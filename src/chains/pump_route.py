"""The canonical T0 route: build a Pump trade locally, with no quote round trip.

Entry used to require two network round trips before a byte of the transaction
existed -- a quote from Jupiter, then a transaction from Jupiter -- on the one
path where latency is the entire product. Both were avoidable. The curve state
already arrives on the stream, the pricing already runs locally with Rust
parity, and every account `buy_v2` and `sell_v2` need is either a fixed
program, a derivable PDA, or an associated token account. Nothing on the entry
path actually required asking a third party what the trade should be.

Jupiter keeps the three jobs it is genuinely better at -- routing after
migration, an independent cross-check on our own pricing, and the fallback
when the curve state is stale -- and loses the one it should never have had,
which is being a mandatory dependency of a sub-second decision.

The account lists are GENERATED from the vendored IDL, not written here. The
first version of this module transcribed them from the prose tables in
docs/instructions/BUY.md, and on three flags those tables disagree with the
program: `fee_recipient` and `buyback_fee_recipient` are writable and the
tables say they are not, `global_volume_accumulator` is not writable and the
tables say it is, and `sharing_config` is derived under the Pump Fees program
rather than under Pump. A transaction built from the prose declares the wrong
mutability on two accounts and derives a third from the wrong program, and it
fails without pointing at why. See src/chains/idl.py.

What remains the caller's to supply is exactly what nobody can derive: the two
fee recipients, which are chosen from the on-chain Global config's published
sets, and the token programs, which depend on how the mint was created. The
recipients ship with this repo (idl/FEE_RECIPIENTS.md) because Pump publishes
all twenty-four addresses, so the desk selects one rather than being blocked.
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.chains.idl import (
    PUMP_FEES_IDL, PUMP_IDL, AccountMeta, IdlError, IdlError as RouteError, account_names,
    build_accounts, discriminator, encode_u64_args, program_id, unresolvable,
)
from src.chains.idl import IDL_STATUS as ROUTE_STATUS

logger = logging.getLogger(__name__)

PUMP_ROUTE_SCHEMA_VERSION = "v2"

PUMP_PROGRAM = program_id(PUMP_IDL) if ROUTE_STATUS == "OK" else ""
# Stated outright in pump-public-docs/docs/PUMP_PROGRAM_README.md, and used
# below as an independent check that the seeds and the program id are both
# right: deriving `["global"]` under the Pump program must reproduce it.
PUBLISHED_GLOBAL = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"

WSOL_MINT = "So11111111111111111111111111111111111111112"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
ASSOCIATED_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

_RECIPIENTS_PATH = Path(__file__).resolve().parents[2] / "idl" / "FEE_RECIPIENTS.md"
_BASE58 = re.compile(r"`([1-9A-HJ-NP-Za-km-z]{32,44})`")


@lru_cache(maxsize=1)
def fee_recipients() -> Dict[str, Tuple[str, ...]]:
    """The published recipient sets, parsed from the vendored doc.

    Pump publishes all twenty-four: eight normal, eight reserved for mayhem
    coins, eight buyback. Choosing one is a routing decision, not a guess, so
    the desk is not blocked on an operator supplying an address that is
    already public. The set a coin belongs to is NOT inferred here -- mayhem
    mode is a property of the coin, and picking the wrong set produces a
    transaction the program rejects.
    """
    groups: Dict[str, List[str]] = {"normal": [], "reserved": [], "buyback": []}
    current: Optional[str] = None
    try:
        text = _RECIPIENTS_PATH.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive
        logger.error("fee recipients unreadable at %s: %s", _RECIPIENTS_PATH, exc)
        return {name: () for name in groups}
    for line in text.splitlines():
        lowered = line.lower()
        if lowered.startswith("## "):
            if "normal" in lowered:
                current = "normal"
            elif "reserved" in lowered:
                current = "reserved"
            elif "buyback" in lowered:
                current = "buyback"
            else:
                current = None
            continue
        if current and (match := _BASE58.search(line)):
            groups[current].append(match.group(1))
    return {name: tuple(values) for name, values in groups.items()}


def select_fee_recipient(mint: str, *, mayhem: bool = False) -> str:
    """Pick a recipient deterministically from the mint.

    Deterministic rather than random so the same mint always writes to the
    same fee account within a session, which spreads load across the eight
    without making two concurrent trades on one coin contend for the same
    account -- and so a failing transaction can be reproduced exactly.
    """
    published = fee_recipients()["reserved" if mayhem else "normal"]
    if not published:
        return ""
    return published[sum(mint.encode()) % len(published)]


def select_buyback_recipient(mint: str) -> str:
    published = fee_recipients()["buyback"]
    if not published:
        return ""
    return published[sum(mint.encode()) % len(published)]


@dataclass(frozen=True)
class PumpRouteConfig:
    """The values no derivation can supply.

    Every field has a defensible default EXCEPT the token programs, which
    depend on how the mint was created: `create_v2` coins are Token-2022 and
    older ones are not. Getting that wrong changes every associated token
    address in the instruction, so it is passed per trade rather than assumed
    once.
    """

    base_token_program: str = TOKEN_2022_PROGRAM
    quote_token_program: str = TOKEN_PROGRAM
    quote_mint: str = WSOL_MINT
    mayhem: bool = False

    def blocked_reason(self) -> Optional[str]:
        if ROUTE_STATUS != "OK":
            return ROUTE_STATUS
        if not fee_recipients()["normal"] or not fee_recipients()["buyback"]:
            return "published fee recipients unreadable"
        return None


@dataclass
class PreparedInstruction:
    """A complete Pump instruction, ready to sign. Never a submitted one."""

    status: str
    program_id: str = ""
    accounts: List[AccountMeta] = field(default_factory=list)
    data: bytes = b""
    detail: str = ""
    expected_accounts: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "program_id": self.program_id,
                "accounts": [item.to_dict() for item in self.accounts],
                "data_hex": self.data.hex(), "detail": self.detail}


class NativePumpRoute:
    """Builds `buy_v2` / `sell_v2` locally from streamed curve state."""

    def __init__(self, config: Optional[PumpRouteConfig] = None):
        self.config = config or PumpRouteConfig()
        self.program = PUMP_PROGRAM

    def build_buy(self, base_mint: str, creator: str, user: str,
                  amount: int, max_sol_cost: int) -> PreparedInstruction:
        """`buy_v2(amount, max_sol_cost)`.

        `max_sol_cost` is the caller's slippage bound and is never derived
        here: choosing it would be choosing the trade's risk limit, which
        belongs to the sizing decision and is already frozen into the decision
        snapshot before this is called.
        """
        return self._build("buy_v2", base_mint, creator, user, amount, max_sol_cost)

    def build_sell(self, base_mint: str, creator: str, user: str,
                   amount: int, min_sol_output: int) -> PreparedInstruction:
        """`sell_v2(amount, min_sol_output)`."""
        return self._build("sell_v2", base_mint, creator, user, amount, min_sol_output)

    def _build(self, name: str, base_mint: str, creator: str, user: str,
               first: int, second: int) -> PreparedInstruction:
        blocked = self.config.blocked_reason()
        if blocked:
            return PreparedInstruction(status="DATA_BLOCKED", detail=blocked)
        if first <= 0:
            return PreparedInstruction(status="REJECTED", detail="non-positive amount")
        if second <= 0:
            return PreparedInstruction(
                status="REJECTED",
                detail="no protective bound; an unbounded trade is not a trade")
        if not creator:
            return PreparedInstruction(
                status="DATA_BLOCKED",
                detail="no curve creator; creator_vault is underivable")
        config = self.config
        supplied = {
            "base_mint": base_mint,
            "quote_mint": config.quote_mint,
            "base_token_program": config.base_token_program,
            "quote_token_program": config.quote_token_program,
            "associated_token_program": ASSOCIATED_TOKEN_PROGRAM,
            "system_program": SYSTEM_PROGRAM,
            # `fee_config` is declared before `fee_program` in the account
            # list but derived UNDER it, so resolution order alone cannot
            # supply it. Read from the pump_fees IDL's own published address
            # rather than hardcoded here.
            "fee_program": program_id(PUMP_FEES_IDL),
            "user": user,
            "fee_recipient": select_fee_recipient(base_mint, mayhem=config.mayhem),
            "buyback_fee_recipient": select_buyback_recipient(base_mint),
            # Named by the IDL as a seed path on an account we do not pass.
            "bonding_curve.creator": creator,
            "associated_base_user": associated_token_address(
                user, base_mint, config.base_token_program),
        }
        try:
            accounts = build_accounts(PUMP_IDL, name, supplied)
            data = encode_u64_args(PUMP_IDL, name, (first, second))
        except (IdlError, ValueError, TypeError) as exc:
            return PreparedInstruction(status="REJECTED", detail=f"unbuildable: {exc}")
        expected = len(account_names(PUMP_IDL, name))
        if len(accounts) != expected:  # pragma: no cover - structurally impossible
            return PreparedInstruction(
                status="REJECTED", expected_accounts=expected,
                detail=f"built {len(accounts)} accounts, {name} takes {expected}")
        return PreparedInstruction(status="OK", program_id=self.program,
                                   accounts=accounts, data=data,
                                   expected_accounts=expected)

    def report(self) -> Dict[str, Any]:
        blocked = self.config.blocked_reason()
        recipients = fee_recipients()
        return {
            "schema": PUMP_ROUTE_SCHEMA_VERSION,
            "status": "OK" if not blocked else "DATA_BLOCKED",
            "detail": blocked or "",
            "program": self.program,
            "buy_accounts": len(account_names(PUMP_IDL, "buy_v2")) if not blocked else 0,
            "sell_accounts": len(account_names(PUMP_IDL, "sell_v2")) if not blocked else 0,
            "fee_recipients": {name: len(values) for name, values in recipients.items()},
        }


def associated_token_address(owner: str, mint: str, token_program: str) -> str:
    """The ATA, under the Associated Token Program's documented seeds.

    `[owner, token_program, mint]` -- the token program is PART of the seed,
    which is why a Token-2022 base mint and an SPL quote mint give the same
    owner different addresses. Passing one where the other belongs is the
    easiest way to build a transaction that fails on chain.
    """
    from solders.pubkey import Pubkey

    address, _bump = Pubkey.find_program_address(
        [bytes(Pubkey.from_string(owner)),
         bytes(Pubkey.from_string(token_program)),
         bytes(Pubkey.from_string(mint))],
        Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM))
    return str(address)


def derived_global() -> str:
    """`["global"]` under the Pump program, for the import-time check below."""
    from solders.pubkey import Pubkey

    address, _bump = Pubkey.find_program_address(
        [b"global"], Pubkey.from_string(PUMP_PROGRAM))
    return str(address)


if ROUTE_STATUS == "OK":
    # An independent confirmation of both the seed and the program id: the
    # README states the global config's address outright and separately states
    # that it is derived from ["global"]. Deriving one and getting the other
    # means both readings are right. If this ever fails, building transactions
    # would be worse than not trading.
    _derived = derived_global()
    if _derived != PUBLISHED_GLOBAL:  # pragma: no cover - fatal by design
        ROUTE_STATUS = (f"DATA_BLOCKED: derived global {_derived} does not match "
                        f"the published {PUBLISHED_GLOBAL}")
        logger.error(ROUTE_STATUS)
