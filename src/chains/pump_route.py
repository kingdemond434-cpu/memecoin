"""The canonical T0 route: build a Pump trade locally, with no Jupiter round trip.

Entry used to require two network round trips before a byte of the transaction
existed -- a quote from Jupiter, then a transaction from Jupiter -- on the one
path where latency is the entire product. Both were avoidable. The curve state
already arrives on the stream, the pricing already runs locally with Rust
parity, and every account `buy_v2` and `sell_v2` need is either a fixed
program, a published PDA, or an associated token account. Nothing on the entry
path actually required asking a third party what the trade should be.

So the route is built here: quote off the streamed curve, derive the accounts,
encode the instruction, and hand the result to the existing signing and
submission path. Jupiter keeps three jobs it is genuinely better at -- routing
after migration, an independent cross-check on our own pricing, and a fallback
when the curve state is stale -- and loses the one it should never have had,
which is being a mandatory dependency of a sub-second decision.

Everything here is transcribed from Pump's published docs
(pump-public-docs/docs/instructions/BUY.md and SELL.md), never recalled. The
seeds are quoted in the constants below so a reader can check them against the
source without leaving the file, and the derivation is verified against a
published address at import time: `[b"global"]` under the Pump program must
produce `4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf`, which the README states
outright. If that assertion ever fails, either the program moved or the
derivation is wrong, and in both cases building transactions would be worse
than not trading.

Two addresses the docs do NOT publish are required inputs with no defaults:
the Pump Fees program, and the fee recipients that live in the on-chain Global
config. A guessed program id does not fail loudly -- it produces a transaction
that is rejected at best and lands somewhere unintended at worst -- so the
route reports DATA_BLOCKED until an operator supplies them from a source they
can point at.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by the availability test
    from solders.pubkey import Pubkey
    ROUTE_STATUS = "OK"
except ImportError as exc:  # pragma: no cover - defensive
    Pubkey = None  # type: ignore[assignment]
    ROUTE_STATUS = f"DATA_BLOCKED: solders unavailable ({exc})"

PUMP_ROUTE_SCHEMA_VERSION = "v1"

# Published in pump-public-docs/docs/PUMP_PROGRAM_README.md.
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUBLISHED_GLOBAL = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"

# Published in BUY.md / SELL.md, quoted verbatim so the seeds can be checked
# against the source without leaving this file.
SEED_GLOBAL = b"global"                                    # [b"global"]
SEED_BONDING_CURVE = b"bonding-curve"                      # [b"bonding-curve", base_mint]
SEED_CREATOR_VAULT = b"creator-vault"                      # [b"creator-vault", creator]
SEED_SHARING_CONFIG = b"sharing-config"                    # [b"sharing-config", base_mint]
SEED_GLOBAL_VOLUME = b"global_volume_accumulator"          # [b"global_volume_accumulator"]
SEED_USER_VOLUME = b"user_volume_accumulator"              # [b"user_volume_accumulator", user]
SEED_FEE_CONFIG = b"fee_config"                            # [b"fee_config", pump_program_id]
SEED_EVENT_AUTHORITY = b"__event_authority"                # [b"__event_authority"]

# Fixed programs, each stated in BUY.md.
WSOL_MINT = "So11111111111111111111111111111111111111112"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
ASSOCIATED_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

BUY_V2_ACCOUNTS = 27
SELL_V2_ACCOUNTS = 26


def instruction_discriminator(name: str) -> bytes:
    """Anchor's `sha256("global:<name>")[:8]`. Recomputed, never transcribed."""
    return hashlib.sha256(f"global:{name}".encode()).digest()[:8]


def encode_args(name: str, first: int, second: int) -> bytes:
    """`<discriminator><u64 LE><u64 LE>`, the shape both instructions take."""
    if first < 0 or second < 0 or first >= 2 ** 64 or second >= 2 ** 64:
        raise ValueError("instruction arguments must fit in u64")
    return (instruction_discriminator(name)
            + int(first).to_bytes(8, "little")
            + int(second).to_bytes(8, "little"))


@dataclass(frozen=True)
class AccountMeta:
    pubkey: str
    is_signer: bool = False
    is_writable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"pubkey": self.pubkey, "is_signer": self.is_signer,
                "is_writable": self.is_writable}


@dataclass(frozen=True)
class PumpRouteConfig:
    """The addresses the published docs do not derive.

    `fee_program` is named but never given an address in the docs, and
    `fee_recipient` / `buyback_fee_recipient` are selected from the on-chain
    Global config rather than derived. None of them has a default: a guessed
    program id does not fail loudly, it produces a transaction that is
    rejected at best and lands somewhere unintended at worst.
    """

    fee_program: str = ""
    fee_recipient: str = ""
    buyback_fee_recipient: str = ""
    base_token_program: str = TOKEN_2022_PROGRAM
    quote_token_program: str = TOKEN_PROGRAM
    quote_mint: str = WSOL_MINT

    def blocked_reason(self) -> Optional[str]:
        missing = [name for name in
                   ("fee_program", "fee_recipient", "buyback_fee_recipient")
                   if not getattr(self, name)]
        if missing:
            return ("not published in pump-public-docs; supply from the on-chain "
                    f"Global config or an operator-verified source: {', '.join(missing)}")
        return None


def _key(value: str) -> "Pubkey":
    return Pubkey.from_string(value)


def derive(seeds: Sequence[bytes], program: str = PUMP_PROGRAM) -> str:
    """Program-derived address for these seeds, as base58."""
    address, _bump = Pubkey.find_program_address(list(seeds), _key(program))
    return str(address)


def associated_token_address(owner: str, mint: str, token_program: str) -> str:
    """The ATA, derived under the Associated Token Program's documented seeds.

    `[owner, token_program, mint]` -- and the token program is part of the
    seed, which is why a Token-2022 base mint and an SPL quote mint produce
    different ATAs for the same owner. Passing one where the other belongs is
    the single easiest way to build a transaction that fails on chain.
    """
    address, _bump = Pubkey.find_program_address(
        [bytes(_key(owner)), bytes(_key(token_program)), bytes(_key(mint))],
        _key(ASSOCIATED_TOKEN_PROGRAM))
    return str(address)


@dataclass
class PreparedInstruction:
    """A complete Pump instruction, ready to sign. Never a submitted one."""

    status: str
    program_id: str = PUMP_PROGRAM
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

    def __init__(self, config: Optional[PumpRouteConfig] = None,
                 program: str = PUMP_PROGRAM):
        self.config = config or PumpRouteConfig()
        self.program = program

    # -- derivations -------------------------------------------------------

    def bonding_curve(self, base_mint: str) -> str:
        return derive([SEED_BONDING_CURVE, bytes(_key(base_mint))], self.program)

    def creator_vault(self, creator: str) -> str:
        return derive([SEED_CREATOR_VAULT, bytes(_key(creator))], self.program)

    def sharing_config(self, base_mint: str) -> str:
        return derive([SEED_SHARING_CONFIG, bytes(_key(base_mint))], self.program)

    def user_volume_accumulator(self, user: str) -> str:
        return derive([SEED_USER_VOLUME, bytes(_key(user))], self.program)

    @property
    def global_config(self) -> str:
        return derive([SEED_GLOBAL], self.program)

    @property
    def event_authority(self) -> str:
        return derive([SEED_EVENT_AUTHORITY], self.program)

    @property
    def global_volume_accumulator(self) -> str:
        return derive([SEED_GLOBAL_VOLUME], self.program)

    @property
    def fee_config(self) -> str:
        # Seeded by the PUMP program id but derived under the FEES program:
        # `[b"fee_config", pump_program_id]` is a Pump Fees PDA, per the docs.
        return derive([SEED_FEE_CONFIG, bytes(_key(self.program))],
                      self.config.fee_program)

    # -- account assembly --------------------------------------------------

    def _common(self, base_mint: str, creator: str, user: str) -> Dict[str, str]:
        config = self.config
        curve = self.bonding_curve(base_mint)
        vault = self.creator_vault(creator)
        accumulator = self.user_volume_accumulator(user)
        return {
            "global": self.global_config,
            "base_mint": base_mint,
            "quote_mint": config.quote_mint,
            "base_token_program": config.base_token_program,
            "quote_token_program": config.quote_token_program,
            "associated_token_program": ASSOCIATED_TOKEN_PROGRAM,
            "fee_recipient": config.fee_recipient,
            "associated_quote_fee_recipient": associated_token_address(
                config.fee_recipient, config.quote_mint, config.quote_token_program),
            "buyback_fee_recipient": config.buyback_fee_recipient,
            "associated_quote_buyback_fee_recipient": associated_token_address(
                config.buyback_fee_recipient, config.quote_mint, config.quote_token_program),
            "bonding_curve": curve,
            "associated_base_bonding_curve": associated_token_address(
                curve, base_mint, config.base_token_program),
            "associated_quote_bonding_curve": associated_token_address(
                curve, config.quote_mint, config.quote_token_program),
            "user": user,
            "associated_base_user": associated_token_address(
                user, base_mint, config.base_token_program),
            "associated_quote_user": associated_token_address(
                user, config.quote_mint, config.quote_token_program),
            "creator_vault": vault,
            "associated_creator_vault": associated_token_address(
                vault, config.quote_mint, config.quote_token_program),
            "sharing_config": self.sharing_config(base_mint),
            # Present in the map for both instructions, but only placed in the
            # buy's account list -- `sell_v2` omits it, and that asymmetry is
            # enforced by the ordering below rather than by what exists here.
            "global_volume_accumulator": self.global_volume_accumulator,
            "user_volume_accumulator": accumulator,
            "associated_user_volume_accumulator": associated_token_address(
                accumulator, config.quote_mint, config.quote_token_program),
            "fee_config": self.fee_config,
            "fee_program": config.fee_program,
            "system_program": SYSTEM_PROGRAM,
            "event_authority": self.event_authority,
            "program": self.program,
        }

    def build_buy(self, base_mint: str, creator: str, user: str,
                  amount: int, max_sol_cost: int) -> PreparedInstruction:
        """`buy_v2(amount, max_sol_cost)` over 27 accounts.

        `max_sol_cost` is the caller's slippage bound and is never derived
        here: choosing it would be choosing the trade's risk limit, which
        belongs to the sizing decision and is already frozen into the decision
        snapshot before this is called.
        """
        return self._build("buy_v2", base_mint, creator, user, amount, max_sol_cost)

    def build_sell(self, base_mint: str, creator: str, user: str,
                   amount: int, min_sol_output: int) -> PreparedInstruction:
        """`sell_v2(amount, min_sol_output)` over 26 accounts."""
        return self._build("sell_v2", base_mint, creator, user, amount, min_sol_output)

    def _build(self, name: str, base_mint: str, creator: str, user: str,
               first: int, second: int) -> PreparedInstruction:
        if ROUTE_STATUS != "OK":
            return PreparedInstruction(status="DATA_BLOCKED", detail=ROUTE_STATUS)
        blocked = self.config.blocked_reason()
        if blocked:
            return PreparedInstruction(status="DATA_BLOCKED", detail=blocked)
        if first <= 0:
            return PreparedInstruction(status="REJECTED", detail="non-positive amount")
        if second <= 0:
            return PreparedInstruction(
                status="REJECTED",
                detail="no protective bound; an unbounded trade is not a trade")
        try:
            keys = self._common(base_mint, creator, user)
            data = encode_args(name, first, second)
        except (ValueError, TypeError) as exc:
            return PreparedInstruction(status="REJECTED", detail=f"unbuildable: {exc}")

        buying = name == "buy_v2"
        # The two lists differ in more than length. `buy_v2` includes
        # `global_volume_accumulator`; `sell_v2` omits it. `user` is writable
        # and signer on a buy, and signer but NOT writable on a sell. Copying
        # one list to the other is the obvious mistake, so both are written
        # out in full rather than derived from each other.
        order: List[Tuple[str, bool, bool]] = [
            ("global", False, False),
            ("base_mint", False, False),
            ("quote_mint", False, False),
            ("base_token_program", False, False),
            ("quote_token_program", False, False),
            ("associated_token_program", False, False),
            ("fee_recipient", False, False),
            ("associated_quote_fee_recipient", False, True),
            ("buyback_fee_recipient", False, False),
            ("associated_quote_buyback_fee_recipient", False, True),
            ("bonding_curve", False, True),
            ("associated_base_bonding_curve", False, True),
            ("associated_quote_bonding_curve", False, True),
            ("user", True, buying),
            ("associated_base_user", False, True),
            ("associated_quote_user", False, True),
            ("creator_vault", False, True),
            ("associated_creator_vault", False, True),
            ("sharing_config", False, False),
        ]
        if buying:
            order.append(("global_volume_accumulator", False, True))
        order.extend([
            ("user_volume_accumulator", False, True),
            ("associated_user_volume_accumulator", False, True),
            ("fee_config", False, False),
            ("fee_program", False, False),
            ("system_program", False, False),
            ("event_authority", False, False),
            ("program", False, False),
        ])
        expected = BUY_V2_ACCOUNTS if buying else SELL_V2_ACCOUNTS
        accounts = [AccountMeta(keys[field_name], signer, writable)
                    for field_name, signer, writable in order]
        if len(accounts) != expected:
            # A wrong-length account list is a transaction that fails, or
            # worse, succeeds against the wrong account.
            return PreparedInstruction(
                status="REJECTED", expected_accounts=expected,
                detail=f"built {len(accounts)} accounts, {name} takes {expected}")
        return PreparedInstruction(status="OK", program_id=self.program,
                                   accounts=accounts, data=data,
                                   expected_accounts=expected)

    def report(self) -> Dict[str, Any]:
        blocked = self.config.blocked_reason() if ROUTE_STATUS == "OK" else ROUTE_STATUS
        return {
            "schema": PUMP_ROUTE_SCHEMA_VERSION,
            "status": "OK" if not blocked else "DATA_BLOCKED",
            "detail": blocked or "",
            "program": self.program,
            "global": self.global_config if ROUTE_STATUS == "OK" else "",
            "event_authority": self.event_authority if ROUTE_STATUS == "OK" else "",
        }


if ROUTE_STATUS == "OK":
    # The derivation is checked against an address the docs state outright. If
    # this fails, either the program moved or the seeds are wrong, and in both
    # cases building transactions would be worse than not trading at all.
    _derived_global = derive([SEED_GLOBAL], PUMP_PROGRAM)
    if _derived_global != PUBLISHED_GLOBAL:  # pragma: no cover - fatal by design
        ROUTE_STATUS = (f"DATA_BLOCKED: derived global {_derived_global} does not "
                        f"match the published {PUBLISHED_GLOBAL}")
        logger.error(ROUTE_STATUS)
