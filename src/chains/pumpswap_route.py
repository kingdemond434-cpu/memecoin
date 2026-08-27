"""PumpSwap `buy` / `sell` construction, and the Pool account decoder.

This was the last thing marked DATA_BLOCKED on the grounds that Pump's prose
docs do not publish PumpSwap's account lists. They do not -- but the IDL does,
and it is published in the same repository. The block was never about the
information being unavailable; it was about looking in the wrong file.

So the same machinery that builds `buy_v2` builds these: the account list,
every flag, every PDA seed and the program each PDA is derived under all come
out of idl/pump_amm.json. `buy` takes 23 accounts, `sell` takes 21, and the
difference is `global_volume_accumulator` and `user_volume_accumulator`, which
only the buy carries.

Two things here are not shared with the bonding curve and are easy to get
wrong by analogy.

The pool is not derived from the mint alone. Its seeds are
`["pool", index, creator, base_mint, quote_mint]`, so one coin can have
several pools and the index is part of the address. Deriving with a guessed
index produces a real address that is not this pool, which is worse than
failing. The pool address is therefore an input, read from the migration
event or from the account itself, never assumed.

And `coin_creator` is not `creator`. `creator` is whoever opened the pool;
`coin_creator` is who receives creator fees, and it is the one the creator
vault is derived from. The Pool layout carries both, adjacently, with the same
type -- reading the wrong field yields a valid-looking vault that belongs to
somebody else.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.chains.idl import (
    PUMP_AMM_IDL, PUMP_FEES_IDL, AccountMeta, IdlError, account_names,
    build_accounts, discriminator, encode_u64_args, program_id, unresolvable,
)
from src.chains.idl import IDL_STATUS
from src.chains.pump_route import (
    ASSOCIATED_TOKEN_PROGRAM, SYSTEM_PROGRAM, TOKEN_2022_PROGRAM, TOKEN_PROGRAM,
    WSOL_MINT, PreparedInstruction, associated_token_address,
    select_fee_recipient,
)

logger = logging.getLogger(__name__)

PUMPSWAP_ROUTE_SCHEMA_VERSION = "v1"
PUMPSWAP_PROGRAM = program_id(PUMP_AMM_IDL) if IDL_STATUS == "OK" else ""

# Anchor account discriminator for `Pool`, taken from the IDL rather than
# recomputed, then checked against the derivation in the test suite.
POOL_DISCRIMINATOR = bytes.fromhex("f19a6d0411b16dbc")

# Offsets into the Pool account, derived from the IDL's field order. Written
# out so a layout change upstream shows up as a failing offset test rather
# than as silently misread reserves.
_POOL_LAYOUT: Tuple[Tuple[str, int, str], ...] = (
    ("pool_bump", 1, "u8"),
    ("index", 2, "u16"),
    ("creator", 32, "pubkey"),
    ("base_mint", 32, "pubkey"),
    ("quote_mint", 32, "pubkey"),
    ("lp_mint", 32, "pubkey"),
    ("pool_base_token_account", 32, "pubkey"),
    ("pool_quote_token_account", 32, "pubkey"),
    ("lp_supply", 8, "u64"),
    ("coin_creator", 32, "pubkey"),
    ("is_mayhem_mode", 1, "bool"),
    ("is_cashback_coin", 1, "bool"),
    ("virtual_quote_reserves", 16, "i128"),
)
POOL_SIZE = 8 + sum(width for _, width, _ in _POOL_LAYOUT)


@dataclass
class PoolState:
    """A decoded PumpSwap pool. Every field read, none inferred."""

    status: str
    pool: str = ""
    index: int = 0
    creator: str = ""
    base_mint: str = ""
    quote_mint: str = ""
    # Read by the layout and, until now, discarded. It is the only handle on
    # who controls the pool after migration: LP supply says how much power
    # exists, this says whose accounts to look in for it.
    lp_mint: str = ""
    pool_base_token_account: str = ""
    pool_quote_token_account: str = ""
    lp_supply: int = 0
    coin_creator: str = ""
    is_mayhem_mode: bool = False
    is_cashback_coin: bool = False
    virtual_quote_reserves: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "pool": self.pool, "index": self.index,
                "creator": self.creator, "coin_creator": self.coin_creator,
                "base_mint": self.base_mint, "quote_mint": self.quote_mint,
                "lp_mint": self.lp_mint, "lp_supply": self.lp_supply,
                "is_mayhem_mode": self.is_mayhem_mode,
                "is_cashback_coin": self.is_cashback_coin,
                "virtual_quote_reserves": self.virtual_quote_reserves,
                "detail": self.detail}


def parse_pool(data: bytes, address: str = "") -> PoolState:
    """Decode a Pool account, or say precisely why it could not be decoded.

    The discriminator is checked first. Without that check a differently
    shaped account of the right length decodes into plausible-looking
    pubkeys, and the first sign of trouble is a transaction against a pool
    that does not exist.
    """
    if IDL_STATUS != "OK":
        return PoolState(status="DATA_BLOCKED", detail=IDL_STATUS)
    if len(data) < POOL_SIZE:
        return PoolState(status="DATA_BLOCKED",
                         detail=f"account is {len(data)} bytes, Pool needs {POOL_SIZE}")
    if data[:8] != POOL_DISCRIMINATOR:
        return PoolState(status="DATA_BLOCKED",
                         detail=f"discriminator {data[:8].hex()} is not a Pool")
    from solders.pubkey import Pubkey

    offset = 8
    values: Dict[str, Any] = {}
    for name, width, kind in _POOL_LAYOUT:
        chunk = data[offset:offset + width]
        offset += width
        if kind == "pubkey":
            values[name] = str(Pubkey(bytes(chunk)))
        elif kind == "bool":
            values[name] = bool(chunk[0])
        elif kind == "i128":
            values[name] = int.from_bytes(chunk, "little", signed=True)
        else:
            values[name] = int.from_bytes(chunk, "little")
    return PoolState(
        status="OK", pool=address, index=values["index"],
        creator=values["creator"], base_mint=values["base_mint"],
        quote_mint=values["quote_mint"], lp_mint=values["lp_mint"],
        pool_base_token_account=values["pool_base_token_account"],
        pool_quote_token_account=values["pool_quote_token_account"],
        lp_supply=values["lp_supply"],
        # NOT `creator`. This is the fee recipient the creator vault is
        # derived from, and the two sit adjacent in the layout with the same
        # type -- reading the wrong one yields a vault that belongs to
        # somebody else and looks entirely valid.
        coin_creator=values["coin_creator"],
        is_mayhem_mode=values["is_mayhem_mode"],
        is_cashback_coin=values["is_cashback_coin"],
        virtual_quote_reserves=values["virtual_quote_reserves"])


def derive_pool(index: int, creator: str, base_mint: str, quote_mint: str) -> str:
    """`["pool", index_le_u16, creator, base_mint, quote_mint]`.

    Exposed for verifying a pool address we were told about, not for
    discovering one: a guessed index derives a real address that is not this
    pool, which is worse than failing.
    """
    from solders.pubkey import Pubkey

    address, _bump = Pubkey.find_program_address(
        [b"pool", int(index).to_bytes(2, "little"),
         bytes(Pubkey.from_string(creator)),
         bytes(Pubkey.from_string(base_mint)),
         bytes(Pubkey.from_string(quote_mint))],
        Pubkey.from_string(PUMPSWAP_PROGRAM))
    return str(address)


@dataclass(frozen=True)
class PumpSwapRouteConfig:
    base_token_program: str = TOKEN_2022_PROGRAM
    quote_token_program: str = TOKEN_PROGRAM

    def blocked_reason(self) -> Optional[str]:
        return None if IDL_STATUS == "OK" else IDL_STATUS


class PumpSwapRoute:
    """Builds PumpSwap `buy` / `sell` from a decoded pool."""

    def __init__(self, config: Optional[PumpSwapRouteConfig] = None):
        self.config = config or PumpSwapRouteConfig()
        self.program = PUMPSWAP_PROGRAM

    def build_buy(self, pool: PoolState, user: str, base_amount_out: int,
                  max_quote_amount_in: int) -> PreparedInstruction:
        """`buy(base_amount_out, max_quote_amount_in, track_volume)`.

        The third argument is an `OptionBool`, not a u64, so the data is
        assembled here rather than through the u64 helper -- and it is encoded
        explicitly as absent, because opting a trade into volume tracking is a
        choice about what we tell the protocol, not a default to inherit.
        """
        prepared = self._build("buy", pool, user, base_amount_out, max_quote_amount_in)
        if prepared.ok:
            # Anchor encodes `Option<bool>` as a single presence byte followed
            # by the value when present. Absent is one zero byte.
            prepared.data = prepared.data + b"\x00"
        return prepared

    def build_sell(self, pool: PoolState, user: str, base_amount_in: int,
                   min_quote_amount_out: int) -> PreparedInstruction:
        """`sell(base_amount_in, min_quote_amount_out)`."""
        return self._build("sell", pool, user, base_amount_in, min_quote_amount_out)

    def _build(self, name: str, pool: PoolState, user: str,
               first: int, second: int) -> PreparedInstruction:
        blocked = self.config.blocked_reason()
        if blocked:
            return PreparedInstruction(status="DATA_BLOCKED", detail=blocked, venue="pumpswap")
        if not pool.ok:
            return PreparedInstruction(status="DATA_BLOCKED",
                                       detail=f"pool not decoded: {pool.detail}")
        if not pool.pool:
            return PreparedInstruction(
                status="DATA_BLOCKED",
                detail="pool address unknown; it is an input, never derived from the mint")
        if first <= 0:
            return PreparedInstruction(status="REJECTED", detail="non-positive amount")
        if second <= 0:
            return PreparedInstruction(
                status="REJECTED",
                detail="no protective bound; an unbounded trade is not a trade")
        config = self.config
        from solders.pubkey import Pubkey

        global_config, _bump = Pubkey.find_program_address(
            [b"global_config"], Pubkey.from_string(self.program))
        supplied = {
            "pool": pool.pool,
            "user": user,
            # `["global_config"]`, per the create_config instruction in the
            # IDL. The buy/sell instructions declare it without a pda block,
            # so it has to be derived here rather than resolved for us.
            "global_config": str(global_config),
            "base_mint": pool.base_mint,
            "quote_mint": pool.quote_mint,
            "base_token_program": config.base_token_program,
            "quote_token_program": config.quote_token_program,
            "system_program": SYSTEM_PROGRAM,
            "associated_token_program": ASSOCIATED_TOKEN_PROGRAM,
            "fee_program": program_id(PUMP_FEES_IDL),
            # Mayhem coins draw their protocol fee recipient from a different
            # published set, and the pool itself says which kind it is.
            "protocol_fee_recipient": select_fee_recipient(
                pool.base_mint, mayhem=pool.is_mayhem_mode),
            "user_base_token_account": associated_token_address(
                user, pool.base_mint, config.base_token_program),
            "user_quote_token_account": associated_token_address(
                user, pool.quote_mint, config.quote_token_program),
            "pool_base_token_account": pool.pool_base_token_account,
            "pool_quote_token_account": pool.pool_quote_token_account,
            # The IDL names this seed path on an account we do not pass.
            "pool.coin_creator": pool.coin_creator,
        }
        try:
            accounts = build_accounts(PUMP_AMM_IDL, name, supplied)
            data = encode_u64_args(PUMP_AMM_IDL, name, (first, second))
        except (IdlError, ValueError, TypeError) as exc:
            return PreparedInstruction(status="REJECTED", detail=f"unbuildable: {exc}")
        expected = len(account_names(PUMP_AMM_IDL, name))
        return PreparedInstruction(status="OK", program_id=self.program,
                                   accounts=accounts, data=data,
                                   expected_accounts=expected, venue="pumpswap")

    def report(self) -> Dict[str, Any]:
        blocked = self.config.blocked_reason()
        return {
            "schema": PUMPSWAP_ROUTE_SCHEMA_VERSION,
            "status": "OK" if not blocked else "DATA_BLOCKED",
            "detail": blocked or "",
            "program": self.program,
            "buy_accounts": len(account_names(PUMP_AMM_IDL, "buy")) if not blocked else 0,
            "sell_accounts": len(account_names(PUMP_AMM_IDL, "sell")) if not blocked else 0,
            "pool_size": POOL_SIZE,
        }
