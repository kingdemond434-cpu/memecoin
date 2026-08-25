"""The dynamic fee tiers, decoded from chain instead of transcribed from an image.

The fee engine has reported DATA_BLOCKED for every trade after
2026-09-01 20:00 UTC on the grounds that Pump publishes the tier table only as
`docs/fees.png`, and transcribing numbers out of an image is exactly the kind
of fabrication this codebase refuses.

That was the right refusal and the wrong conclusion. The table is not the
source of truth: the on-chain `FeeConfig` account is, and the docs say so
outright -- "if you implement the fee logic correctly, any future change to
the fee tiers structure above should not affect your code". The image is a
courtesy snapshot of an account anyone can read. So the block is lifted by
reading the account, not by squinting at a picture, and it stays lifted when
Pump changes the tiers.

The layout comes from idl/pump_fees.json. The selection logic is transcribed
from the published `calculateFeeTier`, and its shape matters more than it
looks:

    below the FIRST tier's threshold      -> the first tier
    otherwise, scanning tiers in REVERSE  -> the first whose threshold <= cap

That is not "the tier whose range contains the market cap". Thresholds are
floors, the list is ascending, and the reverse scan therefore lands on the
highest tier the coin has reached. Implementing it as a forward scan over
ceilings gives the same answer for well-formed tables and a different one the
moment a table has a gap -- which is the sort of difference that shows up as
a few basis points of unexplained slippage months later.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.chains.idl import IDL_STATUS, PUMP_FEES_IDL, program_id

logger = logging.getLogger(__name__)

FEE_CONFIG_SCHEMA_VERSION = "v1"

# Anchor account discriminator for `FeeConfig`, from idl/pump_fees.json.
FEE_CONFIG_DISCRIMINATOR = bytes.fromhex("8f3492bbdb7b4c9b")

_FEES_WIDTH = 24          # three u64: lp, protocol, creator
_TIER_WIDTH = 16 + _FEES_WIDTH   # u128 threshold + Fees
_HEADER_WIDTH = 8 + 1 + 32       # discriminator + bump + admin pubkey


@dataclass(frozen=True)
class Fees:
    lp_fee_bps: int = 0
    protocol_fee_bps: int = 0
    creator_fee_bps: int = 0

    @property
    def total_bps(self) -> int:
        """What the trade actually pays.

        All three are charged. Summing only protocol and creator -- the two
        the bonding-curve docs talk about -- understates every PumpSwap trade
        by the LP fee, and understating cost is the direction that makes bad
        trades look acceptable.
        """
        return self.lp_fee_bps + self.protocol_fee_bps + self.creator_fee_bps

    def to_dict(self) -> Dict[str, int]:
        return {"lp_fee_bps": self.lp_fee_bps,
                "protocol_fee_bps": self.protocol_fee_bps,
                "creator_fee_bps": self.creator_fee_bps,
                "total_bps": self.total_bps}


@dataclass(frozen=True)
class FeeTier:
    market_cap_lamports_threshold: int
    fees: Fees

    def to_dict(self) -> Dict[str, Any]:
        return {"market_cap_lamports_threshold": self.market_cap_lamports_threshold,
                **self.fees.to_dict()}


@dataclass
class FeeConfig:
    status: str
    admin: str = ""
    flat_fees: Fees = field(default_factory=Fees)
    fee_tiers: Tuple[FeeTier, ...] = ()
    stable_fee_tiers: Tuple[FeeTier, ...] = ()
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": FEE_CONFIG_SCHEMA_VERSION, "status": self.status,
                "admin": self.admin, "flat_fees": self.flat_fees.to_dict(),
                "tiers": [tier.to_dict() for tier in self.fee_tiers],
                "stable_tiers": len(self.stable_fee_tiers), "detail": self.detail}


def fee_config_address() -> str:
    """`["fee_config", pump_program_id]` under the Pump Fees program."""
    from solders.pubkey import Pubkey

    from src.chains.idl import PUMP_IDL

    address, _bump = Pubkey.find_program_address(
        [b"fee_config", bytes(Pubkey.from_string(program_id(PUMP_IDL)))],
        Pubkey.from_string(program_id(PUMP_FEES_IDL)))
    return str(address)


def _read_fees(data: bytes, offset: int) -> Tuple[Fees, int]:
    values = [int.from_bytes(data[offset + index * 8:offset + index * 8 + 8], "little")
              for index in range(3)]
    return Fees(*values), offset + _FEES_WIDTH


def _read_tiers(data: bytes, offset: int) -> Tuple[Tuple[FeeTier, ...], int]:
    if offset + 4 > len(data):
        raise ValueError("truncated before tier count")
    count = int.from_bytes(data[offset:offset + 4], "little")
    offset += 4
    if count > 4_096:
        # A Borsh vec length read at the wrong offset is a very large number,
        # and allocating on it is how a malformed account becomes an outage.
        raise ValueError(f"implausible tier count {count}; layout is out of step")
    if offset + count * _TIER_WIDTH > len(data):
        raise ValueError(f"account holds {len(data)} bytes, {count} tiers need more")
    tiers: List[FeeTier] = []
    for _ in range(count):
        threshold = int.from_bytes(data[offset:offset + 16], "little")
        offset += 16
        fees, offset = _read_fees(data, offset)
        tiers.append(FeeTier(threshold, fees))
    return tuple(tiers), offset


def parse_fee_config(data: bytes) -> FeeConfig:
    """Decode the FeeConfig account, or say exactly why it could not be decoded."""
    if IDL_STATUS != "OK":
        return FeeConfig(status="DATA_BLOCKED", detail=IDL_STATUS)
    if len(data) < _HEADER_WIDTH + _FEES_WIDTH + 8:
        return FeeConfig(status="DATA_BLOCKED",
                         detail=f"account is {len(data)} bytes, too short for FeeConfig")
    if data[:8] != FEE_CONFIG_DISCRIMINATOR:
        return FeeConfig(status="DATA_BLOCKED",
                         detail=f"discriminator {data[:8].hex()} is not a FeeConfig")
    from solders.pubkey import Pubkey

    try:
        admin = str(Pubkey(bytes(data[9:41])))
        flat_fees, offset = _read_fees(data, 41)
        fee_tiers, offset = _read_tiers(data, offset)
        stable_tiers, _ = _read_tiers(data, offset)
    except (ValueError, IndexError) as exc:
        return FeeConfig(status="DATA_BLOCKED", detail=f"malformed FeeConfig: {exc}")
    if not fee_tiers:
        return FeeConfig(status="DATA_BLOCKED", detail="FeeConfig carries no tiers")
    return FeeConfig(status="OK", admin=admin, flat_fees=flat_fees,
                     fee_tiers=fee_tiers, stable_fee_tiers=stable_tiers)


def bonding_curve_market_cap(mint_supply: int, virtual_sol_reserves: int,
                             virtual_token_reserves: int) -> Optional[int]:
    """`virtualSolReserves * mintSupply / virtualTokenReserves`, in lamports.

    Integer division throughout, matching the published BN arithmetic. Doing
    it in floating point would drift across the tier boundaries, which is
    precisely where the answer changes.
    """
    if virtual_token_reserves <= 0 or mint_supply <= 0 or virtual_sol_reserves < 0:
        return None
    return (virtual_sol_reserves * mint_supply) // virtual_token_reserves


def pool_market_cap(base_mint_supply: int, base_reserve: int,
                    quote_reserve: int) -> Optional[int]:
    """`quoteReserve * baseMintSupply / baseReserve`, for a canonical pool."""
    if base_reserve <= 0 or base_mint_supply <= 0 or quote_reserve < 0:
        return None
    return (quote_reserve * base_mint_supply) // base_reserve


def calculate_fee_tier(tiers: Sequence[FeeTier], market_cap: int) -> Optional[Fees]:
    """The published `calculateFeeTier`, transcribed rather than reinvented.

    Below the first tier's threshold the first tier applies; otherwise the
    scan runs in REVERSE and takes the first tier whose threshold the market
    cap has reached. Thresholds are floors on an ascending list, so this lands
    on the highest tier the coin has reached -- which a forward scan over
    ceilings only reproduces while the table has no gaps.
    """
    if not tiers:
        return None
    first = tiers[0]
    if market_cap < first.market_cap_lamports_threshold:
        return first.fees
    for tier in reversed(list(tiers)):
        if market_cap >= tier.market_cap_lamports_threshold:
            return tier.fees
    return first.fees
