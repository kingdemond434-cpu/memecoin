"""Local PumpSwap (Pump AMM) quoting.

The bonding curve has ``pump_curve``; this is its counterpart for the pool a
coin graduates into. Without it a migrated token has no local price, and the
execution path falls back to a router round trip for every decision -- which
is exactly the dependency the native route existed to remove.

Two things here are taken from the protocol rather than assumed:

* **Effective quote reserves.** ``PUMP_SWAP_README.md`` is explicit that buys
  and sells are priced against ``pool_quote_token_account.amount +
  Pool::virtual_quote_reserves``, not the raw vault balance. The virtual field
  is zero on every pool today, so the two agree -- which is precisely why
  reading only the raw balance would keep working right up until it silently
  stopped.

* **Fees.** The fee basis points are not guessed from a published table. Every
  PumpSwap ``BuyEvent`` / ``SellEvent`` carries the bps *charged on that
  trade* -- lp, protocol, coin creator, cashback and buyback -- and this
  module quotes against those. A pool whose fee schedule has never been
  observed is DATA_BLOCKED, not quoted at a default.

Fee application follows the event fields themselves: on a buy, ``quote_amount_in``
is the AMM leg and ``user_quote_amount_in`` is what actually leaves the wallet,
so a spending budget is treated as fee-INCLUSIVE and the AMM leg is
``budget * 10_000 // (10_000 + total_bps)``. On a sell, ``quote_amount_out`` is
the AMM leg and the fee is taken out of it.

Like ``pump_curve``, every quote is returned with ``verified=False``: these are
local estimates for decision-making, and what gates real capital is a measured
fill, not our own arithmetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.chains.pump_curve import LAMPORTS_PER_SOL, CurveQuote

PUMPSWAP_CURVE_SCHEMA_VERSION = "v1"

# Quoting against a pool whose reserves have not moved in this long is quoting
# against a memory. The number is not a timeout on the pool -- it is the point
# past which we would rather say DATA_BLOCKED than price a trade.
DEFAULT_MAX_STATE_AGE_S = 30.0


def _blocked(reason: str) -> CurveQuote:
    return CurveQuote(0, 0, 0, 0.0, None, verified=False,
                      data_status="DATA_BLOCKED", reason=reason)


@dataclass
class PumpSwapPoolState:
    """Live pool reserves plus the fee schedule actually charged on it.

    Assembled from the stream: ``pool_created`` supplies identity and opening
    reserves, and every subsequent trade event supplies post-trade reserves
    and the fee bps that trade paid.
    """

    pool: str = ""
    base_mint: str = ""
    quote_mint: str = ""
    base_reserves: int = 0
    quote_reserves: int = 0
    virtual_quote_reserves: int = 0
    # Measured, never defaulted. ``None`` means no trade on this pool has been
    # observed yet, and a quote refuses rather than inventing a schedule.
    total_fee_bps: Optional[int] = None
    lp_fee_bps: Optional[int] = None
    protocol_fee_bps: Optional[int] = None
    coin_creator_fee_bps: Optional[int] = None
    coin_creator: str = ""
    base_supply: int = 0
    base_decimals: int = 0
    updated_at: float = 0.0
    slot: int = 0
    source: str = ""

    @property
    def effective_quote_reserves(self) -> int:
        """What the protocol prices against, per the published docs."""
        return self.quote_reserves + self.virtual_quote_reserves

    @property
    def tradeable(self) -> bool:
        return (self.base_reserves > 0 and self.effective_quote_reserves > 0
                and bool(self.pool) and bool(self.base_mint) and bool(self.quote_mint))

    @property
    def price_quote_per_base(self) -> Optional[float]:
        if not self.tradeable:
            return None
        return self.effective_quote_reserves / self.base_reserves

    def age_s(self, now: Optional[float] = None) -> float:
        if self.updated_at <= 0:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - self.updated_at)

    def blocked_reason(self, max_age_s: float = DEFAULT_MAX_STATE_AGE_S,
                       now: Optional[float] = None) -> Optional[str]:
        """Why this state cannot be quoted against, or None."""
        if not self.pool:
            return "no_pool_address"
        if not self.tradeable:
            return "pool_reserves_empty"
        if self.total_fee_bps is None:
            return "fee_schedule_unobserved"
        if self.total_fee_bps < 0 or self.total_fee_bps >= 10_000:
            return f"implausible_fee_bps:{self.total_fee_bps}"
        if max_age_s > 0 and self.age_s(now) > max_age_s:
            return f"state_stale:{self.age_s(now):.1f}s"
        return None

    def to_dict(self) -> dict:
        return {
            "schema": PUMPSWAP_CURVE_SCHEMA_VERSION, "pool": self.pool,
            "base_mint": self.base_mint, "quote_mint": self.quote_mint,
            "base_reserves": self.base_reserves, "quote_reserves": self.quote_reserves,
            "virtual_quote_reserves": self.virtual_quote_reserves,
            "effective_quote_reserves": self.effective_quote_reserves,
            "total_fee_bps": self.total_fee_bps, "coin_creator": self.coin_creator,
            "base_supply": self.base_supply, "updated_at": self.updated_at,
            "slot": self.slot, "source": self.source,
            "price_quote_per_base": self.price_quote_per_base,
        }


def quote_buy(state: PumpSwapPoolState, quote_lamports: int, *,
              max_age_s: float = DEFAULT_MAX_STATE_AGE_S,
              now: Optional[float] = None) -> CurveQuote:
    """Base tokens received for spending ``quote_lamports`` in total.

    The budget is what leaves the wallet, fees included -- so the fee is
    removed first and the remainder is what moves along the constant product.
    Sizing decided to spend this much; quoting the full amount against the
    curve and then paying the fee on top would spend more than was decided.
    """
    if quote_lamports <= 0:
        return _blocked("non_positive_input")
    blocked = state.blocked_reason(max_age_s, now)
    if blocked:
        return _blocked(blocked)

    total_bps = int(state.total_fee_bps or 0)
    amm_leg = (quote_lamports * 10_000) // (10_000 + total_bps)
    fee = quote_lamports - amm_leg
    if amm_leg <= 0:
        return _blocked("input_fully_consumed_by_fee")

    reserves_quote = state.effective_quote_reserves
    base_out = (amm_leg * state.base_reserves) // (reserves_quote + amm_leg)
    if base_out <= 0:
        return _blocked("output_rounds_to_zero")
    # A constant-product pool can never pay out its entire base side; the
    # bound is here so a quote cannot claim inventory the pool does not hold.
    if base_out >= state.base_reserves:
        return _blocked("size_exceeds_pool_inventory")

    spot = state.price_quote_per_base or 0.0
    average = (quote_lamports / LAMPORTS_PER_SOL) / base_out
    spot_sol = spot / LAMPORTS_PER_SOL if spot else 0.0
    impact = ((average - spot_sol) / spot_sol) if spot_sol > 0 else 0.0
    return CurveQuote(
        input_amount=quote_lamports, output_amount=int(base_out), fee_amount=int(fee),
        price_impact_pct=max(0.0, float(impact)),
        average_price_sol_per_token=average, verified=False)


def quote_sell(state: PumpSwapPoolState, base_amount: int, *,
               max_age_s: float = DEFAULT_MAX_STATE_AGE_S,
               now: Optional[float] = None) -> CurveQuote:
    """Quote lamports received for selling ``base_amount``, net of fees."""
    if base_amount <= 0:
        return _blocked("non_positive_input")
    blocked = state.blocked_reason(max_age_s, now)
    if blocked:
        return _blocked(blocked)

    gross = (base_amount * state.effective_quote_reserves) // (state.base_reserves + base_amount)
    if gross <= 0:
        return _blocked("output_rounds_to_zero")
    total_bps = int(state.total_fee_bps or 0)
    # Round the fee UP. Rounding it down overstates the proceeds by a lamport
    # in the direction that flatters the trade.
    fee = -((-gross * total_bps) // 10_000)
    net = gross - fee
    if net <= 0:
        return _blocked("output_fully_consumed_by_fee")

    spot = state.price_quote_per_base or 0.0
    average = (net / LAMPORTS_PER_SOL) / base_amount
    spot_sol = spot / LAMPORTS_PER_SOL if spot else 0.0
    impact = ((spot_sol - average) / spot_sol) if spot_sol > 0 else 0.0
    return CurveQuote(
        input_amount=base_amount, output_amount=int(net), fee_amount=int(fee),
        price_impact_pct=max(0.0, float(impact)),
        average_price_sol_per_token=average, verified=False)


def sell_capacity_base(state: PumpSwapPoolState, max_impact_pct: float = 0.15, *,
                       max_age_s: float = DEFAULT_MAX_STATE_AGE_S,
                       now: Optional[float] = None) -> int:
    """Largest sale in base tokens whose price impact stays within the bound.

    The same question ``pump_curve.sell_capacity_lamports`` answers for the
    curve, asked of the pool: not "what fraction of liquidity is polite" but
    "how much of this position can actually be exited".
    """
    if max_impact_pct <= 0 or state.blocked_reason(max_age_s, now):
        return 0
    low, high = 0, state.base_reserves
    for _ in range(64):
        if low >= high:
            break
        mid = (low + high + 1) // 2
        quote = quote_sell(state, mid, max_age_s=max_age_s, now=now)
        if quote.data_status == "OK" and quote.price_impact_pct <= max_impact_pct:
            low = mid
        else:
            high = mid - 1
    return int(low)
