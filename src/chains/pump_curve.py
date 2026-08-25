"""Native Pump.fun bonding-curve state and local pricing.

Why this exists: the desk previously decided whether a token was tradeable by
asking Jupiter for a route. A newborn Pump token is perfectly tradeable on its
own bonding curve seconds before any aggregator indexes it, so that check
turned "the aggregator has not caught up yet" into "no executable evidence,
do not trade" — excellent research conservatism and exactly wrong for the
first seconds of a launch, which is the only window that matters here.

The curve is a constant-product market over *virtual* reserves held in the
bonding-curve account, so price, size impact and capacity are all computable
locally from one already-streamed account with no network round trip.

VERIFICATION STATUS
-------------------
The layout and the constant-product relation below are implemented from the
public Pump program interface, and the invariants are unit-tested. They have
NOT yet been reconciled against captured mainnet buy/sell transactions in this
environment (no outbound Solana RPC here to capture them). Accordingly:

  * ``quote_buy`` / ``quote_sell`` are safe for research marking and capacity
    estimation, where an error shows up as a mispriced observation.
  * ``CurveQuote.verified`` is False until fixtures confirm the arithmetic
    against real fills. Execution paths must refuse to size real capital on an
    unverified quote rather than trusting hand-derived math.
"""

import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Anchor account discriminator for BondingCurve, sha256("account:BondingCurve")[:8].
BONDING_CURVE_DISCRIMINATOR = bytes((23, 183, 248, 55, 96, 216, 172, 96))

# Pump's published trade fee, in basis points, applied to the quote leg.
DEFAULT_FEE_BPS = 100

LAMPORTS_PER_SOL = 1_000_000_000


@dataclass
class BondingCurveState:
    """Decoded bonding-curve account."""

    virtual_token_reserves: int
    virtual_sol_reserves: int
    real_token_reserves: int
    real_sol_reserves: int
    token_total_supply: int
    complete: bool
    creator: str = ""

    @property
    def tradeable(self) -> bool:
        """A completed curve has migrated; trade the AMM pool instead."""
        return (not self.complete
                and self.virtual_token_reserves > 0
                and self.virtual_sol_reserves > 0)

    @property
    def price_sol_per_token(self) -> Optional[float]:
        """Marginal price from virtual reserves, or None when not tradeable."""
        if not self.tradeable:
            return None
        return (self.virtual_sol_reserves / LAMPORTS_PER_SOL) / self.virtual_token_reserves

    @property
    def progress(self) -> Optional[float]:
        """Fraction of the curve's token inventory already sold, 0..1."""
        if self.token_total_supply <= 0:
            return None
        sold = self.token_total_supply - self.real_token_reserves
        return max(0.0, min(1.0, sold / self.token_total_supply))


@dataclass
class CurveQuote:
    """Locally computed quote. ``verified`` gates real-capital use."""

    input_amount: int
    output_amount: int
    fee_amount: int
    price_impact_pct: float
    average_price_sol_per_token: Optional[float]
    verified: bool = False
    data_status: str = "OK"
    reason: str = ""


def _blocked(reason: str) -> CurveQuote:
    return CurveQuote(0, 0, 0, 0.0, None, verified=False, data_status="DATA_BLOCKED", reason=reason)


def parse_bonding_curve(data: bytes) -> BondingCurveState:
    """Decode a bonding-curve account.

    Raises ValueError rather than returning a partly-filled state, so a layout
    change surfaces immediately instead of silently producing wrong prices.
    """
    if len(data) < 8:
        raise ValueError(f"bonding curve account is {len(data)} bytes; expected at least 8")
    if data[:8] != BONDING_CURVE_DISCRIMINATOR:
        raise ValueError("account discriminator is not Pump BondingCurve")
    if len(data) < 49:
        raise ValueError(f"bonding curve payload is {len(data)} bytes; expected at least 49")

    (virtual_token, virtual_sol, real_token, real_sol, total_supply) = struct.unpack_from("<QQQQQ", data, 8)
    complete_byte = data[48]
    if complete_byte not in (0, 1):
        raise ValueError("bonding curve 'complete' flag is not a bool")

    creator = ""
    if len(data) >= 81:
        from src.chains.yellowstone_grpc import b58encode
        creator = b58encode(data[49:81])

    return BondingCurveState(
        virtual_token_reserves=virtual_token,
        virtual_sol_reserves=virtual_sol,
        real_token_reserves=real_token,
        real_sol_reserves=real_sol,
        token_total_supply=total_supply,
        complete=bool(complete_byte),
        creator=creator,
    )


def quote_buy(state: BondingCurveState, sol_lamports: int, fee_bps: int = DEFAULT_FEE_BPS) -> CurveQuote:
    """Tokens received for spending ``sol_lamports``.

    Constant product over virtual reserves: the fee is taken from the quote
    leg first, then the remainder moves along the curve.
    """
    if sol_lamports <= 0:
        return _blocked("non_positive_input")
    if not state.tradeable:
        return _blocked("curve_complete_or_empty")

    fee = (sol_lamports * fee_bps) // 10_000
    net = sol_lamports - fee
    if net <= 0:
        return _blocked("input_fully_consumed_by_fee")

    # k = virtual_sol * virtual_token is held constant across the trade.
    tokens_out = (net * state.virtual_token_reserves) // (state.virtual_sol_reserves + net)
    if tokens_out <= 0:
        return _blocked("output_rounds_to_zero")
    # The curve cannot deliver more inventory than it actually holds.
    tokens_out = min(tokens_out, state.real_token_reserves) if state.real_token_reserves > 0 else tokens_out

    spot = state.price_sol_per_token or 0.0
    average = (net / LAMPORTS_PER_SOL) / tokens_out if tokens_out else None
    impact = ((average - spot) / spot) if spot > 0 and average else 0.0

    return CurveQuote(
        input_amount=sol_lamports,
        output_amount=int(tokens_out),
        fee_amount=int(fee),
        price_impact_pct=max(0.0, float(impact)),
        average_price_sol_per_token=average,
        verified=False,
    )


def quote_sell(state: BondingCurveState, token_amount: int, fee_bps: int = DEFAULT_FEE_BPS) -> CurveQuote:
    """Lamports received for selling ``token_amount``, net of fee."""
    if token_amount <= 0:
        return _blocked("non_positive_input")
    if not state.tradeable:
        return _blocked("curve_complete_or_empty")

    gross = (token_amount * state.virtual_sol_reserves) // (state.virtual_token_reserves + token_amount)
    if gross <= 0:
        return _blocked("output_rounds_to_zero")
    # Only real SOL can actually be paid out, whatever the virtual curve says.
    if state.real_sol_reserves > 0:
        gross = min(gross, state.real_sol_reserves)

    fee = (gross * fee_bps) // 10_000
    net = gross - fee
    if net <= 0:
        return _blocked("output_fully_consumed_by_fee")

    spot = state.price_sol_per_token or 0.0
    average = (net / LAMPORTS_PER_SOL) / token_amount
    impact = ((spot - average) / spot) if spot > 0 else 0.0

    return CurveQuote(
        input_amount=token_amount,
        output_amount=int(net),
        fee_amount=int(fee),
        price_impact_pct=max(0.0, float(impact)),
        average_price_sol_per_token=average,
        verified=False,
    )


def sell_capacity_lamports(state: BondingCurveState, max_impact_pct: float = 0.15,
                           fee_bps: int = DEFAULT_FEE_BPS) -> int:
    """Largest sale, in tokens, whose price impact stays within the bound.

    Capacity is a function rather than a flat "1% of liquidity" guardrail:
    it answers how much can actually be exited, which is what sizing needs.
    Binary search over the monotone impact curve.
    """
    if not state.tradeable or max_impact_pct <= 0:
        return 0
    low, high = 0, max(state.real_token_reserves, state.virtual_token_reserves)
    if high <= 0:
        return 0
    while low < high:
        mid = (low + high + 1) // 2
        quote = quote_sell(state, mid, fee_bps)
        if quote.data_status == "OK" and quote.price_impact_pct <= max_impact_pct:
            low = mid
        else:
            high = mid - 1
    return low


def observation_from_state(state: BondingCurveState, probe_lamports: int = 10_000_000,
                           fee_bps: int = DEFAULT_FEE_BPS) -> Dict[str, Any]:
    """Research observation derived entirely from streamed curve state.

    Marking a position from local state costs no Jupiter quota and no latency,
    so the price path keeps being collected even while the aggregator is rate
    limited or has not indexed the token at all. It is explicitly labelled as
    curve-derived so it is never confused with an executable Jupiter round
    trip, which remains the independent feasibility check.
    """
    if not state.tradeable:
        return {"type": "route", "feasible": False, "data_status": "DATA_BLOCKED",
                "reason": "curve_complete_or_empty", "measurement": "pump_curve_local"}
    buy = quote_buy(state, probe_lamports, fee_bps)
    if buy.data_status != "OK":
        return {"type": "route", "feasible": False, "data_status": "DATA_BLOCKED",
                "reason": buy.reason, "measurement": "pump_curve_local"}
    back = quote_sell(state, buy.output_amount, fee_bps)
    round_trip = (back.output_amount / probe_lamports) if back.data_status == "OK" else None
    return {
        "type": "route",
        "feasible": back.data_status == "OK" and back.output_amount > 0,
        "data_status": "OK",
        "measurement": "pump_curve_local",
        "price_sol_per_token": state.price_sol_per_token,
        "price_impact_pct": buy.price_impact_pct,
        "round_trip_retention": round_trip,
        "curve_progress": state.progress,
        "real_sol_reserves": state.real_sol_reserves,
        # Curve arithmetic is not yet reconciled against captured fills, so
        # this must not be treated as execution-grade evidence.
        "execution_verified": False,
    }
