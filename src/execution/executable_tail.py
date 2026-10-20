"""What a 1000x is actually worth to a position, not to a chart.

A memecoin can print a 1000x market-cap multiple while the position holding it
realises a fraction of that, because selling into the venue that produced the
number collapses it. Optimising for the printed multiple is optimising for a
screenshot, and the whole point of a tail strategy is that the rare event has
to be CAPTURED, not merely witnessed.

So each multiple gets three numbers instead of one:

    P(reach m)  x  exitable fraction at m  x  net multiple after impact

The first comes from the survival curve. The other two are computed here, from
the desk's own curve arithmetic, by projecting the venue forward to the state
it would be in at that price and quoting a real sale into it.

The projection is exact rather than approximate. On a constant-product curve
price = S**2 / k, so reaching multiple m means the SOL reserve has grown by
exactly sqrt(m) -- verified against `quote_buy` to four decimals at 1, 5, 20
and 60 SOL of inflow. Every lamport of that growth is real inflow from real
buyers, which is what a seller can be paid out of.

**And the finding that falls straight out of the arithmetic: a Pump bonding
curve cannot price a 100x exit at all.** Migration happens at roughly 85 SOL
of real reserves, which from a T0 entry is a price multiple of 14.4x. Every
20x, 100x and 1000x exit this desk will ever take happens on the migrated
pool, never on the curve. A tail model quoting exits against the bonding
curve past 15x is quoting a venue the token has already left.

So the ladder crosses venues. Below migration it is priced on the curve; at
and above it, on the OBSERVED pool.

There is deliberately no fallback to a synthesised one, and the reason is
worth recording because the synthesis was written first and then removed. The
curve sells 99.98% of its own inventory by the time it completes -- 793.1e12
tokens down to 1.5e11 -- so a pool seeded from what the curve has left would
hold less base than a single 0.3 SOL position bought at T0. That is plainly
not what the program seeds a migrated pool with, and no verified figure for
what it does seed was available here. `pool_at_migration` keeps the
arithmetic, and `model_migration` keeps it switched off, so it can be turned
on the day a captured migration confirms the seeding rather than today on the
strength of an assumption.

The consequence is a real finding rather than a gap in this module: **without
observed pool state, every multiple above 14.4x is unmeasurable to this
desk.** That is the entire range the tail strategy exists for, and the rungs
say DATA_BLOCKED rather than quoting a venue nobody has looked at.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EXECUTABLE_TAIL_SCHEMA_VERSION = "v1"

#: Real SOL reserves at which a Pump curve completes and trading moves to a
#: PumpSwap pool. The exact figure is a program constant; what matters here is
#: that a curve projection past it is a projection of a venue that no longer
#: exists.
MIGRATION_REAL_SOL_LAMPORTS = 85_000_000_000

#: Staleness bound reused from the pool module, so an OBSERVED pool is held to
#: the same freshness rule here as everywhere else.
from src.chains.pumpswap_curve import DEFAULT_MAX_STATE_AGE_S

#: The multiples an executable tail curve is reported at.
DEFAULT_TAIL_LADDER: Tuple[float, ...] = (
    2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0)


@dataclass
class ExecutableRung:
    """One multiple, and what reaching it would really be worth."""

    multiple: float
    status: str = "DATA_BLOCKED"
    #: P(M >= multiple), from the survival curve. None when unanswerable.
    survival: Optional[float] = None
    survival_extrapolated: bool = False
    #: Share of the position sellable at this price inside the impact bound.
    exitable_fraction: Optional[float] = None
    #: The multiple actually realised after impact and fees on the way out.
    net_multiple: Optional[float] = None
    #: survival x exitable_fraction x (net_multiple - 1), the contribution
    #: this rung makes to expected capturable upside.
    capturable: Optional[float] = None
    venue: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def project_curve(state: Any, multiple: float) -> Optional[Any]:
    """The same bonding curve after its price has multiplied by ``multiple``.

    Exact, not approximate: price = S**2 / k on a constant product over
    virtual reserves, so S scales as sqrt(multiple) and T follows from k.
    Every lamport of the increase is real inflow, so real reserves rise by the
    same amount -- which is precisely what bounds what a seller can be paid.

    Returns None past migration. A curve that has completed is not a venue,
    and projecting through that boundary would quote a sale into a pool of
    reserves the program has already moved somewhere else.
    """
    from src.chains.pump_curve import BondingCurveState
    if state is None or multiple <= 0 or not getattr(state, "tradeable", False):
        return None
    sol = int(getattr(state, "virtual_sol_reserves", 0) or 0)
    token = int(getattr(state, "virtual_token_reserves", 0) or 0)
    if sol <= 0 or token <= 0:
        return None
    product = sol * token
    grown = int(sol * math.sqrt(float(multiple)))
    if grown <= 0:
        return None
    inflow = grown - sol
    real_sol = int(getattr(state, "real_sol_reserves", 0) or 0) + max(0, inflow)
    if real_sol >= MIGRATION_REAL_SOL_LAMPORTS:
        return None
    new_token = product // grown
    sold = token - new_token
    real_token = int(getattr(state, "real_token_reserves", 0) or 0) - max(0, sold)
    return BondingCurveState(
        virtual_token_reserves=new_token,
        virtual_sol_reserves=grown,
        real_token_reserves=max(0, real_token),
        real_sol_reserves=real_sol,
        token_total_supply=int(getattr(state, "token_total_supply", 0) or 0),
        complete=False,
        creator=str(getattr(state, "creator", "") or ""))


def migration_multiple(state: Any) -> Optional[float]:
    """The price multiple at which this curve completes, from where it is now.

    The single most useful number for a tail strategy on Pump: above it, every
    exit is a pool exit, and any tail model quoting the curve is quoting a
    venue the token has left.
    """
    if state is None or not getattr(state, "tradeable", False):
        return None
    sol = int(getattr(state, "virtual_sol_reserves", 0) or 0)
    real = int(getattr(state, "real_sol_reserves", 0) or 0)
    if sol <= 0:
        return None
    headroom = MIGRATION_REAL_SOL_LAMPORTS - real
    if headroom <= 0:
        return 1.0
    return float(((sol + headroom) / sol) ** 2)


def pool_at_migration(state: Any) -> Optional[Any]:
    """The pool a completing curve would seed, from the curve's own reserves.

    NOT USED BY DEFAULT, and the reason is the point. Seeding a pool from what
    the curve has left over produces a pool holding 1.5e11 base tokens, which
    is less than a single 0.3 SOL T0 position -- so the model says a position
    can exit 0.1% of itself at 20x, which is an artefact of the seeding
    assumption rather than a fact about the venue. The program seeds the
    migrated pool from a separate reserve, and no verified figure for it was
    available here.

    Kept because the arithmetic is right once the seeding is known: pass
    `model_migration=True` the day a captured migration confirms what moves.
    Until then the honest answer above migration is that the desk needs to
    observe the pool.
    """
    from src.chains.pumpswap_curve import PumpSwapPoolState
    at_migration = project_curve_to_migration(state)
    if at_migration is None:
        return None
    base = int(getattr(at_migration, "real_token_reserves", 0) or 0)
    quote = int(getattr(at_migration, "real_sol_reserves", 0) or 0)
    if base <= 0 or quote <= 0:
        return None
    from src.execution.pump_fees import LEGACY_TOTAL_FEE_BPS
    return PumpSwapPoolState(
        pool="modelled_migration", base_mint="modelled", quote_mint="modelled",
        base_reserves=base, quote_reserves=quote, virtual_quote_reserves=0,
        total_fee_bps=LEGACY_TOTAL_FEE_BPS,
        base_supply=int(getattr(state, "token_total_supply", 0) or 0),
        source="modelled_migration")


def project_curve_to_migration(state: Any) -> Optional[Any]:
    """The curve at the instant it completes."""
    multiple = migration_multiple(state)
    if multiple is None:
        return None
    # A hair under, so `project_curve`'s own migration guard does not refuse
    # the very state being asked for.
    return project_curve(state, multiple * 0.999)


def project_pool(pool: Any, multiple: float) -> Optional[Any]:
    """The same pool after its price has multiplied by ``multiple``.

    Identical arithmetic to the curve: constant product means the quote
    reserve scales as sqrt(multiple) and the base reserve follows from k.
    """
    import dataclasses
    if pool is None or multiple <= 0 or not getattr(pool, "tradeable", False):
        return None
    quote = int(getattr(pool, "effective_quote_reserves", 0) or 0)
    base = int(getattr(pool, "base_reserves", 0) or 0)
    if quote <= 0 or base <= 0:
        return None
    product = quote * base
    grown = int(quote * math.sqrt(float(multiple)))
    if grown <= 0:
        return None
    return dataclasses.replace(
        pool, quote_reserves=grown, virtual_quote_reserves=0,
        base_reserves=max(1, product // grown))


def _pool_exitable(projected: Any, position_tokens: int, cost_lamports: int,
                   acceptable_impact: float, *, modelled: bool = False
                   ) -> Tuple[Optional[float], Optional[float], str]:
    """The same question asked of a pool rather than a curve.

    A pool quote refuses stale state, and rightly: a reserve reading from two
    minutes ago is not what a sale would meet. A PROJECTED pool has no age at
    all -- it is a hypothetical future state, not an old reading -- so the
    staleness guard is lifted for it and only for it. Leaving it on would
    reject every modelled rung as out of date, which is a staleness test
    applied to something that was never fresh in the first place.
    """
    from src.chains.pumpswap_curve import quote_sell, sell_capacity_base
    if position_tokens <= 0 or cost_lamports <= 0:
        return None, None, "no position to price"
    age = math.inf if modelled else DEFAULT_MAX_STATE_AGE_S
    capacity = sell_capacity_base(projected, max_impact_pct=acceptable_impact,
                                  max_age_s=age)
    if capacity <= 0:
        return None, None, "nothing sellable inside the impact bound"
    sellable = min(int(position_tokens), int(capacity))
    quote = quote_sell(projected, sellable, max_age_s=age)
    if quote.data_status != "OK" or quote.output_amount <= 0:
        return None, None, f"sale unquotable: {quote.reason or quote.data_status}"
    return (sellable / float(position_tokens),
            quote.output_amount / float(cost_lamports), "")


def _exitable(projected: Any, position_tokens: int, cost_lamports: int,
              acceptable_impact: float) -> Tuple[Optional[float], Optional[float], str]:
    """(exitable fraction, net multiple, detail) selling into a projected venue."""
    from src.chains.pump_curve import quote_sell, sell_capacity_lamports
    if position_tokens <= 0 or cost_lamports <= 0:
        return None, None, "no position to price"
    capacity = sell_capacity_lamports(projected, max_impact_pct=acceptable_impact)
    if capacity <= 0:
        return None, None, "nothing sellable inside the impact bound"
    sellable = min(int(position_tokens), int(capacity))
    quote = quote_sell(projected, sellable)
    if quote.data_status != "OK" or quote.output_amount <= 0:
        return None, None, f"sale unquotable: {quote.reason or quote.data_status}"
    fraction = sellable / float(position_tokens)
    # The net multiple is what the WHOLE position realises: the part that gets
    # out at this price, plus nothing for the part that does not. Crediting
    # the trapped remainder at the marked price is exactly the arithmetic that
    # turns an unrealisable 1000x into a reported one.
    net = quote.output_amount / float(cost_lamports)
    return fraction, net, ""


def executable_tail_curve(
        state: Any, position_tokens: int, cost_lamports: int, *,
        survival: Optional[Any] = None,
        pool: Any = None,
        multiples: Sequence[float] = DEFAULT_TAIL_LADDER,
        acceptable_impact: float = 0.10,
        model_migration: bool = False) -> List[ExecutableRung]:
    """Each multiple, with the probability of reaching it and of keeping it.

    ``survival`` is any callable taking a multiple and returning
    ``(value, basis, extrapolated)`` -- the continuation model's own method.
    Absent, the rungs still report executability, which is the half this
    module exists to compute.
    """
    rungs: List[ExecutableRung] = []
    for multiple in multiples:
        rung = ExecutableRung(multiple=float(multiple))
        if survival is not None:
            try:
                value, basis, extrapolated = survival(float(multiple))
            except Exception as exc:  # pragma: no cover - defensive
                value, basis, extrapolated = None, f"survival raised: {exc}", False
            rung.survival = value
            rung.survival_extrapolated = bool(extrapolated)
            if value is None:
                rung.detail = str(basis)
        projected = project_curve(state, float(multiple))
        if projected is not None:
            rung.venue = "bonding_curve"
            fraction, net, detail = _exitable(
                projected, position_tokens, cost_lamports, acceptable_impact)
        else:
            # Past migration. The exit happens on the pool, so price it there
            # -- on the observed pool when the desk has one, otherwise on the
            # one migration is known to seed, labelled as modelled.
            venue = pool if pool is not None else (
                pool_at_migration(state) if model_migration else None)
            if venue is None:
                rung.venue = "migrated_or_unmeasurable"
                rung.detail = (rung.detail or
                               "past migration and no pool state to price "
                               "against; this exit is unmeasurable")
                rungs.append(rung)
                continue
            rung.venue = ("pool" if pool is not None else "modelled_pool")
            # The multiple is measured from the POSITION'S ENTRY price, and
            # the pool is already trading somewhere above it. Projecting the
            # pool by the whole multiple would compound the move the token has
            # already made onto itself -- which reported a 20x rung realising
            # 114x, because the pool was already ~14x above entry before the
            # projection multiplied it again.
            #
            # So the pool is projected by the ratio still to come: the target
            # price over the price it is at now, both in the same units.
            entry_price = float(cost_lamports) / float(max(1, position_tokens))
            now_price = float(getattr(venue, "price_quote_per_base", 0.0) or 0.0)
            if now_price <= 0 or entry_price <= 0:
                rung.detail = rung.detail or "pool price unreadable"
                rungs.append(rung)
                continue
            remaining = (float(multiple) * entry_price) / now_price
            if remaining < 1.0:
                rung.detail = (rung.detail or
                               f"{multiple:g}x is below the price the pool is "
                               f"already at")
                rungs.append(rung)
                continue
            projected_pool = project_pool(venue, remaining)
            if projected_pool is None:
                rung.detail = rung.detail or "pool projection unavailable"
                rungs.append(rung)
                continue
            fraction, net, detail = _pool_exitable(
                projected_pool, position_tokens, cost_lamports,
                acceptable_impact, modelled=pool is None)
        rung.exitable_fraction = fraction
        rung.net_multiple = net
        if fraction is None or net is None:
            rung.detail = rung.detail or detail
            rungs.append(rung)
            continue
        rung.status = "OK"
        if rung.survival is not None:
            rung.capturable = float(rung.survival * fraction * max(0.0, net - 1.0))
        rungs.append(rung)
    return rungs


def capturable_upside(rungs: Sequence[ExecutableRung]) -> Dict[str, Any]:
    """Summed capturable contribution, and how much of the ladder was priced.

    The coverage number matters as much as the total. A capturable upside
    summed over three rungs out of ten is not a small answer, it is a mostly
    unanswered one, and reporting it as a scalar would hide that.
    """
    priced = [rung for rung in rungs if rung.capturable is not None]
    answered = [rung for rung in rungs if rung.status == "OK"]
    on_curve = [rung for rung in rungs if rung.venue == "bonding_curve"]
    modelled = [rung for rung in rungs if rung.venue == "modelled_pool"
                and rung.status == "OK"]
    return {
        "schema": EXECUTABLE_TAIL_SCHEMA_VERSION,
        "status": "OK" if priced else "DATA_BLOCKED",
        "capturable_upside": (sum(rung.capturable for rung in priced)
                              if priced else None),
        "rungs": len(rungs),
        "priced_rungs": len(priced),
        "executable_rungs": len(answered),
        "rungs_on_the_curve": len(on_curve),
        # Priced against a venue that is modelled rather than observed. Kept
        # separate because "we computed it" and "we saw it" are different
        # claims and only one of them is evidence.
        "rungs_on_a_modelled_pool": len(modelled),
        "highest_priced_multiple": (max(rung.multiple for rung in priced)
                                    if priced else None),
        "ladder": [rung.to_dict() for rung in rungs],
        "detail": ("" if priced else
                   "no rung could be priced: either the survival curve "
                   "cannot answer or the position has no executable exit"),
    }
