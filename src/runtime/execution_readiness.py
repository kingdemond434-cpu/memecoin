"""Everything the execution path already knows, asked before the trade.

Nine methods across six modules were built and never called: the exit template
for this mint, its staged ladder, the impact bound a given size fits inside,
what the competition is paying at this launch age, which feed is actually
carrying events, who the leader is for the slot we would land in, and whether
the challenger route has earned the right to carry a fill.

Every one of them answers a question the desk asks implicitly at entry and
then guesses at. Composed here into one snapshot so the guess becomes a
reading, with two of them promoted from diagnostic to decisional:

* `reference_bid` sets a FLOOR under the priority fee. Everything in that
  corpus landed, so it says what the competition paid and can never say what
  we would have needed -- which makes it a floor and disqualifies it as a
  target. Bidding under the observed market is a choice to lose the race.

* `should_route_through_challenger` decides the route. It is the only question
  the execution path is allowed to ask of the shadow, it latches on demotion,
  and it answered nobody.

The rest are recorded on the decision rather than acted on, and the difference
is stated per field rather than blurred: a readiness snapshot that silently
mixes "this changed the trade" with "this was written down" teaches the
forward ledger nothing about either.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: How far under the observed market a bid may sit before it is raised to the
#: floor. Not zero: the corpus is survivorship-biased upward (only landed
#: transactions are in it), so matching it exactly would overpay on every
#: uncontested launch.
BID_FLOOR_FRACTION = 0.80


def _safe(call: Any, *args: Any, **kwargs: Any) -> Any:
    """Read a subsystem without letting a research surface break the trade."""
    if call is None:
        return None
    try:
        return call(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("execution readiness read failed: %s", exc)
        return None


def exit_readiness_for(desk: Any, token: str) -> Dict[str, Any]:
    """Can this position be left, and by which prepared route?

    A staged ladder built for every position and used for none is pure cost;
    a sell template that has expired is worse, because the position looks
    prepared and is not.
    """
    readiness = getattr(desk, "exit_readiness", None)
    staged = getattr(desk, "staged_exits", None)
    template = _safe(getattr(readiness, "template_for", None), token)
    ladder = _safe(getattr(staged, "ladder_for", None), token)
    return {
        "template_ready": template is not None,
        "ladder_ready": ladder is not None,
        "ladder_rungs": (len(getattr(ladder, "rungs", ()) or ())
                         if ladder is not None else None),
    }


def size_impact(desk: Any, token: str, size_tokens: int) -> Dict[str, Any]:
    """The tightest MEASURED impact bound this size fits inside.

    None is not "no impact". It means the size is worse than every bound the
    frontier measured, which is the answer that should make a position
    smaller and instead made it invisible.
    """
    if size_tokens <= 0:
        return {"impact_bound": None, "status": "DATA_BLOCKED",
                "detail": "no size to price"}
    from src.chains.pump_curve import quote_buy, quote_sell
    from src.execution.tradeability import curve_tradeability
    state = (getattr(desk, "_latest_curve_state", None) or {}).get(token)
    if state is None or not getattr(state, "tradeable", False):
        return {"impact_bound": None, "status": "DATA_BLOCKED",
                "detail": "no tradeable curve state"}
    report = _safe(curve_tradeability, state, quote_buy, quote_sell)
    frontier = getattr(report, "exit", None) if report is not None else None
    if frontier is None or not getattr(frontier, "ok", False):
        return {"impact_bound": None, "status": "DATA_BLOCKED",
                "detail": "no measured exit frontier"}
    bound = _safe(getattr(frontier, "impact_for", None), int(size_tokens))
    return {
        "impact_bound": bound,
        "status": "OK" if bound is not None else "DATA_BLOCKED",
        "detail": ("" if bound is not None else
                   "this size exceeds every measured bound, which is a "
                   "reason to size down rather than an absence of impact"),
    }


def landing_context(desk: Any, age_s: float) -> Dict[str, Any]:
    """Which feed is carrying, who leads the slot, and what others paid."""
    race = getattr(desk, "feed_race", None)
    schedule = getattr(desk, "leader_schedule", None)
    bids = getattr(desk, "observed_bids", None)
    slot = int(getattr(desk, "_latest_slot", 0) or 0)
    node = _safe(getattr(schedule, "node_for", None), slot) if slot else None
    return {
        "best_feed": _safe(getattr(race, "best_feed", None)),
        "leader_identity": (getattr(node, "identity", None)
                            if node is not None else None),
        "leader_known": node is not None,
        "reference_bid_lamports": _safe(
            getattr(bids, "reference_bid", None), float(age_s)),
    }


def bid_floor(desk: Any, age_s: float, proposed: int) -> int:
    """Raise a bid to the observed market when it sits under it.

    DECISIONAL. The corpus holds only transactions that landed, so it is a
    market quote and never a landing probability -- which makes it a floor
    and disqualifies it as a target. The floor is a fraction of it rather than
    all of it, because matching a survivorship-biased corpus exactly overpays
    on every launch nobody else wanted.
    """
    reference = _safe(getattr(getattr(desk, "observed_bids", None),
                              "reference_bid", None), float(age_s))
    if reference is None or reference <= 0:
        return int(proposed)
    return int(max(int(proposed), int(reference * BID_FLOOR_FRACTION)))


def route_choice(desk: Any) -> Dict[str, Any]:
    """Whether the challenger route has earned a fill.

    DECISIONAL. `should_route_through_challenger` is the only question the
    execution path may ask of the shadow: it answers False until the paired
    evidence says PROMOTED, and once DEMOTED it latches, so a route shown to
    lose does not re-litigate on a lucky streak.
    """
    shadow = getattr(desk, "raptor_shadow", None)
    eligible = _safe(getattr(shadow, "should_route_through_challenger", None))
    return {
        "challenger_eligible": bool(eligible),
        "route": "challenger" if eligible else "incumbent",
        "status": "OK" if shadow is not None else "DATA_BLOCKED",
    }


def pre_trade_readiness(desk: Any, token: str, size_tokens: int,
                        age_s: float) -> Dict[str, Any]:
    """One snapshot, with each field labelled by what it actually did."""
    return {
        "status": "OK",
        # Consumed by the trade.
        "decisional": {
            "route": route_choice(desk),
        },
        # Written down beside the trade, for the forward ledger to price.
        "recorded": {
            "exit": exit_readiness_for(desk, token),
            "size": size_impact(desk, token, size_tokens),
            "landing": landing_context(desk, age_s),
        },
    }
