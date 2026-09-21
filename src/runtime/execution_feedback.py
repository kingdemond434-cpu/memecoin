"""What the execution attempt taught the bidder.

`PriorityFeeOptimizer.get_optimal_fee` sets the compute-unit price on every
entry, and it selects from `self.landing_rates`. Nothing ever wrote to that
dict. It was empty for the life of the desk, so every call fell through to a
hardcoded base of 5,000 / 10,000 / 20,000 lamports scaled by a `competition`
argument that was itself hardcoded to 0.5 at both call sites -- a lookup
table of five magic numbers wearing an optimiser's name.

Found 2026-09-19 by sweeping for public methods with no caller, the same
sweep that found `PromotionLedger.demote` unwired. `record_attempt` was
written to close this loop and never called.

Kept out of `src/main.py` because that file has a line budget whose whole
purpose is to push exactly this kind of bookkeeping into a service.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


#: What `competition` was hardcoded to at both call sites, kept as the
#: answer for "nothing has been submitted yet" so the fallback is the old
#: behaviour stated out loud rather than a new invention. It stops being
#: used on the first attempt.
UNMEASURED_COMPETITION = 0.5


def fee_competition(latency: Any) -> float:
    """How often the desk is currently LOSING the race, measured.

    A free function rather than a method because both call sites are in the
    entry path and neither should depend on the caller's type: the desk is
    assembled from mixins in production and from namespaces in tests, and a
    bid that raises AttributeError over bookkeeping is a lost entry.

    The latency ledger already counts the outcome of every submission, and
    the share that failed is what competition means here.
    """
    outcomes = getattr(latency, "outcomes", None) or {}
    try:
        landed = int(outcomes.get("entered", 0))
        lost = int(outcomes.get("submit_failed", 0))
    except (AttributeError, TypeError, ValueError):
        return UNMEASURED_COMPETITION
    total = landed + lost
    return UNMEASURED_COMPETITION if total <= 0 else lost / total


class ExecutionFeedback:
    """Mixin: tell the bidder how its bid did."""

    def _record_fee_outcome(self, priority_fee: int, landed: bool,
                            trace: Any) -> None:
        """Teach the fee optimiser what its own bid did.

        `get_optimal_fee` selects from `landing_rates`, and nothing wrote to
        that dict, so it was empty forever and every call returned the
        hardcoded base. The optimiser could not have learned anything.
        """
        optimiser = getattr(self, "fee_optimizer", None)
        recorder = getattr(optimiser, "record_attempt", None)
        if not callable(recorder):
            return
        micros = getattr(trace, "elapsed_us", None)
        latency_ms = int((micros() if callable(micros) else 0) / 1000)
        try:
            recorder(priority_fee, landed, latency_ms)
        except Exception as exc:  # pragma: no cover - accounting only
            logger.debug("fee outcome not recorded: %s", exc)
