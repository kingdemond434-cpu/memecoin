"""Parameterized position-exit policy.

The thresholds here (stop-loss level, ratchet triggers/fractions, trailing-stop
ratios, max hold time) used to be inline magic numbers in
MemecoinQuantDesk._manage_positions. They are still just guesses by default --
turning them into named fields does not make them evidence-based on its own.
What makes them evidence-based is src/research/exit_policy_trainer.py, which
selects a policy from a small candidate set by replaying real observed price
paths and only ships one that beats both the current default and a trivial
"hold to the end" baseline on held-out chronological data.

The daily-loss kill switch (ElogwEngine.max_daily_loss) is deliberately NOT
part of this policy and is never trainable: a policy that can optimize its own
worst-case circuit breaker is a policy that can learn to remove it.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

EXIT_POLICY_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ExitPolicy:
    hard_stop_multiple: float = 0.70
    cost_recovery_trigger_multiple: float = 2.0
    bank_trigger_multiple_1: float = 5.0
    bank_fraction_1: float = 0.25
    bank_trigger_multiple_2: float = 10.0
    bank_fraction_2: float = 0.20
    continuation_threshold: float = 0.15
    high_water_tight_trail_threshold: float = 5.0
    trail_ratio_wide: float = 0.78
    trail_ratio_tight: float = 0.68
    trail_ratio_mid: float = 0.58
    trail_floor_below_2x: float = 0.70
    trail_floor_above_2x: float = 1.10
    trail_activation_high_water: float = 1.5
    max_hold_seconds: float = 3600.0
    #: The ceiling when the position is under live, calibrated conviction.
    #: Zero means no extension, which is what a desk with no continuation
    #: model gets: the ordinary time stop, unchanged.
    #:
    #: The ordinary hold exists because a position nobody has an opinion
    #: about is capital doing nothing while the rug hazard runs. That reason
    #: evaporates when the model does have an opinion, and the old behaviour
    #: -- one flat hour -- closed a runner at 11x on the clock alone, which
    #: is the single most expensive thing an exit policy can do to a tail
    #: strategy. This is a CEILING and not a licence: the hard stop, the
    #: trail, the hazard exit and the ratchet all still apply throughout, and
    #: past this the time stop fires whatever the conviction says.
    max_conviction_hold_seconds: float = 0.0
    #: The trail that applies UNDER conviction, in place of the ordinary one.
    #:
    #: Conviction widens this stop; it does not remove it. Removing it is what
    #: the monster override did on its own -- the trail is not in
    #: NEVER_SUPPRESSED, so a position inside a monster state had nothing
    #: between its high water and the 0.70x hard stop, and a 100x that round
    #: tripped gave back everything while the state machine waited for three
    #: confirmations across two independent degrade dimensions it may never
    #: have been able to measure.
    #:
    #: 0.45 tolerates a 55% giveback, against 22-42% for the ordinary trail.
    #: That is wide enough to sit through the mid-run pullbacks that shake out
    #: an ordinary stop and narrow enough that a peak is banked as a peak.
    #: A judgement, like the ratios above it, and replaceable by the trainer.
    conviction_trail_ratio: float = 0.45

    @staticmethod
    def default() -> "ExitPolicy":
        """The pre-existing hand-picked thresholds, kept as the safe fallback
        when no chronologically validated policy has been trained yet."""
        return ExitPolicy()

    def trail_ratio(self, high_water: float, continuation: float) -> float:
        if continuation < self.continuation_threshold:
            return self.trail_ratio_wide
        if high_water >= self.high_water_tight_trail_threshold:
            return self.trail_ratio_tight
        return self.trail_ratio_mid

    def trail_floor(self, high_water: float, continuation: float) -> float:
        base = self.trail_floor_above_2x if high_water >= 2 else self.trail_floor_below_2x
        return max(base, high_water * self.trail_ratio(high_water, continuation))

    def conviction_trail_floor(self, high_water: float) -> float:
        """The wider floor a position under conviction is still held to."""
        base = self.trail_floor_above_2x if high_water >= 2 else self.trail_floor_below_2x
        return max(base, high_water * self.conviction_trail_ratio)


#: Exit reasons no conviction may stand down, at any state, ever.
#:
#: The monster override nulls whatever decision this function returns, and it
#: used to null ALL of them -- so a position inside a monster state that
#: collapsed through the hard stop had no stop. That was latent rather than
#: live only because `overrides_ordinary_exit` could never return True without
#: a calibrated probability; switching that on without this set would ship an
#: unbounded loss. `time_stop` is here too, because its conviction extension
#: is applied BELOW, inside this function: once it fires, the ceiling has
#: already been honoured and there is nothing left to override.
NEVER_SUPPRESSED: frozenset = frozenset({
    "hard_stop_loss", "time_stop", "conviction_trailing_stop"})


def evaluate_exit(
    policy: ExitPolicy,
    multiple: float,
    high_water: float,
    continuation: float,
    stages_done: Set[str],
    elapsed_seconds: float,
    conviction: bool = False,
) -> Optional[Tuple[str, float]]:
    """Returns (reason, sell_fraction) if the policy exits now, else None.

    `conviction` says a calibrated continuation model currently believes this
    position has substantial upside left. It extends ONLY the time stop, and
    only as far as `max_conviction_hold_seconds`. It is a parameter of this
    pure function rather than a branch in the caller so that the exit-policy
    trainer replays the conviction path exactly as the desk runs it.

    Pure function: no I/O, no randomness, deterministic given its inputs, so
    the exact same logic can be replayed offline against historical price
    paths in the trainer as is used live in _manage_positions.
    """
    if multiple <= policy.hard_stop_multiple:
        return "hard_stop_loss", 1.0
    # The ratchet is what conviction actually stands down, and it is skipped
    # HERE rather than nulled by the caller.
    #
    # Nulling it outside was a trap that only appeared once conviction could
    # be granted at all: the ratchet is checked before the trail, so under
    # conviction this function returned `profit_ratchet_10x` on every single
    # cycle, the caller discarded it, and `stages_done` was never updated --
    # so the trail below was UNREACHABLE for the whole life of the position.
    # A 100x that round-tripped had no giveback limit and rode to the hard
    # stop. Measured on a replayed path: exited at 5.0x from a 100x peak.
    if not conviction:
        if multiple >= policy.cost_recovery_trigger_multiple and "cost_recovery" not in stages_done:
            return "profit_ratchet_cost_recovery", min(0.50, 1.0 / multiple)
        if multiple >= policy.bank_trigger_multiple_1 and "bank_5x" not in stages_done:
            return "profit_ratchet_5x", policy.bank_fraction_1
        if multiple >= policy.bank_trigger_multiple_2 and "bank_10x" not in stages_done:
            return "profit_ratchet_10x", policy.bank_fraction_2
    activated = high_water >= policy.trail_activation_high_water
    if conviction:
        # Superseded, not skipped. The ordinary trail would fire here and be
        # nulled by the monster override, leaving the position with no
        # giveback limit at all; this is the same rule with a wider band, and
        # it is in NEVER_SUPPRESSED so nothing can null it.
        if multiple <= policy.conviction_trail_floor(high_water) and activated:
            return "conviction_trailing_stop", 1.0
    elif multiple <= policy.trail_floor(high_water, continuation) and activated:
        return "adaptive_profit_trailing_stop", 1.0
    hold_ceiling = policy.max_hold_seconds
    if conviction and policy.max_conviction_hold_seconds > hold_ceiling:
        hold_ceiling = policy.max_conviction_hold_seconds
    if elapsed_seconds >= hold_ceiling:
        return "time_stop", 1.0
    return None


def load_latest_exit_policy(model_dir: str) -> Tuple[Optional[ExitPolicy], Dict[str, Any]]:
    """Load the newest chronologically validated exit policy artifact, if any."""
    if not os.path.isdir(model_dir):
        return None, {}
    candidates = [
        os.path.join(model_dir, name) for name in os.listdir(model_dir)
        if name.startswith("exit-policy-") and name.endswith(".json")
    ]
    for candidate in sorted(candidates, key=os.path.getmtime, reverse=True):
        try:
            data = json.loads(Path(candidate).read_text(encoding="utf-8"))
            if data.get("artifact_version") != EXIT_POLICY_ARTIFACT_VERSION:
                raise ValueError("unsupported exit-policy artifact version")
            report = data.get("validation_report") or {}
            if report.get("status") != "PASSED":
                raise ValueError("exit-policy artifact lacks passed chronological validation")
            policy = ExitPolicy(**data["policy"])
            report = dict(report)
            report["model_path"] = candidate
            return policy, report
        except Exception as exc:
            logger.error("Exit-policy artifact rejected (%s): %s", candidate, exc)
    return None, {}


def save_exit_policy(model_dir: Path, policy: ExitPolicy, validation_report: Dict[str, Any]) -> Path:
    if validation_report.get("status") != "PASSED":
        raise RuntimeError("refusing to save an exit policy without passed chronological validation")
    model_dir.mkdir(parents=True, exist_ok=True)
    import time
    output = model_dir / f"exit-policy-{int(time.time())}.json"
    output.write_text(json.dumps({
        "artifact_version": EXIT_POLICY_ARTIFACT_VERSION,
        "policy": asdict(policy),
        "validation_report": validation_report,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return output
