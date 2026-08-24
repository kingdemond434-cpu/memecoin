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


def evaluate_exit(
    policy: ExitPolicy,
    multiple: float,
    high_water: float,
    continuation: float,
    stages_done: Set[str],
    elapsed_seconds: float,
) -> Optional[Tuple[str, float]]:
    """Returns (reason, sell_fraction) if the policy exits now, else None.

    Pure function: no I/O, no randomness, deterministic given its inputs, so
    the exact same logic can be replayed offline against historical price
    paths in the trainer as is used live in _manage_positions.
    """
    if multiple <= policy.hard_stop_multiple:
        return "hard_stop_loss", 1.0
    if multiple >= policy.cost_recovery_trigger_multiple and "cost_recovery" not in stages_done:
        return "profit_ratchet_cost_recovery", min(0.50, 1.0 / multiple)
    if multiple >= policy.bank_trigger_multiple_1 and "bank_5x" not in stages_done:
        return "profit_ratchet_5x", policy.bank_fraction_1
    if multiple >= policy.bank_trigger_multiple_2 and "bank_10x" not in stages_done:
        return "profit_ratchet_10x", policy.bank_fraction_2
    floor = policy.trail_floor(high_water, continuation)
    if multiple <= floor and high_water >= policy.trail_activation_high_water:
        return "adaptive_profit_trailing_stop", 1.0
    if elapsed_seconds >= policy.max_hold_seconds:
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
