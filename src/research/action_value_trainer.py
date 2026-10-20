"""Training the action-value policy, with a gate it cannot talk its way past.

Every exit-policy search has the same failure available to it: raise the win
rate, cut the drawdown, and quietly stop holding the trades that pay for
everything. Aggregate metrics improve on every axis while the book gets worse,
because a book whose returns come from a handful of tokens is destroyed by
whatever sells those tokens early. The improvement is real and the wealth is
gone.

So the acceptance criteria here are asymmetric on purpose. A candidate must
beat the incumbent on chronological out-of-sample growth, AND separately
demonstrate that it did not do so by killing the right tail. Failing either
rejects it. There is no aggregate score good enough to buy its way past the
tail check, because that is precisely the trade a search will otherwise make.

Chronological, never random. Splitting a launch dataset at random puts the
same market regime on both sides and every candidate looks like it
generalises. The split here is by launch time, whole launches only, so a
candidate is tested on a market it has not seen.
"""

import json
import logging
import math
import argparse
import gzip
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.research.lifecycle_replay import (
    DEFAULT_EXIT_RULES, Cell, Lifecycle, Mark, replay_lifecycle,
    sniper_scoreboard,
)

logger = logging.getLogger(__name__)

ACTION_VALUE_ARTIFACT_VERSION = 1

# A candidate may not reduce tail capture on the monster subset by more than
# this, however good its aggregate looks. Chosen tight: the whole point is
# that this constraint binds.
MAX_TAIL_CAPTURE_REGRESSION = 0.05
# Nor may it raise the share of big winners exited far too early.
MAX_PREMATURE_EXIT_REGRESSION = 0.02
# Below this many out-of-sample launches nothing is measured, only sampled.
MIN_OOS_LAUNCHES = 40
# A launch is a "monster" for gate purposes when this much was feasible.
MONSTER_FEASIBLE_MULTIPLE = 10.0


@dataclass
class PolicyMetrics:
    """What one policy did on one set of launches."""

    launches: int
    priced_cells: int
    mean_net_sol: Optional[float]
    mean_log_growth: Optional[float]
    tail_capture_on_monsters: Optional[float]
    premature_exit_rate: Optional[float]
    monster_launches: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _log_growth(net_sol: float, stake_sol: float) -> Optional[float]:
    """Log wealth change from one cell, or None when the stake is unknown.

    Growth, not profit: a book compounds, so summing profit ranks a policy
    that risks everything above one that risks a sensible fraction.
    """
    if stake_sol <= 0:
        return None
    wealth = 1.0 + net_sol / stake_sol
    return math.log(wealth) if wealth > 0 else None


def measure_policy(cells: Sequence[Cell], launches: int) -> PolicyMetrics:
    """Score one policy's replayed cells, including the tail statistics."""
    priced = [cell for cell in cells if cell.ok]
    monsters = [cell for cell in priced
                if (cell.max_feasible_multiple or 0) >= MONSTER_FEASIBLE_MULTIPLE]
    captures = [cell.tail_capture for cell in monsters if cell.tail_capture is not None]
    # Exited below half of what was feasible, on a launch where a monster was
    # available. This is the leak the gate exists to detect.
    premature = [cell for cell in monsters
                 if (cell.exit_multiple or 0) < 0.5 * (cell.max_feasible_multiple or 0)]
    growths = [value for value in
               (_log_growth(cell.net_sol, cell.filled_sol) for cell in priced)
               if value is not None]

    return PolicyMetrics(
        launches=launches,
        priced_cells=len(priced),
        mean_net_sol=(float(np.mean([cell.net_sol for cell in priced])) if priced else None),
        mean_log_growth=(float(np.mean(growths)) if growths else None),
        tail_capture_on_monsters=(float(np.mean(captures)) if captures else None),
        premature_exit_rate=(len(premature) / len(monsters) if monsters else None),
        monster_launches=len(monsters),
    )


def chronological_split(lifecycles: Sequence[Lifecycle],
                        train_fraction: float = 0.7) -> Tuple[List[Lifecycle], List[Lifecycle]]:
    """Split by launch time, whole launches only.

    A random split puts the same regime on both sides and every candidate
    looks like it generalises. Whole launches, because splitting one launch's
    marks across the boundary lets a candidate see its own future.
    """
    ordered = sorted(lifecycles, key=lambda life: life.created_at)
    cut = int(len(ordered) * max(0.0, min(1.0, train_fraction)))
    return ordered[:cut], ordered[cut:]


@dataclass
class GateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    incumbent: Optional[Dict[str, Any]] = None
    candidate: Optional[Dict[str, Any]] = None


def tail_preservation_gate(
    incumbent: PolicyMetrics,
    candidate: PolicyMetrics,
    max_tail_regression: float = MAX_TAIL_CAPTURE_REGRESSION,
    max_premature_regression: float = MAX_PREMATURE_EXIT_REGRESSION,
    min_launches: int = MIN_OOS_LAUNCHES,
) -> GateResult:
    """Accept only if growth improved AND the right tail survived.

    Asymmetric on purpose. No aggregate score is good enough to buy past the
    tail check, because trading the tail for the aggregate is exactly the move
    a search makes when only the aggregate is scored.
    """
    reasons: List[str] = []

    if candidate.launches < min_launches:
        reasons.append(
            f"only {candidate.launches} out-of-sample launches, below {min_launches}; "
            "this is a sample, not a measurement")

    if candidate.mean_log_growth is None or incumbent.mean_log_growth is None:
        reasons.append("log growth unmeasurable on one side; nothing to compare")
    elif candidate.mean_log_growth <= incumbent.mean_log_growth:
        reasons.append(
            f"growth did not improve: {candidate.mean_log_growth:.6f} vs "
            f"{incumbent.mean_log_growth:.6f}")

    # The tail checks. A candidate with no monsters to be measured on has not
    # demonstrated tail preservation -- it has demonstrated nothing about it,
    # which is not the same and must not pass.
    if candidate.monster_launches == 0 or incumbent.tail_capture_on_monsters is None:
        reasons.append(
            "no monster-subset evidence; tail preservation is unproven, not proven")
    else:
        if candidate.tail_capture_on_monsters is None:
            reasons.append("candidate tail capture unmeasurable on the monster subset")
        else:
            regression = incumbent.tail_capture_on_monsters - candidate.tail_capture_on_monsters
            if regression > max_tail_regression:
                reasons.append(
                    f"tail capture fell {regression:.4f} on the monster subset, "
                    f"above the {max_tail_regression:.4f} allowance")
        if (candidate.premature_exit_rate is not None
                and incumbent.premature_exit_rate is not None):
            worsening = candidate.premature_exit_rate - incumbent.premature_exit_rate
            if worsening > max_premature_regression:
                reasons.append(
                    f"premature exits rose {worsening:.4f}, above the "
                    f"{max_premature_regression:.4f} allowance")

    return GateResult(passed=not reasons, reasons=reasons,
                      incumbent=incumbent.to_dict(), candidate=candidate.to_dict())


def evaluate_candidate(
    lifecycles: Sequence[Lifecycle],
    incumbent_rule_name: str,
    candidate_rule_name: str,
    exit_rules: Dict[str, Callable],
    delays: Sequence[float] = (0.0,),
    sizes: Sequence[float] = (1.0,),
    train_fraction: float = 0.7,
    round_trip_cost: float = 0.02,
) -> Tuple[GateResult, Dict[str, Any]]:
    """Replay both policies on held-out launches and run the gate."""
    _, oos = chronological_split(lifecycles, train_fraction)
    if not oos:
        return (GateResult(passed=False, reasons=["no out-of-sample launches"]), {})

    def cells_for(rule_name: str) -> List[Cell]:
        rule = {rule_name: exit_rules[rule_name]}
        collected: List[Cell] = []
        for life in oos:
            collected.extend(replay_lifecycle(life, delays, sizes, rule, round_trip_cost))
        return collected

    incumbent_cells = cells_for(incumbent_rule_name)
    candidate_cells = cells_for(candidate_rule_name)
    result = tail_preservation_gate(
        measure_policy(incumbent_cells, len(oos)),
        measure_policy(candidate_cells, len(oos)),
    )
    report = {
        "incumbent_rule": incumbent_rule_name,
        "candidate_rule": candidate_rule_name,
        "oos_launches": len(oos),
        "incumbent_scoreboard": sniper_scoreboard(incumbent_cells, len(oos)),
        "candidate_scoreboard": sniper_scoreboard(candidate_cells, len(oos)),
    }
    return result, report


def select_policy(
    lifecycles: Sequence[Lifecycle],
    exit_rules: Dict[str, Callable],
    incumbent_rule_name: str,
    delays: Sequence[float] = (0.0,),
    sizes: Sequence[float] = (1.0,),
    train_fraction: float = 0.7,
) -> Dict[str, Any]:
    """Pick the best candidate that clears the gate, or ship nothing.

    Shipping nothing is the expected outcome most of the time and is not a
    failure of the trainer. A search that always finds a winner has found
    overfitting.
    """
    if len(lifecycles) < MIN_OOS_LAUNCHES:
        return {"status": "DATA_BLOCKED", "shipped": None,
                "detail": (f"{len(lifecycles)} launches, below the "
                           f"{MIN_OOS_LAUNCHES} needed for a chronological verdict")}

    evaluated: List[Dict[str, Any]] = []
    winner: Optional[Tuple[str, float]] = None
    for name in exit_rules:
        if name == incumbent_rule_name:
            continue
        gate, report = evaluate_candidate(
            lifecycles, incumbent_rule_name, name, exit_rules, delays, sizes, train_fraction)
        growth = (gate.candidate or {}).get("mean_log_growth")
        evaluated.append({"candidate": name, "passed": gate.passed,
                          "reasons": gate.reasons, "mean_log_growth": growth})
        if gate.passed and growth is not None and (winner is None or growth > winner[1]):
            winner = (name, growth)

    return {
        "status": "OK",
        "shipped": winner[0] if winner else None,
        "incumbent": incumbent_rule_name,
        "evaluated": evaluated,
        "detail": ("no candidate cleared the tail-preservation gate"
                   if winner is None else f"{winner[0]} cleared the gate"),
    }


def save_report(model_dir: Path, report: Dict[str, Any]) -> Path:
    """Persist the verdict, including a rejection.

    A rejected run is the more informative record: it says the search looked
    and found nothing worth shipping, which is what stops the same candidate
    being re-proposed next week as though it were new.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "last_action_value_report.json"
    payload = {"artifact_version": ACTION_VALUE_ARTIFACT_VERSION,
               "generated_at": time.time(), **report}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def lifecycles_from_storage(storage: Path) -> List[Lifecycle]:
    """Rebuild launch lifecycles from the episodes on disk.

    The trainer had no way in. Every other trainer in this package reads
    `--storage` and turns it into rows; this one took `Sequence[Lifecycle]`
    from a caller that did not exist, so `python -m` on it imported the
    module, defined some functions and exited 0 having done nothing -- which
    a supervisor reads as success followed by a missing report.
    """
    lifecycles: List[Lifecycle] = []
    for path in sorted(storage.rglob("*.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, ValueError):
            continue
        outcome = episode.get("final_outcome") or {}
        if outcome.get("status") not in (None, "OK"):
            continue
        created = float(episode.get("created_at", 0) or 0)
        if created <= 0:
            continue
        marks: List[Mark] = []
        for item in episode.get("market_observations") or []:
            try:
                timestamp = float(item.get("timestamp", 0) or 0)
                multiple = float(item.get("price_multiple", 0) or 0)
            except (TypeError, ValueError):
                continue
            if timestamp <= 0 or multiple <= 0:
                continue
            depth = item.get("executable_sol")
            marks.append(Mark(
                timestamp=timestamp, multiple=multiple,
                executable_sol=(float(depth) if depth is not None else None),
                feasible=bool(item.get("feasible", True))))
        if len(marks) < 2:
            # A lifecycle with one mark has no path to replay, and replaying
            # the outcome alone would be scoring a decision nobody could
            # have made.
            continue
        marks.sort(key=lambda mark: mark.timestamp)
        rug_time = outcome.get("rug_time")
        lifecycles.append(Lifecycle(
            token=str(episode.get("token", "") or path.stem),
            created_at=created, marks=marks,
            migrated=bool(outcome.get("migrated")),
            rugged=bool(outcome.get("rugged")),
            rug_time=(float(rug_time) if rug_time is not None else None)))
    return lifecycles


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default="data/launch_episodes")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--min-lifecycles", type=int, default=200)
    # The rule a candidate has to beat. "hold" is the honest incumbent: it is
    # what the desk does when it has no policy, and a candidate that cannot
    # beat doing nothing has not earned a promotion.
    parser.add_argument("--incumbent", default="hold")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lifecycles = lifecycles_from_storage(Path(args.storage))
    model_dir = Path(args.model_dir)
    if len(lifecycles) < args.min_lifecycles:
        # DATA_BLOCKED is written, not merely returned. A supervisor that
        # sees no report cannot tell "too little data" from "the trainer
        # crashed", and it should never have to guess.
        save_report(model_dir, {
            "status": "DATA_BLOCKED",
            "lifecycles": len(lifecycles),
            "reason": (f"{len(lifecycles)} replayable lifecycle(s); "
                       f"{args.min_lifecycles} required. A lifecycle needs at "
                       "least two observed executable marks, so launches that "
                       "died before anything traded contribute nothing"),
        })
        print(json.dumps({"status": "DATA_BLOCKED",
                          "lifecycles": len(lifecycles)}, indent=1))
        return 0

    report = select_policy(lifecycles, DEFAULT_EXIT_RULES, args.incumbent)
    save_report(model_dir, report)
    print(json.dumps({"status": report.get("status"),
                      "lifecycles": len(lifecycles)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
