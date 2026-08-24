"""Chronological selection of an exit policy from real observed price paths.

Replays evaluate_exit -- the exact function _manage_positions uses live --
against the route-feasible price path of each persisted launch episode, and
scores each candidate policy by mean realized log growth (the same objective
ElogwEngine optimizes for sizing). Episodes are split chronologically by
launch so no episode's own outcome informs the policy that is scored on it.

A candidate only ships if, on the held-out fold, it beats BOTH the current
default policy and a trivial hold-to-the-end baseline. That guards against
the usual failure mode of tuning exits on a lucky in-sample window and
discovering the "improvement" was noise.

Only route-feasible marks are used as exit prices: an exit the router could
not actually have filled is not a real exit, and treating a quoted mid as a
fill is how backtests learn to sell into liquidity that never existed.
"""

import argparse
import gzip
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.strategies.exit_policy import ExitPolicy, evaluate_exit, save_exit_policy

EXECUTION_COST = 0.003


def load_price_paths(storage: Path) -> List[Tuple[str, float, List[Tuple[float, float, bool]]]]:
    """Returns (token, created_at, [(elapsed_seconds, multiple, route_feasible)]) per episode."""
    paths: List[Tuple[str, float, List[Tuple[float, float, bool]]]] = []
    for path in storage.glob("*/*.json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if (episode.get("final_outcome") or {}).get("status") != "OK":
            continue
        created_at = float(episode.get("created_at", 0) or 0)
        observations = sorted(
            (item for item in (episode.get("market_observations") or [])
             if item.get("timestamp") is not None),
            key=lambda item: float(item["timestamp"]),
        )
        marks: List[Tuple[float, float, bool]] = []
        entry_price: Optional[float] = None
        for item in observations:
            multiple = float(item.get("price_multiple", 0) or 0)
            if multiple <= 0:
                price = float(item.get("price_usd", 0) or 0)
                if price <= 0:
                    continue
                if entry_price is None:
                    entry_price = price
                multiple = price / max(entry_price, 1e-12)
            feasible = item.get("route_feasible", item.get("feasible")) is True
            impact = float(item.get("price_impact_pct", 1) or 1)
            marks.append((float(item["timestamp"]) - created_at, multiple, feasible and impact <= 0.15))
        if len(marks) >= 2:
            paths.append((str(episode.get("token", path.stem)), created_at, marks))
    return sorted(paths, key=lambda item: item[1])


def simulate(policy: ExitPolicy, marks: Sequence[Tuple[float, float, bool]]) -> float:
    """Realized log growth of one episode under a policy, net of execution cost.

    Sells only at route-feasible marks. Any unsold remainder is valued at the
    last feasible mark; if the route never became feasible again, the
    remainder is written down to the observed final multiple, because an
    unexitable position is not worth its quoted mid.
    """
    remaining = 1.0
    proceeds = 0.0
    high_water = marks[0][1]
    stages: set = set()
    continuation = 0.0  # no trained predictor in replay; matches the live fallback
    last_feasible = next((multiple for _, multiple, feasible in marks if feasible), marks[0][1])

    for elapsed, multiple, feasible in marks:
        high_water = max(high_water, multiple)
        if feasible:
            last_feasible = multiple
        if remaining <= 0:
            break
        decision = evaluate_exit(policy, multiple, high_water, continuation, stages, elapsed)
        if not decision:
            continue
        reason, fraction = decision
        if reason == "profit_ratchet_cost_recovery":
            stages.add("cost_recovery")
        elif reason == "profit_ratchet_5x":
            stages.add("bank_5x")
        elif reason == "profit_ratchet_10x":
            stages.add("bank_10x")
        if not feasible:
            # The policy wanted out but the router could not fill; it does not
            # get credit for an exit that was not executable at this mark.
            continue
        sold = min(remaining, remaining * fraction if fraction < 1.0 else remaining)
        proceeds += sold * multiple
        remaining -= sold

    if remaining > 0:
        final_multiple, final_feasible = marks[-1][1], marks[-1][2]
        proceeds += remaining * (final_multiple if final_feasible else min(final_multiple, last_feasible))
    return math.log(max(proceeds - EXECUTION_COST, 1e-9))


def candidate_policies() -> Dict[str, ExitPolicy]:
    """A deliberately small, human-legible grid.

    Kept small on purpose: every extra candidate scored on the same held-out
    fold inflates the chance that the winner is just the luckiest draw rather
    than the best policy.
    """
    base = ExitPolicy.default()
    candidates = {"default": base}
    for stop in (0.60, 0.70, 0.80):
        for trail_wide in (0.70, 0.78, 0.86):
            for hold_seconds in (1800.0, 3600.0, 7200.0):
                key = f"stop{stop:.2f}_trail{trail_wide:.2f}_hold{int(hold_seconds)}"
                candidates[key] = replace(
                    base, hard_stop_multiple=stop, trail_ratio_wide=trail_wide,
                    max_hold_seconds=hold_seconds,
                )
    return candidates


def hold_baseline(marks: Sequence[Tuple[float, float, bool]]) -> float:
    final_multiple, final_feasible = marks[-1][1], marks[-1][2]
    if not final_feasible:
        feasible = [multiple for _, multiple, ok in marks if ok]
        final_multiple = min(final_multiple, feasible[-1] if feasible else final_multiple)
    return math.log(max(final_multiple - EXECUTION_COST, 1e-9))


def train_exit_policy(storage: Path, model_dir: Path, min_episodes: int = 60) -> Dict[str, Any]:
    paths = load_price_paths(storage)
    report: Dict[str, Any] = {"created_at": time.time(), "episodes": len(paths)}
    if len(paths) < min_episodes:
        report.update({"status": "DATA_BLOCKED", "reason": f"need_at_least_{min_episodes}_route_feasible_episodes"})
        _persist(model_dir, report)
        return report

    split_at = max(1, min(len(paths) - 1, int(len(paths) * 0.8)))
    train, oos = paths[:split_at], paths[split_at:]
    report.update({"train_episodes": len(train), "oos_episodes": len(oos),
                   "split": "strict_chronological_80_20"})

    candidates = candidate_policies()
    train_scores = {
        name: float(np.mean([simulate(policy, marks) for _, _, marks in train]))
        for name, policy in candidates.items()
    }
    best_name = max(train_scores, key=train_scores.get)
    best_policy = candidates[best_name]

    oos_best = float(np.mean([simulate(best_policy, marks) for _, _, marks in oos]))
    oos_default = float(np.mean([simulate(candidates["default"], marks) for _, _, marks in oos]))
    oos_hold = float(np.mean([hold_baseline(marks) for _, _, marks in oos]))

    passed = best_name != "default" and oos_best > oos_default and oos_best > oos_hold
    report.update({
        "status": "PASSED" if passed else "REJECTED",
        "selected_policy": best_name,
        "train_elogw": train_scores[best_name],
        "oos_elogw": oos_best,
        "oos_elogw_default_policy": oos_default,
        "oos_elogw_hold_baseline": oos_hold,
        "candidates_scored": len(candidates),
        "objective": "mean realized log growth over route-feasible exits, net of execution cost",
        "note": "continuation probability is 0 in replay (no trained predictor), matching the live fallback",
    })
    if passed:
        report["model_path"] = str(save_exit_policy(model_dir, best_policy, report))
    _persist(model_dir, report)
    return report


def _persist(model_dir: Path, report: Dict[str, Any]):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "last_exit_policy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Select a chronologically validated exit policy")
    parser.add_argument("--storage", default="data/launch_episodes")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--min-episodes", type=int, default=60)
    args = parser.parse_args()
    report = train_exit_policy(Path(args.storage), Path(args.model_dir), args.min_episodes)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("status") in {"PASSED", "DATA_BLOCKED", "REJECTED"} else 1)


if __name__ == "__main__":
    main()
