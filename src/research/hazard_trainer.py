"""Chronological calibration trainer for the continuous rug-hazard model.

ContinuousRugHazardModel scores exit hazard from a hand-picked, uncalibrated
combination of weighted signals (src/strategies/rug_hazard.py). This module
turns that raw score into an empirically calibrated probability by replaying
the leakage-free half of that scoring (collect_observation_signals /
score_signals -- both pure functions of a token's own observation timeline)
against persisted point-in-time launch episodes, chronologically split by
episode so no token's outcome ever informs its own training fold.

Wallet-reputation-dependent signals (smart_wallet_exit, insider_sell) are
deliberately excluded from replay: wallet_intel reputation is live,
continuously-updated state that is never point-in-time snapshotted per
episode anywhere in this codebase, so replaying it against historical data
would silently leak information the model would not have had at that moment.
Calibrating only the leakage-free half is honest; faithfully "replaying"
the wallet half would not be.
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.strategies.rug_hazard import (
    DEFAULT_TRIGGER_WEIGHTS, HAZARD_ARTIFACT_VERSION, collect_observation_signals, score_signals,
)

CHECKPOINT_OFFSETS = (5, 10, 20, 30, 60, 120, 300, 600)
HORIZON_SECONDS = 30


def _episode_files(storage: Path):
    yield from storage.glob("*/*.json.gz")


def build_samples(storage: Path) -> List[Tuple[str, float, float, float]]:
    """Returns (token, episode_created_at, raw_hazard, label) sorted chronologically.

    label is 1.0 iff the episode ruggs within HORIZON_SECONDS of the
    checkpoint; only checkpoints strictly before any eventual rug are used,
    since a hazard score computed after the rug already happened is not a
    meaningful "will this rug in the next 30s" training example.
    """
    samples: List[Tuple[str, float, float, float]] = []
    for path in _episode_files(storage):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        outcome = episode.get("final_outcome") or {}
        if outcome.get("status") != "OK":
            continue
        observations = sorted(episode.get("market_observations") or [], key=lambda item: item.get("timestamp", 0))
        if not observations:
            continue
        created_at = float(episode.get("created_at", observations[0].get("timestamp", 0)))
        rugged = bool(outcome.get("rugged"))
        rug_time = outcome.get("rug_time")
        rug_time = float(rug_time) if isinstance(rug_time, (int, float)) else None
        last_offset = float(observations[-1].get("timestamp", created_at)) - created_at
        for offset in CHECKPOINT_OFFSETS:
            if offset > last_offset:
                break
            if rugged and rug_time is not None and offset >= rug_time:
                break
            now = created_at + offset
            window = [item for item in observations if float(item.get("timestamp", created_at)) <= now]
            if not window:
                continue
            raw = score_signals(collect_observation_signals(window, now), DEFAULT_TRIGGER_WEIGHTS)
            label = 1.0 if (rugged and rug_time is not None and 0 <= rug_time - offset <= HORIZON_SECONDS) else 0.0
            samples.append((str(episode.get("token", path.stem)), created_at, raw, label))
    return sorted(samples, key=lambda item: item[1])


def chronological_split(
    samples: List[Tuple[str, float, float, float]], train_fraction: float = 0.8,
) -> Tuple[List[Tuple[str, float, float, float]], List[Tuple[str, float, float, float]]]:
    first_seen: Dict[str, float] = {}
    for token, created_at, _, _ in samples:
        first_seen[token] = min(first_seen.get(token, created_at), created_at)
    ordered_tokens = sorted(first_seen, key=lambda token: (first_seen[token], token))
    if len(ordered_tokens) < 2:
        return [], []
    split_at = max(1, min(len(ordered_tokens) - 1, int(len(ordered_tokens) * train_fraction)))
    train_tokens = set(ordered_tokens[:split_at])
    train = [sample for sample in samples if sample[0] in train_tokens]
    oos = [sample for sample in samples if sample[0] not in train_tokens]
    return train, oos


def _brier(y_true: List[float], y_prob: List[float]) -> float:
    truth, probability = np.asarray(y_true), np.asarray(y_prob)
    return float(np.mean((truth - probability) ** 2))


def train_hazard_calibration(
    storage: Path, model_dir: Path, min_samples: int = 200, min_positive: int = 15,
) -> Dict[str, Any]:
    samples = build_samples(storage)
    report: Dict[str, Any] = {"created_at": time.time(), "samples": len(samples), "horizon_seconds": HORIZON_SECONDS}
    if len(samples) < min_samples:
        report.update({"status": "DATA_BLOCKED", "reason": f"need_at_least_{min_samples}_hazard_checkpoints"})
        _persist_report(model_dir, report)
        return report

    train, oos = chronological_split(samples)
    report.update({
        "train_samples": len(train), "oos_samples": len(oos),
        "train_episodes": len({item[0] for item in train}), "oos_episodes": len({item[0] for item in oos}),
    })
    train_positive = sum(item[3] for item in train)
    oos_positive = sum(item[3] for item in oos)
    report.update({"train_positive": train_positive, "oos_positive": oos_positive})
    if not train or not oos or train_positive < min_positive or oos_positive < max(5, min_positive // 3):
        report.update({"status": "DATA_BLOCKED", "reason": "insufficient_rug_positive_class_coverage"})
        _persist_report(model_dir, report)
        return report

    train_raw = [item[2] for item in train]
    train_label = [item[3] for item in train]
    oos_raw = [item[2] for item in oos]
    oos_label = [item[3] for item in oos]

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(train_raw, train_label)
    calibrated_oos = calibrator.predict(oos_raw)

    train_rate = float(np.mean(train_label))
    brier = _brier(oos_label, calibrated_oos)
    baseline_brier = _brier(oos_label, [train_rate] * len(oos_label))
    raw_brier = _brier(oos_label, np.clip(oos_raw, 0, 1))
    brier_skill = baseline_brier - brier
    passed = brier_skill > 0 and brier <= raw_brier

    report.update({
        "status": "PASSED" if passed else "REJECTED",
        "brier": brier, "baseline_brier": baseline_brier, "raw_uncalibrated_brier": raw_brier,
        "brier_skill_vs_base_rate": brier_skill,
        "split": "strict_chronological_80_20",
        "excludes": ["smart_wallet_exit", "insider_sell"],
        "exclusion_reason": "wallet reputation is not point-in-time snapshotted; excluded to avoid lookahead leakage",
    })
    if passed:
        model_dir.mkdir(parents=True, exist_ok=True)
        output = model_dir / f"hazard-calibration-{int(time.time())}.joblib"
        import joblib
        joblib.dump({
            "artifact_version": HAZARD_ARTIFACT_VERSION,
            "calibrator": calibrator,
            "trigger_weights": DEFAULT_TRIGGER_WEIGHTS,
            "trained_at": time.time(),
            "validation_report": report,
        }, output)
        report["model_path"] = str(output)
    _persist_report(model_dir, report)
    return report


def _persist_report(model_dir: Path, report: Dict[str, Any]):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "last_hazard_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a chronologically validated rug-hazard calibration")
    parser.add_argument("--storage", default="data/launch_episodes")
    parser.add_argument("--model-dir", default="models/hazard")
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--min-positive", type=int, default=15)
    args = parser.parse_args()
    report = train_hazard_calibration(Path(args.storage), Path(args.model_dir), args.min_samples, args.min_positive)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("status") in {"PASSED", "DATA_BLOCKED", "REJECTED"} else 1)


if __name__ == "__main__":
    main()
