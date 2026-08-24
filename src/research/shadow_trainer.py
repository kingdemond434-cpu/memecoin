"""Chronological shadow-only trainer for persisted point-in-time episodes."""

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.strategies.multihead_predictor import (
    ElogwEngine, MultiHeadPredictor, PredictionFeatures, PredictionTarget,
)


SNAPSHOT_ORDER = ("t10s", "t30s", "t1m")


def _number(mapping: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float, bool)) and np.isfinite(value) else default


def snapshot_to_features(episode: Dict[str, Any], snapshot: Dict[str, Any]) -> PredictionFeatures:
    deployer = snapshot.get("deployer_features") or {}
    wallet = snapshot.get("wallet_features") or {}
    flow = snapshot.get("flow_features") or {}
    liquidity = snapshot.get("liquidity_features") or {}
    social = snapshot.get("social_features") or {}
    token = snapshot.get("token_features") or {}
    graph = snapshot.get("entity_graph_features") or {}
    statuses = [
        bool(deployer.get("has_profile")), bool(wallet), flow.get("status") == "OK",
        liquidity.get("status") == "OK", bool(social.get("mention_count")),
        token.get("status") == "OK", graph.get("status") == "OK",
    ]
    return PredictionFeatures(
        token=str(episode.get("token", "")), chain=str(episode.get("chain", "solana")),
        timestamp=_number(snapshot, "timestamp", _number(episode, "created_at", 0)),
        deployer_rug_rate=_number(deployer, "rug_rate"),
        deployer_success_rate=_number(deployer, "success_rate"),
        deployer_avg_multiple=_number(deployer, "avg_max_multiple"),
        deployer_cluster_risk=_number(graph, "deployer_cluster_risk"),
        funding_wallet_risk=_number(graph, "funding_wallet_risk"),
        initial_buyers=int(_number(wallet, "initial_buyer_count")),
        smart_buyers=int(_number(wallet, "smart_buyer_count")),
        insider_buyers=int(_number(wallet, "insider_buyer_count")),
        buyer_acceleration=_number(flow, "buy_acceleration"),
        buy_velocity=_number(flow, "buy_velocity"),
        sol_volume=_number(wallet, "total_sol_volume"),
        organic_ratio=_number(flow, "organic_ratio"),
        bundle_concentration=_number(flow, "bundle_concentration"),
        liquidity_usd=_number(liquidity, "liquidity_usd"),
        liquidity_locked=bool(liquidity.get("liquidity_locked")),
        ownership_renounced=bool(token.get("ownership_renounced")),
        can_mint=bool(token.get("can_mint")), can_freeze=bool(token.get("can_freeze")),
        social_velocity=_number(social, "avg_velocity"),
        social_acceleration=_number(social, "acceleration"),
        social_credibility=_number(social, "avg_credibility"),
        chain_before_social=_number(social, "chain_before_pct"),
        cross_platform=bool(social.get("cross_platform")),
        holder_concentration=_number(token, "top_10_pct") / 100,
        top_10_pct=_number(token, "top_10_pct"),
        data_coverage=sum(statuses) / len(statuses),
        wallet_history_available=bool(wallet.get("smart_buyer_count") is not None),
        social_available=bool(social.get("mention_count")),
        coordination_available=flow.get("status") == "OK",
        time_since_launch=max(0.0, _number(snapshot, "timestamp") - _number(episode, "created_at")),
    )


def snapshot_labels(snapshot: Dict[str, Any]) -> Dict[PredictionTarget, float]:
    labels = snapshot.get("labels") or {}
    rug_time = labels.get("label_rug_time")
    rugged = bool(labels.get("label_rug"))
    slippage = _number(snapshot.get("liquidity_features") or {}, "price_impact_pct")
    return {
        PredictionTarget.P_2X: float(bool(labels.get("label_2x"))),
        PredictionTarget.P_5X: float(bool(labels.get("label_5x"))),
        PredictionTarget.P_10X: float(bool(labels.get("label_10x"))),
        PredictionTarget.P_50X: float(bool(labels.get("label_50x"))),
        PredictionTarget.P_MIGRATION: float(bool(labels.get("label_migration"))),
        PredictionTarget.P_RUG_30S: float(rugged and rug_time is not None and float(rug_time) <= 30),
        PredictionTarget.P_RUG_5M: float(rugged and rug_time is not None and float(rug_time) <= 300),
        PredictionTarget.EXPECTED_SLIPPAGE: float(np.clip(slippage, 0, 1)),
        PredictionTarget.EXPECTED_HOLD_TIME: max(0.0, _number(labels, "time_to_peak")),
    }


def load_samples(storage: Path) -> List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]]:
    samples: List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]] = []
    for path in storage.glob("*/*.json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        outcome = episode.get("final_outcome") or {}
        if outcome.get("status") != "OK":
            continue
        snapshots = episode.get("snapshots") or {}
        for name in SNAPSHOT_ORDER:
            snapshot = snapshots.get(name)
            if not snapshot or (snapshot.get("labels") or {}).get("label_2x") is None:
                continue
            samples.append((snapshot_to_features(episode, snapshot), snapshot_labels(snapshot), outcome))
    return sorted(samples, key=lambda item: item[0].timestamp)


def _brier(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    truth, probability = np.asarray(list(y_true)), np.asarray(list(y_prob))
    return float(np.mean((truth - probability) ** 2))


def validate_oos(
    predictor: MultiHeadPredictor,
    train_samples: List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]],
    oos_samples: List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]],
) -> Dict[str, Any]:
    predictions = predictor.predict_batch([item[0] for item in oos_samples])
    if any(item is None for item in predictions):
        return {"status": "DATA_BLOCKED", "reason": "one_or_more_prediction_heads_failed"}
    brier: Dict[str, float] = {}
    baseline: Dict[str, float] = {}
    for target in (
        PredictionTarget.P_2X, PredictionTarget.P_5X, PredictionTarget.P_10X,
        PredictionTarget.P_50X, PredictionTarget.P_MIGRATION,
        PredictionTarget.P_RUG_30S, PredictionTarget.P_RUG_5M,
    ):
        truth = [item[1][target] for item in oos_samples]
        probability = [getattr(prediction, target.value) for prediction in predictions]
        train_rate = float(np.mean([item[1][target] for item in train_samples]))
        brier[target.value] = _brier(truth, probability)
        baseline[target.value] = _brier(truth, [train_rate] * len(truth))

    realized_logs: List[float] = []
    trade_count = 0
    for prediction, (_, _, outcome) in zip(predictions, oos_samples):
        bins = ElogwEngine.probability_bins(prediction)
        expected_log = sum(probability * math.log(1 + 0.01 * (gross - prediction.expected_slippage - 0.003))
                           for _, probability, gross in bins)
        if expected_log <= 0 or prediction.p_2x < 0.10 or prediction.p_5x < 0.05:
            continue
        feasible = outcome.get("feasible_exit_multiple")
        if outcome.get("rugged"):
            realized_return = -0.98
        elif feasible is None:
            continue
        else:
            realized_return = float(np.clip(float(feasible) - 1, -0.98, 49))
        realized_logs.append(math.log(1 + 0.01 * (realized_return - prediction.expected_slippage - 0.003)))
        trade_count += 1

    mean_brier_skill = float(np.mean([
        baseline[key] - brier[key] for key in brier
    ]))
    net_elogw = float(np.mean(realized_logs)) if realized_logs else -float("inf")
    passed = len(oos_samples) >= 50 and trade_count >= 10 and mean_brier_skill > 0 and net_elogw > 0
    return {
        "status": "PASSED" if passed else "REJECTED",
        "oos_samples": len(oos_samples), "shadow_policy_trades": trade_count,
        "mean_brier_skill": mean_brier_skill, "net_elogw_proxy": net_elogw,
        "brier": brier, "baseline_brier": baseline,
        "split": "strict_chronological_80_20",
        "warning": "net_elogw_proxy uses route-feasible observed outcomes; forward shadow remains mandatory",
    }


def train_shadow(storage: Path, model_dir: Path, min_samples: int = 250) -> Dict[str, Any]:
    samples = load_samples(storage)
    report: Dict[str, Any] = {"created_at": time.time(), "samples": len(samples)}
    if len(samples) < min_samples:
        report.update({"status": "DATA_BLOCKED", "reason": f"need_at_least_{min_samples}_labeled_snapshots"})
    else:
        split = int(len(samples) * 0.8)
        train_samples, oos_samples = samples[:split], samples[split:]
        predictor = MultiHeadPredictor(str(model_dir))
        predictor.initialize_models()
        for features, labels, _ in train_samples:
            predictor.add_training_sample(features, labels)
        training = predictor.train(min_samples=max(100, int(len(train_samples) * 0.5)))
        report["training"] = training
        if not predictor._is_trained:
            report.update({"status": "DATA_BLOCKED", "reason": "one_or_more_heads_lack_chronological_class_coverage"})
        else:
            report.update(validate_oos(predictor, train_samples, oos_samples))
            if report["status"] == "PASSED":
                model_dir.mkdir(parents=True, exist_ok=True)
                output = model_dir / f"multihead-shadow-{int(time.time())}-{predictor.model_version}.joblib"
                predictor.save(str(output), report)
                report["model_path"] = str(output)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "last_training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train a chronologically validated shadow challenger")
    parser.add_argument("--storage", default="data/launch_episodes")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--min-samples", type=int, default=250)
    args = parser.parse_args()
    report = train_shadow(Path(args.storage), Path(args.model_dir), args.min_samples)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report.get("status") in {"PASSED", "DATA_BLOCKED", "REJECTED"} else 1)


if __name__ == "__main__":
    main()
