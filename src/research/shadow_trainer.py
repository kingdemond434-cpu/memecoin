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
    market = snapshot.get("market_features") or {}
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
        funding_wallet_reuse=_number(graph, "funding_wallet_reuse"),
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
        holder_concentration_delta=_number(token, "top_10_delta_pct") / 100,
        holder_concentration_velocity=_number(token, "top_10_velocity_pct_per_second") / 100,
        top_10_pct=_number(token, "top_10_pct"),
        token_extension_risk=_number(token, "extension_risk"),
        meme_launch_rate_1h=_number(market, "meme_launch_rate_1h"),
        sol_change_24h=_number(market, "sol_change_24h"),
        btc_change_24h=_number(market, "btc_change_24h"),
        sol_btc_beta=_number(market, "sol_btc_beta"),
        solana_tvl_change=_number(market, "solana_tvl_change"),
        priority_fee_p90=_number(market, "priority_fee_p90"),
        fee_pressure=_number(market, "fee_pressure"),
        data_coverage=sum(statuses) / len(statuses),
        wallet_history_available=bool(wallet.get("smart_buyer_count") is not None),
        social_available=bool(social.get("mention_count")),
        coordination_available=(flow.get("status") == "OK"
                                and int(flow.get("observed_trade_count", 0) or 0) >= 3),
        flow_available=flow.get("status") == "OK",
        time_since_launch=max(0.0, _number(snapshot, "timestamp") - _number(episode, "created_at")),
    )


def snapshot_labels(
    snapshot: Dict[str, Any], episode: Dict[str, Any], outcome: Dict[str, Any]
) -> Dict[PredictionTarget, float]:
    labels = snapshot.get("labels") or {}
    rug_time = outcome.get("rug_time")
    if rug_time is not None:
        rug_time = (_number(episode, "created_at") + float(rug_time)
                    - _number(snapshot, "timestamp", _number(episode, "created_at")))
    rugged = bool(outcome.get("rugged")) and rug_time is not None and rug_time >= 0
    slippage = _number(snapshot.get("liquidity_features") or {}, "price_impact_pct")
    result = {
        PredictionTarget.P_2X: float(bool(labels.get("label_2x"))),
        PredictionTarget.P_5X: float(bool(labels.get("label_5x"))),
        PredictionTarget.P_10X: float(bool(labels.get("label_10x"))),
        PredictionTarget.P_50X: float(bool(labels.get("label_50x"))),
        PredictionTarget.P_MIGRATION: float(bool(outcome.get("migrated"))),
        PredictionTarget.P_RUG_30S: float(rugged and rug_time is not None and float(rug_time) <= 30),
        PredictionTarget.P_RUG_5M: float(rugged and rug_time is not None and float(rug_time) <= 300),
        PredictionTarget.EXPECTED_SLIPPAGE: float(np.clip(slippage, 0, 1)),
        PredictionTarget.EXPECTED_HOLD_TIME: max(0.0, _number(labels, "time_to_peak")),
    }
    feasible = labels.get("feasible_exit_multiple")
    if isinstance(feasible, (int, float)) and np.isfinite(feasible) and feasible > 0:
        result[PredictionTarget.EXPECTED_FEASIBLE_MULTIPLE] = float(np.clip(feasible, 0.02, 50))
    return result


def _repair_legacy_outcome(episode: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Derive labels from persisted observations when older outcome rows omitted them."""
    repaired = dict(outcome)
    observations = episode.get("market_observations") or []
    migration_types = {"migration", "token_migrated", "graduation"}
    if not repaired.get("migrated") and any(
        item.get("migrated") is True or str(item.get("type", "")).lower() in migration_types
        for item in observations
    ):
        repaired["migrated"] = True
    if repaired.get("rugged") and repaired.get("rug_time") is None:
        created_at = _number(episode, "created_at")
        explicit = sorted(
            (item for item in observations if item.get("rugged") and item.get("timestamp") is not None),
            key=lambda item: float(item["timestamp"]),
        )
        rug_at = float(explicit[0]["timestamp"]) if explicit else None
        prices = sorted(
            (item for item in observations
             if item.get("timestamp") is not None
             and _number(item, "price_multiple", _number(item, "price_usd")) > 0),
            key=lambda item: float(item["timestamp"]),
        )
        if rug_at is None and prices:
            if all(_number(item, "price_multiple") > 0 for item in prices):
                multiples = [_number(item, "price_multiple") for item in prices]
            else:
                entry = _number(prices[0], "price_usd")
                multiples = [_number(item, "price_usd") / max(entry, 1e-12) for item in prices]
            peak = multiples[0]
            for item, multiple in zip(prices, multiples):
                peak = max(peak, multiple)
                if 1 - multiple / max(peak, 1e-12) >= 0.90:
                    rug_at = float(item["timestamp"])
                    break
        if rug_at is not None:
            repaired["rug_time"] = max(0.0, rug_at - created_at)
    return repaired


def load_samples(storage: Path) -> List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]]:
    samples: List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]] = []
    for path in storage.glob("*/*.json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episode = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        outcome = _repair_legacy_outcome(episode, episode.get("final_outcome") or {})
        if outcome.get("status") != "OK":
            continue
        observations = [item for item in (episode.get("market_observations") or [])
                        if _number(item, "price_multiple", _number(item, "price_usd")) > 0]
        timestamps = sorted({_number(item, "timestamp") for item in observations if _number(item, "timestamp") > 0})
        recognized = any(
            (item.get("signature") and item.get("program"))
            or item.get("measurement") in {"jupiter_round_trip_probe", "decoded_onchain_reserve_event"}
            for item in observations
        )
        if len(timestamps) < 3 or timestamps[-1] - timestamps[0] < 1 or not recognized:
            continue
        snapshots = episode.get("snapshots") or {}
        for name in SNAPSHOT_ORDER:
            snapshot = snapshots.get(name)
            if not snapshot or (snapshot.get("labels") or {}).get("label_2x") is None:
                continue
            if (outcome.get("rugged") and outcome.get("rug_time") is not None
                    and _number(snapshot, "timestamp") >= _number(episode, "created_at") + float(outcome["rug_time"])):
                continue
            samples.append((snapshot_to_features(episode, snapshot), snapshot_labels(snapshot, episode, outcome), outcome))
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
    feasible_pairs = [
        (float(outcome["feasible_exit_multiple"]), float(prediction.expected_feasible_multiple))
        for prediction, (_, _, outcome) in zip(predictions, oos_samples)
        if outcome.get("feasible_exit_multiple") is not None
    ]
    train_feasible = [float(item[2]["feasible_exit_multiple"]) for item in train_samples
                      if item[2].get("feasible_exit_multiple") is not None]
    feasible_mae = (float(np.mean([abs(actual - predicted) for actual, predicted in feasible_pairs]))
                    if feasible_pairs else float("inf"))
    feasible_baseline = float(np.median(train_feasible)) if train_feasible else 0.0
    feasible_baseline_mae = (float(np.mean([abs(actual - feasible_baseline) for actual, _ in feasible_pairs]))
                             if feasible_pairs and train_feasible else float("inf"))
    net_elogw = float(np.mean(realized_logs)) if realized_logs else -float("inf")
    passed = (
        len(oos_samples) >= 50 and trade_count >= 10 and mean_brier_skill > 0 and net_elogw > 0
        and len(feasible_pairs) >= 10 and feasible_mae < feasible_baseline_mae
    )
    return {
        "status": "PASSED" if passed else "REJECTED",
        "oos_samples": len(oos_samples), "shadow_policy_trades": trade_count,
        "mean_brier_skill": mean_brier_skill, "net_elogw_proxy": net_elogw,
        "feasible_return_samples": len(feasible_pairs), "feasible_return_mae": feasible_mae,
        "feasible_return_baseline_mae": feasible_baseline_mae,
        "brier": brier, "baseline_brier": baseline,
        "split": "strict_chronological_80_20",
        "warning": "net_elogw_proxy uses route-feasible observed outcomes; forward shadow remains mandatory",
    }


def chronological_episode_split(
    samples: List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]],
    train_fraction: float = 0.8,
) -> Tuple[
    List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]],
    List[Tuple[PredictionFeatures, Dict[PredictionTarget, float], Dict[str, Any]]],
]:
    """Split entire launch episodes, preventing one token from leaking across folds."""
    first_seen: Dict[str, float] = {}
    for features, _, _ in samples:
        first_seen[features.token] = min(first_seen.get(features.token, features.timestamp), features.timestamp)
    ordered_tokens = sorted(first_seen, key=lambda token: (first_seen[token], token))
    if len(ordered_tokens) < 2:
        return [], []
    split_at = max(1, min(len(ordered_tokens) - 1, int(len(ordered_tokens) * train_fraction)))
    train_tokens = set(ordered_tokens[:split_at])
    train = [sample for sample in samples if sample[0].token in train_tokens]
    oos = [sample for sample in samples if sample[0].token not in train_tokens]
    return train, oos


def train_shadow(storage: Path, model_dir: Path, min_samples: int = 250) -> Dict[str, Any]:
    samples = load_samples(storage)
    report: Dict[str, Any] = {"created_at": time.time(), "samples": len(samples)}
    if len(samples) < min_samples:
        report.update({"status": "DATA_BLOCKED", "reason": f"need_at_least_{min_samples}_labeled_snapshots"})
    else:
        train_samples, oos_samples = chronological_episode_split(samples)
        report.update({
            "train_samples": len(train_samples), "oos_samples": len(oos_samples),
            "train_episodes": len({item[0].token for item in train_samples}),
            "oos_episodes": len({item[0].token for item in oos_samples}),
        })
        if not train_samples or not oos_samples:
            report.update({"status": "DATA_BLOCKED", "reason": "need_at_least_two_distinct_launch_episodes"})
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "last_training_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            return report
        predictor = MultiHeadPredictor(str(model_dir))
        predictor.initialize_models()
        for features, labels, _ in train_samples:
            predictor.add_training_sample(features, labels)
        training = predictor.train(min_samples=100)
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
