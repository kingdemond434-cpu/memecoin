"""The single feature implementation shared by training and live inference.

Training-serving skew is silent and fatal: a model can pass chronological
out-of-sample validation on feature definition A and then be handed feature
definition B in shadow or live, so its measured edge simply does not transfer.
The repository previously had exactly that, with five features diverging:

    feature                training source          live source
    organic_ratio          flow_features            public_coordination
    bundle_concentration   flow_features            public_coordination
    sol_volume             wallet_features          flow / recent buys
    deployer_cluster_risk  entity_graph_features    assess_launch_risk()
    funding_wallet_risk    entity_graph_features    never populated at all

The fix is structural rather than a matter of care: both paths build the same
point-in-time snapshot groups and then call ``build_features`` here. There is
no second implementation to drift from.

FEATURE_SCHEMA_VERSION is stamped into every model artifact. A model trained
under one version must not be served under another, so a bump fails closed
instead of silently mismatching.
"""

from typing import Any, Dict

import numpy as np

from src.strategies.multihead_predictor import PredictionFeatures

FEATURE_SCHEMA_VERSION = "v2"

SNAPSHOT_GROUPS = (
    "deployer_features", "wallet_features", "flow_features", "liquidity_features",
    "social_features", "token_features", "market_features", "entity_graph_features",
)


def number(mapping: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Numeric read that never turns a missing value into a silent zero-by-accident.

    Missingness is reported separately through the *_available indicators, so
    the model can learn that absence itself carries information.
    """
    value = (mapping or {}).get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return default


def build_features(episode: Dict[str, Any], snapshot: Dict[str, Any]) -> PredictionFeatures:
    """Build the model input from point-in-time snapshot groups.

    ``episode`` supplies identity and launch time; ``snapshot`` supplies the
    feature groups captured at one instant. Both historical replay and live
    inference call this exact function with the same group shapes.
    """
    deployer = snapshot.get("deployer_features") or {}
    market = snapshot.get("market_features") or {}
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
    created_at = number(episode, "created_at")
    timestamp = number(snapshot, "timestamp", created_at)

    return PredictionFeatures(
        token=str(episode.get("token", "")),
        chain=str(episode.get("chain", "solana")),
        timestamp=timestamp,
        deployer_rug_rate=number(deployer, "rug_rate"),
        deployer_success_rate=number(deployer, "success_rate"),
        deployer_avg_multiple=number(deployer, "avg_max_multiple"),
        deployer_cluster_risk=number(graph, "deployer_cluster_risk"),
        funding_wallet_risk=number(graph, "funding_wallet_risk"),
        funding_wallet_reuse=number(graph, "funding_wallet_reuse"),
        wallet_quality_weighted_flow=number(graph, "actor_adjusted_flow"),
        sybil_discount=number(graph, "sybil_discount"),
        smart_wallet_sync_evidence=number(graph, "smart_wallet_sync_evidence"),
        initial_buyers=int(number(wallet, "initial_buyer_count")),
        smart_buyers=int(number(wallet, "smart_buyer_count")),
        insider_buyers=int(number(wallet, "insider_buyer_count")),
        buyer_acceleration=number(flow, "buy_acceleration"),
        buy_velocity=number(flow, "buy_velocity"),
        sol_volume=number(wallet, "total_sol_volume"),
        organic_ratio=number(flow, "organic_ratio"),
        bundle_concentration=number(flow, "bundle_concentration"),
        liquidity_usd=number(liquidity, "liquidity_usd"),
        liquidity_locked=bool(liquidity.get("liquidity_locked")),
        ownership_renounced=bool(token.get("ownership_renounced")),
        can_mint=bool(token.get("can_mint")),
        can_freeze=bool(token.get("can_freeze")),
        social_velocity=number(social, "avg_velocity"),
        social_acceleration=number(social, "acceleration"),
        social_price_disagreement=number(social, "price_disagreement"),
        social_credibility=number(social, "avg_credibility"),
        chain_before_social=number(social, "chain_before_pct"),
        cross_platform=bool(social.get("cross_platform")),
        holder_concentration=number(token, "top_10_pct") / 100,
        holder_concentration_delta=number(token, "top_10_delta_pct") / 100,
        holder_concentration_velocity=number(token, "top_10_velocity_pct_per_second") / 100,
        top_10_pct=number(token, "top_10_pct"),
        top_20_pct=number(token, "top_20_pct"),
        top_20_delta_pct=number(token, "top_20_delta_pct"),
        deployer_pct=number(token, "dev_pct"),
        insider_pct=number(token, "insider_pct"),
        bundler_pct=number(token, "bundler_pct"),
        fresh_wallet_pct=number(token, "fresh_wallet_pct"),
        whale_pct=number(token, "whale_pct"),
        connected_cluster_pct=number(token, "connected_cluster_pct"),
        dev_recent_sells=number(token, "dev_recent_sells"),
        dev_sell_supply_pct=number(token, "dev_sell_supply_pct"),
        dev_hard_veto_count=number(token, "dev_hard_veto_count"),
        capital_rotation_flow=number(token, "capital_rotation_flow"),
        token_extension_risk=number(token, "extension_risk"),
        # Market/regime group: whether the whole memecoin market is hot or
        # dead conditions every other signal, so it belongs in the vector
        # rather than being implicitly assumed constant.
        meme_launch_rate_1h=number(market, "meme_launch_rate_1h"),
        sol_change_24h=number(market, "sol_change_24h"),
        btc_change_24h=number(market, "btc_change_24h"),
        sol_btc_beta=number(market, "sol_btc_beta"),
        solana_tvl_change=number(market, "solana_tvl_change"),
        priority_fee_p90=number(market, "priority_fee_p90"),
        fee_pressure=number(market, "fee_pressure"),
        data_coverage=sum(statuses) / len(statuses),
        wallet_history_available=bool(wallet.get("smart_buyer_count") is not None),
        social_available=bool(social.get("mention_count")),
        social_price_disagreement_available=(
            social.get("price_disagreement_status") == "OK"),
        holder_trajectory_available=(token.get("top_20_delta_pct") is not None),
        holder_owner_enrichment_available=any(
            token.get(name) is not None for name in (
                "dev_pct", "insider_pct", "bundler_pct", "fresh_wallet_pct",
                "whale_pct", "connected_cluster_pct")),
        actor_flow_available=(graph.get("actor_adjusted_flow") is not None),
        dev_state_available=(token.get("dev_state_status") == "OK"),
        capital_rotation_available=(token.get("capital_rotation_status") == "OK"),
        coordination_available=(flow.get("status") == "OK"
                                and int(number(flow, "observed_trade_count")) >= 3),
        flow_available=flow.get("status") == "OK",
        time_since_launch=max(0.0, timestamp - created_at),
    )
