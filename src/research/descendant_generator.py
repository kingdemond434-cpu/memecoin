"""Turn measured failure attribution into falsifiable descendant hypotheses."""

import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List


MECHANISM_FEATURES = {
    "rug_loss": ("dev_wallet_events", "authority_mutation", "exit_capacity"),
    "execution_miss": ("observe_to_submit_ms", "landing_ms", "priority_fee"),
    "premature_exit": ("smart_wallet_flow", "distribution_probability", "curve_regime"),
    "missed_monster": ("wallet_quality_flow", "curve_acceleration", "social_price_disagreement"),
    "sizing_leak": ("exit_capacity", "uncertainty", "portfolio_correlation"),
}


def generate_descendants(findings: Iterable[Any], top_n: int = 10) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str], float] = defaultdict(float)
    for finding in findings:
        leak = getattr(getattr(finding, "leak", None), "value", None) or str(
            getattr(finding, "leak", "unattributed"))
        evidence = getattr(finding, "evidence", {}) or {}
        reason = str(evidence.get("rejection_reason") or evidence.get("exit_reason")
                     or evidence.get("failure_reason") or "unattributed")
        grouped[(leak, reason)] += max(0.0, float(
            getattr(finding, "forgone_log_growth", 0.0) or 0.0))
    if not grouped:
        return {"status": "DATA_BLOCKED", "hypotheses": [],
                "detail": "no attributed failures"}
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:top_n]
    hypotheses: List[Dict[str, Any]] = []
    for (leak, reason), loss in ranked:
        features = MECHANISM_FEATURES.get(leak, ("failure_context",))
        identity = hashlib.sha256(f"{leak}|{reason}|{','.join(features)}".encode()).hexdigest()[:16]
        hypotheses.append({
            "hypothesis_id": f"descendant-{identity}",
            "parent_failure": {"leak": leak, "reason": reason,
                               "forgone_log_growth": loss},
            "candidate_features": list(features),
            "target": "net_oos_log_growth_after_costs",
            "validation": "chronological_out_of_sample",
            "falsifier": "no positive OOS log-growth improvement after all execution costs",
            "authority": "none_until_promoted",
        })
    return {"status": "OK", "hypotheses": hypotheses}
