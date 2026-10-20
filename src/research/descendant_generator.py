"""Turn measured failure attribution into falsifiable descendant hypotheses."""

import hashlib
import copy
import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from src.research.research_contracts import validate_exact_candidate

FAILURE_TAXONOMY = {
    "no_mechanism": "identify a forced/incentivized transactor and a discriminating public observable",
    "wrong_proxy": "replace the measured proxy with a PIT observable tied to the proposed mechanism",
    "wrong_chain_protocol": "supply verified native events, fees, finality and buy/sell simulations for the target protocol",
    "wrong_lifecycle_stage": "rebuild the observable and replay at the explicitly identified lifecycle stage",
    "wrong_horizon": "supply a changed holding/entry horizon justified by measured executable timing",
    "latency_death": "demonstrate measured event-to-fill latency below the frozen edge survival boundary",
    "cost_death": "demonstrate lower measured total fees, tips, gas, slippage and failed-transaction cost",
    "route_fill_death": "supply verified native route/fill and emergency-exit evidence",
    "rug_honeypot_death": "supply a pre-decision native sellability or authority observable",
    "sample_insufficiency": "supply additional independent chronological cohorts without changing the strategy",
    "leakage": "repair availability timestamps and rerun all affected untouched cohorts",
    "selection_bias": "enumerate every inspected frozen variant and repair matched controls",
    "crowding": "measure follower delay, shared actor dependence and executable capacity at proposed size",
    "regime_dependence": "define and measure the distinguishing regime before the decision",
    "data_defect": "repair the named data defect with independently checked native fixtures",
    "implementation_defect": "repair the named implementation defect and verify parity on immutable fixtures",
    "validator_false_negative": "supply a frozen validation attack demonstrating a specific canonical validator defect",
    "genuine_no_edge": "supply a new economic mechanism or independently measured enabling change; do not recycle parameters",
}
_LEAK_DIAGNOSES = {"rug_loss": "rug_honeypot_death", "execution_miss": "route_fill_death",
    "premature_exit": "wrong_horizon", "missed_monster": "validator_false_negative",
    "sizing_leak": "crowding", "under_sized": "crowding", "over_sized": "crowding"}


def diagnose_failure(evidence: Dict[str, Any]) -> str | None:
    """Use attributed facts, never infer a rug/no-edge from a price collapse."""
    named = evidence.get("diagnosis")
    if named in FAILURE_TAXONOMY and evidence.get("diagnosis_evidence_refs"):
        return named
    if evidence.get("future_input_detected") is True: return "leakage"
    if evidence.get("trials_enumerated") is False: return "selection_bias"
    if evidence.get("implementation_parity_failed") is True: return "implementation_defect"
    if evidence.get("data_integrity_failed") is True: return "data_defect"
    if evidence.get("native_sell_simulation_failed") is True or evidence.get("rug_confirmed") is True:
        return "rug_honeypot_death"
    samples, required = evidence.get("samples"), evidence.get("required_samples")
    if type(samples) is int and type(required) is int and 0 <= samples < required:
        return "sample_insufficiency"
    if evidence.get("route_success") is False or evidence.get("fill_success") is False:
        return "route_fill_death"
    for before, after, diagnosis in (("net_before_delay", "net_after_delay", "latency_death"),
                                      ("net_before_cost", "net_after_cost", "cost_death")):
        if (type(evidence.get(before)) in (int, float) and type(evidence.get(after)) in (int, float)
                and math.isfinite(evidence[before]) and math.isfinite(evidence[after])
                and evidence[before] > 0 >= evidence[after]): return diagnosis
    if (evidence.get("canonical_controls_complete") is True
            and evidence.get("canonical_receipt_ref") and type(evidence.get("net_elogw_lower")) in (int, float)
            and math.isfinite(evidence["net_elogw_lower"]) and evidence["net_elogw_lower"] <= 0):
        return "genuine_no_edge"
    return None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _parent(payload, framework):
    identity, seal = payload.get("parent_id"), payload.get("parent_frozen_hash")
    parent = framework.hypotheses.get(identity)
    if parent is not None and parent.frozen_hash == seal and framework._exact_frozen_valid(parent):
        return parent
    raw = framework.frozen_specs.get(seal)
    if raw:
        from src.strategies.champion_challenger import HypothesisSpec
        parent = HypothesisSpec(**raw)
        if parent.hypothesis_id == identity and framework._exact_frozen_valid(parent): return parent
    raise ValueError("descendant requires the full exact frozen parent version")


def _request(variable, parent, refs, *, reason, kind="DATA_REQUEST", extra=None):
    return {"kind": kind, "payload": {"variable": variable, "parent_id": parent.hypothesis_id,
        "parent_frozen_hash": parent.frozen_hash, "mechanism": parent.mechanism,
        "source_classes": ["canonical execution receipts", "public native observations", "immutable native fixtures"],
        "pit_requirements": "availability precedes decision; repair must not inspect untouched cohorts",
        "expected_consumer": "failure/survivor canonical experiment compiler", "strategy_family": parent.trial_family,
        "acquisition_cost": None, "validation_plan": reason, "falsifier": parent.falsifier,
        "evidence_refs": list(refs), "reason": reason, **dict(extra or {})}}


def _descendant(parent, patch, change, refs, now):
    allowed = {"data_inputs", "feature_program", "exact_entry", "size_rule", "fee_policy", "landing_assumptions",
        "execution_assumptions", "exit_rule", "emergency_exit", "holding_time_model", "entry_horizon_seconds",
        "max_decision_latency_ms", "chain", "protocol", "lifecycle_stage", "native_execution_contract"}
    if not isinstance(patch, dict) or not patch or set(patch) - allowed:
        raise ValueError("descendant patch is unsupported or attempts to change gates/safety authority")
    candidate = copy.deepcopy(parent.exact_candidate)
    for key in ("candidate_hash", "frozen_at", "status"):
        candidate.pop(key, None)
    candidate.update(copy.deepcopy(patch))
    changed_native = any(candidate[key] != parent.exact_candidate[key] for key in ("chain", "protocol", "lifecycle_stage"))
    if changed_native:
        native = candidate.get("native_execution_contract")
        if (not isinstance(native, dict) or native.get("chain") != candidate["chain"]
                or native.get("protocol") != candidate["protocol"]
                or any(not native.get(key) for key in ("decoder_fixture_ref", "buy_simulation_ref", "sell_simulation_ref"))):
            raise ValueError("cross-chain descendants need verified native semantics, never copied parameters")
    candidate.update(candidate_id="descendant-" + _digest({"parent": parent.frozen_hash, "patch": patch, "change": change}),
        created_at=now, parent_ids=[parent.hypothesis_id], agent_identity="deterministic-failure-survivor-science",
        parent_frozen_hashes={parent.hypothesis_id: parent.frozen_hash}, source_lineage=list(dict.fromkeys(list(refs))),
        enabling_change=copy.deepcopy(change), novelty_signature=_digest({"parent": parent.frozen_hash, "patch": patch}))
    admitted = validate_exact_candidate(candidate)
    if not admitted.ok: raise ValueError("descendant remains nonexact: " + "; ".join(admitted.reasons))
    return {"kind": "EXACT_CANDIDATE", "payload": {"candidate": candidate, "enabling_change": change,
        "parent_frozen_hash": parent.frozen_hash, "revalidation_required": True},
        "axes": {"chain": candidate["chain"], "protocol": candidate["protocol"], "lifecycle_stage": candidate["lifecycle_stage"]},
        "enabling_change": change["change_id"]}


def consume_descendant(kind, payload, framework, *, evidence_refs, now):
    parent = _parent(payload, framework)
    if kind == "FAILURE_DESCENDANT":
        facts = payload.get("attribution")
        if not isinstance(facts, dict):
            return {"disposition": "DATA_REQUEST", "detail": {"parent_id": parent.hypothesis_id},
                "children": [_request("attributed_failure_diagnosis", parent, evidence_refs,
                    reason="a losing mark alone does not identify mechanism, rug, latency or validator failure")]}
        diagnosis = diagnose_failure(facts)
        if diagnosis is None:
            return {"disposition": "DATA_REQUEST", "detail": {"diagnosis": "UNATTRIBUTED"},
                "children": [_request("attributed_failure_diagnosis", parent, evidence_refs,
                    reason="name one diagnosis with measured facts and canonical evidence", extra={"taxonomy": list(FAILURE_TAXONOMY)})]}
        change = payload.get("enabling_change")
        valid_change = (isinstance(change, dict) and isinstance(change.get("change_id"), str) and bool(change["change_id"])
            and change.get("addresses_diagnosis") == diagnosis and change.get("evidence_refs")
            and set(change["evidence_refs"]) <= set(evidence_refs)
            and type(change.get("observed_at")) in (int, float) and math.isfinite(change["observed_at"])
            and parent.frozen_at < change["observed_at"] <= now and change.get("description"))
        if not valid_change:
            children = [] if diagnosis == "genuine_no_edge" else [_request("enabling_change_for_" + diagnosis, parent, evidence_refs,
                reason=FAILURE_TAXONOMY[diagnosis], extra={"diagnosis": diagnosis})]
            return {"disposition": "NEGATIVE_KNOWLEDGE", "detail": {"diagnosis": diagnosis,
                "parent_id": parent.hypothesis_id, "resurrection": "BLOCKED_WITHOUT_NAMED_OBSERVED_CHANGE",
                "required_change": FAILURE_TAXONOMY[diagnosis]}, "children": children}
        patch = payload.get("proposed_patch")
        if not patch:
            return {"disposition": "DEEPENING_TASK", "detail": {"diagnosis": diagnosis, "enabling_change": change},
                "children": [_request("exact_repair_program", parent, evidence_refs, kind="REPRESENTATION_REQUEST",
                    reason=FAILURE_TAXONOMY[diagnosis], extra={"enabling_change": change})]}
        child = _descendant(parent, patch, change, evidence_refs, now)
        return {"disposition": "FAILURE_DESCENDANT", "detail": {"diagnosis": diagnosis, "enabling_change": change}, "children": [child]}
    survival = payload.get("survival_evidence")
    if (not isinstance(survival, dict) or survival.get("verdict") != "SURVIVOR"
            or survival.get("spec_hash") != parent.frozen_hash or survival.get("evidence_ref") not in evidence_refs):
        raise ValueError("survivor work requires a canonical survivor receipt bound to the frozen parent")
    children = []
    for feature in list(parent.exact_candidate["feature_program"])[:8]:
        def ablate(node):
            if isinstance(node, dict):
                if node.get("field") == feature: return {"op": "constant", "value": True}
                return {key: ablate(value) for key, value in node.items()}
            if isinstance(node, list): return [ablate(value) for value in node]
            return node
        entry = ablate(parent.exact_candidate["exact_entry"])
        if entry == parent.exact_candidate["exact_entry"]: continue
        change = {"change_id": "survivor-feature-ablation:" + feature, "addresses_diagnosis": "mechanism_minimization",
                  "evidence_refs": [survival["evidence_ref"]], "observed_at": now,
                  "description": "remove one entry-feature condition to test minimum necessary mechanism"}
        children.append(_descendant(parent, {"exact_entry": entry}, change, evidence_refs, now))
    children.append(_request("survivor_failure_boundary", parent, evidence_refs, kind="VALIDATION_ATTACK",
        reason="perturb timing/cost/actor labels and cross named regime boundaries; preserve frozen safety vetoes",
        extra={"entry_program": parent.exact_candidate["exact_entry"], "required_controls": ["source_ablation", "actor_shuffle", "latency_stress", "route_loss"]}))
    children.append(_request("target_native_mechanism_translation", parent, evidence_refs,
        reason="translate the economic mechanism only after new chain-native observables, fees, finality and exits are measured"))
    children.append(_request("portfolio_complementarity_and_decay", parent, evidence_refs,
        reason="measure survivor contribution against incumbents and cash; collect new untouched decay cohorts and both-leg replacement costs"))
    return {"disposition": "SURVIVOR_DESCENDANT", "detail": {"parent_id": parent.hypothesis_id,
        "parent_frozen_hash": parent.frozen_hash, "scope": "RESEARCH_ONLY_NO_PROMOTION"}, "children": children}


MECHANISM_FEATURES = {
    "rug_loss": ("dev_wallet_events", "authority_mutation", "exit_capacity"),
    "execution_miss": ("observe_to_submit_ms", "landing_ms", "priority_fee"),
    "premature_exit": ("smart_wallet_flow", "distribution_probability", "curve_regime"),
    "missed_monster": ("wallet_quality_flow", "curve_acceleration", "social_price_disagreement"),
    "sizing_leak": ("exit_capacity", "uncertainty", "portfolio_correlation"),
}


def generate_descendants(findings: Iterable[Any], top_n: int = 10) -> Dict[str, Any]:
    if type(top_n) is not int or not 1 <= top_n <= 32:
        raise ValueError("descendant work bound must be an integer between 1 and 32")
    grouped: Dict[tuple[str, str], float] = defaultdict(float)
    for index, finding in enumerate(findings):
        if index >= 1024:
            return {"status": "DATA_BLOCKED", "hypotheses": [],
                    "detail": "attribution input exceeds 1024-item bound; partition durable findings before conversion"}
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
            "diagnosis": _LEAK_DIAGNOSES.get(leak, "UNATTRIBUTED"),
            "status": "MECHANISM_HYPOTHESIS",
            "resurrection": "BLOCKED_UNTIL_NAMED_ENABLING_CHANGE",
        })
    return {"status": "OK", "hypotheses": hypotheses}
