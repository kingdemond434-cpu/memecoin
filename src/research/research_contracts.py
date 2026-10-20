"""Versioned, fail-closed L0 admission for research proposals.

This is a schema boundary, not a validator or promotion path. Passing L0 says
only that a frozen proposal can be sent to the existing scientific gauntlet.
Rules are bounded JSON trees; source text and Python expressions are refused.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

CANDIDATE_SCHEMA_VERSION = 1
RULE_FIELDS = ("universe_rule", "exact_entry", "hard_vetoes", "size_rule",
               "exit_rule", "emergency_exit")
TEXT_FIELDS = ("candidate_id", "agent_identity", "trial_family", "chain",
               "protocol", "lifecycle_stage", "mechanism", "actor_incentive",
               "falsifier", "expected_behavior", "holding_time_model",
               "novelty_signature")
OBJECT_FIELDS = ("feature_program", "fee_policy", "landing_assumptions",
                 "execution_assumptions")
LIST_FIELDS = ("source_lineage", "failure_modes", "alternative_explanations")
OPS = {"all", "any", "not", "gt", "gte", "lt", "lte", "eq", "ne",
       "in", "constant", "fraction", "take_profit", "stop_loss", "timeout"}


@dataclass(frozen=True)
class AdmissionResult:
    status: str
    reasons: tuple[str, ...]
    candidate_hash: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "EXACT_CANDIDATE"


def canonical_candidate_hash(payload: Mapping[str, Any]) -> str:
    """The seal excludes only lifecycle metadata, never executable contents."""
    body = {k: v for k, v in payload.items()
            if k not in {"candidate_hash", "frozen_at", "status"}}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                     allow_nan=False, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_candidate_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    """Conservative executable identity and numeric parameter-family identity.

    Natural-language equivalence is never guessed. The declared economic
    mechanism is part of identity; IDs, clocks, attribution and narrative
    descriptions cannot manufacture another independent executable program.
    """
    admission = validate_exact_candidate(payload)
    if not admission.ok:
        raise ValueError("nonexact candidate has no canonical executable niche")
    fields = ("schema_version", "chain", "protocol", "lifecycle_stage", "mechanism", "actor_incentive",
              "universe_rule", "data_inputs", "feature_program", "entry_horizon_seconds",
              "max_decision_latency_ms", "exact_entry", "hard_vetoes", "size_rule", "fee_policy",
              "landing_assumptions", "exit_rule", "emergency_exit", "holding_time_model", "execution_assumptions")
    def normalize(value: Any, *, parameters: bool = False) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item, parameters=parameters) for key, item in sorted(value.items())
                    if key not in {"source", "source_ref", "source_lineage", "agent_identity", "provenance"}}
        if isinstance(value, list):
            return [normalize(item, parameters=parameters) for item in value]
        if type(value) in (int, float) and parameters:
            return {"numeric_parameter": True}
        if isinstance(value, str):
            return " ".join(value.split())
        return value
    body = {name: payload[name] for name in fields}
    # Input declaration order does not change any mapping or executable rule.
    body["data_inputs"] = sorted(body["data_inputs"], key=lambda item: item["name"])
    # Mapped observable/feature aliases are alpha-renamed by their executable
    # expression. A spelling change cannot create another program niche.
    inputs = {}
    for item in body["data_inputs"]:
        descriptor = {key: value for key, value in item.items() if key != "name"}
        if not item.get("snapshot_group") or not item.get("snapshot_key"):
            descriptor["observable_name"] = item["name"]
        inputs[item["name"]] = {"input": normalize(descriptor)}
    features = {}
    for name, expression in body["feature_program"].items():
        if expression["op"] == "identity":
            features[name] = inputs[expression["input"]]
        else:
            features[name] = {"op": expression["op"], "left": inputs[expression["left"]],
                              "right": inputs[expression["right"]]}
    def expand_rule(rule: Any) -> Any:
        if isinstance(rule, dict):
            result = {key: expand_rule(value) for key, value in rule.items() if key != "field"}
            if "field" in rule:
                result["expression"] = features.get(rule["field"], inputs.get(rule["field"], {"unmapped_field": rule["field"]}))
            if isinstance(result.get("op"), str) and result["op"] in {"all", "any"}:
                result["args"] = sorted(result["args"], key=lambda value: json.dumps(value, sort_keys=True))
            return result
        if isinstance(rule, list):
            return [expand_rule(item) for item in rule]
        return rule
    for field in RULE_FIELDS:
        body[field] = expand_rule(body[field])
    body["data_inputs"] = sorted(inputs.values(), key=lambda value: json.dumps(value, sort_keys=True))
    body["feature_program"] = sorted(features.values(), key=lambda value: json.dumps(value, sort_keys=True))
    program = normalize(body)
    parameters = normalize(body, parameters=True)
    return {
        "program_signature": hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "parameter_family_signature": hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
    }


def _rule_valid(rule: Any, depth: int = 0) -> bool:
    if (depth > 12 or not isinstance(rule, dict) or not isinstance(rule.get("op"), str)
            or rule["op"] not in OPS):
        return False
    op = rule["op"]
    if op in {"all", "any"}:
        args = rule.get("args")
        return (isinstance(args, list) and 0 < len(args) <= 32
                and all(_rule_valid(arg, depth + 1) for arg in args))
    if op == "not":
        return _rule_valid(rule.get("arg"), depth + 1)
    if op == "constant":
        return isinstance(rule.get("value"), bool)
    if op in {"fraction", "take_profit", "stop_loss", "timeout"}:
        value = rule.get("value")
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0
                and (op != "fraction" or value <= 1))
    return (isinstance(rule.get("field"), str) and bool(rule["field"].strip())
            and "value" in rule and rule["value"] is not None
            and (op != "in" or isinstance(rule["value"], list)))


def validate_exact_candidate(payload: Mapping[str, Any]) -> AdmissionResult:
    reasons = []
    if not isinstance(payload, Mapping):
        return AdmissionResult("BLOCKED", ("candidate must be an object",))
    if payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        reasons.append("unsupported candidate schema_version")
    for name in TEXT_FIELDS:
        if not isinstance(payload.get(name), str) or not payload[name].strip():
            reasons.append(f"missing measurable {name}")
    for name in OBJECT_FIELDS:
        if not isinstance(payload.get(name), dict) or not payload[name]:
            reasons.append(f"missing {name}")
    for name in LIST_FIELDS:
        if not isinstance(payload.get(name), list) or not payload[name]:
            reasons.append(f"missing {name}")
    if not isinstance(payload.get("parent_ids"), list):
        reasons.append("parent_ids must be explicit, including an empty root list")
    elif (any(not isinstance(parent, str) or not parent for parent in payload["parent_ids"])
          or payload.get("candidate_id") in payload["parent_ids"]
          or len(set(payload["parent_ids"])) != len(payload["parent_ids"])):
        reasons.append("parent_ids must contain distinct named ancestors, never the candidate itself")
    for name in RULE_FIELDS:
        if not _rule_valid(payload.get(name)):
            reasons.append(f"invalid executable {name}")
    for name in ("created_at", "entry_horizon_seconds", "max_decision_latency_ms"):
        value = payload.get(name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            reasons.append(f"invalid {name}")
    inputs = payload.get("data_inputs")
    if not isinstance(inputs, list) or not inputs:
        reasons.append("missing data_inputs")
    else:
        for item in inputs:
            if (not isinstance(item, dict) or not item.get("name")
                    or not item.get("source") or item.get("point_in_time") is not True
                    or not item.get("availability_timestamp")
                    or not item.get("event_timestamp")):
                reasons.append("every data input needs source, PIT and event/availability timestamps")
    program = payload.get("feature_program")
    if isinstance(program, dict):
        available = {item.get("name") for item in inputs or []
                     if isinstance(item, dict) and isinstance(item.get("name"), str)} if isinstance(inputs, list) else set()
        for name, expression in program.items():
            if (not isinstance(name, str) or not name
                    or not isinstance(expression, dict)
                    or not isinstance(expression.get("op"), str)
                    or expression["op"] not in {"identity", "ratio", "difference"}):
                reasons.append("feature_program must use named identity/ratio/difference expressions")
                continue
            refs = ([expression.get("input")] if expression["op"] == "identity"
                    else [expression.get("left"), expression.get("right")])
            if any(not isinstance(ref, str) or ref not in available for ref in refs):
                reasons.append(f"feature {name} references an unavailable observable")
    try:
        seal = canonical_candidate_hash(payload)
    except (ValueError, TypeError):
        return AdmissionResult("BLOCKED", tuple(reasons + ["candidate is not finite canonical JSON"]))
    if payload.get("candidate_hash") and payload["candidate_hash"] != seal:
        reasons.append("candidate content hash mismatch")
    return AdmissionResult("BLOCKED" if reasons else "EXACT_CANDIDATE",
                           tuple(reasons), seal)
