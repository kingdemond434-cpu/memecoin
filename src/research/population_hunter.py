"""Bounded PIT actor/population/narrative materialization, called by research.

Facts and source text belong to the canonical evidence ledger. This component
keeps reference-only observations, frozen control assignments and measured
outcomes in a durable incremental index; it starts no worker and has no trading
interface. Identity labels, language and source operators require provenance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Optional, Sequence

from src.research.actor_store import ActorStore, EdgeKind, native_node
from src.research.population_index import BUCKETS, STORAGE, PopulationIndex
from src.strategies.wallet_value import FollowOutcome, WalletValueModel


COMPONENT = "population_hunter_v1"
KINDS = {"ACTOR_OBSERVATION", "FOLLOW_OUTCOME", "NARRATIVE_OBSERVATION", "SOURCE_OUTCOME"}
STRATA = ("chain_id", "protocol", "lifecycle_stage", "regime", "capital_size", "entry_delay_seconds")
STAGES = {"MENTION", "WALLET_PARTICIPATION", "LIQUIDITY_RESPONSE", "PRICE_RESPONSE", "SATURATION"}
LABELS = {"ORIGINAL", "CLONE", "PARODY", "IMPERSONATION", "MIGRATION", "PROMOTION"}
LABEL_METHODS = {"OFFICIAL_PROJECT_STATEMENT", "VERIFIED_CONTRACT_LINEAGE", "SOURCE_AUTHENTICITY_REVIEW"}
QUALITY = ("authentic", "organic", "edited", "deleted", "paid_promotion", "bot",
           "contract_address_accurate", "independently_confirmed", "false_positive", "reliable")
COSTS = ("fees_usd", "gas_usd", "tips_usd", "failed_transaction_usd", "crowding_usd")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} requires an explicit finite {'positive' if positive else 'nonnegative'} measurement")
    return float(value)


def _refs(values: Sequence[str]) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values or len(values) > 256 or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("canonical evidence references required")
    return sorted(set(values))


def _required(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} required")
    return value.strip()


class PopulationHunter:
    """Consume one evidenced observation at a time, with restart-safe receipts.

    Re-delivery returns the original children even if a downstream canonical
    receipt failed after our checkpoint. No counts, graph edges or assignments
    are repeated. Capacity exhaustion retains existing work and reports debt.
    """

    def __init__(self, framework: Any, actor_store: Optional[ActorStore] = None, *, max_observations: int = 20_000):
        self.framework = framework
        self.actor_store = actor_store if actor_store is not None else ActorStore()
        self.max_observations = max(1, int(max_observations))
        self.state = framework.get_research_component(COMPONENT) or {
            "schema": 1, "actors": {}, "outcomes": {}, "narratives": {},
            "source_outcomes": {}, "controls": {}, "receipts": {}, "receipt_observed_at": {}}
        self.index = None
        state_path = getattr(framework, "state_path", None)
        if state_path:
            from pathlib import Path
            path = Path(state_path).with_suffix(".population.sqlite")
            if self.state.get("storage") == STORAGE:
                if self.state.get("file") != path.name:
                    raise RuntimeError("population index descriptor path mismatch")
                self.index = PopulationIndex(path, required=True)
                self.state = framework._research_checkpoint(self.index.load())
            else:
                self.state.setdefault("receipt_observed_at", {})
                self.index = PopulationIndex(path)
                self.index.initialize(self.state)
                framework.checkpoint_research_component(COMPONENT, self.index.descriptor)
        self.state.setdefault("receipt_observed_at", {})
        for row in self.state["actors"].values():
            self._graph(row)

    def close(self) -> None:
        if self.index is not None:
            self.index.close()

    def _graph(self, row: dict) -> None:
        relation = row.get("relation")
        if not relation:
            return
        self.actor_store.observe_native(chain_id=row["chain_id"], chain_family=row["chain_family"],
            source=row["address"], target=relation["target"], kind=EdgeKind(relation["kind"]),
            event_at=row["event_at"], observed_at=row["observed_at"],
            evidence_refs=row["evidence_refs"], observation_id=row["id"],
            source_type=relation.get("source_type", "wallet"), target_type=relation.get("target_type", "wallet"),
            target_chain_id=relation.get("target_chain_id"), target_chain_family=relation.get("target_chain_family"))

    @staticmethod
    def _axes(payload: Mapping[str, Any]) -> dict:
        axes = {"chain": payload.get("chain_id"), "chain_family": payload.get("chain_family"),
                "protocol": payload.get("protocol"), "lifecycle_stage": payload.get("lifecycle_stage"),
                "capital_size": payload.get("capital_size"), "entry_delay": payload.get("entry_delay_seconds"),
                "data_source": payload.get("source_id"), "narrative": payload.get("narrative_id")}
        if payload.get("language"):
            axes["region_language"] = f"{payload.get('region') or 'UNKNOWN'}:{payload['language']}"
        return {key: str(value) for key, value in axes.items() if value is not None}

    def _request(self, variable: str, mechanism: str, payload: dict, refs: list[str], *, kind: str = "DATA_REQUEST") -> dict:
        body = {"variable": variable, "mechanism": mechanism,
            "source_classes": [payload.get("platform") or "canonical_native_observation"],
            "pit_requirements": ["available_at <= decision_at", "frozen pre-outcome sampling and controls"],
            "expected_consumer": "population_hunter", "strategy_family": "actor_narrative_discovery",
            "acquisition_cost": None, "validation_plan": "chronological held-out delayed executable outcomes versus matched controls",
            "falsifier": "net delayed executable edge is nonpositive or source independence is unverified",
            "evidence_refs": refs, "subject": payload.get("actor_id") or payload.get("narrative_id") or payload.get("observation_id")}
        body["request_id"] = _hash([kind, body])
        return {"kind": kind, "payload": body, "axes": self._axes(payload)}

    def consume(self, kind: str, payload: Mapping[str, Any], evidence_refs: Sequence[str], now: float) -> dict:
        kind = str(kind).upper()
        try:
            refs = _refs(evidence_refs)
            now = _number(now, "now")
            p = dict(payload)
            if len(json.dumps(p, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 131_072:
                raise ValueError("bounded population observation exceeds 128 KiB; retain raw canonical evidence and submit a compact fact")
            identity = _hash([kind, p, refs])
            if kind not in KINDS:
                raise ValueError("unsupported population observation kind")
            event_at = _number(p.get("event_at"), "event_at")
            observed_at = _number(p.get("observed_at"), "observed_at")
            if not event_at <= observed_at <= now:
                raise ValueError("event_at <= observed_at <= now required")
            if identity in self.state["receipts"]:
                return deepcopy(self.state["receipts"][identity])
            if len(self.state["receipts"]) >= self.max_observations:
                return {"disposition": "BLOCKED", "detail": {"reason": "compact index capacity reached; retain canonical input and increase reviewed capacity"}, "children": []}
            # All per-identity rows are immutable. Copy the seven maps, not
            # every historical receipt and child request for each new fact.
            state = {"schema": self.state["schema"], **{name: dict(self.state[name]) for name in BUCKETS}}
            method = getattr(self, "_" + kind.lower())
            detail, children = method(state, p, refs, identity)
            result = {"disposition": "OBSERVED", "detail": detail, "children": children}
            state["receipts"][identity] = result
            state["receipt_observed_at"][identity] = observed_at
            if self.index is None:
                self.framework.checkpoint_research_component(COMPONENT, state)
            else:
                changes = {name: {key: value for key, value in state[name].items()
                                  if key not in self.state[name] or value is not self.state[name][key]}
                           for name in BUCKETS}
                changes = {name: rows for name, rows in changes.items() if rows}
                changes = self.framework._research_checkpoint(changes)
                self.index.apply(changes)
            self.state = state
            return deepcopy(result)
        except (ValueError, TypeError, KeyError) as exc:
            return {"disposition": "DATA_BLOCKED", "detail": {"reason": str(exc)},
                    "enabling_change": "population-contract:" + kind, "children": []}

    @staticmethod
    def _actor_id(p: Mapping[str, Any]) -> str:
        return native_node(_required(p, "chain_id"), "wallet", _required(p, "address"),
                           _required(p, "chain_family"))

    def _actor_observation(self, state: dict, p: dict, refs: list[str], identity: str) -> tuple:
        actor_id = self._actor_id(p)
        semantic_id = _hash([actor_id, p.get("source_id"), p["observation_id"], p.get("relation")]) if p.get("observation_id") else identity
        if semantic_id in state["actors"]:
            original = state["actors"][semantic_id]
            if original["event_at"] != p["event_at"]:
                raise ValueError("native observation ID has conflicting event time; explicit correction required")
            cached = state["receipts"][original["receipt_key"]]
            return deepcopy(cached["detail"]), deepcopy(cached["children"])
        row = {key: p.get(key) for key in (*STRATA, "chain_family", "address", "event_at", "observed_at", "source_id")}
        row.update(id=semantic_id, receipt_key=identity, actor_id=actor_id, evidence_refs=refs, population="UNKNOWN")
        selection = p.get("selection") or {}
        if selection:
            selected_at = _number(selection.get("selected_at"), "selected_at")
            if not selected_at <= p["observed_at"] or selection.get("uses_outcomes") is not False:
                raise ValueError("sampling must be declared by observation time without outcome selection")
            for field in ("rule_id", "sampling_frame_ref"):
                _required(selection, field)
            if selection.get("population") not in {"FAMOUS_BENCHMARK", "UNBIASED"}:
                raise ValueError("population requires explicit famous benchmark or declared-frame unbiased sampling")
            if selection["population"] == "FAMOUS_BENCHMARK":
                _required(selection, "benchmark_ref")
            additional = selection.get("additional_populations", [])
            if not isinstance(additional, list) or any(name != "FAMOUS_BENCHMARK" for name in additional):
                raise ValueError("additional population membership requires an explicit benchmark contract")
            if additional:
                _required(selection, "benchmark_ref")
            row["population"] = selection["population"]
            row["population_memberships"] = sorted(set([row["population"], *additional]))
            row["selection"] = {key: selection[key] for key in ("selected_at", "rule_id", "sampling_frame_ref", "uses_outcomes", "benchmark_ref", "additional_populations") if key in selection}
        relation = p.get("relation")
        if relation:
            EdgeKind(_required(relation, "kind"))
            row["relation"] = {"kind": relation["kind"], "target": _required(relation, "target"),
                "target_type": relation.get("target_type", "wallet"), "source_type": relation.get("source_type", "wallet"),
                "target_chain_id": relation.get("target_chain_id"), "target_chain_family": relation.get("target_chain_family")}
            if relation.get("target_chain_id") and not relation.get("target_chain_family"):
                raise ValueError("cross-chain target requires an explicit target chain family")
            self._graph(row)
        state["actors"][semantic_id] = row
        subject = {**p, "actor_id": actor_id}
        children = [self._request("delayed_executable_follow_outcomes", "observable wallet action may precede executable net growth", subject, refs)]
        if row["population"] == "UNKNOWN":
            children.append(self._request("pre_outcome_sampling_frame", "fame and survivor selection can manufacture apparent wallet edge", subject, refs))
        control = self._freeze_controls(state, row, p.get("control_plan"))
        if control["status"] != "MATCHED":
            children.append(self._request("matched_losing_shuffled_delay_controls", "a mechanism must exceed frozen matched negative and randomized controls", subject, refs))
        structure = self.actor_structure(p["chain_id"], p["chain_family"], p["address"], as_of=p["observed_at"])
        if relation:
            children.append(self._request("native_relation_" + relation["kind"],
                "observed native relationships may explain repeated participation; address association does not establish identity or edge",
                subject, refs, kind="MECHANISM_HYPOTHESIS"))
        return {"observation_id": semantic_id, "actor_id": actor_id, "population": row["population"], "controls": control,
                "structure": structure, "identity": "UNKNOWN", "followable": False, "status": "AWAITING_EXECUTABLE_OUTCOMES"}, children

    def actor_structure(self, chain_id: str, chain_family: str, address: str, *, as_of: float) -> dict:
        """Observed funding/reuse/participation structure, without identity claims.

        Shared target IDs are only graph associations. Missing historical
        coverage cannot make an address fresh, independent, or an insider.
        """
        moment = _number(as_of, "as_of")
        node = native_node(chain_id, "wallet", address, chain_family)
        family = self.actor_store.family(node, moment)
        counts, shared, refs = {}, {}, set()
        for kind in EdgeKind:
            edges = self.actor_store.out_edges(node, kind, moment)
            counts[kind.value] = len(edges)
            peers = set()
            for edge in edges[:64]:
                refs.update(edge.evidence_refs)
                peers.update(other.source for other in self.actor_store.in_edges(edge.target, kind, moment)
                             if other.source != node)
            if peers:
                shared[kind.value] = sorted(peers)[:64]
        return {"node": node, "as_of": moment, "outgoing_relation_counts": counts,
            "funding_ancestry": family.to_dict(), "shared_target_associations": shared,
            "evidence_refs": sorted(refs), "fresh_wallet_status": "UNKNOWN",
            "economic_identity": "UNKNOWN", "association_scan_limit_per_kind": 64}

    def _freeze_controls(self, state: dict, row: dict, plan: Any) -> dict:
        if not plan or row["population"] == "UNKNOWN":
            return {"status": "DATA_BLOCKED", "reason": "declared selection and randomized-delay plan required"}
        trial_id = _required(plan, "trial_id")
        seed = _required(plan, "seed")
        grid = sorted(set(_number(x, "delay_grid_seconds", positive=True) for x in plan.get("delay_grid_seconds", [])))
        if not 2 <= len(grid) <= 64:
            raise ValueError("bounded predeclared delay grid requires at least two distinct positive choices")
        selected_at = row["selection"]["selected_at"]
        trial_key = _hash([row["actor_id"], trial_id])
        if trial_key in state["controls"]:
            old = state["controls"][trial_key]
            if old["plan_hash"] != _hash(plan):
                raise ValueError("frozen control plan cannot be changed")
            return old
        complete_strata = all(row.get(k) is not None for k in STRATA)
        eligible = {r["actor_id"]: r for r in state["actors"].values()
                    if complete_strata
                    and r["observed_at"] <= selected_at and r["actor_id"] != row["actor_id"]
                    and r["population"] in {"UNBIASED", "FAMOUS_BENCHMARK"}
                    and all(r.get(k) == row.get(k) for k in STRATA)}
        losers = sorted({o["actor_id"] for o in state["outcomes"].values()
                         if o["available_at"] <= selected_at and o["observed_at"] <= selected_at
                         and o["multiple"] < 1 and o["actor_id"] in eligible})
        ordered = sorted(eligible, key=lambda actor: _hash([seed, trial_id, actor]))
        chosen = ordered[0] if ordered else None
        populations = {name for r in [*eligible.values(), row] for name in r.get("population_memberships", [r["population"]])}
        result = {"status": "MATCHED" if chosen and losers and {"UNBIASED", "FAMOUS_BENCHMARK"} <= populations else "DATA_BLOCKED", "trial_id": trial_id,
            "actor_id": row["actor_id"], "selected_at": selected_at, "plan_hash": _hash(plan),
            "observed_at": row["observed_at"], "matched_populations": sorted(populations),
            "missing_matching_strata": [k for k in STRATA if row.get(k) is None],
            "matched_actor_ids": sorted(eligible), "losing_actor_id": losers[0] if losers else None,
            "shuffled_actor_id": chosen, "shuffled_mapping_rule": "seeded hash assignment excluding self; no returns used",
            "delay_seconds": grid[int(_hash([seed, trial_id, row["actor_id"]]), 16) % len(grid)],
            "delay_grid_seconds": grid, "evidence_refs": row["evidence_refs"]}
        state["controls"][trial_key] = result
        return result

    def _follow_outcome(self, state: dict, p: dict, refs: list[str], identity: str) -> tuple:
        actor_id = self._actor_id(p)
        outcome_id, token_id = _required(p, "outcome_id"), _required(p, "token_id")
        scope = _required(p, "execution_scope")
        if scope not in {"ACTOR_DELAYED_PAPER", "ACTOR_EXECUTED_FOLLOW"}:
            raise ValueError("only explicitly attributed actor paper/follow receipts are accepted; candidate-native simulation is a separate scope")
        failed_entry = p.get("execution_status") == "FAILED_ENTRY"
        resolution_clock = "resolution_at" if failed_entry else "exit_at"
        clock = [_number(p.get(k), k) for k in ("signal_available_at", "decision_at", "entry_at", resolution_clock, "available_at", "observed_at")]
        if clock != sorted(clock) or p["event_at"] > p["available_at"]:
            raise ValueError("executable outcome clocks must be chronological and available before observation")
        delay = _number(p.get("delay_seconds"), "delay_seconds", positive=True)
        if not math.isclose(delay, p["entry_at"] - p["signal_available_at"], abs_tol=1e-6):
            raise ValueError("delay must measure our actual entry after signal availability")
        if p.get("crowding_observed") is not True:
            raise ValueError("measured crowding required")
        capital = _number(p.get("capital_at_risk_usd"), "capital_at_risk_usd", positive=True)
        entry = _number(p.get("entry_fill_cost_usd"), "entry_fill_cost_usd")
        proceeds = _number(p.get("exit_fill_proceeds_usd"), "exit_fill_proceeds_usd")
        costs = {k: _number((p.get("costs") or {}).get(k), k) for k in COSTS}
        if failed_entry:
            if (p.get("entry_filled") is not False or p.get("exit_filled") is not False
                    or _number(p.get("inventory_at_resolution_raw"), "inventory_at_resolution_raw") != 0
                    or entry != 0 or proceeds != 0
                    or _number(p.get("exit_capacity_ratio"), "exit_capacity_ratio") != 0):
                raise ValueError("resolved failed entry requires explicitly zero acquired inventory, fill cost, proceeds and exit capacity")
        else:
            if p.get("execution_status", "FILLED") != "FILLED" or any(p.get(k) is not True for k in (
                    "slippage_in_fills", "impact_in_fills", "entry_filled", "exit_filled")):
                raise ValueError("resolved executable entry/exit and fill slippage/impact required; open failed exits remain blocked")
            if _number(p.get("exit_capacity_ratio"), "exit_capacity_ratio") != 1:
                raise ValueError("unresolved or capacity-limited exit cannot be represented as a full realized return")
            if entry <= 0 or entry > capital:
                raise ValueError("actual positive fill cost cannot exceed declared capital at risk")
        multiple = (capital + proceeds - entry - sum(costs.values())) / capital
        key = _hash([actor_id, outcome_id])
        row = {"id": key, "actor_id": actor_id, "token_id": native_node(p["chain_id"], "token", token_id, p["chain_family"]),
            "available_at": p["available_at"], "observed_at": p["observed_at"], "multiple": multiple,
            "delay_seconds": delay, "regime": p.get("regime", ""), "control_trial_id": p.get("control_trial_id"),
            "signal_available_at": p["signal_available_at"], "source_observation_id": p.get("source_observation_id"),
            "execution_scope": scope, "execution_status": "FAILED_ENTRY" if failed_entry else "FILLED",
            "costs": costs, "evidence_refs": refs}
        if key in state["outcomes"] and state["outcomes"][key] != row:
            raise ValueError("outcome identity already has a different measured value; explicit correction required")
        state["outcomes"][key] = row
        return {"outcome_id": key, "actor_id": actor_id, "executable_multiple": multiple,
                "execution_scope": scope, "execution_status": row["execution_status"],
                "value": self._actor_value(state, actor_id, p["observed_at"], execution_scope=scope)}, []

    def _actor_value(self, state: dict, actor_id: str, as_of: float, *, execution_scope: str = "ACTOR_DELAYED_PAPER") -> dict:
        if execution_scope not in {"ACTOR_DELAYED_PAPER", "ACTOR_EXECUTED_FOLLOW"}:
            raise ValueError("explicit actor execution scope required")
        model = WalletValueModel()
        seen = set()
        visible = sorted((row for row in state["outcomes"].values()
            if row.get("execution_scope") == execution_scope and row["available_at"] <= as_of and row["observed_at"] <= as_of),
            key=lambda r: (r["available_at"], r["id"]))
        by_trial = {}
        for row in visible:
            by_trial.setdefault(row["control_trial_id"], []).append(row)
            # Independent launches, not alternate entries or duplicate control fills.
            key = (row["actor_id"], row["token_id"])
            if row["available_at"] > as_of or row["observed_at"] > as_of or key in seen:
                continue
            seen.add(key)
            model.record(FollowOutcome(row["actor_id"], row["token_id"], row["available_at"],
                row["multiple"], regime=row["regime"], rugged=row["multiple"] <= 0,
                follow_latency_s=row["delay_seconds"]))
        value = model.value(actor_id, now=as_of)
        matched = [c for c in state["controls"].values() if c["actor_id"] == actor_id
                   and c["observed_at"] <= as_of and c["status"] == "MATCHED"]
        control_model = WalletValueModel()
        matched_tokens = set()
        for control in matched:
            required = {actor_id, control["losing_actor_id"], control["shuffled_actor_id"]}
            by_token = {}
            for row in by_trial.get(control["trial_id"], ()):
                if (row["available_at"] <= as_of and row["observed_at"] <= as_of
                        and row["control_trial_id"] == control["trial_id"]
                        and row["available_at"] > control["selected_at"]
                        and math.isclose(row["delay_seconds"], control["delay_seconds"], abs_tol=1e-6)):
                    by_token.setdefault(row["token_id"], {}).setdefault(row["actor_id"], row)
            for token, group in by_token.items():
                if token in matched_tokens or not required <= set(group):
                    continue
                matched_tokens.add(token)
                baseline = sum(max(1e-4, group[a]["multiple"]) for a in required - {actor_id}) / len(required - {actor_id})
                excess = max(1e-4, group[actor_id]["multiple"]) / baseline
                control_model.record(FollowOutcome(actor_id, token, as_of, excess))
        relative_value = control_model.value(actor_id, now=as_of)
        controls_measured = relative_value.ok
        return {"status": value.status, "samples": value.samples, "execution_scope": execution_scope,
            "mean_log_return": value.mean_log_return if value.ok else None,
            "lower_bound": value.lower_bound if value.ok else None,
            "controls_measured": controls_measured,
            "control_samples": relative_value.samples,
            "incremental_lower_bound": relative_value.lower_bound if relative_value.ok else None,
            "followable": bool(value.followable and relative_value.followable),
            "detail": value.detail if not value.ok else "positive delayed net edge and measured frozen controls required"}

    def actor_value(self, chain_id: str, chain_family: str, address: str, *, as_of: float,
                    execution_scope: str = "ACTOR_DELAYED_PAPER") -> dict:
        return self._actor_value(self.state, native_node(chain_id, "wallet", address, chain_family), _number(as_of, "as_of"), execution_scope=execution_scope)

    def _narrative_observation(self, state: dict, p: dict, refs: list[str], identity: str) -> tuple:
        narrative_id, source_id = _required(p, "narrative_id"), _required(p, "source_id")
        semantic_id = _hash([narrative_id, source_id, p["observation_id"]]) if p.get("observation_id") else identity
        platform, language = _required(p, "platform"), _required(p, "language")
        stage = p.get("stage", "MENTION")
        if stage not in STAGES:
            raise ValueError("unknown narrative propagation stage")
        label, provenance = "UNKNOWN", p.get("label_provenance") or {}
        if p.get("label") in LABELS and provenance.get("verified") is True:
            if provenance.get("verification_method") not in LABEL_METHODS:
                raise ValueError("explicit reviewed narrative label provenance required")
            if _number(provenance.get("available_at"), "label available_at") > p["observed_at"]:
                raise ValueError("future label provenance cannot identify a narrative")
            _refs(provenance.get("evidence_refs"))
            label = p["label"]
        operator = None
        if p.get("operator_verified") is True:
            operator = _required(p, "operator_id")
            _refs(p.get("operator_evidence_refs"))
        text = p.get("text")
        content_hash, token_hashes = None, []
        if text is not None:
            if not isinstance(text, str) or len(text) > 32_768:
                raise ValueError("text must be a bounded canonical-ledger-backed source observation")
            normalized = unicodedata.normalize("NFKC", text).casefold()
            content_hash = _hash(normalized)
            token_hashes = sorted({_hash(token) for token in re.findall(r"\w+", normalized, flags=re.UNICODE)})[:128]
        row = {"id": semantic_id, "receipt_key": identity, "observation_id": p.get("observation_id") or semantic_id,
            "narrative_id": narrative_id, "source_id": source_id, "platform": platform,
            "language": language, "region": p.get("region"), "operator_id": operator,
            "operator_evidence_refs": p.get("operator_evidence_refs", []) if operator else [],
            "event_at": p["event_at"], "observed_at": p["observed_at"], "stage": stage,
            "label": label, "label_provenance": {k: provenance[k] for k in ("verified", "verification_method", "available_at", "evidence_refs")} if label != "UNKNOWN" else None,
            "content_hash": content_hash, "token_hashes": token_hashes, "evidence_refs": refs,
            "chain_id": p.get("chain_id"), "token_id": p.get("token_id")}
        if semantic_id in state["narratives"]:
            original = state["narratives"][semantic_id]
            if any(original.get(key) != row.get(key) for key in ("event_at", "content_hash", "label", "language", "operator_id")):
                raise ValueError("narrative observation ID has conflicting facts; explicit correction required")
            cached = state["receipts"][original["receipt_key"]]
            return deepcopy(cached["detail"]), deepcopy(cached["children"])
        state["narratives"][semantic_id] = row
        children = [self._request("source_quality_and_executable_latency", "native-language information may precede delayed executable participation", p, refs)]
        if operator is None:
            children.append(self._request("verified_source_operator_independence", "reposts and common operators do not provide independent corroboration", p, refs, kind="SOURCE_REQUEST"))
        if label == "UNKNOWN":
            children.append(self._request("verified_narrative_lineage", "clone and promotion labels require provenance rather than textual resemblance", p, refs, kind="REPRESENTATION_REQUEST"))
        return {"observation_id": row["observation_id"], "label": label,
                "propagation": self._narrative_report(state, narrative_id, p["observed_at"])}, children

    def _source_outcome(self, state: dict, p: dict, refs: list[str], identity: str) -> tuple:
        observation_id = _required(p, "observation_id")
        matches = [r for r in state["narratives"].values() if r["observation_id"] == observation_id and r["observed_at"] <= p["event_at"]]
        if len(matches) != 1:
            raise ValueError("source evaluation must reference one already observable narrative observation")
        evaluations = p.get("evaluations") or {}
        if any(k not in QUALITY or type(v) is not bool for k, v in evaluations.items()):
            raise ValueError("source quality uses actual explicit boolean evaluations only")
        if not evaluations and p.get("executable_net_return") is None:
            raise ValueError("at least one actually evaluated quality metric required")
        _refs(p.get("evaluation_evidence_refs"))
        row = {"observation_id": observation_id, "source_id": matches[0]["source_id"],
            "observed_at": p["observed_at"], "event_at": p["event_at"], "evaluations": dict(evaluations),
            "evidence_refs": refs, "evaluation_evidence_refs": p["evaluation_evidence_refs"]}
        if p.get("executable_net_return") is not None:
            value = p["executable_net_return"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("finite actually executable source return required")
            _refs(p.get("entry_exit_evidence_refs"))
            if _number(p.get("evaluated_at"), "evaluated_at") > p["observed_at"]:
                raise ValueError("source return unavailable at observation")
            follow = state["outcomes"].get(_required(p, "follow_outcome_id"))
            if (not follow or follow["observed_at"] > p["observed_at"]
                    or follow["source_observation_id"] != observation_id
                    or matches[0]["observed_at"] > follow["signal_available_at"]
                    or not math.isclose(value, follow["multiple"] - 1, abs_tol=1e-9)):
                raise ValueError("source return must reference an already validated delayed executable outcome explicitly attributed to this observation")
            row["executable_net_return"] = value
        for existing in state["source_outcomes"].values():
            if existing["observation_id"] == observation_id:
                if any(k in existing["evaluations"] and existing["evaluations"][k] != v for k, v in evaluations.items()):
                    raise ValueError("source metric already evaluated differently; explicit correction required")
                if ("executable_net_return" in row and "executable_net_return" in existing
                        and row["executable_net_return"] != existing["executable_net_return"]):
                    raise ValueError("source return already evaluated differently; explicit correction required")
        state["source_outcomes"][identity] = row
        return {"source_id": row["source_id"], "evaluated_observation_id": observation_id}, []

    @staticmethod
    def _narrative_report(state: dict, narrative_id: str, as_of: float) -> dict:
        rows = [r for r in state["narratives"].values() if r["narrative_id"] == narrative_id and r["observed_at"] <= as_of]
        rows.sort(key=lambda r: (r["observed_at"], r["id"]))
        unique, hashes = [], set()
        for row in rows:
            if row["content_hash"] is not None and row["content_hash"] in hashes:
                continue
            if row["content_hash"] is not None:
                hashes.add(row["content_hash"])
            unique.append(row)
        independent = [r for r in unique if r["operator_id"] is not None]
        operators = {r["operator_id"] for r in independent}
        stages = {stage: min((r["observed_at"] for r in rows if r["stage"] == stage), default=None) for stage in sorted(STAGES)}
        return {"observations": len(rows), "distinct_sources": len({r["source_id"] for r in rows}),
            "verified_independent_operators": len(operators) if independent else None,
            "independent_confirmation": len(operators) >= 2,
            "independent_cross_platform": len(operators) >= 2 and len({r["platform"] for r in independent}) >= 2,
            "independent_cross_region": len(operators) >= 2 and len({r["region"] for r in independent if r["region"]}) >= 2,
            "independent_platforms": sorted({r["platform"] for r in independent}),
            "independent_regions": sorted({r["region"] for r in independent if r["region"]}),
            "languages": sorted({r["language"] for r in rows}), "stage_first_observed_at": stages,
            "first_seen_at": min((r["observed_at"] for r in rows), default=None),
            "source_clock_first_mention_at": min((r["event_at"] for r in rows), default=None),
            "reposted_content": len(rows) - len(unique), "labels": sorted({r["label"] for r in rows})}

    def narrative(self, narrative_id: str, *, as_of: float) -> dict:
        return self._narrative_report(self.state, narrative_id, _number(as_of, "as_of"))

    def report(self, *, as_of: Optional[float] = None) -> dict:
        cutoff = math.inf if as_of is None else _number(as_of, "as_of")
        actors = [r for r in self.state["actors"].values() if r["observed_at"] <= cutoff]
        narratives = [r for r in self.state["narratives"].values() if r["observed_at"] <= cutoff]
        source_ids = sorted({r["source_id"] for r in narratives})
        sources = {}
        for source in source_ids:
            rows = [r for r in self.state["source_outcomes"].values() if r["source_id"] == source and r["observed_at"] <= cutoff]
            metrics = {}
            for metric in QUALITY:
                evaluated = {r["observation_id"]: r["evaluations"][metric] for r in rows if metric in r["evaluations"]}
                values = list(evaluated.values())
                metrics[metric] = {"numerator": sum(values), "denominator": len(values),
                                   "rate": sum(values) / len(values) if values else None}
            returns = list({r["observation_id"]: r["executable_net_return"] for r in rows if "executable_net_return" in r}.values())
            sources[source] = {"observations": sum(r["source_id"] == source for r in narratives),
                "evaluated_observations": len({r["observation_id"] for r in rows}), "metrics": metrics,
                "mean_executable_net_return": sum(returns) / len(returns) if returns else None,
                "executable_return_denominator": len(returns), "lead_time_seconds": None,
                "cross_source_correlation": None, "acquisition_cost": None}
        population = {name: len({r["actor_id"] for r in actors if name in r.get("population_memberships", [r["population"]])})
                      for name in ("FAMOUS_BENCHMARK", "UNBIASED", "UNKNOWN")}
        controls = [c for c in self.state["controls"].values() if c["observed_at"] <= cutoff]
        population.update(MATCHED_LOSING=len({c["losing_actor_id"] for c in controls if c["losing_actor_id"]}),
                          SHUFFLED_CONTROL=len({c["shuffled_actor_id"] for c in controls if c["shuffled_actor_id"]}),
                          RANDOMIZED_DELAY_CONTROL=len(controls))
        recent = {}
        for row in sorted(actors, key=lambda r: (r["observed_at"], r["id"]), reverse=True):
            recent.setdefault(row["actor_id"], row)
        value_cutoff = cutoff if as_of is not None else max(self.state["receipt_observed_at"].values(), default=0.)
        actor_values = [{"actor_id": actor_id, "population": row["population"],
            "value": self.actor_value(row["chain_id"], row["chain_family"], row["address"], as_of=value_cutoff)}
            for actor_id, row in list(recent.items())[:16]]
        narrative_ids = list(dict.fromkeys(row["narrative_id"] for row in
            sorted(narratives, key=lambda r: (r["observed_at"], r["id"]), reverse=True)))
        narrative_reports = [{"narrative_id": name, **self.narrative(name, as_of=value_cutoff)}
                             for name in narrative_ids[:16]]
        return {"status": "OBSERVATION_ONLY" if actors or narratives else "NO_OBSERVED_INPUTS", "authority": "none",
            "actor_observations": len(actors), "native_actor_count": len({r["actor_id"] for r in actors}),
            "populations": population, "controls": len(controls),
            "actor_values": actor_values, "actor_values_truncated": max(0, len(recent) - len(actor_values)),
            "measured_follow_outcomes": sum(r["observed_at"] <= cutoff for r in self.state["outcomes"].values()),
            "follow_outcomes_by_scope": {scope: sum(r["observed_at"] <= cutoff and r.get("execution_scope") == scope
                for r in self.state["outcomes"].values()) for scope in ("ACTOR_DELAYED_PAPER", "ACTOR_EXECUTED_FOLLOW")},
            "resolved_failed_entries": sum(r["observed_at"] <= cutoff and r.get("execution_status") == "FAILED_ENTRY"
                for r in self.state["outcomes"].values()),
            "narrative_observations": len(narratives), "sources": sources,
            "narratives": narrative_reports, "narratives_truncated": max(0, len(narrative_ids) - len(narrative_reports)),
            "receipt_count": sum(stamp <= cutoff for stamp in self.state["receipt_observed_at"].values()), "capacity": self.max_observations,
            "raw_text_persisted": False, "identity_claims": "UNKNOWN"}
