"""Mechanism recipes and exact training proposals for the existing work queue.

Every catalog entry owns an economic question, typed observables and a decisive
control. Missing observations/execution contracts become explicit acquisition
work. No thresholds, fees, fills or native semantics are invented.
"""
from __future__ import annotations

import copy
import json
import math
import time
from typing import Any, Mapping

from src.research.machine_search import (MachineSearch, SearchBudget, SearchDataError, SEARCH_METHODS,
                                         TARGET_SEMANTICS, digest, training_batch)
from src.research.research_contracts import canonical_candidate_identity, validate_exact_candidate
from src.strategies.champion_challenger import HypothesisSpec, TrialResult


def _lab(title, observables, incentive, discriminator, failure, experiment):
    return {"title": title, "observables": list(observables), "actor_incentive": incentive,
            "questions": {"forced_transactor": incentive, "early_information": observables[0],
                "discriminating_observation": discriminator, "failure_condition": failure,
                "cheapest_experiment": experiment}, "falsifier": failure, "experiment": experiment}


MECHANISM_LABS = {
    "prelaunch_intent": _lab("public prelaunch intent", ("prelaunch_artifact_lead_seconds", "verified_actor_recurrence"),
        "a publicly committed launcher must deploy or abandon a dated project", "verified artifacts precede unrelated public launches",
        "matched abandoned announcements predict equally well", "timestamp verified artifacts and match abandoned public projects"),
    "first_block": _lab("first-block microstructure", ("independent_first_block_buyers", "same_block_concentration"),
        "launch participants compete for finite first-block curve supply", "independent delayed arrivals survive Sybil compression",
        "edge vanishes after measured block delay and failed fills", "replay native first-block entries against shuffled ordering"),
    "bonding_curve": _lab("bonding-curve demand and exhaustion", ("curve_net_buy_velocity", "curve_progress"),
        "new demand must cross an increasingly expensive curve", "net buyer acceleration leads executable reserve change",
        "matched reserve-state controls explain the entire effect", "condition on PIT reserves and ablate buyer velocity"),
    "migration": _lab("liquidity migration and graduation", ("migration_distance", "confirmed_liquidity_commitment"),
        "a graduation rule forces liquidity onto a native destination venue", "verified migration commitment precedes route availability",
        "migration route and sell readiness remove apparent gain", "compare verified graduations with failed migrations at matched distance"),
    "actor_recurrence": _lab("actor/funder/deployer recurrence", ("actor_prior_executable_log_growth", "actor_reactivation_age"),
        "recurring public creators reuse observable funding and launch infrastructure", "PIT actor ancestry predicts subsequent delayed executable economics",
        "matched losing and shuffled actors perform equally", "freeze actor identity and compare matched losing creator cohorts"),
    "wallet_swarm": _lab("independent followable actor arrivals", ("independent_followable_actor_count", "actor_arrival_acceleration"),
        "independent skilled actors compete for available supply", "actor-compressed arrivals lead further executable demand",
        "follower delay, crowding and shared funding remove the effect", "randomize follower delay and shuffle actor labels"),
    "narrative_diffusion": _lab("social ignition and narrative diffusion", ("independent_source_velocity", "narrative_price_residual"),
        "new audiences must acquire tokens to participate in a public narrative", "independent original mentions precede liquidity response",
        "source genealogy and paid-promotion controls remove the lead", "ablate repeaters and align original-source timestamps before price"),
    "cross_chain_rotation": _lab("cross-chain clone and liquidity rotation", ("narrative_chain_lead_seconds", "executable_cross_chain_spread"),
        "cross-chain audiences rotate finite attention and bridged capital", "native executable clone lag follows original-chain attention",
        "bridge delay, gas and unavailable exits consume the spread", "native replay both chains with bridge dependence and clone controls"),
    "regional_waves": _lab("native-language retail waves", ("native_language_source_lead_seconds", "regional_organic_author_count"),
        "regional audiences encounter lawful native-language information at different times", "original regional sources lead local organic wallet arrival",
        "translated-English or paid-repeater controls explain the result", "compare native-language original sources with translation-only queries"),
    "listing_transitions": _lab("public listing and CEX transitions", ("verified_listing_signal", "venue_liquidity_change"),
        "a publicly announced venue transition changes who can acquire the token", "verified announcement precedes measured venue access and depth",
        "false announcements or inaccessible deposits explain apparent alpha", "replay verified listings against matched false-listing announcements"),
    "contract_risk": _lab("rug/honeypot and contract safety", ("sell_simulation_pass", "authority_risk_count"),
        "transfer controls and supply authority constrain actual exits", "native sell failures and authority events precede unrecoverable loss",
        "risk features leak later contract states or only improve ideal marks", "freeze authority state and replay buys plus emergency sell simulation"),
    "execution_optimization": _lab("route, landing and bid optimization", ("route_net_cost_fraction", "landing_probability"),
        "builders and validators ration scarce execution inclusion", "measured route inclusion improves net decision value after tips",
        "failed-transaction and adverse-selection costs erase the improvement", "compare identical frozen signals across measured route/bid policies"),
    "distribution_exits": _lab("distribution and late-chaser exits", ("independent_actor_exit_fraction", "late_chaser_arrival_rate"),
        "early holders sell inventory to arriving cohorts", "independent cohort exits precede depth loss beyond price alone",
        "actor-label shuffle or measured sell capacity removes the edge", "ablate cohort identities and compare executable delayed exits"),
    "monster_tail": _lab("rare-tail continuation", ("durable_actor_retention", "liquidity_absorption_rate"),
        "durable new buyers absorb early inventory while liquidity expands", "independent retention survives initial creator distribution",
        "no-monster periods exceed the declared shadow sleeve loss budget", "compare runner retention with rotation and long no-monster stress"),
    "abandoned_revival": _lab("abandoned-token revival", ("inactivity_duration_seconds", "renewed_independent_actor_count"),
        "new lawful attention can reactivate neglected but sellable inventory", "renewed independent demand appears before revived liquidity",
        "stale quotes or inaccessible exits explain the revival", "compare revived tokens with matched permanently abandoned launches"),
    "capital_velocity": _lab("portfolio replacement and capital velocity", ("replacement_net_log_growth", "capital_lock_seconds"),
        "finite capital committed to one token cannot fund another opportunity", "replacement exceeds both-leg costs and incumbent continuation value",
        "cash or incumbent holding dominates after executable switching costs", "compare each proposal against cash and frozen incumbent alternatives"),
    "research_methods": _lab("new scientific and computational methods", ("source_disagreement", "model_disagreement"),
        "limited inference and validation resources must buy discriminating information", "new representations separate rival mechanisms before held-out evaluation",
        "additional search only repeats existing parameter niches", "freeze competing representation programs and audit all outcome-inspecting trials"),
}

STRATEGY_FAMILIES = {
    "prelaunch_footprint": "prelaunch_intent", "first_block_participation": "first_block",
    "curve_continuation_exhaustion": "bonding_curve", "wallet_swarm_arrival": "wallet_swarm",
    "follower_delay_wallet": "wallet_swarm", "creator_funder_recurrence": "actor_recurrence",
    "sybil_discounted_coordination": "actor_recurrence", "migration_anticipation": "migration",
    "post_migration_continuation_reversal": "migration", "social_price_divergence": "narrative_diffusion",
    "verified_source_authenticity": "narrative_diffusion", "regional_language_leads": "regional_waves",
    "cross_chain_rotation": "cross_chain_rotation", "clone_saturation_original_rebound": "cross_chain_rotation",
    "liquidity_expansion_route_creation": "migration", "listing_cex_transition": "listing_transitions",
    "pool_route_dislocation": "execution_optimization", "creator_cohort_distribution_exits": "distribution_exits",
    "late_chaser_exits": "distribution_exits", "emergency_rug_avoidance": "contract_risk",
    "abandoned_token_revival": "abandoned_revival", "monster_tail_continuation": "monster_tail",
    "opportunity_replacement": "capital_velocity", "execution_only_edges": "execution_optimization",
}

MECHANISM_ALIASES = {
    "wallet_copy_policy": "wallet_swarm", "bundle_detection": "first_block",
    "curve_velocity": "bonding_curve", "liquidity_execution": "execution_optimization",
    "social_propagation": "narrative_diffusion", "risk_constrained_kelly": "capital_velocity",
    "optimal_stopping": "distribution_exits", "survival_hazard": "contract_risk",
    "order_flow_point_process": "bonding_curve", "probability_calibration": "research_methods",
    "public_coordination": "actor_recurrence",
}

EXECUTION_FIELDS = ("chain", "protocol", "lifecycle_stage", "universe_rule", "hard_vetoes", "size_rule",
    "fee_policy", "landing_assumptions", "execution_assumptions", "exit_rule", "emergency_exit",
    "entry_horizon_seconds", "max_decision_latency_ms", "holding_time_model")


def _contaminated(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if name in {"heldout", "held_out", "forward", "forward_outcomes", "agent_opinions",
                        "private_key", "credentials", "authority", "live_authority", "allow_live_trading", "disable_veto"}:
                return True
            if any(marker in name for marker in ("heldout", "held_out", "forward_", "agent_opinion")):
                return True
            if name in {"partition", "research_partition"} and item != "training":
                return True
            if name == "mode" and item not in {"shadow", "simulation", "observation"}:
                return True
            if _contaminated(item): return True
    elif isinstance(value, list):
        return any(_contaminated(item) for item in value)
    return False


class MechanismLaboratory:
    def __init__(self, framework: Any):
        self.framework = framework
        self.state = framework.get_research_component("mechanism_laboratory") or {
            "schema_version": 1, "completed": {}, "trial_keys": [], "lab_calls": {}, "search_archive": {}}

    def _checkpoint(self):
        self.framework.checkpoint_research_component("mechanism_laboratory", self.state)

    @staticmethod
    def _merge_observed(payload: Mapping, now: float) -> dict:
        """Resume from measured rows without inventing a join, label or mapping.

        Data acquisition can wake work before its *other* inputs exist. The
        resumed payload carries those acquired inputs into subsequent requests.
        Snapshot locations must come from a native source contract; a scalar
        measurement alone does not prove a replay feature mapping.
        """
        payload = copy.deepcopy(dict(payload))
        observed = payload.pop("observed_data", None)
        if observed is None: return payload
        if (not isinstance(observed, Mapping) or observed.get("partitions") != ["training"]
                or not isinstance(observed.get("variable"), str)
                or not isinstance(observed.get("rows"), list) or not 1 <= len(observed["rows"]) <= 512):
            raise SearchDataError("resumed data requires bounded explicit training observations")
        context = payload.setdefault("context", {key: value for key, value in payload.items()
            if key not in {"lab_id", "mechanism", "strategy_family", "questions"}})
        contracts = context.setdefault("observable_contracts", {})
        variable, rows = observed["variable"], observed["rows"]
        inputs = context.setdefault("acquired_inputs", {})
        # References and row identities explain any unresolved mapping; the
        # canonical evidence ledger retains the underlying measurements.
        refs = sorted({row["evidence_ref"] for row in rows if isinstance(row, Mapping) and row.get("evidence_ref")})
        inputs[variable] = {"evidence_refs": refs, "observations": len(rows)}
        for row in rows:
            if not isinstance(row, Mapping) or row.get("partition") != "training":
                raise SearchDataError("resumed observation is not training-only")
            stamps = [row.get(key) for key in ("event_at", "available_at", "observed_at")]
            if (not all(type(value) in (int, float) and math.isfinite(value) for value in stamps)
                    or not 0 < stamps[0] <= stamps[1] <= stamps[2] <= now):
                raise SearchDataError("resumed observation has invalid PIT chronology")
        if variable == "chronological_training_observations":
            # A numeric series is insufficient; these rows must actually carry
            # measured executable targets, decisions, fills and episode IDs.
            batch = {"partition": "training", "target_semantics": TARGET_SEMANTICS, "rows": rows}
            training_batch(batch, contracts, now=now)
            context["training_data"] = batch
            return payload
        descriptors = [row.get("observable_contract") for row in rows]
        descriptor = contracts.get(variable)
        if all(isinstance(item, Mapping) for item in descriptors):
            if any(item != descriptors[0] for item in descriptors):
                raise SearchDataError("resumed observable has inconsistent native contracts")
            if descriptor is not None and descriptor != descriptors[0]:
                raise SearchDataError("resumed contract conflicts with frozen source mapping")
            descriptor = contracts[variable] = copy.deepcopy(dict(descriptors[0]))
        if not isinstance(descriptor, Mapping):
            inputs[variable]["missing_contract"] = True
            return payload
        if (descriptor.get("point_in_time") is not True or any(not descriptor.get(field)
                for field in ("dtype", "unit", "source", "event_timestamp", "availability_timestamp", "snapshot_group", "snapshot_key"))):
            raise SearchDataError("resumed source contract lacks typed native PIT mapping")
        execution = context.get("execution_contract", {})
        for row in rows:
            if (row.get("variable") != variable or row.get("unit") != descriptor["unit"]
                    or row.get("source_id") != descriptor["source"]
                    or execution.get("chain") is not None and row.get("chain") != execution["chain"]):
                raise SearchDataError("resumed measurement does not match native observable contract")
        training = context.get("training_data", {}).get("rows", [])
        joined = 0
        for target in training:
            matches = [row for row in rows if row.get("row_id") == target.get("row_id")
                and row.get("episode_id") == target.get("episode_id")
                and row.get("row_id") is not None and row.get("episode_id") is not None
                and row["available_at"] <= target.get("decision_at", float("-inf"))]
            if len(matches) > 1: raise SearchDataError("ambiguous resumed training row identity")
            if matches:
                value = matches[0].get("value")
                dtype = descriptor["dtype"]
                if (dtype == "bool" and type(value) is not bool or dtype == "int" and type(value) is not int
                        or dtype == "float" and (type(value) not in (int, float) or not math.isfinite(value))):
                    raise SearchDataError("resumed observable does not match declared dtype")
                features = target.setdefault("features", {})
                if variable in features and features[variable] != value:
                    raise SearchDataError("resumed measurement conflicts with retained training feature")
                features[variable] = value
                joined += 1
        inputs[variable]["joined_training_rows"] = joined
        return payload

    @staticmethod
    def _request(variable: str, lab_id: str, family: str, evidence_refs, *, reason: str, kind="DATA_REQUEST", extra=None) -> dict:
        lab = MECHANISM_LABS.get(lab_id, MECHANISM_LABS["research_methods"])
        return {"kind": kind, "payload": {"variable": variable, "lab_id": lab_id,
            "mechanism": lab["title"], "source_classes": ["public native events", "verified official interfaces", "licensed PIT history"],
            "pit_requirements": "event_at <= available_at <= decision_at; no future reputation or post-event labels in features",
            "expected_consumer": "canonical exact-strategy compiler", "strategy_family": family,
            "acquisition_cost": None, "validation_plan": lab["experiment"], "falsifier": lab["falsifier"],
            "reason": reason, "evidence_refs": list(evidence_refs), "questions": lab["questions"], **dict(extra or {})}}

    def _compile(self, program: dict, context: dict, lab_id: str, family: str, evidence_refs, now: float) -> HypothesisSpec:
        lab, execution = MECHANISM_LABS[lab_id], context["execution_contract"]
        candidate = {key: copy.deepcopy(execution[key]) for key in EXECUTION_FIELDS}
        identity = "recipe-" + digest({"program": program, "family": family, "execution": execution,
                                       "observables": context["observable_contracts"], "evidence_refs": sorted(evidence_refs)})
        existing = self.framework.hypotheses.get(identity)
        if existing is not None:
            if not self.framework._exact_frozen_valid(existing): raise SearchDataError("existing search identity was changed or unfrozen")
            return existing
        parents = list(dict.fromkeys(list(context.get("parent_ids", [])) + program["search_parents"]))
        parent_hashes = {}
        for parent_id in parents:
            parent = self.framework.hypotheses.get(parent_id)
            if parent is None or not self.framework._exact_frozen_valid(parent):
                raise SearchDataError("every search descendant requires exact frozen parent ancestry")
            parent_hashes[parent_id] = parent.frozen_hash
        candidate.update(schema_version=1, candidate_id=identity, agent_identity="deterministic-lab:" + program["method"],
            created_at=now, parent_ids=parents, trial_family=family + ":" + candidate["chain"] + ":" + candidate["protocol"] + ":" + candidate["lifecycle_stage"],
            mechanism=lab["title"], actor_incentive=lab["actor_incentive"],
            data_inputs=[{"name": name, **copy.deepcopy(descriptor)} for name, descriptor in sorted(context["observable_contracts"].items())],
            feature_program=copy.deepcopy(program["feature_program"]), exact_entry=copy.deepcopy(program["entry_rule"]),
            expected_behavior="hypothesized positive executable log growth; training search is not edge evidence",
            failure_modes=[lab["falsifier"], "latency, crowding or native exit failure", "selection/multiplicity defect"],
            falsifier=lab["falsifier"], alternative_explanations=["common market regime", "source dependence", "search selection"],
            novelty_signature=digest({"lab": lab_id, "program": program["feature_program"], "entry": program["entry_rule"]}),
            source_lineage=list(evidence_refs), parent_frozen_hashes=parent_hashes,
            native_execution_contract=copy.deepcopy(context["native_execution_contract"]),
            training_procedure={"method": program["method"], "training_batch_hash": program["training_batch_hash"]})
        admission = validate_exact_candidate(candidate)
        if not admission.ok: raise SearchDataError("recipe not executable: " + "; ".join(admission.reasons))
        spec = HypothesisSpec(hypothesis_id=identity, mechanism=candidate["mechanism"], target=TARGET_SEMANTICS,
            features=sorted(candidate["feature_program"]), feature_hash=digest(candidate["feature_program"]),
            model_type="bounded_training_recipe", model_params=candidate["training_procedure"], training_window="TRAINING_ONLY",
            threshold=0, sizing_rule=candidate["size_rule"], exit_rule=candidate["exit_rule"],
            execution_policy=candidate["execution_assumptions"], fakeability={}, cost_model=candidate["fee_policy"],
            falsifier=candidate["falsifier"], kill_thesis="canonical gauntlet or forward failure",
            source_provenance=";".join(evidence_refs), trial_family=candidate["trial_family"], created_at=now,
            parent_ids=parents, agent_identity=candidate["agent_identity"], exact_candidate=candidate)
        self.framework.submit_hypothesis(spec)
        if not self.framework.freeze_hypothesis(identity): raise SearchDataError("canonical candidate freezing failed")
        return spec

    def _research(self, payload: dict, evidence_refs, now: float) -> dict:
        requested = payload.get("lab_id") or MECHANISM_ALIASES.get(payload.get("mechanism"))
        if "parent_failure" in payload and not payload.get("attribution"):
            return {"disposition": "DATA_REQUEST", "detail": {"status": "FROZEN_FAILURE_ANCESTRY_REQUIRED"},
                "children": [self._request("frozen_parent_failure_attribution", "research_methods", "unresolved_failure",
                    evidence_refs, reason="legacy forgone-growth findings are not a frozen attributed scientific parent",
                    extra={"required_fields": ["parent_id", "parent_frozen_hash", "attribution"]})]}
        if requested in (None, "all") and not payload.get("mechanism"):
            children = [{"kind": "MECHANISM_LAB", "payload": {"lab_id": lab_id,
                "strategy_family": next((family for family, owner in STRATEGY_FAMILIES.items() if owner == lab_id), "research_methods"),
                "questions": lab["questions"], "source_evidence_refs": list(evidence_refs)}} for lab_id, lab in MECHANISM_LABS.items()]
            return {"disposition": "DEEPENING_TASK", "detail": {"lab_count": len(children), "scope": "RESEARCH_ONLY"}, "children": children}
        if requested not in MECHANISM_LABS:
            child = self._request("economic_mechanism_definition", "research_methods", "unresolved_mechanism", evidence_refs,
                reason="unknown prose is not a measurable economic mechanism; identify actors, constraints and a discriminating observable")
            return {"disposition": "DATA_REQUEST", "detail": {"unresolved_mechanism": payload.get("mechanism")}, "children": [child]}
        lab_id, lab = requested, MECHANISM_LABS[requested]
        family = payload.get("strategy_family") or next((name for name, owner in STRATEGY_FAMILIES.items() if owner == lab_id), "research_methods")
        if family not in STRATEGY_FAMILIES and not (family == "research_methods" and lab_id == "research_methods"):
            raise SearchDataError("unknown strategy family; explicit family registration required")
        if family in STRATEGY_FAMILIES and STRATEGY_FAMILIES[family] != lab_id:
            raise SearchDataError("strategy-family recipe does not match mechanism lab")
        context = payload.get("context", payload)
        if not isinstance(context, Mapping): raise SearchDataError("recipe context must be an object")
        contracts = context.get("observable_contracts")
        missing = [name for name in lab["observables"] if not isinstance(contracts, Mapping) or name not in contracts]
        training_data = context.get("training_data", {})
        if not isinstance(training_data, Mapping): raise SearchDataError("training data must be an explicitly partitioned object")
        training_rows = training_data.get("rows", [])
        if isinstance(contracts, Mapping) and isinstance(training_rows, list) and training_rows:
            for name in contracts:
                if any(isinstance(row, Mapping) and name not in row.get("features", {}) for row in training_rows) and name not in missing:
                    missing.append(name)
        execution = context.get("execution_contract")
        def rule_fields(value):
            if isinstance(value, Mapping):
                result = [value["field"]] if isinstance(value.get("field"), str) else []
                return result + [field for item in value.values() for field in rule_fields(item)]
            if isinstance(value, list): return [field for item in value for field in rule_fields(item)]
            return []
        if isinstance(execution, Mapping):
            for field in rule_fields(execution.get("universe_rule")) + rule_fields(execution.get("hard_vetoes")):
                if (not isinstance(contracts, Mapping) or field not in contracts) and field not in missing:
                    missing.append(field)
        children = [self._request(name, lab_id, family, evidence_refs,
            reason="lab requires this economically named typed PIT observable", extra={"required_dtype": "float", "unit": "must be declared by native source"}) for name in missing]
        for variable, acquired in context.get("acquired_inputs", {}).items():
            if acquired.get("missing_contract"):
                children.append(self._request("typed_native_mapping:" + variable, lab_id, family, evidence_refs,
                    reason="observed scalar values do not establish native replay mapping or dtype",
                    extra={"input_variable": variable, "input_evidence_refs": acquired["evidence_refs"],
                        "required_fields": ["dtype", "unit", "source", "point_in_time", "event_timestamp", "availability_timestamp", "snapshot_group", "snapshot_key"]}))
        missing_execution = [key for key in EXECUTION_FIELDS if not isinstance(execution, Mapping) or execution.get(key) is None]
        native = context.get("native_execution_contract")
        native_fields = ("chain", "protocol", "event_schema_version", "decoder_fixture_ref", "buy_simulation_ref", "sell_simulation_ref")
        if (not isinstance(native, Mapping) or any(not native.get(key) for key in native_fields)
                or isinstance(execution, Mapping) and (native["chain"] != execution.get("chain") or native["protocol"] != execution.get("protocol"))):
            children.append(self._request("native_execution_contract", lab_id, family, evidence_refs,
                reason="verified native event/decoder/buy/sell semantics must be explicit before exact compilation",
                extra={"required_fields": list(native_fields)}))
        if missing_execution:
            children.append(self._request("exact_execution_policy", lab_id, family, evidence_refs,
                reason="fees, fills, sizing, horizon and exits may not be invented", extra={"required_fields": missing_execution}))
        if not context.get("training_data"):
            children.append(self._request("chronological_training_observations", lab_id, family, evidence_refs,
                reason="bounded training-only rows with measured executable targets required",
                extra={"target_semantics": TARGET_SEMANTICS, "minimum_rows": 8}))
        if children:
            return {"disposition": "DATA_REQUEST", "detail": {"lab_id": lab_id, "strategy_family": family,
                "questions": lab["questions"], "missing_observables": missing, "edge_evidence": "UNMEASURED"}, "children": children}
        # Validate all observable mappings before any outcome inspection.
        for name, descriptor in contracts.items():
            if not isinstance(descriptor, Mapping): raise SearchDataError("observable contract must be an object")
            if any(not descriptor.get(field) for field in ("dtype", "unit", "source", "event_timestamp", "availability_timestamp", "snapshot_group", "snapshot_key")):
                raise SearchDataError("observable contract lacks native mapping, dtype, unit or PIT timestamps")
            if any(marker in str(descriptor["snapshot_key"]).lower() for marker in ("future", "label", "outcome", "realized", "final_")):
                raise SearchDataError("native observable mapping points at a future/outcome field")
        budget = SearchBudget(**context.get("search_budget", {}))
        trial_outputs = {}
        def freeze(program):
            return self._compile(program, context, lab_id, family, evidence_refs, now).hypothesis_id
        def record(hid, result):
            key = digest({"hypothesis_id": hid, "batch": result["training_batch_hash"], "program": result["program_id"]})
            spec = self.framework.hypotheses[hid]
            if key not in self.state["trial_keys"]:
                score = result["training_score"]
                # A crash may occur after the canonical trial save and before
                # this component checkpoint. The canonical ledger is the
                # source of truth for retry multiplicity in that interval.
                marker = "training-search-key=" + key
                recorded = any(trial.hypothesis_id == hid and trial.stage == "TRAINING_SEARCH"
                    and trial.spec_hash == spec.frozen_hash and marker in trial.notes
                    for trial in self.framework.trial_results)
                if not recorded:
                    accepted = self.framework.record_trial_result(TrialResult(hypothesis_id=hid, stage="TRAINING_SEARCH",
                        samples=result["selected_count"], metrics={"elogw": score} if score is not None else {}, oos_metrics={},
                        portfolio_impact=0.0, passed=False, timestamp=max(time.time(), spec.frozen_at), spec_hash=spec.frozen_hash,
                        notes="training-only outcome inspection; no heldout verdict; " + marker))
                    if not accepted: raise SearchDataError("canonical training trial binding rejected")
                self.state["trial_keys"].append(key)
                self._checkpoint()
            trial_outputs[hid] = {"candidate": spec.exact_candidate, "training_trial": {"spec_hash": spec.frozen_hash,
                "stage": "TRAINING_SEARCH", "passed": False, "training_batch_hash": result["training_batch_hash"],
                "program_id": result["program_id"], "scope": "TRAINING_ONLY"}, "search_ancestry": result["search_parents"]}
        search = MachineSearch(budget=budget).run(context["training_data"], contracts, now=now,
            freeze=freeze, record=record, methods=context.get("search_methods", SEARCH_METHODS), archive=self.state["search_archive"])
        self.state["search_archive"] = search["archive"]
        children = [{"kind": "EXACT_CANDIDATE", "payload": value,
            "axes": {"chain": execution["chain"], "protocol": execution["protocol"], "lifecycle_stage": execution["lifecycle_stage"]}}
            for value in trial_outputs.values()]
        children.extend(self._request(request["variable"], lab_id, family, evidence_refs,
            kind=request["kind"], reason="machine search requires a measurable representation/acquisition",
            extra=request) for request in search["requests"])
        self.state["lab_calls"][lab_id] = self.state["lab_calls"].get(lab_id, 0) + 1
        return {"disposition": "EXACT_CANDIDATE" if trial_outputs else "REPRESENTATION_REQUEST",
            "detail": {"lab_id": lab_id, "strategy_family": family, "questions": lab["questions"],
                "search": {key: value for key, value in search.items() if key not in {"archive", "programs", "requests"}},
                "raw_outcome_inspecting_variants": len(search["programs"]), "canonical_gauntlet_required": True}, "children": children}

    def consume(self, kind: str, payload: Mapping, evidence_refs, now: float) -> dict:
        try:
            if not isinstance(payload, Mapping) or not evidence_refs or not all(isinstance(ref, str) and ref for ref in evidence_refs):
                raise SearchDataError("canonical payload/evidence references required")
            key = digest({"kind": kind, "payload": payload, "evidence_refs": sorted(evidence_refs)})
            if key in self.state["completed"]:
                self._checkpoint()
                return copy.deepcopy(self.state["completed"][key])
            # This exact annotation is emitted by the pre-existing forgone-
            # growth converter. Remove it before checkpointing/resuming; its
            # presence cannot confer authority or bypass frozen ancestry.
            if ("parent_failure" in payload and payload.get("authority") == "none_until_promoted"):
                payload = {name: value for name, value in payload.items() if name != "authority"}
            if not math.isfinite(now) or _contaminated(payload):
                raise SearchDataError("heldout/forward/authority contamination refused")
            payload = self._merge_observed(payload, now)
            if kind in {"FAILURE_DESCENDANT", "SURVIVOR_DESCENDANT"} or (
                    kind == "MECHANISM_HYPOTHESIS" and all(payload.get(field) for field in
                    ("parent_id", "parent_frozen_hash", "attribution"))):
                from src.research.descendant_generator import consume_descendant
                descendant_kind = "FAILURE_DESCENDANT" if kind == "MECHANISM_HYPOTHESIS" else kind
                result = consume_descendant(descendant_kind, payload, self.framework, evidence_refs=evidence_refs, now=now)
            elif kind in {"MECHANISM_HYPOTHESIS", "MECHANISM_LAB", "MACHINE_SEARCH", "DEEPENING_TASK"}:
                result = self._research(dict(payload), evidence_refs, now)
            else:
                raise SearchDataError("unsupported laboratory work kind")
            # Acquired inputs wake the exact original work item. No request
            # can silently become a terminal DATA_AVAILABLE receipt.
            for child in result.get("children", []):
                if child["kind"] in {"DATA_REQUEST", "REPRESENTATION_REQUEST"}:
                    child["payload"].update(partition="training", resume_kind=kind,
                        resume_payload=copy.deepcopy(payload))
                    execution = payload.get("context", payload).get("execution_contract", {})
                    if execution.get("chain"): child["payload"]["chain"] = execution["chain"]
            self.state["completed"][key] = copy.deepcopy(result)
            self._checkpoint()
            return result
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return {"disposition": "BLOCKED", "detail": {"reason": str(exc), "edge_evidence": "UNMEASURED"}, "children": []}

    def report(self) -> dict:
        return {"status": "RESEARCH_ONLY", "mechanism_labs": MECHANISM_LABS,
            "strategy_families": STRATEGY_FAMILIES, "mechanism_aliases": MECHANISM_ALIASES,
            "implemented_search_paths": list(SEARCH_METHODS), "completed_work_items": len(self.state["completed"]),
            "outcome_inspecting_trials": len(self.state["trial_keys"]), "lab_calls": self.state["lab_calls"],
            "measured_incremental_log_growth": None, "live_authority": "NONE"}
