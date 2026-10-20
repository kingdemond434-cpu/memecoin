"""Consumers for research validation work; never an execution or promotion path.

GAUNTLET payloads name candidate_id and optionally its exact spec_hash. The
configured CandidateReplayHandler produces the L0--L4 receipt, never a caller.
VALIDATION_ATTACK additionally declares corpus_hash and attacks (null, leakage,
delay, cost). FORWARD_SHADOW actions are register, cohort and outcome. Cohorts
declare event_at <= available_at <= decision_at <= observed_at. Outcomes declare
entry_at <= exit_at <= observed_at and reference two canonical ledger rows of
type research_forward_native_execution, phase entry/exit, origin
canonical_native_execution, bound to candidate_id/spec_hash/cohort_id. Each row
has observed_at, available_at, route, size_native, costs_native, filled, plus
execution.evidence_kind. Real rows require signature; paper rows require
quote_ref and simulation_ref. Native cashflow fields in execution are
entry_cost_native, exit_proceeds_native, fees_native, failed_cost_native,
native_to_usd. Declared realized_pnl_usd must equal their recomputed cashflow.

Paper scope observed_forward_shadow uses native_simulation and real_fill=False;
observed_forward uses observed_transaction and real_fill=True. Neither source
configuration nor historical prices constitute a forward execution receipt.
Only canonical framework checkpoints persist this consumer's state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sqlite3
from dataclasses import replace
from typing import Any, Mapping

from src.research.candidate_replay import (
    CandidateReplayHandler, ReplayBlocked, _ExactFeed, _digest, compile_candidate,
)
from src.research.forward_evidence import Outcome
from src.research.gauntlet_feed import CONTROL, EpisodeView
from src.research.trade_evidence import TradeEvidenceLedger, _canonical
from src.strategies.champion_challenger import TrialResult

CHECKPOINT = "research_validation_v1"
NATIVE_ROUTES = frozenset({"pump_curve", "pump_native", "pumpswap_native"})
STAGES = ("L0_SCHEMA_PIT_PROGRAM", "L1_FULL_FROZEN_FAMILY",
          "L2_NATIVE_COST_FILL_FEASIBILITY", "L3_CHRONOLOGICAL_CANONICAL_GAUNTLET",
          "L4_NATIVE_EXECUTION_REPLAY")


def _number(value: Any, *, positive: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


def _safe(value: Any) -> Any:
    """Receipts can describe absent authority, but checkpoints forbid its key."""
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items() if key != "authority"}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


class ResearchValidationService:
    def __init__(self, framework, gauntlet_handler=None, forward_evidence=None):
        self.framework = framework
        self.gauntlet_handler = gauntlet_handler
        self.forward_evidence = forward_evidence
        self.evidence_ledger = None
        self.registry = None
        self.adapters = {}
        self.state = framework.get_research_component(CHECKPOINT) or {
            "schema_version": 1, "operations": {}, "historical": {},
            "experiments": {}, "attacks": {}, "integrations": {}}

    def configure_evidence(self, ledger: TradeEvidenceLedger):
        if not isinstance(ledger, TradeEvidenceLedger):
            raise TypeError("canonical TradeEvidenceLedger required")
        self.evidence_ledger = ledger
        return self

    def configure_integrations(self, registry, adapters):
        self.registry, self.adapters = registry, dict(adapters)
        return self

    def read_evidence(self, reference):
        """Resolve a bounded authenticated row from the configured desk ledger."""
        ledger = self.evidence_ledger
        if ledger is None or not ledger.last_hash:
            raise ReplayBlocked("canonical execution evidence ledger with intact tail required")
        if not isinstance(reference, str):
            raise ReplayBlocked("canonical evidence reference required")
        parts = reference.split(":")
        if len(parts) != 3 or parts[0] != "evidence" or not parts[2].isdigit():
            raise ReplayBlocked("canonical evidence reference format required")
        with ledger.path.open("rb") as handle:
            handle.seek(int(parts[2]))
            raw = handle.readline(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ReplayBlocked("execution receipt exceeds bounded read")
        row = json.loads(raw)
        digest = row.pop("hash")
        if digest != parts[1] or hashlib.sha256(_canonical(row)).hexdigest() != digest:
            raise ReplayBlocked("execution receipt hash mismatch")
        return row

    def _save(self):
        self.framework.checkpoint_research_component(CHECKPOINT, _safe(self.state))

    @staticmethod
    def _result(disposition, detail, children=()):
        return {"disposition": disposition, "detail": detail, "children": list(children)}

    def _blocked(self, variable, reason, payload):
        change = "research_validation:" + variable
        request = {"missing_variable": variable, "reason": reason,
                   "mechanism": "version_bound_research_validation",
                   "source_classes": ["canonical_native_evidence"],
                   "point_in_time_requirement": "event and availability timestamps before decision",
                   "expected_consumer": "ResearchValidationService",
                   "candidate_id": payload.get("candidate_id"),
                   "acquisition_strategy": "supply named missing input then retry original work",
                   "acquisition_cost": None,
                   "validation": "canonical receipt hash, exact frozen binding and chronological checks",
                   "falsifier": "missing, stale, non-native or mismatched evidence remains blocked",
                   "enabling_change": change}
        return self._result("BLOCKED", {"status": "DATA_BLOCKED", "reason": reason,
                            "enabling_change": change},
                            [{"kind": "DATA_REQUEST", "payload": request,
                              "enabling_change": change}])

    def _candidate(self, payload):
        candidate = payload.get("candidate")
        hid = payload.get("candidate_id") or (candidate.get("candidate_id") if isinstance(candidate, Mapping) else None)
        spec = self.framework.hypotheses.get(hid)
        if not spec or not self.framework._exact_frozen_valid(spec):
            raise ReplayBlocked("exact registered frozen candidate unavailable")
        if payload.get("spec_hash", spec.frozen_hash) != spec.frozen_hash:
            raise ReplayBlocked("candidate frozen hash mismatch")
        if candidate is not None and candidate != spec.exact_candidate:
            raise ReplayBlocked("supplied candidate differs from exact registered frozen program")
        return spec

    def consume(self, kind, payload, evidence_refs=(), now=None):
        if not isinstance(payload, Mapping) or not _number(now, positive=True):
            return self._blocked("valid_contract", "object payload and finite positive now required", {})
        refs = list(evidence_refs)
        if not all(isinstance(ref, str) and ref for ref in refs):
            return self._blocked("evidence_refs", "original evidence references must be strings", payload)
        # Original references are part of operation identity. Time is deliberately
        # excluded: the same event cannot be counted again after a restart.
        try:
            self.framework._research_checkpoint({"payload": dict(payload)})
            key = _digest({"kind": kind, "payload": dict(payload), "refs": refs})
        except (TypeError, ValueError):
            return self._blocked("finite_payload", "payload must be finite JSON", payload)
        cached = self.state["operations"].get(key)
        if cached:
            return copy.deepcopy(cached)
        try:
            if kind == "GAUNTLET":
                result = self._gauntlet(payload, refs, now)
            elif kind == "VALIDATION_ATTACK":
                result = self._attack(payload, refs, now)
            elif kind == "FORWARD_SHADOW":
                result = self._forward(payload, refs, now)
            elif kind in {"CHAIN_INTEGRATION", "PROTOCOL_INTEGRATION", "INTEGRATION"}:
                result = self._integration(payload, refs, now)
            else:
                return self._blocked("supported_kind", "unsupported research validation work kind", payload)
        except (ReplayBlocked, ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
            result = self._blocked(str(kind).lower() + "_inputs", str(exc), payload)
        # Missing inputs must be retriable after a named enabling change.
        if result["disposition"] != "BLOCKED":
            self.state["operations"][key] = _safe(copy.deepcopy(result))
        self._save()
        return _safe(result)

    def _handler(self):
        handler = self.gauntlet_handler
        if not isinstance(handler, CandidateReplayHandler) or handler.framework is not self.framework:
            raise ReplayBlocked("configured canonical CandidateReplayHandler required")
        return handler

    def _trial(self, spec, result, *, passed, stage, now, receipt_hash):
        # No zero substitutes for unmeasured metrics. An unmeasured receipt stays
        # visible in checkpoints but does not become a metric-bearing trial.
        if not (_number(result.get("e_log_w")) and type(result.get("n")) is int):
            return False
        elogw = float(result["e_log_w"])
        metrics = {"elogw": elogw}
        if _number(result.get("e_log_w_lower")):
            metrics["elogw_lower"] = result["e_log_w_lower"]
        note = "canonical receipt " + receipt_hash
        prior = [trial for trial in self.framework.trial_results
                 if trial.hypothesis_id == spec.hypothesis_id and trial.spec_hash == spec.frozen_hash
                 and trial.stage == stage and trial.notes == note]
        if prior:
            return prior[0].evidence_status == "BOUND"
        return self.framework.record_trial_result(TrialResult(
            hypothesis_id=spec.hypothesis_id, stage=stage, samples=result["n"],
            metrics=metrics, oos_metrics=dict(metrics), portfolio_impact=elogw,
            passed=passed, timestamp=now, notes=note,
            spec_hash=spec.frozen_hash))

    def _gauntlet(self, payload, refs, now):
        spec = self._candidate(payload)
        if now < spec.frozen_at:
            raise ReplayBlocked("validation precedes candidate freeze")
        handler = self._handler()
        if not _number(handler.split.get("test_end") if isinstance(handler.split, Mapping) else None):
            raise ReplayBlocked("declared historical test cutoff unavailable")
        if handler.split["test_end"] > now:
            raise ReplayBlocked("historical validation cannot inspect future outcome timestamps")
        raw = handler(spec.exact_candidate)
        receipt = _safe(raw["receipt"])
        receipt_hash = _digest(receipt)
        target = spec.hypothesis_id + "@" + spec.frozen_hash
        status = raw["status"]
        recorded = False
        if set(STAGES).issubset(receipt.get("stages", [])):
            test = receipt.get("results", {}).get("test", {}).get(target, {})
            # Even rejected experiments are immutable outcome-inspecting trials.
            recorded = self._trial(spec, test, passed=status == "HISTORICAL_SURVIVOR",
                stage="CHRONOLOGICAL_OOS", now=now, receipt_hash=receipt_hash)
        record = {"receipt": receipt, "receipt_hash": receipt_hash, "status": status,
                  "recorded_trial": recorded, "evidence_refs": refs, "validated_at": now}
        self.state["historical"].setdefault(receipt_hash, record)
        if status == "DATA_BLOCKED":
            return self._blocked("canonical_replay_corpus", receipt.get("reason", "L0-L4 incomplete"), payload)
        if status != "HISTORICAL_SURVIVOR" or not recorded:
            return self._result("NEGATIVE_KNOWLEDGE", record)
        return self._result("DEEPENING_TASK", record, [{"kind": "FORWARD_SHADOW", "payload": {
            "action": "register", "candidate_id": spec.hypothesis_id,
            "spec_hash": spec.frozen_hash, "historical_receipt_hash": receipt_hash,
            "experiment_id": "forward:" + receipt_hash}}])

    def _attack(self, payload, refs, now):
        spec = self._candidate(payload)
        handler = self._handler()
        rows = handler._episodes()
        if payload.get("corpus_hash") != _digest(rows):
            raise ReplayBlocked("declared attack corpus_hash absent or mismatched")
        attacks = payload.get("attacks")
        if (not isinstance(attacks, list) or not attacks or len(attacks) > 4
                or len(set(attacks)) != len(attacks) or not set(attacks) <= {"null", "leakage", "delay", "cost"}):
            raise ReplayBlocked("declare bounded null/leakage/delay/cost attack targets")
        if now < spec.frozen_at:
            raise ReplayBlocked("attack precedes candidate freeze")
        # Validate the declared corpus through the same full-family chronological
        # adapter before any planted control inspects its outcomes.
        baseline = handler(spec.exact_candidate)
        if baseline["status"] == "DATA_BLOCKED":
            raise ReplayBlocked(baseline["receipt"].get("reason", "baseline replay incomplete"))
        if baseline["receipt"]["split"]["test_end"] > now:
            raise ReplayBlocked("planted controls cannot inspect future outcomes")
        family = handler._family(spec.exact_candidate)
        work = len(rows) * len(family) * handler.gauntlet.bootstrap * len(attacks) * (
            len(handler.gauntlet.latencies) + len(handler.gauntlet.cost_multipliers) + 8)
        if work > handler.max_bootstrap_draws:
            raise ReplayBlocked("planted experiments exceed declared bootstrap work budget")
        experiments = {}
        for attack in attacks:
            if attack == "leakage":
                planted = copy.deepcopy(spec.exact_candidate)
                planted["data_inputs"][0]["snapshot_key"] = "future_realized_outcome"
                rejected = False
                try:
                    compile_candidate(planted)
                except ReplayBlocked:
                    rejected = True
                program = compile_candidate(spec.exact_candidate)
                late = copy.deepcopy(rows[0])
                for snapshot in late["snapshots"].values():
                    snapshot["available_at"] = late["created_at"] + 1e9
                unavailable = program.mechanism.rule(EpisodeView(late)) is None
                experiments[attack] = {"future_mapping_rejected": rejected,
                    "late_availability_unknown": unavailable, "falsified": not (rejected and unavailable)}
                continue
            manipulated = copy.deepcopy(rows)
            if attack == "null":
                # Permute only declared mapped feature groups, preserving native
                # outcomes/quotes and timestamps. Constant predictors therefore
                # remain constant and honestly fail this null challenge.
                order = list(range(len(rows)))
                random.Random(721).shuffle(order)
                for row, index in zip(manipulated, order):
                    source = rows[index].get("snapshots", {})
                    for offset, snapshot in row.get("snapshots", {}).items():
                        for group, value in source.get(offset, {}).items():
                            if group.endswith("_features"):
                                snapshot[group] = copy.deepcopy(value)
            observations = []
            names = []
            target = spec.hypothesis_id + "@" + spec.frozen_hash
            for name, member in family:
                names.append(name)
                program = compile_candidate(member.exact_candidate, mechanism_name=name)
                if attack == "delay":
                    delay = payload.get("delay_seconds")
                    if not _number(delay, positive=True) or delay > 3600:
                        raise ReplayBlocked("bounded positive delay_seconds required")
                    program = replace(program, mechanism=replace(program.mechanism,
                        decision_offset_s=program.mechanism.decision_offset_s + delay))
                for row in manipulated:
                    execution = copy.deepcopy(handler._execution(row, program.candidate))
                    if attack == "cost":
                        multiplier = payload.get("cost_multiplier")
                        if not _number(multiplier, positive=True) or not 1 < multiplier <= 100:
                            raise ReplayBlocked("cost_multiplier must be greater than one and <=100")
                        for cell in execution["fills_by_latency"].values():
                            for key in ("round_trip_cost_fraction", "failed_cost_fraction"):
                                if not _number(cell.get(key)) or cell[key] < 0:
                                    raise ReplayBlocked("measured native costs required")
                                cell[key] *= multiplier
                    built = _ExactFeed(program, execution, handler.gauntlet.latencies).observations([row])
                    observations.extend(item for item in built if item.mechanism != CONTROL.name or name == target)
            judged = handler.gauntlet.run(observations, expected_mechanisms=names)
            results = {name: result.to_dict() for name, result in judged.items()}
            measured = results.get(target, {})
            if not measured or measured.get("verdict") == "DATA_BLOCKED":
                raise ReplayBlocked("planted " + attack + " experiment lacks executable evidence")
            experiment = {"results": results, "falsified": measured["verdict"] != "SURVIVOR"
                          if attack != "null" else measured["verdict"] == "SURVIVOR"}
            experiment["recorded_trial"] = self._trial(spec, measured, passed=False,
                stage="VALIDATION_ATTACK", now=now, receipt_hash=_digest(experiment))
            experiments[attack] = experiment
        receipt = {"candidate_id": spec.hypothesis_id, "spec_hash": spec.frozen_hash,
            "corpus_hash": payload["corpus_hash"], "scope": "PLANTED_CONTROLS_ONLY",
            "promotion": False, "experiments": experiments, "estimated_bootstrap_draws": work,
            "evidence_refs": refs, "observed_at": now}
        receipt_hash = _digest(receipt)
        self.state["attacks"].setdefault(receipt_hash, receipt)
        return self._result("NEGATIVE_KNOWLEDGE" if any(value["falsified"] for value in experiments.values())
                            else "DEEPENING_TASK", {"receipt_hash": receipt_hash, **receipt})

    def _forward(self, payload, refs, now):
        spec = self._candidate(payload)
        action, eid = payload.get("action"), payload.get("experiment_id")
        if not isinstance(eid, str) or not eid:
            raise ReplayBlocked("named forward experiment_id required")
        experiments = self.state["experiments"]
        if action == "register":
            record = self.state["historical"].get(payload.get("historical_receipt_hash"), {})
            receipt = record.get("receipt", {})
            if (record.get("status") != "HISTORICAL_SURVIVOR" or not record.get("recorded_trial")
                    or receipt.get("candidate_id") != spec.hypothesis_id
                    or receipt.get("spec_hash") != spec.frozen_hash
                    or not set(STAGES).issubset(receipt.get("stages", []))):
                raise ReplayBlocked("real canonical version-bound L0-L4 survivor receipt required")
            cutoff = max(spec.frozen_at, receipt["split"]["test_end"], record["validated_at"])
            if now < cutoff:
                raise ReplayBlocked("forward registration precedes freeze/historical validation")
            entry = {"candidate_id": spec.hypothesis_id, "spec_hash": spec.frozen_hash,
                     "registered_at": now, "historical_receipt_hash": record["receipt_hash"],
                     "cohorts": {}, "outcomes": {}}
            if eid in experiments:
                old = experiments[eid]
                if (old["candidate_id"], old["spec_hash"], old["historical_receipt_hash"]) != (
                        spec.hypothesis_id, spec.frozen_hash, record["receipt_hash"]):
                    raise ReplayBlocked("experiment_id already bound to another frozen experiment")
                entry = old
            else:
                experiments[eid] = entry
            return self._result("DEEPENING_TASK", {"status": "REGISTERED_FORWARD", "experiment_id": eid,
                                                    "registered_at": entry["registered_at"]})
        experiment = experiments.get(eid)
        if (not experiment or experiment["candidate_id"] != spec.hypothesis_id
                or experiment["spec_hash"] != spec.frozen_hash):
            raise ReplayBlocked("registered matching forward experiment required")
        if action == "cohort":
            cid = payload.get("cohort_id")
            times = [payload.get(key) for key in ("event_at", "available_at", "decision_at", "observed_at")]
            if (not isinstance(cid, str) or not cid or not all(_number(value, positive=True) for value in times)
                    or not experiment["registered_at"] <= times[0] <= times[1] <= times[2] <= times[3] <= now):
                raise ReplayBlocked("untouched forward chronological cohort timestamps required")
            cohorts = experiment["cohorts"]
            if cid in cohorts:
                if cohorts[cid]["payload"] != dict(payload) or cohorts[cid]["refs"] != refs:
                    raise ReplayBlocked("cohort_id reused with conflicting evidence")
            else:
                if cohorts and times[2] <= max(item["payload"]["decision_at"] for item in cohorts.values()):
                    raise ReplayBlocked("forward cohorts must have unique chronological decisions")
                if not refs:
                    raise ReplayBlocked("forward cohort original evidence references required")
                cohort_key = _digest({"candidate_id": spec.hypothesis_id, "spec_hash": spec.frozen_hash,
                                      "cohort_id": cid})
                owner = self.state.setdefault("cohort_owners", {}).get(cohort_key)
                if owner is not None and owner != eid:
                    raise ReplayBlocked("same forward cohort cannot supply multiple experiment sample counts")
                self.state["cohort_owners"][cohort_key] = eid
                cohorts[cid] = {"payload": dict(payload), "refs": refs}
            return self._result("DEEPENING_TASK", {"status": "FORWARD_COHORT_RECORDED", "cohort_id": cid})
        if action != "outcome":
            raise ReplayBlocked("FORWARD_SHADOW action must be register, cohort or outcome")
        return self._outcome(spec, experiment, payload, refs, now)

    def _execution_rows(self, spec, cohort_id, execution, refs, paper, now):
        result = {}
        for phase in ("entry", "exit"):
            reference = execution.get(phase + "_evidence_ref")
            if reference not in refs or not isinstance(reference, str):
                raise ReplayBlocked("original entry/exit canonical evidence refs required")
            row = self.read_evidence(reference)
            body = row.get("payload", {})
            if (row.get("event_type") != "research_forward_native_execution"
                    or not _number(row.get("timestamp"), positive=True) or row["timestamp"] > now
                    or body.get("origin") != "canonical_native_execution" or body.get("phase") != phase
                    or body.get("candidate_id") != spec.hypothesis_id or body.get("spec_hash") != spec.frozen_hash
                    or body.get("cohort_id") != cohort_id or body.get("route") not in NATIVE_ROUTES
                    or body.get("filled") is not True or not _number(body.get("size_native"), positive=True)
                    or not _number(body.get("costs_native")) or body["costs_native"] < 0
                    or not _number(body.get("observed_at"), positive=True)
                    or not _number(body.get("available_at"), positive=True)
                    or not body["observed_at"] <= body["available_at"] <= now):
                raise ReplayBlocked("native execution receipt binding/route/fill/time/cost mismatch")
            kind = "native_simulation" if paper else "observed_transaction"
            if body.get("evidence_kind") != kind:
                raise ReplayBlocked("native paper and observed-transaction evidence cannot be relabeled")
            if paper:
                if not all(isinstance(body.get(key), str) and body[key] for key in ("quote_ref", "simulation_ref")):
                    raise ReplayBlocked("canonical native quote and simulation receipts required")
                quote_row = self.read_evidence(body["quote_ref"])
                quote = quote_row.get("payload", {})
                quoted = quote.get("quote", {})
                simulation_row = self.read_evidence(body["simulation_ref"])
                simulation = simulation_row.get("payload", {})
                if (quote_row.get("event_type") != "research_forward_native_quote"
                        or quote.get("phase") != phase or quote.get("route") != body["route"]
                        or quote.get("observed_at") != body["observed_at"]
                        or quoted.get("data_status") != "OK"
                        or not _number(quoted.get("input_amount"), positive=True)
                        or not _number(quoted.get("output_amount"), positive=True)
                        or not _number(quoted.get("fee_amount")) or quoted["fee_amount"] < 0
                        or simulation_row.get("event_type") != "research_native_simulation_cost"
                        or simulation.get("origin") != "canonical_native_simulation"
                        or simulation.get("phase") != phase
                        or simulation.get("route") != body["route"]
                        or simulation.get("input_amount") != quoted.get("input_amount")
                        or not _number(simulation.get("observed_at"), positive=True)
                        or not simulation["observed_at"] <= body["observed_at"]
                        or simulation.get("simulation", {}).get("err", "UNMEASURED") is not None
                        or not _number(simulation.get("simulation", {}).get("unitsConsumed"), positive=True)):
                    raise ReplayBlocked("paper fill lacks actual native quote/successful simulation receipt observability")
                principal = ((quoted["input_amount"] - quoted["fee_amount"]) if phase == "entry"
                             else (quoted["output_amount"] + quoted["fee_amount"])) / 1_000_000_000
                network, failed = simulation.get("network_fee_native"), simulation.get("failed_cost_native")
                if (not _number(network) or network < 0 or not _number(failed) or failed < 0
                        or not math.isclose(principal, body["size_native"], rel_tol=1e-9, abs_tol=1e-12)
                        or not math.isclose(quoted["fee_amount"] / 1_000_000_000 + network + failed,
                                            body["costs_native"], rel_tol=1e-9, abs_tol=1e-12)):
                    raise ReplayBlocked("paper size/cashflow/fees differ from actual native quote and simulation")
            elif not isinstance(body.get("signature"), str) or not body["signature"]:
                raise ReplayBlocked("observed native transaction signature required")
            result[phase] = body
        return result

    def _outcome(self, spec, experiment, payload, refs, now):
        cohort = experiment["cohorts"].get(payload.get("cohort_id"))
        if not cohort or payload.get("partition") != "forward":
            raise ReplayBlocked("registered untouched forward cohort required; no historical relabel")
        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ReplayBlocked("unique forward event_id required")
        identity = _digest({"payload": dict(payload), "refs": refs})
        prior = experiment["outcomes"].get(event_id)
        if prior:
            if prior["input_hash"] != identity:
                raise ReplayBlocked("event_id reused with conflicting outcome")
            return self._result("DEEPENING_TASK", prior["receipt"])
        if any(item["receipt"]["cohort_id"] == payload["cohort_id"] for item in experiment["outcomes"].values()):
            raise ReplayBlocked("forward cohort already has a resolved outcome; duplicate sample refused")
        scope = payload.get("scope")
        if scope not in {"observed_forward", "observed_forward_shadow"}:
            raise ReplayBlocked("observed native forward scope required")
        paper = scope == "observed_forward_shadow"
        execution = payload.get("execution", {})
        if (execution.get("evidence_kind") != ("native_simulation" if paper else "observed_transaction")
                or execution.get("native") is not True or execution.get("filled") is not True
                or execution.get("route") not in NATIVE_ROUTES):
            raise ReplayBlocked("native route and observed fill evidence required")
        times = [payload.get(key) for key in ("entry_at", "exit_at", "observed_at")]
        if (not all(_number(value, positive=True) for value in times)
                or not cohort["payload"]["decision_at"] <= times[0] <= times[1] <= times[2] <= now):
            raise ReplayBlocked("forward outcome cannot precede registered cohort or use future prices")
        rows = self._execution_rows(spec, payload["cohort_id"], execution, refs, paper, now)
        execution = copy.deepcopy(execution)
        if not paper:
            execution["entry_signature"] = rows["entry"]["signature"]
            execution["exit_signature"] = rows["exit"]["signature"]
        if (rows["entry"]["observed_at"] != times[0] or rows["exit"]["observed_at"] != times[1]
                or any(row["route"] != execution["route"] for row in rows.values())):
            raise ReplayBlocked("outcome times/route differ from actual native receipts")
        for field in ("entry_cost_native", "exit_proceeds_native", "fees_native", "failed_cost_native", "native_to_usd"):
            if not _number(execution.get(field)) or execution[field] < 0:
                raise ReplayBlocked("complete measured native cashflows and fees required")
        if execution["native_to_usd"] <= 0 or execution["entry_cost_native"] <= 0:
            raise ReplayBlocked("positive entry cost and contemporaneous native conversion required")
        if (not math.isclose(rows["entry"]["size_native"], execution["entry_cost_native"], rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(rows["exit"]["size_native"], execution["exit_proceeds_native"], rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(sum(row["costs_native"] for row in rows.values()),
                    execution["fees_native"] + execution["failed_cost_native"], rel_tol=1e-9, abs_tol=1e-12)):
            raise ReplayBlocked("cashflows/fees mismatch canonical native execution receipts")
        pnl = (execution["exit_proceeds_native"] - execution["entry_cost_native"]
               - execution["fees_native"] - execution["failed_cost_native"]) * execution["native_to_usd"]
        equity = payload.get("equity_at_decision_usd")
        if (payload.get("all_costs_included") is not True or not _number(payload.get("realized_pnl_usd"))
                or not math.isclose(pnl, payload["realized_pnl_usd"], rel_tol=1e-9, abs_tol=1e-9)
                or not _number(equity, positive=True) or pnl <= -equity):
            raise ReplayBlocked("recomputed after-cost cashflow and positive equity required")
        if payload.get("record_in_forward_evidence") is True and self.forward_evidence is None:
            raise ReplayBlocked("canonical ForwardEvidence consumer unavailable")
        # Experiment observations survive before the canonical evaluation loop
        # creates a shadow bucket. Reconciliation never creates that bucket.
        receipt = {"status": "FORWARD_OUTCOME_RECORDED", "event_id": event_id,
            "candidate_id": spec.hypothesis_id, "spec_hash": spec.frozen_hash,
            "cohort_id": payload["cohort_id"], "scope": scope, "paper": paper,
            "real_fill": not paper, "pnl": pnl, "elogw": math.log1p(pnl / equity),
            "entry_at": times[0], "exit_at": times[1], "observed_at": times[2],
            "evidence_refs": refs, "framework_recorded": False,
            "execution": copy.deepcopy(execution)}
        # A single owning callback may delegate the global ledger. The normal
        # runtime path already records it, so this is deliberately opt-in.
        if payload.get("record_in_forward_evidence") is True:
            # Save a before/after state intent first. On an interrupted retry the
            # actual ledger must equal one of these snapshots, otherwise an
            # explicit conflict blocks instead of silently doubling the outcome.
            pending_key = _digest({"experiment": payload["experiment_id"], "event": event_id})
            pending = self.state.setdefault("forward_ledger_intents", {}).get(pending_key)
            outcome = Outcome(token=str(payload.get("token", payload["cohort_id"])),
                entered=True, regime=str(payload.get("regime", "unknown")), realized_pnl_usd=pnl,
                equity_at_decision_usd=equity, real_fill=not paper,
                execution_attempted=True, execution_succeeded=True)
            if pending is None:
                before = self.forward_evidence.state()
                projected = copy.deepcopy(self.forward_evidence)
                projected.record(outcome)
                pending = {"input_hash": identity, "before_hash": _digest(before),
                           "after_hash": _digest(projected.state())}
                self.state["forward_ledger_intents"][pending_key] = pending
                self._save()
            if pending["input_hash"] != identity:
                raise ReplayBlocked("global forward ledger event intent conflict")
            current = _digest(self.forward_evidence.state())
            if current == pending["before_hash"]:
                self.forward_evidence.record(outcome)
                if self.forward_evidence.path and not self.forward_evidence.save():
                    raise ReplayBlocked("canonical ForwardEvidence persistence failed")
            elif current != pending["after_hash"]:
                raise ReplayBlocked("global forward ledger changed during pending event; reconciliation required")
        experiment["outcomes"][event_id] = {"input_hash": identity, "receipt": receipt}
        self._save()
        self.reconcile_forward()
        return self._result("DEEPENING_TASK", receipt)

    def reconcile_forward(self):
        """Flush checkpointed observations into existing canonical forward buckets.

        IDs plus frozen bindings make this at most once across a restart. Paper
        may enter shadow only; it never increments the canary position counter.
        No bucket, stage, champion or live permission is created here.
        """
        transferred = 0
        for experiment in self.state["experiments"].values():
            spec = self.framework.hypotheses.get(experiment["candidate_id"])
            if (not spec or not self.framework._exact_frozen_valid(spec)
                    or spec.frozen_hash != experiment["spec_hash"]):
                continue
            for observation in experiment["outcomes"].values():
                receipt = observation["receipt"]
                if receipt.get("framework_recorded"):
                    continue
                bucket = self.framework.shadow_models.get(spec.hypothesis_id)
                if not receipt["paper"] and bucket is None:
                    bucket = self.framework.canary_models.get(spec.hypothesis_id)
                if bucket is None:
                    continue
                existing = [item for item in bucket["forward_results"]
                            if item.get("event_id") == receipt["event_id"]]
                if existing and any(item.get("spec_hash") != spec.frozen_hash
                                    or item.get("evidence_refs") != receipt["evidence_refs"] for item in existing):
                    continue
                if existing or self.framework.record_forward_result(spec.hypothesis_id, copy.deepcopy(receipt)):
                    receipt["framework_recorded"] = True
                    transferred += 1
        if transferred:
            self._save()
        return {"transferred": transferred}

    def _integration(self, payload, refs, now):
        from src.chains.evm_observation import EVMObservationAdapter
        chain, protocol = payload.get("chain"), payload.get("protocol")
        adapter = self.adapters.get(chain)
        if (not isinstance(adapter, EVMObservationAdapter) or self.registry is None
                or self.registry.get_rpc(chain) is not adapter.rpc):
            raise ReplayBlocked("actual registry-bound native observation adapter unavailable; config is not evidence")
        capabilities = payload.get("capabilities")
        supported = {"observation", "historical_backfill", "finality", "native_factory_decode"}
        if not isinstance(capabilities, list) or not capabilities or not set(capabilities) <= supported:
            raise ReplayBlocked("requested native execution/sizing/latency capability has no verified adapter")
        factories = [factory for factory in adapter.factories.values()
                     if not protocol or factory.protocol_id == protocol]
        if not factories or ("native_factory_decode" in capabilities
                and not all(item.decoder != "UNKNOWN" and item.abi_ref and item.source_ref for item in factories)):
            raise ReplayBlocked("actual protocol decoder/source/ABI capability unavailable")
        report = adapter.report()
        if (not refs or not report.get("observed_events") or report.get("last_height") is None
                or report.get("execution_eligible") is not False):
            raise ReplayBlocked("native adapter has no observed finalized event evidence yet")
        # Adapter counters authenticate progress, not which protocol decoded.
        # Resolve its actual canonical acknowledgements rather than granting
        # decoder capability from configured factory names alone.
        with adapter._db() as db:
            observed = db.execute("SELECT evidence_ref FROM evm_observer_refs WHERE adapter_id=? "
                "AND canonical=1 ORDER BY height DESC LIMIT 64", (adapter.adapter_id,)).fetchall()
        matched = []
        for row in observed:
            reference = row["evidence_ref"]
            try:
                event = self.read_evidence(reference)["payload"].get("metadata", {})
            except (ReplayBlocked, OSError, ValueError, KeyError):
                continue
            if (event.get("chain") != chain or (protocol and event.get("protocol") != protocol)
                    or event.get("finality") not in {"SAFE", "FINALIZED"}
                    or event.get("observation_only") is not True or event.get("execution_eligible") is not False):
                continue
            if ("native_factory_decode" in capabilities and
                    event.get("native", {}).get("decoder_status") != "DECODED_OFFICIAL_ABI"):
                continue
            matched.append(reference)
        if not matched:
            raise ReplayBlocked("actual canonical finalized protocol event/decoder receipts unavailable")
        detail = {"status": "OBSERVATION_CAPABILITY_VERIFIED", "chain": chain,
                  "protocol": protocol, "capabilities": capabilities, "adapter": report,
                  "execution_eligible": False, "evidence_refs": list(dict.fromkeys(refs + matched)), "observed_at": now}
        self.state["integrations"][_digest(detail)] = detail
        return self._result("DEEPENING_TASK", detail)

    def report(self):
        experiments = list(self.state["experiments"].values())
        outcomes = [item["receipt"] for experiment in experiments for item in experiment["outcomes"].values()]
        return {"schema_version": 1, "historical_receipts": len(self.state["historical"]),
                "planted_attack_receipts": len(self.state["attacks"]),
                "forward_experiments": len(experiments),
                "forward_cohorts": sum(len(item["cohorts"]) for item in experiments),
                "paper_outcomes": sum(item["paper"] for item in outcomes),
                "real_fill_outcomes": sum(item["real_fill"] for item in outcomes),
                "integration_receipts": len(self.state["integrations"]),
                "production_performance": "UNMEASURED"}
