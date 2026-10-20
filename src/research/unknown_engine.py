"""Bounded conversion work driven by the existing public research loop.

Raw objects live in the desk's existing hash-chained evidence ledger. The
frontier stores references and leases only. This service does not schedule
itself, evaluate alpha, promote models, allocate capital or sign anything.
"""
from __future__ import annotations

from collections import deque
import hashlib
import inspect
import json
import math
import time
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit

from src.research.frontier import AuctionInputs, FrontierIndex
from src.research.research_contracts import validate_exact_candidate
from src.research.trade_evidence import TradeEvidenceLedger
from src.strategies.champion_challenger import HypothesisSpec


LEGAL_RIGHTS = {"PUBLIC", "AUTHORIZED", "LICENSED"}
REQUEST_FIELDS = ("variable", "mechanism", "source_classes", "pit_requirements",
                  "expected_consumer", "strategy_family", "acquisition_cost",
                  "validation_plan", "falsifier")
FORBIDDEN_FIELDS = {"private_key", "password", "api_key", "authorization",
                    "sign_transaction", "allow_live_trading", "disable_veto"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).lower() in FORBIDDEN_FIELDS or _contains_forbidden(item)
                   for key, item in value.items())
    return isinstance(value, (list, tuple)) and any(_contains_forbidden(item) for item in value)


class UnknownOpportunityEngine:
    def __init__(self, framework: Any, frontier: FrontierIndex,
                 evidence: TradeEvidenceLedger, *,
                 gauntlet_handler: Optional[Callable] = None,
                 gauntlet_costs: Optional[Mapping[str, float]] = None,
                 gauntlet_work_bound: Optional[Mapping[str, int]] = None,
                 research_pass_resource_units: Optional[float] = None,
                 proposal_runner: Any = None,
                 worker: str = "canonical-research-conversion"):
        self.framework, self.frontier, self.evidence = framework, frontier, evidence
        self.gauntlet_handler, self.worker = gauntlet_handler, worker
        self.gauntlet_work_bound = dict(gauntlet_work_bound or {})
        self.gauntlet_auction = AuctionInputs()
        self.research_pass_resource_units = research_pass_resource_units
        if research_pass_resource_units is not None and (
                type(research_pass_resource_units) not in (int, float)
                or not math.isfinite(research_pass_resource_units) or research_pass_resource_units <= 0):
            raise ValueError("positive finite research-pass resource envelope required")
        if gauntlet_costs is not None or gauntlet_work_bound is not None:
            self.gauntlet_auction = AuctionInputs(costs=dict(gauntlet_costs or {}))
            cost = self.gauntlet_auction.total_cost
            bound_keys = {"max_episodes", "max_family_variants", "max_bootstrap_draws"}
            if (gauntlet_handler is None or cost is None or cost <= 0
                    or set(self.gauntlet_work_bound) != bound_keys
                    or any(type(value) is not int or value <= 0 for value in self.gauntlet_work_bound.values())):
                raise ValueError("canonical handler needs complete explicit costs and bounded replay work")
            for key, bound in self.gauntlet_work_bound.items():
                actual = getattr(gauntlet_handler, key, None)
                if actual is not None and actual > bound:
                    raise ValueError("canonical replay handler exceeds its declared work bound")
        self.proposal_runner = proposal_runner
        self.training_facts = deque(maxlen=16)
        self.last_proposal_round = {"status": "NOT_RUN", "authority": "none"}
        self.last_pass = {"status": "NOT_RUN", "authority": "none"}
        self.workflows = None

    def configure_workflows(self, workflows: Any) -> None:
        self.workflows = workflows

    def enqueue_work(self, kind: str, payload: dict, *, axes=None,
                     evidence_refs=(), now: Optional[float] = None) -> str:
        """Idempotently retain work before admitting it to the reference queue."""
        now = time.time() if now is None else float(now)
        if _contains_forbidden(payload):
            raise ValueError("credentials or authority instructions cannot enter research work")
        axes = dict(axes or {})
        if kind == "EXACT_CANDIDATE":
            created = payload["candidate"]["created_at"]
            return self.ingest({"source_id": "canonical-research-compiler", "rights": "PUBLIC",
                "event_at": created, "available_at": created, "observed_at": created,
                "exact_candidate": payload["candidate"], "source_lineage": payload["candidate"]["source_lineage"]}, now=now)
        identity = "work-" + _digest([kind, payload, axes])
        if self.frontier.job(identity) is not None:
            return identity
        ref = self._record("research_typed_work", {"work_kind": kind, "point_payload": payload}, now)
        auction = self.workflows.auction(kind) if self.workflows and kind in self.workflows.kinds else AuctionInputs()
        return self.frontier.enqueue(kind, ref, axes=axes, job_id=identity,
            evidence_refs=[ref, *evidence_refs], auction=auction,
            exploration_tags=["unknown_mechanisms"], now=now)

    def notify_evidence(self, changes, refs, now: float) -> None:
        """Only the exact named new evidence can wake incomplete work."""
        state = self.framework.get_research_component("research_evidence_wakes") or {"pending": {}}
        for change in sorted(set(changes)):
            if not isinstance(change, str) or not change:
                raise ValueError("evidence changes must be named")
            state["pending"][change] = {"refs": list(dict.fromkeys(refs)), "cursor": ""}
        self.framework.checkpoint_research_component("research_evidence_wakes", state)
        self._drain_evidence_wakes(now)

    def _drain_evidence_wakes(self, now: float) -> None:
        state = self.framework.get_research_component("research_evidence_wakes") or {"pending": {}}
        for change, work in list(state["pending"].items())[:8]:
            jobs = self.frontier.pending_work(states=("HIBERNATED",), limit=33, enabling_change=change,
                                             after_job_id=work["cursor"])
            for job in jobs[:32]:
                if self.frontier.coordinate(job["axes"])["state"] == "HIBERNATED_LOW_VOI":
                    continue
                self.frontier.wake_job(job["job_id"], enabling_change=change, evidence_refs=work["refs"], now=now)
            if len(jobs) < 33:
                del state["pending"][change]
            else:
                work["cursor"] = jobs[31]["job_id"]
        if state["pending"] or self.framework.get_research_component("research_evidence_wakes"):
            self.framework.checkpoint_research_component("research_evidence_wakes", state)

    def wake_gauntlet_job(self, job_id: str, *, now: Optional[float] = None) -> bool:
        """A named enabling change costs existing work without resetting it."""
        now = time.time() if now is None else float(now)
        if self.gauntlet_handler is None or self.gauntlet_auction.total_cost is None:
            return False
        job = self.frontier.job(job_id)
        if job is None or job["kind"] != "GAUNTLET" or job["state"] not in {"READY", "RETRY", "HIBERNATED"}:
            return False
        change = "canonical-replay-cost-and-work-envelope-configured"
        if job["total_cost"] is not None:
            return False
        coordinate = self.frontier.coordinate(job["axes"])
        if (coordinate["state"] == "HIBERNATED_LOW_VOI"
                and coordinate.get("enabling_change") != change):
            return False
        ref = self._record("unknown_gauntlet_budget_declared", {
            "job_id": job_id, "resource_costs": self.gauntlet_auction.to_dict(),
            "work_bound": self.gauntlet_work_bound, "monetary_cost": None}, now)
        if job["state"] != "HIBERNATED":
            self.frontier.hibernate_job(job_id, enabling_change=change,
                reason="previously unknown replay cost now has an explicit resource envelope", now=now)
        elif job.get("enabling_change") != change:
            return False
        if (coordinate["state"] == "HIBERNATED_LOW_VOI"
                and coordinate.get("enabling_change") == change):
            self.frontier.wake_coordinate(job["axes"], enabling_change=change,
                                           evidence_refs=[ref], now=now)
        self.frontier.wake_job(job_id, enabling_change=change, evidence_refs=[ref],
                               auction=self.gauntlet_auction, now=now)
        return True

    def _queue_gauntlet(self, detail: dict, axes: dict, refs: list[str], now: float) -> str:
        candidate = detail["candidate"]
        niche = detail["novelty"]
        checkpoints = self.framework.get_research_component("gauntlet_work") or {"jobs": {}}
        jobs = checkpoints.setdefault("jobs", {})
        family_key = niche["validation_job_family"]
        previous = jobs.get(family_key)
        if previous:
            job = self.frontier.job(previous["job_id"])
            target = self.framework.hypotheses.get(previous["candidate_id"])
            if (job and job["state"] in {"READY", "RETRY"} and target
                    and target.frozen_hash == previous["spec_hash"]
                    and self.framework._exact_frozen_valid(target)):
                self.wake_gauntlet_job(previous["job_id"], now=now)
                self._record("unknown_gauntlet_family_work_shared", {
                    "candidate_id": candidate["candidate_id"], "job_id": previous["job_id"],
                    "niche": niche, "raw_refs": refs}, now)
                return previous["job_id"]
        detail["gauntlet_work_bound"] = self.gauntlet_work_bound or None
        candidate_ref = self._record("unknown_frozen_candidate", detail, now)
        identity = self.frontier.enqueue("GAUNTLET", candidate_ref, axes=axes,
            evidence_refs=refs + [candidate_ref], auction=self.gauntlet_auction,
            exploration_tags=["strange_falsifiable"], now=now)
        jobs[family_key] = {"job_id": identity, "candidate_id": candidate["candidate_id"],
                            "spec_hash": self.framework.hypotheses[candidate["candidate_id"]].frozen_hash}
        self.framework.checkpoint_research_component("gauntlet_work", checkpoints)
        return identity

    def _wake_checkpointed_gauntlet_work(self, limit: int, now: float) -> None:
        if self.gauntlet_handler is None or self.gauntlet_auction.total_cost is None:
            return
        checkpoint = self.framework.get_research_component("gauntlet_work")
        if not checkpoint:
            return
        pointers = list(checkpoint.get("jobs", {}).values())
        if not pointers:
            return
        start = int(checkpoint.get("budget_cursor", 0)) % len(pointers)
        count = min(int(limit), 16, len(pointers))
        for index in range(count):
            self.wake_gauntlet_job(pointers[(start + index) % len(pointers)]["job_id"], now=now)
        checkpoint["budget_cursor"] = (start + count) % len(pointers)
        self.framework.checkpoint_research_component("gauntlet_work", checkpoint)

    def _record(self, event: str, payload: dict, now: float) -> str:
        reference = self.evidence.record_reference(event, payload, timestamp=now)
        if reference is None:
            raise RuntimeError("canonical evidence write failed; research not acknowledged")
        return reference

    def _load_refs(self, refs: set[str]) -> dict[str, dict]:
        found = {}
        if not self.evidence.path.exists():
            return found
        with self.evidence.path.open("rb") as handle:
            for ref in refs:
                prefix, digest, offset = ref.split(":")
                if prefix != "evidence" or int(offset) < 0:
                    raise ValueError("invalid canonical evidence reference")
                handle.seek(int(offset))
                line = handle.readline()
                if not line:
                    continue
                row = json.loads(line)
                body = dict(row)
                body.pop("hash")
                raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                 default=str).encode()
                if hashlib.sha256(raw).hexdigest() != digest or row["hash"] != digest:
                    raise ValueError("referenced research evidence was modified")
                found[ref] = row["payload"]
        return found

    @staticmethod
    def _triage_budget() -> AuctionInputs:
        # A declared envelope for a bounded local conversion, not a measured
        # monetary cost or an invented expected return. No provider call here.
        return AuctionInputs(costs={"inference": 0, "data": 0, "engineering": 0,
                                    "conversion": 1, "gauntlet": 0})

    def ingest(self, raw: Mapping[str, Any], *, now: Optional[float] = None) -> str:
        now = time.time() if now is None else float(now)
        raw = dict(raw)
        if _contains_forbidden(raw):
            raise ValueError("credentials or authority instructions cannot enter research evidence")
        identity = _digest(raw)
        job_id = "raw-" + identity
        existing = self.frontier.job(job_id)
        if existing is not None:
            self._admit_training_fact(raw, existing["payload_ref"])
            return job_id
        ref = self._record("unknown_raw_evidence", {"raw_id": identity, "raw": raw}, now)
        self._admit_training_fact(raw, ref)
        axes = {key: raw[key] for key in ("chain", "chain_family", "protocol",
                "lifecycle_stage", "region_language") if raw.get(key) is not None}
        axes["data_source"] = str(raw.get("source_id") or "UNDECLARED")
        kind = str(raw.get("discovery_kind") or "source")
        discovery_id = str(raw.get("identity") or raw.get("source_id") or identity)
        # Unknown discovery kinds stay ordinary evidence, never a guessed
        # chain or native protocol decoder.
        if (not self._pit_errors(raw, now)
                and kind in {"chain", "protocol", "source", "mechanism", "actor"}):
            self.frontier.observe_discovery(kind, discovery_id, evidence_refs=[ref],
                axes=axes, family=str(raw.get("chain_family") or "UNKNOWN"),
                version=str(raw.get("protocol_version") or "1"),
                metrics=raw.get("metrics"), contract=raw.get("protocol_contract"), now=now)
        tags = {"chain": "unknown_chains", "protocol": "unknown_protocols",
                "actor": "unknown_actors", "mechanism": "unknown_mechanisms"}
        return self.frontier.enqueue("RAW_EVIDENCE", ref, axes=axes, job_id=job_id,
            evidence_refs=[ref], auction=self._triage_budget(),
            exploration_tags=[tags.get(kind, "new_source_classes")], now=now)

    def _admit_training_fact(self, raw: dict, ref: str) -> None:
        if (raw.get("research_partition") != "training" or raw.get("rights") != "PUBLIC"
                or self._pit_errors(raw, time.time())):
            return
        if ref in {fact["evidence_id"] for fact in self.training_facts}:
            return
        # Deliberately exclude token outcomes, candidate evaluations, model
        # opinions and arbitrary source fields from every independent input.
        self.training_facts.append({"evidence_id": ref,
            "source": str(raw.get("source_url") or raw.get("source_id")),
            "observed_at": raw["event_at"], "available_at": raw["available_at"],
            "content": {key: raw[key] for key in ("title", "summary", "source_id") if key in raw},
            "public": True, "partition": "training"})

    async def run_proposal_pass(self) -> dict:
        if self.proposal_runner is None or not self.training_facts:
            self.last_proposal_round = {"status": "DATA_BLOCKED",
                "reason": "no configured provider runner or public training facts", "authority": "none"}
            return self.last_proposal_round
        facts = list(self.training_facts)
        # A new hourly round is a new attributable, budgeted experiment. The
        # runner itself reuses sealed retries with the same stable identity.
        round_id = _digest({"facts": [fact["evidence_id"] for fact in facts],
                            "hour": int(time.time() // 3600)})
        report = await self.proposal_runner.run_round(round_id=round_id, facts=facts,
            schemas={"candidate": "research_contracts-v1", "provider_role": "research-proposals-only"},
            training_memory=[])
        self.last_proposal_round = report
        for request in report.get("experiment_requests", []):
            ref = self._record("unknown_agent_experiment", request, time.time())
            self.frontier.enqueue("VALIDATION_ATTACK", ref, evidence_refs=[ref],
                job_id="agent-experiment-" + request["experiment_id"],
                exploration_tags=["new_models_tools"])
        return report

    @staticmethod
    def _pit_errors(raw: dict, as_of: float) -> list[str]:
        reasons = []
        if raw.get("rights") not in LEGAL_RIGHTS:
            reasons.append("source access rights unverified")
        if raw.get("rights") == "AUTHORIZED" and not raw.get("authorization_evidence"):
            reasons.append("authorized source needs operator access provenance")
        times = [raw.get(key) for key in ("event_at", "available_at", "observed_at")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in times):
            reasons.append("event, availability and observation timestamps required")
        elif not times[0] <= times[1] <= times[2] <= as_of:
            reasons.append("future or inconsistent point-in-time evidence")
        url = raw.get("source_url")
        if url:
            parsed = urlsplit(str(url))
            if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
                reasons.append("unsupported or credential-bearing source URL")
        return reasons

    def _compile(self, raw: dict, now: float) -> tuple[str, dict]:
        errors = self._pit_errors(raw, now)
        if errors:
            return "BLOCKED", {"reasons": errors}
        candidate = raw.get("exact_candidate")
        if candidate is not None:
            admission = validate_exact_candidate(candidate)
            if not admission.ok:
                return "BLOCKED", {"reasons": list(admission.reasons)}
            # The raw signal may not present a pre-outcome freeze from later
            # than its own collection; the framework freezes it now.
            if float(candidate["created_at"]) > now:
                return "INVALID", {"reasons": ["candidate creation is in the future"]}
            return "EXACT_CANDIDATE", {"candidate": candidate,
                                        "candidate_hash": admission.candidate_hash}
        request = raw.get("data_request") or raw.get("representation_request")
        if request is not None:
            missing = [name for name in REQUEST_FIELDS if request.get(name) is None]
            if missing:
                return "BLOCKED", {"reasons": ["request missing: " + ", ".join(missing)]}
            return ("DATA_REQUEST" if raw.get("data_request") else "REPRESENTATION_REQUEST"), request
        mechanism = raw.get("mechanism")
        if not mechanism:
            return "SOURCE_REQUEST", {"expected_consumer": "mechanism inference",
                "reason": "public finding has no falsifiable mechanism yet",
                "source_ref": raw.get("source_url"), "authority": "none"}
        return "MECHANISM_HYPOTHESIS", {"mechanism": mechanism,
            "observables": list(raw.get("candidate_features") or []),
            "expected_consumer": "exact strategy compiler",
            "falsifier": raw.get("falsifier") or "after-cost chronological edge does not survive",
            "gap": "exact entry/veto/size/exit/PIT/execution assumptions not supplied",
            "authority": "none"}

    def _submit_exact(self, candidate: dict, now: float, *, allow_existing: bool = False) -> str:
        previous = self.framework.hypotheses.get(candidate["candidate_id"])
        if previous is not None and previous.exact_candidate == candidate:
            if not allow_existing:
                return "DUPLICATE"
            if self.framework._exact_frozen_valid(previous):
                return "EXACT_CANDIDATE"
            return "BLOCKED"
        spec = HypothesisSpec(
            hypothesis_id=candidate["candidate_id"], mechanism=candidate["mechanism"],
            target="net_elogw", features=sorted(candidate["feature_program"]),
            feature_hash=_digest(candidate["feature_program"]), model_type="bounded_rule_program",
            model_params={}, training_window="chronological_point_in_time", threshold=0,
            sizing_rule=candidate["size_rule"], exit_rule=candidate["exit_rule"],
            execution_policy=candidate["execution_assumptions"], fakeability={},
            cost_model=candidate["fee_policy"], falsifier=candidate["falsifier"],
            kill_thesis="canonical gauntlet rejection, decay or known hard safety veto",
            source_provenance=";".join(map(str, candidate["source_lineage"])),
            trial_family=candidate["trial_family"], created_at=candidate["created_at"],
            exact_candidate=candidate, parent_ids=candidate["parent_ids"],
            agent_identity=candidate["agent_identity"])
        result = self.framework.submit_hypothesis(spec)
        if result != "accepted":
            return "DUPLICATE" if result == "duplicate" else "BLOCKED"
        if not self.framework.freeze_hypothesis(spec.hypothesis_id):
            return "BLOCKED"
        self.framework.mark_data_blocked(spec.hypothesis_id,
            "frozen exact candidate awaiting canonical chronological/route/forward receipts")
        return "EXACT_CANDIDATE"

    def run_pass(self, *, limit: int = 8, now: Optional[float] = None) -> dict:
        if self.workflows is not None:
            return self._run_workflows_sync(limit=limit, now=now)
        now = time.time() if now is None else float(now)
        self._wake_checkpointed_gauntlet_work(limit, now)
        jobs = self.frontier.lease_jobs(self.worker, limit=limit, lease_seconds=120,
                                       resource_budget=(self.research_pass_resource_units
                                           if self.research_pass_resource_units is not None else float(limit)), now=now)
        payloads = self._load_refs({job["payload_ref"] for job in jobs})
        results = []
        for job in jobs:
            try:
                payload = payloads.get(job["payload_ref"])
                if payload is None:
                    raise ValueError("canonical evidence reference is unavailable")
                if job["kind"] == "RAW_EVIDENCE":
                    disposition, detail = self._compile(payload["raw"], now)
                    if disposition == "EXACT_CANDIDATE":
                        disposition = self._submit_exact(detail["candidate"], now)
                        detail["novelty"] = self.framework.candidate_niche(detail["candidate"]["candidate_id"])
                        if disposition == "EXACT_CANDIDATE" and detail["novelty"].get("disposition") == "DUPLICATE":
                            disposition = "DUPLICATE"
                elif (job["kind"] == "GAUNTLET" and self.gauntlet_handler is not None
                        and self.gauntlet_auction.total_cost is not None):
                    detail = self.gauntlet_handler(payload["candidate"])
                    detail["declared_resource_costs"] = self.gauntlet_auction.to_dict()
                    detail["declared_work_bound"] = self.gauntlet_work_bound
                    disposition = detail.get("disposition", "BLOCKED")
                else:
                    disposition, detail = "BLOCKED", {
                        "reason": "awaiting required observed data or configured canonical consumer",
                        "required_change": job["kind"], "authority": "none"}
                ref = self._record("unknown_conversion_receipt", {
                    "job_id": job["job_id"], "input_ref": job["payload_ref"],
                    "disposition": disposition, "detail": detail, "authority": "none"}, now)
                if disposition == "EXACT_CANDIDATE":
                    self._queue_gauntlet(detail, job["axes"], [ref], now)
                elif disposition in {"DATA_REQUEST", "REPRESENTATION_REQUEST", "SOURCE_REQUEST",
                                      "MECHANISM_HYPOTHESIS"}:
                    self.frontier.enqueue(disposition, ref, axes=job["axes"],
                        evidence_refs=[ref], auction=AuctionInputs(),
                        exploration_tags=["unknown_mechanisms"], now=now)
                self.frontier.ack(job["job_id"], self.worker, job["lease_token"],
                                  disposition=disposition, evidence_refs=[ref], now=now)
                results.append({"job_id": job["job_id"], "disposition": disposition, "receipt": ref})
            except Exception as exc:
                self.frontier.retry(job["job_id"], self.worker, job["lease_token"],
                                    reason=f"{type(exc).__name__}: {exc}", now=now)
                results.append({"job_id": job["job_id"], "disposition": "BLOCKED"})
        self.last_pass = {"status": "OK" if jobs else "NO_COSTED_WORK_DUE",
                          "processed": len(results), "results": results, "authority": "none"}
        return self.last_pass

    def _workflow_jobs(self, limit: int, now: float):
        self._wake_checkpointed_gauntlet_work(limit, now)
        self._drain_evidence_wakes(now)
        self.workflows.prepare(now)
        return self.frontier.lease_jobs(self.worker, limit=limit, lease_seconds=600,
            resource_budget=self.research_pass_resource_units or float(limit),
            kinds=sorted({"RAW_EVIDENCE"} | self.workflows.kinds), now=now)

    def _dispatch_work(self, job: dict, payload: dict, now: float):
        if job["kind"] != "RAW_EVIDENCE":
            refs = list(dict.fromkeys([job["payload_ref"], *job["evidence_refs"]]))
            return self.workflows.consume(job["kind"], payload, refs, job["axes"], now)
        disposition, detail = self._compile(payload["raw"], now)
        if disposition == "EXACT_CANDIDATE":
            disposition = self._submit_exact(detail["candidate"], now,
                allow_existing=payload["raw"].get("source_id") == "canonical-research-compiler")
            detail["novelty"] = self.framework.candidate_niche(detail["candidate"]["candidate_id"])
        children = []
        if disposition in {"DATA_REQUEST", "REPRESENTATION_REQUEST", "SOURCE_REQUEST", "MECHANISM_HYPOTHESIS"}:
            children.append({"kind": disposition, "payload": detail})
        return {"disposition": disposition, "detail": detail, "children": children}

    def _commit_work(self, job: dict, result: dict, now: float) -> dict:
        from src.research.frontier import DISPOSITIONS
        if not isinstance(result, dict) or not isinstance(result.get("detail"), dict):
            raise ValueError("consumer must return a typed result")
        children = result.get("children", [])
        if not isinstance(children, list) or len(children) > 64:
            raise ValueError("consumer exceeded bounded child work contract")
        actual = str(result.get("disposition", "BLOCKED"))
        blocked = actual in {"BLOCKED", "DATA_BLOCKED"}
        disposition = "BLOCKED" if blocked else actual if actual in DISPOSITIONS else "DEEPENING_TASK"
        detail = result["detail"]
        receipts = self.framework.get_research_component("research_conversion_receipts") or {"refs": {}}
        result_key = _digest([job["job_id"], result])
        ref = receipts["refs"].get(result_key)
        if ref is None:
            ref = self._record("unknown_conversion_receipt", {"job_id": job["job_id"],
                "input_ref": job["payload_ref"], "disposition": disposition,
                "consumer_disposition": actual, "detail": detail, "authority": "none"}, now)
            receipts["refs"][result_key] = ref
            self.framework.checkpoint_research_component("research_conversion_receipts", receipts)
        if job["kind"] == "GAUNTLET" and detail.get("status") in {"REJECTED", "HISTORICAL_SURVIVOR"}:
            receipt = detail["receipt"]
            hid, seal = receipt["candidate_id"], receipt["spec_hash"]
            descendant = {"parent_id": hid, "parent_frozen_hash": seal}
            if detail["status"] == "HISTORICAL_SURVIVOR":
                descendant["survival_evidence"] = {"verdict": "SURVIVOR", "spec_hash": seal, "evidence_ref": ref}
                next_kind = "SURVIVOR_DESCENDANT"
            else:
                # A rejected test is an attributed result, not proof of a
                # specific causal failure. The diagnosis consumer requests
                # discriminating evidence before permitting resurrection.
                descendant["attribution"] = {"canonical_receipt_ref": ref,
                    "observed_test_result": receipt.get("results", {}).get("test", {}).get(hid+"@"+seal, {})}
                next_kind = "FAILURE_DESCENDANT"
            self.enqueue_work(next_kind, descendant, axes=job["axes"], evidence_refs=[ref], now=now)
        # Persist every downstream child before acknowledging the parent. A
        # retry reuses semantic child identities, including after a crash.
        for child in children:
            self.enqueue_work(child["kind"], child["payload"], axes=child.get("axes", job["axes"]),
                evidence_refs=[job["payload_ref"], ref], now=now)
        if disposition == "EXACT_CANDIDATE":
            self._queue_gauntlet(detail, job["axes"], [job["payload_ref"], ref], now)
        changes = detail.get("changes", [])
        if changes:
            self.notify_evidence(changes, [job["payload_ref"], ref], now)
        # Malformed raw input has a terminal validation receipt. Incomplete
        # typed work retains its identity and exact named wake condition.
        if blocked and job["kind"] != "RAW_EVIDENCE":
            change = result.get("enabling_change") or detail.get("enabling_change")
            if not change:
                change = next((child.get("enabling_change") for child in children if child.get("enabling_change")), None)
            change = change or "research-input:" + job["job_id"]
            self.frontier.block_leased_job(job["job_id"], self.worker, job["lease_token"],
                enabling_change=change, reason=str(detail.get("reason") or detail.get("status") or "observed input missing"),
                evidence_refs=[ref], now=now)
        else:
            self.frontier.ack(job["job_id"], self.worker, job["lease_token"],
                disposition=disposition, evidence_refs=[ref], now=now)
        return {"job_id": job["job_id"], "disposition": disposition, "receipt": ref}

    def _retry_work(self, job: dict, exc: Exception, now: float) -> dict:
        self.frontier.retry(job["job_id"], self.worker, job["lease_token"],
            reason=f"{type(exc).__name__}: {exc}", now=now)
        return {"job_id": job["job_id"], "disposition": "BLOCKED", "reason": str(exc)}

    def _pass_report(self, jobs: list, results: list) -> dict:
        self.last_pass = {"status": "OK" if jobs else "NO_COSTED_WORK_DUE",
                          "processed": len(results), "results": results, "authority": "none"}
        return self.last_pass

    def _run_workflows_sync(self, *, limit: int, now: Optional[float]) -> dict:
        now = time.time() if now is None else float(now)
        jobs = self._workflow_jobs(limit, now)
        results = []
        for job in jobs:
            try:
                payload = self._load_refs({job["payload_ref"]})[job["payload_ref"]]
                result = self._dispatch_work(job, payload, now)
                if inspect.isawaitable(result):
                    result.close()
                    raise RuntimeError("source acquisition requires the existing asynchronous research pass")
                results.append(self._commit_work(job, result, now))
            except Exception as exc:
                results.append(self._retry_work(job, exc, now))
        return self._pass_report(jobs, results)

    async def run_async_pass(self, *, limit: int = 8, now: Optional[float] = None) -> dict:
        import asyncio
        if self.workflows is None:
            return self.run_pass(limit=limit, now=now)
        now = time.time() if now is None else float(now)
        jobs = self._workflow_jobs(limit, now)
        results = []
        for job in jobs:
            try:
                payload = self._load_refs({job["payload_ref"]})[job["payload_ref"]]
                result = self._dispatch_work(job, payload, now)
                if inspect.isawaitable(result):
                    result = await result
                # Acquisition timestamps come from completion, not a pass's
                # earlier start. Controlled tests can supply a fixed clock.
                results.append(self._commit_work(job, result, now))
            except Exception as exc:
                results.append(self._retry_work(job, exc, now))
            await asyncio.sleep(0)
        return self._pass_report(jobs, results)

    def report(self) -> dict:
        return {"status": "SHADOW_ONLY", "authority": "none", "frontier": self.frontier.report(),
                "last_pass": self.last_pass, "canonical_evidence": self.evidence.report(),
                "proposal_round": self.last_proposal_round,
                "workflows": self.workflows.report() if self.workflows is not None else {"status": "NOT_CONFIGURED"},
                "gauntlet_adapter": {"configured": self.gauntlet_handler is not None,
                    "configuration": getattr(self, "gauntlet_configuration", None),
                    "resource_costs": self.gauntlet_auction.to_dict(),
                    "work_bound": self.gauntlet_work_bound or None, "monetary_cost": None},
                "validated_incremental_log_growth": None,
                "runtime_edge_evidence": "UNMEASURED"}
