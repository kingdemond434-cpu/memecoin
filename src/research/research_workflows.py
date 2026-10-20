"""Consumers of the existing frontier queue, driven by its one research loop."""
from __future__ import annotations

import json
from typing import Any

from src.research.data_requests import ResearchDataService, finite
from src.research.frontier import AuctionInputs


POPULATION = {"ACTOR_OBSERVATION", "NARRATIVE_OBSERVATION", "FOLLOW_OUTCOME", "SOURCE_OUTCOME"}
LABORATORY = {"MECHANISM_HYPOTHESIS", "MECHANISM_LAB", "MACHINE_SEARCH", "FAILURE_DESCENDANT", "SURVIVOR_DESCENDANT", "DEEPENING_TASK"}
VALIDATION = {"GAUNTLET", "VALIDATION_ATTACK", "FORWARD_SHADOW", "CHAIN_INTEGRATION", "PROTOCOL_INTEGRATION", "INTEGRATION"}
DATA = {"DATA_REQUEST", "REPRESENTATION_REQUEST", "SOURCE_REQUEST", "OBSERVABLE_SERIES", "DATA_OBSERVATION", "SOURCE_ACQUISITION"}
INGESTION = {"MINER_BATCH_REPLAY", "DISCOVERY_RESPONSE", "NATIVE_EVENT_RESEARCH", "NATIVE_OBSERVATION"}

# A numeric field is indexed only when its unit is declared here or in the
# source's explicit observable contract. Arbitrary numeric JSON is not alpha.
MEASURED_UNITS = {"price_usd": "USD/token", "liquidity_usd": "USD", "volume_usd": "USD",
    "market_cap_usd": "USD", "notional_sol": "SOL", "independent_buyers": "count",
    "buy_count": "count", "sell_count": "count", "holders": "count", "buy_velocity": "count/second",
    "curve_progress": "fraction", "price_impact": "fraction", "priority_fee": "lamports",
    "buy_acceleration": "count/second^2", "liquidity_change": "USD", "wallet_activity": "count"}


class ResearchWorkflows:
    def __init__(self, engine: Any, *, population: Any, laboratory: Any, validation: Any,
                 data: ResearchDataService | None = None, seed_labs: bool = False, lab_contexts=None):
        self.engine = engine
        self.population, self.laboratory, self.validation = population, laboratory, validation
        self.data = data or ResearchDataService(engine.framework, engine._load_refs)
        self.kinds = POPULATION | LABORATORY | VALIDATION | DATA | INGESTION
        self.configuration_ref = engine._record("research_consumers_configured", {
            "kinds": sorted(self.kinds), "max_rows_per_conversion": self.data.max_points,
            "resource_semantics": "declared bounded operations; monetary cost unmeasured"}, __import__("time").time())
        state = engine.framework.get_research_component("research_workflow_cursor") or {}
        self.cursor = state.get("migration_cursor", "")
        self.blocked_cursor = state.get("blocked_source_cursor", "")
        self.seed_labs = seed_labs
        self.lab_contexts = dict(lab_contexts or {})

    def auction(self, kind: str) -> AuctionInputs:
        if kind == "GAUNTLET":
            return self.engine.gauntlet_auction
        # Four source-acquisition units permit at most two source calls in an
        # eight-unit pass. Search work has its own hard row/variant bounds.
        return AuctionInputs(costs={"inference": 0, "data": 3 if kind == "SOURCE_ACQUISITION" else 0,
                                    "engineering": 0, "conversion": 1, "gauntlet": 0})

    def prepare(self, now: float, limit: int = 64) -> None:
        if self.seed_labs:
            from src.research.mechanism_labs import STRATEGY_FAMILIES
            state = self.engine.framework.get_research_component("research_family_intake") or {"seeded": {}}
            for family, lab_id in sorted(STRATEGY_FAMILIES.items()):
                if family in state["seeded"]:
                    continue
                state["seeded"][family] = self.engine.enqueue_work("MECHANISM_LAB",
                    {"lab_id": lab_id, "strategy_family": family,
                     "context": self.lab_contexts.get(family, self.lab_contexts.get(lab_id, {}))},
                    axes={"representation": family, "public_actor_mechanism": lab_id},
                    evidence_refs=[self.configuration_ref], now=now)
                self.engine.framework.checkpoint_research_component("research_family_intake", state)
                break  # one family admission per bounded existing pass
        jobs = self.engine.frontier.pending_work(kinds=sorted(self.kinds-{"GAUNTLET"}),
                                               limit=limit, after_job_id=self.cursor)
        for job in jobs:
            if job["total_cost"] is None:
                self.engine.frontier.price_pending_job(job["job_id"], auction=self.auction(job["kind"]),
                    evidence_refs=[self.configuration_ref], reason="bounded canonical consumer now registered", now=now)
        self.cursor = jobs[-1]["job_id"] if len(jobs) == limit else ""
        blocked = self.engine.frontier.pending_work(kinds=("SOURCE_REQUEST", "SOURCE_ACQUISITION"),
            states=("HIBERNATED",), limit=16, after_job_id=self.blocked_cursor)
        for job in blocked:
            if not str(job.get("enabling_change", "")).startswith("source-capability:"):
                continue
            payload = self.unwrap(self.engine._load_refs({job["payload_ref"]})[job["payload_ref"]])
            eligible = payload.get("source_id") in self.data.sources if job["kind"] == "SOURCE_ACQUISITION" else False
            if job["kind"] == "SOURCE_REQUEST":
                try: eligible = self.data._source(payload).get("disposition") == "SOURCE_REQUEST"
                except (ValueError, TypeError): eligible = False
            if eligible:
                self.engine.notify_evidence([job["enabling_change"]], [self.configuration_ref], now)
        self.blocked_cursor = blocked[-1]["job_id"] if len(blocked) == 16 else ""
        self.engine.framework.checkpoint_research_component("research_workflow_cursor", {
            "migration_cursor": self.cursor, "blocked_source_cursor": self.blocked_cursor})
        reconcile = getattr(self.validation, "reconcile_forward", None)
        if reconcile is not None:
            reconcile()

    @staticmethod
    def unwrap(payload: dict) -> dict:
        if "point_payload" in payload:
            return payload["point_payload"]
        if "detail" in payload and "disposition" in payload:
            return payload["detail"]
        return payload

    def consume(self, kind: str, payload: dict, refs: list[str], axes: dict, now: float) -> dict:
        payload = self.unwrap(payload)
        if kind in POPULATION:
            return self.population.consume(kind, payload, refs, now)
        if kind in LABORATORY:
            return self.laboratory.consume(kind, payload, refs, now)
        if kind in VALIDATION:
            if kind == "GAUNTLET" and isinstance(payload.get("candidate"), dict):
                candidate = payload["candidate"]
                spec = self.engine.framework.hypotheses.get(candidate.get("candidate_id"))
                if spec is None or spec.exact_candidate != candidate:
                    raise ValueError("queued gauntlet content no longer matches frozen candidate")
                payload = {"candidate_id": candidate["candidate_id"],
                           "spec_hash": payload.get("spec_hash", spec.frozen_hash)}
            if kind in {"CHAIN_INTEGRATION", "PROTOCOL_INTEGRATION", "INTEGRATION"}:
                chain = payload.get("chain") or axes.get("chain")
                for adapter in self.validation.adapters.values():
                    if str(chain) in {adapter.chain_id, adapter.chain_name}:
                        chain = adapter.chain_name
                        break
                payload = {**payload, "chain": payload.get("chain") or axes.get("chain"),
                    "protocol": payload.get("protocol") or axes.get("protocol"),
                    "capabilities": payload.get("capabilities", ["observation"])}
                payload["chain"] = chain
            return self.validation.consume(kind, {**payload, "axes": axes}, refs, now)
        if kind in DATA:
            if kind == "SOURCE_ACQUISITION":
                return self.data.acquire(payload, refs, now)
            return self.data.consume(kind, payload, refs, now)
        if kind == "MINER_BATCH_REPLAY":
            if payload.get("batch_ref"):
                retained = self.engine._load_refs({payload["batch_ref"]}).get(payload["batch_ref"])
                if retained is None:
                    raise ValueError("retained original batch unavailable")
                retained = self.unwrap(retained)
                payload = {**retained, "cursor": payload["cursor"], "batch_ref": payload["batch_ref"]}
            return self._batch(payload, refs, now)
        if kind == "DISCOVERY_RESPONSE":
            body = payload.get("raw_body", payload.get("body"))
            if isinstance(body, str):
                if len(body.encode()) > 2_000_000:
                    return {"disposition": "BLOCKED", "detail": {"reason": "retained source body exceeds bounded parse envelope"},
                            "enabling_change": "source-parser:streaming", "children": []}
                try: body = json.loads(body)
                except ValueError: body = None
            if not isinstance(body, (dict, list)):
                return {"disposition": "BLOCKED", "detail": {"reason": "source response has no supported structured mapping"},
                        "enabling_change": "source-schema:"+str(payload.get("source_url", "unknown")), "children": []}
            rows = body if isinstance(body, list) else body.get("pairs", body.get("data", []))
            if not isinstance(rows, list):
                rows = []
            return {"disposition": "DEEPENING_TASK", "detail": {"status": "DISCOVERY_RESPONSE_ROUTED", "raw_rows": len(rows),
                "native_execution": False, "input_refs": refs}, "children": [{"kind": "MINER_BATCH_REPLAY", "payload": {
                    "miner_id": "dexscreener:public-api", "records": rows, "cursor": 0,
                    "observed_at": payload.get("observed_at"), "rights": "PUBLIC", "source_response_refs": refs}}]}
        if kind in {"NATIVE_EVENT_RESEARCH", "NATIVE_OBSERVATION"}:
            metadata = payload.get("metadata", {})
            chain = axes.get("chain") or metadata.get("chain_id")
            protocol = axes.get("protocol") or metadata.get("protocol")
            return {"disposition": "DEEPENING_TASK", "detail": {"status": "NATIVE_OBSERVATION_ROUTED"}, "children": [
                {"kind": "MECHANISM_LAB", "payload": {"lab_id": "migration", "chain": chain,
                    "protocol": protocol, "observation_refs": refs, "native_event_type": metadata.get("event_type")}, "axes": axes}]}
        raise ValueError("no canonical consumer for work kind")

    def _batch(self, payload: dict, refs: list[str], now: float) -> dict:
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("miner batch lacks retained record list")
        cursor = payload.get("cursor", 0)
        if type(cursor) is not int or not 0 <= cursor <= len(records):
            raise ValueError("invalid retained batch cursor")
        end = min(len(records), cursor+self.data.max_points)
        children = []
        points = []
        for row in records[cursor:end]:
            if not isinstance(row, dict):
                continue
            if row.get("variable") is not None:
                points.append(row)
                continue
            observed = row.get("_fetched_at", payload.get("observed_at"))
            if not finite(observed) or not 0 < observed <= now:
                continue
            # Latest-sample APIs describe the measurement at receipt, never
            # a guessed historic publication time. Availability is receipt.
            contracts = row.get("observable_contracts", {})
            if "chainId" in row:
                row = {**row, "chain": row["chainId"], "token": (row.get("baseToken") or {}).get("address"),
                       "liquidity_usd": (row.get("liquidity") or {}).get("usd")}
                # Numeric strings are accepted only for this documented API
                # field; malformed or nonfinite prices remain unobserved.
                try: row["price_usd"] = float(row["priceUsd"])
                except (KeyError, TypeError, ValueError): pass
            for variable, value in row.items():
                unit = contracts.get(variable, {}).get("unit") or MEASURED_UNITS.get(variable)
                if not unit or not finite(value):
                    continue
                points.append({"variable": variable, "value": value, "unit": unit,
                    "event_at": observed, "available_at": observed, "observed_at": observed,
                    "partition": row.get("research_partition", "observation"),
                    "source_id": payload.get("miner_id") or row.get("_miner"),
                    "chain": row.get("chain") or row.get("chain_id") or payload.get("chain"),
                    "entity_id": row.get("token") or row.get("mint") or row.get("address") or row.get("pairAddress"),
                    "rights": payload.get("rights", "PUBLIC"), "rights_ref": payload.get("rights_ref"),
                    "point_in_time": True, "timestamp_semantics": "sample_observation_time"})
        # Chunk derived points as well as raw rows; no silent transform tail.
        for start in range(0, len(points), self.data.max_points):
            children.append({"kind": "OBSERVABLE_SERIES", "payload": {"points": points[start:start+self.data.max_points], "raw_refs": refs}})
        if end < len(records):
            children.append({"kind": "MINER_BATCH_REPLAY", "payload": {
                "batch_ref": payload.get("batch_ref", refs[0]), "cursor": end}})
        return {"disposition": "DEEPENING_TASK", "detail": {"status": "BATCH_CONVERTED", "cursor": cursor,
            "processed_rows": end-cursor, "deferred_rows": len(records)-end,
            "measured_points": len(points), "unmapped_rows_retained": refs}, "children": children}

    def report(self) -> dict:
        return {"status": "CONSUMERS_REGISTERED", "consumers": sorted(self.kinds),
            "data": self.data.report(), "population": self.population.report(),
            "laboratory": self.laboratory.report(), "validation": self.validation.report(),
            "authority": "none"}
