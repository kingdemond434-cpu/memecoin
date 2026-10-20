"""Bounded data/source/representation consumers over canonical evidence refs.

The catalogue stores descriptors and references in the existing framework
checkpoint. Values live only in the canonical ledger. Acquisitions can call
registered desk sources; a proposal cannot introduce an arbitrary URL.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import time
from typing import Any, Callable, Mapping


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def measured_value(point: Mapping) -> bool:
    return finite(point.get("value")) or (type(point.get("value")) is bool
        and isinstance(point.get("observable_contract"), Mapping)
        and point["observable_contract"].get("dtype") == "bool")


class ResearchDataService:
    COMPONENT = "research_data_catalogue_v1"
    TRANSFORMS = {"level", "change", "acceleration", "rank", "percentile", "z_score",
        "persistence", "duration", "transition", "hazard", "residual", "ratio", "spread",
        "relative_value", "entropy", "concentration", "dispersion", "curvature",
        "cross_chain_differential", "actor_adjusted_value", "source_disagreement",
        "model_disagreement", "state_conditioned_value"}

    def __init__(self, framework: Any, loader: Callable, *, max_points: int = 256,
                 max_catalogue_refs: int = 20000):
        self.framework, self.loader = framework, loader
        self.max_points = min(1000, max(8, int(max_points)))
        self.max_catalogue_refs = max(self.max_points, int(max_catalogue_refs))
        self.state = framework.get_research_component(self.COMPONENT) or {
            "series": {}, "cache": {}, "evicted_refs": 0, "requests": 0,
            "acquisitions": 0, "representations": 0}
        self.sources: dict[str, dict] = {}

    def _save(self):
        self.framework.checkpoint_research_component(self.COMPONENT, self.state)

    def register_source(self, source_id: str, fetch: Callable, *, source_classes=(),
                        rights: str = "PUBLIC", rights_ref: str | None = None, chain: str | None = None) -> None:
        if not source_id or not callable(fetch) or rights not in {"PUBLIC", "AUTHORIZED", "LICENSED"}:
            raise ValueError("named registered source and declared rights required")
        if rights != "PUBLIC" and not rights_ref:
            raise ValueError("restricted source requires access provenance")
        self.sources[source_id] = {"fetch": fetch, "classes": set(source_classes),
                                   "rights": rights, "rights_ref": rights_ref, "chain": chain}

    @staticmethod
    def _points(payload: Mapping) -> list[dict]:
        if isinstance(payload.get("points"), list):
            return payload["points"]
        if payload.get("variable") is not None and "value" in payload:
            return [dict(payload)]
        if isinstance(payload.get("observables"), list):
            return payload["observables"]
        return []

    def observe(self, payload: Mapping, evidence_ref: str, *, now: float) -> list[str]:
        """Index explicit measured series. No timestamp or partition inference."""
        changes = []
        staged = {}
        points = self._points(payload)
        if len(points) > self.max_points:
            raise ValueError("observable batch exceeds declared bound; retain and page the original work")
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                continue
            variable = point.get("variable")
            stamps = [point.get(k) for k in ("event_at", "available_at", "observed_at")]
            if (not isinstance(variable, str) or not variable or not measured_value(point)
                    or any(not finite(value) for value in stamps)
                    or not 0 < stamps[0] <= stamps[1] <= stamps[2] <= now
                    or point.get("rights") not in {"PUBLIC", "AUTHORIZED", "LICENSED"}
                    or not point.get("source_id") or not point.get("unit")
                    or point.get("partition") not in {"training", "validation", "test", "forward", "observation"}
                    or point.get("point_in_time") is not True):
                continue
            if point["rights"] != "PUBLIC" and not point.get("rights_ref"):
                continue
            descriptor = {k: point.get(k) for k in ("variable", "unit", "source_id", "chain",
                "entity_id", "row_id", "episode_id", "partition", "event_at", "available_at", "observed_at")}
            identity = digest(descriptor)
            descriptor["value_hash"] = digest(point["value"])
            descriptor.update(ref=evidence_ref, point_index=index)
            previous = staged.get((variable, identity)) or self.state["series"].get(variable, {}).get(identity)
            if previous is not None and previous.get("value_hash") != descriptor["value_hash"]:
                raise ValueError("conflicting values for one source observation identity")
            if previous is None:
                staged[variable, identity] = descriptor
        for (variable, identity), descriptor in staged.items():
            self.state["series"].setdefault(variable, {})[identity] = descriptor
            changes.append("observable:" + variable)
        # Index pressure is explicit; canonical raw data is retained. An
        # archival acquisition request can re-index a referenced older batch.
        total = sum(len(rows) for rows in self.state["series"].values())
        if total > self.max_catalogue_refs:
            oldest = sorted((row["observed_at"], name, key)
                for name, rows in self.state["series"].items() for key, row in rows.items())
            for _, name, key in oldest[:total-self.max_catalogue_refs]:
                del self.state["series"][name][key]
                self.state["evicted_refs"] += 1
        if changes:
            self._save()
        return sorted(set(changes))

    def _read(self, variable: str, request: dict, now: float) -> list[dict]:
        as_of = request.get("as_of", now)
        if not finite(as_of) or as_of > now:
            raise ValueError("request as_of is in the future")
        descriptors = [row for row in self.state["series"].get(variable, {}).values()
            if row["available_at"] <= as_of and row["observed_at"] <= as_of
            and all(request.get(key) is None or row.get(key) == request[key]
                    for key in ("chain", "entity_id", "unit", "partition"))]
        descriptors.sort(key=lambda row: (row["event_at"], row["available_at"], row["ref"]))
        descriptors = descriptors[-self.max_points:]
        loaded = self.loader({row["ref"] for row in descriptors})
        points = []
        for row in descriptors:
            value = loaded.get(row["ref"])
            if value is None:
                raise ValueError("catalogued canonical evidence is unavailable")
            if "raw" in value:
                value = value["raw"]
            if "point_payload" in value:
                value = value["point_payload"]
            candidates = self._points(value)
            try:
                point = candidates[row["point_index"]]
            except (IndexError, TypeError):
                raise ValueError("catalogue pointer no longer resolves")
            if point.get("variable") != variable or not measured_value(point):
                raise ValueError("catalogue pointer disagrees with canonical evidence")
            points.append({**point, "evidence_ref": row["ref"]})
        return points

    @staticmethod
    def _blocked(reason: str, change: str, *, children=()) -> dict:
        return {"disposition": "BLOCKED", "detail": {"status": "DATA_BLOCKED", "reason": reason},
                "enabling_change": change, "children": list(children)}

    def consume(self, kind: str, payload: dict, evidence_refs: list[str], now: float) -> dict:
        kind = kind.upper()
        request = payload.get("request", payload)
        if not isinstance(request, dict):
            return {"disposition": "INVALID", "detail": {"reason": "request must be an object"}, "children": []}
        key = digest([kind, request, evidence_refs])
        cached = self.state["cache"].get(key)
        if cached is not None:
            return cached
        try:
            if kind in {"OBSERVABLE_SERIES", "DATA_OBSERVATION"}:
                changes = self.observe(request, evidence_refs[0], now=now)
                result = {"disposition": "DEEPENING_TASK", "detail": {
                    "status": "OBSERVED" if changes else "DATA_BLOCKED", "changes": changes}, "children": []}
            elif kind == "REPRESENTATION_REQUEST":
                result = self._representation(request, evidence_refs, now)
            elif kind == "DATA_REQUEST":
                result = self._data(request, evidence_refs, now)
            elif kind == "SOURCE_REQUEST":
                result = self._source(request)
            else:
                result = {"disposition": "INVALID", "detail": {"reason": "unsupported data workflow"}, "children": []}
        except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
            result = {"disposition": "INVALID", "detail": {"reason": str(exc)}, "children": []}
        self.state["requests"] += 1
        # Blocked answers must be reevaluated after their named input changes.
        if result["disposition"] != "BLOCKED":
            self.state["cache"][key] = result
            if len(self.state["cache"]) > 512:
                self.state["cache"].pop(next(iter(self.state["cache"])))
        self._save()
        return result

    def _data(self, request: dict, refs: list[str], now: float) -> dict:
        variable = request.get("variable") or request.get("missing_variable")
        if not isinstance(variable, str) or not variable:
            return self._blocked("a measurable variable must be named", "request-contract:variable")
        rows = self._read(variable, request, now)
        minimum = request.get("min_observations", 1)
        if type(minimum) is not int or not 1 <= minimum <= self.max_points:
            raise ValueError("min_observations exceeds declared data work bound")
        if len(rows) < minimum:
            source_request = {**request, "variable": variable, "expected_consumer": request.get("expected_consumer", "mechanism_lab"),
                              "requested_source_ids": request.get("source_ids", [])}
            return self._blocked("required observed PIT series is incomplete", "observable:" + variable,
                children=[{"kind": "SOURCE_REQUEST", "payload": source_request}])
        partitions = sorted({row["partition"] for row in rows})
        consumer = request.get("resume_kind")
        children = []
        if consumer:
            if consumer not in {"MECHANISM_LAB", "MECHANISM_HYPOTHESIS", "MACHINE_SEARCH", "REPRESENTATION_REQUEST", "VALIDATION_ATTACK",
                                "FAILURE_DESCENDANT", "SURVIVOR_DESCENDANT", "DEEPENING_TASK"}:
                raise ValueError("unsupported data request resume consumer")
            children.append({"kind": consumer, "payload": {**request.get("resume_payload", {}),
                "observed_data": {"variable": variable, "rows": rows, "partitions": partitions}}})
        return {"disposition": "DEEPENING_TASK", "detail": {"status": "DATA_AVAILABLE", "variable": variable,
            "observations": len(rows), "partitions": partitions,
            "evidence_refs": sorted({row["evidence_ref"] for row in rows}), "coverage_truncated": len(rows) == self.max_points},
            "children": children}

    def _source(self, request: dict) -> dict:
        requested = request.get("requested_source_ids") or request.get("source_ids") or []
        classes = set(request.get("source_classes") or [])
        if not (request.get("variable") or request.get("missing_variable")) or not (requested or classes):
            return self._blocked("acquisition needs a measurable variable and named source classes or registered IDs",
                                 "source-request-contract:" + str(request.get("variable", "unresolved")))
        if not isinstance(requested, list) or any(not isinstance(x, str) for x in requested):
            raise ValueError("source IDs must name registered capabilities")
        eligible = [name for name, source in sorted(self.sources.items())
                    if (not requested or name in requested) and (not classes or classes.intersection(source["classes"]))]
        if not eligible:
            return self._blocked("no registered source with lawful access matches this request",
                                 "source-capability:" + str(request.get("variable", "unresolved")))
        cursor = request.get("source_cursor", 0)
        snapshot = request.get("source_page_ids", eligible)
        if (type(cursor) is not int or not 0 <= cursor < len(snapshot)
                or not isinstance(snapshot, list) or len(snapshot) > 256
                or any(name not in eligible for name in snapshot)):
            raise ValueError("source page must name bounded eligible registered capabilities")
        end = min(len(snapshot), cursor+4)
        children = [{"kind": "SOURCE_ACQUISITION", "payload": {"source_id": name, "request": request}}
                    for name in snapshot[cursor:end]]
        if end < len(snapshot):
            children.append({"kind": "SOURCE_REQUEST", "payload": {**request,
                "source_cursor": end, "source_page_ids": snapshot}})
        return {"disposition": "SOURCE_REQUEST", "detail": {"status": "ACQUISITION_QUEUED",
            "sources": snapshot[cursor:end], "deferred_sources": len(snapshot)-end}, "children": children}

    async def acquire(self, payload: dict, evidence_refs: list[str], now: float) -> dict:
        source_id = payload.get("source_id")
        source = self.sources.get(source_id)
        if source is None:
            return self._blocked("registered acquisition capability unavailable", "source-capability:" + str(source_id))
        rows = source["fetch"]()
        if inspect.isawaitable(rows):
            import asyncio
            rows = await asyncio.wait_for(rows, timeout=30)
        if not isinstance(rows, list):
            raise ValueError("registered source did not return a bounded record list")
        self.state["acquisitions"] += 1
        self._save()
        # Persist complete source batch as canonical child before processing a
        # bounded page. Conversion never acknowledges away the deferred tail.
        return {"disposition": "DATA_REQUEST", "detail": {"status": "SOURCE_OBSERVED", "source_id": source_id,
            "records": len(rows), "rights": source["rights"]}, "children": [
            {"kind": "MINER_BATCH_REPLAY", "payload": {"miner_id": source_id, "records": rows,
                "cursor": 0, "observed_at": time.time(), "rights": source["rights"], "rights_ref": source["rights_ref"], "chain": source["chain"],
                "request": payload.get("request", {})}}]}

    def _representation(self, request: dict, refs: list[str], now: float) -> dict:
        transform = str(request.get("transform") or request.get("representation") or "").lower().replace("-", "_")
        if transform not in self.TRANSFORMS or not request.get("mechanism") or not request.get("falsifier"):
            raise ValueError("representation requires an economic mechanism, falsifier and supported transform")
        variable = request.get("input_variable") or request.get("variable")
        if not isinstance(variable, str):
            raise ValueError("representation input variable required")
        rows = self._read(variable, request, now)
        if not rows:
            return self._blocked("representation input series unavailable", "observable:" + variable,
                children=[{"kind": "DATA_REQUEST", "payload": {**request, "variable": variable}}])
        if any(not finite(row["value"]) for row in rows):
            raise ValueError("numeric representation requires explicitly numeric observations")
        peers = []
        paired = transform in {"ratio", "spread", "relative_value", "residual", "cross_chain_differential",
                                "actor_adjusted_value", "source_disagreement", "model_disagreement", "state_conditioned_value"}
        if paired:
            other = request.get("other_variable")
            if not other:
                raise ValueError("paired representation requires an explicit other_variable")
            peer_request = {**request, **request.get("other_scope", {})}
            peers = self._read(other, peer_request, now)
            if not peers:
                return self._blocked("paired representation input unavailable", "observable:" + other,
                    children=[{"kind": "DATA_REQUEST", "payload": {**peer_request, "variable": other}}])
        if len({(row.get("chain"), row.get("entity_id"), row["unit"]) for row in rows}) != 1:
            raise ValueError("representation needs one declared entity, chain and unit scope")
        if len({row["partition"] for row in rows}) != 1:
            raise ValueError("representation must explicitly select one research partition")
        if len({(row["rights"], row.get("rights_ref")) for row in rows}) != 1:
            raise ValueError("representation needs one declared access-rights scope")
        output = []
        window = request.get("window", 16)
        if type(window) is not int or not 2 <= window <= self.max_points:
            raise ValueError("representation window exceeds declared bound")
        for index, row in enumerate(rows):
            history = rows[max(0, index-window+1):index+1]
            row = {**row, "available_at": max(r["available_at"] for r in history),
                   "observed_at": max(r["observed_at"] for r in history)}
            values = [r["value"] for r in history]
            x = values[-1]
            value = None
            unit = ("count" if transform in {"rank", "persistence"} else "seconds" if transform == "duration"
                    else "dimensionless" if transform in {"percentile", "z_score", "transition", "hazard", "entropy", "concentration"}
                    else row["unit"])
            if transform == "level": value = x
            elif transform in {"change", "transition"} and len(values) >= 2:
                value = x-values[-2] if transform == "change" else float(x != values[-2])
            elif transform in {"acceleration", "curvature"} and len(values) >= 3: value = x-2*values[-2]+values[-3]
            elif transform == "rank": value = sum(v < x for v in values) + (sum(v == x for v in values)+1)/2
            elif transform == "percentile": value = sum(v <= x for v in values)/len(values)
            elif transform == "z_score" and len(values) >= 2 and statistics.pstdev(values) > 0:
                value = (x-statistics.mean(values))/statistics.pstdev(values)
            elif transform in {"persistence", "duration"}:
                threshold = request.get("threshold")
                if not finite(threshold): raise ValueError("persistence/duration requires explicit threshold")
                run = 0
                for v in reversed(values):
                    if v < threshold: break
                    run += 1
                value = float(run) if transform == "persistence" else (row["event_at"]-history[-run]["event_at"] if run else 0.0)
            elif transform in {"entropy", "concentration"} and all(v >= 0 for v in values) and sum(values) > 0:
                weights = [v/sum(values) for v in values]
                value = (-sum(p*math.log(p) for p in weights if p > 0) if transform == "entropy" else sum(p*p for p in weights))
            elif transform == "dispersion" and len(values) >= 2: value = statistics.pstdev(values)
            elif transform == "hazard":
                if request.get("interpretation") != "observed_binary_event_frequency" or any(v not in (0, 1) for v in values):
                    raise ValueError("hazard needs observed binary event/risk-set encoding")
                value = sum(values)/len(values)
            elif paired:
                matches = [peer for peer in peers if peer["event_at"] == row["event_at"]
                           and peer["available_at"] <= request.get("as_of", now)]
                if len(matches) != 1: continue
                peer = matches[0]
                if peer["partition"] != row["partition"] or (peer["rights"], peer.get("rights_ref")) != (row["rights"], row.get("rights_ref")):
                    raise ValueError("paired representation requires matching partition and access rights")
                y = peer["value"]
                if transform == "cross_chain_differential" and peer.get("chain") == row.get("chain"):
                    raise ValueError("cross-chain representation requires distinct declared chains")
                if transform == "source_disagreement" and peer["source_id"] == row["source_id"]:
                    raise ValueError("source disagreement requires distinct observed sources")
                if transform in {"spread", "relative_value", "residual", "cross_chain_differential", "source_disagreement", "model_disagreement"}:
                    if peer["unit"] != row["unit"]: raise ValueError("difference representation requires commensurate units")
                    value = x-y
                elif transform == "ratio" and y != 0:
                    value = x/y
                    unit = "dimensionless" if row["unit"] == peer["unit"] else row["unit"]+"/("+peer["unit"]+")"
                elif transform == "actor_adjusted_value":
                    if peer["unit"] != "fraction" or not 0 <= y <= 1: raise ValueError("actor discount requires a measured fraction")
                    value = x*y
                elif transform == "state_conditioned_value":
                    if not finite(request.get("state_value")): raise ValueError("state condition needs explicit numeric state_value")
                    if y == request["state_value"]: value = x
                row = {**row, "available_at": max(row["available_at"], peer["available_at"]),
                       "observed_at": max(row["observed_at"], peer["observed_at"])}
            if value is not None and finite(value):
                if request.get("output_unit") not in {None, unit}:
                    raise ValueError("declared output unit disagrees with representation arithmetic")
                output.append({k: row.get(k) for k in ("event_at", "available_at", "observed_at", "chain", "entity_id",
                    "partition", "rights", "rights_ref") } | {"value": value, "variable": request.get("output_variable", variable+"__"+transform),
                    "source_id": "derived:"+transform, "unit": unit, "point_in_time": True})
        if not output:
            return self._blocked("representation lacks valid aligned observations", "observable:"+variable)
        self.state["representations"] += 1
        children = [{"kind": "OBSERVABLE_SERIES", "payload": {"points": output, "derivation": request,
                     "parent_refs": sorted(set(refs + [r["evidence_ref"] for r in rows+peers]))}}]
        if request.get("resume_kind"):
            children.append({"kind": "DATA_REQUEST", "payload": {**request,
                "variable": request.get("output_variable", variable+"__"+transform), "input_variable": None,
                "min_observations": 1}})
        return {"disposition": "REPRESENTATION_REQUEST", "detail": {"status": "REPRESENTATION_BUILT", "transform": transform,
            "observations": len(output), "mechanism": request["mechanism"], "falsifier": request["falsifier"], "input_refs": refs},
            "children": children}

    def report(self) -> dict:
        return {"status": "ACTIVE" if self.sources or self.state["series"] else "WAITING_FOR_OBSERVATIONS",
            "registered_sources": sorted(self.sources), "variables": {name: len(rows) for name, rows in self.state["series"].items()},
            "requests": self.state["requests"], "acquisitions": self.state["acquisitions"],
            "representations": self.state["representations"], "index_evictions": self.state["evicted_refs"],
            "raw_store": "canonical_evidence_only", "max_points_per_request": self.max_points}
