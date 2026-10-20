"""Bounded training-only program search; canonical freezing precedes scoring.

Algorithms here propose experiments. They never inspect held-out/forward rows,
issue certificates, schedule work or alter capital. The caller supplies the
existing framework-backed freeze/record callbacks and keeps all trial ancestry.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SEARCH_METHODS = ("grammar", "symbolic_regression", "genetic_programming", "decision_tree",
                  "state_machine", "bayesian_optimization", "novelty", "map_elites",
                  "causal_discovery", "graph_search", "anomaly_change_point", "enumeration")
TARGET_SEMANTICS = "net_executable_log_growth_after_costs"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class SearchDataError(ValueError):
    pass


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def training_batch(data: Mapping, contracts: Mapping, *, now: float, max_rows: int = 512,
                   max_features: int = 8) -> dict:
    def contaminated(value):
        if isinstance(value, Mapping):
            return any(any(marker in str(key).lower() for marker in
                       ("heldout", "held_out", "forward", "agent_opinion")) or contaminated(item)
                       for key, item in value.items())
        return isinstance(value, list) and any(contaminated(item) for item in value)
    if contaminated(data):
        raise SearchDataError("heldout/forward/opinion contamination")
    if not isinstance(data, Mapping) or data.get("partition") != "training":
        raise SearchDataError("explicit training partition required; heldout/forward rows forbidden")
    if data.get("target_semantics") != TARGET_SEMANTICS:
        raise SearchDataError("measured executable after-cost training target required")
    rows = data.get("rows")
    if not isinstance(rows, list) or not 8 <= len(rows) <= max_rows:
        raise SearchDataError("training row count outside declared 8..max_rows bound")
    names = sorted(contracts)
    if not 1 <= len(names) <= max_features:
        raise SearchDataError("observable count outside declared feature bound")
    for name in names:
        descriptor = contracts[name]
        if (not isinstance(name, str) or not isinstance(descriptor, Mapping)
                or any(marker in name.lower() for marker in ("future", "label", "outcome", "realized", "final_"))
                or descriptor.get("dtype") not in {"float", "int", "bool"}
                or descriptor.get("point_in_time") is not True or not descriptor.get("unit")
                or not isinstance(descriptor.get("source"), str) or not descriptor["source"].strip()):
            raise SearchDataError("typed, named, PIT observable contracts required")
    identities, x, y = set(), [], []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("partition", "training") != "training":
            raise SearchDataError("mixed or contaminated training partitions")
        if any(key in row for key in ("heldout", "forward", "forward_outcomes", "agent_opinions")):
            raise SearchDataError("heldout/forward/opinion contamination")
        rid = row.get("row_id")
        if not isinstance(rid, str) or not rid or rid in identities:
            raise SearchDataError("distinct stable training row identities required")
        identities.add(rid)
        if not isinstance(row.get("episode_id"), str) or not row["episode_id"]:
            raise SearchDataError("stable episode identity required for temporal/state controls")
        times = [row.get(key) for key in ("event_at", "available_at", "decision_at", "outcome_at")]
        if (not all(finite(value) for value in times)
                or not times[0] <= times[1] <= times[2] <= times[3] <= now):
            raise SearchDataError("training event/availability/decision/outcome PIT order invalid")
        if any(row.get(key) is not True for key in ("costs_observed", "route_observed", "fill_observed")):
            raise SearchDataError("unmeasured costs/routes/fills cannot become training returns")
        features = row.get("features")
        if (not isinstance(features, Mapping) or not finite(row.get("target"))
                or abs(row["target"]) > 1e150):
            raise SearchDataError("finite measured training features/target required")
        values = []
        for name in names:
            value, dtype = features.get(name), contracts[name]["dtype"]
            if (dtype == "bool" and type(value) is not bool
                    or dtype == "int" and type(value) is not int
                    or dtype == "float" and not finite(value)):
                raise SearchDataError("missing or mistyped observable; never substitute zero")
            if abs(float(value)) > 1e150:
                raise SearchDataError("observable exceeds finite numerical work range")
            values.append(float(value))
        x.append(values)
        y.append(row["target"])
    order = sorted(range(len(rows)), key=lambda index: (rows[index]["decision_at"], rows[index]["row_id"]))
    return {"rows": [rows[index] for index in order], "names": names,
            "x": np.asarray(x)[order], "y": np.asarray(y)[order],
            "batch_hash": digest(data)}


@dataclass(frozen=True)
class SearchBudget:
    max_rows: int = 512
    max_features: int = 8
    max_trials: int = 48
    max_depth: int = 3
    max_graph_edges: int = 128

    def __post_init__(self):
        bounds = ((self.max_rows, 8, 512), (self.max_features, 1, 8), (self.max_trials, 1, 96),
                  (self.max_depth, 1, 3), (self.max_graph_edges, 1, 256))
        if any(type(value) is not int or not low <= value <= high for value, low, high in bounds):
            raise ValueError("search work exceeds declared hard bounds")


class MachineSearch:
    def __init__(self, *, budget: SearchBudget | None = None):
        self.budget = budget or SearchBudget()

    def run(self, data: Mapping, contracts: Mapping, *, now: float,
            freeze: Callable[[dict], str], record: Callable[[str, dict], None],
            methods: Sequence[str] = SEARCH_METHODS, archive: Mapping | None = None) -> dict:
        batch = training_batch(data, contracts, now=now, max_rows=self.budget.max_rows,
                               max_features=self.budget.max_features)
        if not methods or len(set(methods)) != len(methods) or set(methods) - set(SEARCH_METHODS):
            raise SearchDataError("distinct supported search methods required")
        names, x, y = batch["names"], batch["x"], batch["y"]
        evaluated, requests, diagnostics, program_cache = [], [], {}, {}
        prior = dict(archive or {})
        reserve = max(1, self.budget.max_trials // len(methods))

        def values(expression: dict) -> np.ndarray:
            if expression["op"] == "identity": return x[:, names.index(expression["input"])]
            left, right = x[:, names.index(expression["left"])], x[:, names.index(expression["right"])]
            if expression["op"] == "difference": return left - right
            if np.any(right == 0): raise SearchDataError("ratio denominator observed at zero")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = left / right
            if not np.all(np.isfinite(result)) or np.any(np.abs(result) > 1e150):
                raise SearchDataError("derived observable outside finite numerical domain")
            return result

        def mask_of(rule: dict, signals: dict) -> np.ndarray:
            op = rule["op"]
            if op == "all": return np.logical_and.reduce([mask_of(arg, signals) for arg in rule["args"]])
            if op == "any": return np.logical_or.reduce([mask_of(arg, signals) for arg in rule["args"]])
            signal = signals[rule["field"]]
            return signal >= rule["value"] if op == "gte" else signal <= rule["value"]

        def attempt(method: str, expression: dict, threshold: float, *, op="gte", parents=(), rule=None, programs=None) -> dict | None:
            if len(evaluated) >= self.budget.max_trials:
                return None
            feature_program = programs or {"signal": expression}
            entry = rule or {"op": op, "field": "signal", "value": float(threshold)}
            def depth(node):
                return 1 + max((depth(arg) for arg in node.get("args", [])), default=0)
            if depth(entry) > self.budget.max_depth:
                return None
            program = {"method": method, "feature_program": feature_program, "entry_rule": entry,
                       "search_parents": list(parents), "training_batch_hash": batch["batch_hash"]}
            program["program_id"] = digest(program)
            if program["program_id"] in program_cache:
                return program_cache[program["program_id"]]
            # Freeze callback must complete before any target-based score,
            # regression fit, quality-diversity update or outcome selection.
            frozen_id = freeze(program)
            signals = {name: values(expr) for name, expr in feature_program.items()}
            selected = mask_of(entry, signals)
            outcomes = y[selected]
            score = float(np.mean(outcomes)) if outcomes.size else None
            lower = (float(np.mean(outcomes) - np.std(outcomes) / math.sqrt(outcomes.size)) if outcomes.size else None)
            result = {**program, "hypothesis_id": frozen_id, "selected_count": int(outcomes.size),
                      "training_score": score, "training_lower_proxy": lower,
                      "coverage": float(np.mean(selected)), "scope": "TRAINING_ONLY_NO_EDGE_VERDICT"}
            record(frozen_id, result)
            evaluated.append(result)
            program_cache[program["program_id"]] = result
            return result

        def base_grid() -> list[tuple[dict, float, str]]:
            return [({"op": "identity", "input": name}, float(np.quantile(x[:, index], q)), op)
                    for index, name in enumerate(names) for q in (0.25, 0.5, 0.75) for op in ("gte", "lte")]

        grid = base_grid()
        for method in methods:
            start = len(evaluated)
            if method in {"grammar", "enumeration"}:
                candidates = grid if method == "grammar" else list(reversed(grid))
                for expression, threshold, op in candidates[:reserve]:
                    attempt(method, expression, threshold, op=op)
            elif method == "symbolic_regression":
                expressions = [{"op": "identity", "input": name} for name in names]
                expressions += [{"op": op, "left": left, "right": right}
                                for left, right in itertools.combinations(names, 2) for op in ("difference", "ratio")]
                fits = []
                for expression in expressions[:reserve]:
                    try: signal = values(expression)
                    except SearchDataError: continue
                    result = attempt(method, expression, float(np.median(signal)))
                    if result is None: break
                    design = np.column_stack((np.ones(len(signal)), signal))
                    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
                    fits.append({"hypothesis_id": result["hypothesis_id"], "expression": expression,
                        "coefficients": coefficients.tolist(), "training_mse": float(np.mean((design @ coefficients - y) ** 2))})
                diagnostics[method] = {"fits": fits, "interpretation": "training least-squares symbolic hypotheses, not causal or OOS proof"}
            elif method in {"genetic_programming", "decision_tree"}:
                parents = sorted([row for row in evaluated if row["training_score"] is not None],
                                 key=lambda row: (-row["training_score"], row["program_id"]))[:reserve + 1]
                if not parents:
                    expression, threshold, op = grid[0]
                    parent = attempt(method, expression, threshold, op=op)
                    parents = [parent] if parent else []
                for index, parent in enumerate(parents[:reserve]):
                    other = parents[(index + 1) % len(parents)]
                    programs, rules = {}, []
                    for slot, ancestor in enumerate((parent, other)):
                        # Existing fitted predicates are exact ASTs. Rename
                        # signals before crossover, never eval generated text.
                        rename = {name: f"b{slot}_{name}" for name in ancestor["feature_program"]}
                        programs.update({rename[name]: expr for name, expr in ancestor["feature_program"].items()})
                        def rename_rule(node):
                            return {key: rename.get(value, value) if key == "field" else
                                [rename_rule(arg) for arg in value] if key == "args" else value for key, value in node.items()}
                        rules.append(rename_rule(ancestor["entry_rule"]))
                    if len(programs) > self.budget.max_features: continue
                    combined = {"op": "all" if method == "decision_tree" else "any", "args": rules}
                    if self.budget.max_depth < 2: continue
                    attempt(method, {"op": "identity", "input": names[0]}, 0,
                        parents=[parent["hypothesis_id"], other["hypothesis_id"]], rule=combined, programs=programs)
                diagnostics[method] = {"procedure": "bounded conjunction leaf induction" if method == "decision_tree" else "bounded AST crossover",
                                       "maximum_depth": self.budget.max_depth}
            elif method == "state_machine":
                signal, median = x[:, 0], float(np.median(x[:, 0]))
                ordered = sorted(range(len(batch["rows"])), key=lambda i: batch["rows"][i]["decision_at"])
                transitions = Counter()
                for previous, current in zip(ordered, ordered[1:]):
                    if batch["rows"][previous].get("episode_id") == batch["rows"][current].get("episode_id"):
                        transitions[(int(signal[previous] >= median), int(signal[current] >= median))] += 1
                requests.append({"kind": "DATA_REQUEST", "variable": "prior_state_" + names[0],
                    "mechanism": "observable state persistence or transition before decision",
                    "construction": {"source": names[0], "threshold": median,
                        "states": ["below_training_median", "at_or_above_training_median"],
                        "transition_counts": [{"from": a, "to": b, "count": count} for (a, b), count in sorted(transitions.items())]},
                    "pit_requirements": "last available observation in same episode strictly before current decision",
                    "falsifier": "transition state adds no executable OOS value beyond current level"})
            elif method in {"bayesian_optimization", "novelty", "map_elites"}:
                chosen = set()
                acquisitions = []
                for _ in range(reserve):
                    candidates = [(i, item) for i, item in enumerate(grid) if i not in chosen]
                    if not candidates: break
                    history = [row for row in evaluated if row["training_score"] is not None]
                    def acquisition(item):
                        index, (expression, threshold, op) = item
                        signal = values(expression)
                        coverage = float(np.mean(signal >= threshold if op == "gte" else signal <= threshold))
                        descriptor = np.array([coverage, names.index(expression["input"]) / max(1, len(names)), float(op == "gte")])
                        points = np.asarray([[row["coverage"], 0, float(row["entry_rule"].get("op") == "gte")] for row in history])
                        if method == "bayesian_optimization" and len(history) >= 2:
                            kernel = np.exp(-np.sum((points[:, None] - points[None, :]) ** 2, axis=2)) + np.eye(len(points)) * 1e-6
                            cross = np.exp(-np.sum((points - descriptor) ** 2, axis=1))
                            mean = cross @ np.linalg.solve(kernel, np.asarray([row["training_score"] for row in history]))
                            variance = max(0.0, 1 - cross @ np.linalg.solve(kernel, cross))
                            return float(mean + math.sqrt(variance)), -index
                        distances = [float(np.linalg.norm(descriptor - point)) for point in points]
                        if method == "map_elites":
                            key = f"coverage:{min(3,int(coverage*4))}/rule:{op}/features:1"
                            return (1.0 if key not in prior.get("map_elites", {}) else 0.0), -index
                        if method == "novelty":
                            distances.extend(float(np.linalg.norm(descriptor - np.asarray(point)))
                                             for point in prior.get("novelty", {}).values())
                        return (min(distances) if distances else 1.0), -index
                    index, (expression, threshold, op) = max(candidates, key=acquisition)
                    acquisitions.append({"grid_index": index, "acquisition": acquisition((index, grid[index]))[0]})
                    chosen.add(index)
                    outcome = attempt(method, expression, threshold, op=op)
                    if method == "map_elites" and outcome is not None and outcome["training_lower_proxy"] is not None:
                        key = f"coverage:{min(3,int(outcome['coverage']*4))}/rule:{op}/features:1"
                        prior.setdefault("map_elites", {})[key] = {"hypothesis_id": outcome["hypothesis_id"],
                            "training_quality": outcome["training_lower_proxy"], "training_batch_hash": batch["batch_hash"]}
                diagnostics[method] = {"acquisitions": acquisitions,
                    "procedure": "RBF Gaussian-process upper-confidence acquisition" if method == "bayesian_optimization" else "behavior-distance exploration" if method == "novelty" else "coverage/direction/observable quality-diversity niches"}
            elif method == "causal_discovery":
                conditional = []
                if len(names) < 2:
                    requests.append({"kind": "DATA_REQUEST", "variable": "pre_treatment_confounder",
                        "mechanism": "distinguish public mechanism from common-cause market regime",
                        "pit_requirements": "measured before treatment and decision", "falsifier": "effect vanishes under matched pre-treatment conditioning"})
                else:
                    for sign in ("gte", "lte")[:reserve]:
                        rule = {"op": "all", "args": [
                            {"op": "gte", "field": "treatment", "value": float(np.median(x[:, 0]))},
                            {"op": sign, "field": "confounder", "value": float(np.median(x[:, 1]))}]}
                        result = attempt(method, {"op": "identity", "input": names[0]}, 0, rule=rule,
                            programs={"treatment": {"op": "identity", "input": names[0]}, "confounder": {"op": "identity", "input": names[1]}})
                        if result: conditional.append({"hypothesis_id": result["hypothesis_id"], "stratum": sign, "training_conditional_mean": result["training_score"]})
                    diagnostics[method] = {"conditional_hypotheses": conditional,
                        "identification": "UNIDENTIFIED: requires negative controls/intervention; observational conditioning is not causal proof"}
            elif method == "graph_search":
                graph = data.get("graph")
                if not isinstance(graph, Mapping) or not isinstance(graph.get("edges"), list):
                    requests.append({"kind": "DATA_REQUEST", "variable": "public_actor_temporal_edges",
                        "mechanism": "compress linked addresses before independent actor counting",
                        "pit_requirements": "edge event and availability times before decision; lawful public provenance",
                        "falsifier": "graph-adjusted signal does not beat degree-preserving shuffled controls"})
                else:
                    edges = graph["edges"]
                    if len(edges) > self.budget.max_graph_edges: raise SearchDataError("graph work bound exceeded")
                    adjacency = {}
                    for edge in edges:
                        if (not isinstance(edge, Mapping) or edge.get("partition") != "training" or not edge.get("source")
                                or not finite(edge.get("event_at")) or not finite(edge.get("available_at"))
                                or not edge["event_at"] <= edge["available_at"] <= now
                                or not isinstance(edge.get("from"), str) or not isinstance(edge.get("to"), str)):
                            raise SearchDataError("graph edge provenance/PIT invalid")
                        adjacency.setdefault(edge["from"], set()).add(edge["to"])
                        adjacency.setdefault(edge["to"], set()).add(edge["from"])
                    unseen, components = set(adjacency), []
                    while unseen:
                        root = min(unseen); queue = deque([root]); component = set()
                        while queue:
                            node = queue.popleft()
                            if node in component: continue
                            component.add(node); queue.extend(sorted(adjacency[node] - component))
                        unseen -= component; components.append(sorted(component))
                    requests.append({"kind": "DATA_REQUEST", "variable": "actor_component_size",
                        "mechanism": "shared temporal ancestry discounts apparent independent conviction",
                        "construction": {"algorithm": "bounded undirected ancestry BFS", "components": components},
                        "pit_requirements": "reconstruct components as of each decision, never using later edges",
                        "falsifier": "linked-address compression fails degree-preserving shuffled-graph control"})
            elif method == "anomaly_change_point":
                for index, name in enumerate(names[:reserve]):
                    signal = x[:, index]
                    center = float(np.median(signal[:max(2, len(signal)//2)]))
                    cumulative = np.cumsum(signal - center)
                    point = int(np.argmax(np.abs(cumulative)))
                    low, high = float(np.quantile(signal, 0.1)), float(np.quantile(signal, 0.9))
                    result = attempt(method, {"op": "identity", "input": name}, high,
                        rule={"op": "any", "args": [{"op": "gte", "field": "signal", "value": high}, {"op": "lte", "field": "signal", "value": low}]})
                    diagnostics.setdefault(method, {"changes": []})["changes"].append({"observable": name,
                        "training_change_index": point, "cumulative_shift": float(cumulative[point]),
                        "hypothesis_id": result["hypothesis_id"] if result else None})
            diagnostics.setdefault(method, {})["outcome_inspecting_trials"] = len(evaluated) - start
        elites = dict(prior.get("map_elites", {}))
        for row in evaluated:
            key = f"coverage:{min(3,int(row['coverage']*4))}/rule:{row['entry_rule']['op']}/features:{len(row['feature_program'])}"
            old = elites.get(key)
            quality = row["training_lower_proxy"]
            if quality is not None and (old is None or quality > old["training_quality"]):
                elites[key] = {"hypothesis_id": row["hypothesis_id"], "training_quality": quality,
                               "training_batch_hash": batch["batch_hash"]}
        novelty = dict(prior.get("novelty", {}))
        for row in evaluated:
            novelty[row["hypothesis_id"]] = [row["coverage"], 0, float(row["entry_rule"].get("op") == "gte")]
        evicted = list(novelty)[:-128]
        novelty = {key: value for key, value in list(novelty.items())[-128:]}
        return {"status": "TRAINING_PROPOSALS_ONLY", "training_batch_hash": batch["batch_hash"],
            "programs": evaluated, "requests": requests, "methods": diagnostics,
            "archive": {"map_elites": elites, "novelty": novelty}, "work_bound": self.budget.__dict__,
            "priority_archive_evictions": evicted,
            "budget_exhausted": len(evaluated) >= self.budget.max_trials,
            "edge_evidence": "UNMEASURED"}
