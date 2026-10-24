"""Bounded exact-program adapter to the desk's existing replay and gauntlet.

This module grants no promotion. It refuses assumptions the native SOL replay
cannot represent; unsupported chains, rules, missing trial ancestors, costs,
fills or chronological partitions remain DATA_BLOCKED.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.gauntlet import Gauntlet, Observation, Verdict
from src.research.gauntlet_feed import CONTROL, EpisodeView, GauntletFeed, Mechanism, SNAPSHOT_KEY_BY_OFFSET
from src.research.lifecycle_replay import ExitRule, Mark, replay_cell
from src.research.research_contracts import canonical_candidate_identity, validate_exact_candidate
from src.strategies.champion_challenger import HypothesisSpec

FEATURE_GROUPS = {"deployer_features", "wallet_features", "flow_features", "liquidity_features",
                  "social_features", "token_features", "market_features", "entity_graph_features"}
COMPARISONS = {"gt", "gte", "lt", "lte", "eq", "ne", "in"}


class ReplayBlocked(ValueError):
    pass


def native_replay_identity(chain: Any, factory: Any) -> tuple[str, str] | None:
    """Recognize retained producer labels without rewriting sealed episodes.

    The desk stores its configured chain display name and decoded program ID.
    Only the existing reviewed Pump adapter is mapped; other native programs
    still require their own replay implementation and execution evidence.
    """
    if not isinstance(chain, str) or not isinstance(factory, str) or chain not in {"solana", "Solana Mainnet", "Solana"}:
        return None
    from src.chains.idl import IDL_STATUS, PUMP_IDL, program_id
    protocols = {"pump", "pump.fun"}
    if IDL_STATUS == "OK":
        protocols.add(program_id(PUMP_IDL))
    return ("solana", "pump") if factory in protocols else None


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def evaluate_rule(rule: Mapping, values: Mapping[str, Any], *, _depth: int = 0) -> bool | None:
    """Three-valued bounded rules; an absent observable is never False/zero."""
    if _depth > 12 or not isinstance(rule, Mapping):
        raise ReplayBlocked("rule nesting or shape unsupported")
    op = rule.get("op")
    if op == "constant" and type(rule.get("value")) is bool:
        return rule["value"]
    if op in {"all", "any"}:
        args = rule.get("args")
        if not isinstance(args, list) or not 0 < len(args) <= 32:
            raise ReplayBlocked("bounded boolean arguments required")
        results = [evaluate_rule(arg, values, _depth=_depth + 1) for arg in args]
        if op == "all":
            return False if False in results else None if None in results else True
        return True if True in results else None if None in results else False
    if op == "not":
        result = evaluate_rule(rule.get("arg"), values, _depth=_depth + 1)
        return None if result is None else not result
    if op not in COMPARISONS or not isinstance(rule.get("field"), str) or "value" not in rule:
        raise ReplayBlocked("unsupported entry/universe/veto operation")
    left, right = values.get(rule["field"]), rule["value"]
    if left is None:
        return None
    if isinstance(left, float) and not math.isfinite(left):
        return None
    try:
        if op == "eq": return left == right
        if op == "ne": return left != right
        if op == "in": return left in right
        if not _finite(left) or not _finite(right): return None
        if op == "gt": return left > right
        if op == "gte": return left >= right
        if op == "lt": return left < right
        if op == "lte": return left <= right
    except (TypeError, ValueError):
        return None
    return None


def _check_boolean(rule: Mapping) -> None:
    # Validate all branches, including ones a short circuit would not visit.
    evaluate_rule(rule, {})


def _exit_predicate(rule: Mapping, mark: Mark, entry_multiple: float, entry_time: float,
                    depth: int = 0) -> bool:
    if depth > 12 or not isinstance(rule, Mapping):
        raise ReplayBlocked("unsupported exit shape")
    op = rule.get("op")
    if op in {"all", "any"}:
        args = rule.get("args")
        if not isinstance(args, list) or not 0 < len(args) <= 32:
            raise ReplayBlocked("bounded exit arguments required")
        values = [_exit_predicate(arg, mark, entry_multiple, entry_time, depth + 1) for arg in args]
        return all(values) if op == "all" else any(values)
    if op == "not":
        return not _exit_predicate(rule.get("arg"), mark, entry_multiple, entry_time, depth + 1)
    if op == "constant" and type(rule.get("value")) is bool:
        return rule["value"]
    threshold = rule.get("value")
    if not _finite(threshold, positive=True):
        raise ReplayBlocked("finite positive exit threshold required")
    if op == "take_profit":
        return mark.multiple / entry_multiple >= threshold
    if op == "stop_loss" and threshold < 1:
        return mark.multiple / entry_multiple <= 1 - threshold
    if op == "timeout":
        return mark.timestamp - entry_time >= threshold
    raise ReplayBlocked("unsupported exact exit; no fallback substituted")


def compile_exit(candidate: Mapping, *, entry_time: float) -> ExitRule:
    rules = (candidate["exit_rule"], candidate["emergency_exit"])
    for rule in rules:
        _exit_predicate(rule, Mark(entry_time, 1.0), 1.0, entry_time)

    def exit_rule(marks: Sequence[Mark], entry_multiple: float) -> Mark | None:
        for mark in marks:
            if mark.feasible and any(_exit_predicate(rule, mark, entry_multiple, entry_time) for rule in rules):
                return mark
        # A take-profit never reached is not permission to invent a terminal exit.
        return None
    return exit_rule


@dataclass(frozen=True)
class CompiledCandidate:
    candidate: dict
    mechanism: Mechanism
    size_native: float


def compile_candidate(candidate: Mapping, *, mechanism_name: str | None = None) -> CompiledCandidate:
    admission = validate_exact_candidate(candidate)
    if not admission.ok:
        raise ReplayBlocked("L0: " + "; ".join(admission.reasons))
    candidate = copy.deepcopy(dict(candidate))
    if candidate["chain"] != "solana" or candidate["protocol"] not in {"pump", "pump.fun"}:
        raise ReplayBlocked("native replay adapter unavailable for this chain/protocol")
    execution = candidate["execution_assumptions"]
    offset = execution.get("decision_offset_seconds")
    equity = execution.get("equity_native")
    if (not _finite(offset) or offset not in SNAPSHOT_KEY_BY_OFFSET
            or offset > candidate["entry_horizon_seconds"] or execution.get("native_unit") != "SOL"
            or not _finite(equity, positive=True)):
        raise ReplayBlocked("explicit supported decision offset, SOL unit and replay equity required")
    size = candidate["size_rule"]
    if size.get("op") != "fraction":
        raise ReplayBlocked("only explicit fraction sizing is supported")
    fee_cap = candidate["fee_policy"].get("max_round_trip_cost_fraction")
    if not _finite(fee_cap) or not 0 <= fee_cap < 1:
        raise ReplayBlocked("explicit round-trip fractional fee cap required")
    for field in ("universe_rule", "exact_entry", "hard_vetoes"):
        _check_boolean(candidate[field])
    compile_exit(candidate, entry_time=1.0)
    for item in candidate["data_inputs"]:
        if item.get("snapshot_group") not in FEATURE_GROUPS or not isinstance(item.get("snapshot_key"), str):
            raise ReplayBlocked("data input requires allowlisted snapshot_group/snapshot_key mapping")
        if any(marker in item["snapshot_key"].lower() for marker in ("label", "future", "outcome", "realized", "max_multiple")):
            raise ReplayBlocked("future/outcome fields cannot become replay features")

    def select(view: EpisodeView) -> bool | None:
        identity = native_replay_identity(view.raw.get("chain"), view.raw.get("factory"))
        if identity is None or identity != native_replay_identity(candidate["chain"], candidate["protocol"]):
            return False
        row = view.snapshot(offset)
        if not isinstance(row, dict):
            return None
        cutoff = view.created_at + offset
        values = {}
        for item in candidate["data_inputs"]:
            event_at, available_at = row.get(item["event_timestamp"]), row.get(item["availability_timestamp"])
            if (not _finite(event_at) or not _finite(available_at) or event_at > cutoff
                    or available_at > cutoff or available_at < event_at):
                return None
            block = row.get(item["snapshot_group"])
            value = block.get(item["snapshot_key"]) if isinstance(block, dict) else None
            if value is None or isinstance(value, (dict, list)):
                return None
            values[item["name"]] = value
        for name, expression in candidate["feature_program"].items():
            op = expression["op"]
            if op == "identity":
                values[name] = values[expression["input"]]
            else:
                left, right = values[expression["left"]], values[expression["right"]]
                if not _finite(left) or not _finite(right) or (op == "ratio" and right == 0):
                    return None
                values[name] = left / right if op == "ratio" else left - right
        universe = evaluate_rule(candidate["universe_rule"], values)
        veto = evaluate_rule(candidate["hard_vetoes"], values)
        entry = evaluate_rule(candidate["exact_entry"], values)
        if veto is True or universe is False or entry is False:
            return False
        if None in (universe, veto, entry):
            return None
        return True
    mechanism = Mechanism(mechanism_name or candidate["candidate_id"], select, float(offset), candidate["trial_family"])
    return CompiledCandidate(candidate, mechanism, equity * size["value"])


class _ExactFeed(GauntletFeed):
    """Keep canonical observation construction; price each observed fee/fill cell."""
    def __init__(self, compiled: CompiledCandidate, execution: dict, delays: Sequence[float]):
        self.compiled, self.execution = compiled, execution
        super().__init__(mechanisms=[compiled.mechanism], delays=delays, size_sol=compiled.size_native,
                         exit_rule_name="exact-" + compiled.mechanism.name, round_trip_cost=0)

    def _returns_by_latency(self, lifecycle: Any, decision_offset_s: float) -> dict:
        results = {}
        for delay in self.delays:
            cell = self.execution["fills_by_latency"].get(str(float(delay)))
            if not isinstance(cell, dict):
                raise ReplayBlocked("missing measured latency-specific fill/cost evidence")
            if (cell.get("evidence_kind") not in {"native_simulation", "observed_transaction"}
                    or not cell.get("source") or type(cell.get("filled")) is not bool
                    or type(cell.get("route_feasible")) is not bool):
                raise ReplayBlocked("explicit native fill/route evidence required")
            cost = cell.get("round_trip_cost_fraction")
            failed_cost = cell.get("failed_cost_fraction")
            if not _finite(cost) or not 0 <= cost < 1 or not _finite(failed_cost) or failed_cost < 0:
                raise ReplayBlocked("observed successful and failed-transaction costs required")
            if delay * 1000 > self.compiled.candidate["max_decision_latency_ms"]:
                results[delay] = None
                continue
            if (not cell["filled"] or not cell["route_feasible"]
                    or cost > self.compiled.candidate["fee_policy"]["max_round_trip_cost_fraction"]):
                results[delay] = -failed_cost
                continue
            entry_time = lifecycle.created_at + decision_offset_s + delay
            latest = lifecycle.marks_upto(decision_offset_s + delay)
            if not latest or not latest[-1].feasible:
                results[delay] = None
                continue
            rule = compile_exit(self.compiled.candidate, entry_time=entry_time)
            priced = replay_cell(lifecycle, decision_offset_s + delay, self.size_sol,
                                 self.exit_rule_name, rule, cost)
            results[delay] = priced.net_sol / priced.filled_sol if priced.ok and priced.filled_sol > 0 else None
        return results

    def _for_episode(self, view: EpisodeView) -> list[Observation]:
        rows = super()._for_episode(view)
        # Gauntlet cost stress needs observed cost, not the base class's default.
        worst_cost = max(cell["round_trip_cost_fraction"] for cell in self.execution["fills_by_latency"].values())
        for row in rows:
            row.chain = self.compiled.candidate["chain"]
            row.cost_fraction = worst_cost
        return rows


class CandidateReplayHandler:
    def __init__(self, framework: Any, episode_path: str | Path, *, chronological_split: Mapping | None = None,
                 gauntlet: Gauntlet | None = None, max_episodes: int = 2000,
                 max_family_variants: int = 32, max_bootstrap_draws: int = 20_000_000):
        self.framework, self.episode_path = framework, Path(episode_path)
        self.split, self.gauntlet = chronological_split, gauntlet or Gauntlet()
        self.max_episodes, self.max_family_variants = max_episodes, max_family_variants
        self.max_bootstrap_draws = max_bootstrap_draws

    def _family(self, candidate: Mapping) -> list[tuple[str, HypothesisSpec]]:
        family = candidate["trial_family"]
        ids = set(self.framework.registry.families.get(family, []))
        ids.update(self.framework.trial_ledger.multiplicity_groups.get(family, []))
        member_ids = set(ids)
        specs = list(self.framework.hypotheses.values())
        specs += [HypothesisSpec(**row) for row in self.framework.frozen_specs.values()]
        specs += [HypothesisSpec(**row["spec"]) for row in self.framework.registry.graveyard]
        selected = {}
        for spec in specs:
            if spec.trial_family != family and spec.hypothesis_id not in member_ids:
                continue
            if not self.framework._exact_frozen_valid(spec):
                raise ReplayBlocked("trial family contains legacy, nonexact, unfrozen or changed ancestry")
            selected[spec.frozen_hash] = spec
            ids.discard(spec.hypothesis_id)
        if ids or not selected or len(selected) > self.max_family_variants:
            raise ReplayBlocked("trial-family history unavailable or exceeds explicit variant budget")
        for trial in self.framework.trial_ledger.trials:
            if ((trial.get("family") == family or trial.get("hypothesis_id") in member_ids)
                    and trial.get("spec_hash") not in selected):
                raise ReplayBlocked("outcome-inspecting trial lacks its frozen strategy version")
        return [(spec.hypothesis_id + "@" + seal, spec) for seal, spec in sorted(selected.items())]

    def _episodes(self) -> list[dict]:
        if isinstance(self.split, Mapping) and "files" in self.split:
            from src.research.replay_corpus import load_sealed_corpus
            rows = load_sealed_corpus(self.episode_path, self.split, max_episodes=self.max_episodes)
            if any(row.get("final_outcome", {}).get("status") != "OK" for row in rows):
                raise ReplayBlocked("sealed corpus contains unresolved launches")
            return rows
        paths = sorted(self.episode_path.glob("*/*.json.gz"))
        if not paths or len(paths) > self.max_episodes:
            raise ReplayBlocked("episode corpus absent or exceeds explicit work budget")
        rows = []
        for path in paths:
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    row = json.load(handle)
            except (OSError, ValueError, EOFError) as exc:
                raise ReplayBlocked("unreadable corpus object; no silent cohort deletion") from exc
            if not isinstance(row, dict) or row.get("final_outcome", {}).get("status") != "OK":
                raise ReplayBlocked("corpus contains unresolved launches")
            rows.append(row)
        return sorted(rows, key=lambda row: (row.get("created_at", 0), row.get("token", "")))

    @staticmethod
    def _execution(episode: dict, candidate: dict) -> dict:
        execution = episode.get("replay_execution")
        identity = native_replay_identity(episode.get("chain"), episode.get("factory"))
        if (identity is None or identity != native_replay_identity(candidate["chain"], candidate["protocol"])
                or not isinstance(execution, dict) or execution.get("native_unit") != "SOL"
                or not execution.get("route") or not isinstance(execution.get("fills_by_latency"), dict)):
            raise ReplayBlocked("native chain/protocol/route/fill metadata absent or mismatched")
        marks = episode.get("market_observations")
        if not isinstance(marks, list) or not marks:
            raise ReplayBlocked("native executable marks unavailable")
        for mark in marks:
            if (not isinstance(mark, dict) or mark.get("route") != execution["route"]
                    or not _finite(mark.get("timestamp")) or not _finite(mark.get("available_at"))
                    or mark["available_at"] > mark["timestamp"]
                    or not _finite(mark.get("executable_sol")) or mark["executable_sol"] < 0
                    or not _finite(mark.get("price_multiple"), positive=True)
                    or type(mark.get("feasible")) is not bool):
                raise ReplayBlocked("mark lacks observed native route/depth/PIT/feasibility")
        return execution

    def __call__(self, candidate: Mapping) -> dict:
        receipt = {"schema_version": 1, "candidate_id": candidate.get("candidate_id"),
                   "scope": "L0-L4_ONLY", "authority": "none", "promotion": False,
                   "required_next_stage": "L5_UNTOUCHED_FORWARD", "stages": []}
        try:
            admission = validate_exact_candidate(candidate)
            if not admission.ok:
                raise ReplayBlocked("L0 contract admission failed: " + "; ".join(admission.reasons))
            current = self.framework.hypotheses.get(candidate["candidate_id"])
            if (current is None or not self.framework._exact_frozen_valid(current)
                    or current.exact_candidate != dict(candidate)):
                raise ReplayBlocked("candidate is not the exact registered frozen program")
            receipt.update(candidate_hash=admission.candidate_hash, spec_hash=current.frozen_hash)
            receipt["stages"].append("L0_SCHEMA_PIT_PROGRAM")
            family = self._family(candidate)
            compiled = [(name, compile_candidate(spec.exact_candidate, mechanism_name=name)) for name, spec in family]
            target = current.hypothesis_id + "@" + current.frozen_hash
            receipt["expected_variants"] = [name for name, _ in compiled]
            receipt["stages"].append("L1_FULL_FROZEN_FAMILY")
            episodes = self._episodes()
            corpus_hash = _digest(episodes)
            split = self.split
            if (not isinstance(split, Mapping) or split.get("corpus_hash") != corpus_hash
                    or not all(_finite(split.get(key)) for key in ("train_end", "validation_end", "test_end", "frozen_at"))
                    or not split["train_end"] < split["validation_end"] < split["test_end"]
                    or split["frozen_at"] > min(spec.frozen_at for _, spec in family)):
                raise ReplayBlocked("content-bound chronological train/validation/test split was not frozen before candidates")
            partitions = {"train": [], "validation": [], "test": []}
            for episode in episodes:
                created = episode.get("created_at")
                if not _finite(created, positive=True) or created > split["test_end"]:
                    raise ReplayBlocked("episode lies outside frozen chronological corpus")
                name = "train" if created <= split["train_end"] else "validation" if created <= split["validation_end"] else "test"
                end = split["train_end"] if name == "train" else split["validation_end"] if name == "validation" else split["test_end"]
                if any(mark.get("timestamp", float("inf")) > end for mark in episode.get("market_observations", [])):
                    raise ReplayBlocked("launch outcome overlaps the next chronological partition")
                partitions[name].append(episode)
            if any(not rows for rows in partitions.values()):
                raise ReplayBlocked("chronological train/validation/test evidence is incomplete")
            estimate = len(episodes) * len(compiled) * self.gauntlet.bootstrap * (len(self.gauntlet.latencies) + len(self.gauntlet.cost_multipliers) + 8)
            if estimate > self.max_bootstrap_draws:
                raise ReplayBlocked("canonical gauntlet exceeds declared bootstrap work budget")
            receipt.update(corpus_hash=corpus_hash, split=dict(split), estimated_bootstrap_draws=estimate,
                           partition_counts={key: len(rows) for key, rows in partitions.items()})
            program_signatures = {name: canonical_candidate_identity(program.candidate)["program_signature"]
                                  for name, program in compiled}
            receipt["validation_work"] = {"raw_frozen_versions": len(compiled),
                "unique_executable_programs": len(set(program_signatures.values())),
                "duplicate_replays_reused": len(compiled) - len(set(program_signatures.values()))}
            receipt["stages"].append("L2_NATIVE_COST_FILL_FEASIBILITY")
            results, coverage = {}, {}
            for partition, rows in partitions.items():
                observations = []
                coverage[partition] = {}
                replay_cache = {}
                for mechanism_name, program in compiled:
                    signature = program_signatures[mechanism_name]
                    if signature in replay_cache:
                        cached = replay_cache[signature]
                        for row in cached:
                            if row.mechanism == CONTROL.name:
                                if mechanism_name == target:
                                    observations.append(replace(row))
                            else:
                                observations.append(replace(row, mechanism=mechanism_name))
                        coverage[partition][mechanism_name] = {"reused_program_signature": signature}
                        continue
                    counters = []
                    variant_observations = []
                    for episode in rows:
                        execution = self._execution(episode, program.candidate)
                        feed = _ExactFeed(program, execution, self.gauntlet.latencies)
                        built = feed.observations([episode])
                        counters.append(feed.coverage())
                        variant_observations.extend(built)
                        observations.extend(row for row in built if row.mechanism != CONTROL.name or mechanism_name == target)
                    replay_cache[signature] = variant_observations
                    coverage[partition][mechanism_name] = counters
                judged = self.gauntlet.run(observations, expected_mechanisms=[name for name, _ in compiled])
                results[partition] = {name: result.to_dict() for name, result in judged.items()}
                if target not in judged:
                    raise ReplayBlocked("target has no executable selected observations")
            receipt.update(results=results, coverage=coverage)
            receipt["stages"].extend(["L3_CHRONOLOGICAL_CANONICAL_GAUNTLET", "L4_NATIVE_EXECUTION_REPLAY"])
            verdicts = [result[target]["verdict"] for result in results.values()]
            if any(verdict == Verdict.DATA_BLOCKED.value for verdict in verdicts):
                raise ReplayBlocked("canonical gauntlet lacks sufficient chronological/selection evidence")
            if any(verdict != Verdict.SURVIVOR.value for verdict in verdicts):
                return {"disposition": "NEGATIVE_KNOWLEDGE", "status": "REJECTED", "receipt": receipt, "authority": "none"}
            return {"disposition": "DEEPENING_TASK", "status": "HISTORICAL_SURVIVOR", "receipt": receipt, "authority": "none"}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            receipt["reason"] = str(exc)
            return {"disposition": "BLOCKED", "status": "DATA_BLOCKED", "receipt": receipt, "authority": "none"}
