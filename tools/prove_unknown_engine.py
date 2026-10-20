"""Retain 15 isolated offline acceptance receipts, never production/live claims.

    python -m tools.prove_unknown_engine --output-dir <new-proof-directory>

The proof reuses the canonical research index, science, gauntlet, evidence,
allocator and execution gates. Unknown or missing integrations are BLOCKED.
No keys are loaded and transaction builders/submission are inert spies. All
mutable state and subprocess test artifacts stay below the output directory.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, Mock, patch

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE_TIME = 1_700_000_100.0
EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED = 0, 1, 4
SOURCE_FILES = ("tools/prove_unknown_engine.py", "src/research/unknown_engine.py",
    "src/research/frontier.py", "src/research/research_contracts.py",
    "src/research/trade_evidence.py", "src/research/gauntlet.py", "src/research/gauntlet_feed.py",
    "src/research/dataset_builder.py", "src/research/forward_evidence.py",
    "src/research/counterfactual_corpus.py", "src/research/lifecycle_replay.py",
    "src/research/benchmark_wallets.py", "src/research/promotion_gate.py",
    "src/chains/yellowstone_grpc.py", "src/strategies/champion_challenger.py",
    "src/strategies/risk_veto.py", "src/strategies/opportunity_allocator.py",
    "src/strategies/asymmetric_tail.py", "src/execution/jupiter_jito.py", "src/main.py",
    "src/research/proposal_rounds.py", "src/research/global_research_miner.py",
    "src/runtime/research_runtime.py")

REQUIREMENTS = (
    ("A01", "dynamic_chain_or_protocol_discovery"),
    ("A02", "unknown_native_event_refusal"),
    ("A03", "public_signal_to_pit_mechanism"),
    ("A04", "delayed_costly_wallet_not_followable"),
    ("A05", "frozen_candidate_genealogy_and_execution"),
    ("A06", "duplicates_and_parameter_neighbours_collapse"),
    ("A07", "planted_leakage_and_multiplicity_traps"),
    ("A08", "known_safety_veto_overrides_high_alpha"),
    ("A09", "route_aware_return_differs_from_marks"),
    ("A10", "strategy_competes_with_open_positions_and_cash"),
    ("A11", "provider_failure_preserves_queued_work"),
    ("A12", "restart_restores_debt_episodes_and_forward_evidence"),
    ("A13", "llm_outage_preserves_deterministic_operations"),
    ("A14", "no_monster_stress_respects_tail_budget"),
    ("A15", "operator_authority_required_for_live_transition"),
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str,
                               allow_nan=False) + "\n", encoding="utf-8")


class MissingProof(Exception):
    def __init__(self, reason: str, observed: dict | None = None):
        super().__init__(reason)
        self.observed = observed or {}


@contextmanager
def isolated_runtime(output: Path):
    """Relative paths and Python caches cannot escape into repository data."""
    previous_cwd, previous_cache = Path.cwd(), sys.dont_write_bytecode
    env_overrides = {"ALLOW_LIVE_TRADING": "", "PYTHONDONTWRITEBYTECODE": "1"}
    previous_env = {key: os.environ.get(key) for key in env_overrides}
    original_connect, original_dns = socket.socket.connect, socket.getaddrinfo
    def connect(sock, address):
        # Windows asyncio uses a loopback self-pipe. No provider is permitted.
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1"}:
            raise RuntimeError("acceptance proof refuses external network connections")
        return original_connect(sock, address)
    def dns(host, *args, **kwargs):
        if host not in {"127.0.0.1", "::1", None}:
            raise RuntimeError("acceptance proof refuses external DNS")
        return original_dns(host, *args, **kwargs)
    sys.path.insert(0, str(REPOSITORY))
    sys.dont_write_bytecode = True
    os.chdir(output)
    for key, value in env_overrides.items():
        os.environ[key] = value
    try:
        with patch.object(socket.socket, "connect", connect), patch.object(socket, "getaddrinfo", dns):
            yield
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.chdir(previous_cwd)
        sys.dont_write_bytecode = previous_cache
        sys.path.remove(str(REPOSITORY))


def exact_candidate(identity="fixture-candidate", *, parents=None) -> dict:
    """A finite executable fixture contract, explicitly unrelated to live edge."""
    return {
        "schema_version": 1, "candidate_id": identity, "agent_identity": "offline-fixture",
        "created_at": FIXTURE_TIME - 10, "parent_ids": parents or [], "trial_family": "fixture-flow",
        "chain": "solana", "protocol": "pump", "lifecycle_stage": "curve",
        "mechanism": "independent public buyers absorb supply", "actor_incentive": "acquire curve supply",
        "data_inputs": [{"name": "buyers", "source": "fixture-native-event", "point_in_time": True,
                         "event_timestamp": "event_at", "availability_timestamp": "available_at"}],
        "feature_program": {"buyer_count": {"op": "identity", "input": "buyers"}},
        "universe_rule": {"op": "constant", "value": True},
        "entry_horizon_seconds": 10, "max_decision_latency_ms": 250,
        "exact_entry": {"op": "gte", "field": "buyer_count", "value": 3},
        "hard_vetoes": {"op": "eq", "field": "mint_authority_active", "value": True},
        "size_rule": {"op": "fraction", "value": .001}, "fee_policy": {"round_trip_bps": 200},
        "landing_assumptions": {"mode": "fixture-replay", "fill_probability": .5},
        "exit_rule": {"op": "take_profit", "value": 1.5},
        "emergency_exit": {"op": "stop_loss", "value": .1},
        "holding_time_model": "timeout after 60s", "expected_behavior": "positive net log-growth",
        "failure_modes": ["crowding"], "falsifier": "nonpositive held-out executable return",
        "alternative_explanations": ["market beta"],
        "execution_assumptions": {"route": "fixture-curve", "buy_delay_ms": 250, "sell_delay_ms": 250},
        "novelty_signature": "fixture-independent-buyers", "source_lineage": ["fixture-public-signal"],
    }


def raw_signal(**changes) -> dict:
    return {"source_id": "fixture-public-source", "source_url": "https://example.invalid/public",
            "rights": "PUBLIC", "event_at": FIXTURE_TIME - 3, "available_at": FIXTURE_TIME - 2,
            "observed_at": FIXTURE_TIME - 1, "chain": "solana",
            "mechanism": "independent public buyers absorb supply", "candidate_features": ["buyers"],
            **changes}


def components(directory: Path):
    from src.research.frontier import FrontierIndex
    from src.research.trade_evidence import TradeEvidenceLedger
    from src.research.unknown_engine import UnknownOpportunityEngine
    from src.strategies.champion_challenger import ChampionChallengerFramework
    directory.mkdir(parents=True, exist_ok=True)
    framework = ChampionChallengerFramework(state_path=str(directory / "science.json"))
    frontier = FrontierIndex(directory / "frontier.sqlite")
    ledger = TradeEvidenceLedger(directory / "research-evidence.jsonl")
    return UnknownOpportunityEngine(framework, frontier, ledger)


def priced_job():
    from src.research.frontier import AuctionInputs
    return AuctionInputs(costs={"inference": 0, "data": 0, "engineering": 0, "conversion": 1, "gauntlet": 0})


def safety_report(**changes):
    return SimpleNamespace(**{**dict(data_status="OK", blocked_checks=[], can_mint=False,
        can_freeze=False, token_extensions=[], sell_route_feasible=True,
        checks={"sell_route": {"price_impact_pct": .01}},
        risk_level=SimpleNamespace(value="safe"), score=100), **changes})


def observations(name="fixture", count=32):
    from src.research.gauntlet import Observation, LATENCY_GRID_S
    return [Observation(mechanism=name, timestamp=FIXTURE_TIME + i, cohort=f"launch-{i}",
        regime=("up/quiet", "down/busy", "flat/frantic")[i % 3], source_family=f"source-{i % 3}",
        net_return_by_latency={delay: .2 for delay in LATENCY_GRID_S}, cost_fraction=.02)
        for i in range(count)]


def scenario_discovery(path: Path) -> dict:
    engine = components(path)
    # Identity derives from a supplied source record, not a configured chain list.
    signal = raw_signal(discovery_kind="chain", chain="chain-" + digest({"native_genesis": "fixture-unlisted"})[:16],
                        chain_family="EVM", metrics={"launches_per_day": 42})
    signal["identity"] = signal["chain"]
    job = engine.ingest(signal, now=FIXTURE_TIME)
    receipt = engine.run_pass(now=FIXTURE_TIME)
    row = engine.frontier.chain(signal["chain"])
    assert row and row["stage"] == "DISCOVERED"
    assert engine.frontier.job(job)["state"] == "ACKED"
    assert row["metrics"]["executable_liquidity"] is None
    return {"input": signal, "chain": row, "conversion": receipt,
            "path": "raw public record -> canonical evidence -> frontier discovery -> leased conversion"}


def scenario_unknown_event(path: Path) -> dict:
    from src.chains.yellowstone_grpc import PumpFunMonitor, YellowstoneClient, b58encode
    seen = []
    monitor = PumpFunMonitor(YellowstoneClient("https://example.invalid/never-connected"), seen.append)
    prefix = bytes.fromhex("aabbccddeeff0011")
    native = {"slot": 42, "transaction": {"signatures": ["fixture-signature"], "message": {
        "accountKeys": [monitor.PUMP_FUN_PROGRAM],
        "instructions": [{"programIdIndex": 0, "accounts": [], "data": b58encode(prefix + b"unknown-layout")}]}}}
    asyncio.run(monitor._on_transaction(native))
    report = monitor.decoder_report()
    assert seen == [] and report["unmatched_prefixes"][prefix.hex()] == 1
    return {"input": native, "decoder": report, "emitted_candidates": seen,
            "path": "native transaction fixture -> production PumpFunMonitor -> refusal counter; callback remains empty",
            "scope_limit": "Solana instruction fixture; no new-chain native decoder parity claimed"}


def scenario_mechanism(path: Path) -> dict:
    engine = components(path)
    signal = raw_signal()
    identity = engine.ingest(signal, now=FIXTURE_TIME)
    receipt = engine.run_pass(now=FIXTURE_TIME)
    assert receipt["results"][0]["disposition"] == "MECHANISM_HYPOTHESIS"
    assert engine.frontier.job(identity)["state"] == "ACKED"
    return {"input": signal, "conversion": receipt, "path": "public signal -> PIT validation -> mechanism receipt -> persistent conversion work"}


def scenario_wallet(path: Path) -> dict:
    from src.research.benchmark_wallets import BenchmarkCorpus, BenchmarkWallet, WalletDecision, FOLLOW_DELAYS_S
    corpus = BenchmarkCorpus(str(path / "wallet-corpus.json"), cost_per_round_trip=.02)
    corpus.register(BenchmarkWallet(address="public-fixture-wallet"))
    for i in range(60):
        corpus.record(WalletDecision(wallet="public-fixture-wallet", token=f"launch-{i}",
            entered_at=FIXTURE_TIME + i, entry_price=1, exit_price=2,
            price_at_delay={delay: 2.1 for delay in FOLLOW_DELAYS_S}))
    verdict = corpus.follow_verdict("public-fixture-wallet")
    assert verdict.wallet_mean_log_return > 0 and verdict.followable is False
    assert all(value < 0 for value in verdict.mean_log_return.values())
    assert corpus.save()
    return {"verdict": asdict(verdict), "delay_prices": {str(d): 2.1 for d in FOLLOW_DELAYS_S},
            "path": "public-wallet fixture decisions -> canonical benchmark corpus -> after-delay/fee followability rejection"}


def scenario_frozen(path: Path) -> dict:
    engine = components(path)
    proposal = exact_candidate(parents=["fixture-ancestor"])
    engine.ingest(raw_signal(exact_candidate=proposal), now=FIXTURE_TIME)
    conversion = engine.run_pass(now=FIXTURE_TIME)
    spec = engine.framework.hypotheses[proposal["candidate_id"]]
    assert spec.frozen_hash and spec.parent_ids == ["fixture-ancestor"]
    assert spec.exact_candidate["execution_assumptions"] == proposal["execution_assumptions"]
    assert engine.framework._exact_frozen_valid(spec)
    sealed = spec.frozen_hash
    spec.exact_candidate["execution_assumptions"]["buy_delay_ms"] = 0
    assert not engine.framework._exact_frozen_valid(spec)
    return {"input": proposal, "conversion": conversion, "frozen_hash": sealed,
            "post_freeze_mutation_valid": False,
            "path": "raw exact contract -> canonical framework registration -> frozen ancestry/execution seal -> mutation refusal"}


def scenario_duplicates(path: Path) -> dict:
    engine = components(path)
    first = exact_candidate()
    for candidate in (first, copy.deepcopy(first), exact_candidate("same-program-new-id")):
        engine._submit_exact(candidate, FIXTURE_TIME)
    neighbour = exact_candidate("parameter-neighbour")
    neighbour["exact_entry"]["value"] = 4
    engine._submit_exact(neighbour, FIXTURE_TIME)
    observed = {"registered_ids": sorted(engine.framework.hypotheses),
                "multiplicity_groups": dict(engine.framework.trial_ledger.multiplicity_groups),
                "same_id_result": engine._submit_exact(first, FIXTURE_TIME),
                "path": "candidate programs -> canonical hypothesis identity and multiplicity registry"}
    assert observed["same_id_result"] == "DUPLICATE"
    if not hasattr(engine.framework, "candidate_niche"):
        raise MissingProof("Executable program and numeric parameter-family niche identity not implemented.", observed)
    niches = {key: engine.framework.candidate_niche(key) for key in observed["registered_ids"]}
    observed["canonical_niches"] = niches
    original = niches[first["candidate_id"]]
    duplicate = niches["same-program-new-id"]
    neighbour_niche = niches["parameter-neighbour"]
    assert original["program_signature"] == duplicate["program_signature"]
    assert original["parameter_family_signature"] == duplicate["parameter_family_signature"] == neighbour_niche["parameter_family_signature"]
    assert original["program_signature"] != neighbour_niche["program_signature"]
    assert duplicate["disposition"] == "DUPLICATE"
    assert neighbour_niche["disposition"] == "PARAMETER_NEIGHBOUR"
    assert sum(row["contributes_independent_breadth"] for row in niches.values()) == 1
    observed["scope_limit"] = "bounded executable/parameter identity; natural-language mechanism equivalence is not guessed and alias/trial history is retained"
    return observed


def scenario_traps(path: Path) -> dict:
    from src.research.research_contracts import validate_exact_candidate
    from src.research.gauntlet import Gauntlet, Verdict
    engine = components(path)
    leaking = exact_candidate("leaking")
    leaking["data_inputs"][0]["point_in_time"] = False
    admission = validate_exact_candidate(leaking)
    assert not admission.ok
    engine.ingest(raw_signal(available_at=FIXTURE_TIME + 5), now=FIXTURE_TIME)
    future = engine.run_pass(now=FIXTURE_TIME)
    assert future["results"][0]["disposition"] == "BLOCKED"
    gauntlet = Gauntlet(min_observations=16, bootstrap=30)
    missing_variant = gauntlet.run(observations(), expected_mechanisms=["fixture", "hidden-outcome-variant"])
    assert missing_variant["fixture"].verdict is Verdict.DATA_BLOCKED
    selected = gauntlet.evaluate("fixture", observations(), pbo=.8)
    assert selected.verdict is Verdict.KILL
    return {"leakage_admission": asdict(admission), "future_signal": future,
            "hidden_variant": missing_variant["fixture"].to_dict(), "selected_champion": selected.to_dict(),
            "fixture_gauntlet_settings": {"min_observations": 16, "bootstrap": 30,
                                            "rejection_behavior_only": True},
            "path": "planted future/PIT input -> L0 refusal; incomplete selection set and PBO trap -> canonical gauntlet refusal",
            "scope_limit": "PBO=.8 is an explicit adversarial fixture, not an observed search statistic"}


def scenario_safety(path: Path, *, run_pytest: bool) -> dict:
    if not run_pytest:
        raise MissingProof("Focused production-entry integration cases were explicitly skipped; unit veto tests do not replace them.")
    target = REPOSITORY / "tests" / "test_entry_safety_authority.py"
    command = [sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider",
               "-p", "pytest_asyncio.plugin", "--basetemp", str(path / "pytest-tmp"),
               "--junitxml", str(path / "junit.xml")]
    allowed_environment = {"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "SYSTEMDRIVE"}
    env = {key: os.environ[key] for key in allowed_environment if key in os.environ}
    env.update(PYTHONPATH=str(REPOSITORY), PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               TEMP=str(path), TMP=str(path), ALLOW_LIVE_TRADING="")
    try:
        completed = subprocess.run(command, cwd=path, env=env, capture_output=True, text=True,
                                   timeout=120, check=False)
    except subprocess.TimeoutExpired as error:
        (path / "pytest-output.txt").write_text(str(error), encoding="utf-8")
        raise MissingProof("Focused integration test subprocess timed out; no successful assertion claimed.") from error
    (path / "pytest-output.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import xml.etree.ElementTree as ET
    suite = ET.parse(path / "junit.xml")
    cases = list(suite.iter("testcase"))
    assert len(cases) >= 7
    assert not any(list(suite.iter(tag)) for tag in ("failure", "error", "skipped")), "integration cases were not all executed successfully"
    return {"command": command, "command_hash": digest(command), "test_source_sha256": file_digest(target),
            "exit_code": completed.returncode, "asserted_cases": len(cases),
            "path": "real MemecoinQuantDesk._evaluate_candidate -> canonical RiskVeto -> execution spies never called",
            "evidence_level": "OFFLINE_ENTRY_INTEGRATION_WITH_MOCKED_EXTERNALS"}


def scenario_route(path: Path) -> dict:
    from src.research.lifecycle_replay import Lifecycle, Mark, replay_cell, hold_to_end
    from src.research.counterfactual_corpus import CounterfactualCorpus
    life = Lifecycle("thin-fixture", FIXTURE_TIME,
        marks=[Mark(FIXTURE_TIME, 1, executable_sol=1), Mark(FIXTURE_TIME + 10, 10, executable_sol=.01)])
    cell = replay_cell(life, 0, 1, "hold", hold_to_end)
    ideal_return = 9.0
    assert cell.ok and cell.net_sol < ideal_return
    corpus = CounterfactualCorpus(str(path / "counterfactuals.jsonl"))
    row = corpus.record("route-comparison", life.token,
        state={"entry_depth_sol": 1, "exit_depth_sol": .01}, chosen_action="enter")
    corpus.resolve("route-comparison", realised_multiple=1 + cell.net_sol,
                   peak_multiple=10)
    corpus.close()
    return {"lifecycle": asdict(life), "executable_cell": asdict(cell), "ideal_mark_return_sol": ideal_return,
            "fixture_round_trip_cost_fraction": .02,
            "path": "observed entry/exit depth fixtures -> canonical lifecycle replay -> frozen counterfactual corpus"}


def scenario_cash(path: Path) -> dict:
    from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator
    allocator = OpportunityAllocator()
    incumbent = Opportunity("held", -.2, 10, 60, 1000, is_open_position=True)
    losing = Opportunity("negative-edge", -.1, 10, 60, 1000)
    waiting = allocator.rank([incumbent, losing], cash_available_usd=500, cash_horizon_seconds=60)
    assert waiting.best is waiting.cash
    assert {item.token for item in waiting.ranked} == {"held", "negative-edge", "__CASH__"}
    assert waiting.cash.metadata["action"] == "WAIT" and waiting.displacements == []
    positive = Opportunity("positive-edge", .1, 10, 60, 1000)
    investing = allocator.rank([incumbent, positive], cash_available_usd=500, cash_horizon_seconds=60)
    assert investing.best.token == positive.token
    assert all(move.challenger.metadata.get("kind") != "CASH" for move in investing.displacements)
    return {"wait_slate": waiting.report(), "invest_slate": investing.report(),
            "wait_ranking": [item.token for item in waiting.ranked],
            "invest_ranking": [item.token for item in investing.ranked],
            "path": "held and candidate E[log W] fixtures + measured idle cash -> one canonical allocator slate -> WAIT or positive challenger"}


def scenario_outage(path: Path) -> dict:
    from src.collectors.event_source import Event, EventSource, SourceClass, SourceMesh
    class FixtureSource(EventSource):
        async def poll(self):
            if self.source_id == "unavailable-provider":
                raise RuntimeError("fixture feed/provider outage")
            return [Event(self.source_id, SourceClass.WEB, FIXTURE_TIME - 2,
                          FIXTURE_TIME - 1, text="public fixture finding")]
    sources = SourceMesh([FixtureSource("unavailable-provider", SourceClass.WEB),
                          FixtureSource("working-provider", SourceClass.WEB)])
    engine = components(path)
    raw_job = engine.ingest(raw_signal(), now=FIXTURE_TIME)
    events = asyncio.run(sources.collect(FIXTURE_TIME))
    assert [event.source_id for event in events] == ["working-provider"]
    assert engine.frontier.job(raw_job)["state"] == "READY"
    def unavailable(*args):
        raise RuntimeError("fixture provider outage before conversion")
    engine._compile = unavailable
    failed = engine.run_pass(now=FIXTURE_TIME)
    assert engine.frontier.job(raw_job)["state"] == "RETRY"
    restarted = components(path)
    before = restarted.frontier.job(raw_job)
    assert before["payload_ref"] == engine.frontier.job(raw_job)["payload_ref"]
    assert restarted.frontier.lease_jobs("early", now=FIXTURE_TIME + 30) == []
    recovered = restarted.run_pass(now=FIXTURE_TIME + 61)
    assert restarted.frontier.job(raw_job)["state"] == "ACKED"
    assert recovered["results"][0]["disposition"] == "MECHANISM_HYPOTHESIS"
    return {"source_health": sources.health(FIXTURE_TIME), "surviving_source_events": [event.to_dict() for event in events],
            "failed_pass": failed, "retry_checkpoint": before, "recovered_pass": recovered,
            "path": "canonical source mesh isolates failed provider; raw queue survives; converter failure -> RETRY -> restart -> same input consumed once"}


def scenario_restart(path: Path) -> dict:
    from src.research.dataset_builder import PointInTimeDatasetBuilder, LaunchEpisode
    from src.research.forward_evidence import ForwardEvidence, Outcome
    engine = components(path / "research")
    axes = {"chain": "fixture-chain", "representation": "source-disagreement"}
    engine.frontier.observe_coordinate(axes, state="DATA_BLOCKED", coverage_debt=4,
                                      evidence_refs=["fixture:missing-history"], now=FIXTURE_TIME)
    def builder():
        return PointInTimeDatasetBuilder(*([None] * 9), storage_path=str(path / "episodes"))
    first = builder()
    first.active_episodes["active"] = LaunchEpisode("active", "solana", FIXTURE_TIME,
        "dev", "pump", "curve", "wsol")
    first.record_market_observation("active", {"timestamp": FIXTURE_TIME + 1, "price_multiple": 1.1})
    asyncio.run(first._flush_active())
    forward = ForwardEvidence(path / "forward.json")
    forward.record(Outcome("closed", True, regime="up/quiet", realized_pnl_usd=2,
                           equity_at_decision_usd=1000))
    assert forward.save()
    second = builder()
    second._load_active_checkpoints()
    restored_forward = ForwardEvidence(path / "forward.json")
    assert restored_forward.load()
    debt = components(path / "research").frontier.coordinate(axes)["coverage_debt"]
    assert debt == 4 and second.active_episodes["active"].market_observations == first.active_episodes["active"].market_observations
    assert restored_forward.state() == forward.state()
    return {"restored_debt": debt, "restored_episode": asdict(second.active_episodes["active"]),
            "restored_forward": restored_forward.state(),
            "path": "frontier debt + active episode + forward close -> canonical checkpoints -> fresh component instances"}


def scenario_llm(path: Path) -> dict:
    from src.strategies.risk_veto import RiskVeto
    from src.research.gauntlet import Gauntlet
    from src.research.proposal_rounds import ProposalRoundRunner
    from src.research.global_research_miner import GlobalResearchMiner
    engine = components(path)
    providers = []
    for name in ("claude", "codex", "deepseek"):
        providers.append(SimpleNamespace(name=name,
            propose=AsyncMock(side_effect=RuntimeError("fixture model provider unavailable"))))
    engine.proposal_runner = ProposalRoundRunner(engine.evidence, providers, provider_timeout_seconds=1)
    miner = GlobalResearchMiner(engine.framework, unknown_engine=engine,
                                ledger_path=str(path / "public-leads.jsonl"))
    signal = raw_signal(research_partition="training", summary="Public independent buyer mechanism fixture")
    async def fixture_collection(*args):
        engine.ingest(signal)
    miner._mine_github = AsyncMock(side_effect=fixture_collection)
    miner._mine_arxiv = AsyncMock()
    miner._mine_rss = AsyncMock()
    asyncio.run(miner.run_once())
    assert miner._mine_github.await_count > 0
    assert all(provider.propose.await_count == 1 for provider in providers)
    report = engine.last_proposal_round
    assert report["status"] == "DATA_BLOCKED"
    assert all(row["status"] == "DATA_BLOCKED" for row in report["providers"])
    collected = engine.last_pass
    risk = RiskVeto().evaluate(safety_report(can_mint=True))
    validation = Gauntlet(bootstrap=20).run(observations(count=200))
    assert collected["processed"] == 1 and risk.status == "VETO" and validation
    return {"proposal_provider_outage": report, "deterministic_conversion": collected, "risk": risk.to_dict(),
            "validation": {key: value.to_dict() for key, value in validation.items()},
            "fixture_gauntlet_bootstrap": 20,
            "path": "GlobalResearchMiner.run_once fixture collection -> UnknownOpportunityEngine conversion -> all configured proposal providers fail and seal -> deterministic RiskVeto/Gauntlet remain operational",
            "scope_limit": "actual provider boundary with injected failures; no paid provider or production outage exercised"}


def scenario_tail(path: Path) -> dict:
    from src.strategies.asymmetric_tail import AsymmetricTailSleeve
    from src.strategies.opportunity_allocator import Opportunity
    sleeve = AsymmetricTailSleeve(10_000)
    for day in range(365):
        now = FIXTURE_TIME + day * 86400
        candidates = [Opportunity(f"loss-{day}-{i}", .01, 10, 60, 1000,
            metadata={"chain": "solana", "actor_id": str(i), "narrative_id": str(i),
                      "safety_clear": True, "exit_ready": True}, elogw_at=lambda size: size * .001)
            for i in range(30)]
        for item in sleeve.propose(candidates, now=now).proposed:
            assert sleeve.record_shadow_entry(item, now=now)
            assert sleeve.record_shadow_exit(item.token, fraction=1, proceeds_usd=0, now=now)
        report = sleeve.report(now=now)
        assert report["gross_loss_usd"] + report["open_risk_usd"] <= report["budget_usd"] + 1e-8
        assert report["daily_loss_usd"] <= 50 + 1e-8 and report["weekly_loss_usd"] <= 150 + 1e-8
    final = sleeve.report(now=FIXTURE_TIME + 365 * 86400)
    assert final["status"] == "SHUTDOWN" and final["gross_loss_usd"] == 300
    write_json(path / "tail-checkpoint.json", sleeve.state())
    return {"days_without_monster": 365, "terminal_equity_usd": 9700, "report": final,
            "path": "canonical allocator proposals -> recorded full-loss shadow fills -> daily/weekly/lifetime shutdown",
            "scope_limit": "one deterministic adverse path proves its budget bound; ruin probability remains UNMEASURED"}


def scenario_authority(path: Path) -> dict:
    from collections import defaultdict
    from src.execution.jupiter_jito import ExecutionEngine, RouteType, TransactionStatus
    from src.research.promotion_gate import PromotionLedger
    ledger = PromotionLedger(path / "promotion.jsonl")
    assert not ledger.authorises_live_capital()[0]
    results = []
    async def check():
        for venue in ("aggregator", "pump"):
            for earned in (False, True):
                engine = ExecutionEngine.__new__(ExecutionEngine)
                engine.dry_run = False
                engine.promotion_ledger = (SimpleNamespace(authorises_live_capital=lambda: (True, "fixture earned stage"))
                                           if earned else ledger)
                engine.native_route_attempts = defaultdict(int)
                engine._record = Mock()
                engine._build_native_signed = AsyncMock(return_value="UNSIGNED_FIXTURE_BYTES")
                engine._submit_signed = AsyncMock(side_effect=AssertionError("submission forbidden in acceptance"))
                quote = SimpleNamespace(output_amount=123, route_type=RouteType.JUPITER_V1)
                engine.jupiter = SimpleNamespace(get_quote=AsyncMock(return_value=quote),
                    get_swap_transaction=AsyncMock(side_effect=AssertionError("build forbidden")))
                engine.tx_builder = SimpleNamespace(sign_versioned_transaction=AsyncMock(
                    side_effect=AssertionError("signing forbidden")))
                engine.prepare_native_route = (lambda *args: SimpleNamespace(ok=True, venue="pump")) if venue == "pump" else lambda *args: None
                engine._native_quote = lambda *args: quote
                for side in ("buy", "sell"):
                    first, second = ("wsol", "token") if side == "buy" else ("token", "wsol")
                    result = await engine.execute_swap(first, second, 100, use_jito=False)
                    assert result.simulated or result.status is TransactionStatus.REJECTED
                    engine._submit_signed.assert_not_awaited()
                    engine.tx_builder.sign_versioned_transaction.assert_not_awaited()
                    results.append({"venue": venue, "side": side, "earned_stage_fixture": earned,
                                    "status": result.status.value, "simulated": result.simulated, "error": result.error})
        return results
    asyncio.run(check())
    return {"attempts": results, "operator_ack_present": False, "submissions": 0, "real_signatures": 0,
            "path": "actual execute_swap buy/sell -> native or aggregator branch -> canonical stage gate and separate operator-ack gate",
            "scope_limit": "earned stage is a test double, transaction build is an inert spy; no live stage was granted"}


def run_proof(output_dir: str | Path | None = None, *, run_pytest: bool = True) -> dict:
    output = Path(output_dir).resolve() if output_dir else Path(tempfile.mkdtemp(prefix="memecoin-unknown-proof-")).resolve()
    if output in {REPOSITORY, REPOSITORY / "data"}:
        raise ValueError("choose a dedicated acceptance output directory")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("proof output already contains a manifest; choose a fresh directory")
    scenarios: list[Callable] = [scenario_discovery, scenario_unknown_event, scenario_mechanism,
        scenario_wallet, scenario_frozen, scenario_duplicates, scenario_traps,
        lambda path: scenario_safety(path, run_pytest=run_pytest), scenario_route, scenario_cash,
        scenario_outage, scenario_restart, scenario_llm, scenario_tail, scenario_authority]
    started = time.time()
    source_hashes_start = {name: file_digest(REPOSITORY / name) for name in SOURCE_FILES}
    with isolated_runtime(output):
        from src.research.trade_evidence import TradeEvidenceLedger
        evidence = TradeEvidenceLedger(output / "acceptance-evidence.jsonl")
        records = []
        for (requirement, name), scenario in zip(REQUIREMENTS, scenarios):
            directory = output / f"{requirement}-{name}"
            directory.mkdir()
            input_record = {"requirement": requirement, "scenario": name, "fixture_clock": FIXTURE_TIME,
                            "scope": "ISOLATED_OFFLINE_FIXTURE_RUNTIME", "production_evidence": "UNMEASURED"}
            write_json(directory / "input.json", input_record)
            status, gap = "PASS", None
            try:
                observed = scenario(directory)
            except MissingProof as error:
                status, gap, observed = "BLOCKED", str(error), error.observed
            except Exception as error:
                status, gap, observed = "FAIL", f"{type(error).__name__}: {error}", {}
            record = {**input_record, "status": status, "gap": gap, "observed": observed,
                      "input_sha256": digest(input_record), "result_sha256": digest(observed),
                      "provider_runtime_evidence": "UNMEASURED", "untouched_forward_evidence": "UNMEASURED",
                      "live_capability": False}
            row = evidence.record("unknown_engine_acceptance", record)
            if row is None:
                raise RuntimeError("canonical acceptance evidence write failed")
            record["canonical_evidence_ref"] = f"{evidence.path.name}#sequence={row['sequence']};sha256={row['hash']}"
            write_json(directory / "result.json", record)
            record["artifact"] = str((directory / "result.json").relative_to(output))
            records.append(record)
        verified, detail, count = evidence.verify(evidence.path)
        assert verified, detail
    files = {str(path.relative_to(output)): {"sha256": file_digest(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()}
    counts = {status: sum(row["status"] == status for row in records) for status in ("PASS", "FAIL", "BLOCKED")}
    source_hashes_end = {name: file_digest(REPOSITORY / name) for name in SOURCE_FILES}
    changed_sources = sorted(name for name in SOURCE_FILES
                             if source_hashes_start[name] != source_hashes_end[name])
    manifest = {"schema": "unknown-engine-acceptance-v1", "output_dir": str(output),
        "scope": "OFFLINE_FIXTURE_RUNTIME_ONLY", "production_status": "UNMEASURED",
        "provider_status": "UNMEASURED", "forward_status": "UNMEASURED", "live_capability": False,
        "started_at": started, "completed_at": time.time(), "counts": counts,
        "python_version": sys.version, "python_executable": sys.executable,
        "source_sha256": source_hashes_start, "source_sha256_at_end": source_hashes_end,
        "sources_changed_during_run": changed_sources,
        "status": "FAIL" if counts["FAIL"] else "BLOCKED" if counts["BLOCKED"] or changed_sources else "PASS",
        "canonical_evidence_chain": {"verified": verified, "records": count, "detail": detail},
        "requirements": [{key: row[key] for key in ("requirement", "scenario", "status", "gap", "artifact", "canonical_evidence_ref")} for row in records],
        "files": files}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-entry-pytest", action="store_true",
                        help="skip subprocess entry checks; A08 remains BLOCKED")
    args = parser.parse_args(argv)
    result = run_proof(args.output_dir, run_pytest=not args.skip_entry_pytest)
    print(json.dumps({key: result[key] for key in ("output_dir", "status", "counts", "production_status")}, indent=2))
    return EXIT_FAIL if result["status"] == "FAIL" else EXIT_BLOCKED if result["status"] == "BLOCKED" else EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
