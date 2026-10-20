"""Actual native quote arithmetic flows into persistent paper-only experiments."""
import copy
from types import SimpleNamespace

import pytest

from src.chains.pump_curve import BondingCurveState
from src.research.native_forward import NativeForwardPaper
from src.research.research_validation import ResearchValidationService
from src.runtime.research_runtime import ResearchRuntime
from test_candidate_replay import episode
from test_research_validation import desk, register


def fresh_episode(now):
    row = episode(0)
    row["created_at"] = now - .25
    row["snapshots"]["t0"].update(event_at=now - .25, available_at=now - .25)
    row.pop("final_outcome")
    row["market_observations"] = []
    return row


def curve():
    return BondingCurveState(100_000_000_000, 30_000_000_000,
        70_000_000_000, 10_000_000_000, 100_000_000_000, False)


def costs(desk, now, *, phase="entry", amount=100_000_000):
    row = {"origin": "canonical_native_simulation", "network_fee_native": .000005,
           "failed_cost_native": 0., "observed_at": now,
           "token": "mint-0", "phase": phase, "input_amount": amount,
           "route": "pump_curve",
           "simulation": {"err": None, "unitsConsumed": 50000},
           "simulation_ref": "offline-native-rpc-simulation-fixture"}
    ref = desk[4].record_reference("research_native_simulation_cost", row, timestamp=now)
    return {**row, "evidence_ref": ref}


def tick(runner, desk, row, state, now, **override):
    position = next(iter(runner.state["positions"].values()), None)
    kwargs = {"state_observed_at": now, "source_ref": "native-account:offline-control",
        "equity_usd": 1000., "native_to_usd": 100., "now": now,
        "fee_bps": 100, "execution_costs": costs(desk, now,
            phase="exit" if position else "entry", amount=position["tokens"] if position else 100_000_000)}
    kwargs.update(override)
    return runner.tick(row["token"], row, state, **kwargs)


def test_native_quote_entry_and_later_exit_reach_bound_paper_consumer(desk):
    _, _, _, validation, _, forward = desk
    _, now = register(desk)
    now += 1
    runner = NativeForwardPaper(validation)
    row, state = fresh_episode(now), curve()
    entry = tick(runner, desk, row, state, now)
    assert entry["disposition"] == "DEEPENING_TASK", entry
    assert entry["detail"]["open_positions"] == 1
    # The canonical timeout rule exits against a fresh second native state,
    # rather than a historical peak, router quote or fabricated model return.
    exit_result = tick(runner, desk, row, state, now + 3)
    assert exit_result["disposition"] == "DEEPENING_TASK", exit_result
    receipt = exit_result["detail"]["results"][0]
    assert receipt["scope"] == "observed_forward_shadow" and receipt["real_fill"] is False
    assert receipt["pnl"] < 0  # unchanged market actually loses roundtrip fees/impact
    assert runner.report()["open_positions"] == 0 and validation.report()["paper_outcomes"] == 1
    assert not forward.real_fills and not desk[0].canary_models and not desk[0].champions
    tick(runner, desk, row, state, now + 4)
    assert validation.report()["paper_outcomes"] == 1


def test_native_paper_position_survives_restart_without_duplicate_entry(desk):
    framework, _, handler, validation, ledger, forward = desk
    _, now = register(desk)
    now += 1
    row, state = fresh_episode(now), curve()
    first = NativeForwardPaper(validation)
    assert tick(first, desk, row, state, now)["detail"]["open_positions"] == 1
    from src.strategies.champion_challenger import ChampionChallengerFramework
    framework2 = ChampionChallengerFramework(state_path=framework.state_path)
    handler.framework = framework2
    validation2 = ResearchValidationService(framework2, handler, forward).configure_evidence(ledger)
    second = NativeForwardPaper(validation2)
    assert second.report()["open_positions"] == 1
    result = tick(second, desk, row, state, now + 3)
    assert result["disposition"] == "DEEPENING_TASK", result
    assert second.report()["open_positions"] == 0 and validation2.report()["paper_outcomes"] == 1


@pytest.mark.parametrize("missing", ["state_time", "real_depth", "fees", "simulation_cost", "future_inputs", "equity", "aggregator"])
def test_missing_native_paper_inputs_make_named_requests_and_never_fill(desk, missing):
    _, _, _, validation, _, _ = desk
    _, now = register(desk)
    now += 1
    runner = NativeForwardPaper(validation)
    row, state = fresh_episode(now), curve()
    override = {}
    if missing == "state_time": override["state_observed_at"] = now - 31
    if missing == "real_depth": state.real_sol_reserves = 0
    if missing == "fees": override["fee_bps"] = None
    if missing == "simulation_cost": override["execution_costs"] = None
    if missing == "future_inputs": row["snapshots"]["t0"]["available_at"] = now + 1
    if missing == "equity": override["equity_usd"] = None
    if missing == "aggregator": state = SimpleNamespace(output_amount=999, route="jupiter")
    result = tick(runner, desk, row, state, now, **override)
    assert result["disposition"] == "BLOCKED", result
    assert result["children"][0]["kind"] == "DATA_REQUEST"
    assert "enabling_change" in result["children"][0]
    assert not runner.report()["open_positions"] and not validation.report()["paper_outcomes"]


def test_old_episode_cannot_be_relabelled_untouched_forward(desk):
    _, _, _, validation, _, _ = desk
    _, now = register(desk)
    runner = NativeForwardPaper(validation)
    result = tick(runner, desk, episode(0), curve(), now)
    assert result["detail"]["status"] == "NO_REGISTERED_FRESH_COHORT"
    assert not runner.report()["open_positions"]


def test_runtime_native_seam_retains_request_in_existing_queue(desk, monkeypatch):
    _, _, _, validation, _, _ = desk
    _, now = register(desk)
    now += 1
    row = fresh_episode(now)
    state = curve()
    row["market_observations"] = [{"signature": "offline-native-trade-control", "timestamp": now,
        "virtual_sol_reserves": state.virtual_sol_reserves,
        "virtual_token_reserves": state.virtual_token_reserves}]
    queued = []
    app = ResearchRuntime()
    app.unknown_engine = SimpleNamespace(workflows=SimpleNamespace(validation=validation),
        enqueue_work=lambda kind, payload, **kwargs: queued.append((kind, payload)))
    app.dataset_builder = SimpleNamespace(active_episodes={row["token"]: row}, _episode_to_dict=lambda item: copy.deepcopy(item))
    app._latest_curve_state = {row["token"]: state}
    app._latest_pool_state = {}
    app.trade_evidence = desk[4]
    app.equity_status, app.wallet_equity_usd, app.sol_price_usd = "OK", 1000., 100.
    app.sol_price_age_s = 0.
    app._cost_model = lambda token, at: {"status": "OK", "assumed": False, "entry_cost": .01}
    import asyncio
    result = asyncio.run(app._native_forward_tick(row["token"], now))
    assert result["disposition"] == "BLOCKED" and queued[0][0] == "DATA_REQUEST"
    # The registered async acquisition path supplies a canonical receipt;
    # config values without a canonical receipt are rejected by the producer.
    async def acquired(self, token, amount, *, buying, now):
        return costs(desk, now, phase="entry" if buying else "exit", amount=amount)
    monkeypatch.setattr("src.research.native_forward.NativeSimulationFeeAdapter.acquire", acquired)
    result = asyncio.run(app._native_forward_tick(row["token"], now))
    assert result["disposition"] == "DEEPENING_TASK", result
    assert app.native_forward_paper.report()["open_positions"] == 1


def test_unsigned_native_fee_adapter_uses_actual_message_and_never_touches_signer(desk):
    import asyncio
    import base64
    from solders.hash import Hash
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    from src.chains.pump_route import NativePumpRoute
    from src.execution.jupiter_jito import ExecutionEngine, SolanaTransactionBuilder
    from src.execution.tx_kernel import TxKernel
    from src.research.native_forward import NativeSimulationFeeAdapter
    payer = str(Pubkey.new_unique())
    mint, creator = str(Pubkey.new_unique()), str(Pubkey.new_unique())
    calls = []
    class PublicOnlySigner:
        public_key = payer
        async def sign_message(self, message):
            raise AssertionError("unsigned simulation must never invoke signer")
    class RPC:
        async def request(self, method, params):
            calls.append((method, params))
            if method == "getLatestBlockhash": return {"value": {"blockhash": str(Hash.new_unique())}}
            if method == "simulateTransaction":
                transaction = VersionedTransaction.from_bytes(base64.b64decode(params[0]))
                assert all(bytes(signature) == bytes(64) for signature in transaction.signatures)
                assert params[1]["sigVerify"] is False
                return {"value": {"err": None, "unitsConsumed": 50000, "logs": ["offline-native-simulation"]}}
            if method == "getFeeForMessage": return {"value": 5000}
            raise AssertionError("unexpected RPC or submission: " + method)
    rpc = RPC()
    builder = SolanaTransactionBuilder(rpc, PublicOnlySigner())
    builder.tx_kernel = TxKernel(mode="solders")
    engine = ExecutionEngine(SimpleNamespace(), rpc, None, None, builder, None,
                             dry_run=True, pump_route=NativePumpRoute())
    state = curve()
    state.creator = creator
    engine.curve_state_provider = lambda token: state if token == mint else None
    assert callable(engine.prepare_native_route) and not hasattr(engine, "_prepare_native_route")
    assert engine.native_compute_unit_limit == 400000
    assert engine.tx_builder is builder and engine.rpc is rpc
    now = desk[1].frozen_at + 20
    adapter = NativeSimulationFeeAdapter(engine, desk[4])
    receipt = asyncio.run(adapter.acquire(mint, 100000000, buying=True, now=now))
    assert receipt["network_fee_native"] == .000005
    assert receipt["phase"] == "entry" and receipt["input_amount"] == 100000000
    assert receipt["evidence_ref"].startswith("evidence:")
    assert {method for method, _ in calls} == {"getLatestBlockhash", "simulateTransaction", "getFeeForMessage"}
    assert desk[3].read_evidence(receipt["evidence_ref"])["payload"]["message_hash"] == receipt["message_hash"]
