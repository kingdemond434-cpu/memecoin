"""Exercise the real research-loop seam with no network/signing authority."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.chains.evm_events import NativeChainEvent
from src.chains.rpc_manager import ChainType
from src.research.frontier import FrontierIndex
from src.research.global_research_miner import GlobalResearchMiner
from src.research.trade_evidence import TradeEvidenceLedger
from src.runtime.wiring import SubsystemWiring


class Desk(SubsystemWiring):
    pass


class HTTP:
    def __init__(self, calls):
        self.calls = calls

    async def get(self, url):
        self.calls.append("public-discovery")
        return 200, json.dumps([{"chainId": "new-unlisted-chain", "tokenAddress": "public-token"}]), {}


def desk_at(tmp_path, calls=None):
    desk = Desk()
    desk.offline = False
    desk.global_config = {"chain_scout_pair_budget": 0}
    desk.http_client = HTTP(calls if calls is not None else [])
    desk.research_frontier = FrontierIndex(tmp_path / "frontier.sqlite")
    desk.trade_evidence = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    return desk


def test_production_loop_drains_after_discovery_and_between_public_collection(tmp_path):
    async def scenario():
        calls = []
        desk = desk_at(tmp_path, calls)
        await desk._setup_opportunity_observation(tmp_path)
        async def proposals():
            calls.append("proposals")
        engine = SimpleNamespace(run_pass=lambda: calls.append("conversion"), run_proposal_pass=proposals,
                                 report=lambda: {"status": "SHADOW_ONLY"})
        miner = GlobalResearchMiner(SimpleNamespace(), unknown_engine=engine,
                                    chain_scout=desk.chain_opportunity_scout,
                                    evm_observers=desk.evm_observers, native_event_sink=desk._retain_native_observation)
        async def research(*args):
            calls.append("public-research")
        miner._mine_github = miner._mine_arxiv = miner._mine_rss = research
        await miner.run_once()
        assert calls[0] == "public-discovery"
        assert calls.index("public-discovery") < calls.index("conversion") < calls.index("public-research") < calls.index("proposals")
        assert calls.count("conversion") > 1
        assert calls[-1] == "proposals"
        assert desk.research_frontier.chain("new-unlisted-chain")["family"] == "UNKNOWN"
        assert desk.research_frontier.report()["cost_blocked"] >= 1
        assert miner.get_stats()["multi_chain_observation"]["evm"][0]["status"] == "NOT_CONFIGURED"
        intact, _, _ = TradeEvidenceLedger.verify(desk.trade_evidence.path)
        assert intact
    asyncio.run(scenario())


def test_failed_canonical_append_cannot_register_a_chain(tmp_path):
    desk = desk_at(tmp_path)
    desk.trade_evidence = SimpleNamespace(record_reference=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="canonical"):
        desk._retain_chain_scout_record({"kind": "source_response", "observed_at": 100, "raw_body": "[]"})
    assert desk.research_frontier.report()["coordinates"] == 0


def test_native_event_uses_common_event_ledger_and_never_solanas_episode_or_execution(tmp_path):
    desk = desk_at(tmp_path)
    native = NativeChainEvent(
        chain="configured-name", chain_id="1", chain_family="EVM", protocol="factory",
        protocol_version="UNKNOWN", event_type="UNKNOWN", block_number=100,
        block_hash="0x" + "ab" * 32, transaction_hash="0x" + "cd" * 32, event_index=0,
        source_at=100, observed_at=200, finality="FINALIZED", source_ref="chain:1:block:100",
        native={"log": {"data": "uninterpreted"}})
    def forbidden(*args, **kwargs):
        raise AssertionError("native research observation reached Solana execution/dataset")
    desk.dataset_builder = SimpleNamespace(record_market_observation=forbidden)
    desk.execution_engine = SimpleNamespace(execute_swap=forbidden)
    ref = desk._retain_native_observation(native)
    assert desk._retain_native_observation(native) == ref
    assert desk.trade_evidence.sequence == 1
    assert desk.research_frontier.chain("1")["stage"] == "OBSERVATION_ONLY"
    payload = json.loads(desk.trade_evidence.path.read_text())["payload"]
    assert payload["source_class"] == "chain"
    assert payload["metadata"]["event_type"] == "UNKNOWN"
    assert payload["metadata"]["execution_eligible"] is False
    assert desk.research_frontier.job(f"native-observation-{native.event_id}")["total_cost"] is None


def test_optional_evm_setup_reuses_configured_manager_and_default_decoder_stays_unknown(tmp_path):
    async def scenario():
        desk = desk_at(tmp_path)
        config = SimpleNamespace(chain_id=1, chain_type=ChainType.EVM,
                                 factories={"uniswap_v2": "0x" + "f1" * 20})
        rpc = SimpleNamespace()
        desk.chain_registry = SimpleNamespace(get_chain=lambda name: config, get_rpc=lambda name: rpc)
        desk.global_config["evm_observation"] = [{"chain": "ethereum", "start_block": 100}]
        await desk._setup_opportunity_observation(tmp_path)
        assert len(desk.evm_observers) == 1
        assert desk.evm_observers[0].rpc is rpc
        assert next(iter(desk.evm_observers[0].factories.values())).decoder == "UNKNOWN"
        assert desk.evm_observation_configuration[0]["status"] == "CONFIGURED_NOT_OBSERVED"
        assert desk.research_frontier.report()["coordinates"] == 0
    asyncio.run(scenario())


def test_offline_setup_and_invalid_observer_config_are_truthful(tmp_path):
    async def scenario():
        desk = desk_at(tmp_path)
        desk.offline = True
        await desk._setup_opportunity_observation(tmp_path)
        assert desk.chain_opportunity_scout is None and not desk.evm_observers
        assert desk.evm_observation_configuration[0]["status"] == "DATA_BLOCKED"
        desk.offline = False
        desk.global_config["evm_observation"] = [{"chain": "missing-start-block"}]
        await desk._setup_opportunity_observation(tmp_path)
        assert not desk.evm_observers
        assert desk.evm_observation_configuration[0]["status"] == "DATA_BLOCKED"
    asyncio.run(scenario())


def test_native_probe_configuration_has_no_four_route_universe_cap(tmp_path):
    async def scenario():
        desk = desk_at(tmp_path)
        config = SimpleNamespace(chain_id=1, chain_type=ChainType.EVM)
        rpc = SimpleNamespace()
        desk.chain_registry = SimpleNamespace(get_chain=lambda name: config, get_rpc=lambda name: rpc)
        route = {"chain": "ethereum", "factory": "0x" + "11" * 20,
                 "router": "0x" + "22" * 20, "token_in": "0x" + "33" * 20,
                 "token_out": "0x" + "44" * 20, "amount_in": 100,
                 "source_ref": "fixture:official-deployment"}
        desk.global_config["evm_native_probes"] = [dict(route) for _ in range(9)]
        await desk._setup_opportunity_observation(tmp_path)
        assert len(desk.evm_native_probes) == 9
        assert all(row["status"] == "CONFIGURED_NOT_OBSERVED" for row in desk.evm_native_configuration)
        assert all(probe.rpc is rpc for probe in desk.evm_native_probes)
    asyncio.run(scenario())


def test_optional_startup_overlaps_chains_isolates_failures_and_reuses_managers(tmp_path):
    async def scenario():
        desk = desk_at(tmp_path)
        configs = {name: SimpleNamespace(chain_id=i + 1, chain_type=ChainType.EVM,
                   factories={"uniswap_v2": "0x" + "f1" * 20})
                   for i, name in enumerate(("slow", "fast", "failed"))}
        managers, calls = {}, []
        fast_ready = asyncio.Event()

        async def start(name):
            calls.append(name)
            if name == "slow":
                await asyncio.wait_for(fast_ready.wait(), 2)
            elif name == "fast":
                fast_ready.set()
            else:
                raise ConnectionError("fixture provider URL must stay private")
            managers[name] = SimpleNamespace()
        desk.chain_registry = SimpleNamespace(get_chain=configs.get, get_rpc=managers.get, start_chain=start)
        desk.global_config["evm_observation"] = [
            {"chain": name, "start_block": 100} for name in ("slow", "fast", "failed", "slow")]
        await desk._setup_opportunity_observation(tmp_path)
        assert calls.count("slow") == 1
        assert len(desk.evm_observers) == 3
        blocked = [row for row in desk.evm_observation_configuration if row["status"] == "DATA_BLOCKED"]
        assert blocked == [{"chain": "failed", "status": "DATA_BLOCKED", "failure_type": "ConnectionError"}]
        assert "provider URL" not in json.dumps(desk.evm_observation_configuration)
    asyncio.run(scenario())
