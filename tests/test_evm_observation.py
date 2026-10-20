"""Official ABI parity plus synthetic chain/reorg controls; no RPC network."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from eth_abi import encode
import pytest

from src.chains.evm_events import (
    NativeChainEvent, PAIR_CREATED_TOPIC, decode_pair_created, select_quote_pair,
)
from src.chains.evm_observation import EVMObservationAdapter, FactoryContract
from src.chains.rpc_manager import ChainType
from src.detection.token_detector import FactoryPoller, MempoolDetector


FACTORY = "0x" + "f1" * 20
TOKEN0 = "0x" + "11" * 20
TOKEN1 = "0x" + "22" * 20
PAIR = "0x" + "33" * 20
ABI_FIXTURE = "tests/fixtures/uniswap_v2_pair_created_abi.json"


def block_hash(height, variant=0):
    return "0x" + f"{variant:02x}" + f"{height:062x}"


def header(height, variant=0, parent=None):
    return {"number": hex(height), "hash": block_hash(height, variant),
            "parentHash": parent or block_hash(height - 1), "timestamp": hex(1700000000 + height)}


def log_for(block, index=0):
    """Invented local values encoded with the official ABI fixture."""
    return {"address": FACTORY, "topics": [PAIR_CREATED_TOPIC,
            "0x" + encode(["address"], [TOKEN0]).hex(), "0x" + encode(["address"], [TOKEN1]).hex()],
            "data": "0x" + encode(["address", "uint256"], [PAIR, 1]).hex(),
            "blockNumber": block["number"], "blockHash": block["hash"],
            "transactionHash": "0x" + f"{int(block['number'], 16):064x}",
            "logIndex": hex(index), "removed": False}


def contract(known=True):
    return FactoryContract(
        "uniswap_v2", FACTORY, version="official-abi-test-v1",
        decoder="uniswap_v2_pair_created" if known else "UNKNOWN",
        abi_ref=ABI_FIXTURE if known else None,
        source_ref="https://github.com/Uniswap/v2-core/blob/master/contracts/interfaces/IUniswapV2Factory.sol" if known else None)


class FakeRPC:
    def __init__(self, first=100, last=102):
        self.blocks = {n: header(n) for n in range(first - 1, last + 1)}
        self.logs = {block["hash"]: [log_for(block)] for block in self.blocks.values()}
        self.requests = []
        self.fail_logs = False

    async def request(self, method, params):
        self.requests.append((method, copy.deepcopy(params)))
        if method == "eth_chainId":
            return "0x1"
        if method == "eth_getBlockByNumber":
            height = max(self.blocks) if params[0] in ("safe", "finalized") else int(params[0], 16)
            return copy.deepcopy(self.blocks.get(height))
        if method == "eth_getLogs":
            if self.fail_logs:
                raise RuntimeError("provider https://secret.invalid/token was unavailable")
            if "blockHash" in params[0]:
                return copy.deepcopy(self.logs.get(params[0]["blockHash"], []))
            return copy.deepcopy([log for height in range(int(params[0]["fromBlock"], 16),
                                                           int(params[0]["toBlock"], 16) + 1)
                                   for log in self.logs.get(self.blocks[height]["hash"], [])])
        if method == "eth_getTransactionReceipt":
            return {"status": "0x1", "blockNumber": "0x64", "logs": [log_for(self.blocks[100])]}
        if method == "eth_getTransactionByHash":
            return {"from": "transaction-sender-is-not-proven-token-deployer"}
        if method == "eth_blockNumber":
            return hex(max(self.blocks))
        raise AssertionError(f"unexpected RPC call {method}")


class CanonicalSink:
    def __init__(self):
        self.events = {}
        self.calls = 0

    def __call__(self, event):
        self.calls += 1
        self.events.setdefault(event.event_id, event.to_event().to_dict())
        return f"canonical-lake:{event.event_id}"


def observer(tmp_path, rpc, **kwargs):
    return EVMObservationAdapter(rpc, chain_name="ethereum-test-control", chain_id=1,
                                  factories=[contract()], start_block=100,
                                  checkpoint_path=tmp_path / "frontier.sqlite", **kwargs)


def test_official_abi_fixture_matches_decoder_and_pair_comes_from_data():
    fixture = json.loads((Path(__file__).parent / "fixtures/uniswap_v2_pair_created_abi.json").read_text())
    assert fixture["fixture_kind"] == "OFFICIAL_ABI_SCHEMA"
    assert fixture["mainnet_transaction_evidence"] is False
    assert [row["indexed"] for row in fixture["event"]["inputs"]] == [True, True, False, False]
    decoded = decode_pair_created(log_for(header(100)))
    assert decoded.token0 == TOKEN0 and decoded.token1 == TOKEN1
    assert decoded.pair == PAIR and decoded.pair != TOKEN0
    assert PAIR_CREATED_TOPIC == "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"
    assert select_quote_pair(decoded, [TOKEN0]) == (TOKEN1, TOKEN0)
    assert select_quote_pair(decoded, [TOKEN1]) == (TOKEN0, TOKEN1)
    assert select_quote_pair(decoded, []) is None


@pytest.mark.parametrize("mutation", ["unknown_topic", "missing_data", "bad_padding", "extra_topic", "removed"])
def test_unknown_or_malformed_native_layout_is_refused(mutation):
    log = log_for(header(100))
    if mutation == "unknown_topic":
        log["topics"][0] = "0x" + "ff" * 32
    elif mutation == "missing_data":
        log["data"] = "0x"
    elif mutation == "bad_padding":
        log["topics"][1] = "0x" + "ff" * 12 + TOKEN0[2:]
    elif mutation == "extra_topic":
        log["topics"].append("0x" + "00" * 32)
    elif mutation == "removed":
        log["removed"] = True
    assert decode_pair_created(log) is None


def test_native_pit_identity_retains_actual_receive_time_and_chain_namespace():
    fields = dict(chain="a", chain_id="1", chain_family="EVM", protocol="p", protocol_version="1",
                  event_type="POOL_CREATED", block_number=100, block_hash=block_hash(100),
                  transaction_hash="tx", event_index=1, source_at=100, observed_at=200,
                  finality="FINALIZED", source_ref="canonical-source:1")
    original = NativeChainEvent(**fields)
    later = NativeChainEvent(**{**fields, "observed_at": 300})
    other_chain = NativeChainEvent(**{**fields, "chain_id": "2"})
    assert original.event_id == later.event_id
    assert original.event_id != other_chain.event_id
    event = original.to_event()
    assert event.source_at == 100 and event.observed_at == 200
    assert event.metadata["execution_eligible"] is False


def test_checkpoint_commits_only_after_canonical_sink_and_restores(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(), CanonicalSink()
        first = observer(tmp_path, rpc, max_blocks_per_pass=1)
        result = await first.poll_once(sink)
        assert result["pass_status"] == "BACKFILL_PENDING" and result["next_block"] == 101
        restarted = observer(tmp_path, rpc, max_blocks_per_pass=2)
        result = await restarted.poll_once(sink)
        assert result["next_block"] == 103 and result["pass_status"] == "CAUGHT_UP"
        assert len(sink.events) == 3
        assert (await restarted.poll_once(sink))["emitted_events"] == 0
        assert all(row["metadata"]["observation_only"] for row in sink.events.values())
        assert all(row["source_at"] < row["observed_at"] for row in sink.events.values())
        assert all(method.startswith("eth_get") or method == "eth_chainId" for method, _ in rpc.requests)
    asyncio.run(scenario())


def test_provider_and_sink_failure_leave_checkpoint_for_retry(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(last=100), CanonicalSink()
        adapter = observer(tmp_path, rpc)
        rpc.fail_logs = True
        failed = await adapter.poll_once(sink)
        assert failed["pass_status"] == "DATA_BLOCKED" and failed["next_block"] == 100
        assert "secret.invalid" not in str(failed)
        rpc.fail_logs = False
        def unavailable_sink(event):
            raise OSError("canonical ledger unavailable")
        assert (await adapter.poll_once(unavailable_sink))["next_block"] == 100
        assert (await observer(tmp_path, rpc).poll_once(sink))["next_block"] == 101
        assert len(sink.events) == 1
    asyncio.run(scenario())


def test_unknown_event_is_retained_without_invented_pool_or_token(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(last=100), CanonicalSink()
        rpc.logs[block_hash(100)][0]["topics"][0] = "0x" + "ab" * 32
        result = await observer(tmp_path, rpc).poll_once(sink)
        event = next(iter(sink.events.values()))
        assert event["metadata"]["event_type"] == "UNKNOWN"
        assert event["token_addresses"] == ()
        assert event["metadata"]["pool_address"] is None
        assert event["metadata"]["native"]["log"]["topics"][0] == "0x" + "ab" * 32
        assert result["unknown_event_rate"] == 1
    asyncio.run(scenario())


def test_reorg_emits_corrections_before_replacement_and_preserves_old_evidence(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(), CanonicalSink()
        adapter = observer(tmp_path, rpc)
        await adapter.poll_once(sink)
        originals = dict(sink.events)
        for height in (101, 102):
            parent = block_hash(100) if height == 101 else block_hash(101, 1)
            rpc.blocks[height] = header(height, 1, parent)
            rpc.logs[block_hash(height, 1)] = [log_for(rpc.blocks[height])]
        result = await observer(tmp_path, rpc).poll_once(sink)
        assert result["reorgs"] == 1 and result["reverted_events_this_pass"] == 2
        assert result["next_block"] == 103
        corrections = [row for row in sink.events.values() if row["metadata"]["event_type"] == "REORG"]
        assert len(corrections) == 2
        assert all(row["metadata"]["reverts_event_id"] in originals for row in corrections)
        assert all(row["metadata"]["finality"] == "REVERTED" for row in corrections)
        assert set(originals).issubset(sink.events)
        assert len(sink.events) == 7  # 3 original, 2 corrections, 2 replacement
    asyncio.run(scenario())


def test_no_fallback_from_missing_finalized_head_to_latest(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(last=100), CanonicalSink()
        request = rpc.request
        async def no_finality(method, params):
            if method == "eth_getBlockByNumber" and params[0] == "finalized":
                return None
            return await request(method, params)
        rpc.request = no_finality
        result = await observer(tmp_path, rpc).poll_once(sink)
        assert result["pass_status"] == "DATA_BLOCKED" and result["next_block"] == 100
        assert not sink.events
    asyncio.run(scenario())


def test_block_budget_failure_does_not_silently_drop_tail(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(last=100), CanonicalSink()
        rpc.logs[block_hash(100)] = [log_for(rpc.blocks[100], i) for i in range(3)]
        adapter = observer(tmp_path, rpc, max_logs_per_block=2)
        assert (await adapter.poll_once(sink))["next_block"] == 100
        assert not sink.events
        assert (await observer(tmp_path, rpc, max_logs_per_block=3).poll_once(sink))["next_block"] == 101
        assert len(sink.events) == 3
    asyncio.run(scenario())


def test_registry_reuses_existing_rpc_and_config_does_not_establish_abi(tmp_path):
    rpc = FakeRPC()
    config = SimpleNamespace(chain_id=1, chain_type=ChainType.EVM, factories={"uniswap_v2": FACTORY})
    registry = SimpleNamespace(get_chain=lambda name: config, get_rpc=lambda name: rpc)
    adapter = EVMObservationAdapter.from_registry(registry, "configured", start_block=100,
                                                checkpoint_path=tmp_path / "frontier.sqlite")
    assert adapter.rpc is rpc
    assert adapter.factories[FACTORY].decoder == "UNKNOWN"
    assert adapter.report()["unknown_event_rate"] is None
    with pytest.raises(ValueError, match="no latest fallback"):
        observer(tmp_path, rpc, finality_tag="latest")


def test_legacy_factory_detectors_use_the_same_reviewed_pair_decoder():
    async def scenario():
        rpc = FakeRPC(last=100)
        config = SimpleNamespace(base_tokens=[TOKEN0], factories={"uniswap_v2": FACTORY})
        received = []
        async def callback(candidate):
            received.append(candidate)
        mempool = MempoolDetector("ethereum-test", config, rpc, callback)
        await mempool._handle_factory_call({"hash": "tx", "from": "sender"}, "uniswap_v2", FACTORY)
        poller = FactoryPoller("ethereum-test", config, rpc, callback)
        poller._last_blocks["uniswap_v2"] = 99
        await poller._poll_factory("uniswap_v2", FACTORY)
        assert len(received) == 2
        assert all(row.pair == PAIR and row.address == TOKEN1 and row.base_token == TOKEN0 for row in received)
        assert all(row.deployer is None and row.metadata["execution_eligible"] is False for row in received)
    asyncio.run(scenario())


def test_reorg_of_first_requested_block_can_rewind_to_measured_anchor(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(last=100), CanonicalSink()
        await observer(tmp_path, rpc).poll_once(sink)
        rpc.blocks[100] = header(100, 1, block_hash(99))
        rpc.logs[block_hash(100, 1)] = [log_for(rpc.blocks[100])]
        result = await observer(tmp_path, rpc).poll_once(sink)
        assert result["reverted_events_this_pass"] == 1
        assert result["next_block"] == 101
    asyncio.run(scenario())


def test_partially_persisted_block_is_retracted_if_provider_reorgs_before_retry(tmp_path):
    async def scenario():
        rpc, sink = FakeRPC(last=100), CanonicalSink()
        rpc.logs[block_hash(100)] = [log_for(rpc.blocks[100], i) for i in range(2)]
        def half_unavailable(event):
            if event.event_index == 1:
                raise OSError("canonical ledger unavailable halfway through block")
            return sink(event)
        result = await observer(tmp_path, rpc).poll_once(half_unavailable)
        assert result["next_block"] == 100 and len(sink.events) == 1
        original = next(iter(sink.events))
        rpc.blocks[100] = header(100, 1, block_hash(99))
        rpc.logs[block_hash(100, 1)] = [log_for(rpc.blocks[100])]
        result = await observer(tmp_path, rpc).poll_once(sink)
        corrections = [row for row in sink.events.values() if row["metadata"]["event_type"] == "REORG"]
        assert len(corrections) == 1
        assert corrections[0]["metadata"]["reverts_event_id"] == original
        assert result["next_block"] == 101
    asyncio.run(scenario())
