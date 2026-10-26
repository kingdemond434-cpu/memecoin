"""LaunchLab discovery against captured legacy, ALT and V1 transactions."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.chains.launchlab import LAUNCHLAB_PROGRAM, LaunchLabMonitor
from src.chains.yellowstone_grpc import (
    SolanaRpcProgramStream, b58decode, b58encode, create_combined_subscription,
    extract_compute_budget, instruction_fields, transaction_parts,
)

FIXTURE = Path(__file__).parent / "fixtures/launchlab_launches.json"


def transactions():
    return json.loads(FIXTURE.read_text())


async def decode(tx):
    out = []
    monitor = LaunchLabMonitor(Mock(), out.append)
    await monitor._on_transaction(tx)
    return out, monitor


@pytest.mark.asyncio
async def test_public_captures_preserve_creator_quote_and_transfer_fee():
    expected = [
        ("J2V7tRwJqwr9a1gpL3VMQRuJVbebcH2YbvNezfiMhGUr", "9ePSmJyUxsXcWcjxeUpqJSxq3R65MvuFuLPkSWvU43TX", None),
        ("E3myentprJHSWdUvD4LDS3A48QMnUNQVUaHwSmDNsm7k", "FkKXAcfrso5t53QVxnKzHQaypDpCc7jQF2qpZENhYz2A", None),
        ("H4Jz31z5JKrPKtbndUqvpj29KK6kMFdcnduwgmMu6BPE", "GpTXmkdvrTajqkzX1fBmC4BUjSboF9dHgfnqPqj8WAc4", 100),
    ]
    for tx, (mint, creator, fee) in zip(transactions(), expected):
        events, monitor = await decode(tx)
        assert len(events) == 1 and not monitor.rejected
        event = events[0]
        assert (event["token"], event["creator"]) == (mint, creator)
        assert event["quote_decimals"] is None
        assert event["timestamp"] == tx["blockTime"]
        assert event["execution_support"] == "DATA_BLOCKED_NATIVE_CURVE_EXECUTION"
        extension = event["transfer_fee_extension"]
        assert (extension["transfer_fee_basis_points"] if extension else None) == fee
    assert event["quote_mint"] == "Xs3oZwbHvqis4NYcf4YKWmEia2eC84wSiVrcYcTqpH8"
    assert event["priority_lamports"] == 100_000
    assert event["compute_unit_price"] is None  # V1 absolute fee, not CU price.


@pytest.mark.asyncio
async def test_creator_is_not_replaced_with_sponsoring_payer():
    tx = transactions()[0]
    create = tx["transaction"]["message"]["instructions"][2]
    create["accounts"][1] = 1
    events, _ = await decode(tx)
    assert events[0]["creator"] == tx["transaction"]["message"]["accountKeys"][1]
    assert events[0]["creator"] != events[0]["payer"]


@pytest.mark.asyncio
async def test_cpi_unknown_platform_and_multiple_launches_are_retained():
    tx = transactions()[0]
    keys, _, _, _ = transaction_parts(tx)
    message = tx["transaction"]["message"]
    create = copy.deepcopy(message["instructions"][2])
    message["instructions"] = []
    tx["meta"]["innerInstructions"] = [{"index": 0, "instructions": [create]}]
    keys[create["accounts"][3]] = "11111111111111111111111111111111"
    message["accountKeys"] = keys
    other = copy.deepcopy(create)
    message["accountKeys"].append("So11111111111111111111111111111111111111112")
    other["accounts"][6] = len(message["accountKeys"]) - 1
    tx["meta"]["innerInstructions"][0]["instructions"].append(other)
    events, monitor = await decode(tx)
    assert len(events) == 2 and not monitor.rejected
    assert events[0]["token"] != events[1]["token"]
    assert all(e["platform_config"] == "11111111111111111111111111111111" for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["failed", "missing_meta", "negative_index", "truncated", "unknown_tail"])
async def test_invalid_packets_never_become_launches(failure):
    tx = transactions()[0]
    if failure == "failed":
        tx["meta"]["err"] = {"InstructionError": [2, "Custom"]}
    elif failure == "missing_meta":
        del tx["meta"]
    else:
        create = tx["transaction"]["message"]["instructions"][2]
        if failure == "negative_index":
            create["accounts"][6] = -1
        else:
            raw = b58decode(create["data"])
            create["data"] = b58encode(raw[:20] if failure == "truncated" else raw + b"bad")
    events, monitor = await decode(tx)
    assert not events and monitor.rejected


@pytest.mark.asyncio
async def test_all_three_initializer_layouts_come_from_idl():
    from src.chains.idl import load_idl
    tx = transactions()[0]
    keys, _, _, _ = transaction_parts(tx)
    create = tx["transaction"]["message"]["instructions"][2]
    original = b58decode(create["data"])
    # Public capture is Option::None with SDK's ten zero padding bytes.
    payload = original[8:-11]
    schema2022 = next(i for i in load_idl("raydium_launchlab")["instructions"] if i["name"] == "initialize_with_token_2022")
    named = {a["name"]: keys[create["accounts"][n]] for n,a in enumerate(schema2022["accounts"])}
    named["base_token_program"] = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    for name in ("initialize", "initialize_v2"):
        schema = next(i for i in load_idl("raydium_launchlab")["instructions"] if i["name"] == name)
        accounts = [named.get(a["name"], "11111111111111111111111111111111") for a in schema["accounts"]]
        data = bytes(schema["discriminator"]) + (payload[:-1] if name == "initialize" else payload)
        event = LaunchLabMonitor(Mock(), Mock()).decode(accounts, list(range(len(accounts))), data)
        assert event["instruction"] == name and event["creator"] == named["creator"]


@pytest.mark.asyncio
async def test_geyser_cpi_keys_and_v1_config():
    from src.chains.generated import solana_storage_pb2 as p
    tx = transactions()[2]
    keys, instructions, signature, slot = transaction_parts(tx)
    message = p.Message(account_keys=[b58decode(k) for k in keys], versioned=True)
    message.config.priority_fee = 100_000
    message.config.compute_unit_limit = 450_000
    inner = []
    for item in instructions:
        program, accounts, data = instruction_fields(item)
        compiled = p.CompiledInstruction(program_id_index=program, accounts=bytes(accounts), data=data)
        inner.append(compiled)
    meta = SimpleNamespace(HasField=lambda _: False, inner_instructions=[SimpleNamespace(instructions=inner)],
                           loaded_writable_addresses=[], loaded_readonly_addresses=[])
    packet = SimpleNamespace(slot=slot, transaction=SimpleNamespace(signature=b58decode(signature),
        transaction=SimpleNamespace(message=message), meta=meta))
    events, _ = await decode(packet)
    assert len(events) == 1 and events[0]["priority_lamports"] == 100_000
    assert events[0]["token"] == "H4Jz31z5JKrPKtbndUqvpj29KK6kMFdcnduwgmMu6BPE"


@pytest.mark.asyncio
async def test_rpc_and_geyser_subscriptions_admit_v1_launches():
    assert LAUNCHLAB_PROGRAM in create_combined_subscription().transactions["memecoin_programs"].accounts_include
    for name in ("InitializeV2", "InitializeWithToken2022"):
        assert SolanaRpcProgramStream._looks_like_pool_creation(["Program log: Instruction: " + name])
    rpc = SimpleNamespace(request=AsyncMock(return_value=transactions()[2]))
    stream = SolanaRpcProgramStream(rpc, [LAUNCHLAB_PROGRAM])
    out = []
    LaunchLabMonitor(stream, out.append)
    with patch("src.chains.yellowstone_grpc.asyncio.sleep", new=AsyncMock()):
        await stream._fetch_confirmed_transaction("captured-signature")
    assert len(out) == 1
    assert rpc.request.call_args.args[1][1]["maxSupportedTransactionVersion"] == 1


def test_v1_ignores_noop_budget_instructions_and_missing_config_is_unknown():
    tx = transactions()[2]
    keys, instructions, _, _ = transaction_parts(tx)
    budget = extract_compute_budget(keys, instructions, tx_data=tx)
    assert budget.priority_lamports == 100_000 and budget.unit_price_micro_lamports is None
    del tx["transaction"]["message"]["transactionConfig"]
    assert extract_compute_budget(keys, instructions, tx_data=tx).priority_lamports is None


def test_idl_and_public_fixture_are_pinned():
    root = Path(__file__).parents[1]
    for relative, expected in {
        "idl/raydium_launchlab.json": "4949983335d0a345c91b76f88e91e1e9680230e8190be5e5b9fe537df927668f",
        "tests/fixtures/launchlab_launches.json": "599ff4b61939d455d87ff8460c823c044b6e5d60cf44a198f109a5d3404a7ead",
    }.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


@pytest.mark.asyncio
async def test_runtime_archives_all_launches_without_pump_reserves_or_execution_authority(tmp_path):
    from src.chains.launchpads import LaunchpadRegistry
    from src.main import MemecoinQuantDesk
    from src.research.trade_evidence import TradeEvidenceLedger
    from src.runtime.feed_race import FeedRace
    from src.runtime.ingestion import MinedRecordIngestion
    value = SimpleNamespace(
        _stream_events={}, feed_race=FeedRace(), current_regime="unknown",
        launchpads=LaunchpadRegistry(), trade_evidence=TradeEvidenceLedger(tmp_path / "launch.jsonl"),
        launch_census=Mock(), wallet_intel=Mock(), dataset_builder=Mock(),
        genealogy=Mock(), info_graph=Mock(), observed_bids=Mock(),
        detection_engine=SimpleNamespace(_on_candidate=AsyncMock()),
        _assess_identity=Mock(), _record_prelaunch_launch=Mock(),
        _prelaunch_context=Mock(return_value={}),
    )
    value._note_launch_venue = lambda event: MemecoinQuantDesk._note_launch_venue(value, event)
    value._record_hot_creator = lambda token, event: MinedRecordIngestion._record_hot_creator(value, token, event)
    for tx in transactions():
        events, _ = await decode(tx)
        await MinedRecordIngestion._on_launchpad_event(value, events[0])
    assert value.launch_census.see.call_count == 3
    assert value.dataset_builder.start_episode.call_count == 3
    # The existing registry's three distinct observations now attest discovery.
    candidate = value.detection_engine._on_candidate.call_args.args[0]
    assert candidate.factory == LAUNCHLAB_PROGRAM
    assert candidate.base_token == "Xs3oZwbHvqis4NYcf4YKWmEia2eC84wSiVrcYcTqpH8"
    assert candidate.metadata["execution_support"] == "DATA_BLOCKED_NATIVE_CURVE_EXECUTION"
    assert TradeEvidenceLedger.verify(value.trade_evidence.path)[0]
    assert len(value.trade_evidence.path.read_text().splitlines()) == 3
    await MinedRecordIngestion._on_launchpad_event(value, events[0])
    assert value.dataset_builder.start_episode.call_count == 3  # racing duplicate
    assert not hasattr(value, "_latest_curve_state")  # handler never needs Pump state


@pytest.mark.asyncio
@pytest.mark.parametrize('intake_failure',[False,True])
async def test_native_capture_hands_off_before_episode_enrichment_and_retains_failures(tmp_path,caplog,intake_failure):
    from src.chains.pump_route import WSOL_MINT
    from src.research.fork_probe import ForkProbe
    from src.research.trade_evidence import TradeEvidenceLedger
    from src.runtime.ingestion import MinedRecordIngestion
    events,_=await decode(transactions()[0])
    event={**events[0],'quote_mint':WSOL_MINT}  # Synthetic quote control.
    ledger=TradeEvidenceLedger(tmp_path/'launch.jsonl')
    probe=ForkProbe(tmp_path/'fork.sqlite')
    if intake_failure:
        probe.enqueue_native_launch=Mock(side_effect=OSError('private-provider-text'))
    surface=SimpleNamespace(_stream_events={},observed_bids=Mock(),_note_launch_venue=Mock(),
        launchpads=SimpleNamespace(specs={}),trade_evidence=ledger,fork_probe=probe,
        _record_hot_creator=Mock(side_effect=RuntimeError('synthetic enrichment failure')))
    with pytest.raises(RuntimeError,match='enrichment failure'):
        await MinedRecordIngestion._on_launchpad_event(surface,event)
    assert TradeEvidenceLedger.verify(ledger.path)[0]
    assert len(ledger.path.read_text().splitlines())==1
    if intake_failure:
        assert surface._stream_events['native_probe_intake_failed']==1
        assert 'private-provider-text' not in caplog.text
    else:
        assert probe.report()['targets']==1 and probe.report()['canonical_native_source_links']==1
        raw=json.loads(ledger.path.read_text())
        assert probe.db.execute('SELECT ref,available FROM native_target_sources').fetchone()==(
            'native-launch:'+raw['hash'],raw['timestamp'])
    probe.close()


@pytest.mark.asyncio
async def test_failed_canonical_launch_append_does_not_enter_queue_or_enrichment():
    from src.runtime.ingestion import MinedRecordIngestion
    events,_=await decode(transactions()[0])
    surface=SimpleNamespace(_stream_events={},observed_bids=Mock(),_note_launch_venue=Mock(),
        launchpads=SimpleNamespace(specs={}),trade_evidence=SimpleNamespace(record=Mock(return_value=None)),
        fork_probe=SimpleNamespace(enqueue_native_launch=Mock()),_record_hot_creator=Mock())
    await MinedRecordIngestion._on_launchpad_event(surface,events[0])
    surface.fork_probe.enqueue_native_launch.assert_not_called()
    surface._record_hot_creator.assert_not_called()
    assert surface._stream_events['native_launch:canonical_failed']==1


@pytest.mark.asyncio
async def test_native_monitor_is_wired_before_rpc_start():
    from src.runtime.wiring import SubsystemWiring
    observed = {}
    class Stream:
        def __init__(self, rpc, programs):
            observed["programs"] = programs
            self.handlers = []
        def on(self, kind, callback):
            self.handlers.append(callback)
        async def start(self):
            assert any(getattr(h, "__self__", None).__class__ is LaunchLabMonitor for h in self.handlers)
            observed["started"] = True
    value = SimpleNamespace(offline=False, solana_rpc=Mock(), _on_pump_event=AsyncMock(),
        _on_raydium_event=AsyncMock(), _on_launchpad_event=AsyncMock())
    with patch("src.runtime.wiring.SolanaRpcProgramStream", Stream), patch.dict("os.environ", {"YELLOWSTONE_GRPC_URL": ""}):
        await SubsystemWiring._setup_yellowstone(value)
    assert observed["started"] and LAUNCHLAB_PROGRAM in observed["programs"]
