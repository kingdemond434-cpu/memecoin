"""Production source hooks retain actual facts through the canonical queue."""
from types import SimpleNamespace
import asyncio
from collections import deque

from src.collectors.event_source import Event, SourceClass
from src.research.frontier import FrontierIndex
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine
from src.runtime.source_intelligence import SourceIntelligence
from src.strategies.champion_challenger import ChampionChallengerFramework


class Desk(SourceIntelligence):
    pass


def test_nonempty_source_queue_yields_between_durable_records(monkeypatch):
    async def scenario():
        value = Desk()
        value._running = True
        value.transports = {}
        value.social_intel = SimpleNamespace()
        pending = deque(range(20))
        indexed, heartbeat = [], []
        async def start(*args, **kwargs): return {}
        async def mesh_start(): return 1
        async def next_event(): return pending.popleft()
        value.source_mesh = SimpleNamespace(start=mesh_start, next_event=next_event)
        def index(event, **kwargs):
            indexed.append(event)
            if not pending:
                value._running = False
        value._index_source_event = index
        monkeypatch.setattr("src.runtime.source_intelligence.start_transports", start)
        async def health():
            await asyncio.sleep(0)
            heartbeat.append(len(indexed))
        await asyncio.gather(value._source_consumer_loop(), health())
        assert indexed == list(range(20))
        assert 0 < heartbeat[0] < len(indexed)
    asyncio.run(scenario())


def desk(tmp_path):
    value = Desk()
    value.unknown_engine = UnknownOpportunityEngine(ChampionChallengerFramework(
        state_path=str(tmp_path / "champion.json")), FrontierIndex(tmp_path / "frontier.sqlite"),
        TradeEvidenceLedger(tmp_path / "evidence.jsonl"))
    value.transports = {"rss:test": SimpleNamespace(kind="rss")}
    return value


def test_actual_source_event_enters_canonical_intake_before_existing_index(tmp_path):
    value = desk(tmp_path)
    value._source_events = {}
    posts, decisions = [], []
    value.source_genealogy = SimpleNamespace(record=posts.append)
    value.launch_census = SimpleNamespace(_records={})
    value.request_redecision = decisions.append
    event = Event("rss:test", SourceClass.NEWS, 10, 11, text="本地猫币观察", language="zh",
                  token_addresses=["NativeToken"], author_id="publisher-handle",
                  metadata={"chain_id": "solana", "region": "CN", "link": "https://example.org/fixture"})
    value._index_source_event(event)
    assert value._source_events["NativeToken"] == [event]
    assert len(posts) == 1 and decisions == ["NativeToken"]
    job = value.unknown_engine.frontier.job(value.population_research_intake["job_id"])
    row = value.unknown_engine._load_refs({job["payload_ref"]})[job["payload_ref"]]["point_payload"]
    assert row["text"] == event.text and row["language"] == "zh" and row["platform"] == "rss"
    assert row["event_at"] == 10 and row["observed_at"] == 11
    assert row.get("operator_id") is None  # author handle is not an operator identity
    assert row["narrative_id"] == "token:solana:NativeToken"
    assert TradeEvidenceLedger.verify(value.unknown_engine.evidence.path)[0]


def test_decoded_first_buy_enqueues_native_case_preserving_actor_once(tmp_path):
    value = desk(tmp_path)
    value._actor_seen, value._actor_entries = {}, {}
    value.hot_state = SimpleNamespace(active_tokens={"Token"}, touch_token=lambda token: None)
    value.wallet_intel, value.sol_price_usd = None, 0
    entries = []
    value.wallet_independence = SimpleNamespace(record_entries=entries.extend)
    value.launch_census = SimpleNamespace(_records={})
    value.buyer_dna = SimpleNamespace(depth=25)
    event = {"wallet": "AbCnativeWallet", "side": "buy", "timestamp": 10, "signature": "fixture-decoded-signature"}
    value._record_actor_entry("Token", event, {"timestamp": 10, "notional_sol": 1})
    job_id = value.population_research_intake["job_id"]
    value._record_actor_entry("Token", event, {"timestamp": 10, "notional_sol": 1})
    assert value.population_research_intake["job_id"] == job_id and len(entries) == 1
    job = value.unknown_engine.frontier.job(job_id)
    row = value.unknown_engine._load_refs({job["payload_ref"]})[job["payload_ref"]]["point_payload"]
    assert row["address"] == "AbCnativeWallet" and row["chain_family"] == "SOLANA"
    assert row["relation"] == {"kind": "bought", "target": "Token", "target_type": "token"}
    assert "selection" not in row and "control_plan" not in row
    assert "capital_size" not in row and "entry_delay_seconds" not in row


def test_follow_callback_forwards_actual_envelope_without_making_mark_into_return(tmp_path):
    value = desk(tmp_path)
    payload = {"observation_id": "resolved-native-paper", "observed_at": 20,
               "entry_fill_cost_usd": 100, "exit_fill_proceeds_usd": 90, "costs": {"fees_usd": 2}}
    job_id = value._record_population_follow_outcome(payload, evidence_refs=["fixture:native-paper"], now=20)
    job = value.unknown_engine.frontier.job(job_id)
    retained = value.unknown_engine._load_refs({job["payload_ref"]})[job["payload_ref"]]["point_payload"]
    assert retained == payload
    assert "gas_usd" not in retained["costs"] and "executable_multiple" not in retained
    assert job["kind"] == "FOLLOW_OUTCOME"  # consumer must reject this deliberately incomplete fixture


def test_ledger_failure_is_visible_while_legacy_source_index_still_works(tmp_path, caplog):
    value = desk(tmp_path)
    value._source_events = {}
    value.source_genealogy = SimpleNamespace(record=lambda post: None)
    value.launch_census = SimpleNamespace(_records={})
    value.request_redecision = lambda token: None
    value.unknown_engine.evidence.record_reference = lambda *args, **kwargs: None
    event = Event("rss:test", SourceClass.NEWS, 10, 11, text="fixture", token_addresses=["Token"])
    value._index_source_event(event)
    assert value._source_events["Token"] == [event]
    assert value.population_research_intake["status"] == "DATA_BLOCKED"
    assert value.population_research_intake["failed_inputs"] == 1
    assert "DATA_BLOCKED" in caplog.text


def test_absent_engine_is_optional_and_does_not_create_background_work():
    value = Desk()
    assert value._record_population_follow_outcome({"observed_at": 1}, now=1) is None


def test_actual_census_recruits_declared_frame_without_outcome_selection_and_survives_restart(tmp_path):
    from src.research.population_hunter import PopulationHunter

    value = desk(tmp_path)
    value.launch_census = SimpleNamespace(_records={"Token": SimpleNamespace(detected_at=1)})
    value.benchmark_corpus = SimpleNamespace(wallets={"AbC": SimpleNamespace(label="declared-study-only")})
    job_id = value._record_population_actor("Token", {"wallet": "AbC", "timestamp": 10,
        "signature": "native-fixture-signature"}, {"timestamp": 10})
    job = value.unknown_engine.frontier.job(job_id)
    raw = value.unknown_engine._load_refs({job["payload_ref"]})[job["payload_ref"]]["point_payload"]
    frame = value.unknown_engine.framework.get_research_component("population_sampling_frame_v1")
    assert raw["selection"]["population"] == "UNBIASED"
    assert raw["selection"]["sampling_frame_ref"] == frame["reference"]
    assert frame["declared_at"] <= raw["selection"]["selected_at"] == raw["observed_at"]
    assert raw["selection"]["uses_outcomes"] is False
    assert raw["selection"]["additional_populations"] == ["FAMOUS_BENCHMARK"]
    assert raw["selection"]["benchmark_ref"] == frame["reference"]
    assert frame["representativeness"] == "UNBIASED_WITHIN_DECLARED_OBSERVED_STREAM_ONLY"
    assert len(set(raw["control_plan"]["delay_grid_seconds"])) >= 2
    hunter = PopulationHunter(value.unknown_engine.framework)
    result = hunter.consume("ACTOR_OBSERVATION", raw, [job["payload_ref"]], now=raw["observed_at"])
    assert result["detail"]["population"] == "UNBIASED"
    assert hunter.report()["populations"]["FAMOUS_BENCHMARK"] == 1
    assert hunter.report()["populations"]["UNBIASED"] == 1
    assert result["detail"]["controls"]["delay_seconds"] > 0
    assert result["detail"]["controls"]["status"] == "DATA_BLOCKED"  # actual capital/regime matching still unknown
    assert hunter.report()["actor_values"][0]["value"]["lower_bound"] is None
    restarted = desk(tmp_path)
    restarted.launch_census = value.launch_census
    assert restarted._population_sampling_frame("Token")["reference"] == frame["reference"]
    assert TradeEvidenceLedger.verify(value.unknown_engine.evidence.path)[0]
