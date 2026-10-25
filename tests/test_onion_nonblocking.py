"""Bulk public-source admission must preserve delivery without freezing asyncio."""
import asyncio
import threading
from types import SimpleNamespace

from src.collectors.event_source import Event, SourceClass
from src.collectors.onion_discovery import deliver
from src.collectors.onion_frontier import OnionFrontierTransport
from src.runtime.source_intelligence import SourceIntelligence
from tests.test_onion_collection import URL


def test_consumer_waits_for_durable_ack_in_worker_while_health_can_run(monkeypatch):
    async def scenario():
        owner = threading.get_ident()
        entered, released = threading.Event(), threading.Event()
        phases = []
        value = SourceIntelligence()
        event = Event("public:directory", SourceClass.WEB, 1, 2, text=URL)
        def enqueue(urls, source, at):
            assert threading.get_ident() != owner
            phases.append("admit")
            entered.set()
            assert released.wait(5), "async health did not run while SQLite delivery waited"
            phases.append("committed")
        def acknowledge(observation_id, job_id):
            assert phases[-1] == "committed" and job_id == "canonical-job"
            phases.append("ack")
            value._running = False
        frontier = SimpleNamespace(source_id="continuous", enqueue=enqueue)
        value._running = True
        value.transports = {event.source_id: SimpleNamespace(kind="onion_directory", acknowledge=acknowledge)}
        value.social_intel = SimpleNamespace()
        value._source_events = {}
        value._record_population_narrative = lambda event, **kwargs: "canonical-job"
        async def start(*args, **kwargs): return {}
        async def mesh_start(): return 1
        async def next_event(): return event
        value.source_mesh = SimpleNamespace(start=mesh_start, next_event=next_event)
        monkeypatch.setattr("src.runtime.source_intelligence.start_transports", start)
        monkeypatch.setattr("src.collectors.onion_discovery.restore", lambda _: 0)
        monkeypatch.setattr("src.collectors.onion_discovery._frontier", lambda _: frontier)
        async def health():
            while not entered.is_set():
                await asyncio.sleep(0.001)
            phases.append("health")
            released.set()
        await asyncio.gather(value._source_consumer_loop(), health())
        assert phases == ["admit", "health", "committed", "ack"]
    asyncio.run(scenario())


def test_admission_failure_never_acknowledges_parent():
    ack = []
    def fail(*args): raise OSError("disk unavailable")
    value = SimpleNamespace(transports={"directory": SimpleNamespace(kind="onion_directory", acknowledge=lambda *a:ack.append(a))})
    event = Event("directory", SourceClass.WEB, 1, 2, text=URL)
    assert deliver(value, event, "canonical", frontier=SimpleNamespace(source_id="frontier",enqueue=fail), prepared=True) is False
    assert not ack


def test_frontier_lease_waits_for_writer_without_blocking_loop(tmp_path):
    async def scenario():
        value = OnionFrontierTransport(tmp_path / "frontier.sqlite")
        original = value._lease
        entered, released = threading.Event(), threading.Event()
        owner = threading.get_ident()
        def slow_lease(now):
            assert threading.get_ident() != owner
            entered.set()
            assert released.wait(5)
            return original(now)
        value._lease = slow_lease
        async def health():
            while not entered.is_set():
                await asyncio.sleep(0.001)
            released.set()
        result, _ = await asyncio.gather(value.fetch(), health())
        assert result == []
        await value.stop()
    asyncio.run(scenario())
