"""Canonical source persistence yields without moving owner-bound state."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from src.collectors.event_source import Event, SourceClass
from src.research.public_claim_links import PublicClaimLinks
from src.research.trade_evidence import TradeEvidenceLedger
from tests.test_population_callbacks import desk

ADDRESS = "So11111111111111111111111111111111111111112"


def event():
    return Event("rss:test", SourceClass.NEWS, 10, 11, text="public wallet launch claim",
                 token_addresses=[ADDRESS], metadata={"chain_id":"solana"})


def test_real_canonical_write_yields_and_thread_bound_claims_stay_on_owner(tmp_path):
    async def scenario():
        value = desk(tmp_path)
        value.global_config = {"ops_state_dir": str(tmp_path)}
        value.public_claim_links = PublicClaimLinks(tmp_path / "claims.sqlite")
        owner = threading.get_ident()
        entered, released = threading.Event(), threading.Event()
        original = value.unknown_engine.enqueue_work
        phases = []
        def blocked(*args, **kwargs):
            assert threading.get_ident() != owner
            phases.append("persist")
            entered.set()
            assert released.wait(5), "health could not run during a canonical commit"
            return original(*args, **kwargs)
        value.unknown_engine.enqueue_work = blocked
        async def health():
            while not entered.is_set():
                await asyncio.sleep(0.001)
            phases.append("health")
            released.set()
        job, _ = await asyncio.gather(value._record_population_narrative_async(event()), health())
        assert phases == ["persist", "health"]
        assert value.unknown_engine.frontier.job(job)["kind"] == "NARRATIVE_OBSERVATION"
        # This real SQLite connection was created on the owner thread. An
        # accidental worker-side finalization would be rejected and lose links.
        items = value.public_claim_links.page()["items"]
        assert len(items) == 1 and items[0]["source_job"] == job
        assert TradeEvidenceLedger.verify(value.unknown_engine.evidence.path)[0]
        value.public_claim_links.close()
        value.unknown_engine.frontier.close()
    asyncio.run(scenario())


def test_failed_async_intake_does_not_retry_synchronously(tmp_path):
    async def scenario():
        value = desk(tmp_path)
        value.unknown_engine.enqueue_work = Mock(side_effect=OSError("disk unavailable"))
        assert await value._record_population_narrative_async(event()) is None
        assert value.unknown_engine.enqueue_work.call_count == 1
        assert value.population_research_intake["failed_inputs"] == 1
        assert value.population_research_intake["status"] == "DATA_BLOCKED"
        value.unknown_engine.frontier.close()
    asyncio.run(scenario())


def test_cancel_waits_for_started_persistence_without_acknowledging_source(tmp_path):
    async def scenario():
        value = desk(tmp_path)
        entered, released, committed = threading.Event(), threading.Event(), threading.Event()
        def persist(*args, **kwargs):
            entered.set()
            assert released.wait(5)
            committed.set()
            return "durable-job"
        value._enqueue_population_work = persist
        task = asyncio.create_task(value._record_population_narrative_async(event()))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        released.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("source cancellation was swallowed")
        assert committed.is_set()
        assert getattr(value, "public_claim_links", None) is None
        value.unknown_engine.frontier.close()
    asyncio.run(scenario())
