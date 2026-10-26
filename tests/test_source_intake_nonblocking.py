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


def test_neural_queue_commit_yields_and_keeps_exactly_one_canonical_source(tmp_path):
    async def scenario():
        from src.research.neural_sentiment import SentimentResearch
        value=desk(tmp_path);value.global_config={'ops_state_dir':str(tmp_path)}
        neural=SentimentResearch(tmp_path/'neural.sqlite');value.neural_sentiment=neural
        entered,released=threading.Event(),threading.Event();owner=threading.get_ident();calls=[]
        original=neural.enqueue
        def blocked(payload,**kwargs):
            assert threading.get_ident()!=owner
            assert value.unknown_engine.frontier.job(kwargs['source_job'])['payload_ref']==kwargs['source_ref']
            calls.append(kwargs);entered.set();assert released.wait(5)
            return original(payload,**kwargs)
        neural.enqueue=blocked
        task=asyncio.create_task(value._record_population_narrative_async(event()))
        try:
            assert await asyncio.to_thread(entered.wait,3)
            # A real socket request must complete while the durable queue is gated.
            from aiohttp import web
            from aiohttp.test_utils import TestClient,TestServer
            async def health(request):return web.json_response({'responsive':True})
            app=web.Application();app.router.add_get('/health',health)
            async with TestClient(TestServer(app)) as client:
                response=await asyncio.wait_for(client.get('/health'),1)
                assert (await response.json())['responsive'] and not released.is_set()
        finally:released.set()
        job=await task
        assert len(calls)==1 and calls[0]['source_job']==job
        assert neural.report()['inputs']==1 and neural.report()['canonical_windows']==0
        assert TradeEvidenceLedger.verify(value.unknown_engine.evidence.path)[0]
        neural.close();value.public_claim_links.close();value.unknown_engine.frontier.close()
    asyncio.run(scenario())


def test_cancellation_drains_started_neural_commit_before_owner_can_close(tmp_path):
    async def scenario():
        from src.research.neural_sentiment import SentimentResearch
        value=desk(tmp_path);neural=SentimentResearch(tmp_path/'neural.sqlite');value.neural_sentiment=neural
        entered,released=threading.Event(),threading.Event();original=neural.enqueue
        def blocked(*args,**kwargs):
            entered.set();assert released.wait(5);return original(*args,**kwargs)
        neural.enqueue=blocked
        task=asyncio.create_task(value._record_population_narrative_async(event()))
        assert await asyncio.to_thread(entered.wait,3)
        task.cancel();await asyncio.sleep(0);assert not task.done()
        released.set()
        try:await task
        except asyncio.CancelledError:pass
        else:raise AssertionError('cancellation swallowed')
        assert neural.report()['inputs']==1 and getattr(value,'public_claim_links',None) is None
        assert TradeEvidenceLedger.verify(value.unknown_engine.evidence.path)[0]
        neural.close();value.unknown_engine.frontier.close()
    asyncio.run(scenario())


def test_neural_failure_retains_canonical_work_and_reports_one_retryable_failure(tmp_path):
    async def scenario():
        from src.research.neural_sentiment import SentimentResearch
        value=desk(tmp_path);value.global_config={'ops_state_dir':str(tmp_path)}
        neural=SentimentResearch(tmp_path/'neural.sqlite');value.neural_sentiment=neural
        neural.enqueue=Mock(side_effect=OSError('sensitive fixture must not be copied to failure type'))
        job=await value._record_population_narrative_async(event())
        assert job and value.unknown_engine.frontier.job(job)['payload_ref']
        assert neural.enqueue.call_count==neural.intake_failures==1
        assert neural.last_intake_error=='OSError' and neural.report()['inputs']==0
        assert value.public_claim_links.page()['items'][0]['source_job']==job
        neural.close();value.public_claim_links.close();value.unknown_engine.frontier.close()
    asyncio.run(scenario())
