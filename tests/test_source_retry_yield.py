"""A retrying backlog must yield without promoting retry work to an ACK."""
import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
from src.runtime.source_intelligence import SourceIntelligence


class Desk(SourceIntelligence):
    pass


def fixture(monkeypatch, mode):
    value=Desk();value._running=True;value.transports={};value.social_intel=SimpleNamespace()
    pending=deque(range(20));retried=[];acked=[];indexed=[]
    async def start(*args,**kwargs):return {}
    async def mesh_start():return 1
    async def next_event():return pending.popleft()
    async def record(event):
        if mode=='canonical_error':raise OSError('fixture')
        return None if mode in ('canonical_unavailable','retry_error') else 'work-fixture'
    async def retry(event):
        retried.append(event)
        if not pending:value._running=False
        if mode=='retry_error':raise OSError('fixture')
    async def deliver(*args,**kwargs):return mode=='success'
    async def ack(event,*args,**kwargs):acked.append(event)
    def index(event,**kwargs):
        indexed.append(event)
        if not pending:value._running=False
    value.source_mesh=SimpleNamespace(start=mesh_start,next_event=next_event,journal=object(),
        retry_event=retry,acknowledge_event=ack,delivery_receipt=lambda event:object())
    value.unknown_engine=SimpleNamespace(frontier=SimpleNamespace(job_payload_ref=lambda _:None))
    value._record_population_narrative_async=record;value._index_source_event=index
    monkeypatch.setattr('src.runtime.source_intelligence.start_transports',start)
    monkeypatch.setattr('src.collectors.onion_discovery.restore',lambda _:None)
    monkeypatch.setattr('src.collectors.onion_discovery._frontier',lambda _:object())
    monkeypatch.setattr('src.collectors.onion_discovery.deliver_async',deliver)
    return value,pending,retried,acked,indexed


@pytest.mark.parametrize('mode',['canonical_unavailable','onion_unavailable','canonical_error','retry_error'])
def test_immediate_retry_allows_heartbeat_before_backlog_finishes(monkeypatch,mode):
    async def scenario():
        value,pending,retried,acked,indexed=fixture(monkeypatch,mode);heartbeat=[]
        async def health():heartbeat.append(len(pending))
        await asyncio.gather(value._source_consumer_loop(),health())
        assert 0 < heartbeat[0] < 20
        assert not pending and not acked and not indexed
        assert set(retried)==set(range(20))
    asyncio.run(scenario())


@pytest.mark.parametrize('mode',['canonical_unavailable','onion_unavailable'])
def test_prestart_cancellation_does_not_consume_or_ack(monkeypatch,mode):
    async def scenario():
        value,pending,retried,acked,indexed=fixture(monkeypatch,mode)
        task=asyncio.create_task(value._source_consumer_loop())
        async def shutdown():task.cancel()
        await shutdown()
        with pytest.raises(asyncio.CancelledError):await task
        assert len(pending)==20 and not retried and not acked and not indexed
    asyncio.run(scenario())


@pytest.mark.parametrize('stage',['intake','delivery','ack'])
def test_cancel_inflight_operation_preserves_unconfirmed_event(monkeypatch,stage):
    async def scenario():
        value,pending,retried,acked,indexed=fixture(monkeypatch,'success')
        entered=asyncio.Event()
        async def blocked(*args,**kwargs):
            entered.set();await asyncio.Event().wait()
        if stage=='intake':value._record_population_narrative_async=blocked
        elif stage=='delivery':monkeypatch.setattr('src.collectors.onion_discovery.deliver_async',blocked)
        else:value.source_mesh.acknowledge_event=blocked
        task=asyncio.create_task(value._source_consumer_loop())
        await entered.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert len(pending)==19 and not retried and not acked and not indexed
    asyncio.run(scenario())


def test_successful_ack_and_index_are_not_retried_at_cancelled_yield(monkeypatch):
    async def scenario():
        value,pending,retried,acked,indexed=fixture(monkeypatch,'success')
        task=asyncio.create_task(value._source_consumer_loop())
        async def shutdown():task.cancel()
        stopper=asyncio.create_task(shutdown())
        with pytest.raises(asyncio.CancelledError):await task
        await stopper
        assert acked==indexed==[0] and not retried and len(pending)==19
    asyncio.run(scenario())


@pytest.mark.parametrize('mode',['canonical_unavailable','onion_unavailable'])
def test_retry_boundary_allows_cancel_without_losing_unacked_work(monkeypatch,mode):
    async def scenario():
        value,pending,retried,acked,indexed=fixture(monkeypatch,mode)
        task=asyncio.create_task(value._source_consumer_loop())
        async def shutdown():task.cancel()
        stopper=asyncio.create_task(shutdown())
        with pytest.raises(asyncio.CancelledError):await task
        await stopper
        assert retried==[0] and len(pending)==19 and not acked and not indexed
        assert set(pending)|set(retried)==set(range(20))
    asyncio.run(scenario())
