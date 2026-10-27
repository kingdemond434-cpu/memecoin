"""Claim commits cannot stall source health or expose uncommitted claims."""
import asyncio
import threading
import sqlite3

import pytest

from src.research.public_claim_links import PublicClaimLinks
from tests.test_population_callbacks import desk
from tests.test_source_intake_nonblocking import event


def test_claim_commit_yields_and_readers_see_only_committed_wal_snapshot(tmp_path):
    async def scenario():
        from aiohttp import web
        from aiohttp.test_utils import TestClient,TestServer
        value=desk(tmp_path);value.global_config={'ops_state_dir':str(tmp_path)}
        store=PublicClaimLinks(tmp_path/'claims.sqlite');value.public_claim_links=store
        owner=threading.get_ident();entered,released=threading.Event(),threading.Event()
        original=store.record;threads=[]
        def record(**kwargs):
            assert threading.get_ident()!=owner
            return original(**kwargs)
        store.record=record
        def trace(sql):
            if sql=='COMMIT':
                threads.append(threading.get_ident());entered.set();released.wait(30)
        store.db.set_trace_callback(trace)
        async def health(request):
            return web.json_response({'healthy':True,'visible_links':len(store.page()['items']),
                                      'reported_links':store.report()['links']})
        app=web.Application();app.router.add_get('/health',health)
        task=asyncio.create_task(value._record_population_narrative_async(event()))
        try:
            assert await asyncio.to_thread(entered.wait,10)
            async with TestClient(TestServer(app)) as client:
                response=await asyncio.wait_for(client.get('/health'),5)
                assert await response.json()=={'healthy':True,'visible_links':0,'reported_links':0}
        finally:released.set()
        job=await task
        assert threads and all(t!=owner for t in threads)
        assert store.page()['items'][0]['source_job']==job
        store.close();value.unknown_engine.frontier.close()
    asyncio.run(scenario())


def test_cancelled_claim_commit_drains_before_source_can_ack_or_store_closes(tmp_path):
    async def scenario():
        from src.collectors.event_source import SourceMesh
        value=desk(tmp_path);value.global_config={'ops_state_dir':str(tmp_path)}
        store=PublicClaimLinks(tmp_path/'claims.sqlite');value.public_claim_links=store
        mesh=SourceMesh(journal_path=tmp_path/'journal.sqlite');value.source_mesh=mesh
        await mesh.publish_batch([event()]);source=await mesh.next_event()
        entered,released=threading.Event(),threading.Event();original=store.record
        def blocked(**kwargs):
            entered.set();assert released.wait(30)
            return original(**kwargs)
        store.record=blocked
        task=asyncio.create_task(value._record_population_narrative_async(source))
        try:
            assert await asyncio.to_thread(entered.wait,10)
            task.cancel();await asyncio.sleep(0)
            assert not task.done()
            assert store.page()['items']==[]
        finally:released.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert len(store.page()['items'])==1
        assert mesh.health()['canonical_acknowledged']==0 and mesh.pending==1
        store.close();mesh.close();value.unknown_engine.frontier.close()
    asyncio.run(scenario())


def test_serialized_concurrent_writers_and_independent_readers_preserve_idempotency(tmp_path):
    async def scenario():
        store=PublicClaimLinks(tmp_path/'claims.sqlite')
        args={'source_job':'work-fixture','source_id':'public:fixture',
              'addresses':['0x'+'a'*40],'topics':['claim'],'observed_at':10}
        await asyncio.gather(*(asyncio.to_thread(store.record,**args) for _ in range(20)))
        results=await asyncio.gather(*(asyncio.to_thread(store.page) for _ in range(4)))
        assert all(len(r['items'])==1 for r in results)
        assert store.report()['links']==1 and not store.report()['insider_identity_verified']
        store.close()
        with pytest.raises(sqlite3.ProgrammingError):store.page()
    asyncio.run(scenario())

def test_existing_narrative_hook_runs_on_owner_after_exactly_one_claim_write(tmp_path):
    async def scenario():
        value=desk(tmp_path);value.global_config={'ops_state_dir':str(tmp_path)}
        store=PublicClaimLinks(tmp_path/'claims.sqlite');value.public_claim_links=store
        owner=threading.get_ident();order=[];original_record=store.record
        def record(**kwargs):
            assert threading.get_ident()!=owner
            result=original_record(**kwargs);order.append('claim')
            return result
        store.record=record;original_hook=value._record_population_narrative
        def hook(source,**kwargs):
            assert threading.get_ident()==owner
            assert len(store.page()['items'])==1
            order.append('owner')
            return original_hook(source,**kwargs)
        value._record_population_narrative=hook
        job=await value._record_population_narrative_async(event())
        assert order==['claim','owner']
        assert store.page()['items'][0]['source_job']==job
        store.close();value.unknown_engine.frontier.close()
    asyncio.run(scenario())
