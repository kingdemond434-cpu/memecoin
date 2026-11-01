import asyncio
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import pytest

from src.collectors.event_source import Event, SourceClass, SourceMesh
from src.collectors.source_journal import SourceJournal


def event(source='feed', number=0, **kwargs):
    return Event(source, SourceClass.NEWS, 1., 2., text=f'public {source}:{number}', **kwargs)


def test_restart_retains_every_overflow_and_rejects_stale_or_changed_ack(tmp_path):
    path=tmp_path/'intake.sqlite'
    journal=SourceJournal(path)
    journal.append([event(number=i) for i in range(2000)], 100.)
    first, receipt, generation=journal.lease(110., 5.)
    journal.close()
    journal=SourceJournal(path)
    # Retry the same observation after a process loss. A first-delivery clock
    # is stable, and the old lease cannot acknowledge the newer attempt.
    with journal.db:journal.db.execute('UPDATE intake SET due=1000 WHERE id!=?',(receipt['id'],))
    assert journal.lease(114.) is None
    replay, replay_receipt, newer=journal.lease(116.)
    assert replay==first and replay_receipt==receipt and newer==generation+1
    with pytest.raises(ValueError):journal.acknowledge(first,receipt,generation,'work-job','evidence:ref',117.)
    with pytest.raises(ValueError):journal.acknowledge(replace(replay,text='changed'),receipt,newer,'work-job','evidence:ref',117.)
    assert journal.report(117.)['pending_events']==2000
    journal.acknowledge(replay,receipt,newer,'work-job','evidence:ref',117.)
    report=journal.report(118.)
    assert report['pending_events']==1999 and report['canonical_acknowledged']==1
    assert report['accepted_observations']==2000 and report['evicted_events']==0
    journal.append([first],119.)
    assert journal.report(120.)['accepted_observations']==2000
    journal.close()


def test_rare_sources_keep_a_turn_and_onion_outbox_repeats_share_one_row(tmp_path):
    journal=SourceJournal(tmp_path/'intake.sqlite')
    journal.append([event('hot',i) for i in range(3000)],10.)
    onion=event('onion',metadata={'onion_observation_id':'observation'})
    journal.append([onion,replace(onion,observed_at=12.),event('rare')],11.)
    received=[]
    for n in range(3):
        value,receipt,generation=journal.lease(20.+n)
        received.append(value.source_id)
        journal.acknowledge(value,receipt,generation,'work-'+value.source_id,'evidence:ref',25.+n)
        journal.append([event('hot',4000+n)],20.+n)
    assert set(received)=={'hot','onion','rare'}
    assert journal.report(30.)['accepted_observations']==3005
    journal.close()


def test_corruption_and_noncanonical_ack_never_remove_observations(tmp_path):
    journal=SourceJournal(tmp_path/'intake.sqlite')
    journal.append([event()],10.)
    value,receipt,generation=journal.lease(20.)
    with pytest.raises(ValueError):journal.acknowledge(value,receipt,generation,'','',21.)
    with pytest.raises(ValueError):journal.acknowledge(value,receipt,generation,'work-job','ref',19.)
    with journal.db:journal.db.execute("UPDATE intake SET body=replace(body,'public','altered'),due=0")
    with pytest.raises(ValueError):journal.acknowledge(value,receipt,generation,'work-job','ref',21.)
    with pytest.raises(ValueError,match='integrity'):journal.lease(22.)
    assert journal.report(22.)['pending_events']==1
    journal.close()


@pytest.mark.asyncio
async def test_disk_commit_is_nonblocking_and_cancellation_drains_it(tmp_path):
    # Load test server utilities before arming the timed blocked-write fixture.
    from aiohttp import web
    from aiohttp.test_utils import TestClient,TestServer
    mesh=SourceMesh(max_queue=2,journal_path=tmp_path/'intake.sqlite')
    entered,released=threading.Event(),threading.Event()
    original=mesh.journal.append
    owner=threading.get_ident()
    def blocked(*args):
        assert threading.get_ident()!=owner
        entered.set();assert released.wait(5)
        return original(*args)
    mesh.journal.append=blocked
    task=asyncio.create_task(mesh.publish_batch([event(number=i) for i in range(100)]))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        async def health(request):return web.json_response({'healthy':True})
        app=web.Application();app.router.add_get('/health',health)
        async with TestClient(TestServer(app)) as client:
            assert (await (await client.get('/health')).json())['healthy']
        task.cancel();await asyncio.sleep(0);assert not task.done()
    finally:released.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert mesh.pending==100 and mesh.health()['dropped_events']==0
    mesh.close()


@pytest.mark.asyncio
async def test_failed_write_retries_admitted_batch_without_dedupe_loss(tmp_path):
    mesh=SourceMesh(journal_path=tmp_path/'intake.sqlite')
    original=mesh.journal.append;calls=[]
    def fail_once(*args):
        calls.append(1)
        if len(calls)==1:raise OSError('disk fixture')
        return original(*args)
    mesh.journal.append=fail_once
    await mesh.publish_batch([event()])
    leased=await asyncio.wait_for(mesh.next_event(),2)
    assert leased.text==event().text and len(calls)==2
    assert mesh.health()['journal_write_failures']==1
    await mesh.acknowledge_event(leased,'work-job','evidence:ref')
    assert mesh.pending==0 and mesh.health()['canonical_acknowledged']==1
    mesh.close()


@pytest.mark.asyncio
async def test_runtime_ack_is_bound_to_real_canonical_job_and_late_availability(tmp_path,monkeypatch):
    from tests.test_population_callbacks import desk
    value=desk(tmp_path)
    mesh=SourceMesh(max_queue=1,journal_path=tmp_path/'intake.sqlite')
    value.source_mesh=mesh;value.global_config={'ops_state_dir':str(tmp_path)}
    value.social_intel=SimpleNamespace();value._running=True;value.transports={}
    source=event('rss:test',metadata={'source_intake_receipt':{'first_delivered_at':0},'research_available_at':0})
    await mesh.publish_batch([source])
    observed=[]
    def index(value_event,**kwargs):
        assert mesh.health()['canonical_acknowledged']==1
        observed.append(value_event);value._running=False
    value._index_source_event=index
    await asyncio.wait_for(value._source_consumer_loop(),60)
    row=mesh.journal.db.execute('SELECT canonical_job,canonical_ref,first_delivery FROM intake').fetchone()
    job=value.unknown_engine.frontier.job(row[0])
    assert job['payload_ref']==row[1]
    payload=value.unknown_engine._load_refs({row[1]})[row[1]]['point_payload']
    assert payload['observed_at']==source.observed_at
    assert payload['research_available_at']==row[2]>source.observed_at
    assert payload['source_intake_receipt']['first_delivered_at']==row[2]
    assert len(observed)==1 and mesh.pending==0
    await mesh.stop();mesh.close();value.public_claim_links.close();value.unknown_engine.frontier.close()


def test_concurrent_connections_never_lease_the_same_generation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path=tmp_path/'intake.sqlite';first=SourceJournal(path);second=SourceJournal(path)
    first.append([event(number=n) for n in range(20)],10.)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(journal.lease,20.) for journal in (first,second)]
        leases=[future.result() for future in futures]
    assert leases[0][1]['id']!=leases[1][1]['id']
    assert first.report(21.)['pending_events']==20
    first.close();second.close()


@pytest.mark.asyncio
async def test_failed_canonical_intake_releases_lease_without_ack(tmp_path,monkeypatch):
    from tests.test_population_callbacks import desk
    value=desk(tmp_path);mesh=SourceMesh(journal_path=tmp_path/'intake.sqlite')
    value.source_mesh=mesh;value.social_intel=SimpleNamespace();value.transports={};value._running=True
    await mesh.publish_batch([event()])
    async def fail(_):
        value._running=False
        return None
    value._record_population_narrative_async=fail
    value._index_source_event=lambda *args,**kwargs:pytest.fail('failed intake must not be indexed or acknowledged')
    await asyncio.wait_for(value._source_consumer_loop(),10)
    assert mesh.pending==1 and mesh.health()['canonical_acknowledged']==0
    assert mesh.journal.db.execute('SELECT canonical_ref FROM intake').fetchone()[0] is None
    assert mesh.delivery_receipt(event()) is None
    await mesh.stop();mesh.close();value.unknown_engine.frontier.close()
