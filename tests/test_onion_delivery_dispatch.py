import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.collectors.event_source import Event, SourceClass
from src.collectors.onion_discovery import deliver_async
from tests.test_onion_collection import URL


def test_plain_public_event_uses_no_worker_or_transport_write(monkeypatch):
    async def scenario():
        value = SimpleNamespace(transports={})
        event = Event('public', SourceClass.WEB, 1, 2, text='ordinary public claim')
        def forbidden(*a, **kw):pytest.fail('empty delivery dispatched a worker')
        monkeypatch.setattr('asyncio.to_thread', forbidden)
        assert await deliver_async(value, event, 'work-one') is True
        assert await deliver_async(value, event, None) is False
    asyncio.run(scenario())


@pytest.mark.parametrize('kind', ['onion', 'onion_directory'])
def test_linkless_onion_outbox_still_acknowledged_in_worker(kind):
    async def scenario():
        owner = threading.get_ident();calls=[]
        def ack(identity, job):
            assert threading.get_ident() != owner
            calls.append((identity, job))
        value = SimpleNamespace(transports={'public': SimpleNamespace(kind=kind, acknowledge=ack)})
        event = Event('public', SourceClass.WEB, 1, 2, text='linkless page',
                      metadata={'onion_observation_id': 'observation'})
        assert await deliver_async(value, event, 'work-one') is True
        assert calls == [('observation', 'work-one')]
    asyncio.run(scenario())


@pytest.mark.parametrize('field', ['text', 'urls', 'metadata'])
def test_public_discovery_material_still_admitted_before_return(field):
    async def scenario():
        owner=threading.get_ident();calls=[]
        def enqueue(urls, source, at):
            assert threading.get_ident()!=owner
            calls.append((urls, source, at))
        event=Event('public', SourceClass.WEB, 1, 2, text='ordinary')
        if field=='text':event=replace(event,text=URL)
        elif field=='urls':event=replace(event,urls=[URL])
        else:event=replace(event,metadata={'discovered_onion_links':[URL]})
        frontier=SimpleNamespace(source_id='continuous',enqueue=enqueue)
        assert await deliver_async(SimpleNamespace(transports={}),event,'work-one',frontier=frontier)
        assert calls==[([URL], 'frontier_job:work-one', 2)]
    asyncio.run(scenario())


def test_cancellation_drains_started_admission_and_ack_before_return():
    async def scenario():
        entered,released=threading.Event(),threading.Event();calls=[]
        def enqueue(*args):
            entered.set()
            assert released.wait(5)
            calls.append('commit')
        def ack(*args):calls.append('ack')
        value=SimpleNamespace(transports={'public':SimpleNamespace(kind='onion_directory',acknowledge=ack)})
        event=Event('public',SourceClass.WEB,1,2,text=URL)
        frontier=SimpleNamespace(source_id='continuous',enqueue=enqueue)
        task=asyncio.create_task(deliver_async(value,event,'work-one',frontier=frontier))
        while not entered.is_set():await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not calls
        released.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert calls==['commit','ack']
    asyncio.run(scenario())


def test_failed_child_admission_retains_parent_outbox():
    async def scenario():
        def fail(*args):raise OSError('PRIVATE_SENTINEL')
        calls=[]
        value=SimpleNamespace(transports={'public':SimpleNamespace(kind='onion_directory',acknowledge=lambda *a:calls.append(a))})
        event=Event('public',SourceClass.WEB,1,2,text=URL)
        assert await deliver_async(value,event,'work-one',frontier=SimpleNamespace(source_id='continuous',enqueue=fail)) is False
        assert calls==[]
    asyncio.run(scenario())


def test_real_consumer_plain_event_keeps_canonical_ack_and_inventory(tmp_path, monkeypatch):
    from tests.test_population_callbacks import desk
    from src.collectors.event_source import SourceMesh
    async def scenario():
        value=desk(tmp_path);mesh=SourceMesh(journal_path=tmp_path/'intake.sqlite')
        value.source_mesh=mesh;value.global_config={'ops_state_dir':str(tmp_path)}
        value.social_intel=SimpleNamespace();value._running=True;value.transports={}
        event=Event('public',SourceClass.WEB,1,2,text='ordinary public claim')
        await mesh.publish_batch([event])
        def forbidden(*a,**kw):pytest.fail('plain delivery dispatched onion worker')
        monkeypatch.setattr('src.collectors.onion_discovery.deliver',forbidden)
        indexed=[]
        def index(*a,**kw):
            assert mesh.journal.report(10)['canonical_acknowledged']==1
            indexed.append(True);value._running=False
        monkeypatch.setattr(value,'_index_source_event',index)
        try:
            await asyncio.wait_for(value._source_consumer_loop(),30)
            assert indexed==[True]
            row=mesh.journal.db.execute('SELECT body,canonical_job,canonical_ref FROM intake').fetchone()
            assert row[0] is None and row[1].startswith('work-') and row[2]
            phases=value._source_phase_timing.report()['phases']
            for name in ('canonical_ref_dispatch_wait','canonical_ref_worker',
                         'journal_ack_dispatch_wait','journal_ack_worker'):
                assert phases[name]['attempts']==phases[name]['completed']==1
        finally:
            await mesh.stop();mesh.close()
            if getattr(value,'public_claim_links',None):value.public_claim_links.close()
            value.unknown_engine.frontier.close()
    asyncio.run(scenario())
