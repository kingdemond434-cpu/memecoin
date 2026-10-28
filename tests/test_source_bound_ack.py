import asyncio
import threading

import pytest

from src.collectors.event_source import SourceMesh
from src.runtime.source_timing import SourceTiming
from tests.test_source_intake_nonblocking import event


async def ready(path):
    mesh=SourceMesh(journal_path=path)
    await mesh.publish_batch([event()])
    return mesh,await mesh.next_event()


def test_binding_then_full_ack_uses_one_worker_dispatch(tmp_path,monkeypatch):
    async def scenario():
        mesh,item=await ready(tmp_path/'journal.sqlite');timing=SourceTiming();calls=[];owner=threading.get_ident()
        original=mesh.journal.acknowledge;disk=mesh._disk_call;dispatches=[]
        async def dispatch(*args,**kw):
            dispatches.append(True);return await disk(*args,**kw)
        def lookup(job):
            assert job=='work-one' and threading.get_ident()!=owner
            calls.append(('lookup',threading.get_ident()));return 'evidence:one'
        def ack(*args):
            calls.append(('ack',threading.get_ident()));return original(*args)
        monkeypatch.setattr(mesh,'_disk_call',dispatch);monkeypatch.setattr(mesh.journal,'acknowledge',ack)
        try:
            await mesh.acknowledge_event(item,'work-one',lookup=lookup,timing=timing)
            assert len(dispatches)==1
            assert calls[0][0]=='lookup' and calls[1][0]=='ack' and calls[0][1]==calls[1][1]
            assert mesh.journal.report(20)['canonical_acknowledged']==1 and not mesh._journal_leases
            assert mesh.journal.db.execute('PRAGMA synchronous').fetchone()[0]==2
            phases=timing.report()['phases']
            assert set(phases)=={'canonical_ack_dispatch_wait','canonical_ack_worker','canonical_ref_inline','journal_ack_inline'}
            assert all(row['completed']==1 for row in phases.values())
        finally:await mesh.stop();mesh.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('failure',['missing','lookup_error','stale'])
def test_failed_binding_or_stale_lease_retains_raw_body_and_counters(tmp_path,failure):
    async def scenario():
        mesh,item=await ready(tmp_path/'journal.sqlite')
        def lookup(job):
            if failure=='lookup_error':raise OSError('fixture')
            return None if failure=='missing' else 'evidence:one'
        if failure=='stale':
            with mesh.journal.db:mesh.journal.db.execute('UPDATE intake SET generation=generation+1')
        try:
            with pytest.raises((ValueError,OSError)):
                await mesh.acknowledge_event(item,'work-one',lookup=lookup)
            assert mesh.journal.report(20)['canonical_acknowledged']==0
            assert mesh.journal.db.execute('SELECT body FROM intake').fetchone()[0] is not None
            assert mesh.delivery_receipt(item) is not None
        finally:await mesh.stop();mesh.close()
    asyncio.run(scenario())


def test_cancelled_started_binding_drains_exact_ack_before_owner_can_close(tmp_path):
    async def scenario():
        mesh,item=await ready(tmp_path/'journal.sqlite');entered,released=threading.Event(),threading.Event()
        def lookup(job):
            entered.set();assert released.wait(5);return 'evidence:one'
        try:
            task=asyncio.create_task(mesh.acknowledge_event(item,'work-one',lookup=lookup))
            while not entered.is_set():await asyncio.sleep(.001)
            task.cancel();await asyncio.sleep(0)
            assert not task.done() and mesh.journal.report(20)['canonical_acknowledged']==0
            released.set()
            with pytest.raises(asyncio.CancelledError):await task
            assert mesh.journal.report(20)['canonical_acknowledged']==1
            assert mesh.journal.db.execute('SELECT body FROM intake').fetchone()[0] is None
        finally:released.set();await mesh.stop();mesh.close()
    asyncio.run(scenario())


def test_lookup_and_explicit_reference_are_not_ambiguously_combined(tmp_path):
    async def scenario():
        mesh,item=await ready(tmp_path/'journal.sqlite')
        def forbidden(job):pytest.fail('ambiguous lookup executed')
        try:
            with pytest.raises(ValueError):
                await mesh.acknowledge_event(item,'work-one','evidence:one',lookup=forbidden)
            assert mesh.journal.report(20)['canonical_acknowledged']==0
            assert mesh.journal.db.execute('SELECT body FROM intake').fetchone()[0] is not None
        finally:await mesh.stop();mesh.close()
    asyncio.run(scenario())
