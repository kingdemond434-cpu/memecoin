"""Bounded lease transactions preserve durability, source turns and exact ACKs."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.collectors.event_source import Event, SourceClass, SourceMesh
from src.collectors.source_journal import SourceJournal


def events(source, count):
    return [Event(source, SourceClass.NEWS, 1, 2, text=f'public {source} {i}') for i in range(count)]


def test_batch_matches_sequential_fair_turns_with_one_commit(tmp_path):
    batch = SourceJournal(tmp_path/'batch.sqlite')
    serial = SourceJournal(tmp_path/'serial.sqlite')
    for journal in (batch, serial):
        journal.append(events('hot', 30)+events('rare', 2)+events('third', 2), 10)
    statements=[]
    batch.db.set_trace_callback(statements.append)
    actual=batch.lease_many(20, limit=8)
    expected=[serial.lease(20) for _ in range(8)]
    assert actual == expected
    assert [r[0].source_id for r in actual[:3]] == ['hot','rare','third']
    assert sum(s=='COMMIT' for s in statements)==1
    assert batch.db.execute('PRAGMA synchronous').fetchone()[0]==2
    assert batch.report(21)['lease_batches_this_run']==1
    assert batch.report(21)['leased_events_this_run']==8
    assert batch.report(21)['canonical_acknowledged']==0
    batch.close();serial.close()


def test_corrupt_later_member_rolls_back_entire_batch(tmp_path):
    journal=SourceJournal(tmp_path/'journal.sqlite')
    journal.append(events('hot',2),10)
    with journal.db:
        journal.db.execute("UPDATE intake SET body='corrupt' WHERE rowid=2")
    with pytest.raises(ValueError,match='integrity'):
        journal.lease_many(20,limit=2)
    assert journal.db.execute('SELECT sum(generation),count(first_delivery) FROM intake').fetchone()==(0,0)
    assert journal.db.execute('SELECT ticket FROM source_service').fetchone()==(0,)
    assert journal.report(21)['lease_batches_this_run']==0
    assert journal.report(21)['pending_events']==2
    journal.close()


def test_restart_replays_unused_batch_with_stable_clocks_and_stale_ack_guard(tmp_path):
    path=tmp_path/'journal.sqlite';journal=SourceJournal(path)
    journal.append(events('hot',8),10)
    leased=journal.lease_many(20,limit=8,lease_seconds=5)
    event,receipt,generation=leased[0]
    journal.acknowledge(event,receipt,generation,'work-first','evidence:first',21)
    journal.close();journal=SourceJournal(path)
    replay=journal.lease_many(26,limit=8)
    assert len(replay)==7
    assert all(row[1]['first_delivered_at']==20 and row[2]==2 for row in replay)
    event,receipt,generation=leased[1]
    with pytest.raises(ValueError,match='stale'):
        journal.acknowledge(event,receipt,generation,'work-stale','evidence:stale',27)
    assert journal.report(28)['pending_events']==7
    for event,receipt,generation in replay:
        journal.acknowledge(event,receipt,generation,'work-final','evidence:final',29)
    assert journal.report(30)['pending_events']==0
    journal.close()


def test_concurrent_connections_get_disjoint_batches(tmp_path):
    path=tmp_path/'journal.sqlite';first=SourceJournal(path);second=SourceJournal(path)
    first.append(events('hot',16),10)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(j.lease_many,20,limit=8) for j in (first,second)]
        leased=[row for future in futures for row in future.result()]
    assert len(leased)==len({r[1]['id'] for r in leased})==16
    assert first.report(21)['pending_events']==16
    first.close();second.close()


@pytest.mark.asyncio
async def test_mesh_new_source_waits_at_most_one_bounded_prefetch(tmp_path):
    mesh=SourceMesh(journal_path=tmp_path/'journal.sqlite',journal_batch_size=4)
    await mesh.publish_batch(events('hot',20))
    first=await mesh.next_event()
    assert mesh.health()['prefetched_events']==3
    await mesh.publish_batch(events('rare',1))
    await mesh.acknowledge_event(first,'work-first','evidence:first')
    for _ in range(3):
        event=await mesh.next_event()
        assert event.source_id=='hot'
        await mesh.acknowledge_event(event,'work-hot','evidence:hot')
    assert (await mesh.next_event()).source_id=='rare'
    assert mesh.health()['dropped_events']==0
    mesh.close()


@pytest.mark.asyncio
async def test_concurrent_mesh_waiters_share_one_bounded_buffer(tmp_path):
    mesh=SourceMesh(journal_path=tmp_path/'journal.sqlite',journal_batch_size=4)
    await mesh.publish_batch(events('hot',8))
    delivered=await asyncio.gather(*(mesh.next_event() for _ in range(8)))
    assert len({mesh.delivery_receipt(e)['id'] for e in delivered})==8
    assert mesh.health()['prefetched_events']==0
    assert mesh.health()['lease_batches_this_run']==2
    for event in delivered:
        await mesh.acknowledge_event(event,'work-fixture','evidence:fixture')
    assert mesh.pending==0
    mesh.close()


@pytest.mark.asyncio
async def test_cancelled_batch_wait_drains_commit_and_all_raw_rows_replay(tmp_path):
    mesh=SourceMesh(journal_path=tmp_path/'journal.sqlite',journal_batch_size=4)
    await mesh.publish_batch(events('hot',4))
    entered,released=threading.Event(),threading.Event()
    original=mesh.journal.lease_many
    def blocked(*args,**kwargs):
        entered.set()
        assert released.wait(30)
        return original(*args,**kwargs)
    mesh.journal.lease_many=blocked
    task=asyncio.create_task(mesh.next_event())
    try:
        assert await asyncio.to_thread(entered.wait,10)
        task.cancel();await asyncio.sleep(0)
        assert not task.done()
    finally:
        released.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert mesh.journal.db.execute('SELECT count(body),sum(generation) FROM intake').fetchone()==(4,4)
    assert mesh.health()['canonical_acknowledged']==0
    assert mesh.health()['prefetched_events']==0
    deadline=mesh.journal.db.execute('SELECT max(due) FROM intake').fetchone()[0]
    assert len(original(deadline+1,limit=4))==4
    mesh.close()


@pytest.mark.parametrize('limit',[0,33,True,1.5])
def test_invalid_batch_capacity_is_refused(tmp_path,limit):
    with pytest.raises(ValueError):SourceMesh(journal_batch_size=limit)
    journal=SourceJournal(tmp_path/'journal.sqlite')
    with pytest.raises(ValueError):journal.lease_many(20,limit=limit)
    journal.close()

@pytest.mark.parametrize('now,duration',[(20.,1e-300),(1e308,1e308)])
def test_deadline_must_advance_without_overflow(tmp_path,now,duration):
    journal=SourceJournal(tmp_path/'journal.sqlite')
    journal.append(events('hot',1),10)
    with pytest.raises(ValueError,match='deadline'):
        journal.lease_many(now,lease_seconds=duration)
    assert journal.db.execute('SELECT generation FROM intake').fetchone()==(0,)
    journal.close()


@pytest.mark.asyncio
async def test_closed_mesh_unused_buffer_remains_recoverable(tmp_path):
    path=tmp_path/'journal.sqlite'
    mesh=SourceMesh(journal_path=path,journal_batch_size=4)
    await mesh.publish_batch(events('hot',4))
    event=await mesh.next_event()
    await mesh.acknowledge_event(event,'work-first','evidence:first')
    assert mesh.health()['prefetched_events']==3
    mesh.close()
    journal=SourceJournal(path)
    deadline=journal.db.execute('SELECT max(due) FROM intake').fetchone()[0]
    assert len(journal.lease_many(deadline+1))==3
    assert journal.report(deadline+1)['canonical_acknowledged']==1
    journal.close()
