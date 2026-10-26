import asyncio
import threading

import pytest

from src.collectors.event_source import Event, SourceClass, SourceMesh


def event(source, number, *, durable=False):
    return Event(source, SourceClass.WEB, 1., 2., text=f'{source}:{number}',
                 metadata={'onion_observation_id': str(number)} if durable else {})


@pytest.mark.asyncio
async def test_a_hot_backlog_cannot_delay_a_durable_source_until_the_queue_drains():
    mesh = SourceMesh(max_queue=100)
    for number in range(100):
        mesh._publish(event('hot', number), 2.)
    cold = event('onion:continuous-frontier', 'claim', durable=True)
    mesh._publish(cold, 3.)
    for number in range(100, 1000):
        mesh._publish(event('hot', number), 4.)
    assert mesh.pending == 100
    assert mesh.health(4.)['pending_by_source'] == {'hot': 99, cold.source_id: 1}
    # The rare observation survives overflow, then receives its turn without
    # waiting for 99 hotter observations or treating dequeue as a durable ACK.
    assert (await mesh.next_event()).source_id == 'hot'
    assert await mesh.next_event() == cold
    assert mesh.health(4.)['dropped_by_source'] == {'hot': 901}
    mesh._publish(cold, 5.)
    assert cold in mesh.drain(limit=2)


def test_each_nonempty_source_gets_a_turn_under_continuous_hot_refill():
    mesh = SourceMesh(max_queue=200)
    for number in range(100):
        mesh._publish(event('hot', number), 2.)
    sources = [f'public:{number}' for number in range(30)]
    for source in sources:
        mesh._publish(event(source, 'rare'), 3.)
    received = []
    for turn in range(len(sources)+1):
        received.extend(row.source_id for row in mesh.drain(limit=1))
        for number in range(20):
            mesh._publish(event('hot', f'refill:{turn}:{number}'), 4.)
    assert set(received) == {'hot', *sources}
    assert len(received) == len(set(received))
    assert mesh.pending <= mesh.max_queue
    assert mesh.health(5.)['delivered_by_source'] == dict.fromkeys(received, 1)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_consume_the_next_observation():
    mesh = SourceMesh()
    waiter = asyncio.create_task(mesh.next_event())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    next_waiter = asyncio.create_task(mesh.next_event())
    await asyncio.sleep(0)
    fresh = event('public', 'new')
    mesh._publish(fresh, 2.)
    assert await asyncio.wait_for(next_waiter, 1) == fresh
    assert mesh.pending == 0
    assert mesh.health(3.)['pending_by_source'] == {}


@pytest.mark.asyncio
async def test_two_waiters_receive_distinct_events_and_remaining_waiter_keeps_waiting():
    mesh = SourceMesh()
    first = asyncio.create_task(mesh.next_event())
    second = asyncio.create_task(mesh.next_event())
    await asyncio.sleep(0)
    one = event('one', 1)
    mesh._publish(one, 2.)
    done, pending = await asyncio.wait([first, second], timeout=1, return_when=asyncio.FIRST_COMPLETED)
    assert len(done) == len(pending) == 1
    assert next(iter(done)).result() == one
    two = event('two', 2)
    mesh._publish(two, 3.)
    assert await asyncio.wait_for(next(iter(pending)), 1) == two
    assert mesh.pending == 0


@pytest.mark.asyncio
async def test_worker_health_reads_are_consistent_during_owner_loop_mutation():
    mesh = SourceMesh(max_queue=20)
    started, stop = threading.Event(), threading.Event()
    def snapshots():
        seen = 0
        started.set()
        while not stop.is_set():
            report = mesh.health(10.)
            assert sum(report['pending_by_source'].values()) == report['pending_events']
            assert sum(report['dropped_by_source'].values()) == report['dropped_events']
            assert set(report['pending_by_source']) == set(report['oldest_pending_age_by_source'])
            seen += 1
        return seen
    worker = asyncio.create_task(asyncio.to_thread(snapshots))
    try:
        while not started.is_set():await asyncio.sleep(0)
        for number in range(1000):
            mesh._publish(event(f'source:{number % 17}', number), 2.)
            if number % 3 == 0:mesh.drain(limit=1)
            await asyncio.sleep(0)
    finally:stop.set()
    assert await worker > 0
