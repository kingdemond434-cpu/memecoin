import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.runtime.offload import OffloadedPool


def test_actual_drainer_yields_owner_turn_and_preserves_fifo_tail():
    async def scenario():
        seen, observed = [], []
        clock = [0.0]; ready = asyncio.Event(); loop = asyncio.get_running_loop()
        owner = threading.get_ident()
        def sink(name, rows):
            assert threading.get_ident() == owner
            seen.append(rows[0]); clock[0] += .02
            if len(seen) == 1:
                loop.call_soon(lambda: (observed.append(list(seen)), ready.set()))
        pool = OffloadedPool(SimpleNamespace(), sink=sink, drain_interval_s=.05)
        for n in range(50):
            pool._publish('fixture', [n])
        pool._running = True
        with patch('src.runtime.offload.time.perf_counter', side_effect=lambda: clock[0]):
            task = asyncio.create_task(pool._drain_loop())
            try:
                await asyncio.wait_for(ready.wait(), 2)
                assert observed == [[0]]
                assert pool._queue.qsize() == 49 and pool.dropped == 0
            finally:
                pool._running = False; task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert pool.drain() == 49  # Manual count-only contract retained.
        assert seen == list(range(50))
        report = pool.report()
        assert report['delivered'] == report['sink_returned'] == 50
        assert report['sink_errors'] == report['sink_interrupted'] == 0
        assert report['sink_seconds_total'] == pytest.approx(1.)
        assert report['sink_seconds_max'] == pytest.approx(.02)
    asyncio.run(scenario())


def test_sink_return_failure_and_interruption_are_distinct():
    def sink(name, rows):
        if rows[0] == 1:
            raise ValueError('fixture')
        if rows[0] == 2:
            raise asyncio.CancelledError()
    pool = OffloadedPool(SimpleNamespace(), sink=sink)
    for n in range(4):
        pool._publish('fixture', [n])
    with pytest.raises(asyncio.CancelledError):
        pool.drain()
    report = pool.report()
    assert (report['delivered'], report['sink_returned'], report['sink_errors'], report['sink_interrupted']) == (3,1,1,1)
    assert pool._queue.qsize() == 1 and pool.dropped == 0
    assert pool.drain() == 1
    assert pool.report()['sink_returned'] == 2
    assert 'not durable' in report['delivery_scope']


@pytest.mark.parametrize('quantum', [0, -1, float('nan'), float('inf'), True])
def test_invalid_quantum_does_not_dequeue(quantum):
    pool = OffloadedPool(SimpleNamespace(), sink=lambda *_: None)
    pool._publish('fixture', [1])
    with pytest.raises(ValueError):
        pool.drain(max_seconds=quantum)
    assert pool._queue.qsize() == 1 and pool.delivered == 0
