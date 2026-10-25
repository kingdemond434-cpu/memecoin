import time
import asyncio

import pytest

from src.research.data_miners import DataMinerPool, Enriches, MinerSpec


@pytest.mark.asyncio
async def test_full_batch_retained_before_chunked_delivery_without_losing_tail():
    retained, delivered, dispositions = [], [], []
    def retain(name, rows, now):
        retained.extend(rows)
        return "evidence:batch:0"
    pool = DataMinerPool(retain_batch=retain,
        retain_disposition=lambda *args: dispositions.append(args),
        on_records=lambda name, rows: delivered.extend(rows))
    async def fetch():
        return [{"index": i} for i in range(5)]
    pool.register(MinerSpec("fixture", Enriches.NARRATIVE, 60, max_records=2), fetch)
    result = (await pool.run_due(time.time() + 100))[0]
    assert [row["index"] for row in retained] == list(range(5))
    assert len(delivered) == 5 and result.status == "OK"
    assert dispositions[0][1] == "DELIVERED"
    assert "deferred=0" in dispositions[0][2]


@pytest.mark.asyncio
async def test_failed_durable_write_blocks_delivery():
    delivered = []
    def broken(*args):
        raise OSError("disk full")
    pool = DataMinerPool(retain_batch=broken,
                         on_records=lambda *args: delivered.append(args))
    async def fetch():
        return [{"index": 0}]
    pool.register(MinerSpec("fixture", Enriches.NARRATIVE, 60), fetch)
    result = (await pool.run_due(time.time() + 100))[0]
    assert result.status == "DATA_BLOCKED"
    assert not delivered


@pytest.mark.asyncio
async def test_consumer_failure_preserves_batch_and_retry_reference():
    receipts = []
    def fail(*args):
        raise ValueError("consumer unavailable")
    pool = DataMinerPool(retain_batch=lambda *args: "evidence:retained:0",
        retain_disposition=lambda *args: receipts.append(args), on_records=fail)
    async def fetch():
        return [{"index": 0}]
    pool.register(MinerSpec("fixture", Enriches.NARRATIVE, 60), fetch)
    result = (await pool.run_due(time.time() + 100))[0]
    assert result.status == "DATA_BLOCKED"
    assert receipts[0][0:2] == ("evidence:retained:0", "RETRY_REQUIRED")


@pytest.mark.asyncio
async def test_slow_source_does_not_hold_unrelated_sources_to_its_cadence():
    held, fast_twice = asyncio.Event(), asyncio.Event()
    fast_calls = 0
    cancelled = asyncio.Event()
    async def slow():
        held.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async def fast():
        nonlocal fast_calls
        fast_calls += 1
        if fast_calls == 2: fast_twice.set()
        return [{'reading':fast_calls}]
    pool = DataMinerPool(concurrency=2)
    pool.register(MinerSpec('slow',Enriches.NARRATIVE,60),slow)
    pool.register(MinerSpec('fast',Enriches.MARKET_CONTEXT,.01),fast)
    pool._next_due = {'slow':0,'fast':0}
    await pool.start()
    try:
        await asyncio.wait_for(held.wait(), 2)
        await asyncio.wait_for(fast_twice.wait(), 3)
        assert not cancelled.is_set() and fast_calls >= 2
    finally:
        await pool.stop()
    assert cancelled.is_set() and not pool._inflight and not pool._active


@pytest.mark.asyncio
async def test_acquisition_workflow_cannot_overlap_same_miner_already_running():
    held, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def fetch():
        nonlocal calls
        calls += 1
        held.set()
        await release.wait()
        return [{'reading':1}]
    pool = DataMinerPool()
    pool.register(MinerSpec('one',Enriches.NARRATIVE,1),fetch)
    task = asyncio.create_task(pool._run_one('one',time.time()))
    await held.wait()
    duplicate = await pool._run_one('one',time.time()+10)
    assert duplicate.status == 'IDLE' and calls == 1
    release.set()
    assert (await task).ok
