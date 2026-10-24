import time

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
