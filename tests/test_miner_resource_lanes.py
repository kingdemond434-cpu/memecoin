"""Prove external waits cannot occupy the offline inference execution slot."""
import asyncio
import time

import pytest

from src.research.data_miners import DataMinerPool, Enriches, MinerSpec


@pytest.mark.asyncio
async def test_offline_work_completes_while_network_slot_remains_held():
    held, release = asyncio.Event(), asyncio.Event()
    network_calls = []
    async def network():
        network_calls.append(1)
        held.set()
        await release.wait()
        return [{'source':'network'}]
    async def local():
        return [{'source':'offline'}]
    retained=[]
    pool=DataMinerPool(concurrency=1, retain_batch=lambda name,rows,now: retained.append(name) or 'canonical:'+name)
    for name in ('network:a','network:b'):
        pool.register(MinerSpec(name,Enriches.NARRATIVE,1),network)
    pool.register(MinerSpec('local',Enriches.NARRATIVE,1,resource_class='LOCAL_CPU'),local)
    first=asyncio.create_task(pool._run_one('network:a',time.time()))
    await asyncio.wait_for(held.wait(),1)
    second=asyncio.create_task(pool._run_one('network:b',time.time()))
    try:
        result=await asyncio.wait_for(pool._run_one('local',time.time()),1)
        assert result.ok and retained == ['local']
        assert network_calls == [1] and not first.done() and not second.done()
        report=pool.report()
        local_row=next(row for row in report['miners'] if row['miner_id']=='local')
        assert local_row['resource_class']=='LOCAL_CPU'
        assert local_row['last_queue_wait_ms'] >= 0 and local_row['last_fetch_ms'] >= 0
        assert report['resource_concurrency']=={'NETWORK':1,'LOCAL_CPU':1}
    finally:
        release.set()
        await asyncio.gather(first,second)


@pytest.mark.asyncio
async def test_cpu_lane_stays_serial_and_shutdown_drains_running_and_queued_work():
    held=asyncio.Event()
    entered=[]
    async def local():
        entered.append(1)
        held.set()
        await asyncio.Event().wait()
    pool=DataMinerPool(concurrency=2)
    for name in ('local:a','local:b'):
        pool.register(MinerSpec(name,Enriches.NARRATIVE,1,resource_class='LOCAL_CPU'),local)
    pool._next_due={'local:a':0,'local:b':0}
    await pool.run_due(wait=False)
    await asyncio.wait_for(held.wait(),1)
    await asyncio.sleep(0)
    assert entered == [1] and len(pool._active)==2
    await pool.stop()
    assert not pool._inflight and not pool._active


def test_unknown_resource_declaration_is_not_registered():
    pool=DataMinerPool()
    with pytest.raises(ValueError, match='resource'):
        pool.register(MinerSpec('bad',Enriches.NARRATIVE,1,resource_class='arbitrary'),None)
    assert not pool._specs and not pool._callables
