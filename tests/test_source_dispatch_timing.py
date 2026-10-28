import asyncio
import threading
from unittest.mock import patch

import pytest

from src.collectors.event_source import SourceMesh
from src.runtime.source_timing import SourceTiming


def test_exact_dispatch_and_worker_clocks_result_and_cardinality():
    timing=SourceTiming()
    with patch('src.runtime.source_timing.time.perf_counter',side_effect=[1,3,4,9,10]):
        call=timing.worker('canonical_ref',lambda a:a+1,4)
        assert call()==5
    phases=timing.report()['phases']
    assert set(phases)=={'canonical_ref_dispatch_wait','canonical_ref_worker'}
    assert phases['canonical_ref_dispatch_wait']['total_seconds']==2
    assert phases['canonical_ref_worker']['total_seconds']==5
    assert all(v['completed']==1 for v in phases.values())
    with pytest.raises(ValueError):timing.worker('PRIVATE_SENTINEL',lambda:None)
    assert 'PRIVATE_SENTINEL' not in timing.report()['phases']


def test_worker_failure_is_not_dispatch_failure_or_durable_success():
    timing=SourceTiming()
    def fail():raise OSError('fixture')
    with pytest.raises(OSError):timing.worker('journal_ack',fail)()
    phases=timing.report()['phases']
    assert phases['journal_ack_dispatch_wait']['completed']==1
    assert phases['journal_ack_worker']['interrupted']==1
    assert phases['journal_ack_worker']['completed']==0


def test_timed_disk_call_drains_cancelled_worker_before_return():
    async def scenario():
        entered,released=threading.Event(),threading.Event();finished=[];timing=SourceTiming()
        def write():
            entered.set();assert released.wait(5);finished.append(True)
        task=asyncio.create_task(SourceMesh._disk_call(write,timing=timing,phase='journal_ack'))
        while not entered.is_set():await asyncio.sleep(.001)
        task.cancel();await asyncio.sleep(0)
        assert not task.done() and not finished
        released.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert finished==[True]
        phases=timing.report()['phases']
        assert phases['journal_ack_worker']['completed']==1
        assert phases['journal_ack_worker']['interrupted']==0
        assert phases['journal_ack_owner_resume']['interrupted']==1
    asyncio.run(scenario())


def test_timed_actual_journal_ack_retains_full_exact_receipt_contract(tmp_path):
    from tests.test_source_intake_nonblocking import event
    async def scenario():
        mesh=SourceMesh(journal_path=tmp_path/'journal.sqlite');timing=SourceTiming()
        try:
            await mesh.publish_batch([event()]);item=await mesh.next_event()
            await mesh.acknowledge_event(item,'work-one','evidence:one',timing=timing)
            assert mesh.journal.report(20)['canonical_acknowledged']==1
            assert not mesh._journal_leases
            assert mesh.journal.db.execute('PRAGMA synchronous').fetchone()[0]==2
            phases=timing.report()['phases']
            assert phases['journal_ack_worker']['completed']==1
            assert phases['journal_ack_dispatch_wait']['completed']==1
            with pytest.raises(KeyError):await mesh.acknowledge_event(item,'work-one','evidence:one',timing=timing)
        finally:
            await mesh.stop();mesh.close()
    asyncio.run(scenario())


def test_resumption_clock_is_measured_directly_not_parent_subtraction():
    timing = SourceTiming()
    with patch('src.runtime.source_timing.time.perf_counter', side_effect=[1,3,4,9,10,17]):
        worker = timing.worker('canonical_ack', lambda: 'value')
        assert worker() == 'value'
        timing.resumed('canonical_ack', worker, True)
    phases = timing.report()['phases']
    assert phases['canonical_ack_owner_resume']['total_seconds'] == 7
    assert phases['canonical_ack_worker']['total_seconds'] == 5
    assert phases['canonical_ack_dispatch_wait']['total_seconds'] == 2


def test_undispatched_work_has_no_invented_resumption():
    timing = SourceTiming()
    worker = timing.worker('canonical_ack', lambda: None)
    timing.resumed('canonical_ack', worker, False)
    assert timing.report()['phases'] == {}
    with pytest.raises(ValueError):
        timing.resumed('PRIVATE_SENTINEL', worker, False)


def test_worker_error_reaches_caller_and_records_interrupted_return():
    async def scenario():
        timing = SourceTiming()
        def fail():
            raise OSError('fixture')
        with pytest.raises(OSError, match='fixture'):
            await SourceMesh._disk_call(fail, timing=timing, phase='canonical_ack')
        phases = timing.report()['phases']
        assert phases['canonical_ack_worker']['interrupted'] == 1
        assert phases['canonical_ack_owner_resume']['interrupted'] == 1
        assert phases['canonical_ack_owner_resume']['attempts'] == 1
    asyncio.run(scenario())
