import asyncio,threading,time
from types import SimpleNamespace
import pytest
from src.runtime.readiness_timing import ReadinessRefreshTiming
from src.runtime.reporting import ReportingSurface


class Surface(ReportingSurface):
    def readiness(self,**kwargs):return {'mode':'DRY_RUN'}


def seeded(surface):
    surface._readiness_cache=({'mode':'old'},time.time()-1000,time.monotonic()-10)
    return surface


@pytest.mark.asyncio
async def test_real_frontier_wait_has_phase_age_and_one_refresh():
    entered,release=asyncio.Event(),asyncio.Event();calls=[];s=seeded(Surface())
    async def frontier():calls.append(1);entered.set();await release.wait();return None
    s.unknown_engine=SimpleNamespace(frontier_snapshot=frontier)
    await s._readiness_snapshot();task=s._readiness_refresh_task
    try:
        await asyncio.wait_for(entered.wait(),5)
        view=await s._readiness_snapshot();d=view['diagnostic_snapshot']
        assert d['refresh_phase']=='frontier_snapshot' and d['refresh_pending_seconds']>=0
        assert d['age_seconds']>=1000 and s._readiness_refresh_task is task and calls==[1]
        release.set();await task
        done=await s._readiness_snapshot()
        assert done['mode']=='DRY_RUN' and done['diagnostic_snapshot']['refresh_phase']=='idle'
        assert not done['diagnostic_snapshot']['refresh_in_progress']
    finally:release.set();await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_real_projection_worker_wait_is_distinct_and_coalesced():
    entered,release=threading.Event(),threading.Event();calls=[]
    class Blocked(Surface):
        def readiness(self,**kwargs):
            calls.append(1);entered.set();assert release.wait(30);return {'mode':'new'}
    s=seeded(Blocked());await s._readiness_snapshot();task=s._readiness_refresh_task
    try:
        assert await asyncio.to_thread(entered.wait,5)
        view=await s._readiness_snapshot()
        assert view['diagnostic_snapshot']['refresh_phase']=='projection_worker'
        assert s._readiness_refresh_task is task and calls==[1] and view['mode']=='old'
        release.set();await task
        assert (await s._readiness_snapshot())['mode']=='new'
    finally:release.set();await asyncio.gather(task,return_exceptions=True)


def test_actual_worker_boundaries_distinguish_dispatch_and_owner_resume():
    timing=ReadinessRefreshTiming();entered,release=threading.Event(),threading.Event()
    def work():entered.set();assert release.wait(30)
    call=timing.worker(work)
    assert timing.report()['refresh_phase']=='projection_dispatch'
    thread=threading.Thread(target=call);thread.start()
    try:
        assert entered.wait(5) and timing.report()['refresh_phase']=='projection_worker'
        release.set();thread.join(5);assert not thread.is_alive()
        assert timing.report()['refresh_phase']=='projection_owner_resume'
        timing.finish();assert timing.report()['refresh_phase']=='idle'
        assert timing.report()['refresh_pending_seconds']==0
    finally:release.set();thread.join(30)


@pytest.mark.asyncio
async def test_cancelled_frontier_clears_wait_without_extra_refresh():
    entered=asyncio.Event();s=seeded(Surface())
    async def frontier():entered.set();await asyncio.Event().wait()
    s.unknown_engine=SimpleNamespace(frontier_snapshot=frontier)
    await s._readiness_snapshot();task=s._readiness_refresh_task
    await asyncio.wait_for(entered.wait(),5);task.cancel()
    await asyncio.gather(task,return_exceptions=True)
    assert s._readiness_refresh_task is None
    assert s._readiness_refresh_timing.report()['refresh_phase']=='idle'
