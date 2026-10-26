import asyncio
import threading
import time
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
from src.runtime.reporting import ReportingSurface


@pytest.mark.asyncio
async def test_full_projection_is_coalesced_without_blocking_health():
    entered,release=threading.Event(),threading.Event();calls=[]
    class Surface(ReportingSurface):
        _starting_phase=''
        _running=True
        dry_run=True
        start_time=time.time()
        def readiness(self):
            calls.append(1);entered.set()
            if not release.wait(5):raise RuntimeError('projection gate expired')
            return {'mode':'DRY_RUN'}
    surface=Surface();app=web.Application()
    app.router.add_get('/status',surface._status_endpoint);app.router.add_get('/health',surface._health_endpoint)
    try:
        async with TestClient(TestServer(app)) as client:
            first=asyncio.create_task(client.get('/status'));second=asyncio.create_task(client.get('/status'))
            assert await asyncio.to_thread(entered.wait,3)
            health=await (await client.get('/health')).json()
            assert health['status']=='healthy' and not release.is_set()
            one_response,two_response=await first,await second
            assert one_response.status==two_response.status==503
            one,two=await one_response.json(),await two_response.json()
            assert one['phase']==two['phase']=='diagnostics'
            assert one['diagnostic_snapshot']['available'] is False and len(calls)==1
            task=surface._readiness_refresh_task
            release.set()
            await task
            response=await client.get('/status')
            assert response.status==200
            cached=await response.json()
            assert cached['mode']=='DRY_RUN' and len(calls)==1
            assert cached['diagnostic_snapshot']['age_seconds']>=0
    finally:release.set()


@pytest.mark.asyncio
async def test_initial_projection_failure_stays_unready_and_retry_hides_error_text():
    class Surface(ReportingSurface):
        _starting_phase=''
        start_time=time.time()
        def readiness(self):raise RuntimeError('private-provider-url-must-not-be-published')
    surface=Surface()
    first=await surface._status_endpoint(None)
    assert first.status==503
    task=surface._readiness_refresh_task
    with pytest.raises(RuntimeError):await task
    retry=await surface._status_endpoint(None)
    assert retry.status==503
    assert 'RuntimeError' in retry.text and 'private-provider' not in retry.text
    assert surface._readiness_refresh_task is not task
    await asyncio.gather(surface._readiness_refresh_task,return_exceptions=True)


@pytest.mark.asyncio
async def test_background_projection_failure_exposes_error_on_dated_cache():
    class Surface(ReportingSurface):
        def readiness(self):raise RuntimeError('injected projection failure')
    surface=Surface()
    surface._readiness_cache=({'mode':'DRY_RUN'},time.time()-10,time.monotonic()-10)
    view=await surface._readiness_snapshot()
    task=surface._readiness_refresh_task
    assert view['mode']=='DRY_RUN' and view['diagnostic_snapshot']['refresh_in_progress']
    with pytest.raises(RuntimeError):await task
    # Freeze only the refresh cadence for this read, not the reported source age.
    cached=surface._readiness_cache
    surface._readiness_cache=(cached[0],cached[1],time.monotonic())
    view=await surface._readiness_snapshot()
    assert view['diagnostic_snapshot']['refresh_error_type']=='RuntimeError'
    assert view['diagnostic_snapshot']['age_seconds']>=10
