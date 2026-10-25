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
            release.set()
            one=await (await first).json();two=await (await second).json()
            assert one['mode']==two['mode']=='DRY_RUN' and len(calls)==1
            assert one['diagnostic_snapshot']['age_seconds']>=0
    finally:release.set()


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
