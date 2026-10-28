import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient,TestServer
from src.runtime.reporting import ReportingSurface


@pytest.mark.asyncio
async def test_actual_health_loop_file_publication_yields_for_http(tmp_path,monkeypatch):
    owner=threading.get_ident();entered,released=threading.Event(),threading.Event();observations=[]
    path=tmp_path/'readiness.json';path.write_text('{"old":true}')
    snapshot={'mode':'DRY_RUN','counts':{'accepted':7}}
    class Surface(ReportingSurface):
        _running=True;_starting_phase='';dry_run=True;start_time=time.time()
        global_config={'readiness_path':str(path)}
        memory=SimpleNamespace(observe=lambda:observations.append(threading.get_ident()))
        async def _readiness_snapshot(self):return snapshot
        def _persist_readiness(self,value):
            assert threading.get_ident()!=owner
            entered.set();assert released.wait(5)
            return super()._persist_readiness(value)
    original_sleep=asyncio.sleep
    async def cadence(delay):await original_sleep(0 if delay==10 else delay)
    monkeypatch.setattr('src.runtime.reporting.asyncio.sleep',cadence)
    monkeypatch.delenv('ALLOW_LIVE_TRADING',raising=False)
    surface=Surface();app=web.Application();app.router.add_get('/health',surface._health_endpoint)
    task=asyncio.create_task(surface._health_loop())
    try:
        assert await asyncio.to_thread(entered.wait,3)
        async with TestClient(TestServer(app)) as client:
            response=await asyncio.wait_for(client.get('/health'),2)
            value=await response.json()
            assert value['status']=='healthy' and value['live_submission_locked'] is True
        assert json.loads(path.read_text())=={'old':True} and not task.done()
        assert observations==[owner]*6
        surface._running=False;released.set();await task
        assert json.loads(path.read_text())==snapshot
    finally:
        surface._running=False;released.set();await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_drains_real_atomic_replacement_before_return(tmp_path,monkeypatch):
    path=tmp_path/'readiness.json';path.write_text('{"old":true}')
    surface=ReportingSurface();surface.global_config={'readiness_path':str(path)}
    entered,released=threading.Event(),threading.Event();original=Path.replace
    def gated_replace(value,target):
        if value==path.with_suffix('.json.tmp'):
            entered.set();assert released.wait(5)
        return original(value,target)
    monkeypatch.setattr(Path,'replace',gated_replace)
    task=asyncio.create_task(surface._publish_readiness({'new':True}))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        task.cancel();await asyncio.sleep(0)
        assert not task.done() and json.loads(path.read_text())=={'old':True}
        released.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert json.loads(path.read_text())=={'new':True}
        assert not path.with_suffix('.json.tmp').exists()
    finally:released.set();await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_existing_persistence_failure_remains_nonfatal(tmp_path,monkeypatch):
    path=tmp_path/'readiness.json';path.write_text('{"old":true}')
    surface=ReportingSurface();surface.global_config={'readiness_path':str(path)}
    def fail(*args):raise OSError('fixture storage failure')
    monkeypatch.setattr(Path,'replace',fail)
    await surface._publish_readiness({'new':True})
    assert json.loads(path.read_text())=={'old':True}
