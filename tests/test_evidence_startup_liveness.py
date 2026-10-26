"""Actual startup seam: a gated disk constructor must not hold HTTP liveness."""
import asyncio
import json
import threading
import time

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from src.research.trade_evidence import TradeEvidenceLedger
from src.runtime.reporting import ReportingSurface
from src.runtime.wiring import SubsystemWiring
from tools.verify_evidence_backup import verify


class Desk(SubsystemWiring, ReportingSurface):
    pass


@pytest.mark.asyncio
async def test_certified_recovery_keeps_startup_health_responsive_and_publishes_only_complete_ledger(tmp_path,monkeypatch):
    ledger_path=tmp_path/'trade_evidence.jsonl'
    TradeEvidenceLedger(ledger_path).record('synthetic_source',{'observed':True})
    certificate=tmp_path/'certificate.json';certificate.write_text(json.dumps(verify(ledger_path)))
    entered,release=threading.Event(),threading.Event()
    thread_ids=[];main_thread=threading.get_ident()
    def constructor(path,**kwargs):
        thread_ids.append(threading.get_ident())
        assert kwargs['recovery_certificate']==str(certificate)
        entered.set()
        if not release.wait(5):raise RuntimeError('synthetic recovery gate expired')
        return TradeEvidenceLedger(path,**kwargs)
    monkeypatch.setattr('src.runtime.wiring.TradeEvidenceLedger',constructor)
    desk=Desk();desk.global_config={'ops_state_dir':str(tmp_path),'evidence_recovery_certificate':str(certificate)}
    desk.trade_evidence=None;desk._starting_phase='prediction';desk.start_time=time.time()
    app=web.Application();app.router.add_get('/health',desk._health_endpoint);app.router.add_get('/status',desk._status_endpoint)
    task=None
    try:
        async with TestClient(TestServer(app)) as client:
            task=asyncio.create_task(desk._setup_trade_evidence())
            assert await asyncio.to_thread(entered.wait,3)
            assert desk.trade_evidence is None and len(thread_ids)==1 and thread_ids[0]!=main_thread
            health=await asyncio.wait_for(client.get('/health'),1)
            assert (await health.json())['phase']=='prediction' and not release.is_set()
            status=await asyncio.wait_for(client.get('/status'),1)
            assert status.status==503 and (await status.json())['phase']=='prediction'
            release.set();await task
            assert desk.trade_evidence.sequence==1
            assert desk.trade_evidence.recovery_mode=='CERTIFIED_PREFIX_SHA256_AND_VERIFIED_TAIL'
            assert desk.trade_evidence.write_failures==0
    finally:
        release.set()
        if task is not None:await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_constructor_exception_is_not_swallowed_or_replaced_with_partial_state(tmp_path,monkeypatch):
    def unavailable(*args,**kwargs):raise OSError('synthetic source unavailable')
    monkeypatch.setattr('src.runtime.wiring.TradeEvidenceLedger',unavailable)
    desk=Desk();desk.global_config={'ops_state_dir':str(tmp_path)};desk.trade_evidence=None
    with pytest.raises(OSError,match='source unavailable'):
        await desk._setup_trade_evidence()
    assert desk.trade_evidence is None
