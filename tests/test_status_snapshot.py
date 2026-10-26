import asyncio
import threading
import time
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from src.research.frontier import FrontierIndex
from src.research.global_research_miner import GlobalResearchMiner
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine
from src.runtime.regime import RegimeAndEvidence
from src.runtime.reporting import ReportingSurface
from src.strategies.champion_challenger import ChampionChallengerFramework


def engine(root):
    return UnknownOpportunityEngine(ChampionChallengerFramework(state_path=str(root/'framework.json')),
        FrontierIndex(root/'frontier.sqlite'),TradeEvidenceLedger(root/'evidence.jsonl'))


@pytest.mark.asyncio
async def test_slow_status_query_is_coalesced_without_blocking_health(tmp_path, monkeypatch):
    desk=engine(tmp_path);entered=threading.Event();release=threading.Event();calls=[]
    def report():
        calls.append(1);entered.set()
        if not release.wait(5):raise RuntimeError('test reader gate expired')
        return {'pending':42}
    monkeypatch.setattr(desk.frontier,'report',report)
    class Surface(ReportingSurface):
        unknown_engine=desk
        _starting_phase=''
        _running=True
        dry_run=True
        start_time=time.time()
        def readiness(self, *, unknown_engine_report=None):return {'engine':unknown_engine_report}
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
            assert one['diagnostic_snapshot']['available'] is False
            assert one_response.headers['Retry-After']=='1'
            task=surface._readiness_refresh_task
            release.set()
            await task
            assert len(calls)==1
            response=await client.get('/status')
            assert response.status==200
            cached=await response.json()
            assert cached['engine']['frontier']['pending']==42
            assert cached['engine']['frontier']['snapshot_age_seconds']>=0 and len(calls)==1
    finally:
        release.set();desk.frontier.close()


@pytest.mark.asyncio
async def test_refresh_failure_retains_dated_snapshot_and_reports_error(tmp_path, monkeypatch):
    desk=engine(tmp_path)
    monkeypatch.setattr(desk.frontier,'report',lambda:{'pending':42})
    try:
        first=await desk.frontier_snapshot()
        def unavailable():raise RuntimeError('injected database unavailable')
        monkeypatch.setattr(desk.frontier,'report',unavailable)
        stale=await desk.frontier_snapshot(max_age_seconds=0)
        assert stale['pending']==42 and stale['refresh_in_progress']
        task=desk._frontier_report_task
        with pytest.raises(RuntimeError):await task
        last=await desk.frontier_snapshot(max_age_seconds=3600)
        assert last['pending']==42 and last['refresh_error_type']=='RuntimeError'
        assert last['snapshot_observed_at']==first['snapshot_observed_at']
    finally:desk.frontier.close()


def test_production_research_miner_is_not_a_market_regime_measurement(tmp_path, monkeypatch):
    miner=object.__new__(GlobalResearchMiner)
    def forbidden(*args,**kwargs):raise AssertionError('regime requested full research diagnostics')
    monkeypatch.setattr(miner,'get_stats',forbidden)
    class Surface(RegimeAndEvidence):
        global_research=miner
        dataset_builder=SimpleNamespace(current_market_state=lambda:{})
        global_config={}
    assert Surface().current_regime=='unknown'


def test_research_projection_reuses_the_same_engine_report():
    def forbidden():raise AssertionError('duplicate queue report')
    miner=SimpleNamespace(leads=[],data_status={},observation_status={},conversion_status={},
                          unknown_engine=SimpleNamespace(report=forbidden))
    supplied={'frontier':{'pending':42}}
    result=GlobalResearchMiner.get_stats(miner,engine_report=supplied)
    assert result['unknown_opportunity_engine'] is supplied
