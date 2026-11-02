import json,sqlite3,time
from pathlib import Path
import pytest
from src.research.frontier import FrontierIndex,AuctionInputs,FrontierDiagnosticBudgetExceeded
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.unknown_engine import UnknownOpportunityEngine
from src.strategies.champion_challenger import ChampionChallengerFramework


def fixture(tmp_path,rows=1):
    index=FrontierIndex(tmp_path/'frontier.sqlite')
    auction=AuctionInputs(probability_unique=.5,expected_delta_logw=1,novelty=1,coverage_debt=1,urgency=1,
        costs={'inference':1,'data':0,'engineering':0,'conversion':0,'gauntlet':0})
    job=index.enqueue('DATA_REQUEST','lake:fixture',auction=auction,now=100)
    if rows>1:
        with index._db() as db:
            template=dict(db.execute('SELECT * FROM jobs WHERE job_id=?',(job,)).fetchone());cols=list(template)
            values=[tuple(({**template,'job_id':f'fixture:{n}'})[k] for k in cols) for n in range(rows-1)]
            db.executemany('INSERT INTO jobs ('+','.join(cols)+') VALUES ('+','.join('?' for _ in cols)+')',values)
    return index


def test_readonly_snapshot_matches_full_report_and_bypasses_writer_connection(tmp_path,monkeypatch):
    index=fixture(tmp_path)
    try:
        expected=index.report(now=101);before=index.path.read_bytes()
        def forbidden():pytest.fail('diagnostic used writer connection')
        monkeypatch.setattr(index,'_db',forbidden)
        assert index.diagnostic_report(now=101)==expected and index.path.read_bytes()==before
    finally:index.close()


def test_vm_budget_aborts_without_partial_counts_or_lost_work(tmp_path):
    index=fixture(tmp_path,128)
    try:
        with pytest.raises(FrontierDiagnosticBudgetExceeded):index.diagnostic_report(vm_step_budget=100)
        assert index.report(now=101)['pending']==128
        assert index.diagnostic_report(now=101,vm_step_budget=1000000)['pending']==128
    finally:index.close()


def test_deadline_interrupts_vm_without_disabling_full_report(tmp_path):
    index=fixture(tmp_path,128);ticks=iter([0,0,3])
    try:
        with pytest.raises(FrontierDiagnosticBudgetExceeded):index.diagnostic_report(clock=lambda:next(ticks))
        assert index.report(now=101)['pending']==128
    finally:index.close()


@pytest.mark.parametrize('kwargs',[{'vm_step_budget':True},{'vm_step_budget':99},
    {'vm_step_budget':1000001},{'deadline_seconds':float('nan')},{'deadline_seconds':6}])
def test_invalid_policy_and_missing_file_do_not_create_database(tmp_path,kwargs):
    index=object.__new__(FrontierIndex);index.path=tmp_path/'absent.sqlite'
    with pytest.raises(ValueError):index.diagnostic_report(**kwargs)
    assert not index.path.exists()


@pytest.mark.asyncio
async def test_budget_failure_retains_old_snapshot_and_error_without_fabricating_counts(tmp_path,monkeypatch):
    index=fixture(tmp_path,128)
    desk=UnknownOpportunityEngine(ChampionChallengerFramework(state_path=str(tmp_path/'framework.json')),
        index,TradeEvidenceLedger(tmp_path/'evidence.jsonl'))
    try:
        first=await desk.frontier_snapshot();assert first['pending']==128
        original=index.diagnostic_report
        monkeypatch.setattr(index,'diagnostic_report',lambda:original(vm_step_budget=100))
        stale=await desk.frontier_snapshot(max_age_seconds=0)
        assert stale['pending']==128 and stale['refresh_in_progress']
        with pytest.raises(FrontierDiagnosticBudgetExceeded):await desk._frontier_report_task
        last=await desk.frontier_snapshot(max_age_seconds=3600)
        assert last['pending']==128 and last['snapshot_observed_at']==first['snapshot_observed_at']
        assert last['refresh_error_type']=='FrontierDiagnosticBudgetExceeded'
        assert index.report(now=101)['pending']==128
    finally:index.close()


def test_late_query_result_is_not_returned_as_a_fresh_report(tmp_path,monkeypatch):
    index=fixture(tmp_path);ticks=iter([0,0,3])
    try:
        monkeypatch.setattr(index,'report',lambda **kwargs:{'pending':999})
        with pytest.raises(FrontierDiagnosticBudgetExceeded):index.diagnostic_report(clock=lambda:next(ticks))
    finally:index.close()


@pytest.mark.asyncio
async def test_initial_budget_failure_does_not_invent_snapshot(tmp_path,monkeypatch):
    index=fixture(tmp_path,128)
    desk=UnknownOpportunityEngine(ChampionChallengerFramework(state_path=str(tmp_path/'framework.json')),
        index,TradeEvidenceLedger(tmp_path/'evidence.jsonl'))
    original=index.diagnostic_report
    monkeypatch.setattr(index,'diagnostic_report',lambda:original(vm_step_budget=100))
    try:
        with pytest.raises(FrontierDiagnosticBudgetExceeded):await desk.frontier_snapshot()
        assert desk._frontier_report_cache is None and desk._frontier_report_task is None
        assert desk._frontier_report_error=='FrontierDiagnosticBudgetExceeded'
        assert index.report(now=101)['pending']==128
    finally:index.close()


@pytest.mark.asyncio
async def test_health_loop_retains_governor_and_cadence_after_initial_budget_failure(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from src.runtime.reporting import ReportingSurface
    observations=[];published=[];calls=[];cadence=[];original_sleep=asyncio.sleep
    class Surface(ReportingSurface):
        _running=True
        memory=SimpleNamespace(observe=lambda:observations.append(1))
        async def _readiness_snapshot(self):
            calls.append(1)
            if len(calls)==1:raise FrontierDiagnosticBudgetExceeded('fixture budget')
            return {'mode':'DRY_RUN'}
        async def _publish_readiness(self,snapshot):published.append(snapshot);self._running=False
    async def controlled(delay):cadence.append(delay);await original_sleep(0)
    monkeypatch.setattr('src.runtime.reporting.asyncio.sleep',controlled)
    await Surface()._health_loop()
    assert len(observations)==12 and cadence==[10]*12 and calls==[1,1]
    assert published==[{'mode':'DRY_RUN'}]
