import json,os
import pytest
from ops.source_phase_snapshot import read_source_phase_snapshot,MAX_BYTES,SCOPE


def fixture(tmp_path):
    p=tmp_path/'readiness.json'
    row=dict(attempts=2,completed=1,interrupted=1,total_seconds=4,max_seconds=3,last_seconds=1)
    p.write_text(json.dumps(dict(live_submission_locked=True,private_config='secret-fixture-value',
        source_phase_timing=dict(scope=SCOPE,phases=dict(canonical_ack_owner_resume=row,
        unknown_private_phase=dict(secret='secret-fixture-value'))))))
    return p


def run(p,**kwargs):
    return read_source_phase_snapshot(p,continue_check=lambda:True,
        resource_check=lambda:dict(status='VERIFIED',cpu_rate=200),**kwargs)


def test_real_snapshot_redaction_freshness_and_scope(tmp_path):
    p=fixture(tmp_path);before=p.read_bytes();r=run(p,now=lambda:p.stat().st_mtime+5)
    assert r['snapshot_fresh'] and r['live_submission_locked'] is True
    assert set(r['phases'])=={'canonical_ack_owner_resume'}
    assert 'secret-fixture-value' not in json.dumps(r) and p.read_bytes()==before
    assert not any(r[k] for k in ('http_health_verified','stall_cause_established','source_latency_verified','source_write_requested'))
    assert not run(p,now=lambda:p.stat().st_mtime+181)['snapshot_fresh']
    assert run(p,now=lambda:p.stat().st_mtime-1)['clock_skew_detected']


@pytest.mark.parametrize('key,value',[('attempts',True),('completed',-1),('total_seconds',float('nan')),('last_seconds','secret-fixture-value'),('interrupted',0)])
def test_invalid_metrics_not_exported(tmp_path,key,value):
    p=fixture(tmp_path);d=json.loads(p.read_text());d['source_phase_timing']['phases']['canonical_ack_owner_resume'][key]=value;p.write_text(json.dumps(d))
    r=run(p);assert not r['phases'] and r['invalid_phases']==['canonical_ack_owner_resume']
    assert 'secret-fixture-value' not in json.dumps(r)


def test_unavailable_scope_and_duplicate_keys(tmp_path):
    p=fixture(tmp_path);d=json.loads(p.read_text());d['source_phase_timing']['scope']='private-scope';p.write_text(json.dumps(d))
    assert not run(p)['timing_available']
    p.write_text('{"source_phase_timing":{},"source_phase_timing":{}}')
    with pytest.raises(ValueError):run(p)


def test_budget_and_admission_before_read(tmp_path):
    p=fixture(tmp_path)
    with pytest.raises(RuntimeError):read_source_phase_snapshot(p,continue_check=lambda:False,resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))
    with p.open('wb') as f:f.truncate(MAX_BYTES+1)
    with pytest.raises(ValueError):run(p)


def test_changed_file_during_paced_read_rejected(tmp_path):
    p=fixture(tmp_path);tick=[0]
    def sleep(dt):
        tick[0]+=dt;p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):run(p,clock=lambda:tick[0],sleep=sleep)


def test_fresh_publication_cannot_hide_stale_projection(tmp_path):
    p=fixture(tmp_path);d=json.loads(p.read_text());observed=p.stat().st_mtime
    d['diagnostic_snapshot']=dict(observed_at=observed-1000,refresh_in_progress=True,refresh_error_type='secret-fixture-value')
    p.write_text(json.dumps(d));r=run(p,now=lambda:observed+5)
    assert r['snapshot_fresh'] and not r['projection_fresh']
    assert r['projection_age_seconds']==1005 and r['projection_refresh_in_progress'] is True
    assert r['projection_refresh_error_present'] and 'secret-fixture-value' not in json.dumps(r)
    assert run(fixture(tmp_path))['projection_fresh'] is None


def test_refresh_wait_metadata_allowlist_and_legacy_unknown(tmp_path):
    p=fixture(tmp_path);assert run(p)['refresh_phase'] is None
    d=json.loads(p.read_text());stamp=p.stat().st_mtime
    d['diagnostic_snapshot']=dict(refresh_phase='frontier_snapshot',refresh_pending_seconds=120,
        refresh_phase_seconds=100,refresh_wait_observed_at=stamp)
    p.write_text(json.dumps(d));r=run(p,now=lambda:stamp+5)
    assert r['refresh_phase']=='frontier_snapshot' and r['refresh_wait_age_seconds']==5
    assert r['refresh_pending_seconds']==120 and r['refresh_phase_seconds']==100
    d['diagnostic_snapshot'].update(refresh_phase='secret-fixture-value',refresh_pending_seconds=True)
    p.write_text(json.dumps(d));r=run(p)
    assert r['refresh_phase'] is None and r['refresh_pending_seconds'] is None
    assert 'secret-fixture-value' not in json.dumps(r)
