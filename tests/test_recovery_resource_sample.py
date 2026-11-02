import json,os,sys
import pytest
import ops.recovery_resource_sample as module


def pair():
    a=dict(status='OWN_PROCESS_RESOURCE_SAMPLE',pid=7,creation_100ns=123,
        observation_start_monotonic=1.,observation_end_monotonic=1.1,
        **{k:100 for k in module.COUNTERS},working_set_bytes=100,
        peak_working_set_bytes=150,private_commit_bytes=200)
    b=dict(a,observation_start_monotonic=3.,observation_end_monotonic=3.1,
        working_set_bytes=90,private_commit_bytes=250,
        **{k:a[k]+10 for k in module.COUNTERS})
    b['cpu_kernel_100ns']=a['cpu_kernel_100ns']+10000000
    b['cpu_user_100ns']=a['cpu_user_100ns']+20000000
    return a,b


def test_delta_uses_documented_units_and_allows_memory_shrink():
    a,b=pair();r=module.resource_delta(a,b)
    assert r['cpu_seconds']==3 and r['read_transfer_bytes']==10 and r['read_operations']==10
    assert r['working_set_change_bytes']==-10 and r['private_commit_after_bytes']==250
    assert r['elapsed_observation_seconds']==pytest.approx(2.1)
    assert not r['physical_disk_bytes_verified'] and not r['quant_performance_verified']
    assert not r['memory_cap_claimed'] and not r['child_processes_included']


@pytest.mark.parametrize('kind',['pid','creation','io_regression','cpu_regression','overlap','nan',
    'bool_counter','negative_memory','bad_peak','bad_status'])
def test_invalid_counter_domains_are_rejected_not_clamped(kind):
    a,b=pair()
    if kind=='pid':b['pid']=8
    elif kind=='creation':b['creation_100ns']=124
    elif kind=='io_regression':b['read_transfer_bytes']=99
    elif kind=='cpu_regression':b['cpu_user_100ns']=99
    elif kind=='overlap':b['observation_start_monotonic']=1.05
    elif kind=='nan':b['observation_end_monotonic']=float('nan')
    elif kind=='bool_counter':b['write_transfer_bytes']=True
    elif kind=='negative_memory':b['private_commit_bytes']=-1
    elif kind=='bad_peak':b['peak_working_set_bytes']=1
    elif kind=='bad_status':b['status']='UNSUPPORTED'
    with pytest.raises(ValueError):module.resource_delta(a,b)


def test_native_error_is_sanitized_and_unsupported_has_no_zero_sample(monkeypatch):
    monkeypatch.setattr(module.sys,'platform','win32')
    def fail():raise PermissionError('private diagnostic marker')
    monkeypatch.setattr(module,'_native_sample',fail)
    r=module.sample_own_process()
    assert r['status']=='RESOURCE_SAMPLE_UNAVAILABLE' and r['error_type']=='PermissionError'
    assert 'private diagnostic marker' not in repr(r)
    monkeypatch.setattr(module.sys,'platform','linux')
    assert module.sample_own_process()['status']=='RESOURCE_SAMPLE_UNSUPPORTED'


def samples(monkeypatch,a,b):
    values=iter([a,b]);monkeypatch.setattr(module,'sample_own_process',lambda:next(values))


def test_missing_or_invalid_initial_sample_starts_no_tick(monkeypatch):
    def no_tick(*a,**k):raise AssertionError('must not start work')
    monkeypatch.setattr(module,'run_fragment_tick',no_tick)
    for sample in (dict(status='RESOURCE_SAMPLE_UNAVAILABLE'),dict(status='OWN_PROCESS_RESOURCE_SAMPLE')):
        monkeypatch.setattr(module,'sample_own_process',lambda:sample)
        r=module.run_observed_fragment_tick()
        assert not r['tick_started'] and r['status']=='TICK_NOT_STARTED_RESOURCE_SAMPLE_UNAVAILABLE'


def test_success_preserves_real_tick_result_and_counter_delta(monkeypatch):
    a,b=pair();samples(monkeypatch,a,b);tick=dict(status='STEP_BUDGET_REACHED',last_checkpointed_progress_sha256='a'*64)
    monkeypatch.setattr(module,'run_fragment_tick',lambda *a,**k:tick)
    r=module.run_observed_fragment_tick()
    assert r['status']=='TICK_RESOURCE_OBSERVATION_RECORDED' and r['tick_result'] is tick
    assert r['measurement']['cpu_seconds']==3 and not r['recovery_verified']


def test_missing_final_measurements_do_not_discard_or_retry_completed_tick(monkeypatch):
    a,b=pair();samples(monkeypatch,a,dict(status='RESOURCE_SAMPLE_UNAVAILABLE'));calls=[]
    tick=dict(status='SOURCE_COPY_BUDGET_REACHED',last_checkpointed_progress_sha256='b'*64)
    def execute(*a,**k):calls.append(1);return tick
    monkeypatch.setattr(module,'run_fragment_tick',execute)
    r=module.run_observed_fragment_tick()
    assert r['status']=='TICK_COMPLETE_RESOURCE_OBSERVATION_UNAVAILABLE' and r['tick_result'] is tick
    assert calls==[1] and r['measurement']['status']=='RESOURCE_DELTA_UNAVAILABLE'


def test_lost_tick_result_requires_reconciliation_and_never_leaks_error(monkeypatch):
    a,b=pair();samples(monkeypatch,a,b);calls=[]
    def execute(*a,**k):calls.append(1);raise OSError('private diagnostic marker')
    monkeypatch.setattr(module,'run_fragment_tick',execute)
    r=module.run_observed_fragment_tick()
    assert r['status']=='TICK_RESULT_UNAVAILABLE_RECONCILIATION_REQUIRED' and r['tick_result'] is None
    assert calls==[1] and 'private diagnostic marker' not in repr(r)


def test_unresolved_tick_is_not_promoted_by_successful_sampling(monkeypatch):
    a,b=pair();samples(monkeypatch,a,b)
    tick=dict(status='RECONCILIATION_REQUIRED',unresolved_step_copy_upper_bound=1048576)
    monkeypatch.setattr(module,'run_fragment_tick',lambda *a,**k:tick)
    r=module.run_observed_fragment_tick()
    assert r['status']=='TICK_RECONCILIATION_REQUIRED' and r['tick_result'] is tick


@pytest.mark.skipif(sys.platform!='win32',reason='current-process Win32 APIs only')
def test_actual_windows_native_counter_sample_is_valid():
    a=module.sample_own_process();b=module.sample_own_process()
    assert a['status']==b['status']=='OWN_PROCESS_RESOURCE_SAMPLE'
    assert a['pid']==os.getpid() and a['creation_100ns']>0 and a['working_set_bytes']>0
    r=module.resource_delta(a,b)
    assert r['cpu_seconds']>=0 and r['elapsed_observation_seconds']>=0
