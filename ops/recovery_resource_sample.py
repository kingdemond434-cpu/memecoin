"""Read-only current-process counters around one finite recovery tick."""
import ctypes,math,os,sys,time
from ops.canonical_recovery_tick import run_fragment_tick

COUNTERS=('read_operations','write_operations','other_operations','read_transfer_bytes',
          'write_transfer_bytes','other_transfer_bytes','cpu_kernel_100ns','cpu_user_100ns')
MEMORY=('working_set_bytes','peak_working_set_bytes','private_commit_bytes')
SCOPE='own process between non-atomic samples; includes instrumentation and other activity'


def _native_sample():
    from ctypes import wintypes
    class IO(ctypes.Structure):
        _fields_=[(name,ctypes.c_ulonglong) for name in COUNTERS[:6]]
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in (
            'peak_working_set_bytes','working_set_bytes','peak_paged_pool','paged_pool','peak_nonpaged_pool',
            'nonpaged_pool','pagefile_commit','peak_pagefile_commit','private_commit_bytes')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    current=kernel.GetCurrentProcess;current.argtypes=[];current.restype=wintypes.HANDLE
    io_call=kernel.GetProcessIoCounters;io_call.argtypes=[wintypes.HANDLE,ctypes.POINTER(IO)];io_call.restype=wintypes.BOOL
    times=kernel.GetProcessTimes;times.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;times.restype=wintypes.BOOL
    memory_call=psapi.GetProcessMemoryInfo;memory_call.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];memory_call.restype=wintypes.BOOL
    started=time.monotonic();handle=current();io=IO();memory=Memory();memory.cb=ctypes.sizeof(memory)
    creation,exit_time,kernel_time,user_time=[wintypes.FILETIME() for _ in range(4)]
    if not times(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel_time),ctypes.byref(user_time)):
        raise OSError('Own-process timing sample unavailable')
    if not io_call(handle,ctypes.byref(io)):raise OSError('Own-process IO sample unavailable')
    if not memory_call(handle,ctypes.byref(memory),ctypes.sizeof(memory)):raise OSError('Own-process memory sample unavailable')
    def ticks(value):return (value.dwHighDateTime<<32)+value.dwLowDateTime
    return dict(status='OWN_PROCESS_RESOURCE_SAMPLE',pid=os.getpid(),creation_100ns=ticks(creation),
        cpu_kernel_100ns=ticks(kernel_time),cpu_user_100ns=ticks(user_time),
        observation_start_monotonic=started,observation_end_monotonic=time.monotonic(),
        **{name:getattr(io,name) for name in COUNTERS[:6]},**{name:getattr(memory,name) for name in MEMORY})


def _validate(value):
    if not isinstance(value,dict) or value.get('status')!='OWN_PROCESS_RESOURCE_SAMPLE':
        raise ValueError('Native resource sample required')
    for name in ('pid','creation_100ns')+COUNTERS+MEMORY:
        if type(value.get(name)) is not int or value[name]<0:raise ValueError('Unsigned native counter required')
    if value['pid']<=0 or value['peak_working_set_bytes']<value['working_set_bytes']:
        raise ValueError('Resource sample identity or memory mismatch')
    for name in ('observation_start_monotonic','observation_end_monotonic'):
        if type(value.get(name)) not in (int,float) or not math.isfinite(value[name]) or value[name]<0:
            raise ValueError('Finite observation clock required')
    if value['observation_end_monotonic']<value['observation_start_monotonic']:
        raise ValueError('Observation clock regressed')
    return value


def sample_own_process():
    if sys.platform!='win32':return dict(status='RESOURCE_SAMPLE_UNSUPPORTED',scope=SCOPE)
    try:return _validate(_native_sample())
    except Exception as exc:return dict(status='RESOURCE_SAMPLE_UNAVAILABLE',error_type=type(exc).__name__,scope=SCOPE)


def resource_delta(before,after):
    before=_validate(before);after=_validate(after)
    if (before['pid'],before['creation_100ns'])!=(after['pid'],after['creation_100ns']):
        raise ValueError('Resource sample process changed')
    if after['observation_start_monotonic']<before['observation_end_monotonic']:
        raise ValueError('Resource sample intervals overlap or regress')
    delta={name:after[name]-before[name] for name in COUNTERS}
    if any(v<0 for v in delta.values()):raise ValueError('Resource counter regressed')
    return dict(status='OWN_PROCESS_RESOURCE_DELTA',scope=SCOPE,pid=before['pid'],
        elapsed_observation_seconds=after['observation_end_monotonic']-before['observation_start_monotonic'],
        cpu_seconds=(delta['cpu_kernel_100ns']+delta['cpu_user_100ns'])/10000000,
        **delta,working_set_before_bytes=before['working_set_bytes'],working_set_after_bytes=after['working_set_bytes'],
        working_set_change_bytes=after['working_set_bytes']-before['working_set_bytes'],
        process_lifetime_peak_working_set_after_bytes=after['peak_working_set_bytes'],
        private_commit_before_bytes=before['private_commit_bytes'],private_commit_after_bytes=after['private_commit_bytes'],
        child_processes_included=False,physical_io_cap_claimed=False,physical_disk_bytes_verified=False,
        quant_performance_verified=False,memory_cap_claimed=False)


def run_observed_fragment_tick(*args,**kwargs):
    """Sampling failure before work starts nothing; failure after work preserves its result.

    Current-process APIs only; no other process handles, elevation, settings or tasks.
    This observer does not replace the tick's CPU/admission/pacing/pin gates.
    """
    before=sample_own_process()
    try:_validate(before);ready=True
    except Exception:ready=False
    if not ready:
        return dict(status='TICK_NOT_STARTED_RESOURCE_SAMPLE_UNAVAILABLE',tick_started=False,before=before,
                    recovery_verified=False,deployment_backup_accepted=False)
    result=None;error=None
    try:result=run_fragment_tick(*args,**kwargs)
    except Exception as exc:error=type(exc).__name__
    after=sample_own_process()
    try:measurement=resource_delta(before,after)
    except Exception as exc:measurement=dict(status='RESOURCE_DELTA_UNAVAILABLE',error_type=type(exc).__name__,scope=SCOPE)
    status=('TICK_RESULT_UNAVAILABLE_RECONCILIATION_REQUIRED' if result is None else
            'TICK_RECONCILIATION_REQUIRED' if result.get('status')=='RECONCILIATION_REQUIRED' else
            'TICK_RESOURCE_OBSERVATION_RECORDED' if measurement['status']=='OWN_PROCESS_RESOURCE_DELTA' else
            'TICK_COMPLETE_RESOURCE_OBSERVATION_UNAVAILABLE')
    return dict(status=status,tick_started=True,tick_result=result,tick_error_type=error,before=before,after=after,
        measurement=measurement,recovery_verified=False,deployment_backup_accepted=False,
        physical_io_cap_claimed=False,memory_cap_claimed=False,quant_performance_verified=False)
