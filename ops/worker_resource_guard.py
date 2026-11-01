"""Fail closed unless this Windows worker is already inside its CPU job."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

MANIFEST = Path('C:/ProgramData/MemecoinResourceLimits/roles.json')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS')


def current_job_cpu():
    if os.name != 'nt':
        raise RuntimeError('Windows CPU job verification required')
    from ctypes import wintypes
    class CpuRate(ctypes.Structure):
        _fields_ = [('flags', ctypes.c_uint32), ('rate', ctypes.c_uint32)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.QueryInformationJobObject
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                      wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL
    info, returned = CpuRate(), wintypes.DWORD()
    # NULL queries the current process's immediate job, not a named object
    # in another session. A missing/incorrect job is never an uncapped fallback.
    if not query(None, 15, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'flags': info.flags, 'rate': info.rate}


def development_job_cpu():
    """Read the named ancestor job and prove this process belongs to it.

    Sandboxed descendants can have an immediate job with no CPU rate of its own.
    A named-job rate alone is insufficient: IsProcessInJob must prove current
    process membership. Query-only handles never assign or change job limits.
    """
    if os.name != 'nt':
        raise RuntimeError('Windows CPU job verification required')
    from ctypes import wintypes
    class CpuRate(ctypes.Structure):
        _fields_ = [('flags', wintypes.DWORD), ('rate', wintypes.DWORD)]
    class BasicLimits(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                    ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t),
                    ('maximum', ctypes.c_size_t), ('active', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                    ('scheduling', wintypes.DWORD)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    opened = kernel.OpenJobObjectW
    opened.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    opened.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    member = kernel.IsProcessInJob
    member.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    member.restype = wintypes.BOOL
    query = kernel.QueryInformationJobObject
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                      wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    job = opened(4, False, 'Local\\MemecoinCPU-development')  # JOB_OBJECT_QUERY only
    if not job:
        raise RuntimeError('Development ancestor job unavailable')
    try:
        belongs = wintypes.BOOL()
        if not member(kernel.GetCurrentProcess(), job, ctypes.byref(belongs)) or not belongs.value:
            raise RuntimeError('Current process is not in the development ancestor job')
        cpu, basic, returned = CpuRate(), BasicLimits(), wintypes.DWORD()
        for kind, info in ((15, cpu), (2, basic)):
            if not query(job, kind, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
                raise RuntimeError('Development ancestor limits unavailable')
        return {'flags': cpu.flags, 'rate': cpu.rate, 'current_process_member': True,
                'priority_class': basic.priority, 'limit_flags': basic.flags}
    finally:
        close(job)


def verify(manifest_path=MANIFEST, *, env=None, query=None, ancestor_query=None):
    env = os.environ if env is None else env
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    roles = manifest['roles']
    rates = [entry['cpu_rate'] for entry in roles.values()]
    if (manifest.get('version') != 1 or not rates
            or any(type(rate) is not int or not 1 <= rate <= 1000 for rate in rates)
            or sum(rates) > 1000):
        raise ValueError('Invalid shared CPU budget')
    expected = roles['development']['cpu_rate']
    if expected > 200:
        raise ValueError('Development may use at most two percent CPU')
    kernel = (current_job_cpu if query is None else query)()
    source = 'CURRENT_JOB'
    if kernel.get('flags') != 5 or kernel.get('rate') != expected:
        # Test-injected immediate queries cannot accidentally consult a real host
        # ancestor. Production always proves membership with query-only Win32 APIs.
        if query is not None and ancestor_query is None:
            raise RuntimeError('Current worker job does not match the hard CPU budget')
        kernel = (development_job_cpu if ancestor_query is None else ancestor_query)()
        if (kernel.get('flags') != 5 or kernel.get('rate') != expected
                or kernel.get('current_process_member') is not True
                or kernel.get('priority_class') != 16384
                or kernel.get('limit_flags', 0) & 0x2020 != 0x2020):
            raise RuntimeError('Development ancestor job does not contain this capped worker')
        source = 'NAMED_ANCESTOR_JOB_WITH_CURRENT_PROCESS_MEMBERSHIP'
    if (any(env.get(name) != '1' for name in THREAD_VARIABLES)
            or env.get('TOKENIZERS_PARALLELISM', '').lower() != 'false'):
        raise RuntimeError('Single-thread native environment required')
    return {'status': 'VERIFIED', 'cpu_rate': expected, 'cpu_percent': expected/100,
            'kernel_flags': kernel['flags'], 'kernel_evidence_source': source, 'total_configured_percent': sum(rates)/100,
            'native_threads': 1, 'memory_or_io_cap_claimed': False}
