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


def verify(manifest_path=MANIFEST, *, env=None, query=None):
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
    if kernel.get('flags') != 5 or kernel.get('rate') != expected:
        raise RuntimeError('Current worker job does not match the hard CPU budget')
    if (any(env.get(name) != '1' for name in THREAD_VARIABLES)
            or env.get('TOKENIZERS_PARALLELISM', '').lower() != 'false'):
        raise RuntimeError('Single-thread native environment required')
    return {'status': 'VERIFIED', 'cpu_rate': expected, 'cpu_percent': expected/100,
            'kernel_flags': kernel['flags'], 'total_configured_percent': sum(rates)/100,
            'native_threads': 1, 'memory_or_io_cap_claimed': False}
