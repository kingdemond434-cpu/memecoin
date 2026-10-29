"""Isolated Windows job-budget fixture; never imports bot/runtime configuration."""
import ctypes,ctypes.wintypes as w,hashlib,json,os,subprocess,sys,time
from pathlib import Path

kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenJobObjectW.argtypes=[w.DWORD,w.BOOL,w.LPCWSTR];kernel.OpenJobObjectW.restype=w.HANDLE
kernel.GetCurrentProcess.restype=w.HANDLE
kernel.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)]
kernel.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,w.LPVOID,w.DWORD,w.LPVOID]
kernel.CloseHandle.argtypes=[w.HANDLE]

def membership():
    handle=kernel.OpenJobObjectW(4,False,'Local\\MemecoinCPU-fixture')
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        value=w.BOOL()
        assert kernel.IsProcessInJob(kernel.GetCurrentProcess(),handle,ctypes.byref(value)) and value.value
        rate=(w.DWORD*2)()
        assert kernel.QueryInformationJobObject(handle,15,ctypes.byref(rate),ctypes.sizeof(rate),None)
        assert list(rate)==[5,100]
        return {'pid':os.getpid(),'in_budget_job':True,'cpu_rate':rate[1],'control_flags':rate[0]}
    finally:kernel.CloseHandle(handle)

folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
record=membership()
record['deterministic_result']=hashlib.pbkdf2_hmac('sha256',b'cpu-fixture',b'fixed-salt',30000).hex()
if '--grandchild' in sys.argv:
    deadline=time.monotonic()+7
    while time.monotonic()<deadline:hashlib.sha256(b'x'*2048).digest()
    (folder/('grandchild-'+str(os.getpid())+'.json')).write_text(json.dumps(record))
else:
    record['thread_environment']={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')}
    children=[subprocess.Popen([sys.executable,__file__,str(folder),'--grandchild'],creationflags=subprocess.CREATE_NO_WINDOW) for _ in range(2)]
    record['breakaway_rejected']=False
    try:
        escaped=subprocess.Popen([sys.executable,'-c','pass'],creationflags=subprocess.CREATE_BREAKAWAY_FROM_JOB|subprocess.CREATE_NO_WINDOW)
        escaped.wait(timeout=20)
    except OSError as exc:record['breakaway_rejected']=exc.winerror==5
    for child in children:assert child.wait(timeout=120)==0
    assert record['breakaway_rejected']
    assert all(value=='1' for value in record['thread_environment'].values())
    (folder/'parent.json').write_text(json.dumps(record))
