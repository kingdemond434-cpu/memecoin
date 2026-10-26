"""Exercise native VM fixtures behind the production process boundary.

Direct local-bank execution caused a reproducible later SQLite selection
failure in Linux CPython 3.11's full suite. The same selector passed alone;
prefix bisection identified the new VM tests as the trigger. The underlying
native/runtime interaction is unresolved. Do not let native VM state enter
the host process in tests when production deliberately isolates it too.
Every child runs the original test assertions; failures propagate to pytest.
"""
import asyncio
import functools
import inspect
import os
import re
from pathlib import Path
import subprocess
import sys

WORKER='MEMECOIN_PAPER_TEST_WORKER'


def isolated_vm(function):
    def child():
        node=os.environ['PYTEST_CURRENT_TEST'].rsplit(' (',1)[0]
        env={key:os.environ[key] for key in ('SYSTEMROOT','WINDIR','TEMP','TMP') if key in os.environ}
        env.update({WORKER:'1','PYTHONIOENCODING':'utf-8','ALLOW_LIVE_TRADING':''})
        result=subprocess.run([sys.executable,'-m','pytest',node,'-q','--tb=short'],
            cwd=Path(__file__).resolve().parents[1],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            encoding='utf-8',errors='replace',timeout=90,
            creationflags=0x08000000 if sys.platform=='win32' else 0)
        assert result.returncode==0 and re.search(r'\b1 passed\b',result.stdout), result.stdout[-12000:]
    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def run(*args,**kwargs):
            if os.environ.get(WORKER)=='1':return await function(*args,**kwargs)
            await asyncio.to_thread(child)
    else:
        @functools.wraps(function)
        def run(*args,**kwargs):
            if os.environ.get(WORKER)=='1':return function(*args,**kwargs)
            child()
    return run
