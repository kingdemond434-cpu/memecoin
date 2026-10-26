"""Temporary offline CI test-order bisection; no service or provider access."""
import os
import subprocess
import sys

TARGET='tests/test_perps_acquisition.py::test_partial_http_failure_keeps_other_books_and_continues_universe_after_restart'


def pytest_collection_modifyitems(items):
    prefix=os.environ.get('MEMECOIN_DIAGNOSTIC_PREFIX')
    if prefix is None:return
    target=next(item for item in items if item.nodeid==TARGET)
    preceding=items[:items.index(target)]
    items[:]=preceding[:int(prefix)]+[target]


def main():
    collected=subprocess.check_output([sys.executable,'-m','pytest','tests','--collect-only','-q'],text=True)
    names=[line.strip() for line in collected.splitlines() if line.startswith('tests/') and '::' in line]
    high=names.index(TARGET)
    def fails(prefix):
        env={**os.environ,'MEMECOIN_DIAGNOSTIC_PREFIX':str(prefix)}
        run=subprocess.run([sys.executable,'-m','pytest','tests','-q','--tb=short','-p','tools.diagnose_perps_order'],
            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        failure=('FAILED '+TARGET) in run.stdout
        print(f'PREFIX={prefix} TARGET_FAILED={failure} EXIT={run.returncode}',flush=True)
        if run.returncode and not failure:print(run.stdout[-6000:],flush=True)
        return failure
    if fails(0):
        print('COLLECTION_ONLY_CONTAMINATION',flush=True);return
    if not fails(high):
        print('FULL_PREFIX_DID_NOT_REPRODUCE',flush=True);return
    low=0
    while high-low>1:
        mid=(low+high)//2
        if fails(mid):high=mid
        else:low=mid
    print('FIRST_TRIGGER_PREDECESSOR='+names[high-1],flush=True)
    print('LAST_PASS_PREFIX='+str(low),flush=True)


if __name__=='__main__':main()
