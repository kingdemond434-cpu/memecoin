"""Single bounded server developer; sequential resumptions and reviewed actions.

Installed as worker.py beside worker_resource_guard.py on the authorized VPS.
No credentials, scheduled-task registration or production deployment live here.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

try:
    from .worker_resource_guard import verify
except ImportError:
    from worker_resource_guard import verify

ROOT = Path('C:/Users/Administrator/memecoin-autonomous')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def process_lock(root):
    import msvcrt
    with (root/'worker.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def child_environment(root, environ=None):
    env = dict(os.environ if environ is None else environ)
    for name in ('GH_TOKEN', 'GITHUB_TOKEN'):
        env.pop(name, None)  # use the authorized machine-protected gh wrapper
    for name in ('ALLOW_LIVE_TRADING', 'SOLANA_PRIVATE_KEY', 'MEMECOIN_SIGNER_SOCKET',
                 'MEMECOIN_WALLET_MODE'):
        env[name] = ''
    env.update(PATH=str(root/'bin')+';C:/Program Files/Git/cmd;C:/Program Files/Python314;'+env.get('PATH',''),
        GH_CONFIG_DIR=str(root/'github-auth'), GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    return env


def completion_ready(path):
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return False
    if not isinstance(report, dict):
        return False
    return (report.get('all_blueprints_complete_excluding_live') is True
        and report.get('outstanding_requirements') == []
        and isinstance(report.get('dashboard_url'), str) and report['dashboard_url'].startswith('https://')
        and isinstance(report.get('evidence'), list) and bool(report['evidence']))


def build_prompt(root, stamp):
    handoff = root/'handoff'
    prompt = (handoff/'WORKER-INSTRUCTIONS.txt').read_text(encoding='utf-8-sig')
    current = handoff/'CONTINUATION-20261031.txt'
    if not current.exists():
        raise RuntimeError('Current resource and acceptance instructions required')
    prompt += '\n\nLatest user-authorized continuation and resource requirements:\n'
    prompt += current.read_text(encoding='utf-8-sig')
    prompt += ('\nThis is scheduled run '+stamp+'. Read the current review, resource fence, ledger and checkpoint. '
               'Implement the next coherent unfinished increment, verify it, and checkpoint exact remaining requirements. '
               'The supervisor will resume sequential work while incomplete; do not create another worker.')
    return prompt


def run_cycle(root, guard, *, popen=subprocess.Popen):
    logs, handoff = root/'logs', root/'handoff'
    status = logs/'worker-status.json'
    prior = json.loads(status.read_text(encoding='utf-8-sig')) if status.exists() else {}
    if prior.get('state') == 'COMPLETION_REVIEW_PENDING':
        return 'STOP'
    if time.time() < prior.get('next_eligible_at', 0):
        return 'WAIT'
    stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())+'-'+uuid.uuid4().hex[:8]
    thread = prior.get('thread_id')
    if thread:
        uuid.UUID(thread)
    prompt = build_prompt(root, stamp)
    state = {'state':'RUNNING', 'run_id':stamp, 'started_at':time.time(),
        'supervisor_pid':os.getpid(), 'thread_id':thread,
        'consecutive_failures':prior.get('consecutive_failures',0),
        'live_trading_enabled':False, 'laptop_required':False, 'resource_guard':guard}
    save(status, state)
    cmd = [str(root/'bin/codex.exe'), 'exec', '--ignore-user-config', '--approve-for-me',
        '-C', str(root/'repo'), '--add-dir', str(handoff), '--add-dir', str(logs)]
    if thread:
        cmd += ['resume', thread]
    cmd += ['--json', '-o', str(logs/(stamp+'-last-message.txt')), '-']
    # No --dangerously-bypass-approvals-and-sandbox, no never-approve mode,
    # and no added model/API key/account override.
    with (logs/(stamp+'-events.jsonl')).open('w', encoding='utf-8') as out, \
            (logs/(stamp+'-stderr.log')).open('w', encoding='utf-8') as err:
        proc = popen(cmd, cwd=root/'repo', env=child_environment(root), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=err, text=True, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        state['codex_pid'] = proc.pid
        save(status, state)
        try:
            proc.stdin.write(prompt); proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # collect the actual exit and retain retry state
        for line in proc.stdout:
            out.write(line); out.flush()
            try:
                event = json.loads(line)
                if event.get('type') == 'thread.started':
                    identifier = event.get('thread_id'); uuid.UUID(identifier)
                    state['thread_id'] = identifier
                state.update(last_event_at=time.time(), last_event_type=event.get('type'))
                if event.get('type') in ('item.started','item.completed'):
                    state['last_item_type'] = event.get('item',{}).get('type')
                save(status, state)
            except (ValueError, TypeError, AttributeError):
                pass
        code = proc.wait()
    failures = state['consecutive_failures']+1 if code else 0
    delay = min(21600, 1800*2**min(failures-1,4)) if code else 120
    state.update(exit_code=code, finished_at=time.time(), consecutive_failures=failures,
        state='RETRY_BACKOFF' if code else 'WAITING_NEXT_RUN', next_eligible_at=time.time()+delay)
    if code == 0 and completion_ready(handoff/'completion.json'):
        state['state'] = 'COMPLETION_REVIEW_PENDING'
    save(status, state)
    return 'STOP' if state['state'] == 'COMPLETION_REVIEW_PENDING' else 'WAIT'


def serve(root=ROOT, *, resource_check=verify, cycle=run_cycle, sleeper=time.sleep, locker=process_lock):
    logs = root/'logs'
    with locker(root) as acquired:
        if not acquired:
            return
        # A check-only start is deliberately possible with PAUSE retained;
        # this verifies the actual scheduled account/job before any model run.
        checking = (root/'RESOURCE-CHECK-ONLY').exists()
        while checking or ((root/'READY').exists() and not (root/'PAUSE').exists()):
            try:
                guard = resource_check()
                save(logs/'resource-guard.json', {**guard, 'checked_at':time.time(), 'pid':os.getpid()})
            except Exception as exc:
                status = logs/'worker-status.json'
                previous = json.loads(status.read_text(encoding='utf-8-sig')) if status.exists() else {}
                save(status, {**previous, 'state':'RESOURCE_GUARD_BLOCKED',
                    'error_type':type(exc).__name__, 'observed_at':time.time(),
                    'live_trading_enabled':False, 'laptop_required':False})
                return
            if checking:
                return
            if cycle(root, guard) == 'STOP':
                return
            # Quiet, bounded polling of the local pause/backoff markers. The
            # single lock covers every resumed turn and every wait period.
            sleeper(30)


if __name__ == '__main__':
    try:
        serve()
    except Exception as exc:
        save(ROOT/'logs/supervisor-error.json', {'time':time.time(), 'error_type':type(exc).__name__})
        raise SystemExit(1)
