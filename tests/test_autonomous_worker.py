"""Supervisor fixtures use fake model children, never credentials or VPS data."""
from contextlib import nullcontext
import io
import json
import uuid

import pytest

from ops import autonomous_worker as worker
from ops import worker_resource_guard as guard


@pytest.fixture
def root(tmp_path):
    for name in ('logs','handoff','repo'):
        (tmp_path/name).mkdir()
    (tmp_path/'READY').touch()
    (tmp_path/'handoff/WORKER-INSTRUCTIONS.txt').write_text('Base scope: paper only.',encoding='utf-8')
    (tmp_path/'handoff/CONTINUATION-20261031.txt').write_text('Protect quant; preserve CPU wrappers.',encoding='utf-8')
    return tmp_path


@pytest.fixture
def policy(tmp_path):
    path=tmp_path/'roles.json'
    path.write_text(json.dumps({'version':1,'roles':{
        'spot':{'cpu_rate':500},'prediction':{'cpu_rate':100},'capture':{'cpu_rate':150},
        'terminal':{'cpu_rate':50},'development':{'cpu_rate':200}}}),encoding='utf-8')
    env={name:'1' for name in guard.THREAD_VARIABLES}
    env['TOKENIZERS_PARALLELISM']='false'
    return path,env


def test_guard_requires_kernel_rate_manifest_and_threads(policy):
    path,env=policy
    result=guard.verify(path,env=env,query=lambda:{'flags':5,'rate':200})
    assert result['cpu_percent'] == 2 and result['total_configured_percent'] == 10
    with pytest.raises(RuntimeError):
        guard.verify(path,env=env,query=lambda:{'flags':1,'rate':200})
    with pytest.raises(RuntimeError):
        guard.verify(path,env=env,query=lambda:{'flags':5,'rate':300})
    with pytest.raises(RuntimeError):
        guard.verify(path,env={**env,'OMP_NUM_THREADS':'16'},query=lambda:{'flags':5,'rate':200})


@pytest.mark.parametrize('development,spot',[(201,500),(200,501),(True,500),(-1,500)])
def test_invalid_resource_contract_cannot_start_a_model(policy,development,spot):
    path,env=policy
    p=json.loads(path.read_text())
    p['roles']['development']['cpu_rate']=development
    p['roles']['spot']['cpu_rate']=spot
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        guard.verify(path,env=env,query=lambda:pytest.fail('invalid policy must not reach native query'))


def test_child_environment_drops_unrelated_github_and_trading_authority(root):
    env=worker.child_environment(root,{'GH_TOKEN':'unrelated','GITHUB_TOKEN':'other',
        'ALLOW_LIVE_TRADING':'armed','SOLANA_PRIVATE_KEY':'fixture-not-a-key','OMP_NUM_THREADS':'1'})
    assert 'GH_TOKEN' not in env and 'GITHUB_TOKEN' not in env
    assert env['ALLOW_LIVE_TRADING'] == env['SOLANA_PRIVATE_KEY'] == ''
    assert env['OMP_NUM_THREADS'] == '1' and env['GIT_TERMINAL_PROMPT'] == '0'


def test_preflight_checks_actual_guard_without_resuming_even_with_pause(root):
    (root/'PAUSE').touch();(root/'RESOURCE-CHECK-ONLY').touch()
    worker.serve(root,resource_check=lambda:{'status':'VERIFIED','cpu_rate':200},
        cycle=lambda *args:pytest.fail('preflight must not invoke a model'),locker=lambda r:nullcontext(True))
    proof=json.loads((root/'logs/resource-guard.json').read_text())
    assert proof['cpu_rate'] == 200 and proof['pid'] > 0


def test_guard_failure_keeps_thread_and_never_spawns(root):
    thread=str(uuid.uuid4())
    worker.save(root/'logs/worker-status.json',{'thread_id':thread})
    def blocked():raise RuntimeError('fixture: no capped job')
    worker.serve(root,resource_check=blocked,cycle=lambda *args:pytest.fail('no child allowed'),
                 locker=lambda r:nullcontext(True))
    state=json.loads((root/'logs/worker-status.json').read_text())
    assert state['state'] == 'RESOURCE_GUARD_BLOCKED' and state['thread_id'] == thread


def test_pause_or_existing_lock_prevents_another_worker(root):
    (root/'PAUSE').touch()
    worker.serve(root,resource_check=lambda:pytest.fail('paused'),locker=lambda r:nullcontext(True))
    (root/'PAUSE').unlink()
    worker.serve(root,resource_check=lambda:pytest.fail('lock held'),locker=lambda r:nullcontext(False))


class PromptSink:
    def __init__(self):self.value=''
    def write(self,text):self.value+=text
    def close(self):pass


class Child:
    def __init__(self,thread,code=0):
        self.pid=123
        self.code=code
        self.stdin=PromptSink()
        self.stdout=io.StringIO(json.dumps({'type':'thread.started','thread_id':thread})+'\n'
            +json.dumps({'type':'item.completed','item':{'type':'command_execution'}})+'\n')
    def wait(self):return self.code


@pytest.mark.parametrize('code,state,delay',[(0,'WAITING_NEXT_RUN',120),(1,'RETRY_BACKOFF',1800)])
def test_sequential_resume_uses_auto_review_and_records_backoff(root,monkeypatch,code,state,delay):
    monkeypatch.setattr(worker.time,'time',lambda:10000)
    thread=str(uuid.uuid4())
    worker.save(root/'logs/worker-status.json',{'thread_id':thread})
    child=Child(thread,code)
    captured={}
    def popen(cmd,**kw):captured.update(cmd=cmd,kwargs=kw);return child
    assert worker.run_cycle(root,{'status':'VERIFIED'},popen=popen) == 'WAIT'
    status=json.loads((root/'logs/worker-status.json').read_text())
    assert status['state'] == state and status['next_eligible_at'] == 10000+delay
    assert status['thread_id'] == thread and not status['live_trading_enabled']
    cmd=captured['cmd']
    assert '--approve-for-me' in cmd and cmd[cmd.index('resume')+1] == thread
    assert not any('dangerously' in x or 'danger-full-access' in x for x in cmd)
    assert 'Protect quant' in child.stdin.value and 'do not create another worker' in child.stdin.value


def test_account_backoff_and_review_pending_do_not_repeat_api_calls(root,monkeypatch):
    monkeypatch.setattr(worker.time,'time',lambda:100)
    status=root/'logs/worker-status.json'
    worker.save(status,{'next_eligible_at':200})
    assert worker.run_cycle(root,{},popen=lambda *a,**kw:pytest.fail('backoff')) == 'WAIT'
    worker.save(status,{'state':'COMPLETION_REVIEW_PENDING'})
    assert worker.run_cycle(root,{},popen=lambda *a,**kw:pytest.fail('pending review')) == 'STOP'


def test_continuation_loop_remains_single_and_obeys_a_new_pause(root):
    calls=[]
    def cycle(r,proof):calls.append(proof);return 'WAIT'
    def sleep(seconds):
        assert seconds == 30
        if len(calls) == 2:(root/'PAUSE').touch()
    worker.serve(root,resource_check=lambda:{'status':'VERIFIED'},cycle=cycle,sleeper=sleep,
                 locker=lambda r:nullcontext(True))
    assert len(calls) == 2


def test_false_or_incomplete_completion_cannot_stop_continuation(root):
    path=root/'handoff/completion.json'
    worker.save(path,{'all_blueprints_complete_excluding_live':True,'outstanding_requirements':['API missing'],
                     'dashboard_url':'https://fixture.invalid','evidence':['synthetic']})
    assert not worker.completion_ready(path)
    worker.save(path,{'all_blueprints_complete_excluding_live':True,'outstanding_requirements':[],
                     'dashboard_url':'http://127.0.0.1:18083','evidence':['synthetic']})
    assert not worker.completion_ready(path)
    path.write_text('not-json')
    assert not worker.completion_ready(path)
