import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from ops.bounded_preflight import inspect_task_xml, snapshot_bounded_task, verify_tested_release


def xml():
    return '''<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"><RegistrationInfo><Description>fixture</Description></RegistrationInfo><Principals><Principal id="Author"><UserId>S-1-5-19</UserId><LogonType>ServiceAccount</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals><Settings><Priority>7</Priority><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy></Settings><Actions Context="Author"><Exec><Command>C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe</Command><Arguments>-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "C:\\ProgramData\\MemecoinResourceLimits\\start-cpu-bounded.ps1" -ManifestPath "C:\\ProgramData\\MemecoinResourceLimits\\roles.json" -Role spot</Arguments><WorkingDirectory>C:\\ProgramData\\MemecoinResourceLimits</WorkingDirectory></Exec></Actions></Task>'''


def test_task_snapshot_covers_entire_definition_without_raw_actions():
    before = inspect_task_xml(xml(), 'spot')
    after = inspect_task_xml(xml().replace('fixture', 'changed description'), 'spot')
    assert before['wrapped'] and before['priority'] == 7
    assert before['definition_sha256'] != after['definition_sha256']
    assert 'Arguments' not in json.dumps(before)
    assert 'Command' not in json.dumps(before)


@pytest.mark.parametrize('old,new', [('S-1-5-19','S-1-5-18'), ('LeastPrivilege','HighestAvailable'),
    ('ServiceAccount','Password'), ('<Priority>7','<Priority>5'), ('IgnoreNew','Parallel'),
    ('-Role spot','-Role capture'), ('start-cpu-bounded.ps1','unwrapped.ps1'),
    ('-Role spot','-Role spot -Command injected'),
    ('<Exec>','<ComHandler>'), ('<Actions Context="Author">','<Actions Context="Author"><Exec/>')])
def test_task_identity_or_action_change_rejected(old,new):
    with pytest.raises((ValueError, ET.ParseError)):
        inspect_task_xml(xml().replace(old,new), 'spot')


def test_task_declarations_rejected():
    with pytest.raises(ValueError):
        inspect_task_xml('<!DOCTYPE Task>'+xml(), 'spot')


def test_export_is_read_only_and_strips_unrelated_tokens(monkeypatch):
    monkeypatch.setenv('GH_TOKEN','fixture-secret');monkeypatch.setenv('GITHUB_TOKEN','fixture-other')
    def run(command, **kwargs):
        assert '.GetTask(' in command[-1] and ').Definition' in command[-1]
        assert 'Schedule.Service' in command[-1] and '$service.Connect();' in command[-1]
        assert 'Set-ScheduledTask' not in command[-1]
        assert 'GH_TOKEN' not in kwargs['env'] and 'GITHUB_TOKEN' not in kwargs['env']
        assert kwargs['timeout'] == 60
        return SimpleNamespace(stdout=json.dumps({'xml': xml(), 'properties': properties()}))
    assert snapshot_bounded_task('spot',run=run)['wrapped']


def test_export_failure_suppresses_private_output():
    def denied(*a, **kw):
        raise RuntimeError('PRIVATE-EXAMPLE-ERROR')
    with pytest.raises(RuntimeError) as exc:
        snapshot_bounded_task('spot',run=denied)
    assert 'PRIVATE-EXAMPLE-ERROR' not in str(exc.value)


@pytest.fixture(scope='module')
def revisions(tmp_path_factory):
    root=tmp_path_factory.mktemp('isolated-revision-fixture'); repo=root/'repo';repo.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    (repo/'source.txt').write_text('fixture source');git('add','.');git('commit','-m','fixture source')
    tested=git('rev-parse','HEAD'); isolated=root/'isolated'
    subprocess.run(['git','clone','--local',str(repo),str(isolated)],check=True,capture_output=True)
    git('commit','--allow-empty','-m','fixture equivalent tree');equivalent=git('rev-parse','HEAD')
    (repo/'source.txt').write_text('changed source');git('add','.');git('commit','-m','fixture differing source')
    changed=git('rev-parse','HEAD')
    return root,repo,isolated,tested,equivalent,changed


def report(revisions,path):
    root,repo,isolated,tested,equivalent,changed=revisions
    row=dict(status='PASS',revision=tested,checkout=str(isolated),exit_code=0,
             counts=dict(passed=1,failed=0,errors=0),isolated_checkout_clean=True,
             native_fields_preflight=True,native_source_unchanged=True,production_state_copied=False)
    path.write_text(json.dumps(row));return row


def test_actual_equivalent_merge_tree_accepted_but_no_deployment_claim(revisions,tmp_path):
    path=tmp_path/'report.json';report(revisions,path)
    result=verify_tested_release(revisions[1],revisions[4],path)
    assert result['status']=='EXACT_TESTED_TREE_VERIFIED'
    assert not result['backup_verified'] and not result['ci_verified'] and not result['deployed']


def test_actual_changed_tree_rejected(revisions,tmp_path):
    path=tmp_path/'report.json';report(revisions,path)
    with pytest.raises(ValueError,match='differs'):
        verify_tested_release(revisions[1],revisions[5],path)


@pytest.mark.parametrize('key,value',[('revision','0'*40),('status','FAIL'),('exit_code',True),
    ('isolated_checkout_clean',False),('native_fields_preflight',False),
    ('native_source_unchanged',False),('production_state_copied',True)])
def test_report_claims_cannot_replace_actual_gates(revisions,tmp_path,key,value):
    path=tmp_path/'report.json';row=report(revisions,path);row[key]=value;path.write_text(json.dumps(row))
    with pytest.raises((ValueError,RuntimeError)):
        verify_tested_release(revisions[1],revisions[4],path)


def test_actual_dirty_isolated_checkout_rejected(revisions,tmp_path):
    path=tmp_path/'report.json';report(revisions,path)
    dirty=revisions[2]/'source.txt';original=dirty.read_bytes()
    try:
        dirty.write_text('dirty fixture')
        with pytest.raises(ValueError,match='Clean'):
            verify_tested_release(revisions[1],revisions[4],path)
    finally:dirty.write_bytes(original)


def properties():
    return dict(user_id='S-1-5-19',logon_type=5,run_level=0,priority=7,multiple_instances=2)


def test_omitted_defaults_need_actual_typed_properties():
    omitted = xml().replace('<LogonType>ServiceAccount</LogonType>','').replace('<RunLevel>LeastPrivilege</RunLevel>','').replace('<Priority>7</Priority>','')
    with pytest.raises(ValueError):
        inspect_task_xml(omitted,'spot')
    assert inspect_task_xml(omitted,'spot',definition_properties=properties())['wrapped']


@pytest.mark.parametrize('key,value',[('logon_type',1),('run_level',1),('priority',5),('multiple_instances',0),('user_id','S-1-5-18')])
def test_typed_properties_cannot_override_conflicting_identity(key,value):
    actual=properties();actual[key]=value
    with pytest.raises(ValueError):
        inspect_task_xml(xml(),'spot',definition_properties=actual)


@pytest.mark.parametrize('name',['LOCAL SERVICE','NT AUTHORITY\\LOCAL SERVICE'])
def test_native_localservice_name_matches_xml_builtin_sid(name):
    actual=properties();actual['user_id']=name
    assert inspect_task_xml(xml(),'spot',definition_properties=actual)['principal']=='LOCAL_SERVICE'
