import json
import subprocess
from pathlib import Path

import pytest
from ops.native_runtime_preflight import inspect_runtime, probe_private_runtime, PRIVATE
from ops.worker_resource_guard import THREAD_VARIABLES


def envelope(stage):
    root=(stage/PRIVATE).resolve()
    return dict(executable=str(root/'Scripts/python.exe'),prefix=str(root),isolated=1,
                dont_write_bytecode=True,extensions=[str(root/'Lib/site-packages/solana_fastpath/core.pyd')],
                cpu=dict(status='VERIFIED',cpu_rate=200,native_threads=1,total_configured_percent=10,kernel_flags=5))


def test_location_and_cpu_proof_never_equals_binary_abi_or_deploy(tmp_path):
    result=inspect_runtime(envelope(tmp_path),tmp_path)
    assert result['private_extension_location_verified']
    assert not result['native_binary_digest_verified'] and not result['native_abi_tested']
    assert not result['deployment_authorized']


@pytest.mark.parametrize('key,value',[('isolated',True),('isolated',0),('dont_write_bytecode',False),
    ('executable','shared/.venv/Scripts/python.exe'),('prefix','shared'),
    ('extensions',[]),('extensions',['shared/native.pyd']),('extensions',['a.pyd','b.pyd'])])
def test_shared_or_unisolated_runtime_rejected(tmp_path,key,value):
    row=envelope(tmp_path);row[key]=value
    with pytest.raises(ValueError):inspect_runtime(row,tmp_path)


@pytest.mark.parametrize('key,value',[('kernel_flags',0),('native_threads',True),('cpu_rate',0),('native_threads',4),('total_configured_percent',11)])
def test_child_must_prove_its_own_containment(tmp_path,key,value):
    row=envelope(tmp_path);row['cpu'][key]=value
    with pytest.raises(ValueError):inspect_runtime(row,tmp_path)


def test_launcher_uses_only_private_direct_child_with_clean_environment(tmp_path,monkeypatch):
    root=tmp_path/PRIVATE;(root/'Scripts').mkdir(parents=True)
    (root/'Scripts/python.exe').write_bytes(b'fixture only; never executed')
    for k in ('GH_TOKEN','GITHUB_TOKEN','HELIUS_API_KEY','ALLOW_LIVE_TRADING','SIGNER_KEY'):
        monkeypatch.setenv(k,'private fixture sentinel')
    for k in THREAD_VARIABLES:monkeypatch.setenv(k,'1')
    monkeypatch.setenv('TOKENIZERS_PARALLELISM','false')
    def run(args,**kw):
        assert args[:4]==[str(root.resolve()/'Scripts/python.exe'),'-I','-B','-c']
        assert kw['timeout']==60 and kw['check'] and kw['cwd']==str(root.resolve())
        assert not any(k in kw['env'] for k in ('GH_TOKEN','GITHUB_TOKEN','HELIUS_API_KEY','ALLOW_LIVE_TRADING','SIGNER_KEY'))
        assert all(kw['env'][k]=='1' for k in THREAD_VARIABLES)
        assert 'import solana_fastpath' in args[4]
        return subprocess.CompletedProcess(args,0,json.dumps(envelope(tmp_path)),'')
    assert probe_private_runtime(stage=tmp_path,run=run,
        resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))['isolated_child']


def test_failed_probe_hides_private_child_errors(tmp_path):
    root=tmp_path/PRIVATE;(root/'Scripts').mkdir(parents=True)
    (root/'Scripts/python.exe').write_bytes(b'fixture')
    def run(*args,**kw):raise subprocess.CalledProcessError(1,'private action',stderr='private provider details')
    with pytest.raises(RuntimeError) as caught:
        probe_private_runtime(stage=tmp_path,run=run,resource_check=lambda:dict(status='VERIFIED',cpu_rate=200))
    assert 'provider' not in str(caught.value) and caught.value.__suppress_context__
