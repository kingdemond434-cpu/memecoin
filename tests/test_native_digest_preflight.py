import hashlib,json,subprocess
import pytest
from ops.native_runtime_preflight import PRIVATE,inspect_runtime
from ops.native_digest_preflight import probe_digest_bound_runtime


def fixture(tmp_path):
    root=tmp_path/PRIVATE
    (root/'Scripts').mkdir(parents=True)
    (root/'Scripts/python.exe').write_bytes(b'not executed')
    (root/'Lib').mkdir();(root/'Lib/native.pyd').write_bytes(b'fixture')
    envelope=dict(executable=str(root/'Scripts/python.exe'),prefix=str(root),isolated=1,
        abi_smoke=dict(status='OFFLINE_NATIVE_ABI_SMOKE_PASSED',fixed_vectors=5,
            native_abi_smoke_tested=True,full_native_abi_tested=False,
            binary_provenance_verified=False,deployment_authorized=False,
            network_used=False,services_changed=False),
        dont_write_bytecode=True,extensions=[str(root/'Lib/native.pyd')],
        cpu=dict(status='VERIFIED',cpu_rate=200,kernel_flags=5,native_threads=1,total_configured_percent=10))
    def run(*args,**kwargs):return subprocess.CompletedProcess(args,0,json.dumps(envelope),'')
    args=dict(stage=tmp_path,relative_path='Lib/native.pyd',expected_size=7,
        expected_sha256=hashlib.sha256(b'fixture').hexdigest(),continue_check=lambda:True,
        resource_check=lambda:dict(status='VERIFIED',cpu_rate=200),run=run)
    return args,envelope


def test_binds_matching_bytes_and_child_location(tmp_path):
    args,_=fixture(tmp_path);result=probe_digest_bound_runtime(**args)
    assert result['native_binary_digest_verified']
    assert not result['native_abi_tested'] and not result['deployment_authorized']
    assert result['artifact_after']['bytes_read']==7


def test_digest_failure_prevents_child_launch(tmp_path):
    args,_=fixture(tmp_path);args['expected_sha256']='0'*64
    args['run']=lambda *a,**kw:pytest.fail('must not import mismatching binary')
    with pytest.raises(ValueError):probe_digest_bound_runtime(**args)


def test_different_child_extension_rejected(tmp_path):
    args,row=fixture(tmp_path);row['extensions']=[str(tmp_path/PRIVATE/'Lib/other.pyd')]
    with pytest.raises(RuntimeError):probe_digest_bound_runtime(**args)


def test_mutation_in_child_fails_second_digest(tmp_path):
    args,_=fixture(tmp_path);original=args['run']
    def run(*a,**kw):
        (tmp_path/PRIVATE/'Lib/native.pyd').write_bytes(b'changed')
        return original(*a,**kw)
    args['run']=run
    with pytest.raises(ValueError):probe_digest_bound_runtime(**args)


def test_no_artifact_read_without_cpu_proof(tmp_path):
    args,_=fixture(tmp_path);args['resource_check']=lambda:dict(status='UNVERIFIED')
    with pytest.raises(RuntimeError,match='CPU'):probe_digest_bound_runtime(**args)


def test_optional_exact_location_gate_in_inspector(tmp_path):
    _,row=fixture(tmp_path)
    with pytest.raises(ValueError):inspect_runtime(row,tmp_path,expected_extension=tmp_path/'wrong.pyd')


@pytest.mark.parametrize('value',[None,{},dict(status='OFFLINE_NATIVE_ABI_SMOKE_PASSED')])
def test_missing_or_incomplete_smoke_rejected(tmp_path,value):
    args,row=fixture(tmp_path);row['abi_smoke']=value
    with pytest.raises(RuntimeError):probe_digest_bound_runtime(**args)


@pytest.mark.parametrize('key,value',[('fixed_vectors',True),('fixed_vectors',4),
    ('native_abi_smoke_tested',False),('deployment_authorized',True),
    ('full_native_abi_tested',True),('network_used',True)])
def test_smoke_contract_cannot_claim_broader_authority(tmp_path,key,value):
    args,row=fixture(tmp_path);row['abi_smoke'][key]=value
    with pytest.raises(RuntimeError):probe_digest_bound_runtime(**args)


def test_composed_probe_requests_smoke_in_actual_child_script(tmp_path):
    args,_=fixture(tmp_path);original=args['run']
    def run(command,**kwargs):
        assert 'if True:' in command[4]
        assert 'checker.check_native_abi_smoke(solana_fastpath)' in command[4]
        assert kwargs['timeout']==60
        return original(command,**kwargs)
    args['run']=run
    result=probe_digest_bound_runtime(**args)
    assert result['native_abi_smoke_tested'] and not result['native_abi_tested']
