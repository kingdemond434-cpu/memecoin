import hashlib
import pytest
from ops.native_artifact_digest import verify_native_digest
from ops.native_runtime_preflight import PRIVATE


def setup(tmp_path):
    path=tmp_path/PRIVATE/'Lib'/'native.pyd'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'trusted fixture')
    return dict(stage=tmp_path, relative_path='Lib/native.pyd',
                expected_size=15, expected_sha256=hashlib.sha256(b'trusted fixture').hexdigest(),
                continue_check=lambda: True, bytes_per_second=1024*1024)


def test_matches_without_authorizing_abi_or_deployment(tmp_path):
    result=verify_native_digest(**setup(tmp_path))
    assert result['native_binary_digest_verified'] and result['bytes_read']==15
    assert not result['native_abi_tested'] and not result['deployment_authorized']


@pytest.mark.parametrize('key,value',[
    ('relative_path','../outside.pyd'),('relative_path','Lib/native.dll'),
    ('expected_sha256','0'*64),('expected_sha256','x'*64),
    ('expected_size',True),('expected_size',14),('max_bytes',13),
    ('deadline_seconds',float('nan')),('deadline_seconds',61),
    ('bytes_per_second',1048577),('continue_check',lambda: False),
])
def test_rejects_bad_expectation_or_admission(tmp_path,key,value):
    args=setup(tmp_path);args[key]=value
    with pytest.raises((ValueError,RuntimeError)):verify_native_digest(**args)


def test_changed_during_admission_rejected(tmp_path):
    args=setup(tmp_path);calls=0
    def admission():
        nonlocal calls
        calls+=1
        if calls==2:
            (tmp_path/PRIVATE/'Lib/native.pyd').write_bytes(b'changed bytes!')
        return True
    args['continue_check']=admission
    with pytest.raises(ValueError):verify_native_digest(**args)
