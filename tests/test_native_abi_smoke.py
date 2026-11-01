import hashlib
from types import SimpleNamespace
import pytest
from ops.native_abi_smoke import check_native_abi_smoke


def native():
    def decode(value):
        if value=='0':raise ValueError('invalid')
        return bytes(32)
    return SimpleNamespace(b58encode=lambda value:'1'*32,b58decode=decode,
        anchor_discriminator=lambda value:hashlib.sha256(('global:'+value).encode()).digest()[:8],
        hazard_predict=lambda *args:.5)


def test_fixed_surface_does_not_authorize_execution():
    result=check_native_abi_smoke(native())
    assert result['fixed_vectors']==5 and result['native_abi_smoke_tested']
    assert not any(result[k] for k in ('full_native_abi_tested','binary_provenance_verified',
        'deployment_authorized','network_used','services_changed'))


@pytest.mark.parametrize('name,replacement',[
    ('b58encode',lambda value: 'wrong'),('b58decode',lambda value: bytes(32)),
    ('anchor_discriminator',lambda value: bytes(8)),('hazard_predict',lambda *args:float('nan')),
    ('hazard_predict',lambda *args:True),('b58decode',lambda value:[1]*32),
])
def test_bad_results_rejected(name,replacement):
    module=native();setattr(module,name,replacement)
    with pytest.raises(RuntimeError):check_native_abi_smoke(module)


def test_private_exception_suppressed():
    module=native()
    def fail(*args):raise RuntimeError('private provider sentinel')
    module.hazard_predict=fail
    with pytest.raises(RuntimeError) as caught:check_native_abi_smoke(module)
    assert 'sentinel' not in str(caught.value) and caught.value.__suppress_context__


def test_missing_binding_rejected():
    module=native();del module.hazard_predict
    with pytest.raises(RuntimeError):check_native_abi_smoke(module)
