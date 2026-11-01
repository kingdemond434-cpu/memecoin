import pytest
from ops.native_test_manifest import bind_parity_manifest,CLASS,NAMES


def args():
    xml='<testsuites><testsuite>'+''.join('<testcase classname="'+CLASS+'" name="'+n+'" />' for n in sorted(NAMES))+'</testsuite></testsuites>'
    return dict(revision='a'*40,tree='b'*40,before={'sha256':'c'*64,'size':123},
                after={'sha256':'c'*64,'size':123},junit=xml.encode(),exit_code=0)


def test_binds_scope_without_claiming_build_or_deploy():
    result=bind_parity_manifest(**args())
    assert result['passed']==6 and result['artifact_size']==123
    assert not any(result[k] for k in ('build_provenance_verified','independent_execution_attestation',
        'full_native_abi_tested','deployment_authorized'))


@pytest.mark.parametrize('key,value',[('revision','main'),('tree','x'*40),('exit_code',True),
    ('exit_code',1),('after',{'sha256':'d'*64,'size':123}),('junit',b'<!DOCTYPE x><x/>'),
    ('junit',b'bad xml'),('junit',b'x'*65537)], ids=lambda value: 'case')
def test_rejects_invalid_binding(key,value):
    row=args();row[key]=value
    with pytest.raises(ValueError):bind_parity_manifest(**row)


@pytest.mark.parametrize('mutation',[lambda x:x.replace(b'<testcase',b'<testcase><skipped/></testcase><testcase',1),
 lambda x:x.replace(CLASS.encode(),b'wrong',1),lambda x:x.replace(b'<testsuite>',b'<testsuite><failure/>',1)])
def test_no_skip_wrong_scope_or_failure_acceptance(mutation):
    row=args();row['junit']=mutation(row['junit'])
    with pytest.raises(ValueError):bind_parity_manifest(**row)
