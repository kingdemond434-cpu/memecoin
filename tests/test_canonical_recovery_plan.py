import copy
import pytest
from ops.canonical_recovery_plan import compose_recovery_plan


def fixture():
    a=dict(offset=100,sequence=1,last_hash='a'*64)
    b=dict(offset=200,sequence=2,last_hash='b'*64)
    c=dict(offset=300,sequence=3,last_hash='c'*64)
    def row(x,y,p):
        return dict(status='SINGLE_CANONICAL_SEGMENT_READBACK_VERIFIED',bytes=100,
            sha256='d'*64,start_cursor=x,next_cursor=y,pending_path=p,
            pending_published=True,deployment_backup_accepted=False)
    return a,[row(a,b,'C:/private/a.segment.pending'),row(b,c,'C:/private/b.segment.pending')]


def test_contiguous_plan_has_no_recovery_authority_or_input_mutation():
    a,rows=fixture();old=copy.deepcopy(rows);r=compose_recovery_plan(a,rows)
    assert r['segment_bytes']==200 and r['segment_records']==2
    assert all(r[k] is False for k in ('baseline_authenticated','segment_contents_revalidated',
        'full_prefix_chain_verified','cross_store_checkpoint_verified','recovery_verified',
        'deployment_backup_accepted','source_write_requested'))
    r['segments'][0]['next_cursor']['offset']=999
    assert rows==old


@pytest.mark.parametrize('field,value',[
    ('bytes',True),('bytes',1048577),('sha256','bad'),('pending_published',1),
    ('deployment_backup_accepted',True),('pending_path','C:/private/../a.segment.pending'),
    ('status','PASS')])
def test_invalid_receipt(field,value):
    a,rows=fixture();rows[0][field]=value
    with pytest.raises(ValueError):compose_recovery_plan(a,rows)


@pytest.mark.parametrize('field,value',[('offset',201),('sequence',4),('last_hash','f'*64)])
def test_gap_overlap_or_hash_conflict(field,value):
    a,rows=fixture();rows[1]['start_cursor']=dict(rows[1]['start_cursor'],**{field:value})
    with pytest.raises(ValueError):compose_recovery_plan(a,rows)


def test_duplicate_windows_reference_and_limits():
    a,rows=fixture();rows[1]['pending_path']='c:\\PRIVATE\\A.segment.pending'
    with pytest.raises(ValueError):compose_recovery_plan(a,rows)
    for bad in ([],rows*17,tuple(rows)):
        with pytest.raises(ValueError):compose_recovery_plan(a,bad)


def test_anchor_boolean_and_unknown_keys_rejected():
    a,rows=fixture()
    for bad in (dict(a,offset=True),dict(a,extra=0)):
        with pytest.raises(ValueError):compose_recovery_plan(bad,rows)
