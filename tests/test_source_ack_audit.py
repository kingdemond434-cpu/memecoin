import sqlite3

import pytest

from src.collectors.event_source import Event, SourceClass
from src.collectors.source_journal import SourceJournal
from src.research.frontier import FrontierIndex
from src.research.trade_evidence import TradeEvidenceLedger
from tools.source_ack_audit import audit_acknowledgements


def fixture(tmp_path, *, acknowledge=True, prefix_rows=0):
    journal=SourceJournal(tmp_path/'journal.sqlite')
    frontier=FrontierIndex(tmp_path/'frontier.sqlite')
    evidence=TradeEvidenceLedger(tmp_path/'evidence.jsonl')
    for _ in range(prefix_rows):
        evidence.record_reference('fixture_padding',{'padding':'x'*4096},timestamp=2)
    journal.append([Event('source',SourceClass.NEWS,1,2,text='private fixture')],3)
    event,receipt,generation=journal.lease(4)
    ref=evidence.record_reference('research_typed_work',{'work_kind':'NARRATIVE_OBSERVATION',
        'point_payload':{'observed_at':2,'research_available_at':4,'source_intake_receipt':receipt}},timestamp=4)
    frontier.enqueue('NARRATIVE_OBSERVATION',ref,job_id='work-fixture',now=4)
    if acknowledge:
        journal.acknowledge(event,receipt,generation,'work-fixture',ref,5)
    journal.close();frontier.close()
    return tmp_path/'journal.sqlite',tmp_path/'frontier.sqlite',tmp_path/'evidence.jsonl'


def test_real_ack_target_receipt_and_hash_verified_without_mutation(tmp_path):
    paths=fixture(tmp_path)
    before=[p.read_bytes() for p in paths]
    result=audit_acknowledgements(*paths)
    assert result['status']=='PASS' and result['verified_acknowledgements']==1
    assert result['full_prefix_chain_verified'] is False
    assert 'private fixture' not in str(result)
    assert [p.read_bytes() for p in paths]==before


@pytest.mark.parametrize('corruption',['frontier','receipt','ledger'])
def test_corrupt_targets_fail_without_exporting_contents(tmp_path,corruption):
    paths=fixture(tmp_path)
    if corruption=='ledger':
        paths[2].write_bytes(paths[2].read_bytes().replace(b'"observed_at":2',b'"observed_at":9'))
    else:
        with sqlite3.connect(paths[0] if corruption=='receipt' else paths[1]) as db:
            db.execute("UPDATE intake SET body_sha256='wrong'" if corruption=='receipt'
                       else "UPDATE jobs SET payload_ref='wrong'")
    result=audit_acknowledgements(*paths)
    assert result['status']=='FAIL' and result['verified_acknowledgements']==0


def test_missing_databases_are_not_created(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        audit_acknowledgements(tmp_path/'missing',tmp_path/'frontier',tmp_path/'ledger')
    assert not (tmp_path/'missing').exists()


from tools.source_ack_tail_audit import audit_recent_acknowledgements


def test_tail_real_ack_is_verified_without_content_or_mutation(tmp_path):
    paths=fixture(tmp_path);before=[p.read_bytes() for p in paths]
    result=audit_recent_acknowledgements(*paths)
    assert result['status']=='SAMPLED_BINDINGS_VERIFIED'
    assert result['verified_acknowledgements']==1
    assert not result['sqlite_integrity_verified'] and not result['durable_owner_application_verified']
    assert 'private fixture' not in str(result)
    assert [p.read_bytes() for p in paths]==before


@pytest.mark.parametrize('corruption',['frontier','receipt','ledger'])
def test_tail_corrupt_target_rejected(tmp_path,corruption):
    paths=fixture(tmp_path)
    if corruption=='ledger':
        paths[2].write_bytes(paths[2].read_bytes().replace(b'"observed_at":2',b'"observed_at":9'))
    else:
        with sqlite3.connect(paths[0] if corruption=='receipt' else paths[1]) as db:
            db.execute("UPDATE intake SET body_sha256='wrong'" if corruption=='receipt' else "UPDATE jobs SET payload_ref='wrong'")
    result=audit_recent_acknowledgements(*paths)
    assert result['status']=='FAIL_SAMPLED_BINDINGS' and result['verified_acknowledgements']==0


def test_tail_pending_is_not_a_fill_or_ack(tmp_path):
    result=audit_recent_acknowledgements(*fixture(tmp_path,acknowledge=False))
    assert result['status']=='NO_ACK_TARGET_SAMPLE'
    assert result['pending_receipts']==1 and result['verified_acknowledgements']==0


def test_tail_bounds_large_prefix_and_partial_final_row(tmp_path):
    paths=fixture(tmp_path,prefix_rows=40)
    with paths[2].open('ab') as stream:stream.write(b'{"unfinished":')
    result=audit_recent_acknowledgements(*paths,tail_bytes=4096,max_lines=1)
    assert result['status']=='SAMPLED_BINDINGS_VERIFIED'
    assert result['bytes_read']==4096 and result['lines_inspected']==1
    assert result['incomplete_tail'] and result['leading_line_discarded']
    assert not result['full_prefix_chain_verified']


def test_tail_retained_non_target_retry_does_not_claim_acceptance(tmp_path):
    import json
    paths=fixture(tmp_path)
    row=json.loads(paths[2].read_bytes().splitlines()[0])
    ledger=TradeEvidenceLedger(paths[2])
    ledger.record_reference('research_typed_work',row['payload'],timestamp=6)
    result=audit_recent_acknowledgements(*paths)
    assert result['status']=='SAMPLED_BINDINGS_VERIFIED'
    assert result['non_target_rows']==1 and result['verified_acknowledgements']==1


@pytest.mark.parametrize('which,table',[(0,'intake'),(1,'jobs')])
def test_tail_rejects_unindexed_selection_before_scan(tmp_path,which,table):
    paths=fixture(tmp_path)
    columns = ('id','body_sha256','accepted','first_delivery','canonical_job','canonical_ref','acknowledged','body') if which==0 else ('job_id','kind','payload_ref')
    with sqlite3.connect(paths[which]) as db:
        row=db.execute('SELECT '+','.join(columns)+' FROM '+table).fetchone()
    replacement=tmp_path/'unindexed.sqlite'
    with sqlite3.connect(replacement) as db:
        db.execute('CREATE TABLE '+table+'('+','.join(columns)+')')
        db.execute('INSERT INTO '+table+' VALUES('+','.join('?' for _ in columns)+')',row)
    paths=tuple(replacement if i==which else path for i,path in enumerate(paths))
    result=audit_recent_acknowledgements(*paths)
    assert result['status']=='RESOURCE_BLOCKED'
    assert result['budget_reason']=='unindexed_'+table+'_lookup'


def test_tail_queries_do_not_scan_integrity_or_sort_ack_history(tmp_path,monkeypatch):
    paths=fixture(tmp_path);queries=[];original=sqlite3.connect
    def connect(*args,**kwargs):
        db=original(*args,**kwargs);db.set_trace_callback(queries.append);return db
    monkeypatch.setattr('tools.source_ack_tail_audit.sqlite3.connect',connect)
    assert audit_recent_acknowledgements(*paths)['verified_acknowledgements']==1
    joined=' '.join(queries).lower()
    assert 'quick_check' not in joined and 'integrity_check' not in joined
    assert 'count(' not in joined and 'order by' not in joined
    assert 'where id=' in joined and 'where job_id=' in joined


def test_tail_deadline_stops_without_claiming_integrity(tmp_path):
    paths=fixture(tmp_path);tick=0
    def clock():
        nonlocal tick
        tick+=1;return tick
    result=audit_recent_acknowledgements(*paths,deadline_seconds=.1,clock=clock)
    assert result['status']=='RESOURCE_BLOCKED' and result['budget_reason']=='deadline'
    assert not result['sqlite_integrity_verified']


@pytest.mark.parametrize('field,value',[('sample_limit',True),('tail_bytes',2**21),('max_lines',0),('vm_budget',True),('deadline_seconds',float('nan'))])
def test_tail_invalid_resource_policy_rejected(tmp_path,field,value):
    with pytest.raises(ValueError):
        audit_recent_acknowledgements(*fixture(tmp_path),**{field:value})


def test_tail_malformed_ack_reference_is_a_failure_not_hidden_as_retry(tmp_path):
    paths=fixture(tmp_path)
    with sqlite3.connect(paths[0]) as db:db.execute("UPDATE intake SET canonical_ref='wrong'")
    result=audit_recent_acknowledgements(*paths)
    assert result['status']=='FAIL_SAMPLED_BINDINGS'
    assert result['non_target_rows']==0 and result['verified_acknowledgements']==0


def test_tail_reports_only_verified_journal_clock_ranges(tmp_path):
    result=audit_recent_acknowledgements(*fixture(tmp_path))
    assert result['journal_timings']['intake_to_first_delivery_seconds']=={'count':1,'min':1.0,'max':1.0}
    assert result['journal_timings']['first_delivery_to_ack_seconds']=={'count':1,'min':1.0,'max':1.0}
    assert result['journal_timings']['intake_to_ack_seconds']=={'count':1,'min':2.0,'max':2.0}
    empty=audit_recent_acknowledgements(*fixture(tmp_path/'pending',acknowledge=False))
    assert empty['journal_timings']['intake_to_ack_seconds']=={'count':0,'min':None,'max':None}


def test_tail_explicit_windows_progress_without_automatically_expanding_budget(tmp_path):
    paths=fixture(tmp_path);ledger=TradeEvidenceLedger(paths[2])
    for _ in range(40):ledger.record_reference('fixture_padding',{'padding':'x'*4096},timestamp=6)
    cursor=None;verified=0;results=[]
    for _ in range(4):
        result=audit_recent_acknowledgements(*paths,tail_bytes=65536,end_offset=cursor)
        results.append(result);assert result['bytes_read']<=65536
        verified+=result['verified_acknowledgements']
        following=result['next_window_end']
        if following is None:break
        assert following < result['window_end'];cursor=following
    assert results[0]['status']=='NO_ACK_TARGET_SAMPLE'
    assert verified==1
    assert all(not r['full_prefix_chain_verified'] for r in results)


@pytest.mark.parametrize('cursor',[True,-1,10**20])
def test_tail_invalid_cursor_rejected(tmp_path,cursor):
    with pytest.raises(ValueError):audit_recent_acknowledgements(*fixture(tmp_path),end_offset=cursor)


def test_tail_oversized_boundary_is_explicitly_unsampled(tmp_path):
    paths=fixture(tmp_path)
    with paths[2].open('ab') as stream:stream.write(b'x'*4096)
    result=audit_recent_acknowledgements(*paths,tail_bytes=1024)
    assert result['status']=='NO_ACK_TARGET_SAMPLE'
    assert result['boundary_row_not_sampled'] and result['incomplete_tail']
    assert result['next_window_end'] < result['window_end']


def test_tail_no_candidate_does_not_open_or_imply_verification_of_stores(tmp_path,monkeypatch):
    paths=fixture(tmp_path);ledger=TradeEvidenceLedger(paths[2])
    ledger.record_reference('fixture_padding',{'padding':'x'*4096},timestamp=6)
    monkeypatch.setattr('tools.source_ack_tail_audit.sqlite3.connect',lambda *a,**k:pytest.fail('no receipt candidate needs a store lookup'))
    result=audit_recent_acknowledgements(*paths,tail_bytes=1024)
    assert result['status']=='NO_ACK_TARGET_SAMPLE' and result['candidate_rows']==0
    assert not result['sqlite_integrity_verified']
