import sqlite3

import pytest

from src.collectors.event_source import Event, SourceClass
from src.collectors.source_journal import SourceJournal
from src.research.frontier import FrontierIndex
from src.research.trade_evidence import TradeEvidenceLedger
from tools.source_ack_audit import audit_acknowledgements


def fixture(tmp_path):
    journal=SourceJournal(tmp_path/'journal.sqlite')
    frontier=FrontierIndex(tmp_path/'frontier.sqlite')
    evidence=TradeEvidenceLedger(tmp_path/'evidence.jsonl')
    journal.append([Event('source',SourceClass.NEWS,1,2,text='private fixture')],3)
    event,receipt,generation=journal.lease(4)
    ref=evidence.record_reference('research_typed_work',{'work_kind':'NARRATIVE_OBSERVATION',
        'point_payload':{'observed_at':2,'research_available_at':4,'source_intake_receipt':receipt}},timestamp=4)
    frontier.enqueue('NARRATIVE_OBSERVATION',ref,job_id='work-fixture',now=4)
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
