import sqlite3

import pytest

from src.collectors.event_source import Event, SourceClass
from src.collectors.source_journal import SourceJournal
from tools.source_journal_audit import audit


def test_audit_reconciles_pending_and_ack_without_mutation(tmp_path):
    path = tmp_path / 'journal.sqlite'
    journal = SourceJournal(path)
    journal.append([Event('private-source', SourceClass.NEWS, 1, 2, text='private body')], 3)
    before = journal.db.total_changes
    report = audit(path, now=10)
    assert report['status'] == 'PASS'
    assert report['pending'] == 1 and report['oldest_pending_age_seconds'] == 7
    assert 'private' not in str(report)
    assert journal.db.total_changes == before
    event, receipt, generation = journal.lease(11)
    journal.acknowledge(event, receipt, generation, 'work-test', 'ref-test', 12)
    report = audit(path, now=13)
    assert report['status'] == 'PASS' and report['acknowledged'] == 1
    assert report['sampled_pending_bodies'] == 0
    assert report['canonical_references_externally_verified'] is False
    journal.close()


def test_audit_detects_body_counter_and_receipt_corruption(tmp_path):
    path = tmp_path / 'journal.sqlite'
    journal = SourceJournal(path)
    journal.append([Event('source', SourceClass.NEWS, 1, 2, text='body')], 3)
    with journal.db:
        journal.db.execute("UPDATE intake SET body='{}', acknowledged=4")
        journal.db.execute('UPDATE intake_counts SET accepted=8')
        journal.db.execute('UPDATE source_service SET pending=9')
    report = audit(path, now=10)
    assert report['status'] == 'FAIL'
    assert report['invalid_sampled_bodies'] == 1
    assert report['invalid_receipt_states'] == 1
    assert not report['checks']['counters_match']
    assert not report['checks']['source_counts_match']
    journal.close()


def test_missing_audit_database_is_not_created(tmp_path):
    path = tmp_path / 'missing.sqlite'
    with pytest.raises(sqlite3.OperationalError):
        audit(path)
    assert not path.exists()
