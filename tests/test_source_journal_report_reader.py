from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

import pytest

from src.collectors.source_journal import SourceJournal
from tests.test_source_journal import event


def test_slow_health_scan_does_not_serialize_ack_or_intake(tmp_path, monkeypatch):
    journal = SourceJournal(tmp_path / "intake.sqlite")
    journal.append([event(number=i) for i in range(100)], 10.)
    item, receipt, generation = journal.lease(20.)
    entered, release = threading.Event(), threading.Event()
    connect = sqlite3.connect
    def reader(*args, **kwargs):
        db = connect(*args, **kwargs)
        def trace(sql):
            if sql.startswith("SELECT count(*)"):
                entered.set()
                assert release.wait(10)
        db.set_trace_callback(trace)
        return db
    monkeypatch.setattr("src.collectors.source_journal.sqlite3.connect", reader)
    with ThreadPoolExecutor(max_workers=2) as pool:
        report = pool.submit(journal.report, 30.)
        try:
            assert entered.wait(5)
            def write():
                journal.acknowledge(item, receipt, generation, "work-test", "evidence:ref", 30.)
                journal.append([event(number=101)], 31.)
            pool.submit(write).result(timeout=5)
        finally:
            release.set()
        old = report.result(timeout=5)
    # All database fields retain the original snapshot despite concurrent commits.
    assert old["accepted_observations"] == old["pending_events"] == 100
    assert old["canonical_acknowledged"] == 0
    assert old["pending_by_source"] == {"feed": 100}
    assert old["due_events"] == 99
    new = journal.report(40.)
    assert new["accepted_observations"] == 101
    assert new["pending_events"] == 100
    assert new["canonical_acknowledged"] == 1
    journal.close()
    with pytest.raises(sqlite3.ProgrammingError):
        journal.report(40.)


def test_report_does_not_wait_for_or_observe_uncommitted_writer(tmp_path):
    journal = SourceJournal(tmp_path / "intake.sqlite")
    journal.append([event()], 10.)
    entered, release = threading.Event(), threading.Event()
    def trace(sql):
        if sql == "COMMIT":
            entered.set()
            assert release.wait(10)
    journal.db.set_trace_callback(trace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        write = pool.submit(journal.append, [event(number=1)], 11.)
        try:
            assert entered.wait(5)
            report = pool.submit(journal.report, 20.).result(timeout=5)
            assert report["accepted_observations"] == 1
            assert report["pending_events"] == report["due_events"] == 1
        finally:
            release.set()
        write.result(timeout=5)
    assert journal.report(20.)["accepted_observations"] == 2
    journal.close()
