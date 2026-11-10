"""The desk's durable writers refuse to grow into the volume's reserve."""
from collections import namedtuple

import pytest

from src.collectors.event_source import Event, SourceClass
from src.collectors.source_journal import SourceJournal
from src.research.trade_evidence import TradeEvidenceLedger
from src.runtime import disk_floor
from src.runtime.disk_floor import DiskFloor

Usage = namedtuple("Usage", "total used free")


def _floor(tmp_path, free, floor=100, **kw):
    state = {"free": free}
    guard = DiskFloor(tmp_path, floor_bytes=floor, cache_seconds=0,
                      usage=lambda _: Usage(1000, 0, state["free"]), **kw)
    return guard, state


def test_blocks_only_below_floor_and_counts(tmp_path):
    guard, state = _floor(tmp_path, free=100)
    assert not guard.blocked()
    state["free"] = 99
    assert guard.blocked() and guard.blocked()
    assert guard.report() == {"floor_bytes": 100, "free_bytes": 99, "below_floor": True,
                              "refusals": 2, "read_failures": 0}


def test_zero_floor_disables_and_unreadable_volume_does_not_block(tmp_path):
    guard, _ = _floor(tmp_path, free=0, floor=0)
    assert not guard.blocked()
    def broken(_): raise OSError("gone")
    unknown = DiskFloor(tmp_path, floor_bytes=100, cache_seconds=0, usage=broken)
    assert not unknown.blocked() and unknown.read_failures == 1


def test_reading_is_cached(tmp_path):
    calls, now = [], [0.0]
    guard = DiskFloor(tmp_path, floor_bytes=100, cache_seconds=5,
                      usage=lambda _: calls.append(1) or Usage(1000, 0, 500), clock=lambda: now[0])
    for _ in range(50): guard.blocked()
    assert len(calls) == 1
    now[0] = 6.0; guard.blocked()
    assert len(calls) == 2


def test_env_floor(monkeypatch):
    monkeypatch.setenv("MEMECOIN_DISK_FLOOR_BYTES", "12345")
    assert disk_floor.configured_floor() == 12345
    monkeypatch.setenv("MEMECOIN_DISK_FLOOR_BYTES", "junk")
    assert disk_floor.configured_floor() == disk_floor.DEFAULT_FLOOR_BYTES
    monkeypatch.delenv("MEMECOIN_DISK_FLOOR_BYTES")
    assert disk_floor.configured_floor() == 20 * 1024 ** 3
    monkeypatch.setenv("MEMECOIN_DISK_FLOOR_BYTES", "0")
    assert disk_floor.configured_floor() == 0


def test_ledger_refuses_below_floor_and_resumes_with_chain_intact(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "evidence.jsonl")
    ledger.disk_floor, state = _floor(tmp_path, free=500)
    assert ledger.record("a", {"i": 1})["sequence"] == 1
    size = ledger.path.stat().st_size
    state["free"] = 10
    assert ledger.record("a", {"i": 2}) is None
    assert ledger.path.stat().st_size == size and ledger.write_failures == 1
    assert ledger.report()["disk_floor"]["below_floor"] is True
    state["free"] = 500
    assert ledger.record("a", {"i": 3})["sequence"] == 2
    assert TradeEvidenceLedger.verify(ledger.path) == (True, "OK", 2)


def _event(n):
    return Event(source_id="s", source_class=list(SourceClass)[0],
                 source_at=1.0, observed_at=2.0, text=f"t{n}")


def test_journal_backpressures_below_floor_but_still_drains(tmp_path):
    journal = SourceJournal(tmp_path / "intake.sqlite")
    journal.disk_floor, state = _floor(tmp_path, free=500)
    assert journal.append([_event(1)], 10.0) == 1
    state["free"] = 10
    with pytest.raises(OSError):
        journal.append([_event(2)], 11.0)
    report = journal.report(12.0)
    assert report["accepted_observations"] == 1 and report["disk_floor"]["refusals"] == 1
    leased = journal.lease(12.0)            # draining is never blocked
    assert leased is not None
    event, receipt, generation = leased
    journal.acknowledge(event, receipt, generation, "work-1", "ref", 13.0)
    assert journal.report(14.0)["pending_events"] == 0
    journal.close()
