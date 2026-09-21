"""SolArchive parquet backend.

The interesting failures here are the silent ones -- a wrong schema returning
zero rows, a clock error selecting the wrong day, a missing partition looking
like a quiet stretch of chain. Each gets a test that would pass if the code
merely returned an empty list, and fails unless the code says WHY it is empty.
"""

import asyncio

import pytest

from src.research.history_warehouse import (
    Backend, ExtractionError, ExtractionPlan, HistoryWarehouse, Window)
from src.research.solarchive import (
    ARCHIVE_EPOCH, SlotAnchor, SlotClock, SolArchiveBackend, anchors_from_rows,
    slot_window_for_days)

PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

#: Two anchors a day apart at a clean 0.5 s/slot, so every expectation below
#: is arithmetic a reader can check rather than a fixture they must trust.
DAY_ZERO = 1_700_000_000.0
ANCHORS = [SlotAnchor(slot=1_000_000, unix_time=DAY_ZERO),
           SlotAnchor(slot=1_172_800, unix_time=DAY_ZERO + 86_400.0)]


class FakeReader:
    def __init__(self, days, columns=None, rows=None, sizes=None):
        self.days = list(days)
        self._columns = list(columns or [
            "block_slot", "block_timestamp", "signature", "signer",
            "accounts", "instructions"])
        self._rows = list(rows or [])
        self._sizes = dict(sizes or {})
        self.scans = []

    def list_partitions(self, table):
        return list(self.days)

    def columns(self, table, day):
        return list(self._columns)

    def size_bytes(self, table, day):
        return int(self._sizes.get(day, 1_000))

    def scan(self, table, days, *, start_slot, end_slot, programs):
        self.scans.append({"days": list(days), "start": start_slot,
                           "end": end_slot, "programs": list(programs)})
        return [row for row in self._rows
                if start_slot <= int(row["block_slot"]) <= end_slot]


def _days_around(count=4):
    from datetime import datetime, timedelta, timezone
    first = datetime.fromtimestamp(DAY_ZERO, tz=timezone.utc).date()
    return [(first + timedelta(days=offset - 1)).isoformat()
            for offset in range(count)]


# --- the clock -----------------------------------------------------------

def test_clock_refuses_to_answer_with_one_anchor():
    clock = SlotClock([ANCHORS[0]])
    assert not clock.calibrated
    with pytest.raises(ExtractionError) as caught:
        clock.time_of(1_000_000)
    assert "uncalibrated" in str(caught.value)


def test_clock_refuses_when_every_anchor_names_the_same_slot():
    clock = SlotClock([SlotAnchor(slot=5, unix_time=1.0),
                       SlotAnchor(slot=5, unix_time=2.0)])
    assert not clock.calibrated


def test_clock_fits_the_observed_rate_not_a_nominal_one():
    clock = SlotClock(ANCHORS)
    assert clock.calibrated
    assert clock.seconds_per_slot == pytest.approx(0.5, rel=1e-9)
    # Emphatically NOT Solana's nominal 400ms: the point of fitting is that
    # the archive is read against the rate the chain actually ran at.
    assert abs(clock.seconds_per_slot - 0.4) > 0.05


def test_clock_residual_widens_the_selection_it_cannot_justify():
    noisy = list(ANCHORS) + [SlotAnchor(slot=1_086_400,
                                        unix_time=DAY_ZERO + 43_200 + 200_000)]
    clock = SlotClock(noisy)
    assert clock.residual_s > 86_400
    window = Window(start_slot=1_050_000, end_slot=1_060_000)
    # With no fixed padding at all, the extra days come from the fit's own
    # admitted error -- a clock that misses its anchors reads more of the
    # archive, rather than confidently reading the wrong day.
    assert len(clock.dates_for(window, pad_days=0)) > len(
        SlotClock(ANCHORS).dates_for(window, pad_days=0))


def test_anchors_skip_rows_that_cannot_supply_both_halves():
    anchors = anchors_from_rows([
        {"block_slot": 10, "block_timestamp": 100.0},
        {"block_slot": 20},                      # no time
        {"block_timestamp": 200.0},              # no slot
        {"block_slot": 0, "block_timestamp": 5},  # nonsense slot
        "not a row",
    ])
    assert [anchor.slot for anchor in anchors] == [10]


def test_slot_window_for_days_round_trips():
    clock = SlotClock(ANCHORS)
    days = _days_around()
    window = slot_window_for_days(days, clock)
    assert window.start_slot < window.end_slot
    covered = clock.dates_for(window, pad_days=0)
    assert set(days).issubset(set(covered))


# --- verification --------------------------------------------------------

def test_verify_fails_loudly_on_a_renamed_column():
    reader = FakeReader(_days_around(), columns=["slot", "ts", "sig"])
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS))
    result = backend.verify()
    assert not result["verified"]
    joined = " ".join(result["problems"])
    assert "missing required columns" in joined
    assert "block_slot" in joined


def test_verify_fails_on_an_empty_listing_rather_than_reporting_a_quiet_chain():
    backend = SolArchiveBackend(FakeReader([]), SlotClock(ANCHORS))
    result = backend.verify()
    assert not result["verified"]
    assert any("listing is empty" in problem for problem in result["problems"])


def test_verify_fails_when_the_clock_is_uncalibrated():
    reader = FakeReader(_days_around())
    backend = SolArchiveBackend(reader, SlotClock([ANCHORS[0]]))
    result = backend.verify()
    assert not result["verified"]
    assert any("uncalibrated" in problem for problem in result["problems"])


def test_verify_passes_on_the_bigquery_shape_and_reports_the_rate():
    reader = FakeReader(_days_around())
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS))
    result = backend.verify()
    assert result["verified"], result["problems"]
    assert result["tables"]["seconds_per_slot"] == pytest.approx(0.5)
    assert result["tables"]["partitions"] == 4


def test_extraction_before_verification_is_refused():
    reader = FakeReader(_days_around())
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS))
    with pytest.raises(ExtractionError) as caught:
        backend.launches(Window(1_000_000, 1_010_000), PUMP)
    assert "before verify()" in str(caught.value)


# --- selection and extraction -------------------------------------------

def test_selection_only_reads_days_the_window_can_touch():
    days = _days_around(10)
    reader = FakeReader(days)
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]
    window = Window(start_slot=1_000_000, end_slot=1_010_000)  # ~83 minutes
    selection = backend.select(window)
    assert len(selection.present) <= 2
    assert len(selection.present) < len(days)


def test_a_missing_partition_is_reported_for_repair_not_swallowed():
    days = _days_around(10)
    absent = days[2]
    reader = FakeReader([day for day in days if day != absent])
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]
    window = slot_window_for_days([absent], SlotClock(ANCHORS))
    rows = backend.launches(window, PUMP)
    assert rows == []
    assert absent in backend.repair_windows()
    assert backend.coverage()["days_missing"] >= 1


def test_days_before_the_archive_epoch_are_named_not_counted_as_missing():
    reader = FakeReader(_days_around())
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]
    old = SlotClock([SlotAnchor(slot=1_000, unix_time=1_500_000_000.0),
                     SlotAnchor(slot=173_800, unix_time=1_500_086_400.0)])
    backend.clock = old
    selection = backend.select(Window(1_000, 2_000))
    assert selection.before_epoch
    assert selection.requested[0] < ARCHIVE_EPOCH
    assert not selection.missing


def test_scan_receives_both_predicates_so_a_day_is_not_read_whole():
    days = _days_around(6)
    rows = [{"block_slot": 1_000_500, "block_timestamp": DAY_ZERO + 250,
             "signature": "sig-a", "signer": "dep", "accounts": [PUMP],
             "instructions": []},
            {"block_slot": 1_900_000, "block_timestamp": DAY_ZERO + 450_000,
             "signature": "sig-b", "signer": "dep", "accounts": [PUMP],
             "instructions": []}]
    reader = FakeReader(days, rows=rows)
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]
    got = backend.launches(Window(1_000_000, 1_001_000), PUMP)
    assert [row["signature"] for row in got] == ["sig-a"]
    scan = reader.scans[-1]
    assert scan["start"] == 1_000_000 and scan["end"] == 1_001_000
    assert scan["programs"] == [PUMP]


def test_the_scan_budget_bites_before_the_bytes_are_read():
    days = _days_around(6)
    reader = FakeReader(days, sizes={day: 10 ** 9 for day in days})
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS),
                                scan_budget_bytes=1_000)
    assert backend.verify()["verified"]
    with pytest.raises(ExtractionError) as caught:
        backend.launches(Window(1_000_000, 1_010_000), PUMP)
    assert "budget" in str(caught.value)
    assert reader.scans == []


def test_coverage_reports_ratio_rather_than_row_count():
    days = _days_around(6)
    reader = FakeReader(days[:2])
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]
    for day in days:
        backend.launches(slot_window_for_days([day], SlotClock(ANCHORS)), PUMP)
    coverage = backend.coverage()
    assert 0.0 < coverage["coverage_ratio"] < 1.0
    assert coverage["days_missing"] > 0


# --- the warehouse runner ------------------------------------------------

class AsyncBackend:
    """Shaped like RpcBackend: `launches` is a coroutine function."""

    kind = Backend.RPC
    bytes_scanned = 0

    def __init__(self):
        self.calls = 0

    def verify(self):
        return {"verified": True, "problems": [], "tables": {}}

    async def launches(self, window, program):
        self.calls += 1
        return [{"block_slot": window.start_slot, "mint": "MINT", "kind":
                 "create", "signer": "dep", "block_timestamp": DAY_ZERO}]


def test_run_drives_an_async_backend_instead_of_scoring_it_a_failure():
    backend = AsyncBackend()
    warehouse = HistoryWarehouse(backend)
    plan = ExtractionPlan(windows=[Window(1, 2)], backend=Backend.RPC,
                          programs=(PUMP,))
    launches, report = warehouse.run(plan)
    assert backend.calls == 1
    assert report.windows_failed == 0
    assert report.windows_done == 1
    assert [launch.token for launch in launches] == ["MINT"]


def test_run_inside_a_loop_says_to_use_arun_rather_than_failing_opaquely():
    async def scenario():
        warehouse = HistoryWarehouse(AsyncBackend())
        plan = ExtractionPlan(windows=[Window(1, 2)], backend=Backend.RPC,
                              programs=(PUMP,))
        _, report = warehouse.run(plan)
        return report

    report = asyncio.run(scenario())
    assert report.windows_failed == 1
    assert "arun()" in report.failures[0]["error"]


def test_arun_extracts_from_inside_a_running_loop():
    async def scenario():
        warehouse = HistoryWarehouse(AsyncBackend())
        plan = ExtractionPlan(windows=[Window(1, 2)], backend=Backend.RPC,
                              programs=(PUMP,))
        return await warehouse.arun(plan)

    launches, report = asyncio.run(scenario())
    assert report.windows_done == 1 and report.windows_failed == 0
    assert [launch.token for launch in launches] == ["MINT"]


def test_arun_still_drives_a_synchronous_backend():
    days = _days_around(4)
    rows = [{"block_slot": 1_000_500, "block_timestamp": DAY_ZERO + 250,
             "signature": "sig-a", "signer": "dep", "mint": "MINT",
             "kind": "create", "accounts": [PUMP], "instructions": []}]
    reader = FakeReader(days, rows=rows)
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]

    async def scenario():
        warehouse = HistoryWarehouse(backend)
        plan = ExtractionPlan(windows=[Window(1_000_000, 1_001_000)],
                              backend=Backend.SOLARCHIVE, programs=(PUMP,))
        return await warehouse.arun(plan)

    launches, report = asyncio.run(scenario())
    assert report.windows_done == 1
    assert [launch.token for launch in launches] == ["MINT"]


# --- the partition cache -------------------------------------------------

def test_a_day_read_twice_is_fetched_once(tmp_path):
    days = _days_around(6)
    rows = [{"block_slot": 1_000_500, "block_timestamp": DAY_ZERO + 250,
             "signature": "sig-a", "signer": "dep", "accounts": [PUMP],
             "instructions": []}]
    reader = FakeReader(days, rows=rows)
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0,
                                cache_dir=tmp_path / "cache")
    assert backend.verify()["verified"]
    window = Window(1_000_000, 1_001_000)
    first = backend.launches(window, PUMP)
    scans_after_first = len(reader.scans)
    second = backend.launches(window, PUMP)
    assert first == second
    assert len(reader.scans) == scans_after_first, "the archive was re-read"
    assert backend.cache_hits >= 1


def test_a_cached_day_costs_no_scan_budget(tmp_path):
    days = _days_around(6)
    reader = FakeReader(days, sizes={day: 10 ** 6 for day in days})
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0,
                                cache_dir=tmp_path / "cache")
    assert backend.verify()["verified"]
    window = Window(1_000_000, 1_001_000)
    backend.launches(window, PUMP)
    spent = backend.bytes_scanned
    backend.launches(window, PUMP)
    assert backend.bytes_scanned == spent, (
        "billing a cached day again exhausts the budget on reads that never "
        "happened")


def test_the_window_predicate_still_applies_to_cached_rows(tmp_path):
    days = _days_around(6)
    rows = [{"block_slot": 1_000_500, "block_timestamp": DAY_ZERO + 250,
             "signature": "in", "signer": "dep", "accounts": [PUMP],
             "instructions": []},
            {"block_slot": 1_050_000, "block_timestamp": DAY_ZERO + 25_000,
             "signature": "out", "signer": "dep", "accounts": [PUMP],
             "instructions": []}]
    reader = FakeReader(days, rows=rows)
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0,
                                cache_dir=tmp_path / "cache")
    assert backend.verify()["verified"]
    wide = Window(1_000_000, 1_100_000)
    backend.launches(wide, PUMP)
    narrow = backend.launches(Window(1_000_000, 1_001_000), PUMP)
    assert [row["signature"] for row in narrow] == ["in"]


def test_a_corrupt_cache_file_is_re_read_not_reported_as_an_empty_day(tmp_path):
    days = _days_around(6)
    rows = [{"block_slot": 1_000_500, "block_timestamp": DAY_ZERO + 250,
             "signature": "sig-a", "signer": "dep", "accounts": [PUMP],
             "instructions": []}]
    reader = FakeReader(days, rows=rows)
    cache = tmp_path / "cache"
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0,
                                cache_dir=cache)
    assert backend.verify()["verified"]
    window = Window(1_000_000, 1_001_000)
    backend.launches(window, PUMP)
    for path in cache.glob("*.json"):
        path.write_text("{truncated")
    again = backend.launches(window, PUMP)
    assert [row["signature"] for row in again] == ["sig-a"], (
        "a half-written cache file is not evidence of a quiet day")


def test_without_a_cache_dir_nothing_changes(tmp_path):
    days = _days_around(6)
    reader = FakeReader(days)
    backend = SolArchiveBackend(reader, SlotClock(ANCHORS), pad_days=0)
    assert backend.verify()["verified"]
    window = Window(1_000_000, 1_001_000)
    backend.launches(window, PUMP)
    backend.launches(window, PUMP)
    assert len(reader.scans) == 2
    assert backend.cache_hits == 0
