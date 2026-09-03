"""SolArchive: the same history as BigQuery, without the invoice.

The warehouse backend can enumerate a launch universe, and it bills for every
byte it scans. The RPC backend bills nothing and cannot enumerate anything.
Between them sits a third shape that neither covers: a public columnar archive
of the decoded chain, sitting on object storage, readable with predicate
pushdown from a local process.

SolArchive is that archive. It is derived from the same
``bigquery-public-data.crypto_solana_mainnet_us`` tables the warehouse backend
targets -- which is why this module reuses ``REQUIRED_COLUMNS`` unchanged
rather than declaring a second, hopeful schema. It publishes daily partitions
in Parquet with SHA256 checksums, from October 2020 forward.

**What this buys.** Reading a day of Pump activity out of a partitioned
parquet file costs one HTTP range read per row group the predicate survives.
Two years of deployer history becomes a local extraction against files instead
of several hundred thousand ``getSignaturesForAddress`` pages. RPC stops being
the corpus and becomes what it should always have been: the repair tool for
days the archive does not cover.

**Three things this module refuses to do.**

*It does not guess the schema.* ``verify()`` asks the reader for the columns of
a real partition and checks them against ``REQUIRED_COLUMNS``. An upstream
rename fails naming the columns it wanted, instead of returning zero rows that
look like a quiet history -- the same discipline the warehouse backend already
enforces, for the same reason.

*It does not guess the calendar.* The extraction plan is expressed in slots;
the archive is partitioned by date. Converting between them requires knowing
how fast the chain actually ran, which varies. So ``SlotClock`` must be
calibrated from at least two OBSERVED ``(slot, unix_time)`` anchors -- from the
chain, from a block the desk already has -- and an uncalibrated clock raises
rather than assuming 400ms. The conversion then pads the selected date range by
a stated margin, so residual clock error costs bytes rather than rows.

*It does not report coverage it did not read.* Days the archive has no
partition for are returned as ``missing_days`` on the report, which is what the
RPC repair pass consumes. A window whose partitions were partly absent is not
silently thinner than a window that was complete.

**On the reader.** No parquet engine is constructed here and no credentials
live here. The caller injects a reader -- DuckDB, Polars, pyarrow.dataset, or a
fake in a test -- and this module speaks to it through four methods. That is
the only way to test an extractor without either a live account or a fabricated
result, and it is why the absence of DuckDB on a box is a DATA_BLOCKED rather
than an import error at startup.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.research.history_warehouse import (
    REQUIRED_COLUMNS, Backend, ExtractionError, Window)

logger = logging.getLogger(__name__)

#: The published archive. Overridable in full -- a mirror, a local sync, an
#: R2 bucket the desk populated itself -- because a single hostname is a
#: single point of failure and this one is not ours.
DEFAULT_SOLARCHIVE_REPO = "solarchive/solarchive"
DEFAULT_SOLARCHIVE_BASE = "hf://datasets/solarchive/solarchive"

#: Earliest partition the archive claims. Extraction below this is not a
#: failure to report; it is a date range the archive never covered, and the
#: caller is told so rather than shown an empty result.
ARCHIVE_EPOCH = "2020-10-01"

#: Partition directory names are dates. Anything else in the listing is not a
#: partition, and is ignored rather than parsed into a wrong day.
_PARTITION_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

#: How far either side of the computed date range to widen the partition
#: selection. The slot clock is calibrated from observed anchors and is still
#: an approximation; one day of slack is cheap (a partition that contributes
#: no rows costs one predicate-rejected scan) and a missed day is expensive
#: (a hole in the corpus that nothing downstream can see).
DEFAULT_DATE_PAD_DAYS = 1

#: Refuse to plan a parquet extraction wider than this. The archive is free to
#: read and not free to read carelessly: a full-history scan with no slot
#: predicate is hours of egress for a corpus the desk did not ask for.
DEFAULT_PARQUET_BUDGET_BYTES = 250 * 1024 ** 3


def _day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class SlotAnchor:
    """One OBSERVED (slot, time) pair. Not a constant -- a measurement."""

    slot: int
    unix_time: float
    source: str = ""


class SlotClock:
    """Slots to wall-clock, fitted to anchors the desk actually observed.

    Solana's slot time is nominally 400ms and empirically is not: it drifts
    with skipped slots and network conditions, and the drift compounds over
    the years of history this module exists to read. A hardcoded 400ms over
    two years is off by weeks, which would select the wrong partitions and
    produce a corpus with a silent hole in it.

    So the clock is a least-squares fit through observed anchors, and it
    refuses to answer at all with fewer than two. ``residual_s`` reports how
    badly the fit misses its own anchors, which is the honest input to how
    much date padding a caller should ask for.
    """

    def __init__(self, anchors: Sequence[SlotAnchor]):
        self.anchors: Tuple[SlotAnchor, ...] = tuple(
            sorted(anchors, key=lambda anchor: anchor.slot))
        self._slope: Optional[float] = None
        self._intercept: Optional[float] = None
        self.residual_s: float = 0.0
        if len(self.anchors) >= 2:
            self._fit()

    def _fit(self) -> None:
        slots = [float(anchor.slot) for anchor in self.anchors]
        times = [float(anchor.unix_time) for anchor in self.anchors]
        count = float(len(slots))
        mean_slot = sum(slots) / count
        mean_time = sum(times) / count
        variance = sum((slot - mean_slot) ** 2 for slot in slots)
        if variance <= 0:
            # Every anchor names the same slot: no rate is observable.
            return
        covariance = sum((slot - mean_slot) * (time_ - mean_time)
                         for slot, time_ in zip(slots, times))
        self._slope = covariance / variance
        self._intercept = mean_time - self._slope * mean_slot
        self.residual_s = max(
            abs(time_ - (self._intercept + self._slope * slot))
            for slot, time_ in zip(slots, times))

    @property
    def calibrated(self) -> bool:
        return self._slope is not None and self._slope > 0

    @property
    def seconds_per_slot(self) -> Optional[float]:
        return self._slope

    def time_of(self, slot: int) -> float:
        if not self.calibrated:
            raise ExtractionError(
                "slot clock uncalibrated: needs at least two observed "
                f"(slot, time) anchors with distinct slots, has "
                f"{len(self.anchors)}")
        assert self._slope is not None and self._intercept is not None
        return self._intercept + self._slope * float(slot)

    def dates_for(self, window: Window, *,
                  pad_days: int = DEFAULT_DATE_PAD_DAYS) -> List[str]:
        """The UTC dates a slot window could plausibly touch, padded.

        Padding is applied in whole days on both sides, and additionally
        widened by the clock's own worst residual, so a fit that is visibly
        poor selects more partitions rather than quietly dropping rows.
        """
        slack = float(max(0, pad_days)) * 86_400.0 + self.residual_s
        start = self.time_of(window.start_slot) - slack
        end = self.time_of(window.end_slot) + slack
        if end < start:
            start, end = end, start
        first = datetime.fromtimestamp(start, tz=timezone.utc).date()
        last = datetime.fromtimestamp(end, tz=timezone.utc).date()
        days: List[str] = []
        cursor = first
        while cursor <= last:
            days.append(cursor.isoformat())
            cursor = cursor + timedelta(days=1)
        return days


@dataclass
class PartitionSelection:
    """Which days a window resolved to, and which of them the archive has."""

    window_key: str
    requested: List[str] = field(default_factory=list)
    present: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    before_epoch: List[str] = field(default_factory=list)
    estimated_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"window": self.window_key,
                "requested_days": len(self.requested),
                "present_days": len(self.present),
                "missing_days": list(self.missing),
                "before_epoch_days": len(self.before_epoch),
                "estimated_bytes": self.estimated_bytes}


class ParquetReader:
    """The interface a concrete engine must satisfy.

    Four methods, because that is all an extractor needs and every additional
    one is a thing a test fake has to imitate correctly. Implementations exist
    for DuckDB and for a directory of local files; a test passes a fake.
    """

    def list_partitions(self, table: str) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def columns(self, table: str, day: str) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def size_bytes(self, table: str, day: str) -> int:  # pragma: no cover
        raise NotImplementedError

    def scan(self, table: str, days: Sequence[str], *, start_slot: int,
             end_slot: int, programs: Sequence[str]
             ) -> List[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


class DuckDBParquetReader:
    """A DuckDB-backed reader over the published parquet layout.

    Constructed only when asked for, so a box without DuckDB reports a
    DATA_BLOCKED with an installable remedy rather than failing to import the
    research package at startup.
    """

    def __init__(self, base: str = DEFAULT_SOLARCHIVE_BASE, *,
                 connection: Any = None, listing: Any = None):
        if connection is None:
            try:
                import duckdb  # noqa: F401  (imported for its side effect)
            except ImportError as exc:  # pragma: no cover - environment
                raise ExtractionError(
                    "duckdb is not installed; the SolArchive backend needs a "
                    "parquet engine (pip install duckdb) or an injected "
                    f"reader ({exc})") from exc
            import duckdb
            connection = duckdb.connect()
            connection.execute("INSTALL httpfs; LOAD httpfs;")
        self.connection = connection
        self.base = base.rstrip("/")
        #: How partition days are discovered. Injected so a mirror with a
        #: different listing mechanism -- or a test -- does not need DuckDB to
        #: enumerate an HTTP directory, which it cannot do.
        self.listing = listing

    def _path(self, table: str, day: str) -> str:
        return f"{self.base}/{table}/{day}/*.parquet"

    def list_partitions(self, table: str) -> List[str]:
        if self.listing is None:
            raise ExtractionError(
                "no partition listing available: object stores are not "
                "enumerable over plain HTTP, so the SolArchive reader needs "
                "an injected `listing` (an index file, an S3/R2 client, or a "
                "local mirror path)")
        days = [match.group(1)
                for entry in self.listing(table)
                for match in [_PARTITION_RE.search(str(entry))] if match]
        return sorted(set(days))

    def columns(self, table: str, day: str) -> List[str]:
        rows = self.connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{self._path(table, day)}') "
            "LIMIT 0").fetchall()
        return [str(row[0]) for row in rows]

    def size_bytes(self, table: str, day: str) -> int:
        # Parquet metadata carries compressed sizes; asking for them is far
        # cheaper than reading, which is the point of estimating at all.
        try:
            rows = self.connection.execute(
                "SELECT sum(total_compressed_size) FROM parquet_metadata("
                f"'{self._path(table, day)}')").fetchall()
        except Exception as exc:
            logger.debug("size probe failed for %s/%s: %s", table, day, exc)
            return 0
        if not rows or rows[0][0] is None:
            return 0
        return int(rows[0][0])

    def scan(self, table: str, days: Sequence[str], *, start_slot: int,
             end_slot: int, programs: Sequence[str]) -> List[Dict[str, Any]]:
        if not days:
            return []
        paths = ", ".join(f"'{self._path(table, day)}'" for day in days)
        # Both predicates push down: block_slot into row-group statistics,
        # the program list into the accounts array. Without the slot predicate
        # this reads whole days for a window that may span minutes.
        query = (
            "SELECT block_slot, block_timestamp, signature, signer, "
            "accounts, instructions "
            f"FROM read_parquet([{paths}], union_by_name = true) "
            "WHERE block_slot BETWEEN ? AND ? "
            "AND len(list_intersect(accounts, ?)) > 0 "
            "ORDER BY block_slot")
        cursor = self.connection.execute(
            query, [int(start_slot), int(end_slot), list(programs)])
        names = [str(item[0]) for item in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


class SolArchiveBackend:
    """Read the decoded chain out of daily parquet partitions.

    Shares the ``WarehouseBackend`` contract -- ``verify``, ``estimate``,
    ``launches``, ``bytes_scanned`` -- so ``HistoryWarehouse`` runs a parquet
    plan with no knowledge that it is one.
    """

    kind = Backend.SOLARCHIVE

    def __init__(self, reader: Any, clock: SlotClock, *,
                 table: str = "transactions",
                 scan_budget_bytes: int = DEFAULT_PARQUET_BUDGET_BYTES,
                 pad_days: int = DEFAULT_DATE_PAD_DAYS,
                 epoch: str = ARCHIVE_EPOCH,
                 repo: str = DEFAULT_SOLARCHIVE_REPO):
        self.reader = reader
        self.clock = clock
        self.table = table
        self.scan_budget_bytes = int(scan_budget_bytes)
        self.pad_days = int(pad_days)
        self.epoch = epoch
        self.repo = repo
        self.verified = False
        self.bytes_scanned = 0
        self.selections: List[PartitionSelection] = []
        self._partitions: Optional[set] = None

    # -- verification ----------------------------------------------------

    def partitions(self, *, refresh: bool = False) -> set:
        if self._partitions is None or refresh:
            try:
                self._partitions = set(self.reader.list_partitions(self.table))
            except ExtractionError:
                raise
            except Exception as exc:
                raise ExtractionError(
                    f"could not list SolArchive partitions: "
                    f"{type(exc).__name__}: {exc}") from exc
        return self._partitions

    def verify(self) -> Dict[str, Any]:
        """Check the archive is reachable and shaped the way we read it.

        Deliberately the same three questions the warehouse backend asks --
        does it exist, does it have partitions, does it have the columns --
        because a parquet archive fails in exactly the same invisible way a
        renamed table does.
        """
        problems: List[str] = []
        detail: Dict[str, Any] = {"repo": self.repo, "table": self.table}

        if not self.clock.calibrated:
            problems.append(
                "slot clock uncalibrated: supply at least two observed "
                "(slot, unix_time) anchors before planning a parquet "
                "extraction")
        else:
            detail["seconds_per_slot"] = round(
                float(self.clock.seconds_per_slot or 0.0), 6)
            detail["clock_residual_s"] = round(self.clock.residual_s, 3)

        days: List[str] = []
        try:
            days = sorted(self.partitions())
        except ExtractionError as exc:
            problems.append(str(exc))

        if not problems and not days:
            problems.append(
                "archive listing is empty: either the repo path is wrong or "
                "the listing mechanism returned nothing, and both look "
                "identical to a chain with no history")

        if days:
            detail["first_partition"] = days[0]
            detail["last_partition"] = days[-1]
            detail["partitions"] = len(days)
            probe = days[-1]
            try:
                columns = set(self.reader.columns(self.table, probe))
            except Exception as exc:
                problems.append(
                    f"could not read the schema of partition {probe}: "
                    f"{type(exc).__name__}: {exc}")
            else:
                detail["columns"] = sorted(columns)
                missing = [name for name in REQUIRED_COLUMNS["launches"]
                           if name not in columns]
                if missing:
                    problems.append(
                        f"partition {probe} is missing required columns "
                        f"{missing}; expected the BigQuery public-dataset "
                        f"shape {list(REQUIRED_COLUMNS['launches'])}")

        self.verified = not problems
        return {"verified": self.verified, "problems": problems,
                "tables": detail}

    # -- planning --------------------------------------------------------

    def select(self, window: Window) -> PartitionSelection:
        """Resolve a slot window to the partitions that can answer it."""
        selection = PartitionSelection(window_key=window.key())
        selection.requested = self.clock.dates_for(
            window, pad_days=self.pad_days)
        available = self.partitions()
        epoch = self.epoch
        for day in selection.requested:
            if day < epoch:
                selection.before_epoch.append(day)
            elif day in available:
                selection.present.append(day)
            else:
                selection.missing.append(day)
        selection.estimated_bytes = sum(
            self._size(day) for day in selection.present)
        return selection

    def _size(self, day: str) -> int:
        try:
            return int(self.reader.size_bytes(self.table, day) or 0)
        except Exception as exc:
            logger.debug("partition size unavailable for %s: %s", day, exc)
            return 0

    def estimate(self, window: Window, program: str) -> int:
        del program  # partition size is per day, not per program
        return self.select(window).estimated_bytes

    # -- extraction ------------------------------------------------------

    def launches(self, window: Window, program: str) -> List[Dict[str, Any]]:
        if not self.verified:
            raise ExtractionError(
                "SolArchive backend used before verify() passed; a parquet "
                "archive with the wrong schema returns zero rows, which is "
                "indistinguishable from a quiet slot range")
        selection = self.select(window)
        self.selections.append(selection)
        if selection.estimated_bytes and (
                self.bytes_scanned + selection.estimated_bytes
                > self.scan_budget_bytes):
            raise ExtractionError(
                f"parquet scan budget exhausted: window {window.key()} would "
                f"read {selection.estimated_bytes} bytes on top of "
                f"{self.bytes_scanned}, over the {self.scan_budget_bytes} "
                "budget")
        if not selection.present:
            return []
        rows = self.reader.scan(
            self.table, selection.present,
            start_slot=window.start_slot, end_slot=window.end_slot,
            programs=[program])
        self.bytes_scanned += selection.estimated_bytes
        return [row for row in rows if isinstance(row, dict)]

    # -- repair ----------------------------------------------------------

    def repair_windows(self) -> List[str]:
        """Days no partition covered, for the RPC pass to fill.

        This is the whole reason RPC still exists in the design. Without it,
        an archive that stops publishing on a Tuesday produces a corpus that
        is silently missing Wednesday, and every model trained on it inherits
        the hole without a diagnostic anywhere.
        """
        missing: List[str] = []
        for selection in self.selections:
            missing.extend(selection.missing)
        return sorted(set(missing))

    def coverage(self) -> Dict[str, Any]:
        requested = sum(len(item.requested) for item in self.selections)
        present = sum(len(item.present) for item in self.selections)
        return {
            "windows": len(self.selections),
            "days_requested": requested,
            "days_present": present,
            "days_missing": len(self.repair_windows()),
            "coverage_ratio": (present / requested) if requested else 0.0,
            "bytes_scanned": self.bytes_scanned,
            "repair_days": self.repair_windows(),
            "selections": [item.to_dict() for item in self.selections],
        }


def slot_window_for_days(days: Sequence[str], clock: SlotClock) -> Window:
    """The inverse: a date range expressed as the slot window covering it.

    Useful when the desk knows what it wants in calendar terms ("the week
    BONK launched") and the extraction plan needs slots.
    """
    if not days:
        raise ExtractionError("no days given")
    if not clock.calibrated:
        raise ExtractionError("slot clock uncalibrated")
    slope = float(clock.seconds_per_slot or 0.0)
    ordered = sorted(days)
    start_time = _day(ordered[0]).timestamp()
    end_time = _day(ordered[-1]).timestamp() + 86_400.0
    anchor = clock.anchors[0]
    start_slot = int(anchor.slot + (start_time - anchor.unix_time) / slope)
    end_slot = int(anchor.slot + (end_time - anchor.unix_time) / slope)
    return Window(start_slot=max(0, start_slot), end_slot=max(0, end_slot))


def anchors_from_rows(rows: Iterable[Dict[str, Any]]) -> List[SlotAnchor]:
    """Build clock anchors out of chain rows the desk already holds.

    Any row carrying both a slot and a block time is an anchor. Rows carrying
    one or neither are skipped rather than defaulted, because an anchor with an
    invented timestamp corrupts every partition selection that follows it.
    """
    anchors: List[SlotAnchor] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        slot = row.get("block_slot", row.get("slot"))
        stamp = row.get("block_timestamp", row.get("blockTime"))
        if slot is None or stamp is None:
            continue
        try:
            slot_value = int(slot)
            time_value = float(stamp)
        except (TypeError, ValueError):
            continue
        if slot_value <= 0 or time_value <= 0:
            continue
        anchors.append(SlotAnchor(slot=slot_value, unix_time=time_value,
                                  source="observed_chain_row"))
    return anchors
