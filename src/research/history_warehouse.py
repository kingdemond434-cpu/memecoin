"""Bulk historical extraction: buying years of the chain instead of waiting.

The forward ledger accrues at the rate the desk observes launches. That is the
right way to learn our own execution, and a needlessly slow way to learn the
market -- because the market's history is already written down. Every Pump
launch, every First25 sequence, every deployer's prior forty tokens, every
funding edge is public and has been for years. What separates us from a desk
with a three-year corpus is not access; it is extraction.

So this is the extraction layer. It emits ``backfill.RawLaunch`` records,
which the existing reconstruction path already knows how to turn into
point-in-time episodes -- stamped as reconstructed, so nothing downstream can
confuse them with observed ones.

Three backends, in descending order of throughput and ascending order of
availability:

  WAREHOUSE   a SQL warehouse holding the decoded chain (BigQuery's public
              Solana dataset, or Dune). Millions of rows per query. This is
              the only route that makes a full launch universe practical.
  INDEXER     a transaction-history API. Good for targeted gaps -- one
              deployer's history, one token's trades -- and far too slow to
              enumerate a universe.
  RPC         getSignaturesForAddress against the program, walked backwards.
              Always available, needs no account, and is slow enough that it
              is a completeness backstop rather than a plan.

**On table names.** The warehouse backends take their table identifiers from
configuration, with the documented public defaults as a starting point, and
VERIFY the schema before the first extraction. A wrong identifier then fails
loudly, naming the columns it expected, instead of returning zero rows that
look like a quiet history. Guessing a schema and silently producing an empty
corpus is the failure mode that wastes a week.

**On cost.** Every backend declares an estimated scan before it runs, and the
plan refuses to exceed a stated budget. A warehouse query over an unpartitioned
multi-terabyte table is a real invoice, and discovering that afterwards is how
free tiers stop being free.

**On completeness.** Extraction is checkpointed by slot range, so an
interrupted run resumes rather than restarting, and coverage is reported as
the slot ranges actually retrieved -- not as a row count, which cannot
distinguish a complete window from a truncated one.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.research.backfill import RawLaunch

logger = logging.getLogger(__name__)

HISTORY_WAREHOUSE_SCHEMA_VERSION = "v1"

PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"

#: Documented public dataset defaults. Overridable, and VERIFIED before use --
#: see the module docstring on why a wrong name must fail loudly.
DEFAULT_BIGQUERY_TABLES: Dict[str, str] = {
    "transactions": "bigquery-public-data.crypto_solana_mainnet_us.Transactions",
    "blocks": "bigquery-public-data.crypto_solana_mainnet_us.Blocks",
    "token_transfers": "bigquery-public-data.crypto_solana_mainnet_us.Token Transfers",
}

#: Columns each extraction needs. Checked against the backend's reported
#: schema so a rename upstream is a clear error rather than an empty corpus.
REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "launches": ("block_slot", "block_timestamp", "signature", "signer",
                 "accounts", "instructions"),
    "trades": ("block_slot", "block_timestamp", "signature", "signer",
               "accounts", "instructions"),
    "transfers": ("block_slot", "block_timestamp", "source", "destination",
                  "value"),
}

#: Refuse a plan estimated above this many bytes scanned. BigQuery's public
#: tier gives a fixed monthly allowance; one careless unpartitioned scan can
#: consume it entirely.
DEFAULT_SCAN_BUDGET_BYTES = 900 * 1024**3  # 900 GiB, inside a 1 TiB allowance


class Backend(Enum):
    WAREHOUSE = "warehouse"
    INDEXER = "indexer"
    #: Public columnar archive of the decoded chain, read with predicate
    #: pushdown off object storage. Free like RPC, enumerable like a
    #: warehouse. See ``src/research/solarchive.py``.
    SOLARCHIVE = "solarchive"
    RPC = "rpc"


class ExtractionError(RuntimeError):
    """A backend could not answer, with the reason attached."""


@dataclass
class Window:
    """A slot range to extract. Checkpointing is by window, not by row."""

    start_slot: int
    end_slot: int

    @property
    def slots(self) -> int:
        return max(0, self.end_slot - self.start_slot)

    def key(self) -> str:
        return f"{self.start_slot}-{self.end_slot}"


@dataclass
class ExtractionPlan:
    """What to pull, from where, and what it is allowed to cost."""

    windows: List[Window]
    backend: Backend
    scan_budget_bytes: int = DEFAULT_SCAN_BUDGET_BYTES
    programs: Tuple[str, ...] = (PUMP_PROGRAM, PUMP_AMM_PROGRAM)

    def total_slots(self) -> int:
        return sum(window.slots for window in self.windows)


# --- queries -------------------------------------------------------------
# Written out rather than generated so a reader can see exactly what is being
# asked for, and so a schema change shows up as a diff rather than as a
# behaviour change buried in a builder.

LAUNCH_QUERY = """
SELECT
  block_slot, block_timestamp, signature, signer, accounts, instructions
FROM `{transactions}`
WHERE block_slot BETWEEN @start_slot AND @end_slot
  AND status = 'Success'
  AND EXISTS (
    SELECT 1 FROM UNNEST(instructions) AS ix
    WHERE ix.program_id = @program
  )
ORDER BY block_slot, signature
"""

TRANSFER_QUERY = """
SELECT
  block_slot, block_timestamp, source, destination, value
FROM `{token_transfers}`
WHERE block_slot BETWEEN @start_slot AND @end_slot
  AND destination IN UNNEST(@addresses)
ORDER BY block_slot
"""


class WarehouseBackend:
    """A SQL warehouse holding the decoded chain.

    The client is injected rather than constructed. This module has no
    credentials, issues no network calls of its own, and can therefore be
    tested against a fake that returns rows -- which is the only way to test
    an extractor without either a live account or a fabricated result.
    """

    kind = Backend.WAREHOUSE

    def __init__(self, client: Any, tables: Optional[Dict[str, str]] = None,
                 *, scan_budget_bytes: int = DEFAULT_SCAN_BUDGET_BYTES):
        self.client = client
        self.tables = dict(tables or DEFAULT_BIGQUERY_TABLES)
        self.scan_budget_bytes = int(scan_budget_bytes)
        self.verified = False
        self.bytes_scanned = 0

    def verify(self) -> Dict[str, Any]:
        """Check the schema before extracting anything.

        A wrong table name returns zero rows on most warehouses, which is
        indistinguishable from a quiet slot range and wastes however long it
        takes someone to notice. This turns that into an error naming the
        columns that were expected.
        """
        problems: List[str] = []
        for logical, needed in (("launches", REQUIRED_COLUMNS["launches"]),
                                ("transfers", REQUIRED_COLUMNS["transfers"])):
            table = self.tables.get(
                "transactions" if logical == "launches" else "token_transfers", "")
            if not table:
                problems.append(f"{logical}: no table configured")
                continue
            try:
                columns = set(self.client.schema(table))
            except Exception as exc:
                problems.append(f"{logical}: {table} unreadable ({exc})")
                continue
            missing = [column for column in needed if column not in columns]
            if missing:
                problems.append(
                    f"{logical}: {table} is missing {', '.join(missing)}")
        self.verified = not problems
        return {"verified": self.verified, "problems": problems,
                "tables": dict(self.tables)}

    def estimate(self, window: Window, program: str) -> int:
        """Bytes this query will scan, from the warehouse's dry run."""
        query = LAUNCH_QUERY.format(**self.tables)
        return int(self.client.estimate(query, {
            "start_slot": window.start_slot, "end_slot": window.end_slot,
            "program": program}))

    def launches(self, window: Window, program: str) -> List[Dict[str, Any]]:
        if not self.verified:
            raise ExtractionError(
                "schema not verified; refusing to extract into a corpus that "
                "may be silently empty")
        estimate = self.estimate(window, program)
        if self.bytes_scanned + estimate > self.scan_budget_bytes:
            raise ExtractionError(
                f"window {window.key()} would scan {estimate} bytes, taking the "
                f"run past its {self.scan_budget_bytes} budget")
        query = LAUNCH_QUERY.format(**self.tables)
        rows = list(self.client.query(query, {
            "start_slot": window.start_slot, "end_slot": window.end_slot,
            "program": program}))
        self.bytes_scanned += estimate
        return rows


class RpcBackend:
    """Walk the program's signatures backwards. Always available, always slow.

    This exists so that a desk with no warehouse account is not stuck at zero.
    It cannot enumerate a universe in reasonable time and does not pretend to;
    it is the completeness backstop for a specific window a warehouse missed.
    """

    kind = Backend.RPC

    def __init__(self, rpc: Any, *, page: int = 1_000):
        self.rpc = rpc
        self.page = int(page)
        self.verified = True
        self.bytes_scanned = 0

    def verify(self) -> Dict[str, Any]:
        return {"verified": True, "problems": [],
                "tables": {"note": "RPC needs no schema"}}

    def estimate(self, window: Window, program: str) -> int:
        return 0

    async def launches(self, window: Window, program: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        before: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"limit": self.page, "commitment": "confirmed"}
            if before:
                params["before"] = before
            batch = await self.rpc.request(
                "getSignaturesForAddress", [program, params])
            batch = [row for row in (batch or []) if isinstance(row, dict)]
            if not batch:
                break
            for row in batch:
                slot = int(row.get("slot", 0) or 0)
                if slot < window.start_slot:
                    return rows
                if slot <= window.end_slot and row.get("err") is None:
                    rows.append({"block_slot": slot,
                                 "block_timestamp": row.get("blockTime"),
                                 "signature": row.get("signature")})
            before = batch[-1].get("signature")
            if not before:
                break
        return rows


@dataclass
class ExtractionReport:
    """What was actually retrieved, expressed as coverage rather than volume."""

    windows_planned: int = 0
    windows_done: int = 0
    windows_failed: int = 0
    rows: int = 0
    launches_built: int = 0
    bytes_scanned: int = 0
    failures: List[Dict[str, str]] = field(default_factory=list)
    covered: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        planned = self.windows_planned or 1
        return {
            "schema": HISTORY_WAREHOUSE_SCHEMA_VERSION,
            "status": ("OK" if self.windows_done and not self.windows_failed
                       else "PARTIAL" if self.windows_done else "DATA_BLOCKED"),
            "detail": ("" if self.windows_done else
                       "no window extracted; the corpus is unchanged"),
            "windows_planned": self.windows_planned,
            "windows_done": self.windows_done,
            "windows_failed": self.windows_failed,
            # Coverage, not row count. A row count cannot tell a complete
            # window from a truncated one, and truncation is the failure that
            # silently biases every model trained on the result.
            "coverage": round(self.windows_done / planned, 4),
            "rows": self.rows,
            "launches_built": self.launches_built,
            "bytes_scanned": self.bytes_scanned,
            "windows_covered": list(self.covered),
            "failures": list(self.failures),
        }


class HistoryWarehouse:
    """Runs an extraction plan, checkpoints it, and emits RawLaunch records."""

    def __init__(self, backend: Any, *, checkpoint: Optional[Path] = None,
                 build: Optional[Callable[[List[Dict[str, Any]]], List[RawLaunch]]] = None):
        self.backend = backend
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.build = build or self._default_build
        self.done: set = set()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if self.checkpoint is None or not self.checkpoint.exists():
            return
        try:
            state = json.loads(self.checkpoint.read_text())
            self.done = set(state.get("windows") or [])
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("extraction checkpoint unreadable: %s", exc)

    def _save_checkpoint(self) -> None:
        if self.checkpoint is None:
            return
        try:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint.write_text(json.dumps(
                {"schema": HISTORY_WAREHOUSE_SCHEMA_VERSION,
                 "windows": sorted(self.done)}))
        except OSError as exc:
            logger.warning("extraction checkpoint unwritable: %s", exc)

    @staticmethod
    def _default_build(rows: Sequence[Dict[str, Any]]) -> List[RawLaunch]:
        """Group warehouse rows into per-mint launches.

        Deliberately conservative: a row set that does not identify a mint
        produces nothing rather than a launch with a guessed identity. A
        corpus polluted with misattributed launches is worse than a smaller
        one, because the error is invisible at training time.
        """
        by_mint: Dict[str, List[Dict[str, Any]]] = {}
        created: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            mint = str(row.get("mint", "") or "")
            if not mint:
                continue
            if row.get("kind") == "create":
                created.setdefault(mint, row)
            else:
                by_mint.setdefault(mint, []).append(row)
        launches: List[RawLaunch] = []
        for mint, creation in created.items():
            trades = sorted(by_mint.get(mint, []),
                            key=lambda item: (int(item.get("block_slot", 0) or 0),
                                              str(item.get("signature", ""))))
            launches.append(RawLaunch(
                token=mint,
                creator=str(creation.get("signer", "") or ""),
                created_at=float(creation.get("block_timestamp", 0) or 0),
                bonding_curve=str(creation.get("bonding_curve", "") or ""),
                trades=trades))
        return launches

    def _fetch(self, window: Window, program: str) -> List[Dict[str, Any]]:
        """Ask the backend for a window, whether or not it is async.

        `RpcBackend.launches` is a coroutine function -- it has to be, it makes
        network calls -- while the warehouse and parquet backends are plain
        functions reading through an injected client. Calling both the same way
        used to hand `list.extend` a coroutine object, which raises TypeError,
        which this loop scored as a failed window. The completeness backstop
        was therefore incapable of extracting anything, and said so only as an
        opaque per-window failure.
        """
        result = self.backend.launches(window, program)
        if not inspect.isawaitable(result):
            return list(result or [])
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return list(asyncio.run(result) or [])
        result.close()
        raise ExtractionError(
            f"{type(self.backend).__name__}.launches is async and run() was "
            "called from inside an event loop; await arun() instead")

    async def _afetch(self, window: Window, program: str) -> List[Dict[str, Any]]:
        result = self.backend.launches(window, program)
        if inspect.isawaitable(result):
            result = await result
        return list(result or [])

    def verify(self) -> Dict[str, Any]:
        return self.backend.verify()

    def _skip(self, window: Window, report: ExtractionReport) -> bool:
        if window.key() not in self.done:
            return False
        report.windows_done += 1
        report.covered.append(window.key())
        return True

    def _absorb(self, window: Window, rows: List[Dict[str, Any]],
                report: ExtractionReport,
                launches: List[RawLaunch]) -> None:
        report.rows += len(rows)
        built = self.build(rows)
        launches.extend(built)
        report.launches_built += len(built)
        report.windows_done += 1
        report.covered.append(window.key())
        self.done.add(window.key())

    def _finish(self, report: ExtractionReport) -> None:
        report.bytes_scanned = int(getattr(self.backend, "bytes_scanned", 0))
        self._save_checkpoint()

    def run(self, plan: ExtractionPlan) -> Tuple[List[RawLaunch], ExtractionReport]:
        """Extract every window not already checkpointed."""
        report = ExtractionReport(windows_planned=len(plan.windows))
        launches: List[RawLaunch] = []
        for window in plan.windows:
            if self._skip(window, report):
                continue
            rows: List[Dict[str, Any]] = []
            failed = False
            for program in plan.programs:
                try:
                    rows.extend(self._fetch(window, program))
                except Exception as exc:
                    failed = True
                    report.failures.append(
                        {"window": window.key(), "program": program,
                         "error": f"{type(exc).__name__}: {exc}"})
                    break
            if failed:
                report.windows_failed += 1
                continue
            self._absorb(window, rows, report, launches)
        self._finish(report)
        return launches, report

    async def arun(self, plan: ExtractionPlan
                   ) -> Tuple[List[RawLaunch], ExtractionReport]:
        """The same extraction from inside a running event loop.

        The async backends are the ones that talk to the network, so this is
        the form a live runtime uses; `run` remains for offline batch work.
        """
        report = ExtractionReport(windows_planned=len(plan.windows))
        launches: List[RawLaunch] = []
        for window in plan.windows:
            if self._skip(window, report):
                continue
            rows: List[Dict[str, Any]] = []
            failed = False
            for program in plan.programs:
                try:
                    rows.extend(await self._afetch(window, program))
                except Exception as exc:
                    failed = True
                    report.failures.append(
                        {"window": window.key(), "program": program,
                         "error": f"{type(exc).__name__}: {exc}"})
                    break
            if failed:
                report.windows_failed += 1
                continue
            self._absorb(window, rows, report, launches)
        self._finish(report)
        return launches, report


def windows_between(start_slot: int, end_slot: int, *,
                    slots_per_window: int = 500_000) -> List[Window]:
    """Split a slot range into extractable windows.

    Sized so one window is a tractable query and a failure costs one window
    rather than the run. Half a million slots is roughly two days of Solana.
    """
    if end_slot <= start_slot:
        return []
    windows: List[Window] = []
    cursor = int(start_slot)
    step = max(1, int(slots_per_window))
    while cursor < end_slot:
        upper = min(end_slot, cursor + step)
        windows.append(Window(start_slot=cursor, end_slot=upper))
        cursor = upper
    return windows
