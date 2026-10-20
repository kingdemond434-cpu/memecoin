"""What a position could actually have realised, recorded rather than modelled.

`executable_tail.py` projects a venue forward and asks what a sale into it
would fetch. That projection has to be calibrated against something, and the
something does not exist yet: the desk records that a token reached 100x and
does not record whether anything could have been sold there.

This is that record, and it has to live after migration, because that is where
the tail is. A Pump bonding curve completes near 85 SOL of real reserves,
which from a T0 entry is 14.4x -- so every 20x, 100x and 1000x this desk will
ever see happens on a PumpSwap pool. A tail corpus built on curve observations
is a corpus that stops just before the interesting part.

So the lineage is followed end to end:

    mint -> curve state -> completion -> pool -> reserve path -> exits

and at every observed pool state one question is asked of a REFERENCE
POSITION: if we held this, what could we have sold right now, inside an impact
we would accept, and what would it have fetched?

Two numbers come out of that per mint, and the gap between them is the whole
point:

    max_price_multiple       -- what the chart printed
    max_executable_multiple  -- what a position could have taken out

A corpus of the first is a corpus of screenshots. Everything downstream --
tail calibration, the capturable-upside objective, ticket sizing -- needs the
second, and nothing in this repository was producing it.

Observed, never inferred. A mint whose pool the desk never saw is recorded as
unmeasured rather than filled in from the curve, because the whole purpose of
the corpus is to be the ground truth a projection is checked against, and a
projection checked against itself confirms nothing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from collections import deque

logger = logging.getLogger(__name__)

MIGRATION_LINEAGE_SCHEMA_VERSION = "v1"

#: Pool observations kept per mint. A migrated token's interesting life is
#: hours, and one observation per trade would be unbounded; this keeps the
#: shape of the path without keeping every tick.
DEFAULT_SAMPLES_PER_MINT = 512

#: Mints tracked. Bounded, oldest first.
DEFAULT_MAX_MINTS = 4_096

#: The rungs an executable path is summarised at.
DEFAULT_RUNGS: Tuple[float, ...] = (
    2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0, 1000.0)


@dataclass
class PoolSample:
    """One observation of a migrated pool, and what it would have paid us."""

    at: float
    base_reserves: int
    quote_reserves: int
    price_quote_per_base: float
    #: Price relative to the position's entry. The multiple the chart shows.
    price_multiple: Optional[float] = None
    #: What the reference position could actually have taken out here, as a
    #: multiple of what it cost. None when the sale could not be quoted.
    executable_multiple: Optional[float] = None
    exitable_fraction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MigrationRecord:
    """One mint's journey from its curve to whatever its pool did next."""

    mint: str
    first_seen: float = 0.0
    #: Last curve state observed before completion, as reserves.
    curve_sol_reserves: int = 0
    curve_token_reserves: int = 0
    curve_observed_at: float = 0.0
    migrated_at: Optional[float] = None
    pool: str = ""
    #: The reference position this mint's executable path is measured for.
    position_tokens: int = 0
    cost_lamports: int = 0
    entry_price: Optional[float] = None
    samples: List[PoolSample] = field(default_factory=list)

    @property
    def migrated(self) -> bool:
        return self.migrated_at is not None

    @property
    def max_price_multiple(self) -> Optional[float]:
        values = [s.price_multiple for s in self.samples
                  if s.price_multiple is not None]
        return max(values) if values else None

    @property
    def max_executable_multiple(self) -> Optional[float]:
        values = [s.executable_multiple for s in self.samples
                  if s.executable_multiple is not None]
        return max(values) if values else None

    @property
    def capture_ratio(self) -> Optional[float]:
        """Executable over printed. The number the whole corpus exists for."""
        printed = self.max_price_multiple
        taken = self.max_executable_multiple
        if not printed or printed <= 0 or taken is None:
            return None
        return float(taken / printed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mint": self.mint, "first_seen": self.first_seen,
            "curve_sol_reserves": self.curve_sol_reserves,
            "curve_token_reserves": self.curve_token_reserves,
            "curve_observed_at": self.curve_observed_at,
            "migrated_at": self.migrated_at, "pool": self.pool,
            "position_tokens": self.position_tokens,
            "cost_lamports": self.cost_lamports, "entry_price": self.entry_price,
            "max_price_multiple": self.max_price_multiple,
            "max_executable_multiple": self.max_executable_multiple,
            "capture_ratio": self.capture_ratio,
            "samples": [s.to_dict() for s in self.samples],
        }


class MigrationLineage:
    """Follows mints across the migration boundary and prices the exits."""

    def __init__(self, path: Optional[str] = None, *,
                 acceptable_impact: float = 0.10,
                 samples_per_mint: int = DEFAULT_SAMPLES_PER_MINT,
                 max_mints: int = DEFAULT_MAX_MINTS):
        self.path = Path(path) if path else None
        self.acceptable_impact = float(acceptable_impact)
        self.samples_per_mint = int(samples_per_mint)
        self.max_mints = int(max_mints)
        self._records: Dict[str, MigrationRecord] = {}
        self._order: Deque[str] = deque()

    # -- observation ------------------------------------------------------

    def _record(self, mint: str, at: float) -> MigrationRecord:
        record = self._records.get(mint)
        if record is None:
            record = MigrationRecord(mint=mint, first_seen=at)
            self._records[mint] = record
            self._order.append(mint)
            self._evict()
        return record

    def _evict(self) -> None:
        while len(self._order) > self.max_mints:
            stale = self._order.popleft()
            self._records.pop(stale, None)

    def set_reference_position(self, mint: str, position_tokens: int,
                               cost_lamports: int,
                               at: Optional[float] = None) -> None:
        """The position whose exits this mint's path is measured for.

        Set once, when the desk enters (or would have entered). Every sample
        after it is priced for THIS size, because exitability is a property of
        the pair and not of the pool alone -- the same 100x is fully
        realisable for a small ticket and mostly unreachable for a large one.
        """
        record = self._record(mint, time.time() if at is None else float(at))
        if position_tokens <= 0 or cost_lamports <= 0:
            return
        record.position_tokens = int(position_tokens)
        record.cost_lamports = int(cost_lamports)
        record.entry_price = float(cost_lamports) / float(position_tokens)

    def observe_curve(self, mint: str, state: Any,
                      at: Optional[float] = None) -> None:
        """The last curve state before completion, kept for the crossing."""
        if state is None:
            return
        moment = time.time() if at is None else float(at)
        record = self._record(mint, moment)
        record.curve_sol_reserves = int(
            getattr(state, "real_sol_reserves", 0) or 0)
        record.curve_token_reserves = int(
            getattr(state, "real_token_reserves", 0) or 0)
        record.curve_observed_at = moment

    def observe_migration(self, mint: str, pool: Any,
                          at: Optional[float] = None) -> None:
        """The crossing itself, from the pool the mint actually landed in."""
        moment = time.time() if at is None else float(at)
        record = self._record(mint, moment)
        if record.migrated_at is None:
            record.migrated_at = moment
        if pool is not None:
            record.pool = str(getattr(pool, "pool", "") or "")
        self.observe_pool(mint, pool, moment)

    def observe_pool(self, mint: str, pool: Any,
                     at: Optional[float] = None) -> Optional[PoolSample]:
        """One pool reading, priced for the reference position."""
        if pool is None or not getattr(pool, "tradeable", False):
            return None
        moment = time.time() if at is None else float(at)
        record = self._record(mint, moment)
        price = float(getattr(pool, "price_quote_per_base", 0.0) or 0.0)
        if price <= 0:
            return None
        sample = PoolSample(
            at=moment,
            base_reserves=int(getattr(pool, "base_reserves", 0) or 0),
            quote_reserves=int(getattr(pool, "effective_quote_reserves", 0) or 0),
            price_quote_per_base=price)
        if record.entry_price and record.entry_price > 0:
            sample.price_multiple = price / record.entry_price
            fraction, executable = self._price_exit(pool, record)
            sample.exitable_fraction = fraction
            sample.executable_multiple = executable
        record.samples.append(sample)
        if len(record.samples) > self.samples_per_mint:
            # Thin the middle rather than the ends: the opening and the peak
            # are the two readings a tail study cannot lose.
            record.samples = (record.samples[:1]
                              + record.samples[2::2]
                              + record.samples[-1:])
        return sample

    def _price_exit(self, pool: Any, record: MigrationRecord
                    ) -> Tuple[Optional[float], Optional[float]]:
        """What the reference position could take out of this pool right now."""
        from src.chains.pumpswap_curve import quote_sell, sell_capacity_base
        if record.position_tokens <= 0 or record.cost_lamports <= 0:
            return None, None
        capacity = sell_capacity_base(pool, max_impact_pct=self.acceptable_impact)
        if capacity <= 0:
            return None, None
        sellable = min(record.position_tokens, int(capacity))
        quote = quote_sell(pool, sellable)
        if quote.data_status != "OK" or quote.output_amount <= 0:
            return None, None
        return (sellable / float(record.position_tokens),
                quote.output_amount / float(record.cost_lamports))

    # -- reading ----------------------------------------------------------

    def get(self, mint: str) -> Optional[MigrationRecord]:
        return self._records.get(mint)

    def executable_path(self, mint: str,
                        rungs: Sequence[float] = DEFAULT_RUNGS
                        ) -> Dict[str, Any]:
        """At each printed rung this mint reached, what was actually takeable.

        Answered only from observations. A rung the price never reached is
        absent, and a rung reached at a moment the desk did not sample is
        unmeasured -- both of which are different from "not executable", and
        collapsing them would manufacture a tail-capture rate.
        """
        record = self._records.get(mint)
        if record is None or not record.samples:
            return {"status": "DATA_BLOCKED", "mint": mint,
                    "detail": "no pool observation for this mint"}
        priced = [s for s in record.samples if s.price_multiple is not None]
        if not priced:
            return {"status": "DATA_BLOCKED", "mint": mint,
                    "detail": "no reference position, so no exit was priced"}
        ladder: Dict[str, Any] = {}
        for rung in rungs:
            reached = [s for s in priced if s.price_multiple >= rung]
            if not reached:
                continue
            first = min(reached, key=lambda s: s.at)
            ladder[f"{rung:g}x"] = {
                "reached_at": first.at,
                "executable_multiple": first.executable_multiple,
                "exitable_fraction": first.exitable_fraction,
                "measured": first.executable_multiple is not None,
            }
        return {
            "status": "OK", "mint": mint, "pool": record.pool,
            "migrated": record.migrated,
            "samples": len(record.samples),
            "max_price_multiple": record.max_price_multiple,
            "max_executable_multiple": record.max_executable_multiple,
            "capture_ratio": record.capture_ratio,
            "ladder": ladder,
        }

    def report(self) -> Dict[str, Any]:
        """The corpus, and the one statistic it exists to produce."""
        migrated = [r for r in self._records.values() if r.migrated]
        priced = [r for r in self._records.values()
                  if r.capture_ratio is not None]
        ratios = sorted(r.capture_ratio for r in priced)
        median = (ratios[len(ratios) // 2] if ratios else None)
        printed = [r.max_price_multiple for r in priced
                   if r.max_price_multiple is not None]
        return {
            "schema": MIGRATION_LINEAGE_SCHEMA_VERSION,
            "status": "OK" if priced else "DATA_BLOCKED",
            "mints_tracked": len(self._records),
            "migrated": len(migrated),
            "with_a_reference_position": sum(
                1 for r in self._records.values() if r.position_tokens > 0),
            "priced": len(priced),
            # The headline. Below 1.0 means the printed multiple overstates
            # what a position of this size could have taken out, and by how
            # much. A corpus without this number is a corpus of screenshots.
            "median_capture_ratio": median,
            "highest_printed_multiple": (max(printed) if printed else None),
            "highest_executable_multiple": (
                max(r.max_executable_multiple for r in priced) if priced else None),
            "detail": ("" if priced else
                       "no mint has both a reference position and a pool "
                       "observation; nothing has been priced, so no capture "
                       "ratio exists"),
        }

    # -- persistence ------------------------------------------------------

    def save(self) -> bool:
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "schema": MIGRATION_LINEAGE_SCHEMA_VERSION,
                "acceptable_impact": self.acceptable_impact,
                "records": [r.to_dict() for r in self._records.values()],
            }), encoding="utf-8")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("migration lineage not saved: %s", exc)
            return False

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("migration lineage not loaded: %s", exc)
            return False
        if payload.get("schema") != MIGRATION_LINEAGE_SCHEMA_VERSION:
            return False
        self.acceptable_impact = float(
            payload.get("acceptable_impact", self.acceptable_impact))
        self._records.clear()
        self._order.clear()
        for raw in payload.get("records", ()):
            record = MigrationRecord(
                mint=str(raw.get("mint", "")),
                first_seen=float(raw.get("first_seen", 0.0)),
                curve_sol_reserves=int(raw.get("curve_sol_reserves", 0)),
                curve_token_reserves=int(raw.get("curve_token_reserves", 0)),
                curve_observed_at=float(raw.get("curve_observed_at", 0.0)),
                migrated_at=raw.get("migrated_at"),
                pool=str(raw.get("pool", "")),
                position_tokens=int(raw.get("position_tokens", 0)),
                cost_lamports=int(raw.get("cost_lamports", 0)),
                entry_price=raw.get("entry_price"),
                samples=[PoolSample(**item) for item in raw.get("samples", ())])
            if not record.mint:
                continue
            self._records[record.mint] = record
            self._order.append(record.mint)
        return True
