"""Every launch that happened, not only the ones we had an opinion about.

The forward ledger counts outcomes for tokens that reached a decision. That
makes every ratio it reports conditional on our own filters, which is the one
bias that cannot be corrected after the fact: if a screen throws away the
launches that become monsters, a ledger fed downstream of that screen shows a
clean record and a rising win rate while the actual opportunity is being
discarded upstream, silently, forever.

So this counts the DENOMINATOR. Every `token_created` the stream carries is
seen here, whatever happens to it afterwards, and each is tracked through the
funnel:

    SEEN ──► SCREENED (with the reason)
         └─► DECIDED (with the action) ──► ENTERED

Later, from the same stream, each launch is resolved with what it actually did
-- peak multiple, whether it migrated, whether it rugged -- again regardless of
whether we touched it. That is what makes the useful question answerable:

    of the launches that became monsters, which screen threw them away?

No other measurement identifies a losing filter. A win rate cannot: a filter
that rejects everything has no losses. Attribution over entered trades cannot:
the discarded launches are not in it. Only a census over all launches, resolved
independently of our own actions, can price what a screen costs.

Two disciplines carry through.

**Unresolved is never zero.** A launch whose peak we never observed has
`peak_multiple = None` and is excluded from rates, with its count stated
separately. Treating unobserved launches as flat would make every screen look
free, which is the error that runs in the expensive direction.

**Bounded memory, unbounded counting.** Pump.fun produces tens of thousands of
launches a day and this runs on a small box that has already been OOM-killed
once. Per-launch detail is capped and the oldest resolved records are spilled
to disk, but the aggregate counters are updated on the way out, so evicting
detail never changes a reported total. A census that dies at 3am has measured
nothing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LAUNCH_CENSUS_SCHEMA_VERSION = "v1"

#: What counts as a monster for census purposes. Matches the action-value
#: trainer's threshold so "missed monster" means the same thing in both.
MONSTER_MULTIPLE = 10.0

#: Per-launch detail kept in memory. Beyond this the oldest RESOLVED records
#: are spilled; unresolved ones are never evicted, because a launch we are
#: still waiting on is the only kind whose detail we still need.
DEFAULT_MAX_RECORDS = 20_000

#: How long an unresolved launch is kept before being written off as
#: unobserved. Long enough to catch a slow migration, short enough that a
#: restart-heavy week does not fill memory with launches nobody will resolve.
DEFAULT_RESOLVE_WINDOW_S = 6 * 3_600.0


class Stage(Enum):
    """How far into our own funnel a launch got."""

    SEEN = "seen"
    SCREENED = "screened"
    DECIDED = "decided"
    ENTERED = "entered"


@dataclass
class LaunchRecord:
    """One launch, as the stream saw it and as it turned out."""

    mint: str
    creator: str = ""
    detected_at: float = 0.0
    regime: str = "unknown"
    stage: Stage = Stage.SEEN
    #: Why it never reached a decision. The single most valuable field here:
    #: it is what a missed monster is attributed to.
    screen_reason: str = ""
    decided_action: str = ""
    # Resolution, from the stream and independent of anything we did.
    peak_multiple: Optional[float] = None
    migrated: Optional[bool] = None
    rugged: Optional[bool] = None
    rug_mechanism: str = ""
    resolved_at: float = 0.0

    @property
    def resolved(self) -> bool:
        return self.peak_multiple is not None

    @property
    def is_monster(self) -> Optional[bool]:
        """None when unresolved. Never False for lack of measurement."""
        if self.peak_multiple is None:
            return None
        return self.peak_multiple >= MONSTER_MULTIPLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mint": self.mint, "creator": self.creator,
            "detected_at": self.detected_at, "regime": self.regime,
            "stage": self.stage.value, "screen_reason": self.screen_reason,
            "decided_action": self.decided_action,
            "peak_multiple": self.peak_multiple, "migrated": self.migrated,
            "rugged": self.rugged, "rug_mechanism": self.rug_mechanism,
            "resolved_at": self.resolved_at,
        }


@dataclass
class _Totals:
    """Counters that survive eviction of the detail they came from."""

    seen: int = 0
    screened: int = 0
    decided: int = 0
    entered: int = 0
    resolved: int = 0
    monsters: int = 0
    rugs: int = 0
    migrated: int = 0
    #: monsters by the stage they died at, and by the reason they were
    #: screened. This is the attribution that pays for the whole module.
    monsters_by_stage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    monsters_by_screen: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    screened_by_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rugs_by_mechanism: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class LaunchCensus:
    """The denominator, and what each filter costs against it."""

    def __init__(self, path: Optional[Path] = None, *,
                 max_records: int = DEFAULT_MAX_RECORDS,
                 resolve_window_s: float = DEFAULT_RESOLVE_WINDOW_S,
                 spill_path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.spill_path = (Path(spill_path) if spill_path
                           else (self.path.with_name("launch_census.jsonl")
                                 if self.path else None))
        self.max_records = max(100, int(max_records))
        self.resolve_window_s = float(resolve_window_s)
        # Insertion-ordered so eviction is oldest-first without a sort.
        self._records: "OrderedDict[str, LaunchRecord]" = OrderedDict()
        self._totals = _Totals()
        self.spilled = 0
        self.expired_unresolved = 0

    # --- the funnel ------------------------------------------------------

    def see(self, mint: str, *, creator: str = "", at: Optional[float] = None,
            regime: str = "unknown") -> Optional[LaunchRecord]:
        """Record a launch. Called for EVERY token_created, unconditionally.

        Idempotent: the stream can replay, and a launch counted twice would
        inflate the denominator and make every screen look cheaper than it is.
        """
        if not mint:
            return None
        existing = self._records.get(mint)
        if existing is not None:
            return existing
        record = LaunchRecord(mint=mint, creator=creator,
                              detected_at=float(at if at is not None else time.time()),
                              regime=regime)
        self._records[mint] = record
        self._totals.seen += 1
        self._evict_if_needed()
        return record

    def screen(self, mint: str, reason: str) -> None:
        """A launch filtered out before it reached a decision.

        The reason is required and is not allowed to be empty: an unattributed
        screen is a launch that vanished, and the whole point here is that
        nothing vanishes without a name attached.
        """
        record = self._records.get(mint)
        if record is None or record.stage != Stage.SEEN:
            return
        record.stage = Stage.SCREENED
        record.screen_reason = str(reason or "unattributed")
        self._totals.screened += 1
        self._totals.screened_by_reason[record.screen_reason] += 1

    def decide(self, mint: str, action: str) -> None:
        """A launch that reached the decision path, whatever it decided."""
        record = self._records.get(mint)
        if record is None:
            return
        if record.stage in (Stage.SEEN, Stage.SCREENED):
            if record.stage == Stage.SCREENED:
                # It was screened and then decided anyway; undo the screen so
                # the funnel stays a partition rather than double counting.
                self._totals.screened -= 1
                self._totals.screened_by_reason[record.screen_reason] -= 1
                record.screen_reason = ""
            record.stage = Stage.DECIDED
            self._totals.decided += 1
        record.decided_action = str(action or "")

    def enter(self, mint: str) -> None:
        """A position was actually taken."""
        record = self._records.get(mint)
        if record is None:
            return
        if record.stage != Stage.ENTERED:
            if record.stage != Stage.DECIDED:
                self._totals.decided += 1
            record.stage = Stage.ENTERED
            self._totals.entered += 1

    # --- resolution, independent of what we did --------------------------

    def resolve(self, mint: str, *, peak_multiple: Optional[float] = None,
                migrated: Optional[bool] = None, rugged: Optional[bool] = None,
                rug_mechanism: str = "", at: Optional[float] = None) -> None:
        """What the launch actually did, from the stream.

        The peak only ever ratchets upward: resolution arrives incrementally
        as the token trades, and taking the latest reading rather than the
        highest would record a monster's post-peak price as its outcome.
        """
        record = self._records.get(mint)
        if record is None:
            return
        now = float(at if at is not None else time.time())
        first_resolution = record.peak_multiple is None
        if peak_multiple is not None:
            value = float(peak_multiple)
            was_monster = record.is_monster
            record.peak_multiple = (value if record.peak_multiple is None
                                    else max(record.peak_multiple, value))
            record.resolved_at = now
            if first_resolution:
                self._totals.resolved += 1
            # Counted at the moment it crosses, so a token that ratchets past
            # the threshold later is still counted exactly once.
            if record.is_monster and not was_monster:
                self._totals.monsters += 1
                self._totals.monsters_by_stage[record.stage.value] += 1
                if record.stage == Stage.SCREENED:
                    self._totals.monsters_by_screen[
                        record.screen_reason or "unattributed"] += 1
        if migrated is not None and record.migrated is None:
            record.migrated = bool(migrated)
            if record.migrated:
                self._totals.migrated += 1
        if rugged is not None and record.rugged is None:
            record.rugged = bool(rugged)
            if record.rugged:
                self._totals.rugs += 1
                mechanism = rug_mechanism or "unclassified"
                record.rug_mechanism = mechanism
                self._totals.rugs_by_mechanism[mechanism] += 1
        elif rug_mechanism and not record.rug_mechanism:
            record.rug_mechanism = rug_mechanism

    # --- memory discipline -----------------------------------------------

    def _evict_if_needed(self) -> None:
        """Spill oldest resolved detail. Totals are already counted.

        Unresolved records are kept: a launch we are still waiting on is the
        only kind whose detail we still need. If the census is entirely
        unresolved and over cap, the oldest are written off as unobserved
        rather than growing without bound -- an OOM measures nothing at all,
        which is strictly worse than a stated gap.
        """
        if len(self._records) <= self.max_records:
            return
        cutoff = time.time() - self.resolve_window_s
        for mint in list(self._records.keys()):
            if len(self._records) <= self.max_records:
                break
            record = self._records[mint]
            if record.resolved:
                self._spill(record)
                del self._records[mint]
            elif record.detected_at < cutoff:
                self.expired_unresolved += 1
                self._spill(record)
                del self._records[mint]
        # Still over after both passes: the stream is outrunning resolution.
        # Drop the oldest regardless, counting them as unobserved.
        while len(self._records) > self.max_records:
            _mint, record = self._records.popitem(last=False)
            if not record.resolved:
                self.expired_unresolved += 1
            self._spill(record)

    def _spill(self, record: LaunchRecord) -> None:
        """Append one record to the lake before its detail leaves memory."""
        if self.spill_path is None:
            return
        try:
            self.spill_path.parent.mkdir(parents=True, exist_ok=True)
            with self.spill_path.open("a") as handle:
                handle.write(json.dumps(record.to_dict()) + "\n")
            self.spilled += 1
        except OSError as exc:
            logger.debug("census spill failed for %s: %s", record.mint, exc)

    # --- what it is all for ----------------------------------------------

    def missed_monster_report(self) -> Dict[str, Any]:
        """Which filter is throwing away the launches that matter.

        Rates are over RESOLVED launches only. The unresolved count is stated
        beside them rather than folded in, because a launch we never priced is
        not evidence that a screen was right.
        """
        totals = self._totals
        monsters = totals.monsters
        by_stage = dict(totals.monsters_by_stage)
        reached = by_stage.get(Stage.DECIDED.value, 0) + by_stage.get(Stage.ENTERED.value, 0)
        screened_away = by_stage.get(Stage.SCREENED.value, 0)
        never_reached = by_stage.get(Stage.SEEN.value, 0)
        ranked = sorted(totals.monsters_by_screen.items(),
                        key=lambda item: item[1], reverse=True)
        return {
            "resolved_launches": totals.resolved,
            "unresolved_launches": len(self._records) - sum(
                1 for record in self._records.values() if record.resolved),
            "expired_unobserved": self.expired_unresolved,
            "monsters_resolved": monsters,
            "monsters_entered": by_stage.get(Stage.ENTERED.value, 0),
            "monsters_decided_not_entered": by_stage.get(Stage.DECIDED.value, 0),
            "monsters_screened_out": screened_away,
            "monsters_never_reached_a_screen": never_reached,
            # The headline. Of the monsters we resolved, what share did our
            # own filters discard before anything could decide on them?
            "monster_capture_rate": (reached / monsters) if monsters else None,
            "costliest_screens": [
                {"reason": reason, "monsters_discarded": count,
                 "total_screened": totals.screened_by_reason.get(reason, 0),
                 # Of everything this screen rejected, how much was a monster.
                 # A screen with a high rate is mispriced; a screen with a high
                 # count is expensive even at a low rate.
                 "monster_share_of_rejections": (
                     count / totals.screened_by_reason[reason]
                     if totals.screened_by_reason.get(reason) else None)}
                for reason, count in ranked[:10]],
            "detail": ("" if monsters else
                       "no resolved monster yet; capture rate is unmeasurable "
                       "and is reported as null rather than as perfect"),
        }

    def report(self) -> Dict[str, Any]:
        totals = self._totals
        return {
            "schema": LAUNCH_CENSUS_SCHEMA_VERSION,
            "status": "OK" if totals.seen else "DATA_BLOCKED",
            "detail": ("" if totals.seen else
                       "no launch has been seen; the denominator is empty and "
                       "every rate computed against it would be undefined"),
            "funnel": {
                "seen": totals.seen,
                "screened_out": totals.screened,
                "reached_a_decision": totals.decided,
                "entered": totals.entered,
                # Seen but neither screened nor decided: launches that fell
                # through the funnel without anything happening to them. A
                # rising number here is a pipeline defect, not a policy.
                "unaccounted": max(0, totals.seen - totals.screened - totals.decided),
            },
            "outcomes": {
                "resolved": totals.resolved,
                "monsters": totals.monsters,
                "rugs": totals.rugs,
                "migrated": totals.migrated,
                "rug_share_of_resolved": (totals.rugs / totals.resolved
                                          if totals.resolved else None),
                "monster_share_of_resolved": (totals.monsters / totals.resolved
                                              if totals.resolved else None),
            },
            "screens": dict(sorted(totals.screened_by_reason.items(),
                                   key=lambda item: item[1], reverse=True)),
            "rug_mechanisms": dict(sorted(totals.rugs_by_mechanism.items(),
                                          key=lambda item: item[1], reverse=True)),
            "missed_monsters": self.missed_monster_report(),
            "memory": {"in_memory": len(self._records),
                       "cap": self.max_records,
                       "spilled_to_disk": self.spilled},
        }

    # --- persistence -----------------------------------------------------

    def state(self) -> Dict[str, Any]:
        return {
            "schema": LAUNCH_CENSUS_SCHEMA_VERSION,
            "totals": {
                "seen": self._totals.seen, "screened": self._totals.screened,
                "decided": self._totals.decided, "entered": self._totals.entered,
                "resolved": self._totals.resolved, "monsters": self._totals.monsters,
                "rugs": self._totals.rugs, "migrated": self._totals.migrated,
                "monsters_by_stage": dict(self._totals.monsters_by_stage),
                "monsters_by_screen": dict(self._totals.monsters_by_screen),
                "screened_by_reason": dict(self._totals.screened_by_reason),
                "rugs_by_mechanism": dict(self._totals.rugs_by_mechanism),
            },
            "spilled": self.spilled,
            "expired_unresolved": self.expired_unresolved,
            "records": [record.to_dict() for record in self._records.values()],
        }

    def save(self) -> bool:
        """Atomic write. A census that resets on restart never reaches any
        threshold, however long it runs."""
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", dir=str(self.path.parent), delete=False)
            with handle:
                json.dump(self.state(), handle)
            os.replace(handle.name, self.path)
            return True
        except OSError as exc:
            logger.warning("launch census save failed: %s", exc)
            return False

    def load(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        try:
            state = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("launch census unreadable: %s", exc)
            return False
        totals = state.get("totals") or {}
        self._totals = _Totals(
            seen=int(totals.get("seen", 0)), screened=int(totals.get("screened", 0)),
            decided=int(totals.get("decided", 0)), entered=int(totals.get("entered", 0)),
            resolved=int(totals.get("resolved", 0)),
            monsters=int(totals.get("monsters", 0)), rugs=int(totals.get("rugs", 0)),
            migrated=int(totals.get("migrated", 0)),
            monsters_by_stage=defaultdict(int, totals.get("monsters_by_stage") or {}),
            monsters_by_screen=defaultdict(int, totals.get("monsters_by_screen") or {}),
            screened_by_reason=defaultdict(int, totals.get("screened_by_reason") or {}),
            rugs_by_mechanism=defaultdict(int, totals.get("rugs_by_mechanism") or {}))
        self.spilled = int(state.get("spilled", 0))
        self.expired_unresolved = int(state.get("expired_unresolved", 0))
        self._records.clear()
        for row in state.get("records") or []:
            mint = row.get("mint")
            if not mint:
                continue
            self._records[mint] = LaunchRecord(
                mint=mint, creator=row.get("creator", ""),
                detected_at=float(row.get("detected_at", 0.0) or 0.0),
                regime=row.get("regime", "unknown"),
                stage=Stage(row.get("stage", "seen")),
                screen_reason=row.get("screen_reason", ""),
                decided_action=row.get("decided_action", ""),
                peak_multiple=row.get("peak_multiple"),
                migrated=row.get("migrated"), rugged=row.get("rugged"),
                rug_mechanism=row.get("rug_mechanism", ""),
                resolved_at=float(row.get("resolved_at", 0.0) or 0.0))
        return True
