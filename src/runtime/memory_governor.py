"""Shed load before the kernel sheds the process.

This desk was OOM-killed once already, twelve hours into a run, taking the
accumulated forward evidence with it. That is the worst possible failure for
a system whose only real bottleneck is evidence: it does not degrade, it
deletes.

Swap helps and is worth having, but it is not the fix. The fix is that the
process notices it is approaching the ceiling and gives something up, because
every subsystem here has something it can give up cheaply:

  miner concurrency   fewer simultaneous fetches. Costs latency on context
                      that is not on the hot path anyway.
  census detail       spill per-launch rows to disk sooner. Costs nothing at
                      all: the totals are already counted before eviction.
  mark history        shorter price paths per token. Costs some slot-value
                      precision on tokens we are not in.
  episode cache       flush completed episodes to the lake. Costs a write.

None of these touch the decision path. That is the design constraint: a desk
under memory pressure must get quieter, never dumber. Shedding the models or
the hazard tracker to save memory would trade the thing the desk is for the
ability to keep running, which is not a trade worth making -- better to die
loudly and restart than to trade blind.

Three bands, because a single threshold either fires too late to help or so
early that it is always on:

  CALM     below the soft mark. Nothing happens, and anything previously
           given up is taken back.
  TRIM     over the soft mark. Cheap reductions, reversible.
  SHED     over the hard mark. Everything reversible plus an immediate flush,
           and the fact is reported loudly, because a desk that spends its
           life in SHED is a desk on a box too small for it and no amount of
           trimming fixes that.

Reading RSS has no portable stdlib answer, so it is read from /proc where that
exists and reported as unmeasured where it does not. An unmeasured footprint
disables the governor rather than defaulting it to calm: a governor that
believes memory is fine because it cannot see it is worse than none, since it
also suppresses the operator's suspicion.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_GOVERNOR_SCHEMA_VERSION = "v1"

#: Fractions of the ceiling at which each band begins.
DEFAULT_SOFT_FRACTION = 0.70
DEFAULT_HARD_FRACTION = 0.85

#: How long a band must hold before acting. Stops a single allocation spike
#: from trimming everything and then immediately restoring it.
DEFAULT_DWELL_S = 20.0


class Band(Enum):
    CALM = "calm"
    TRIM = "trim"
    SHED = "shed"
    UNMEASURED = "unmeasured"


def read_rss_bytes() -> Optional[int]:
    """This process's resident set, or None where it cannot be read.

    None disables the governor. A governor that assumes calm because it cannot
    see the number would suppress the very suspicion that leads someone to
    look, which is worse than not having one.
    """
    try:
        with open("/proc/self/statm", "r") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            return None
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return None


def detect_ceiling_bytes() -> Optional[int]:
    """The cgroup memory limit, which is what actually kills us.

    Preferred over total system memory because the unit runs under a cgroup
    with its own cap, and on a shared box the system total is a number nobody
    is allowed to reach.
    """
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text in ("max", ""):
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        # An absurd cap means "unlimited" expressed as a huge number.
        if 0 < value < (1 << 62):
            return value
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(pages) * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        return None


@dataclass
class Relief:
    """One thing the desk can give up, and how to give it up."""

    name: str
    #: Called on entering TRIM. Must be cheap, reversible, and must not touch
    #: the decision path.
    trim: Callable[[], Any]
    #: Called on entering SHED, in addition to trim. May block briefly.
    shed: Optional[Callable[[], Any]] = None
    #: Called on returning to CALM.
    restore: Optional[Callable[[], Any]] = None
    detail: str = ""


class MemoryGovernor:
    """Watches the footprint and trades context for survival, never accuracy."""

    def __init__(self, *, ceiling_bytes: Optional[int] = None,
                 soft_fraction: float = DEFAULT_SOFT_FRACTION,
                 hard_fraction: float = DEFAULT_HARD_FRACTION,
                 dwell_s: float = DEFAULT_DWELL_S,
                 read_rss: Callable[[], Optional[int]] = read_rss_bytes):
        self.ceiling_bytes = ceiling_bytes or detect_ceiling_bytes()
        self.soft_fraction = float(soft_fraction)
        self.hard_fraction = float(hard_fraction)
        self.dwell_s = float(dwell_s)
        self._read_rss = read_rss
        self.reliefs: List[Relief] = []
        self.band = Band.CALM
        self._candidate: Optional[Band] = None
        self._candidate_since = 0.0
        self.last_rss = 0
        self.transitions = 0
        self.trims = 0
        self.sheds = 0
        self.peak_fraction = 0.0
        self.history: List[Dict[str, Any]] = []

    def register(self, relief: Relief) -> None:
        self.reliefs.append(relief)

    # --- the loop --------------------------------------------------------

    def observe(self, now: Optional[float] = None) -> Band:
        """One reading. Returns the band actually in force."""
        moment = time.time() if now is None else now
        rss = self._read_rss()
        if rss is None or not self.ceiling_bytes:
            self.band = Band.UNMEASURED
            return self.band
        self.last_rss = int(rss)
        fraction = rss / self.ceiling_bytes
        self.peak_fraction = max(self.peak_fraction, fraction)
        target = (Band.SHED if fraction >= self.hard_fraction
                  else Band.TRIM if fraction >= self.soft_fraction
                  else Band.CALM)
        if target is self.band:
            self._candidate = None
            return self.band
        # Escalation is immediate; de-escalation must dwell. Coming down
        # quickly would restore the caches that put us here and oscillate.
        escalating = (target is Band.SHED
                      or (target is Band.TRIM and self.band is Band.CALM))
        if not escalating:
            if self._candidate is not target:
                self._candidate = target
                self._candidate_since = moment
                return self.band
            if moment - self._candidate_since < self.dwell_s:
                return self.band
        self._enter(target, fraction, moment)
        return self.band

    def _enter(self, band: Band, fraction: float, now: float) -> None:
        previous = self.band
        self.band = band
        self._candidate = None
        self.transitions += 1
        self.history.append({"at": now, "from": previous.value, "to": band.value,
                             "fraction": round(fraction, 4)})
        self.history = self.history[-50:]
        if band is Band.CALM:
            self._call_all("restore")
            logger.info("memory governor back to calm at %.0f%% of ceiling",
                        fraction * 100)
            return
        if band is Band.TRIM:
            self.trims += 1
            self._call_all("trim")
            logger.warning("memory governor trimming at %.0f%% of ceiling",
                           fraction * 100)
            return
        self.sheds += 1
        self._call_all("trim")
        self._call_all("shed")
        # Loud on purpose. A desk that lives in SHED is on a box too small for
        # it, and trimming harder will not fix that.
        logger.error("memory governor SHEDDING at %.0f%% of ceiling; if this "
                     "persists the host is undersized for this workload",
                     fraction * 100)

    def _call_all(self, action: str) -> None:
        for relief in self.reliefs:
            call = getattr(relief, action, None)
            if call is None:
                continue
            try:
                call()
            except Exception as exc:
                # Relief must never be able to kill the desk it is protecting.
                logger.warning("memory relief %s.%s failed: %s",
                               relief.name, action, exc)

    def report(self) -> Dict[str, Any]:
        ceiling = self.ceiling_bytes or 0
        fraction = (self.last_rss / ceiling) if ceiling else None
        return {
            "schema": MEMORY_GOVERNOR_SCHEMA_VERSION,
            "status": ("DATA_BLOCKED" if self.band is Band.UNMEASURED
                       else "OK" if self.band is Band.CALM
                       else "DEGRADED"),
            "detail": (
                "resident set cannot be read on this host; the governor is "
                "disabled rather than assuming the footprint is fine"
                if self.band is Band.UNMEASURED else
                "shedding context to stay inside the memory ceiling"
                if self.band is Band.SHED else
                "trimming caches" if self.band is Band.TRIM else ""),
            "band": self.band.value,
            "rss_bytes": self.last_rss or None,
            "ceiling_bytes": ceiling or None,
            "fraction": round(fraction, 4) if fraction is not None else None,
            "peak_fraction": round(self.peak_fraction, 4) or None,
            "soft_at": self.soft_fraction,
            "hard_at": self.hard_fraction,
            "transitions": self.transitions,
            "trims": self.trims,
            "sheds": self.sheds,
            "reliefs": [{"name": r.name, "detail": r.detail}
                        for r in self.reliefs],
            "recent": self.history[-10:],
        }
