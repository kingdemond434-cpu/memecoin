"""Where the milliseconds actually go, measured rather than assumed.

This desk had no end-to-end latency measurement at all. Not a slow one, none:
`perf_counter` appeared zero times in `src/`. For a system whose entire
premise is arriving before other people, that is the worst possible thing not
to instrument, and it makes every speed decision a guess. Rewrite the builder
in Rust? Buy a faster stream? Move the box to Frankfurt? Without this you
cannot rank those, and the first two were guesses that measurement has already
partly refuted -- the whole Python build-and-sign step turns out to be about
120 microseconds, which is 0.03% of a slot.

So the path is decomposed into stages that map onto DIFFERENT INVESTMENTS,
because a decomposition that does not change what you would buy is a
decomposition that is only interesting:

  chain_to_receive   block time to the moment the bytes hit our socket.
                     Provider quality plus physical distance. Fix with money:
                     a better stream, a closer box.
  receive_to_decode  protobuf parse, discriminator match, event construction.
                     Fix with code, and this is where Rust actually pays.
  decode_to_decide   screens, models, the policy kernel. Fix with code.
  decide_to_build    account derivation and instruction assembly.
  build_to_sign      message compile and Ed25519.
  sign_to_submit     serialisation and the outbound request.
  submit_to_land     the network and the leader. Not ours, and measured
                     anyway, because it decides whether a tip was worth it.

Three disciplines, all of them the same discipline this codebase applies to
everything else.

**An unmarked stage is unmeasured, never zero.** A trace that skips
`build_to_sign` because the desk never built anything reports that stage as
having no samples, not as being instant. Zero-filling would make a path that
never runs look like the fastest one.

**The clock is stated, not implied.** Durations use `perf_counter_ns`, which
is monotonic and immune to NTP steps. The one term that cannot use it is
`chain_to_receive`, because block time comes from the cluster and our receive
time from this host, so the two are only comparable to whatever NTP skew the
box carries -- tens of milliseconds, occasionally worse. That term is
reported with its own caveat attached rather than quoted to the microsecond
alongside terms that deserve it.

**Percentiles, not means.** A mean latency is dominated by the median and a
sniper is killed by the tail: the p99 is the launch that mattered. Samples
are kept in a bounded reservoir per stage so the memory is fixed regardless of
how long the desk runs.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LATENCY_SCHEMA_VERSION = "v1"

#: The hot path in order. Every stage names a different thing you would buy or
#: build to fix it; a stage that does not is not worth a separate bucket.
STAGES: Tuple[str, ...] = (
    "chain_to_receive",
    "receive_to_decode",
    "decode_to_dispatch",
    "dispatch_to_decide",
    "decide_to_build",
    "build_to_sign",
    "sign_to_submit",
    "submit_to_land",
)

#: Stages we can actually do something about in code, as opposed to by buying
#: a closer box or a better stream. Used by the report to say where the next
#: hour of engineering should go.
OURS: Tuple[str, ...] = (
    "receive_to_decode", "decode_to_dispatch", "dispatch_to_decide",
    "decide_to_build", "build_to_sign", "sign_to_submit",
)

#: Samples kept per stage. Fixed memory: the desk runs for weeks and the
#: reservoir must not grow with uptime.
DEFAULT_RESERVOIR = 4_096

#: Traces held open at once. A trace is abandoned when its launch is screened
#: and never decided on, which is most of them, so this is bounded and the
#: oldest are dropped rather than accumulated.
DEFAULT_OPEN_TRACES = 2_048


def now_ns() -> int:
    """Monotonic nanoseconds. Never wall clock: an NTP step mid-trace would
    otherwise produce a negative duration, and a negative duration in a
    percentile is worse than a missing sample."""
    return time.perf_counter_ns()


@dataclass
class Trace:
    """One event's journey, from the chain to a landed transaction."""

    key: str
    started_ns: int
    #: Wall-clock at trace start, kept ONLY to compare against block time.
    #: Never used for a duration between two of our own marks.
    started_wall: float = 0.0
    #: The cluster's own timestamp for the block this came from, if the stream
    #: carried one. Absent for streams that do not, and then chain_to_receive
    #: is simply not measured rather than being guessed at.
    block_time: Optional[float] = None
    slot: Optional[int] = None
    marks: Dict[str, int] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    outcome: str = ""

    def mark(self, name: str, at_ns: Optional[int] = None) -> None:
        """Record reaching a point. Idempotent: the FIRST mark wins.

        A retry that re-marks `sign` would otherwise overwrite the original
        and report the retry's duration as the path's, which flatters exactly
        the case that went badly.
        """
        if name in self.marks:
            return
        self.marks[name] = now_ns() if at_ns is None else at_ns
        self.order.append(name)

    def elapsed_us(self, first: str, second: str) -> Optional[float]:
        left, right = self.marks.get(first), self.marks.get(second)
        if left is None or right is None:
            return None
        return (right - left) / 1_000.0

    def total_us(self) -> Optional[float]:
        if not self.order:
            return None
        return (self.marks[self.order[-1]] - self.started_ns) / 1_000.0


class _Reservoir:
    """Bounded samples with percentiles. Sorted on read, not on write."""

    __slots__ = ("samples", "count", "total", "worst")

    def __init__(self, capacity: int):
        self.samples: Deque[float] = deque(maxlen=capacity)
        self.count = 0
        self.total = 0.0
        self.worst = 0.0

    def add(self, value: float) -> None:
        self.samples.append(value)
        self.count += 1
        self.total += value
        self.worst = max(self.worst, value)

    def percentile(self, fraction: float) -> Optional[float]:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1,
                    max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    def to_dict(self) -> Dict[str, Any]:
        if not self.count:
            # Never zero. A stage with no samples has not been measured, and
            # reporting 0.0 would make the path that never runs look fastest.
            return {"samples": 0, "data_status": "DATA_BLOCKED",
                    "p50_us": None, "p90_us": None, "p99_us": None,
                    "max_us": None, "mean_us": None}
        return {
            "samples": self.count,
            "data_status": "OK",
            "p50_us": round(self.percentile(0.50) or 0.0, 1),
            "p90_us": round(self.percentile(0.90) or 0.0, 1),
            "p99_us": round(self.percentile(0.99) or 0.0, 1),
            "max_us": round(self.worst, 1),
            "mean_us": round(self.total / self.count, 1),
        }


class LatencyLedger:
    """Every stage's distribution, and which one to fix next."""

    def __init__(self, *, reservoir: int = DEFAULT_RESERVOIR,
                 max_open: int = DEFAULT_OPEN_TRACES,
                 stages: Sequence[str] = STAGES):
        self.stages = tuple(stages)
        self._buckets: Dict[str, _Reservoir] = {
            stage: _Reservoir(reservoir) for stage in self.stages}
        self._total = _Reservoir(reservoir)
        self._open: Dict[str, Trace] = {}
        self.max_open = int(max_open)
        self.opened = 0
        self.closed = 0
        self.abandoned = 0
        self.outcomes: Dict[str, int] = {}
        #: Wall-clock minus block-time samples, kept apart from everything
        #: else because they carry NTP skew and the others do not.
        self._chain_lag = _Reservoir(reservoir)

    # --- tracing ---------------------------------------------------------

    def open(self, key: str, *, block_time: Optional[float] = None,
             slot: Optional[int] = None) -> Trace:
        """Start a trace at the moment bytes arrive. Cheap: two clock reads."""
        trace = Trace(key=key, started_ns=now_ns(), started_wall=time.time(),
                      block_time=block_time, slot=slot)
        if len(self._open) >= self.max_open:
            # Oldest first. Most launches are screened and never decided on,
            # so most traces are abandoned by design and that is not an error.
            oldest = next(iter(self._open))
            self._open.pop(oldest, None)
            self.abandoned += 1
        self._open[key] = trace
        self.opened += 1
        if block_time:
            lag_us = max(0.0, (trace.started_wall - float(block_time)) * 1e6)
            self._chain_lag.add(lag_us)
            self._buckets["chain_to_receive"].add(lag_us)
        return trace

    def mark(self, key: str, stage_end: str, at_ns: Optional[int] = None) -> None:
        """Mark a point on an open trace. A missing trace is a no-op.

        Deliberately silent: the hot path must not raise because a launch was
        screened before it got here, and a decision loop that can throw from
        its own instrumentation is instrumentation that costs money.
        """
        trace = self._open.get(key)
        if trace is not None:
            trace.mark(stage_end, at_ns)

    def close(self, key: str, outcome: str = "") -> Optional[Trace]:
        """Finish a trace and fold its stages into the distributions."""
        trace = self._open.pop(key, None)
        if trace is None:
            return None
        trace.outcome = outcome
        self.closed += 1
        self.outcomes[outcome or "unspecified"] = (
            self.outcomes.get(outcome or "unspecified", 0) + 1)
        previous_mark: Optional[str] = None
        previous_ns = trace.started_ns
        for name in trace.order:
            stage = name if previous_mark is None else name
            if stage in self._buckets and stage != "chain_to_receive":
                self._buckets[stage].add((trace.marks[name] - previous_ns) / 1_000.0)
            previous_mark = name
            previous_ns = trace.marks[name]
        total = trace.total_us()
        if total is not None:
            self._total.add(total)
        return trace

    # --- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """The distributions, plus the one sentence that decides what to do.

        `dominant_stage` is the point of the whole module. It names the stage
        with the largest p99 among the ones WE control, because that is the
        next hour of engineering; `chain_to_receive` is reported separately
        because when that dominates the answer is money, not code, and mixing
        them produces a number nobody can act on.
        """
        stages = {stage: bucket.to_dict()
                  for stage, bucket in self._buckets.items()}
        measured = {stage: row for stage, row in stages.items()
                    if row["data_status"] == "OK"}
        ours = {stage: row for stage, row in measured.items() if stage in OURS}
        dominant = max(ours, key=lambda stage: ours[stage]["p99_us"]) if ours else None
        chain = stages.get("chain_to_receive", {})
        unmeasured = sorted(stage for stage, row in stages.items()
                            if row["data_status"] != "OK")

        detail = ""
        if not measured:
            status = "DATA_BLOCKED"
            detail = ("nothing has been traced yet; every speed claim about "
                      "this desk is currently unverified")
        else:
            status = "OK"
            if dominant:
                p99 = ours[dominant]["p99_us"]
                detail = (f"{dominant} is the slowest stage we control "
                          f"(p99 {p99/1000:.1f}ms)")
            if chain.get("data_status") == "OK" and dominant:
                chain_p99 = chain["p99_us"]
                if chain_p99 > ours[dominant]["p99_us"]:
                    detail = (f"the wire dominates: chain-to-receive p99 is "
                              f"{chain_p99/1000:.0f}ms against {dominant} at "
                              f"{ours[dominant]['p99_us']/1000:.1f}ms -- this is "
                              "a provider and geography problem, not a code one")

        return {
            "schema": LATENCY_SCHEMA_VERSION,
            "status": status,
            "detail": detail,
            "traces_opened": self.opened,
            "traces_closed": self.closed,
            "traces_abandoned": self.abandoned,
            "outcomes": dict(sorted(self.outcomes.items())),
            "dominant_controllable_stage": dominant,
            "unmeasured_stages": unmeasured,
            "total": self._total.to_dict(),
            "stages": stages,
            "chain_to_receive_caveat": (
                "block time comes from the cluster and receive time from this "
                "host, so this term carries the box's NTP skew -- tens of "
                "milliseconds, occasionally worse. Read it as a magnitude, "
                "not a measurement. Every other stage is perf_counter_ns on "
                "one clock and is exact."),
        }

    def budget_report(self, budgets_us: Dict[str, float]) -> Dict[str, Any]:
        """Which stages are over a declared budget, at p99.

        A budget makes a regression visible before it costs a trade. Without
        one, a stage that doubles is just a number that got bigger.
        """
        rows = []
        breaching = []
        for stage, budget in sorted(budgets_us.items()):
            bucket = self._buckets.get(stage)
            row = bucket.to_dict() if bucket else {"data_status": "DATA_BLOCKED"}
            p99 = row.get("p99_us")
            over = p99 is not None and p99 > budget
            if over:
                breaching.append(stage)
            rows.append({"stage": stage, "budget_us": budget,
                         "p99_us": p99, "over_budget": over,
                         "data_status": row.get("data_status")})
        return {
            "status": "DEGRADED" if breaching else "OK",
            "detail": ("over budget at p99: " + ", ".join(breaching)
                       if breaching else ""),
            "breaching": breaching,
            "stages": rows,
        }
