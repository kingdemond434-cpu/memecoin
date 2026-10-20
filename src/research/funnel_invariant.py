"""Exactly zero is a different fact from low, and only one of them is a bug.

The desk ran for 8.76 days, took 150,278 decisions across 14,097 launches, and
entered nothing. Every dashboard was green. Each individual number looked like
a strict policy doing its job, and the one observation that would have named
the defect in a sentence was never made: a stage with a hundred launches
upstream of it and EXACTLY ZERO in it is not a selective filter, it is a
disconnected wire.

That is the whole of this module. It walks the money path in order:

    seen -> decision_ready -> decided -> entered -> closed -> measured

and for each consecutive pair asks one question: did enough arrive upstream
that downstream could not still be zero by chance? A filter that passes 1 in
500 is a policy; a filter that has passed 0 of 100,000 is a wire. The
threshold is what separates them, and it is deliberately low, because the cost
of asking early is a line of output and the cost of asking late is a fortnight
of evidence the promotion ladder cannot use.

Three stages downstream of `entered` matter as much as the entry itself,
because the promotion criteria are ratios over CLOSED positions:
`net_log_growth` is None exactly when nothing has been entered, and
`rug_loss_share` and `monster_enrichment` are ratios over an empty
denominator. A desk that enters and never closes is as stuck as one that never
enters, and looks healthier.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FUNNEL_INVARIANT_SCHEMA_VERSION = "v1"

#: Upstream arrivals above which a downstream zero stops being explainable by
#: a strict policy. Low on purpose: the question is cheap and the alternative
#: is discovering the break after a fortnight of unusable evidence.
DEFAULT_BREAK_THRESHOLD = 100

#: The money path, in order. Each entry is (name, upstream, downstream).
#: `measured` is included because three promotion criteria are ratios over
#: closed positions, so a desk that enters and never closes is as stuck as one
#: that never enters -- and looks healthier.
STAGE_PATH: Tuple[Tuple[str, str], ...] = (
    ("seen", "decision_ready"),
    ("decision_ready", "decided"),
    ("decided", "entered"),
    ("entered", "closed"),
    ("closed", "measured"),
)


@dataclass
class StageBreak:
    """One link of the path that is not carrying anything."""

    upstream: str
    downstream: str
    upstream_count: int
    downstream_count: int
    threshold: int
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunnelVerdict:
    """Whether the money path is connected, and where it is not."""

    status: str = "DATA_BLOCKED"
    counts: Dict[str, int] = field(default_factory=dict)
    breaks: List[StageBreak] = field(default_factory=list)
    threshold: int = DEFAULT_BREAK_THRESHOLD
    detail: str = ""

    @property
    def connected(self) -> bool:
        return self.status == "OK" and not self.breaks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": FUNNEL_INVARIANT_SCHEMA_VERSION,
            "status": self.status,
            "connected": self.connected,
            "threshold": self.threshold,
            "counts": dict(self.counts),
            "breaks": [item.to_dict() for item in self.breaks],
            "detail": self.detail,
        }


def stage_counts(census_report: Optional[Dict[str, Any]],
                 evidence_report: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, int]:
    """The five counts the path is walked over, from reports already produced.

    `decided` is the launches that reached ANY decision, entered or not, and
    `entered` is the census's own entry counter rather than the ledger's --
    they are different measurements of the same event and a disagreement
    between them is itself worth seeing.
    """
    funnel = ((census_report or {}).get("funnel") or {})
    outcomes = ((census_report or {}).get("outcomes") or {})
    evidence = evidence_report or {}
    decision_ready = int(funnel.get("decision_ready", 0) or 0)
    # A launch that has already moved on from DECISION_READY is not sitting in
    # that disposition any more, so the live count understates how many ever
    # reached it. Everything downstream of it did pass through it.
    reached_decision = int(funnel.get("reached_a_decision", 0) or 0)
    entered = int(funnel.get("entered", 0) or 0)
    closed = int(evidence.get("entered", evidence.get("closed_positions", 0)) or 0)
    measured = int(evidence.get("net_log_growth") is not None)
    return {
        "seen": int(funnel.get("seen", 0) or 0),
        "decision_ready": decision_ready + reached_decision + entered,
        "decided": reached_decision + entered,
        "entered": entered,
        "closed": closed,
        "measured": measured,
    }


def check(census_report: Optional[Dict[str, Any]],
          evidence_report: Optional[Dict[str, Any]] = None, *,
          threshold: int = DEFAULT_BREAK_THRESHOLD) -> FunnelVerdict:
    """Walk the money path and name every link carrying nothing."""
    counts = stage_counts(census_report, evidence_report)
    verdict = FunnelVerdict(counts=counts, threshold=int(threshold))
    if not counts.get("seen"):
        verdict.detail = ("no launch has been seen; the path has not been "
                          "exercised and there is nothing to conclude")
        return verdict
    verdict.status = "OK"
    for upstream, downstream in STAGE_PATH:
        above = int(counts.get(upstream, 0))
        below = int(counts.get(downstream, 0))
        if above < threshold or below > 0:
            continue
        verdict.breaks.append(StageBreak(
            upstream=upstream, downstream=downstream,
            upstream_count=above, downstream_count=below,
            threshold=int(threshold),
            detail=(f"{above} launches reached {upstream} and exactly zero "
                    f"reached {downstream}; a filter that has passed none of "
                    f"{above} is a wire, not a policy")))
    if verdict.breaks:
        verdict.detail = (f"{len(verdict.breaks)} link(s) of the money path "
                          f"are carrying nothing")
    else:
        verdict.detail = "every exercised link of the money path is carrying"
    return verdict


def check_desk(desk: Any, *, threshold: int = DEFAULT_BREAK_THRESHOLD
               ) -> FunnelVerdict:
    """The same question asked of a live desk rather than of two reports."""
    census = getattr(desk, "launch_census", None)
    evidence = getattr(desk, "forward_evidence", None)
    try:
        census_report = census.report() if census is not None else None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("census report unavailable: %s", exc)
        census_report = None
    evidence_report: Optional[Dict[str, Any]] = None
    if evidence is not None:
        # Read the counters directly: the ledger's own report is a promotion
        # document and its shape is allowed to change for reasons that have
        # nothing to do with this question.
        evidence_report = {
            "entered": int(getattr(evidence, "entered", 0) or 0),
            "net_log_growth": getattr(evidence, "net_log_growth", None),
        }
        if not evidence_report["entered"]:
            evidence_report["net_log_growth"] = None
    return check(census_report, evidence_report, threshold=threshold)
