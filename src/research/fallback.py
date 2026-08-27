"""Never blocked, never pretending: a graded ladder of ways to know something.

DATA_BLOCKED is the right answer to "is this measured", and the wrong answer to
"what should we do now". A desk that refuses to act without a primary
measurement stands still through most of the opportunity set, because on a
thirty-second-old launch most facts have no primary measurement yet. But a
desk that substitutes a guess and forgets it was a guess is worse: it sizes a
proxy as though it were a reading, which is how a system with excellent
discipline everywhere else still blows up.

So every fact this desk needs is declared as a LADDER. Each rung is a
different way of knowing, ordered by how much it deserves to be trusted:

  MEASURED      read directly from the chain or from the venue. The thing
                itself, now.
  CORROBORATED  read from an independent second source that agrees. Slightly
                behind the primary in latency, ahead of it in confidence.
  RECONSTRUCTED derived from history -- the same deployer's last forty
                launches, the same funder's cluster. Real evidence about a
                real actor, but about their past rather than about this token.
  PROXY         a different quantity that correlates. Holder concentration
                stood in for by the First25 buy distribution. Genuinely
                informative, genuinely not the thing asked for.
  PRIOR         the base rate over the whole population. Always available,
                almost never specific.
  ABSENT        nothing at all. Rare once a ladder is built, and reported.

Every resolution carries which rung answered it, and a CONFIDENCE MULTIPLIER
that falls as the rung does. That multiplier is not decoration: it flows into
the same shrink the disagreement model already applies, so acting on a prior
produces a position a fraction of the size that acting on a measurement does.
The desk therefore always has an answer and never forgets what kind of answer
it is.

Two rules make this safe rather than merely convenient.

**A rung must be independently checkable.** A ladder whose fallback is "the
last value we saw" is a ladder that reports stale data as fresh. Staleness is
a property of the rung, and a rung past its own freshness window is skipped.

**Confidence multiplies, it never resets.** Three proxies do not add up to a
measurement. Resolving from a low rung shrinks the position and nothing
downstream can talk it back up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FALLBACK_SCHEMA_VERSION = "v1"


class Rung(Enum):
    """How we know it. Ordered best to worst; the order is load-bearing."""

    MEASURED = "measured"
    CORROBORATED = "corroborated"
    RECONSTRUCTED = "reconstructed"
    PROXY = "proxy"
    PRIOR = "prior"
    ABSENT = "absent"


#: How much of the position a fact resolved at each rung justifies. A prior is
#: worth acting on at a tenth of the size a measurement is; ABSENT is worth
#: nothing, which is what makes "we have a ladder" different from "we always
#: have an answer".
RUNG_CONFIDENCE: Dict[Rung, float] = {
    Rung.MEASURED: 1.00,
    Rung.CORROBORATED: 0.90,
    Rung.RECONSTRUCTED: 0.60,
    Rung.PROXY: 0.35,
    Rung.PRIOR: 0.10,
    Rung.ABSENT: 0.0,
}


@dataclass
class Source:
    """One way of learning one fact."""

    name: str
    rung: Rung
    fetch: Callable[[Dict[str, Any]], Optional[Any]]
    #: Beyond this age a value from this source is not used. A ladder whose
    #: fallback is "whatever we saw last" reports stale data as fresh, which is
    #: the failure this whole module is supposed to prevent.
    max_age_s: Optional[float] = None
    detail: str = ""


@dataclass
class Resolution:
    """A value, how it was obtained, and how much it should be sized on."""

    fact: str
    value: Any = None
    rung: Rung = Rung.ABSENT
    source: str = ""
    detail: str = ""
    age_s: Optional[float] = None
    #: Which rungs were tried and why each declined. The operator-facing half:
    #: a fact resolving at PRIOR when a MEASURED rung exists is a broken
    #: source, not a quiet market.
    attempted: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return RUNG_CONFIDENCE.get(self.rung, 0.0)

    @property
    def usable(self) -> bool:
        return self.rung is not Rung.ABSENT

    @property
    def data_status(self) -> str:
        """OK only for a real reading. Everything else names its own rung, so
        a caller logging this cannot accidentally record a prior as a fact."""
        if self.rung in (Rung.MEASURED, Rung.CORROBORATED):
            return "OK"
        if self.rung is Rung.ABSENT:
            return "DATA_BLOCKED"
        return f"DEGRADED:{self.rung.value}"

    def to_dict(self) -> Dict[str, Any]:
        return {"fact": self.fact, "value": self.value, "rung": self.rung.value,
                "source": self.source, "detail": self.detail,
                "confidence": self.confidence, "data_status": self.data_status,
                "age_s": self.age_s,
                "attempted": [{"source": name, "why": why}
                              for name, why in self.attempted]}


class FactLadder:
    """The declared ways of knowing one fact, in order."""

    def __init__(self, fact: str, sources: Sequence[Source]):
        self.fact = fact
        # Sorted defensively: a ladder declared out of order would silently
        # prefer a proxy to a measurement, and nothing would look wrong.
        self.sources = sorted(sources, key=lambda item: list(Rung).index(item.rung))

    def resolve(self, context: Optional[Dict[str, Any]] = None,
                now: Optional[float] = None) -> Resolution:
        context = dict(context or {})
        moment = time.time() if now is None else now
        attempted: List[Tuple[str, str]] = []
        for source in self.sources:
            try:
                answer = source.fetch(context)
            except Exception as exc:
                attempted.append((source.name, f"{type(exc).__name__}: {exc}"))
                continue
            if answer is None:
                attempted.append((source.name, "no value"))
                continue
            value, age = self._unpack(answer, moment)
            if source.max_age_s is not None and age is not None and age > source.max_age_s:
                attempted.append((
                    source.name,
                    f"stale: {age:.1f}s old, limit {source.max_age_s:.1f}s"))
                continue
            return Resolution(fact=self.fact, value=value, rung=source.rung,
                              source=source.name, detail=source.detail,
                              age_s=age, attempted=attempted)
        return Resolution(fact=self.fact, rung=Rung.ABSENT, attempted=attempted,
                          detail="every rung declined; this fact is genuinely "
                                 "unavailable rather than merely degraded")

    @staticmethod
    def _unpack(answer: Any, now: float) -> Tuple[Any, Optional[float]]:
        """A source may return a bare value or (value, observed_at)."""
        if (isinstance(answer, tuple) and len(answer) == 2
                and isinstance(answer[1], (int, float))):
            return answer[0], max(0.0, now - float(answer[1]))
        return answer, None


class FallbackResolver:
    """Every fact's ladder, and what the ladders have actually been doing."""

    def __init__(self) -> None:
        self._ladders: Dict[str, FactLadder] = {}
        self._counts: Dict[str, Dict[str, int]] = {}

    def declare(self, fact: str, sources: Sequence[Source]) -> None:
        self._ladders[fact] = FactLadder(fact, sources)
        self._counts.setdefault(fact, {})

    def resolve(self, fact: str, context: Optional[Dict[str, Any]] = None,
                now: Optional[float] = None) -> Resolution:
        ladder = self._ladders.get(fact)
        if ladder is None:
            return Resolution(fact=fact, rung=Rung.ABSENT,
                              detail="no ladder declared for this fact")
        resolution = ladder.resolve(context, now)
        bucket = self._counts.setdefault(fact, {})
        bucket[resolution.rung.value] = bucket.get(resolution.rung.value, 0) + 1
        return resolution

    def resolve_many(self, facts: Sequence[str],
                     context: Optional[Dict[str, Any]] = None,
                     now: Optional[float] = None) -> Dict[str, Resolution]:
        return {fact: self.resolve(fact, context, now) for fact in facts}

    @staticmethod
    def combined_confidence(resolutions: Sequence[Resolution]) -> float:
        """One multiplier for a decision resting on several resolved facts.

        The WEAKEST rung dominates, and the product pulls it further down.
        Three proxies do not add up to a measurement: a decision resting on a
        prior is a decision made in the dark regardless of how many other
        facts were read cleanly, and averaging would let the clean readings
        vote the dark one up.
        """
        rows = [item for item in resolutions if item is not None]
        if not rows:
            return 0.0
        weakest = min(item.confidence for item in rows)
        product = 1.0
        for item in rows:
            product *= item.confidence
        # Geometric mean keeps the product from collapsing to nothing on a long
        # fact list, but the weakest rung still caps the result.
        geometric = product ** (1.0 / len(rows))
        return min(weakest, geometric) if weakest < 1.0 else geometric

    def report(self) -> Dict[str, Any]:
        """Which facts are actually being measured, and which are guesses.

        The line that matters is `degraded_facts`: a fact that resolves at
        PRIOR while a MEASURED rung is declared for it is a broken source
        wearing the appearance of a quiet market.
        """
        rows = []
        degraded = []
        for fact, counts in sorted(self._counts.items()):
            total = sum(counts.values())
            if not total:
                rows.append({"fact": fact, "resolutions": 0,
                             "rungs": {}, "measured_share": None})
                continue
            measured = (counts.get(Rung.MEASURED.value, 0)
                        + counts.get(Rung.CORROBORATED.value, 0))
            share = measured / total
            rows.append({"fact": fact, "resolutions": total,
                         "rungs": dict(sorted(counts.items())),
                         "measured_share": round(share, 4),
                         "absent": counts.get(Rung.ABSENT.value, 0)})
            if share < 0.5:
                degraded.append(fact)
        return {
            "schema": FALLBACK_SCHEMA_VERSION,
            "status": "OK" if not degraded else "DEGRADED",
            "detail": ("" if not degraded else
                       "these facts are usually being inferred rather than "
                       "read: " + ", ".join(degraded)),
            "facts_declared": len(self._ladders),
            "degraded_facts": degraded,
            "facts": rows,
        }
