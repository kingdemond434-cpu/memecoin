"""What every decision must be able to show it consulted.

Four components were reported wired and were not. The `s.replace()` anchors
that were supposed to add their imports had silently failed to match, the
tests bound fakes into the collaborator slots so the constructor was never
exercised, and the modules sat there fully built, fully unit-tested, and
completely unreachable. Nothing failed. The suite was green. The desk simply
did not use them.

That failure mode has no natural detector. A module that is never called
raises nothing, logs nothing and changes no number, so the only way it
surfaces is if somebody notices its absence -- and the whole reason it went
missing is that nobody did. Unit tests cannot catch it either, because a unit
test of an orphan passes exactly as happily as a unit test of a live
component.

So the contract is inverted. Rather than each module proving it works, every
decision proves which modules it consulted, and this manifest names the ones
it must. A decision record that omits a declared contributor is a defect --
not a gap in evidence, a gap in *wiring* -- and the two are distinguished
carefully:

    MISSING    the module never ran. This is the orphan, and it fails.
    BLOCKED    the module ran and said it could not answer. This passes.
    PRESENT    the module ran and contributed.

BLOCKED passing is the point. A source mesh with no transport, a fee schedule
whose tiers are published only as an image, a buyer cohort with no scored
wallets -- all of those are honest DATA_BLOCKED answers, and demanding a
number from them is how fabricated inputs get introduced. What is not
acceptable is silence, because silence and "contributed nothing" are
indistinguishable from the outside, and one of them is a bug.

The manifest is also a runtime surface, not only a test fixture. Coverage per
module goes into the weekly audit pack, so a component that stops being
consulted -- because a branch was reordered, an exception got swallowed, a
config key was renamed -- shows up as its contribution rate falling to zero
while every test still passes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "v1"

PRESENT = "PRESENT"
BLOCKED = "BLOCKED"
MISSING = "MISSING"

# Statuses a contributor may report that mean "consulted, could not answer".
_BLOCKED_PREFIXES = ("DATA_BLOCKED", "BLOCKED", "NOT_MEASURED", "UNAVAILABLE",
                     "NO_FETCHER", "MEASURING", "REJECTED")


@dataclass(frozen=True)
class Contributor:
    """One module that must be visible in a decision of the given stage.

    ``key`` is the slot it occupies in the record's ``intelligence`` map. It is
    a flat key on purpose: a nested path invites the record to be assembled in
    several places, and a contributor that can be written from several places
    can be dropped from one of them without the others noticing.
    """

    module: str
    key: str
    why: str
    optional: bool = False


# Everything that must reach an ENTRY decision. Ordered roughly by how early
# in the pipeline it speaks, which is also the order an operator reads them.
ENTRY_CONTRIBUTORS: Tuple[Contributor, ...] = (
    Contributor("chains.rug_detector", "safety",
                "Mint and pool authorities decide whether the token can be "
                "confiscated after we buy it."),
    Contributor("strategies.rug_hazard", "hazard",
                "The competing-risk hazard at entry is the denominator every "
                "forward return is discounted by."),
    Contributor("strategies.multihead_predictor", "prediction",
                "The forward distribution itself. Absent, nothing downstream "
                "has anything to price."),
    Contributor("strategies.actor_graph", "actors",
                "Who is buying, whether they are independent, and whether the "
                "cohort is a swarm or one entity wearing forty wallets."),
    Contributor("collectors.event_source", "sources",
                "Which public source spoke first and how late we were to it -- "
                "the only honest measure of our information latency."),
    Contributor("strategies.source_genealogy", "source_dna",
                "Whether that source's posts have historically been tradeable "
                "or merely a place distributors advertise."),
    Contributor("strategies.authenticity", "authenticity",
                "Whether the token is the entity it claims to be. A copycat "
                "priced as the real thing is the most expensive miss there is."),
    Contributor("execution.pump_fees", "cost_model",
                "The fee actually charged at this market cap and date, not a "
                "constant that stopped being true on the first of September."),
    Contributor("strategies.prelaunch_intent", "prelaunch",
                "What the deployer did before the mint existed."),
    Contributor("strategies.public_coordination", "coordination",
                "Whether a public group is being used to organise the buy side."),
    Contributor("strategies.social_intelligence", "social",
                "Public attention, and how much of it arrived before we did."),
    Contributor("strategies.information_graph", "information",
                "Which wallets lead which, so a follower's buy is not counted "
                "as independent confirmation of a leader's."),
    Contributor("strategies.opportunity_allocator", "opportunity",
                "Whether this is the best use of the next dollar, not merely "
                "an acceptable one."),
    Contributor("strategies.reentry", "reentry",
                "Whether we have exited this token before, and what that "
                "should cost us to undo."),
    Contributor("strategies.mega_event", "mega_event",
                "Whether capital is being withheld for a rare authenticated event."),
    Contributor("strategies.champion_challenger", "authority",
                "Which model holds trading authority, and at what promotion stage."),
    Contributor("strategies.elogw_engine", "sizing",
                "The size, and the risk-constrained Kelly reasoning behind it."),
)

# Everything that must reach a decision about an OPEN position.
POSITION_CONTRIBUTORS: Tuple[Contributor, ...] = (
    Contributor("strategies.distribution", "distribution",
                "Whether the early holders are handing the token to us."),
    Contributor("strategies.monster", "monster",
                "Whether this is the rare one that should not be ratcheted away."),
    Contributor("strategies.escape", "escape",
                "The probability the exit lands before the collapse it is "
                "running from."),
    Contributor("strategies.escape", "hazard_mechanisms",
                "WHICH way the position dies. Speed answers a seller and does "
                "not answer a frozen mint, so the race is only meaningful "
                "once the hazard is decomposed."),
    Contributor("strategies.escape", "exit_latency",
                "How long our sells actually take to land. A constant here "
                "prices the race we usually run, not the one we run while "
                "something is collapsing."),
    Contributor("execution.tradeability", "exit_capacity",
                "What share of the position the venue can actually absorb."),
    Contributor("strategies.action_value", "action_value",
                "Every move priced against one forward distribution."),
    Contributor("strategies.exit_policy", "ratchet",
                "The fallback threshold policy, for states the action-value "
                "engine cannot price."),
    Contributor("strategies.rug_hazard", "hazard",
                "The hazard, refreshed -- an exit decided on entry-time hazard "
                "is decided on evidence since contradicted."),
)

_STAGES: Mapping[str, Tuple[Contributor, ...]] = {
    "entry": ENTRY_CONTRIBUTORS,
    "position": POSITION_CONTRIBUTORS,
}


def classify(evidence: Any) -> str:
    """PRESENT, BLOCKED or MISSING for one contributor's slot.

    ``None`` is MISSING and an empty mapping is MISSING, because both are what
    a slot looks like when nothing ever wrote to it. A module that genuinely
    has nothing to say must say so -- ``{"status": "DATA_BLOCKED", ...}`` --
    rather than leaving the slot the way an unwired module leaves it.
    """
    if evidence is None:
        return MISSING
    if isinstance(evidence, Mapping):
        if not evidence:
            return MISSING
        status = str(evidence.get("status", "")).upper()
        if not status:
            return PRESENT
        if any(status.startswith(prefix) for prefix in _BLOCKED_PREFIXES):
            return BLOCKED
        return PRESENT
    if isinstance(evidence, str):
        status = evidence.upper()
        if not status:
            return MISSING
        return BLOCKED if any(status.startswith(p) for p in _BLOCKED_PREFIXES) else PRESENT
    return PRESENT


@dataclass
class ContributionReport:
    stage: str
    verdicts: Dict[str, str] = field(default_factory=dict)
    unknown_keys: List[str] = field(default_factory=list)

    @property
    def orphans(self) -> List[str]:
        """Declared contributors that never ran. Each one is a wiring defect."""
        return sorted(key for key, verdict in self.verdicts.items() if verdict == MISSING)

    @property
    def blocked(self) -> List[str]:
        return sorted(key for key, verdict in self.verdicts.items() if verdict == BLOCKED)

    @property
    def contributing(self) -> List[str]:
        return sorted(key for key, verdict in self.verdicts.items() if verdict == PRESENT)

    @property
    def ok(self) -> bool:
        return not self.orphans

    @property
    def coverage(self) -> float:
        """Share of declared contributors that were consulted at all.

        Deliberately counts BLOCKED as covered. Coverage measures wiring, and
        conflating it with evidence would make the honest answer -- "this
        cannot be measured" -- look like a broken connection.
        """
        if not self.verdicts:
            return 0.0
        consulted = len(self.verdicts) - len(self.orphans)
        return consulted / len(self.verdicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA_VERSION, "stage": self.stage,
            "coverage": self.coverage, "orphans": self.orphans,
            "blocked": self.blocked, "contributing": self.contributing,
            "unknown_keys": sorted(self.unknown_keys),
        }


def audit(stage: str, intelligence: Optional[Mapping[str, Any]]) -> ContributionReport:
    """Check one decision's intelligence map against the manifest for its stage."""
    contributors = _STAGES.get(stage)
    if contributors is None:
        raise KeyError(f"unknown decision stage {stage!r}")
    record = intelligence or {}
    report = ContributionReport(stage=stage)
    for contributor in contributors:
        report.verdicts[contributor.key] = classify(record.get(contributor.key))
    declared = {contributor.key for contributor in contributors}
    # Extra keys are reported rather than ignored: a slot nothing declares is
    # either a contributor somebody forgot to add here, or a typo that means
    # the real slot is empty. Both are worth seeing.
    report.unknown_keys = [key for key in record if key not in declared]
    return report


class CoverageTracker:
    """Contribution rates across many decisions, for the weekly audit pack.

    The signal an operator actually needs is not one decision's coverage but
    the trend: a module whose contribution rate falls from 90% to 0% between
    two audit packs has been disconnected, and no test will say so.
    """

    def __init__(self, stage: str):
        if stage not in _STAGES:
            raise KeyError(f"unknown decision stage {stage!r}")
        self.stage = stage
        self.decisions = 0
        self._counts: Dict[str, Dict[str, int]] = {
            contributor.key: {PRESENT: 0, BLOCKED: 0, MISSING: 0}
            for contributor in _STAGES[stage]
        }

    def record(self, intelligence: Optional[Mapping[str, Any]]) -> ContributionReport:
        report = audit(self.stage, intelligence)
        self.decisions += 1
        for key, verdict in report.verdicts.items():
            self._counts[key][verdict] += 1
        if report.orphans:
            logger.warning("decision reached with orphaned intelligence: %s",
                           ", ".join(report.orphans))
        return report

    def report(self) -> Dict[str, Any]:
        if not self.decisions:
            return {"schema": MANIFEST_SCHEMA_VERSION, "stage": self.stage,
                    "status": "DATA_BLOCKED", "decisions": 0,
                    "detail": "no decisions observed yet"}
        rates = {
            key: {
                "contributed": counts[PRESENT] / self.decisions,
                "blocked": counts[BLOCKED] / self.decisions,
                "orphaned": counts[MISSING] / self.decisions,
            }
            for key, counts in self._counts.items()
        }
        orphaned = sorted(key for key, counts in self._counts.items() if counts[MISSING])
        return {
            "schema": MANIFEST_SCHEMA_VERSION, "stage": self.stage,
            "status": "OK" if not orphaned else "ORPHANED",
            "decisions": self.decisions, "orphaned": orphaned, "rates": rates,
        }
