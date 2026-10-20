"""Why the desk is not ready, traced back to the thing to actually fix.

The promotion gate is strict and correct: an unmeasured criterion FAILS,
because treating "we did not measure the rug-loss share" as satisfying
"rug-loss share below 15%" is how a gate becomes decorative, and it fails in
the direction that promotes.

What it cannot say is WHY a criterion is unmeasured, and that is the only
question an operator actually has. A gate reporting

    monster_enrichment was not measured; required >= 2.0
    net_log_growth was not measured
    rug_loss_share was not measured; required <= 0.2

reads like three problems. It is one problem, three levels down:

    monster_enrichment unmeasured
      because no position was ever entered
        because every entry was DATA_BLOCKED on the prediction
          because no model artifact exists
            because nothing ever ran the trainer

Three of those levels are things nobody can act on and the fourth is a
half-hour fix. A report that lists the symptoms and not the cause sends an
operator to tune thresholds when the problem is that a script has no caller
-- which is exactly what happened here, for weeks.

So this walks the chain. Every link is a fact the desk already reports; the
only thing added is the arrow between them, and the rule that the DEEPEST
unsatisfied link is the one worth naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

READINESS_SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class Link:
    """One step in the chain, and how to tell whether it is satisfied."""

    name: str
    satisfied_when: Callable[[Dict[str, Any]], bool]
    #: Said when this link is the deepest unsatisfied one -- so it names an
    #: action, not a state. "No model artifact" is a state; "the trainer has
    #: never run" is where somebody goes.
    blocked_says: str
    #: Said when this link is satisfied, for the trail above the blockage.
    satisfied_says: str = ""


def _positive(key: str) -> Callable[[Dict[str, Any]], bool]:
    def check(facts: Dict[str, Any]) -> bool:
        value = facts.get(key)
        return isinstance(value, (int, float)) and value > 0
    return check


#: The chain, shallowest LAST. Ordered so the walk reports the deepest
#: unsatisfied link, which is the one that has to move before anything above
#: it can even be measured.
DEFAULT_CHAIN: Sequence[Link] = (
    Link("stream",
         _positive("launches_seen"),
         "the desk has not seen a launch; the chain feed is not delivering, "
         "so nothing downstream can be measured",
         "launches are arriving"),
    Link("episodes",
         _positive("resolved_episodes"),
         "no launch has RESOLVED into an episode on disk; without resolved "
         "outcomes there is nothing to fit a model to",
         "episodes are resolving"),
    Link("training",
         lambda facts: bool(facts.get("training_rounds")),
         "no training round has run. Every trainer in this repository is a "
         "__main__ that had no caller, so the desk collected evidence and "
         "never fitted anything to it",
         "training has run"),
    Link("model",
         lambda facts: bool(facts.get("model_trained")),
         "no model artifact has passed its gate. The trainer's own report "
         "says which criterion it failed and by how much -- read that rather "
         "than the gate below, which can only say the result is unmeasured",
         "a model artifact is live"),
    Link("costing",
         lambda facts: bool(facts.get("cost_model_ok")),
         "trade cost is DATA_BLOCKED, so no expected value can be computed "
         "net of cost and nothing can clear an entry bar; the on-chain "
         "FeeConfig account has not been read",
         "trades can be priced"),
    Link("entries",
         _positive("entries"),
         "nothing has been entered, so every criterion about what entries "
         "DO -- enrichment, growth, rug-loss share -- is unmeasurable by "
         "construction rather than failing",
         "positions are being entered"),
    Link("fills",
         _positive("real_fills"),
         "no real fill exists. In DRY_RUN this is expected and correct: the "
         "desk is not submitting, so execution evidence cannot accumulate "
         "and the live stages cannot be reached from here at all",
         "fills are landing"),
)


@dataclass
class Readiness:
    blocked_at: Optional[str]
    reason: str
    satisfied: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.blocked_at is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": READINESS_SCHEMA_VERSION,
            "status": "OK" if self.ready else "BLOCKED",
            "blocked_at": self.blocked_at,
            "reason": self.reason,
            "satisfied": list(self.satisfied),
            "pending": list(self.pending),
            "facts": dict(self.facts),
            "detail": ("the deepest unsatisfied link in the chain from feed "
                       "to fill; the promotion gate can only say a criterion "
                       "is unmeasured, which is a symptom several levels "
                       "above whatever has to be fixed"),
        }


def diagnose(facts: Dict[str, Any],
             chain: Sequence[Link] = DEFAULT_CHAIN) -> Readiness:
    """Walk the chain and name the deepest thing that is not true yet.

    ONE reason, not a list. A list of seven failures where six are
    consequences of the seventh reads as a system in trouble everywhere,
    and sends whoever reads it to fix the wrong thing.
    """
    satisfied: List[str] = []
    for index, link in enumerate(chain):
        try:
            ok = bool(link.satisfied_when(facts))
        except Exception:  # pragma: no cover - a fact provider misbehaving
            ok = False
        if ok:
            satisfied.append(link.name)
            continue
        return Readiness(
            blocked_at=link.name,
            reason=link.blocked_says,
            satisfied=satisfied,
            pending=[item.name for item in chain[index:]],
            facts=dict(facts))
    return Readiness(
        blocked_at=None,
        reason=("every link from the feed through to real fills is satisfied; "
                "what remains is the promotion gate's own thresholds, which "
                "are now measurable rather than blocked"),
        satisfied=satisfied, pending=[], facts=dict(facts))
