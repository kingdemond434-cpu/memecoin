"""Which evidence actually moved the decision, and by how much.

Coverage says a module was consulted. It does not say the module mattered.
Those are different failures with the same symptom: a component that is
disconnected and a component that is connected but contributes nothing both
show up as trades that would have happened anyway, and only one of them is
fixed by wiring.

So every action is also scored counterfactually. For each input, the decision
is re-priced with that input replaced by its BASELINE -- what we would have
believed without the measurement -- and the difference in the chosen action's
Q is that input's contribution. The sign matters as much as the size: an input
that consistently pushes toward worse actions is worse than one that does
nothing, and averaging absolute values would hide exactly that.

The baseline is the load-bearing choice and it is never the flattering one.

For escape and capacity the baseline is 1.0 -- fully escapable, fully
sellable -- because that is what the desk would have assumed before it learned
to measure them, and it is the assumption that makes every trapped position
look liquid. Measuring them can therefore only ever reduce Q, and the size of
that reduction is precisely their worth: it is the amount of optimism they
removed. An input whose contribution is zero here is not neutral, it is
telling us the position was genuinely liquid and the measurement bought
nothing on this trade.

For the alternative-growth rate the baseline is None -- no opportunity cost --
because a desk that cannot rank cross-sectionally charges none.

For the replacement candidate and the add size the baseline is absence: those
actions simply were not available.

Aggregated over many decisions this answers a question nothing else does.
A component whose contribution distribution is a spike at zero is not earning
its latency, whatever its coverage says. One whose contribution is large and
negative on trades that lost money is earning it loudly.

This is attribution over DECISIONS, not over model features. Feature
importance says what a model leaned on; this says what the desk would have
done differently, which is the only version of the question capital cares
about.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONTRIBUTION_SCHEMA_VERSION = "v1"

# What we would have believed about each input without measuring it. Never the
# pessimistic value: a baseline that assumed the worst would make every
# measurement look like it added optimism, which is backwards.
ABLATIONS: Tuple[Tuple[str, Any], ...] = (
    ("escape_probability", 1.0),
    ("exit_capacity_ratio", 1.0),
    ("alternative_growth_per_second", None),
    ("replacement_bins", None),
    ("add_fraction", None),
    ("reentry_bins", None),
)


@dataclass
class Contribution:
    """One input's effect on one decision."""

    component: str
    delta_q: Optional[float]
    baseline_action: str = ""
    changed_action: bool = False
    status: str = "OK"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"component": self.component, "delta_q": self.delta_q,
                "baseline_action": self.baseline_action,
                "changed_action": self.changed_action,
                "status": self.status, "detail": self.detail}


@dataclass
class DecisionContribution:
    token: str
    action: str
    q: float
    contributions: List[Contribution] = field(default_factory=list)

    @property
    def decisive(self) -> List[str]:
        """Inputs without which a DIFFERENT action would have been taken.

        The strongest statement available about an input: not that it shifted
        a number, but that the desk would have done something else."""
        return sorted(item.component for item in self.contributions
                      if item.changed_action)

    def to_dict(self) -> Dict[str, Any]:
        return {"token": self.token, "action": self.action, "q": self.q,
                "decisive": self.decisive,
                "contributions": [item.to_dict() for item in self.contributions]}


def action_value_contributions(policy: Any, state: Any, *,
                               token: str = "") -> Optional[DecisionContribution]:
    """Leave-one-out contribution of every measured input to one action choice.

    Re-scores the state once per input. That is a handful of pure arithmetic
    evaluations over the same bins, so it is cheap enough to run on every
    decision rather than on a sample -- which matters, because the decisions
    worth attributing are the rare ones and a sample misses them.
    """
    chosen = policy.score(state)
    if chosen.status != "OK":
        return None
    result = DecisionContribution(token=token, action=chosen.action.value, q=chosen.q)
    for field_name, baseline in ABLATIONS:
        current = getattr(state, field_name, None)
        if current is None or current == baseline:
            # Never measured on this decision, so it cannot have contributed.
            # Reported rather than omitted: "did not contribute" and "was not
            # present" are different sentences about a component.
            result.contributions.append(Contribution(
                component=field_name, delta_q=None, status="NOT_MEASURED",
                detail="input absent from this decision"))
            continue
        ablated = dataclass_replace(state, **{field_name: baseline})
        without = policy.score(ablated)
        if without.status != "OK":
            result.contributions.append(Contribution(
                component=field_name, delta_q=None, status="DATA_BLOCKED",
                detail=f"unpriceable without this input: {without.detail}"))
            continue
        # Compare the chosen action's value under both, so the delta is about
        # the DECISION and not about whichever action happened to win each
        # time. Whether the winner changed is reported separately.
        with_q = chosen.score_of(chosen.action)
        without_q = without.score_of(chosen.action)
        delta = (None if with_q is None or without_q is None else with_q - without_q)
        result.contributions.append(Contribution(
            component=field_name,
            delta_q=None if delta is None else float(delta),
            baseline_action=without.action.value,
            changed_action=without.action is not chosen.action,
            status="OK" if delta is not None else "DATA_BLOCKED",
        ))
    return result


@dataclass
class GateFlip:
    """A gate that changed an entry decision, recorded as it happened.

    Some components do not shift a number, they veto. The re-entry premium,
    the authenticity floor, the mega-event reserve and the capital contest all
    turn a yes into a no or back again, and a Q-delta cannot see that at all.
    """

    gate: str
    token: str
    before: bool
    after: bool
    reason: str = ""

    @property
    def flipped(self) -> bool:
        return self.before != self.after


class ContributionLedger:
    """Contribution distributions per component, across many decisions.

    One decision's attribution is an anecdote. The distribution is the
    finding: a component whose contribution is a spike at zero is not earning
    its latency however good its coverage looks, and one that is decisive on
    one trade in fifty is earning it more than one that nudges every trade.
    """

    def __init__(self, capacity: int = 4_096):
        self.capacity = max(1, int(capacity))
        self.decisions = 0
        self._deltas: Dict[str, List[float]] = defaultdict(list)
        self._decisive: Dict[str, int] = defaultdict(int)
        self._not_measured: Dict[str, int] = defaultdict(int)
        self._gate_flips: Dict[str, int] = defaultdict(int)
        self._gate_seen: Dict[str, int] = defaultdict(int)

    def record(self, contribution: Optional[DecisionContribution]) -> None:
        if contribution is None:
            return
        self.decisions += 1
        for item in contribution.contributions:
            if item.status == "NOT_MEASURED":
                self._not_measured[item.component] += 1
                continue
            if item.delta_q is None or not math.isfinite(item.delta_q):
                continue
            samples = self._deltas[item.component]
            samples.append(float(item.delta_q))
            if len(samples) > self.capacity:
                del samples[0]
            if item.changed_action:
                self._decisive[item.component] += 1

    def record_gate(self, flip: GateFlip) -> None:
        self._gate_seen[flip.gate] += 1
        if flip.flipped:
            self._gate_flips[flip.gate] += 1

    def report(self) -> Dict[str, Any]:
        if not self.decisions and not self._gate_seen:
            return {"schema": CONTRIBUTION_SCHEMA_VERSION, "status": "DATA_BLOCKED",
                    "decisions": 0, "detail": "no decisions attributed yet"}
        components: Dict[str, Any] = {}
        for component, samples in self._deltas.items():
            array = np.asarray(samples, dtype=float)
            components[component] = {
                "observations": int(array.size),
                "mean_delta_q": float(array.mean()),
                "median_delta_q": float(np.median(array)),
                # The share of decisions where the input moved the objective by
                # something the policy could actually distinguish from noise.
                "share_nonzero": float(np.mean(np.abs(array) > 1e-9)),
                "decisive_decisions": self._decisive.get(component, 0),
                "not_measured": self._not_measured.get(component, 0),
            }
        for component, count in self._not_measured.items():
            components.setdefault(component, {
                "observations": 0, "mean_delta_q": None, "median_delta_q": None,
                "share_nonzero": 0.0, "decisive_decisions": 0,
                "not_measured": count,
            })
        inert = sorted(name for name, stats in components.items()
                       if stats["observations"] and not stats["share_nonzero"])
        return {
            "schema": CONTRIBUTION_SCHEMA_VERSION, "status": "OK",
            "decisions": self.decisions,
            "components": components,
            # Consulted on every decision and never changed one. Not the same
            # as disconnected, and not the same as working either.
            "inert_components": inert,
            "gates": {gate: {"evaluated": count,
                             "flipped": self._gate_flips.get(gate, 0)}
                      for gate, count in self._gate_seen.items()},
        }
