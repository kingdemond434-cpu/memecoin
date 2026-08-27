"""One action-value function over every move a position can make.

Until now the decisions were split across four components that could disagree.
`should_trade` opened, `plan_scale_in` added, `evaluate_exit` ratcheted and
trailed, and `MonsterStateMachine` overrode the ratchet. Each was individually
defensible and collectively they had no shared objective: the monster detector
could be right about a 30x while the ratchet, reasoning from a different
quantity, sold it at 2x. A detector and an unrelated exit rule is exactly how
that happens, and no amount of improving either one fixes it.

So every action is scored on one axis:

    Q(a | s) = E[log W after taking a] - E[log W after holding]

Hold is the baseline and scores exactly zero by construction. An action is
taken when it beats the baseline, which means "do nothing" wins ties and a
policy that cannot tell the difference does not churn the book.

The actions are the ones a position can actually take:

    HOLD, ADD, BANK_10/25/50/75, EXIT, REENTER, REPLACE

Three properties are load-bearing.

Every action prices the same forward distribution. Banking 25% and adding are
evaluated against one executable-return distribution at one instant, not
against a threshold on price and a Kelly fraction computed separately. That is
the whole point: two components cannot disagree about a number they both read
from the same place.

Banking is not free. Selling a slice pays exit cost and gives up that slice's
share of the forward distribution, and both are charged. A bank that looked
free would always beat holding, and a policy that banks on every tick converts
a runner into a fee schedule.

Opportunity cost is inside the objective, not bolted on. EXIT and REPLACE
credit the freed capital with what it earns elsewhere over the horizon the
hold would have consumed -- the same cross-sectional quantity
`OpportunityAllocator` ranks on, so the two agree by construction.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ACTION_VALUE_SCHEMA_VERSION = "v1"


class Action(Enum):
    # Entry is part of the same decision, not a separate one. A desk whose
    # entry hurdle and exit policy are different objects will, sooner or
    # later, buy something its own exit policy would sell -- and the two will
    # both be individually defensible while doing it.
    IGNORE = "ignore"
    PROBE = "probe"
    HOLD = "hold"
    ADD = "add"
    BANK_10 = "bank_10"
    BANK_25 = "bank_25"
    BANK_50 = "bank_50"
    BANK_75 = "bank_75"
    EXIT = "exit"
    REENTER = "reenter"
    REPLACE = "replace"

    @property
    def bank_fraction(self) -> float:
        """Share of the position this action sells. Zero for non-selling actions."""
        return {
            Action.BANK_10: 0.10, Action.BANK_25: 0.25,
            Action.BANK_50: 0.50, Action.BANK_75: 0.75,
            Action.EXIT: 1.00, Action.REPLACE: 1.00,
        }.get(self, 0.0)

    @property
    def is_entry(self) -> bool:
        """Actions only available when nothing is held."""
        return self in {Action.IGNORE, Action.PROBE}

    @property
    def frees_capital(self) -> bool:
        """Whether the freed capital can be redeployed this cycle.

        A partial bank frees cash but leaves the position open, so its
        remaining exposure still occupies the slot. Only a full exit releases
        the slot itself, which is what the opportunity-cost term prices.
        """
        return self in {Action.EXIT, Action.REPLACE}


@dataclass
class PositionState:
    """Everything an action needs to be priced, at one instant.

    Every field that could be unobserved is Optional. Nothing here defaults to
    a number: an action chosen because an unmeasured input read as zero is the
    same class of error as fabricating the input.
    """

    held_fraction: float
    current_multiple: float
    # Disjoint (probability, gross return) bins for the forward distribution
    # from HERE -- not from entry. What the position has already made is sunk.
    forward_bins: Sequence[Tuple[float, float]] = ()
    exit_cost: float = 0.0
    entry_cost: float = 0.0
    exit_capacity_ratio: Optional[float] = None
    escape_probability: Optional[float] = None
    expected_remaining_seconds: Optional[float] = None
    # Best alternative growth per dollar per second, from the allocator.
    alternative_growth_per_second: Optional[float] = None
    add_fraction: Optional[float] = None
    add_capacity_fraction: Optional[float] = None
    # The small first commitment a flat book can make. Distinct from ADD,
    # which prices adding to something already held: a probe buys the right
    # to observe the position's own fills, which is information no amount of
    # watching from outside produces.
    probe_fraction: Optional[float] = None
    reentry_bins: Optional[Sequence[Tuple[float, float]]] = None
    # A specific better candidate this position could be swapped into, with
    # its own forward distribution and size. Supplied by the allocator.
    replacement_bins: Optional[Sequence[Tuple[float, float]]] = None
    replacement_fraction: Optional[float] = None

    def blocked_reason(self) -> Optional[str]:
        if not self.forward_bins:
            return "no forward distribution"
        total = sum(probability for probability, _ in self.forward_bins)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            return f"forward distribution sums to {total:.6f}, not 1"
        if self.held_fraction < 0 or self.held_fraction > 1:
            return "held fraction out of range"
        if self.current_multiple < 0:
            return "negative multiple"
        if self.exit_capacity_ratio is None:
            return "exit capacity not measured"
        if self.escape_probability is None:
            return "escape probability not measured"
        return None


@dataclass
class ActionScore:
    action: Action
    q: float
    status: str = "OK"
    detail: str = ""

    @property
    def feasible(self) -> bool:
        return self.status == "OK" and math.isfinite(self.q)


@dataclass
class Decision:
    status: str
    action: Action = Action.HOLD
    q: float = 0.0
    scores: List[ActionScore] = field(default_factory=list)
    detail: str = ""
    # Which implementation produced this, and whether the other one agreed.
    # Declared rather than attached dynamically so a decision can always say
    # where it came from -- a Rust answer and a Python answer that cannot be
    # told apart afterwards make a parity ledger unauditable.
    kernel: Optional[Dict[str, Any]] = None

    def score_of(self, action: Action) -> Optional[float]:
        for score in self.scores:
            if score.action is action:
                return score.q if score.feasible else None
        return None


def _expected_log(bins: Sequence[Tuple[float, float]], wealth_of) -> float:
    """E[log W] over the bins, or -inf if any outcome wipes the book."""
    total = 0.0
    for probability, gross in bins:
        wealth = wealth_of(gross)
        if wealth <= 0:
            return -float("inf")
        total += probability * math.log(wealth)
    return total


class ActionValuePolicy:
    """Scores every action against holding, on one objective.

    The default weights are not learned and are not presented as if they were.
    Where a trained action-value model exists it replaces the analytic scoring
    entirely; until then this is an explicit, inspectable baseline whose
    numbers all come from the same forward distribution -- which is already
    the property the split components lacked.
    """

    def __init__(self, min_edge: float = 1e-4, max_add_fraction: float = 0.05):
        # An action must beat holding by more than estimation noise. Without a
        # margin the policy churns on differences it cannot actually measure.
        self.min_edge = max(0.0, min_edge)
        self.max_add_fraction = max(0.0, max_add_fraction)
        self._model: Optional[Any] = None
        self._model_version: str = ""

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def load_model(self, model: Any, version: str) -> bool:
        """Adopt a trained action-value model.

        A model marked loaded but never consulted is worse than none: the
        readiness surface reports `trained: true` while every decision is
        still analytic, so the promotion ladder can advance on a model that
        has never priced a trade. `score` consults it whenever it is present.
        """
        if not hasattr(model, "predict"):
            logger.warning("action-value model rejected: no predict")
            return False
        self._model, self._model_version = model, str(version)
        return True

    def _model_scores(self, state: PositionState) -> Optional[Dict[Action, float]]:
        """Q per action from the loaded model, or None when it cannot answer.

        A model that raises, or returns a shape that does not cover the action
        set, falls back to the analytic scoring rather than to a partial
        answer -- a Q table missing EXIT would silently make exiting
        impossible.
        """
        if self._model is None:
            return None
        try:
            raw = self._model.predict(state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("action-value model inference failed: %s", exc)
            return None
        if not isinstance(raw, dict):
            logger.warning("action-value model returned %s, not a Q map", type(raw))
            return None
        scores: Dict[Action, float] = {}
        for action in Action:
            value = raw.get(action, raw.get(action.value))
            if value is None or not math.isfinite(float(value)):
                logger.warning("action-value model omitted %s; falling back to analytic",
                               action.value)
                return None
            scores[action] = float(value)
        return scores

    # -- component values --------------------------------------------------

    def _hold_value(self, state: PositionState) -> float:
        """E[log W] of doing nothing. The baseline every action is measured against."""
        held = state.held_fraction
        multiple = max(0.0, state.current_multiple)
        cash = 1.0 - held
        position_now = held * multiple

        def wealth(gross: float) -> float:
            captured = self._capture(state, gross)
            return cash + position_now * (1.0 + captured)

        return _expected_log(state.forward_bins, wealth)

    def _capture(self, state: PositionState, gross: float) -> float:
        """Forward return actually capturable, after capacity and escape.

        Upside that cannot be sold is not upside, and upside we are unlikely to
        get out of before a collapse is not upside either. Applying both here
        rather than at one action means no action can be priced on a return the
        position could not have realised.

        Neither input has a permissive default. `blocked_reason` refuses the
        whole state when either is unmeasured, because reading unknown escape
        as "fully escapable" is the single most flattering assumption
        available here: it makes every trapped position look liquid, and it
        does so most strongly on exactly the tokens where escape is hardest to
        measure.
        """
        if gross <= 0:
            return gross
        capacity = float(np.clip(state.exit_capacity_ratio, 0.0, 1.0))
        escape = float(np.clip(state.escape_probability, 0.0, 1.0))
        return gross * capacity * escape

    def _bank_value(self, state: PositionState, fraction: float) -> float:
        """E[log W] after selling ``fraction`` of the position now.

        The sold slice realises at the current multiple less exit cost, and
        stops participating in the forward distribution. Charging both is what
        stops banking looking free -- a free bank always beats holding, and a
        policy that banks every tick turns a runner into a fee schedule.
        """
        held = state.held_fraction
        multiple = max(0.0, state.current_multiple)
        sellable = held * fraction
        capacity = state.exit_capacity_ratio
        capacity = 1.0 if capacity is None else float(np.clip(capacity, 0.0, 1.0))
        # Only the part the venue can absorb is actually sold.
        sold = sellable * capacity
        remaining = held - sold

        proceeds = sold * multiple * (1.0 - state.exit_cost)
        cash = (1.0 - held) + proceeds
        position_now = remaining * multiple

        def wealth(gross: float) -> float:
            return cash + position_now * (1.0 + self._capture(state, gross))

        return _expected_log(state.forward_bins, wealth)

    def _add_value(self, state: PositionState) -> Tuple[float, str]:
        """E[log W] after adding. New capital enters at today's price."""
        if state.add_fraction is None:
            return -float("inf"), "no add size supplied"
        added = float(state.add_fraction)
        if added <= 0:
            return -float("inf"), "non-positive add size"
        ceiling = self.max_add_fraction
        if state.add_capacity_fraction is not None:
            ceiling = min(ceiling, float(state.add_capacity_fraction))
        if added > ceiling:
            return -float("inf"), f"add exceeds capacity ceiling {ceiling:.4f}"

        held = state.held_fraction
        multiple = max(0.0, state.current_multiple)
        cash = 1.0 - held - added
        if cash < 0:
            return -float("inf"), "add exceeds available cash"
        position_now = held * multiple + added

        def wealth(gross: float) -> float:
            return (cash + position_now * (1.0 + self._capture(state, gross))
                    - added * state.entry_cost)

        return _expected_log(state.forward_bins, wealth), "ok"

    def _redeploy_bonus(self, state: PositionState) -> float:
        """Log growth the freed capital earns elsewhere over the same horizon.

        The same quantity `OpportunityAllocator` ranks on, so exiting to fund a
        better opportunity and displacing a position to fund one cannot reach
        different conclusions about the same pair.
        """
        rate = state.alternative_growth_per_second
        seconds = state.expected_remaining_seconds
        if rate is None or seconds is None or rate <= 0 or seconds <= 0:
            return 0.0
        # rate is growth per dollar per second; the freed capital is the
        # position's current mark.
        freed = state.held_fraction * max(0.0, state.current_multiple)
        return float(rate * seconds * freed)

    def _replace_value(self, state: PositionState,
                       exit_value: float) -> Tuple[float, str]:
        """Exit and immediately fund a specific named alternative.

        Distinct from EXIT: exiting credits the generic best-alternative rate,
        while replacing commits to one candidate whose own distribution is
        supplied. Charging the entry cost of that candidate is what stops
        REPLACE from looking like a free upgrade over EXIT.
        """
        if state.replacement_bins is None:
            return -float("inf"), "no replacement candidate supplied"
        if state.replacement_fraction is None or state.replacement_fraction <= 0:
            return -float("inf"), "no replacement size supplied"
        added = float(state.replacement_fraction)
        freed = state.held_fraction * max(0.0, state.current_multiple)
        if added > freed + (1.0 - state.held_fraction):
            return -float("inf"), "replacement exceeds capital the exit would free"

        capacity = float(np.clip(state.exit_capacity_ratio, 0.0, 1.0))
        sold = state.held_fraction * capacity
        proceeds = sold * max(0.0, state.current_multiple) * (1.0 - state.exit_cost)
        cash = (1.0 - state.held_fraction) + proceeds - added
        if cash < 0:
            return -float("inf"), "replacement exceeds realised proceeds"
        stranded = (state.held_fraction - sold) * max(0.0, state.current_multiple)

        def wealth(gross: float) -> float:
            return (cash + stranded + added * (1.0 + gross)
                    - added * state.entry_cost)

        value = _expected_log(state.replacement_bins, wealth)
        return value, "ok"

    def _probe_value(self, state: PositionState) -> Tuple[float, str]:
        """E[log W] of a small first commitment from a flat book.

        Priced on the same forward distribution as everything else, so a probe
        cannot be justified by a number nothing else can see. It is a separate
        action from ADD because ADD prices adding to a position that already
        exists, and the two have different capital bases and different
        capacity ceilings.

        Deliberately NOT credited with the information a probe buys. Our own
        fills reveal real depth, real latency and real fee incidence, and that
        is worth something -- but it is worth an amount nobody here has
        measured, and crediting an unmeasured benefit is how every speculative
        position gets justified.
        """
        if state.held_fraction > 0:
            return -float("inf"), "position is open; probing is not the question"
        size = state.probe_fraction
        if size is None or size <= 0:
            return -float("inf"), "no probe size supplied"
        added = float(size)
        cash = 1.0 - added
        if cash < 0:
            return -float("inf"), "probe exceeds available cash"

        def wealth(gross: float) -> float:
            return (cash + added * (1.0 + self._capture(state, gross))
                    - added * state.entry_cost)

        return _expected_log(state.forward_bins, wealth), "ok"

    def _reenter_value(self, state: PositionState) -> Tuple[float, str]:
        """Re-entry is a new trade and competes as one.

        Scored on its OWN forward distribution, never the open position's.
        Reusing the incumbent's distribution would let a token that has already
        been exited inherit the conviction that was built before the exit,
        which is how a book gets attached to yesterday's winner.
        """
        if state.reentry_bins is None:
            return -float("inf"), "no re-entry distribution supplied"
        if state.held_fraction > 0:
            return -float("inf"), "position is still open; re-entry is not the question"
        if state.add_fraction is None or state.add_fraction <= 0:
            return -float("inf"), "no re-entry size supplied"
        added = float(state.add_fraction)
        cash = 1.0 - added
        if cash < 0:
            return -float("inf"), "re-entry exceeds available cash"

        def wealth(gross: float) -> float:
            return cash + added * (1.0 + self._capture(state, gross)) - added * state.entry_cost

        return _expected_log(state.reentry_bins, wealth), "ok"

    # -- public API --------------------------------------------------------

    def score(self, state: PositionState) -> Decision:
        blocked = state.blocked_reason()
        if blocked:
            return Decision(status="DATA_BLOCKED", detail=blocked)

        model_scores = self._model_scores(state)
        if model_scores is not None:
            scores = [ActionScore(action, model_scores[action] - model_scores[Action.HOLD])
                      for action in Action]
            best = max(scores, key=lambda item: item.q)
            if best.q <= self.min_edge:
                return Decision(status="OK", action=Action.HOLD, q=0.0, scores=scores,
                                detail=f"model {self._model_version}: no action clears "
                                       f"the {self.min_edge:g} margin")
            return Decision(status="OK", action=best.action, q=best.q, scores=scores,
                            detail=f"model {self._model_version} chose {best.action.value}")

        baseline = self._hold_value(state)
        if not math.isfinite(baseline):
            return Decision(status="DATA_BLOCKED",
                            detail="holding has no finite value; the state is not priceable")

        scores: List[ActionScore] = [ActionScore(Action.HOLD, 0.0)]

        for action in (Action.BANK_10, Action.BANK_25, Action.BANK_50, Action.BANK_75):
            value = self._bank_value(state, action.bank_fraction)
            scores.append(ActionScore(action, value - baseline))

        exit_value = self._bank_value(state, 1.0) + self._redeploy_bonus(state)
        scores.append(ActionScore(Action.EXIT, exit_value - baseline))

        # REPLACE is EXIT plus immediately funding a NAMED better candidate.
        # It was in the enum and never scored, which meant the allocator's
        # displacement path and this policy could reach opposite conclusions
        # about the same pair of tokens.
        replace_value, replace_detail = self._replace_value(state, exit_value)
        scores.append(ActionScore(
            Action.REPLACE, replace_value - baseline,
            status="OK" if math.isfinite(replace_value) else "DATA_BLOCKED",
            detail=replace_detail))

        add_value, add_detail = self._add_value(state)
        scores.append(ActionScore(
            Action.ADD, add_value - baseline,
            status="OK" if math.isfinite(add_value) else "DATA_BLOCKED", detail=add_detail))

        reenter_value, reenter_detail = self._reenter_value(state)
        scores.append(ActionScore(
            Action.REENTER, reenter_value - baseline,
            status="OK" if math.isfinite(reenter_value) else "DATA_BLOCKED",
            detail=reenter_detail))

        probe_value, probe_detail = self._probe_value(state)
        scores.append(ActionScore(
            Action.PROBE, probe_value - baseline,
            status="OK" if math.isfinite(probe_value) else "DATA_BLOCKED",
            detail=probe_detail))

        # IGNORE is HOLD seen from a flat book, and scores exactly zero for
        # the same reason: doing nothing is the baseline. It exists as a named
        # action so that a decision not to enter is recorded as a decision
        # rather than as an absence, which is what makes the missed-winner
        # ledger possible at all.
        scores.append(ActionScore(
            Action.IGNORE, 0.0 if state.held_fraction <= 0 else -float("inf"),
            status="OK" if state.held_fraction <= 0 else "DATA_BLOCKED",
            detail="flat" if state.held_fraction <= 0 else "position is open"))

        feasible = [score for score in scores if score.feasible]
        best = max(feasible, key=lambda score: score.q)
        # Hold wins ties and wins anything inside the noise margin, so a policy
        # that cannot tell two actions apart does not churn the book.
        # Doing nothing wins ties and wins anything inside the noise margin, so
        # a policy that cannot tell two actions apart does not churn the book.
        # Which "nothing" it is depends on whether anything is held: a flat
        # book IGNOREs, an open one HOLDs, and recording the difference is
        # what lets a rejected launch be scored against what it went on to do.
        idle = Action.IGNORE if state.held_fraction <= 0 else Action.HOLD
        if best.q <= self.min_edge:
            return Decision(status="OK", action=idle, q=0.0, scores=scores,
                            detail=f"no action clears the {self.min_edge:g} margin")
        return Decision(status="OK", action=best.action, q=best.q, scores=scores,
                        detail=f"{best.action.value} beats holding by {best.q:.6f}")
