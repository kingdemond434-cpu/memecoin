"""Which ticket size makes a small bankroll most likely to survive to the tail.

The existing allocator maximises E[log W], which is the right objective for
growing a bankroll and the wrong one for this question. Maximising expected
log wealth answers "how fast does capital compound"; the stated goal is closer
to

    maximise  P(W_T >= target)
    subject to P(W_t < floor for some t) <= epsilon

and those diverge precisely where it matters. Without the ruin constraint the
target-probability optimiser is degenerate: it bets everything on the most
explosive coin, which maximises the lottery payoff and produces a near-certain
dead account. With it, the question becomes real and has a measurable answer.

The answer is not assumed here, it is simulated. A policy is a ticket size, a
concurrency, a reserve and a set of bank-or-hold rules; the simulator replays
it over thousands of launch sequences drawn from a MEASURED outcome model and
reports how often the bankroll reaches the target and how often it dies first.

Two disciplines make it worth trusting:

**The outcome model must be measured, not chosen.** `LaunchOutcomeModel` takes
a survival curve and a capture function -- both of which the desk now produces
from its own arithmetic -- and refuses to run on an empty one. A simulator fed
invented probabilities returns invented advice with a confidence interval
attached, which is worse than no answer.

**Capture is a function of ticket size, not a constant.** This is the finding
the whole module is built around. Measured on a real migrated pool, the same
1000x printed move is worth 95.4% to a 0.1 SOL ticket and 2.4% to a 10 SOL
ticket. An optimiser that treats the multiple as size-independent will
recommend large tickets and be wrong by more than an order of magnitude.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import asdict, dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

logger = logging.getLogger(__name__)

WEALTH_TARGET_SCHEMA_VERSION = "v1"

#: Sequences simulated per policy. Enough that a probability near 1% is stable
#: to a few tenths of a percent, cheap enough to sweep a grid.
DEFAULT_TRIALS = 2_000

#: Seeded, so a recommendation is a property of the evidence rather than of
#: the moment it was computed.
DEFAULT_SEED = 20260919

#: Ruin budget. A policy that dies more often than this does not get to be
#: preferred for reaching the target, however often it reaches it.
DEFAULT_MAX_RUIN = 0.20


@dataclass(frozen=True)
class TailPolicy:
    """One way to play a bankroll at the tail."""

    #: Share of the CURRENT bankroll per ticket.
    ticket_fraction: float
    #: How many tickets may be open at once.
    max_concurrent: int = 10
    #: Share of the bankroll never risked.
    reserve_fraction: float = 0.0
    #: Multiple at which half the position is banked. None holds it all.
    bank_at_multiple: Optional[float] = None
    #: Share banked when that multiple is reached.
    bank_fraction: float = 0.5

    @property
    def name(self) -> str:
        parts = [f"f{self.ticket_fraction:.3f}", f"n{self.max_concurrent}"]
        if self.reserve_fraction:
            parts.append(f"r{self.reserve_fraction:.2f}")
        if self.bank_at_multiple:
            parts.append(f"bank{self.bank_at_multiple:g}x"
                         f"@{self.bank_fraction:.2f}")
        return "/".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "name": self.name}


class LaunchOutcomeModel:
    """Draws a launch's realised multiple for a ticket of a given size.

    The survival curve gives P(M >= m) for the PRINTED multiple. The capture
    function turns that into what a ticket of this size could actually take
    out, which is where the size dependence enters and where every
    size-independent tail model goes wrong.
    """

    def __init__(self, survival: Sequence[Tuple[float, float]],
                 capture: Optional[Callable[[float, float], float]] = None, *,
                 floor_multiple: float = 0.0,
                 round_trip_cost: float = 0.04):
        self.survival = sorted(
            ((float(level), float(value)) for level, value in survival),
            key=lambda item: item[0])
        self.capture = capture
        self.round_trip_cost = float(round_trip_cost)
        # What a launch that never reached the FIRST measured rung realises.
        # Zero -- a total loss -- is the deliberate default, because the first
        # rung is usually 2x and the desk's own exit policy is what decides
        # between 0 and 2x. That behaviour is a policy, not a property of the
        # launch distribution, so it is supplied rather than invented here.
        self.floor_multiple = float(floor_multiple)
        measured = [value for _level, value in self.survival if value > 0]
        #: Derived, not drawn separately. P(never reaching the first rung) is
        #: one minus the probability of reaching it, and drawing a rug from an
        #: independent coin on top of that double-counts the same mass -- the
        #: first version did exactly that and produced a model in which no
        #: launch ever reached 2x.
        self.miss_probability = max(
            0.0, 1.0 - (max(measured) if measured else 0.0))

    @property
    def usable(self) -> bool:
        return len(self.survival) >= 2 and any(v > 0 for _, v in self.survival)

    def expected_multiple(self) -> Optional[float]:
        """E[realised multiple] per launch, before capture and costs.

        Reported with every simulation, and it is the first number to read.
        A sweep is a function of its input distribution, and a distribution
        implying a +105% edge per launch will conclude that almost any policy
        gets rich -- which is a fact about the assumption, not about the
        market. Without this on the face of the report that conclusion looks
        like a finding.
        """
        if not self.usable:
            return None
        total = 0.0
        previous = 0.0
        for level, probability in reversed(self.survival):
            # Mass in [this rung, the next one up) times this rung's payoff.
            total += max(0.0, probability - previous) * level
            previous = probability
        total += self.miss_probability * self.floor_multiple
        return float(total)

    def draw(self, rng: random.Random, ticket_usd: float) -> float:
        """One launch's realised multiple on capital, net of the round trip.

        ONE uniform draw, read against the survival curve directly: a launch
        reaches m exactly when u <= P(M >= m), so the realised rung is the
        highest one the draw clears. Everything below the first rung is the
        miss, and no separate rug coin is tossed -- tossing one double-counts
        the mass the curve has already accounted for.
        """
        draw = rng.random()
        printed: Optional[float] = None
        for level, probability in self.survival:
            if draw <= probability:
                printed = level
        if printed is None:
            return max(0.0, self.floor_multiple * (1.0 - self.round_trip_cost))
        taken = (self.capture(printed, ticket_usd) if self.capture
                 else printed)
        return max(0.0, taken * (1.0 - self.round_trip_cost))


@dataclass
class PolicyOutcome:
    """What one policy did across every simulated sequence."""

    policy: TailPolicy
    trials: int = 0
    hit_target: float = 0.0
    ruined: float = 0.0
    median_terminal: float = 0.0
    p10_terminal: float = 0.0
    p90_terminal: float = 0.0
    mean_log_growth: float = 0.0
    admissible: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(), "trials": self.trials,
            "hit_target": round(self.hit_target, 5),
            "ruined": round(self.ruined, 5),
            "median_terminal": round(self.median_terminal, 2),
            "p10_terminal": round(self.p10_terminal, 2),
            "p90_terminal": round(self.p90_terminal, 2),
            "mean_log_growth": round(self.mean_log_growth, 6),
            "admissible": self.admissible, "detail": self.detail,
        }


def _percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(share * len(ordered))))
    return ordered[index]


class WealthTargetSimulator:
    """Replays policies over drawn launch sequences and counts the outcomes."""

    def __init__(self, model: LaunchOutcomeModel, *,
                 trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED,
                 max_ruin: float = DEFAULT_MAX_RUIN):
        self.model = model
        self.trials = int(trials)
        self.seed = int(seed)
        self.max_ruin = float(max_ruin)

    def _sequence(self, rng: random.Random, policy: TailPolicy,
                  bankroll: float, target: float, floor: float,
                  launches: int) -> Tuple[float, bool, bool]:
        """One life of the bankroll. Returns (terminal, hit target, ruined)."""
        equity = float(bankroll)
        hit = False
        for _ in range(launches):
            if equity < floor:
                return equity, hit, True
            if equity >= target:
                hit = True
                # Reaching the target is the objective; continuing to risk the
                # bankroll afterwards measures a different question.
                break
            investable = equity * (1.0 - policy.reserve_fraction)
            ticket = investable * policy.ticket_fraction
            if ticket <= 0:
                break
            open_tickets = min(policy.max_concurrent,
                               max(1, int(investable / max(ticket, 1e-9))))
            staked = min(investable, ticket * open_tickets)
            returned = 0.0
            for _slot in range(open_tickets):
                multiple = self.model.draw(rng, ticket)
                if (policy.bank_at_multiple
                        and multiple >= policy.bank_at_multiple):
                    banked = policy.bank_fraction * policy.bank_at_multiple
                    held = (1.0 - policy.bank_fraction) * multiple
                    multiple = banked + held
                returned += ticket * multiple
            equity = equity - staked + returned
        return equity, hit or equity >= target, equity < floor

    def evaluate(self, policy: TailPolicy, *, bankroll: float, target: float,
                 floor: float, launches: int = 200) -> PolicyOutcome:
        outcome = PolicyOutcome(policy=policy, trials=self.trials)
        if not self.model.usable:
            outcome.detail = ("the outcome model has no measured survival "
                              "curve; nothing can be simulated from it")
            return outcome
        rng = random.Random(self.seed)
        terminals: List[float] = []
        hits = 0
        ruins = 0
        for _ in range(self.trials):
            terminal, hit, ruined = self._sequence(
                rng, policy, bankroll, target, floor, launches)
            terminals.append(terminal)
            hits += int(hit)
            ruins += int(ruined)
        outcome.hit_target = hits / self.trials
        outcome.ruined = ruins / self.trials
        outcome.median_terminal = _percentile(terminals, 0.5)
        outcome.p10_terminal = _percentile(terminals, 0.10)
        outcome.p90_terminal = _percentile(terminals, 0.90)
        outcome.mean_log_growth = float(sum(
            math.log(max(1e-9, value / bankroll)) for value in terminals
        ) / self.trials)
        outcome.admissible = outcome.ruined <= self.max_ruin
        outcome.detail = ("" if outcome.admissible else
                          f"ruin {outcome.ruined:.1%} exceeds the "
                          f"{self.max_ruin:.0%} budget")
        return outcome

    def sweep(self, policies: Sequence[TailPolicy], *, bankroll: float,
              target: float, floor: float, launches: int = 200
              ) -> Dict[str, Any]:
        """Every policy, ranked by target probability among the survivors.

        The ruin constraint is applied as a FILTER rather than a penalty
        weight. A weight lets a high enough target probability buy its way
        past any amount of ruin, which is exactly the degenerate answer this
        module exists to avoid.
        """
        outcomes = [self.evaluate(policy, bankroll=bankroll, target=target,
                                  floor=floor, launches=launches)
                    for policy in policies]
        admissible = [item for item in outcomes if item.admissible]
        admissible.sort(key=lambda item: -item.hit_target)
        best = admissible[0] if admissible else None
        expected = self.model.expected_multiple()
        # An edge this large is almost certainly an artefact of the input
        # curve rather than a property of the market, and a sweep run on it
        # will conclude that nearly any policy succeeds. Saying so on the face
        # of the report is the difference between a simulation and a
        # prophecy.
        implausible = expected is not None and expected > 1.20
        return {
            "schema": WEALTH_TARGET_SCHEMA_VERSION,
            "status": "OK" if outcomes and self.model.usable else "DATA_BLOCKED",
            "expected_multiple_per_launch": expected,
            "implied_edge_per_launch": (None if expected is None
                                        else expected - 1.0),
            "input_edge_implausible": implausible,
            "input_warning": (
                "" if not implausible else
                f"the supplied survival curve implies E[multiple] = "
                f"{expected:.2f} per launch, a {expected - 1.0:+.0%} edge. "
                f"Every conclusion below is a consequence of that assumption "
                f"and not evidence about the market."),
            "bankroll": bankroll, "target": target, "floor": floor,
            "launches_per_life": launches, "trials": self.trials,
            "max_ruin": self.max_ruin,
            "policies_tested": len(outcomes),
            "admissible": len(admissible),
            "best": best.to_dict() if best else None,
            "ranked": [item.to_dict() for item in
                       sorted(outcomes, key=lambda i: (-int(i.admissible),
                                                       -i.hit_target))],
            "detail": ("" if best else
                       "no policy stayed inside the ruin budget; the bankroll "
                       "is too small for this target under this outcome "
                       "model, and saying so is the answer"),
        }


def default_policy_grid(
        fractions: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.25),
        concurrency: Sequence[int] = (1, 5, 20, 50),
        reserves: Sequence[float] = (0.0, 0.3),
) -> List[TailPolicy]:
    """A grid, not a shortlist. Choosing before simulating is the mistake."""
    return [TailPolicy(ticket_fraction=fraction, max_concurrent=count,
                       reserve_fraction=reserve)
            for fraction in fractions
            for count in concurrency
            for reserve in reserves]
