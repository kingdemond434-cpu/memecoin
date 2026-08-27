"""Monster-hold: the state machine that stops a 20x from becoming a 2x.

Every exit rule in this repository before this one shares an assumption that
is wrong on exactly the trades that matter: that a large unrealised gain is
evidence the move is ending. It is not. A token at +500% has told you the
demand was real; it has told you nothing about whether the demand is finished.
A ratchet that banks harder the higher a position goes is optimal for ordinary
winners and catastrophic for the rare launch that would have carried the
account, because it is guaranteed to sell that one first and hardest.

So the exit question is posed here as a comparison rather than a threshold:

    V_hold = E[future capturable upside, net of distribution and rug risk]
    V_exit = profit locked now + what that capital earns redeployed elsewhere

and a full exit happens when V_exit > V_hold. Price level appears nowhere in
that comparison. What appears is remaining upside, the risk of not capturing
it, and the opportunity cost of the capital -- which is the same
cross-sectional question `OpportunityAllocator` asks, so the two agree by
construction rather than by coincidence.

Three properties do most of the work:

Hysteresis. The failure mode this state machine exists to prevent is: monster
runs, one whale sells, hazard spikes for a single tick, the bot dumps
everything, the token continues 15x. Downgrades out of a monster state
therefore require persistent evidence across independent dimensions -- smart
money leaving AND buyer quality falling AND acceleration rolling over -- held
for several consecutive evaluations. One signal, once, is noise.

Catastrophic bypass. Hysteresis must never apply to sellability loss, malicious
control changes or liquidity removal. Those exit immediately, on one
observation, from any state. Patience about a rug is not patience.

Calibration gating. The machine cannot enter a monster state on an
uncalibrated belief. Monster probability has to arrive from a validated model
or the machine stays in NORMAL and ordinary exits apply unchanged. An override
that lets a position ignore its stop, granted on an unvalidated number, is the
most expensive possible fabrication.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MONSTER_SCHEMA_VERSION = "v1"


class MonsterState(Enum):
    """Where a position sits in the life of a move.

    Ordered by conviction, not by price. A token can reach MONSTER_HOLD at 1.4x
    and a different one can sit in PUMP_DETECTED at 8x, because the states
    describe the evidence, not the multiple.
    """

    NORMAL = "normal"
    PUMP_DETECTED = "pump_detected"
    EARLY_CONFIRMATION = "early_confirmation"
    MONSTER_CANDIDATE = "monster_candidate"
    MONSTER_HOLD = "monster_hold"
    MASS_FOMO = "mass_fomo"
    SATURATION = "saturation"
    DISTRIBUTION = "distribution"


# Conviction ordering. Advancing along this is cheap; retreating along it is
# what hysteresis governs.
_STATE_ORDER = {state: index for index, state in enumerate([
    MonsterState.NORMAL, MonsterState.PUMP_DETECTED, MonsterState.EARLY_CONFIRMATION,
    MonsterState.MONSTER_CANDIDATE, MonsterState.MONSTER_HOLD, MonsterState.MASS_FOMO,
    MonsterState.SATURATION, MonsterState.DISTRIBUTION,
])}

# States in which the position is deliberately allowed to tolerate more
# volatility than the ordinary trailing stop would.
MONSTER_STATES = frozenset({MonsterState.MONSTER_CANDIDATE, MonsterState.MONSTER_HOLD,
                            MonsterState.MASS_FOMO})


@dataclass
class MonsterEvidence:
    """Point-in-time inputs. Every field is Optional and None means unobserved.

    Nothing here defaults to a number. A monster override granted because an
    unobserved signal read as zero would be the same class of error as
    fabricating the signal outright.
    """

    monster_probability: Optional[float] = None
    monster_probability_calibrated: bool = False
    independent_buyer_acceleration: Optional[float] = None
    smart_wallet_net_accumulation: Optional[float] = None
    buyer_quality_trend: Optional[float] = None
    sell_absorption: Optional[float] = None
    liquidity_expansion: Optional[float] = None
    new_source_discovery_rate: Optional[float] = None
    audience_penetration: Optional[float] = None
    distribution_probability: Optional[float] = None
    distribution_calibrated: bool = False
    rug_probability: Optional[float] = None
    exit_capacity_ratio: Optional[float] = None
    catastrophic_hazard: bool = False

    def observed(self) -> Dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)}


@dataclass
class HoldExitValue:
    """The V_hold versus V_exit comparison, with its inputs kept visible."""

    status: str
    v_hold: float = 0.0
    v_exit: float = 0.0
    detail: str = ""
    remaining_upside: float = 0.0
    survival: float = 0.0
    redeploy_value: float = 0.0

    @property
    def should_exit(self) -> bool:
        return self.status == "OK" and self.v_exit > self.v_hold


@dataclass
class MonsterDecision:
    state: MonsterState
    previous_state: MonsterState
    action: str
    bank_fraction: float = 0.0
    reason: str = ""
    value: Optional[HoldExitValue] = None
    degrade_streak: int = 0
    evidence_dimensions: List[str] = field(default_factory=list)


def hold_versus_exit(
    remaining_upside_multiple: float,
    distribution_probability: float,
    rug_probability: float,
    exit_capacity_ratio: float,
    alternative_growth_per_second: float,
    expected_remaining_seconds: float,
) -> HoldExitValue:
    """Compare holding to exiting, in log-wealth, with no reference to price level.

    ``remaining_upside_multiple`` is E[further multiple from here] -- 1.0 means
    "expected to go nowhere", 3.0 means "expected to triple again". It is a
    forward quantity, so how far the position has already travelled does not
    enter, which is the entire point.

    ``exit_capacity_ratio`` is the share of the position actually sellable at
    an acceptable impact. Upside that cannot be liquidated is not upside, so it
    is applied to the hold branch rather than being ignored.
    """
    if remaining_upside_multiple <= 0 or expected_remaining_seconds <= 0:
        return HoldExitValue("DATA_BLOCKED", detail="no forward upside or horizon supplied")
    if not 0 <= distribution_probability <= 1 or not 0 <= rug_probability <= 1:
        return HoldExitValue("DATA_BLOCKED", detail="risk probabilities out of range")
    capacity = max(0.0, min(1.0, exit_capacity_ratio))
    if capacity <= 0:
        return HoldExitValue("DATA_BLOCKED", detail="position is not liquidatable at any size")

    # Surviving the horizon means neither the rug nor distribution front-runs
    # the upside. A rug takes most of the position; distribution takes the
    # upside but leaves the position roughly where it is.
    survival = (1.0 - rug_probability) * (1.0 - distribution_probability)
    captured = 1.0 + (remaining_upside_multiple - 1.0) * capacity

    v_hold = (
        survival * math.log(max(captured, 1e-9))
        + (1.0 - rug_probability) * distribution_probability * math.log(1.0)
        + rug_probability * math.log(0.02)
    )
    # Exiting realises the position and frees the capital, which then earns the
    # best alternative for the horizon the hold would have consumed. That term
    # is what makes this the same question the allocator asks.
    v_exit = max(0.0, alternative_growth_per_second) * expected_remaining_seconds
    return HoldExitValue(
        status="OK", v_hold=float(v_hold), v_exit=float(v_exit),
        remaining_upside=float(remaining_upside_multiple), survival=float(survival),
        redeploy_value=float(v_exit),
        detail="hold beats exit" if v_hold >= v_exit else "exit beats hold",
    )


class MonsterStateMachine:
    """Tracks one position's conviction state and what to do about it.

    Default banking fractions below are hand-picked and are labelled as such.
    They are the fallback until an exit-value model trained on complete
    trajectories replaces them, exactly as `ExitPolicy` defaults stand in for
    `exit_policy_trainer` output. They are not evidence.
    """

    # Fallback ladders, not learned parameters.
    DEFAULT_BANK_FRACTIONS = {
        MonsterState.MASS_FOMO: 0.15,
        MonsterState.SATURATION: 0.35,
        MonsterState.DISTRIBUTION: 0.80,
    }

    def __init__(
        self,
        monster_probability_threshold: float = 0.15,
        candidate_probability_threshold: float = 0.06,
        degrade_confirmations: int = 3,
        min_degrade_dimensions: int = 2,
        bank_fractions: Optional[Dict[MonsterState, float]] = None,
    ):
        self.monster_probability_threshold = monster_probability_threshold
        self.candidate_probability_threshold = candidate_probability_threshold
        # A monster state is not surrendered on one tick from one signal.
        self.degrade_confirmations = max(1, degrade_confirmations)
        self.min_degrade_dimensions = max(1, min_degrade_dimensions)
        self.bank_fractions = dict(bank_fractions or self.DEFAULT_BANK_FRACTIONS)
        self._states: Dict[str, MonsterState] = {}
        self._degrade_streaks: Dict[str, int] = {}
        self._banked: Dict[str, List[MonsterState]] = {}

    def state_of(self, token: str) -> MonsterState:
        return self._states.get(token, MonsterState.NORMAL)

    def reset(self, token: str) -> None:
        self._states.pop(token, None)
        self._degrade_streaks.pop(token, None)
        self._banked.pop(token, None)

    @staticmethod
    def _degrade_dimensions(evidence: MonsterEvidence) -> List[str]:
        """Independent signals currently arguing the move is over.

        Independent is the operative word: three restatements of the same
        underlying fact are one dimension, so each entry here has to come from
        a different part of the flow.
        """
        dimensions = []
        if (evidence.smart_wallet_net_accumulation is not None
                and evidence.smart_wallet_net_accumulation < 0):
            dimensions.append("smart_wallet_outflow")
        if evidence.buyer_quality_trend is not None and evidence.buyer_quality_trend < 0:
            dimensions.append("buyer_quality_deterioration")
        if (evidence.independent_buyer_acceleration is not None
                and evidence.independent_buyer_acceleration < 0):
            dimensions.append("acceleration_rollover")
        if evidence.sell_absorption is not None and evidence.sell_absorption < 0.4:
            dimensions.append("absorption_failure")
        if (evidence.liquidity_expansion is not None and evidence.liquidity_expansion < 0):
            dimensions.append("liquidity_contraction")
        return dimensions

    def _target_state(self, evidence: MonsterEvidence) -> MonsterState:
        """Where the evidence alone says this position belongs."""
        if evidence.catastrophic_hazard:
            return MonsterState.DISTRIBUTION
        if evidence.distribution_calibrated and (evidence.distribution_probability or 0) >= 0.6:
            return MonsterState.DISTRIBUTION

        # A monster state is reachable only from a calibrated belief.
        probability = evidence.monster_probability if evidence.monster_probability_calibrated else None
        if probability is None:
            return MonsterState.NORMAL

        saturated = (evidence.audience_penetration is not None
                     and evidence.audience_penetration >= 0.85)
        if saturated:
            return MonsterState.SATURATION

        accelerating = (evidence.independent_buyer_acceleration or 0) > 0
        accumulating = (evidence.smart_wallet_net_accumulation or 0) > 0
        broadening = (evidence.new_source_discovery_rate or 0) > 0

        if probability >= self.monster_probability_threshold:
            if broadening and (evidence.audience_penetration or 0) >= 0.5:
                return MonsterState.MASS_FOMO
            if accelerating or accumulating:
                return MonsterState.MONSTER_HOLD
            return MonsterState.MONSTER_CANDIDATE
        if probability >= self.candidate_probability_threshold:
            return (MonsterState.MONSTER_CANDIDATE if (accelerating and accumulating)
                    else MonsterState.EARLY_CONFIRMATION)
        return MonsterState.PUMP_DETECTED if accelerating else MonsterState.NORMAL

    def update(
        self,
        token: str,
        evidence: MonsterEvidence,
        value: Optional[HoldExitValue] = None,
    ) -> MonsterDecision:
        current = self.state_of(token)
        target = self._target_state(evidence)
        dimensions = self._degrade_dimensions(evidence)

        # Catastrophe bypasses everything, from any state, on one observation.
        if evidence.catastrophic_hazard:
            self._states[token] = MonsterState.DISTRIBUTION
            self._degrade_streaks[token] = 0
            return MonsterDecision(
                state=MonsterState.DISTRIBUTION, previous_state=current,
                action="emergency_exit", bank_fraction=1.0,
                reason="catastrophic_hazard", value=value, evidence_dimensions=dimensions,
            )

        advancing = _STATE_ORDER[target] > _STATE_ORDER[current]
        leaving_monster = (current in MONSTER_STATES
                           and _STATE_ORDER[target] < _STATE_ORDER[current])

        if leaving_monster:
            # Retreating out of a monster state needs breadth and persistence.
            # One whale selling once is the exact event that must not eject a
            # position that goes on to run.
            if len(dimensions) >= self.min_degrade_dimensions:
                streak = self._degrade_streaks.get(token, 0) + 1
            else:
                streak = 0
            self._degrade_streaks[token] = streak
            if streak < self.degrade_confirmations:
                return MonsterDecision(
                    state=current, previous_state=current, action="hold",
                    reason=(f"degrade evidence not persistent: {streak}/"
                            f"{self.degrade_confirmations} confirmations across "
                            f"{len(dimensions)} dimensions"),
                    value=value, degrade_streak=streak, evidence_dimensions=dimensions,
                )
        else:
            self._degrade_streaks[token] = 0

        self._states[token] = target
        streak = self._degrade_streaks.get(token, 0)

        if target is MonsterState.DISTRIBUTION:
            return MonsterDecision(
                state=target, previous_state=current, action="bank",
                bank_fraction=self.bank_fractions.get(target, 0.8),
                reason="distribution", value=value, degrade_streak=streak,
                evidence_dimensions=dimensions,
            )

        # The value comparison decides a full exit; it never fires on price.
        if value is not None and value.should_exit and target not in MONSTER_STATES:
            return MonsterDecision(
                state=target, previous_state=current, action="exit", bank_fraction=1.0,
                reason="redeploying beats holding", value=value,
                degrade_streak=streak, evidence_dimensions=dimensions,
            )

        if target in self.bank_fractions:
            banked = self._banked.setdefault(token, [])
            if target not in banked:
                banked.append(target)
                return MonsterDecision(
                    state=target, previous_state=current, action="bank",
                    bank_fraction=self.bank_fractions[target],
                    reason=f"staged banking at {target.value}", value=value,
                    degrade_streak=streak, evidence_dimensions=dimensions,
                )

        if advancing and target in MONSTER_STATES:
            return MonsterDecision(
                state=target, previous_state=current, action="add",
                reason=f"conviction advanced to {target.value}", value=value,
                degrade_streak=streak, evidence_dimensions=dimensions,
            )
        return MonsterDecision(
            state=target, previous_state=current, action="hold",
            reason=f"holding in {target.value}", value=value,
            degrade_streak=streak, evidence_dimensions=dimensions,
        )

    def overrides_ordinary_exit(self, token: str) -> bool:
        """Whether the ordinary ratchet/trail should stand down for this token.

        True only inside a monster state, which is reachable only from a
        calibrated monster probability. Without a validated model this is
        always False and every ordinary exit rule applies unchanged.
        """
        return self.state_of(token) in MONSTER_STATES


def tail_capture_ratio(realized_multiple: float, max_feasible_multiple: float) -> Optional[float]:
    """Realised return over the best return that was actually executable.

    The sniper-native score for exit quality on the monster subset. An exit
    policy that lifts win rate while systematically converting 20x trades into
    2x trades looks better on every conventional metric and destroys wealth;
    this is the number that catches it. Feasible, not peak: credit is only for
    what could have been sold, not for a price that printed on a size nobody
    could have exited.
    """
    if max_feasible_multiple <= 0:
        return None
    return float(max(0.0, realized_multiple) / max_feasible_multiple)


def premature_exit_rates(
    trades: Sequence[Dict[str, Any]],
    thresholds: Sequence[Tuple[float, float]] = ((10.0, 5.0), (20.0, 10.0), (50.0, 20.0)),
) -> Dict[str, Any]:
    """How often a policy sold a big winner far too early.

    Each ``(opportunity, floor)`` pair asks: of the trades where a multiple of
    ``opportunity`` was actually feasible, what share were exited below
    ``floor``? A policy that reduces drawdown by killing these should be
    rejected however good its aggregate numbers look.
    """
    report: Dict[str, Any] = {"sample": len(trades)}
    for opportunity, floor in thresholds:
        eligible = [item for item in trades
                    if float(item.get("max_feasible_multiple", 0) or 0) >= opportunity]
        if not eligible:
            report[f"exited_{opportunity:g}x_below_{floor:g}x"] = None
            continue
        early = sum(1 for item in eligible
                    if float(item.get("realized_multiple", 0) or 0) < floor)
        report[f"exited_{opportunity:g}x_below_{floor:g}x"] = early / len(eligible)
    ratios = [tail_capture_ratio(float(item.get("realized_multiple", 0) or 0),
                                 float(item.get("max_feasible_multiple", 0) or 0))
              for item in trades]
    ratios = [value for value in ratios if value is not None]
    report["tail_capture_ratio"] = (sum(ratios) / len(ratios)) if ratios else None
    return report
