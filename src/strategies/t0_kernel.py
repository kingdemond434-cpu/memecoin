"""The Rust T0 decision kernel, wired in behind measured parity.

`native/solana_fastpath` has carried a complete T0 core -- token state, age
band, executable survival bins, Q over every action, hard safety -- with
parity tests against the Python policy. And the canonical runtime went on
calling the Python policy, because nothing connected the two. A kernel that
exists and is never called is the same defect as one that was never written,
and the only difference is that this one looks finished.

Connecting it by simply swapping the call would be worse. The Rust path has
never decided anything in production, and promoting an unproven implementation
onto the money path because its unit tests pass is precisely the move this
codebase refuses everywhere else. So it is promoted the same way a model is:
on evidence, through a ladder, with a demotion that is automatic and loud.

    OFF        Python decides. Rust is not consulted.
    SHADOW     Python decides. Rust decides too, and every disagreement is
               counted. Nothing Rust says can move capital.
    AUTO       Shadow until the kernel has agreed on `promote_after`
               consecutive decisions, then Rust decides. A single
               disagreement demotes it for the rest of the session.
    RUST       Rust decides from the first call. For tests and benchmarks.

What is compared is the POLICY decision -- the action and its Q -- because
that is what `ActionValuePolicy.score()` computes. Rust is called with
permissive limits so its safety layer cannot bind, and the desk's own safety
layer is untouched and still authoritative. Comparing a policy answer against
a policy-plus-safety answer would report a spurious divergence every time
safety correctly refused, which is the fastest way to make a parity signal
worthless.

Two states are deliberately NOT sent to Rust: one carrying re-entry or
replacement distributions, which the kernel's signature cannot express, and
one whose raw survival inputs the caller did not supply. Both fall to Python
and are counted separately, because "Rust was not asked" and "Rust agreed"
are different facts and a report that merges them overstates coverage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.strategies.action_value import Action, ActionScore, Decision, PositionState

logger = logging.getLogger(__name__)

T0_KERNEL_SCHEMA_VERSION = "v1"

# Q values agreeing to this many decimal places count as agreement. The two
# implementations do the same arithmetic in the same order, so the residual is
# f64 rounding rather than a difference of opinion; a tolerance any looser
# would let a real disagreement through as noise.
DEFAULT_Q_TOLERANCE = 1e-9

# Consecutive agreements before Rust is allowed to decide. Five hundred is
# roughly an hour of an active desk's decisions -- enough that agreement is a
# property of the implementations rather than of one quiet market.
DEFAULT_PROMOTE_AFTER = 500

_NO_SURVIVAL = "caller supplied no raw survival inputs"


class KernelMode(Enum):
    OFF = "off"
    SHADOW = "shadow"
    AUTO = "auto"
    RUST = "rust"


def _load_native() -> Tuple[Optional[Any], str]:
    try:
        import solana_fastpath  # type: ignore

        if not hasattr(solana_fastpath, "t0_decide"):
            return None, "extension present but carries no t0_decide"
        return solana_fastpath, "OK"
    except ImportError as exc:
        return None, f"native extension unavailable: {exc}"


@dataclass
class KernelOutcome:
    """One decision's provenance. Attached to the Decision that is returned."""

    source: str            # "python" | "rust"
    compared: bool = False
    agreed: bool = True
    reason: str = ""
    python_action: str = ""
    rust_action: str = ""
    python_q: Optional[float] = None
    rust_q: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "compared": self.compared,
                "agreed": self.agreed, "reason": self.reason,
                "python_action": self.python_action, "rust_action": self.rust_action,
                "python_q": self.python_q, "rust_q": self.rust_q}


@dataclass
class SurvivalInputs:
    """The raw distribution inputs the Rust kernel builds its own bins from.

    Passed rather than the bins themselves: the kernel derives them, and
    handing it the Python bins would compare Python's arithmetic against
    itself and report a parity it had not established.
    """

    levels: Sequence[float]
    p_rug_30s: float
    p_rug_5m: float
    expected_feasible_multiple: float


class T0Kernel:
    """Routes each decision to Python or Rust, and keeps them honest.

    Wraps the policy rather than replacing it. Callers that do not supply
    survival inputs, or whose state carries the re-entry or replacement
    extensions, get the Python answer and nothing changes for them.
    """

    def __init__(self, policy: Any, *, mode: str = "auto",
                 promote_after: int = DEFAULT_PROMOTE_AFTER,
                 q_tolerance: float = DEFAULT_Q_TOLERANCE):
        self.policy = policy
        try:
            self.mode = KernelMode(str(mode).lower())
        except ValueError:
            logger.warning("unknown T0 kernel mode %r; falling back to shadow", mode)
            self.mode = KernelMode.SHADOW
        self.promote_after = max(1, int(promote_after))
        self.q_tolerance = float(q_tolerance)
        self.native, self.native_status = _load_native()

        self.compared = 0
        self.agreements = 0
        self.divergences = 0
        self.consecutive_agreements = 0
        self.not_expressible = 0
        self.no_survival_inputs = 0
        self.rust_errors = 0
        self.rust_decisions = 0
        self.python_decisions = 0
        # Set on the first disagreement while Rust was authoritative, and
        # never cleared. A kernel that demoted itself once has to be looked
        # at, not quietly re-promoted on the next five hundred agreements.
        self.demoted_reason = ""
        self.divergence_examples: List[Dict[str, Any]] = []

    # --- promotion state -------------------------------------------------

    @property
    def rust_available(self) -> bool:
        return self.native is not None

    @property
    def rust_authoritative(self) -> bool:
        """Whether Rust is currently deciding."""
        if not self.rust_available or self.demoted_reason:
            return False
        if self.mode is KernelMode.RUST:
            return True
        if self.mode is KernelMode.AUTO:
            return self.consecutive_agreements >= self.promote_after
        return False

    # --- the decision ----------------------------------------------------

    def score(self, state: PositionState, *,
              survival: Optional[SurvivalInputs] = None,
              age_seconds: float = 0.0,
              virtual_sol: int = 0, virtual_token: int = 0,
              min_edge: Optional[float] = None,
              max_add_fraction: Optional[float] = None) -> Decision:
        """The policy decision, from whichever implementation is authoritative."""
        python_decision = self.policy.score(state)
        # Read BEFORE this call is counted. Otherwise the call that completes
        # the promoting run is itself decided by Rust, which is authority
        # granted by the same evidence it is still establishing.
        was_authoritative = self.rust_authoritative

        blocked = self._not_expressible(state, survival, virtual_sol, virtual_token)
        if self.mode is KernelMode.OFF or not self.rust_available or blocked:
            if blocked == _NO_SURVIVAL:
                self.no_survival_inputs += 1
            elif blocked:
                self.not_expressible += 1
            self.python_decisions += 1
            return self._tagged(python_decision, KernelOutcome(
                source="python", reason=blocked or self.native_status))

        try:
            rust_decision = self._rust_score(
                state, survival, age_seconds, virtual_sol, virtual_token,
                min_edge, max_add_fraction)
        except Exception as exc:
            # A kernel that raises must never take the desk with it. Counted,
            # because a kernel raising on every call is indistinguishable from
            # one that is not wired at all except by this number.
            self.rust_errors += 1
            self.consecutive_agreements = 0
            logger.warning("T0 kernel raised; using the Python decision: %s", exc)
            self.python_decisions += 1
            return self._tagged(python_decision, KernelOutcome(
                source="python", reason=f"rust raised: {exc}"))

        outcome = self._compare(python_decision, rust_decision)
        self.compared += 1
        if outcome.agreed:
            self.agreements += 1
            self.consecutive_agreements += 1
        else:
            self.divergences += 1
            self.consecutive_agreements = 0
            if was_authoritative and not self.demoted_reason:
                self.demoted_reason = outcome.reason
                logger.error(
                    "T0 kernel DEMOTED: rust and python disagreed while rust was "
                    "authoritative (%s). Python decides for the rest of this "
                    "session.", outcome.reason)
            if len(self.divergence_examples) < 20:
                self.divergence_examples.append(outcome.to_dict())

        if was_authoritative and outcome.agreed and not self.demoted_reason:
            self.rust_decisions += 1
            return self._tagged(rust_decision, outcome)
        self.python_decisions += 1
        return self._tagged(python_decision,
                            KernelOutcome(**{**outcome.to_dict(), "source": "python"}))

    # --- internals -------------------------------------------------------

    def _not_expressible(self, state: PositionState, survival: Optional[SurvivalInputs],
                         virtual_sol: int, virtual_token: int) -> str:
        if state.reentry_bins or state.replacement_bins:
            return "state carries re-entry or replacement distributions"
        if survival is None:
            return _NO_SURVIVAL
        if virtual_sol <= 0 or virtual_token <= 0:
            # Without reserves the kernel's own state is untradeable, its
            # safety layer blocks, and its answer collapses to HOLD. That is
            # not a disagreement about policy -- it is the kernel being asked
            # about a market it was given no prices for -- and counting it as
            # a divergence would demote the kernel on every migrated token.
            return "no curve reserves supplied; the kernel cannot price this market"
        return ""

    def _rust_score(self, state: PositionState, survival: SurvivalInputs,
                    age_seconds: float, virtual_sol: int, virtual_token: int,
                    min_edge: Optional[float],
                    max_add_fraction: Optional[float]) -> Decision:
        # Permissive limits and live=False: what is being compared is the
        # POLICY answer, and the desk's own safety layer is unchanged and
        # still authoritative over it.
        action, q, _band, _allowed, _blocked, _refused, _commit, scores = (
            self.native.t0_decide(
                float(age_seconds), int(virtual_sol), int(virtual_token),
                [float(value) for value in survival.levels],
                float(survival.p_rug_30s), float(survival.p_rug_5m),
                float(survival.expected_feasible_multiple),
                float(state.held_fraction), float(state.current_multiple),
                float(state.exit_cost), float(state.entry_cost),
                state.exit_capacity_ratio, state.escape_probability,
                state.alternative_growth_per_second, state.expected_remaining_seconds,
                state.add_fraction, state.add_capacity_fraction,
                state.probe_fraction,
                float(self.policy.min_edge if min_edge is None else min_edge),
                float(getattr(self.policy, "max_add_fraction", 0.5)
                      if max_add_fraction is None else max_add_fraction),
                False, 1.0, 1.0, 0.0, 0.0, True))
        return Decision(
            status="OK", action=Action(action), q=float(q),
            scores=[ActionScore(action=Action(name), q=float(value),
                                status="OK" if feasible else "INFEASIBLE")
                    for name, value, feasible in scores],
            detail="rust t0 kernel")

    def _compare(self, python_decision: Decision, rust_decision: Decision) -> KernelOutcome:
        outcome = KernelOutcome(
            source="rust", compared=True,
            python_action=python_decision.action.value,
            rust_action=rust_decision.action.value,
            python_q=python_decision.q, rust_q=rust_decision.q)
        if python_decision.status != "OK":
            # Python declining to price is not a disagreement about the
            # answer; there is no answer to disagree with.
            outcome.agreed = True
            outcome.compared = False
            outcome.reason = f"python status {python_decision.status}"
            return outcome
        if python_decision.action is not rust_decision.action:
            outcome.agreed = False
            outcome.reason = (f"action: python {python_decision.action.value}, "
                              f"rust {rust_decision.action.value}")
            return outcome
        if not _close(python_decision.q, rust_decision.q, self.q_tolerance):
            outcome.agreed = False
            outcome.reason = (f"Q on {python_decision.action.value}: python "
                              f"{python_decision.q!r}, rust {rust_decision.q!r}")
            return outcome
        outcome.agreed = True
        return outcome

    @staticmethod
    def _tagged(decision: Decision, outcome: KernelOutcome) -> Decision:
        decision.kernel = outcome.to_dict()
        return decision

    def report(self) -> Dict[str, Any]:
        """Whether the canonical path is on Rust, and on what evidence."""
        total = self.rust_decisions + self.python_decisions
        return {
            "schema": T0_KERNEL_SCHEMA_VERSION,
            "mode": self.mode.value,
            "native": self.native_status,
            "rust_available": self.rust_available,
            "rust_authoritative": self.rust_authoritative,
            "promote_after": self.promote_after,
            "consecutive_agreements": self.consecutive_agreements,
            "compared": self.compared,
            "agreements": self.agreements,
            "divergences": self.divergences,
            "rust_errors": self.rust_errors,
            "decisions_by_rust": self.rust_decisions,
            "decisions_by_python": self.python_decisions,
            # Named separately from agreement: "Rust was not asked" and "Rust
            # agreed" are different facts, and merging them overstates
            # coverage.
            "not_expressible_in_kernel": self.not_expressible,
            "without_survival_inputs": self.no_survival_inputs,
            "rust_share": (self.rust_decisions / total) if total else None,
            "demoted_reason": self.demoted_reason,
            "divergence_examples": list(self.divergence_examples),
            "status": ("OK" if self.rust_available and not self.demoted_reason
                       else "DATA_BLOCKED"),
        }


def _close(left: float, right: float, tolerance: float) -> bool:
    if math.isinf(left) and math.isinf(right):
        return (left > 0) == (right > 0)
    if math.isnan(left) or math.isnan(right):
        return False
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))
