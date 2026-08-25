"""Re-entry as a real post-exit candidate, not an enum value.

`Action.REENTER` existed and could never be chosen. `ActionValuePolicy`
scores it only when `held_fraction == 0`, and the only caller was the loop
over open positions -- where the held fraction is positive by construction.
So the action was scored, always returned -inf, and every log line said
"position is still open; re-entry is not the question". The enum member was a
description of an intention, not a path capital could take.

Re-entry is not a variation on holding. It is a new trade in a token we have
information about, and the information cuts both ways:

    We know why we left.

That is the whole asymmetry. A token exited because the ratchet banked a
winner is a different proposition from a token exited because the hazard
engine said it was about to die, and both are different from a token closed
to fund something better. Treating them alike is how a book gives back a
banked run by buying it again at a worse price, and how it walks back into
the exact rug it just escaped.

So this module does four things and refuses to do a fifth.

It remembers the disposition of every full exit -- not the price, the
*reason*. It gates admission on the reason: some exits bar re-entry
permanently, some require the thing we fled to have measurably receded, and
all require a cooldown, because the price we would buy back at is partly our
own exit's market impact and re-entering inside it means paying for our own
sale twice.

It charges a re-entry premium. Holding through would have cost nothing; the
round trip costs an exit and an entry, so the floor is that round trip, and
above the floor sits a multiplier that grows with how adverse the exit reason
was and with how many times this token has already cycled us. A token we have
round-tripped three times is not an opportunity, it is a counterparty.

It prices the candidate on a distribution built AFTER the exit. A prediction
timestamped at or before the exit is refused outright rather than discounted:
re-using it would let the token inherit the conviction that existed before
the evidence that made us sell, which is the single most flattering thing
this module could do for itself.

And it does not execute. The verdict it returns is a hurdle applied to the
ordinary entry path, so a re-entry competes for capital against every fresh
launch on the same cross-sectional score, sized by the same engine, subject
to the same exposure caps. A re-entry that cannot beat a new launch is not
owed the slot by the fact that we used to own it.

The disposition multipliers below are policy, not measurement. They are
stated once, in one place, versioned, and replaced wholesale by a trained
model when one exists -- the same treatment the analytic action-value weights
get, and for the same reason: an inspectable prior is honest, a prior
smuggled into five call sites is not.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.strategies.action_value import Action, ActionValuePolicy, PositionState
from src.strategies.escape import UNESCAPABLE_MECHANISMS
from src.strategies.opportunity_allocator import Opportunity

logger = logging.getLogger(__name__)

REENTRY_SCHEMA_VERSION = "v1"
REENTRY_SLEEVE = "reentry"


class ExitDisposition(Enum):
    """Why the position was closed. The only input that makes re-entry differ."""

    BANKED_STRENGTH = "banked_strength"
    DISPLACED = "displaced"
    DISTRIBUTION = "distribution"
    HAZARD_ESCAPE = "hazard_escape"
    CATASTROPHIC = "catastrophic"
    UNKNOWN = "unknown"


# Exiting because the token was collapsing in a way we could not have escaped
# is not a trade we get to run again on better terms. There is no premium high
# enough to make it correct, so it is barred rather than priced.
BARRED_DISPOSITIONS = frozenset({ExitDisposition.CATASTROPHIC})

# Dispositions where the reason we left is a live claim about the token, so
# re-entry requires that claim to have measurably weakened -- not merely aged.
EVIDENCE_REQUIRED = frozenset({ExitDisposition.HAZARD_ESCAPE,
                               ExitDisposition.DISTRIBUTION})

# Multiples of the round-trip cost floor. Ordered by how adverse the exit was.
DISPOSITION_PREMIUM = {
    ExitDisposition.BANKED_STRENGTH: 1.0,
    ExitDisposition.DISPLACED: 1.0,
    ExitDisposition.UNKNOWN: 3.0,
    ExitDisposition.DISTRIBUTION: 3.0,
    ExitDisposition.HAZARD_ESCAPE: 5.0,
}


def classify_exit(reason: str) -> ExitDisposition:
    """Map an exit reason string onto a disposition.

    Unrecognised reasons become UNKNOWN, which pays a mid premium and carries
    no evidence requirement. That split is deliberate: demanding that a hazard
    have measurably fallen, when we cannot name what we fled, would block
    re-entry on a technicality rather than on evidence. Not knowing why we
    left is priced, not litigated.
    """
    text = str(reason or "").lower()
    if "catastrophic" in text:
        return ExitDisposition.CATASTROPHIC
    if text.startswith("rug_hazard") or "hazard" in text or "escape" in text:
        return ExitDisposition.HAZARD_ESCAPE
    if "distribution" in text or "dump" in text or "insider" in text:
        return ExitDisposition.DISTRIBUTION
    if "displace" in text or "replace" in text or "contest" in text:
        return ExitDisposition.DISPLACED
    if ("ratchet" in text or "bank" in text or text.startswith("action_")
            or "trail" in text or "monster_" in text or "harvest" in text):
        return ExitDisposition.BANKED_STRENGTH
    return ExitDisposition.UNKNOWN


@dataclass(frozen=True)
class ReentryPolicy:
    """The bar re-entry must clear, stated once.

    ``cooldown_seconds`` is not a superstition about momentum. Our own exit
    consumed the book on the way out; the quote immediately after it is partly
    our own impact, and buying back into it pays that impact a second time.

    ``window_seconds`` bounds how long an exit stays informative at all. Past
    it the token is simply a token we have no position in, and it re-enters
    the world as a fresh candidate through the ordinary path.
    """

    cooldown_seconds: float = 90.0
    window_seconds: float = 1800.0
    max_reentries: int = 2
    min_hazard_improvement: float = 0.25
    # A re-entry whose forward distribution cannot even pay the round trip is
    # rejected before any premium is considered.
    absolute_floor: float = 0.0

    def premium_multiplier(self, disposition: ExitDisposition, reentries: int) -> float:
        base = DISPOSITION_PREMIUM.get(disposition, DISPOSITION_PREMIUM[ExitDisposition.UNKNOWN])
        return base * (1.0 + max(0, int(reentries)))


@dataclass
class ExitRecord:
    """What we knew at the moment we closed the position."""

    token: str
    exited_at: float
    disposition: ExitDisposition
    reason: str = ""
    exit_multiple: float = 1.0
    realized_pnl_usd: float = 0.0
    # The hazard reading that justified the exit, where one existed. None is
    # not zero: an exit taken without a hazard measurement cannot later be
    # cleared by claiming the hazard improved.
    hazard_at_exit: Optional[float] = None
    mechanism_at_exit: Optional[Any] = None
    reentries: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token, "exited_at": self.exited_at,
            "disposition": self.disposition.value, "reason": self.reason,
            "exit_multiple": self.exit_multiple,
            "realized_pnl_usd": self.realized_pnl_usd,
            "hazard_at_exit": self.hazard_at_exit,
            "reentries": self.reentries,
        }


@dataclass
class ReentryVerdict:
    """Whether this token may be re-entered, and at what bar.

    ``status`` is OK, REJECTED or DATA_BLOCKED, and the three are not
    interchangeable. REJECTED means we measured and the answer was no.
    DATA_BLOCKED means we could not measure, which is also not a yes.
    """

    token: str
    status: str
    detail: str = ""
    disposition: ExitDisposition = ExitDisposition.UNKNOWN
    q: Optional[float] = None
    required_q: Optional[float] = None
    reentries: int = 0
    opportunity: Optional[Opportunity] = None

    @property
    def admitted(self) -> bool:
        return self.status == "OK"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token, "status": self.status, "detail": self.detail,
            "disposition": self.disposition.value, "q": self.q,
            "required_q": self.required_q, "reentries": self.reentries,
        }


class ReentryBook:
    """The set of tokens we have exited and might legitimately buy back.

    Bounded by construction: records outside the window are dropped on every
    touch, and the book is capped, because an unbounded memory of every exit
    is a slow leak in a process that is expected to run for weeks.
    """

    def __init__(self, policy: Optional[ReentryPolicy] = None, capacity: int = 512,
                 action_policy: Optional[ActionValuePolicy] = None):
        self.policy = policy or ReentryPolicy()
        self.capacity = max(1, int(capacity))
        self.action_policy = action_policy or ActionValuePolicy()
        self._records: Dict[str, ExitRecord] = {}

    # -- book keeping ------------------------------------------------------

    def record_exit(self, token: str, reason: str, *, exited_at: Optional[float] = None,
                    exit_multiple: float = 1.0, realized_pnl_usd: float = 0.0,
                    hazard_at_exit: Optional[float] = None,
                    mechanism_at_exit: Optional[Any] = None) -> ExitRecord:
        """Register a FULL close. Partial banks are not exits and must not land here.

        A partial bank leaves the position open, so 're-entry' into it is
        simply adding -- which the action-value policy already prices as ADD
        against the live position. Recording one here would let ADD be
        re-litigated through a path that thinks the position is flat.
        """
        now = time.time() if exited_at is None else float(exited_at)
        prior = self._records.get(token)
        record = ExitRecord(
            token=token, exited_at=now, disposition=classify_exit(reason),
            reason=str(reason or ""), exit_multiple=float(exit_multiple),
            realized_pnl_usd=float(realized_pnl_usd),
            hazard_at_exit=hazard_at_exit, mechanism_at_exit=mechanism_at_exit,
            # The count survives the new record. Resetting it on each exit is
            # how a token that cycles us five times looks like a first-timer
            # every single time.
            reentries=prior.reentries if prior else 0,
        )
        self._records[token] = record
        self.prune(now)
        return record

    def note_reentry(self, token: str) -> None:
        """Called when a re-entry actually fills. Raises the bar for the next one."""
        record = self._records.get(token)
        if record is not None:
            record.reentries += 1

    def forget(self, token: str) -> None:
        self._records.pop(token, None)

    def get(self, token: str) -> Optional[ExitRecord]:
        return self._records.get(token)

    def prune(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else float(now)
        stale = [token for token, record in self._records.items()
                 if now - record.exited_at > self.policy.window_seconds]
        for token in stale:
            self._records.pop(token, None)
        # Capacity is enforced against the oldest exits, which are also the
        # least informative -- the opposite order to dropping recent ones.
        while len(self._records) > self.capacity:
            oldest = min(self._records, key=lambda key: self._records[key].exited_at)
            self._records.pop(oldest, None)
            stale.append(oldest)
        return len(stale)

    def candidates(self, now: Optional[float] = None) -> List[ExitRecord]:
        """Records still inside the window and not permanently barred."""
        now = time.time() if now is None else float(now)
        self.prune(now)
        return [record for record in self._records.values()
                if record.disposition not in BARRED_DISPOSITIONS]

    # -- gates -------------------------------------------------------------

    def admits(self, token: str, *, now: Optional[float] = None,
               hazard_now: Optional[float] = None,
               mechanism_now: Optional[Any] = None) -> ReentryVerdict:
        """The cheap gates, in order. Passing means "worth re-evaluating", not "buy".

        Deliberately separate from pricing so the expensive path -- a fresh
        prediction, a fresh risk report, a fresh liquidity probe -- is never
        run for a token that a cooldown or a permanent bar already excludes.
        """
        now = time.time() if now is None else float(now)
        record = self._records.get(token)
        if record is None:
            # Never held, or the window closed. Either way this is not a
            # re-entry and the ordinary entry path owns the decision.
            return ReentryVerdict(token=token, status="OK", detail="not a post-exit candidate")

        verdict = lambda status, detail: ReentryVerdict(  # noqa: E731
            token=token, status=status, detail=detail,
            disposition=record.disposition, reentries=record.reentries)

        if record.disposition in BARRED_DISPOSITIONS:
            return verdict("REJECTED", f"{record.disposition.value} exits are never re-entered")
        for mechanism in (record.mechanism_at_exit, mechanism_now):
            if mechanism is not None and mechanism in UNESCAPABLE_MECHANISMS:
                return verdict("REJECTED",
                               f"{getattr(mechanism, 'value', mechanism)} is unescapable")

        age = now - record.exited_at
        if age > self.policy.window_seconds:
            self.forget(token)
            return ReentryVerdict(token=token, status="OK", detail="re-entry window closed")
        if age < self.policy.cooldown_seconds:
            return verdict("REJECTED",
                           f"inside our own exit impact ({age:.1f}s of "
                           f"{self.policy.cooldown_seconds:.0f}s cooldown)")
        if record.reentries >= self.policy.max_reentries:
            return verdict("REJECTED",
                           f"already re-entered {record.reentries} times")

        if record.disposition in EVIDENCE_REQUIRED:
            if record.hazard_at_exit is None:
                return verdict("DATA_BLOCKED",
                               "exited on evidence that was never quantified; "
                               "nothing to show has improved")
            if hazard_now is None:
                return verdict("DATA_BLOCKED", "hazard not re-measured since the exit")
            ceiling = record.hazard_at_exit * (1.0 - self.policy.min_hazard_improvement)
            if float(hazard_now) > ceiling:
                return verdict("REJECTED",
                               f"hazard {float(hazard_now):.4f} has not fallen below "
                               f"{ceiling:.4f}; the reason we left still holds")

        return verdict("OK", "admitted for re-evaluation")

    # -- pricing -----------------------------------------------------------

    def price(self, token: str, *, bins: Sequence[Tuple[float, float]],
              size_fraction: float, capital_usd: float,
              expected_hold_seconds: Optional[float], liquidity_usd: Optional[float],
              exit_capacity_ratio: Optional[float], escape_probability: Optional[float],
              prediction_at: float, entry_cost: float, exit_cost: float,
              now: Optional[float] = None) -> ReentryVerdict:
        """Score the re-entry on a fresh distribution and charge the premium.

        ``bins`` MUST come from a prediction made after the exit. That is
        checked against ``prediction_at`` rather than trusted, because the
        failure it prevents is silent: a stale distribution re-prices the
        token at the conviction it had before the evidence that made us sell.
        """
        now = time.time() if now is None else float(now)
        record = self._records.get(token)
        if record is None:
            return ReentryVerdict(token=token, status="OK", detail="not a post-exit candidate")

        verdict = lambda status, detail, **kwargs: ReentryVerdict(  # noqa: E731
            token=token, status=status, detail=detail,
            disposition=record.disposition, reentries=record.reentries, **kwargs)

        if float(prediction_at) <= record.exited_at:
            return verdict("DATA_BLOCKED",
                           "prediction predates the exit; re-entry may not inherit "
                           "the conviction the exit contradicted")
        if size_fraction <= 0:
            return verdict("DATA_BLOCKED", "no re-entry size")
        if exit_capacity_ratio is None:
            return verdict("DATA_BLOCKED", "exit capacity not measured")
        if escape_probability is None:
            return verdict("DATA_BLOCKED", "escape probability not measured")

        state = PositionState(
            held_fraction=0.0,
            current_multiple=1.0,
            # With nothing held, holding is all cash and scores exactly zero
            # whatever these bins say -- so Q(REENTER) is the re-entry's own
            # E[log W], which is precisely the number the premium is charged
            # against. The field is populated because the state validator
            # requires a well-formed distribution, not because it is consulted.
            forward_bins=tuple(bins),
            exit_cost=float(exit_cost),
            entry_cost=float(entry_cost),
            exit_capacity_ratio=float(exit_capacity_ratio),
            escape_probability=float(escape_probability),
            add_fraction=float(size_fraction),
            reentry_bins=tuple(bins),
        )
        decision = self.action_policy.score(state)
        if decision.status != "OK":
            return verdict("DATA_BLOCKED", f"unpriceable: {decision.detail}")
        q = decision.score_of(Action.REENTER)
        if q is None:
            reenter = next((score for score in decision.scores
                            if score.action is Action.REENTER), None)
            return verdict("DATA_BLOCKED",
                           f"re-entry unpriceable: {reenter.detail if reenter else 'no score'}")

        # The floor is what holding through would not have cost: one exit and
        # one entry on the capital actually committed.
        round_trip = float(size_fraction) * (float(entry_cost) + float(exit_cost))
        required = max(self.policy.absolute_floor,
                       round_trip * self.policy.premium_multiplier(
                           record.disposition, record.reentries))
        if q <= required:
            return verdict("REJECTED",
                           f"re-entry edge {q:.6f} does not clear the "
                           f"{record.disposition.value} bar {required:.6f}",
                           q=q, required_q=required)

        opportunity = Opportunity(
            token=token,
            # The allocator ranks on edge net of what this trade costs the
            # book, so it is given the premium-adjusted number rather than the
            # gross one. Handing it the gross Q would let a re-entry that only
            # just cleared its own bar outrank a fresh launch that cleared a
            # higher one.
            elogw=q - required,
            capital_usd=float(capital_usd),
            expected_hold_seconds=expected_hold_seconds,
            liquidity_usd=liquidity_usd,
            sleeve=REENTRY_SLEEVE,
            metadata={"reentry": True, "disposition": record.disposition.value,
                      "gross_q": q, "required_q": required,
                      "prior_exit_multiple": record.exit_multiple,
                      "prior_exit_reason": record.reason,
                      "reentries": record.reentries,
                      "seconds_since_exit": now - record.exited_at},
        )
        return verdict("OK", f"re-entry clears the {record.disposition.value} bar",
                       q=q, required_q=required, opportunity=opportunity)

    # -- reporting ---------------------------------------------------------

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else float(now)
        self.prune(now)
        by_disposition: Dict[str, int] = {}
        for record in self._records.values():
            key = record.disposition.value
            by_disposition[key] = by_disposition.get(key, 0) + 1
        return {
            "schema": REENTRY_SCHEMA_VERSION,
            "tracked": len(self._records),
            "capacity": self.capacity,
            "by_disposition": by_disposition,
            "barred": sorted(record.token for record in self._records.values()
                             if record.disposition in BARRED_DISPOSITIONS),
            "cooldown_seconds": self.policy.cooldown_seconds,
            "window_seconds": self.policy.window_seconds,
            "max_reentries": self.policy.max_reentries,
        }
