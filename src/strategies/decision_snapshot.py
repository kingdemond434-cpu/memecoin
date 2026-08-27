"""Freezing a decision so it executes against the state that produced it.

The bug this exists to prevent is subtle and does not look like a bug. The
action policy scores ADD using one `plan_scale_in` result; `_consider_scale_in`
then refreshes portfolio state and calls `plan_scale_in` AGAIN before
submitting. Both calls are correct. Neither is the decision. What executes is a
size computed from a market state the policy never evaluated, and the two
diverge exactly when the market is moving -- which is the only time the size
matters.

Nothing in the logs shows this. The decision is recorded, the fill is recorded,
and they refer to different instants.

So a decision becomes an immutable object: the action, the exact size, the
protective limit, the state sequence it was computed from, and hashes of the
features and model that produced it. Execution takes that object and submits
exactly it. If newer state has arrived, the decision is stale and must be
repriced rather than executed at its old size -- and the reprice is visible as
a reprice, not as the same decision arriving late.

Expiry is separate from staleness on purpose. A decision can be the newest one
and still be too old to act on: on a launch moving in hundreds of milliseconds,
a 4-second-old size is wrong even if nothing has superseded it.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DECISION_SCHEMA_VERSION = "v1"

# Default life of a decision. Short because the state it was computed from
# ages in the same milliseconds the launch does.
DEFAULT_EXPIRY_SECONDS = 1.5


class DecisionStatus(Enum):
    VALID = "valid"
    STALE = "stale"
    EXPIRED = "expired"
    CONSUMED = "consumed"


def state_hash(payload: Dict[str, Any]) -> str:
    """Stable hash of the inputs a decision was computed from."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class DecisionSnapshot:
    """One frozen, executable decision.

    Deliberately carries the exact size and the exact protective limit rather
    than the inputs needed to recompute them. Anything that can be recomputed
    at execution time will be, and then the executed trade is not the decided
    trade.
    """

    token: str
    action: str
    state_seq: int
    size_base_units: int
    # max_sol_cost on a buy, min_sol_output on a sell. The slippage bound the
    # decision was made under, not one derived later from a moved price.
    protective_limit: int
    feature_hash: str
    model_hash: str
    created_at: float = field(default_factory=time.time)
    expiry_seconds: float = DEFAULT_EXPIRY_SECONDS
    q_value: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    _consumed: bool = field(default=False, repr=False)

    @property
    def decision_id(self) -> str:
        return state_hash({
            "token": self.token, "action": self.action, "seq": self.state_seq,
            "size": self.size_base_units, "created_at": self.created_at,
        })

    def status(self, current_seq: int, now: Optional[float] = None) -> DecisionStatus:
        """Whether this decision may still be executed as written."""
        now = time.time() if now is None else now
        if self._consumed:
            return DecisionStatus.CONSUMED
        if current_seq != self.state_seq:
            return DecisionStatus.STALE
        # Checked after staleness so the more specific cause is reported, but
        # independent of it: the newest decision can still be too old to act on.
        if now - self.created_at > self.expiry_seconds:
            return DecisionStatus.EXPIRED
        return DecisionStatus.VALID

    def consume(self) -> None:
        """Mark as executed. A decision may be acted on exactly once.

        Without this a retry loop can submit the same decision twice, which on
        a buy is a double position and on a sell is an oversell.
        """
        self._consumed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": DECISION_SCHEMA_VERSION,
            "decision_id": self.decision_id, "token": self.token,
            "action": self.action, "state_seq": self.state_seq,
            "size_base_units": self.size_base_units,
            "protective_limit": self.protective_limit,
            "feature_hash": self.feature_hash, "model_hash": self.model_hash,
            "created_at": self.created_at, "expiry_seconds": self.expiry_seconds,
            "q_value": self.q_value, "evidence": dict(self.evidence),
        }


class StateSequencer:
    """Monotonic counter bumped whenever state a decision depends on changes.

    Comparing sequence numbers rather than timestamps is deliberate: two
    updates inside one clock tick are indistinguishable by time and perfectly
    distinguishable by sequence, and it is precisely under load that they
    arrive in the same tick.
    """

    def __init__(self):
        self._sequences: Dict[str, int] = {}

    def current(self, token: str) -> int:
        return self._sequences.get(token, 0)

    def bump(self, token: str) -> int:
        seq = self._sequences.get(token, 0) + 1
        self._sequences[token] = seq
        return seq

    def reset(self, token: str) -> None:
        self._sequences.pop(token, None)


@dataclass
class ExecutionOutcome:
    executed: bool
    status: DecisionStatus
    detail: str = ""


def guard(snapshot: DecisionSnapshot, sequencer: StateSequencer,
          now: Optional[float] = None) -> ExecutionOutcome:
    """The single check every execution path must pass.

    Returns rather than raises so the caller records the refusal: a decision
    that went stale is evidence about how fast state is moving relative to how
    fast decisions are made, and that ratio is a latency measurement worth
    keeping.
    """
    status = snapshot.status(sequencer.current(snapshot.token), now)
    if status is DecisionStatus.VALID:
        return ExecutionOutcome(True, status, "decision matches current state")
    return ExecutionOutcome(False, status, {
        DecisionStatus.STALE: "state advanced after this decision was made; reprice",
        DecisionStatus.EXPIRED: "decision older than its expiry; reprice",
        DecisionStatus.CONSUMED: "decision was already executed",
    }[status])
