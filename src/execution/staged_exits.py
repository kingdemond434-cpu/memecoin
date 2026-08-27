"""Exit transactions built before they are needed.

The escape path has a hard shape: something changes, the position has to be
out, and every microsecond between the decision and the signature is spent
while the thing we are escaping continues happening. Building the instruction
at that moment means deriving accounts, encoding arguments and assembling a
message inside the window where none of it can be afforded.

Almost all of that work is knowable in advance. The accounts for a sell do not
depend on the amount; the discriminator does not; the program does not. What
changes between "now" and "when we need it" is exactly two numbers -- the
token amount and its protective minimum -- and one of them is a function of
the other and the current curve.

So a position gets a LADDER at the moment it opens: 10%, 25%, 50%, 75% and a
full exit, each with its accounts derived and cached. When an exit fires, the
work left is encoding two u64s into a prepared instruction. When the curve
moves, only the protective bounds are refreshed, and refreshing them is a
local quote with no round trip.

Three properties this deliberately keeps:

**A stale bound is never used.** A protective minimum computed against a curve
from thirty seconds ago is not protection, it is an invitation. Each rung
carries the state version it was priced against, and a rung priced against a
superseded state is repriced before use -- which costs one local quote, not a
round trip, and is still far cheaper than building from nothing.

**A staged rung is never a decision.** This holds instructions, not intent.
Nothing here decides to sell, and a rung existing must never be a reason to
use it: the action policy decides, and this only ensures that when it does,
the transaction is already most of the way built.

**Emergency is not a separate path.** The 100% rung is the same machinery as
the others. A distinct emergency path is a path that is exercised only during
emergencies, which is the worst possible test schedule.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

STAGED_EXIT_SCHEMA_VERSION = "v1"

# The rungs. Chosen to match the BANK actions the policy can take, so every
# action it may choose has something already built for it -- a ladder that
# does not cover the policy's own action set is a ladder that will be missed
# on exactly the action that was chosen.
DEFAULT_LADDER: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)

# How long a protective bound is trusted before it is repriced. Short: this is
# the number that decides how much worse than expected a fill may be, and one
# priced against a curve that has since moved is not protection.
DEFAULT_BOUND_MAX_AGE_S = 2.0


@dataclass
class StagedRung:
    """One prepared sell, and what it was priced against."""

    fraction: float
    size_tokens: int
    instruction: Any
    min_proceeds: int
    priced_at: float
    state_version: int
    venue: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.instruction is not None and getattr(self.instruction, "ok", False)

    def is_stale(self, state_version: int, now: Optional[float] = None,
                 max_age_s: float = DEFAULT_BOUND_MAX_AGE_S) -> bool:
        """Whether this rung's protective bound can still be trusted."""
        if state_version != self.state_version:
            return True
        return (now if now is not None else time.time()) - self.priced_at > max_age_s

    def to_dict(self) -> Dict[str, Any]:
        return {"fraction": self.fraction, "size_tokens": self.size_tokens,
                "min_proceeds": self.min_proceeds, "priced_at": self.priced_at,
                "state_version": self.state_version, "venue": self.venue,
                "ok": self.ok, "detail": self.detail}


@dataclass
class StagedLadder:
    """Every rung for one position."""

    token: str
    rungs: Dict[float, StagedRung] = field(default_factory=dict)
    built_at: float = 0.0
    size_tokens: int = 0
    detail: str = ""

    def rung_for(self, fraction: float) -> Optional[StagedRung]:
        """The rung at or just below this fraction.

        Below, never above. Rounding an exit UP to the next rung sells more of
        the position than the policy chose, which is a size decision this
        module has no business making.
        """
        eligible = [value for value in sorted(self.rungs) if value <= fraction + 1e-9]
        return self.rungs[eligible[-1]] if eligible else None

    def to_dict(self) -> Dict[str, Any]:
        return {"token": self.token, "built_at": self.built_at,
                "size_tokens": self.size_tokens, "detail": self.detail,
                "rungs": [rung.to_dict() for _, rung in sorted(self.rungs.items())]}


class StagedExits:
    """Keeps a prepared exit ladder per open position.

    Owns no policy and no capital. It is told a position exists, it prepares;
    it is told the state moved, it reprices; it is asked for a rung, it hands
    over an instruction or says why it cannot.
    """

    def __init__(self, *, ladder: Sequence[float] = DEFAULT_LADDER,
                 bound_max_age_s: float = DEFAULT_BOUND_MAX_AGE_S,
                 max_positions: int = 500):
        self.ladder = tuple(sorted({round(float(value), 4) for value in ladder
                                    if 0 < float(value) <= 1.0}))
        self.bound_max_age_s = float(bound_max_age_s)
        self.max_positions = int(max_positions)
        self._ladders: Dict[str, StagedLadder] = {}
        self.built = 0
        self.repriced = 0
        self.served = 0
        self.missed = 0
        self.stale_served = 0
        self.build_failures = 0

    # --- lifecycle -------------------------------------------------------

    def stage(self, token: str, size_tokens: int, *, state_version: int,
              build_sell: Callable[[int, int], Any],
              quote_sell: Callable[[int], Optional[int]],
              slippage_bps: int, venue: str = "",
              now: Optional[float] = None) -> StagedLadder:
        """Build every rung for a position of ``size_tokens``.

        ``build_sell(size, min_proceeds)`` returns a prepared instruction;
        ``quote_sell(size)`` returns local proceeds in the quote unit, or None
        when the venue cannot answer. Both are injected so this module holds
        no venue knowledge -- the ladder is the same shape on a curve and on a
        pool, and a second copy of the routing rules is a second thing to keep
        in step.
        """
        now = time.time() if now is not None else time.time()
        ladder = StagedLadder(token=token, built_at=now, size_tokens=int(size_tokens))
        if size_tokens <= 0:
            ladder.detail = "position holds no tokens"
            self._ladders[token] = ladder
            return ladder
        for fraction in self.ladder:
            size = int(size_tokens * fraction)
            if size <= 0:
                continue
            rung = self._build_rung(fraction, size, state_version, build_sell,
                                    quote_sell, slippage_bps, venue, now)
            if rung is not None:
                ladder.rungs[fraction] = rung
        ladder.detail = f"{len(ladder.rungs)} of {len(self.ladder)} rungs prepared"
        self._ladders[token] = ladder
        self.built += len(ladder.rungs)
        self._evict()
        return ladder

    def _build_rung(self, fraction: float, size: int, state_version: int,
                    build_sell: Callable[[int, int], Any],
                    quote_sell: Callable[[int], Optional[int]],
                    slippage_bps: int, venue: str,
                    now: float) -> Optional[StagedRung]:
        proceeds = None
        try:
            proceeds = quote_sell(size)
        except Exception as exc:
            logger.debug("staging quote failed for %.0f%%: %s", fraction * 100, exc)
        if not proceeds or proceeds <= 0:
            self.build_failures += 1
            return None
        minimum = int(proceeds * (1 - max(0, slippage_bps) / 10_000))
        if minimum <= 0:
            # An unbounded sell is not a prepared sell. Refuse rather than
            # stage something that would authorise any price.
            self.build_failures += 1
            return None
        try:
            instruction = build_sell(size, minimum)
        except Exception as exc:
            self.build_failures += 1
            logger.debug("staging build failed for %.0f%%: %s", fraction * 100, exc)
            return None
        if instruction is None or not getattr(instruction, "ok", False):
            self.build_failures += 1
            return None
        return StagedRung(fraction=fraction, size_tokens=size, instruction=instruction,
                          min_proceeds=minimum, priced_at=now,
                          state_version=state_version, venue=venue)

    def reprice(self, token: str, *, state_version: int,
                build_sell: Callable[[int, int], Any],
                quote_sell: Callable[[int], Optional[int]],
                slippage_bps: int, venue: str = "",
                now: Optional[float] = None) -> int:
        """Refresh the bounds of a staged ladder against the current state.

        Only the volatile part. The accounts do not depend on the amount, so
        the expensive half of the work survives the market moving.
        """
        ladder = self._ladders.get(token)
        if ladder is None:
            return 0
        now = time.time() if now is None else now
        refreshed = 0
        for fraction, rung in list(ladder.rungs.items()):
            if not rung.is_stale(state_version, now, self.bound_max_age_s):
                continue
            rebuilt = self._build_rung(fraction, rung.size_tokens, state_version,
                                       build_sell, quote_sell, slippage_bps, venue, now)
            if rebuilt is not None:
                ladder.rungs[fraction] = rebuilt
                refreshed += 1
        self.repriced += refreshed
        return refreshed

    def release(self, token: str) -> None:
        self._ladders.pop(token, None)

    def _evict(self) -> None:
        while len(self._ladders) > self.max_positions:
            self._ladders.pop(next(iter(self._ladders)))

    # --- use -------------------------------------------------------------

    def take(self, token: str, fraction: float, *, state_version: int,
             now: Optional[float] = None) -> Tuple[Optional[StagedRung], str]:
        """The prepared rung for this exit, or why there is none.

        Returns the rung and a status. A STALE rung is returned with its
        status said plainly rather than withheld: the caller can reprice it
        with one local quote, which is still far cheaper than building from
        nothing, and withholding it would send the escape down the slow path
        for a reason that costs a millisecond to fix.
        """
        ladder = self._ladders.get(token)
        if ladder is None:
            self.missed += 1
            return None, "no ladder staged for this position"
        rung = ladder.rung_for(fraction)
        if rung is None:
            self.missed += 1
            return None, f"no rung at or below {fraction:.0%}"
        if rung.is_stale(state_version, now, self.bound_max_age_s):
            self.stale_served += 1
            return rung, "STALE: bound was priced against a superseded state"
        self.served += 1
        return rung, "OK"

    def ladder_for(self, token: str) -> Optional[StagedLadder]:
        return self._ladders.get(token)

    def report(self) -> Dict[str, Any]:
        """Whether exits are actually being served from the ladder.

        A ladder built for every position and used for none is pure cost.
        `served` against `missed` is the number that says which this is.
        """
        attempts = self.served + self.missed + self.stale_served
        return {
            "schema": STAGED_EXIT_SCHEMA_VERSION,
            "status": "OK" if self.served else "DATA_BLOCKED",
            "ladder": list(self.ladder),
            "staged_positions": len(self._ladders),
            "rungs_built": self.built, "rungs_repriced": self.repriced,
            "served": self.served, "served_stale": self.stale_served,
            "missed": self.missed, "build_failures": self.build_failures,
            "hit_rate": (self.served / attempts) if attempts else None,
            "bound_max_age_s": self.bound_max_age_s,
        }
