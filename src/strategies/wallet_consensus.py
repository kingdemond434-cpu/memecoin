"""Agreement between independent wallets, as hypotheses rather than as a rule.

"Three good wallets bought the same mint within twenty seconds" is a plausible
signal and a terrible constant. Three, twenty and good are all choices, and a
system that hard-codes them has not discovered anything -- it has encoded
somebody's guess and will defend it with whatever returns follow.

So this fires a GRID. Every combination of how many deciders, inside what
window, measured or merely scored, with or without the deployer selling, is a
separately named mechanism, and each one accumulates its own observations for
the gauntlet to kill or keep. The module's job is to say which rules fired on
which launch; nothing here decides to trade, and no rule in the grid is
privileged over any other before the evidence arrives.

Two refusals, both of which change the answer:

**Wallets are not deciders.** Ten addresses funded from one source that enter
over four slots are one person agreeing with themselves. Counting them as ten
manufactures exactly the signal being looked for, which makes this the one
place in the desk where unmeasured independence must BLOCK rather than default
to independent. Elsewhere -- the ring-compression feature, for instance -- the
no-evidence answer is full independence, because penalising a launch for a
ring nobody detected would be inventing the ring. Here the symmetric error is
inventing the consensus, and that one is paid for in capital.

**Measured followability, not a score.** A wallet counts toward agreement when
following it has been SHOWN to grow capital. Counting wallets that merely
score well on a composite means the grid is testing the composite, and it will
find that the composite predicts itself.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (Any, Callable, Deque, Dict, Iterable, List, Optional,
                    Sequence, Tuple)

logger = logging.getLogger(__name__)

CONSENSUS_SCHEMA_VERSION = "v1"

#: How many independent deciders a rule may require.
DECIDER_COUNTS: Tuple[int, ...] = (2, 3, 4)

#: The windows, in seconds, inside which their entries must fall.
WINDOWS_S: Tuple[float, ...] = (10.0, 20.0, 60.0)

#: How long a token's buy record is kept before it stops being a live launch.
DEFAULT_HORIZON_S = 300.0

#: Bound on tracked tokens, so a long-running desk does not accumulate a
#: deque per mint for the life of the process.
DEFAULT_MAX_TOKENS = 4_096


@dataclass(frozen=True)
class ConsensusRule:
    """One hypothesis about what agreement is worth, named for the gauntlet."""

    min_independent: int
    window_s: float
    #: Count only wallets whose followability is measured, never merely scored.
    require_measured: bool = True
    #: Refuse the signal when the deployer is distributing into it.
    forbid_deployer_selling: bool = False

    @property
    def mechanism(self) -> str:
        """Stable identity. The gauntlet scores one row per name, so this has
        to be a function of the rule alone and never of when it ran."""
        parts = [f"consensus:k{self.min_independent}",
                 f"w{self.window_s:g}s",
                 "measured" if self.require_measured else "any"]
        if self.forbid_deployer_selling:
            parts.append("nodevsell")
        return "/".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"mechanism": self.mechanism,
                "min_independent": self.min_independent,
                "window_s": self.window_s,
                "require_measured": self.require_measured,
                "forbid_deployer_selling": self.forbid_deployer_selling}


def default_rule_grid(counts: Sequence[int] = DECIDER_COUNTS,
                      windows: Sequence[float] = WINDOWS_S
                      ) -> Tuple[ConsensusRule, ...]:
    """The full cartesian grid. Deliberately not a shortlist.

    Shortlisting before the evidence is the thing this module exists to avoid,
    and the gauntlet prices the selection that testing a grid creates -- CSCV
    over the surviving candidates is exactly the right answer to "you tried
    thirty-six rules and kept the best one".
    """
    return tuple(
        ConsensusRule(min_independent=count, window_s=float(window),
                      require_measured=measured,
                      forbid_deployer_selling=nodev)
        for count, window, measured, nodev in itertools.product(
            counts, windows, (True, False), (True, False)))


@dataclass
class ConsensusBuy:
    wallet: str
    at: float
    confidence: Optional[float] = None
    measured: bool = False


@dataclass
class ConsensusSignal:
    """Which rules fired on this token, and everything that decided that."""

    status: str
    token: str = ""
    at: float = 0.0
    fired: Tuple[str, ...] = ()
    evaluated: int = 0
    #: Distinct followable wallets seen inside the widest window.
    followable_wallets: int = 0
    #: Those wallets collapsed to independent deciders.
    independent_deciders: Optional[int] = None
    measured_deciders: Optional[int] = None
    deployer_selling: Optional[bool] = None
    tightest_window_s: Optional[float] = None
    mean_confidence: Optional[float] = None
    blocked: List[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": CONSENSUS_SCHEMA_VERSION, "status": self.status,
                "token": self.token, "at": self.at, "fired": list(self.fired),
                "evaluated": self.evaluated,
                "followable_wallets": self.followable_wallets,
                "independent_deciders": self.independent_deciders,
                "measured_deciders": self.measured_deciders,
                "deployer_selling": self.deployer_selling,
                "tightest_window_s": self.tightest_window_s,
                "mean_confidence": self.mean_confidence,
                "blocked": list(self.blocked), "detail": self.detail}


class WalletConsensus:
    """Records who bought what, and reports which agreement rules that satisfies.

    ``followable_provider`` returns {wallet: confidence} for the wallets worth
    listening to, with measured ones above ``measured_floor``.
    ``independence_provider`` collapses a set of wallets to a count of
    independent deciders, returning (count, detail) with a None count when it
    cannot tell.
    """

    def __init__(self, *,
                 followable_provider: Callable[[], Dict[str, float]],
                 independence_provider: Optional[
                     Callable[[Sequence[str]], Tuple[Optional[int], Dict[str, Any]]]] = None,
                 rules: Sequence[ConsensusRule] = (),
                 horizon_s: float = DEFAULT_HORIZON_S,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 measured_floor: float = 0.50):
        self.followable_provider = followable_provider
        self.independence_provider = independence_provider
        self.rules = tuple(rules) or default_rule_grid()
        self.horizon_s = float(horizon_s)
        self.max_tokens = int(max_tokens)
        self.measured_floor = float(measured_floor)
        self._buys: Dict[str, Deque[ConsensusBuy]] = {}
        self._deployer_sold: Dict[str, float] = {}
        self._fired: Dict[str, int] = {}
        self.observed = 0

    # -- recording --------------------------------------------------------

    def observe_buy(self, token: str, wallet: str,
                    at: Optional[float] = None) -> bool:
        """Record a buy. Returns whether the wallet was one worth listening to.

        Buys from wallets nobody is following are dropped rather than stored:
        the question is whether the followable ones agree, and keeping the rest
        would make every busy launch look like consensus.
        """
        if not token or not wallet:
            return False
        confidences = self.followable_provider() or {}
        if wallet not in confidences:
            return False
        moment = time.time() if at is None else float(at)
        confidence = float(confidences[wallet])
        record = self._buys.setdefault(token, deque())
        # One entry per wallet per token. A wallet scaling in is still one
        # decider, and counting its three buys as three would be the same
        # error as counting a ring as three people.
        if any(item.wallet == wallet for item in record):
            return True
        record.append(ConsensusBuy(wallet=wallet, at=moment, confidence=confidence,
                                   measured=confidence >= self.measured_floor))
        self.observed += 1
        self._evict(moment)
        return True

    def observe_deployer_sell(self, token: str, at: Optional[float] = None) -> None:
        if not token:
            return
        moment = time.time() if at is None else float(at)
        self._deployer_sold[token] = moment
        # Deployer sells arrive for tokens no followable wallet ever touched,
        # so this map is not kept in step by the buy record's eviction and
        # would otherwise grow for the life of the process.
        if len(self._deployer_sold) > self.max_tokens:
            for stale in [key for key, when in self._deployer_sold.items()
                          if moment - when > self.horizon_s]:
                self._deployer_sold.pop(stale, None)

    def _evict(self, now: float) -> None:
        for token in list(self._buys):
            record = self._buys[token]
            while record and now - record[0].at > self.horizon_s:
                record.popleft()
            if not record:
                self._buys.pop(token, None)
                self._deployer_sold.pop(token, None)
        while len(self._buys) > self.max_tokens:
            oldest = min(self._buys, key=lambda key: self._buys[key][0].at)
            self._buys.pop(oldest, None)
            self._deployer_sold.pop(oldest, None)

    # -- evaluation -------------------------------------------------------

    def _deciders(self, wallets: Sequence[str]) -> Tuple[Optional[int], Dict[str, Any]]:
        """How many independent people these addresses represent.

        None when nothing can tell, and None must block the rules that depend
        on it. Assuming independence here would count one operator's ring as a
        crowd, which is the precise shape of the signal being looked for and
        therefore the precise shape of the mistake.
        """
        if not wallets:
            return 0, {"reason": "no_wallets"}
        provider = self.independence_provider
        if not callable(provider):
            return None, {"reason": "no_independence_provider"}
        try:
            count, detail = provider(list(wallets))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("independence provider failed: %s", exc)
            return None, {"reason": f"independence_raised:{exc}"}
        return (None if count is None else int(count)), dict(detail or {})

    @staticmethod
    def _tightest_window(times: Sequence[float], count: int) -> Optional[float]:
        """Smallest span containing ``count`` of these entries."""
        if count <= 0 or len(times) < count:
            return None
        ordered = sorted(times)
        return min(ordered[i + count - 1] - ordered[i]
                   for i in range(len(ordered) - count + 1))

    def evaluate(self, token: str, now: Optional[float] = None) -> ConsensusSignal:
        """Which rules this token currently satisfies."""
        moment = time.time() if now is None else float(now)
        record = [item for item in self._buys.get(token, ())
                  if moment - item.at <= self.horizon_s]
        signal = ConsensusSignal(status="OK", token=token, at=moment,
                                 evaluated=len(self.rules),
                                 followable_wallets=len(record))
        # Established before the early return: whether the deployer is
        # distributing is known independently of whether anybody followable
        # has bought, and a signal that reports it as None in that case is
        # claiming not to know something it does know.
        deployer_at = self._deployer_sold.get(token)
        signal.deployer_selling = (deployer_at is not None
                                   and moment - deployer_at <= self.horizon_s)
        if not record:
            signal.detail = "no followable wallet has bought this token"
            return signal

        measured = [item for item in record if item.measured]
        signal.mean_confidence = float(
            sum(item.confidence or 0.0 for item in record) / len(record))

        all_count, _ = self._deciders([item.wallet for item in record])
        measured_count, detail = self._deciders([item.wallet for item in measured])
        signal.independent_deciders = all_count
        signal.measured_deciders = measured_count
        if all_count is None or measured_count is None:
            signal.blocked.append("independence")
            signal.status = "DATA_BLOCKED"
            signal.detail = str(detail.get("reason", "independence unmeasurable"))
            return signal

        fired: List[str] = []
        for rule in self.rules:
            pool = measured if rule.require_measured else record
            deciders = measured_count if rule.require_measured else all_count
            if deciders < rule.min_independent:
                continue
            if rule.forbid_deployer_selling and signal.deployer_selling:
                continue
            # The window is checked on the entries themselves; the decider
            # count has already removed the duplicates a ring would add, so a
            # ring cannot satisfy a window it only fits because it is one
            # person entering repeatedly.
            span = self._tightest_window([item.at for item in pool],
                                         rule.min_independent)
            if span is None or span > rule.window_s:
                continue
            fired.append(rule.mechanism)
            self._fired[rule.mechanism] = self._fired.get(rule.mechanism, 0) + 1

        signal.fired = tuple(fired)
        signal.tightest_window_s = self._tightest_window(
            [item.at for item in record], min(len(record), 2))
        signal.detail = (f"{len(record)} followable wallets, "
                         f"{all_count} independent deciders, "
                         f"{len(fired)} of {len(self.rules)} rules fired")
        return signal

    def report(self) -> Dict[str, Any]:
        """How often each rule has fired. Not how well any of them worked.

        Firing is not evidence. These counts exist so a rule that never fires
        can be told apart from one the gauntlet killed, which are different
        problems with different fixes.
        """
        return {"schema": CONSENSUS_SCHEMA_VERSION,
                "status": "OK" if self.observed else "DATA_BLOCKED",
                "rules": [rule.to_dict() for rule in self.rules],
                "tokens_tracked": len(self._buys),
                "followable_buys_observed": self.observed,
                "fired_counts": dict(sorted(self._fired.items())),
                "never_fired": sorted(rule.mechanism for rule in self.rules
                                      if rule.mechanism not in self._fired),
                "detail": ("" if self.observed else
                           "no followable wallet has been seen buying yet")}
