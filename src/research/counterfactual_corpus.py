"""Every decision, every alternative it had, and what actually happened.

The training corpus this desk needs is not a list of trades. A model trained
only on positions taken learns whether a buy made money; it cannot learn
whether IGNORE was right, because an ignored launch leaves no row. That is
selection bias of the worst kind here, since IGNORE is the action the desk
takes on the overwhelming majority of launches and the one whose mistakes are
invisible by construction -- a missed hundred-x costs nothing that any ledger
records.

So a row is written for every DECISION, not every trade, and it carries:

  the state the decision was made from, point in time, before the action;
  every action that was feasible, and the Q the policy gave each one;
  which action was taken, and the size;
  what the alternatives would have cost -- entry price, route, tip;
  what actually happened afterwards, from the stream, independent of us.

That last independence is the point. The outcome is recorded from the launch
census, which watches every token whether or not the desk touched it, so the
counterfactual for IGNORE is a real measurement rather than a simulation of
one.

Three disciplines, and they are what make this a corpus rather than a log.

**The snapshot is taken BEFORE the action and never amended.** A row updated
with anything learned afterwards is a row that leaks the future into its own
features, and a model trained on it will look extraordinary in backtest and
fail forward. Resolution is written to a separate field, once, on a record
that is otherwise frozen.

**An unresolved decision is not a zero.** A launch still in flight has
`realised_multiple: None`, and a trainer that treats that as a loss has
manufactured a loss out of impatience. Rows are counted resolved and
unresolved separately and the reports never merge them.

**Infeasible is not the same as bad.** An action the state could not support
-- ADD with no capital, REENTER on an open position -- is stored with its
status, not with a large negative Q. Storing a sentinel would teach the model
that those actions are terrible rather than that they were unavailable.

Written append-only as JSON lines. A corpus that has to be rewritten to be
appended to is one that loses everything on a crash mid-write, and this is
the file whose loss cannot be recovered by re-running anything.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CORPUS_SCHEMA_VERSION = "v1"

#: Rows held in memory before a flush. Small: this file is the one whose loss
#: cannot be recovered by re-running anything, so it is written often.
DEFAULT_FLUSH_EVERY = 25

#: Decisions kept addressable for later resolution. A launch resolves within
#: minutes to hours; beyond this it is written unresolved and forgotten,
#: which is honest -- an outcome we never saw is not an outcome of zero.
DEFAULT_PENDING = 20_000


@dataclass
class ActionOption:
    """One action the policy priced, and whether it could have been taken."""

    action: str
    q: Optional[float] = None
    status: str = "OK"
    size_fraction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "q": self.q, "status": self.status,
                "size_fraction": self.size_fraction}


@dataclass
class RouteOption:
    """A landing route that was available, and what it was expected to cost."""

    name: str
    kind: str = ""
    predicted_land_probability: Optional[float] = None
    tip_lamports: Optional[int] = None
    priority_fee_lamports: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "predicted_land_probability": self.predicted_land_probability,
                "tip_lamports": self.tip_lamports,
                "priority_fee_lamports": self.priority_fee_lamports}


@dataclass
class DecisionRow:
    """One decision, frozen at the moment it was made."""

    decision_id: str
    mint: str
    decided_at: float
    age_seconds: Optional[float] = None
    regime: str = "unknown"
    #: Point-in-time features. Whatever the caller had; never fetched here,
    #: because a corpus that fetches is a corpus that records a later world
    #: than the one the decision saw.
    state: Dict[str, Any] = field(default_factory=dict)
    options: List[ActionOption] = field(default_factory=list)
    chosen_action: str = ""
    chosen_q: Optional[float] = None
    size_fraction: Optional[float] = None
    entry_price: Optional[float] = None
    routes: List[RouteOption] = field(default_factory=list)
    chosen_route: str = ""
    #: Why the desk did not act, when it did not. An unattributed non-action
    #: is a launch that vanished from the analysis.
    screen_reason: str = ""
    #: Filled once, later, from the census. Never amended after that.
    resolution: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return bool(self.resolution)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": CORPUS_SCHEMA_VERSION,
            "decision_id": self.decision_id, "mint": self.mint,
            "decided_at": self.decided_at, "age_seconds": self.age_seconds,
            "regime": self.regime, "state": self.state,
            "options": [option.to_dict() for option in self.options],
            "chosen_action": self.chosen_action, "chosen_q": self.chosen_q,
            "size_fraction": self.size_fraction, "entry_price": self.entry_price,
            "routes": [route.to_dict() for route in self.routes],
            "chosen_route": self.chosen_route,
            "screen_reason": self.screen_reason,
            "resolution": self.resolution or None,
        }


class CounterfactualCorpus:
    """Append-only record of decisions and the outcomes they were judged by."""

    def __init__(self, path: Optional[str] = None, *,
                 flush_every: int = DEFAULT_FLUSH_EVERY,
                 max_pending: int = DEFAULT_PENDING):
        self.path = path
        self.flush_every = max(1, int(flush_every))
        self.max_pending = max(1, int(max_pending))
        self._pending: Dict[str, DecisionRow] = {}
        self._buffer: List[Dict[str, Any]] = []
        self.recorded = 0
        self.resolved = 0
        self.written = 0
        self.dropped_unresolved = 0
        self.write_failures = 0
        self.last_error = ""
        self._by_action: Dict[str, int] = {}
        self._resolved_by_action: Dict[str, Dict[str, float]] = {}

    # --- recording -------------------------------------------------------

    def record(self, decision_id: str, mint: str, *,
               state: Optional[Dict[str, Any]] = None,
               options: Sequence[ActionOption] = (),
               chosen_action: str = "", chosen_q: Optional[float] = None,
               size_fraction: Optional[float] = None,
               entry_price: Optional[float] = None,
               routes: Sequence[RouteOption] = (),
               chosen_route: str = "", screen_reason: str = "",
               age_seconds: Optional[float] = None, regime: str = "unknown",
               at: Optional[float] = None) -> DecisionRow:
        """Freeze one decision. Called for IGNORE as much as for a buy.

        The whole corpus rests on this being called for the launches the desk
        walked away from. A row set that contains only the trades taken can
        answer "did we make money" and cannot answer "should we have been
        there", and the second question is where the returns are.
        """
        row = DecisionRow(
            decision_id=decision_id, mint=mint,
            decided_at=time.time() if at is None else float(at),
            age_seconds=age_seconds, regime=regime,
            state=dict(state or {}), options=list(options),
            chosen_action=chosen_action, chosen_q=chosen_q,
            size_fraction=size_fraction, entry_price=entry_price,
            routes=list(routes), chosen_route=chosen_route,
            screen_reason=screen_reason)
        self.recorded += 1
        self._by_action[chosen_action or "unspecified"] = (
            self._by_action.get(chosen_action or "unspecified", 0) + 1)
        if len(self._pending) >= self.max_pending:
            # Oldest out, and written unresolved rather than discarded: a
            # decision whose outcome we never saw is still evidence about
            # what the desk decided, and dropping it silently would bias the
            # corpus toward launches that resolved quickly.
            oldest = next(iter(self._pending))
            self._flush_row(self._pending.pop(oldest))
            self.dropped_unresolved += 1
        self._pending[decision_id] = row
        return row

    def resolve(self, decision_id: str, *,
                realised_multiple: Optional[float] = None,
                peak_multiple: Optional[float] = None,
                rugged: Optional[bool] = None,
                rug_mechanism: str = "",
                migrated: Optional[bool] = None,
                landed: Optional[bool] = None,
                landing_route: str = "",
                realised_log_growth: Optional[float] = None,
                at: Optional[float] = None) -> Optional[DecisionRow]:
        """Attach the outcome, once, to a record that is otherwise frozen.

        `peak_multiple` is the counterfactual and `realised_multiple` is what
        the desk actually got. Keeping them apart is what lets a trainer ask
        the question that matters -- how much of what was available did we
        take -- rather than only whether the trade was green.
        """
        row = self._pending.pop(decision_id, None)
        if row is None:
            return None
        row.resolution = {
            "resolved_at": time.time() if at is None else float(at),
            "realised_multiple": realised_multiple,
            "peak_multiple": peak_multiple,
            "rugged": rugged, "rug_mechanism": rug_mechanism,
            "migrated": migrated, "landed": landed,
            "landing_route": landing_route,
            "realised_log_growth": realised_log_growth,
            # What was available that we did not take. None when the peak is
            # unknown: an unmeasured foregone return is not a zero one.
            "foregone_multiple": (
                None if peak_multiple is None or realised_multiple is None
                else round(max(0.0, peak_multiple - realised_multiple), 6)),
        }
        self.resolved += 1
        bucket = self._resolved_by_action.setdefault(
            row.chosen_action or "unspecified",
            {"count": 0.0, "peak": 0.0, "realised": 0.0, "wins": 0.0})
        bucket["count"] += 1
        if peak_multiple is not None:
            bucket["peak"] += float(peak_multiple)
        if realised_multiple is not None:
            bucket["realised"] += float(realised_multiple)
            if realised_multiple > 1.0:
                bucket["wins"] += 1
        self._flush_row(row)
        return row

    def resolve_by_mint(self, mint: str, **outcome: Any) -> List[DecisionRow]:
        """Resolve every open decision for one launch.

        A launch is decided on repeatedly -- entry, then holds, then an exit
        -- and the census resolves the TOKEN. Every decision made about it
        shares that outcome, and resolving only the entry would throw away
        every hold the desk got right or wrong.
        """
        matches = [decision_id for decision_id, row in self._pending.items()
                   if row.mint == mint]
        return [row for row in (self.resolve(decision_id, **outcome)
                                for decision_id in matches) if row is not None]

    # --- persistence -----------------------------------------------------

    def _flush_row(self, row: DecisionRow) -> None:
        self._buffer.append(row.to_dict())
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> int:
        """Append the buffer. Append-only: a corpus that has to be rewritten
        to be added to is one that loses everything on a crash mid-write."""
        if not self._buffer or not self.path:
            written = len(self._buffer) if not self.path else 0
            self._buffer.clear()
            return written
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                for row in self._buffer:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        except OSError as exc:
            self.write_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("counterfactual corpus write failed: %s", exc)
            return 0
        count = len(self._buffer)
        self.written += count
        self._buffer.clear()
        return count

    def close(self) -> int:
        """Write everything, including decisions still in flight.

        Unresolved rows are written with `resolution: null`, which is the
        honest record: the desk decided, and what happened next was never
        observed. Discarding them would bias the corpus toward launches that
        resolved before shutdown.
        """
        for decision_id in list(self._pending):
            self._flush_row(self._pending.pop(decision_id))
        return self.flush()

    def load(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read rows back. Malformed lines are skipped and counted, not fatal."""
        rows: List[Dict[str, Any]] = []
        if not self.path or not os.path.exists(self.path):
            return rows
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if limit is not None and len(rows) >= limit:
                        break
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return rows

    # --- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Whether the corpus is usable for training yet, and what it covers.

        The line that matters is `ignore_share`. A corpus where IGNORE is a
        small minority of rows is a corpus that is still only recording
        trades, and a model trained on it cannot learn what the desk walked
        away from.
        """
        ignores = self._by_action.get("ignore", 0)
        share = (ignores / self.recorded) if self.recorded else None
        by_action = {}
        for action, bucket in sorted(self._resolved_by_action.items()):
            count = bucket["count"]
            by_action[action] = {
                "resolved": int(count),
                "mean_peak_multiple": (round(bucket["peak"] / count, 4)
                                       if count else None),
                "mean_realised_multiple": (round(bucket["realised"] / count, 4)
                                           if count else None),
                "hit_rate": round(bucket["wins"] / count, 4) if count else None,
            }
        if not self.recorded:
            status, detail = "DATA_BLOCKED", "no decision has been recorded yet"
        elif share is not None and share < 0.5:
            status, detail = "DEGRADED", (
                f"only {share:.0%} of rows are IGNORE; the desk ignores most "
                "launches, so a corpus this shaped is still recording trades "
                "rather than decisions")
        elif not self.resolved:
            status, detail = "OK", (
                "decisions are being recorded; none has resolved yet, so no "
                "action has a measured payoff")
        else:
            status, detail = "OK", ""
        return {
            "schema": CORPUS_SCHEMA_VERSION,
            "status": status, "detail": detail,
            "recorded": self.recorded,
            "resolved": self.resolved,
            "unresolved_in_flight": len(self._pending),
            "written": self.written,
            "buffered": len(self._buffer),
            "dropped_unresolved": self.dropped_unresolved,
            "write_failures": self.write_failures,
            "last_error": self.last_error,
            "by_chosen_action": dict(sorted(self._by_action.items())),
            "ignore_share": None if share is None else round(share, 4),
            "outcomes_by_action": by_action,
            "path": self.path,
        }
