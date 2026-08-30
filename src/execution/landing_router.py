"""Racing independent landing mechanisms, and learning which one actually wins.

The desk already races Jito's seven regions and takes first receipt. That is
one mechanism in seven places. When Jito's auction is congested, or its relay
is having a minute, seven regions of the same auction fail together -- which
is the failure mode the whole idea of racing was supposed to cover.

So routes here are MECHANISMS, not locations: a Jito bundle, a Jito single
transaction, a staked or SWQoS-prioritised RPC, a plain RPC, a
Sender-class multi-path forwarder. They fail for different reasons, which is
the only property that makes redundancy real.

**Identical bytes, one signature, at most one landing.** This is the safety
property the whole module rests on and it is worth being explicit about.
Solana identifies a transaction by its signature; the same signed bytes
submitted through five routes is one transaction that five parties are trying
to deliver, and the runtime executes it at most once. That is what makes
racing free rather than reckless.

Racing two DIFFERENTLY signed variants of the same intent -- say, one with a
Jito tip instruction and one without -- is not racing. It is two transactions,
both of which can land, which is a double-size position taken by accident at
the worst possible moment. This router refuses it: it takes one signed payload
and fans that exact string out. A route that needs different contents is a
different decision and belongs upstream of here.

**First receipt is not landing, and the difference is the whole measurement.**
A route that acknowledges instantly and never lands is worse than one that
acknowledges slowly and always does, and an accept count would rank them the
wrong way round. Every race records which route acknowledged first AND whether
the transaction landed at all, so the learned quantity is
`P(land | route, congestion)` rather than `P(route answers quickly)`.

**A route with no evidence is not a bad route.** Until a route has attempted
enough to measure, its rate reports DATA_BLOCKED and ordering falls back to
declaration order. Ranking an unmeasured route as worst would starve it of the
attempts it needs to be measured at all, which is how a routing table freezes
on whatever happened to work first.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LANDING_ROUTER_SCHEMA_VERSION = "v1"

#: Attempts before a route's landing rate is treated as measured. Below this
#: the rate is reported as DATA_BLOCKED: three landings out of four is not a
#: 75% route, it is four attempts.
MIN_ATTEMPTS_FOR_RATE = 30

#: How long the race waits for a first acknowledgement before giving up on
#: every route. Beyond a slot or two the answer stops being useful.
DEFAULT_RACE_TIMEOUT_S = 3.0


@dataclass
class Route:
    """One independent way of getting bytes to a leader."""

    name: str
    kind: str
    #: Takes the signed base64 transaction, returns an identifier (signature
    #: or bundle id) or None. Must not raise for an ordinary rejection; a
    #: raise is treated as a route failure and counted as one.
    submit: Callable[[str], Awaitable[Optional[str]]]
    #: What using this route costs beyond the transaction's own fees, in
    #: lamports. A tip is not a cost of the ROUTE -- it is inside the
    #: transaction and is paid whoever delivers it -- so this is normally
    #: zero and exists for routes that charge separately.
    surcharge_lamports: int = 0
    enabled: bool = True
    detail: str = ""
    #: Routes that cannot land a bundle-only payload, and vice versa. Checked
    #: by the caller, declared here so a mismatch is visible in the report.
    requires_bundle: bool = False


@dataclass
class RouteStats:
    """What this route has actually done. Never what it promises."""

    attempts: int = 0
    accepted: int = 0
    first_receipts: int = 0
    landed: int = 0
    failures: int = 0
    total_ack_ms: float = 0.0
    total_net_usd: float = 0.0
    resolved: int = 0
    last_error: str = ""

    @property
    def measured(self) -> bool:
        return self.resolved >= MIN_ATTEMPTS_FOR_RATE

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "attempts": self.attempts, "accepted": self.accepted,
            "first_receipts": self.first_receipts, "landed": self.landed,
            "failures": self.failures, "resolved": self.resolved,
            "mean_ack_ms": (round(self.total_ack_ms / self.accepted, 1)
                            if self.accepted else None),
            "last_error": self.last_error,
        }
        if not self.measured:
            # Three landings out of four is four attempts, not a 75% route.
            row.update({"land_rate": None, "mean_net_usd": None,
                        "data_status": "DATA_BLOCKED",
                        "detail": f"{self.resolved} of {MIN_ATTEMPTS_FOR_RATE} "
                                  "attempts needed before a rate means anything"})
            return row
        row.update({
            "land_rate": round(self.landed / self.resolved, 4),
            "mean_net_usd": round(self.total_net_usd / self.resolved, 4),
            "data_status": "OK", "detail": "",
        })
        return row


@dataclass
class RaceOutcome:
    """What happened when one signed payload was fanned out."""

    identifier: str = ""
    winner: str = ""
    winner_kind: str = ""
    ack_ms: Optional[float] = None
    attempted: Tuple[str, ...] = ()
    accepted: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    #: Set once the transaction is reconciled. Until then the race succeeded
    #: and the trade has not.
    landed: Optional[bool] = None

    @property
    def submitted(self) -> bool:
        return bool(self.accepted)

    def to_dict(self) -> Dict[str, Any]:
        return {"identifier": self.identifier, "winner": self.winner,
                "winner_kind": self.winner_kind, "ack_ms": self.ack_ms,
                "attempted": list(self.attempted),
                "accepted": dict(self.accepted), "errors": dict(self.errors),
                "landed": self.landed}


class LandingRouter:
    """Fans one signed payload across independent routes and scores them."""

    def __init__(self, *, race_timeout_s: float = DEFAULT_RACE_TIMEOUT_S,
                 min_attempts: int = MIN_ATTEMPTS_FOR_RATE):
        self.race_timeout_s = float(race_timeout_s)
        self.min_attempts = int(min_attempts)
        self._routes: Dict[str, Route] = {}
        self._stats: Dict[str, RouteStats] = {}
        self.races = 0
        self.races_with_no_acceptance = 0
        self._pending: Dict[str, RaceOutcome] = {}

    # --- registration ----------------------------------------------------

    def register(self, route: Route) -> None:
        self._routes[route.name] = route
        self._stats.setdefault(route.name, RouteStats())

    def enabled_routes(self, *, bundle_capable: bool = True) -> List[Route]:
        return [route for route in self._routes.values()
                if route.enabled and (bundle_capable or not route.requires_bundle)]

    # --- the race --------------------------------------------------------

    async def race(self, signed_tx: str, *, bundle_capable: bool = True,
                   timeout_s: Optional[float] = None) -> RaceOutcome:
        """Submit the SAME bytes through every enabled route; first ack wins.

        Every route is launched and the first acknowledgement returns. The
        remaining submissions are NOT cancelled: they are the redundancy, and
        cancelling them the moment one relay says yes throws away the reason
        for racing in the first place. They finish in the background and their
        outcomes are folded in when they do.
        """
        routes = self.enabled_routes(bundle_capable=bundle_capable)
        outcome = RaceOutcome(attempted=tuple(route.name for route in routes))
        if not routes:
            outcome.errors["router"] = "no enabled landing route"
            return outcome
        self.races += 1
        started = time.perf_counter()
        deadline = self.race_timeout_s if timeout_s is None else float(timeout_s)

        async def attempt(route: Route) -> Tuple[Route, Optional[str], float, str]:
            self._stats[route.name].attempts += 1
            try:
                identifier = await route.submit(signed_tx)
            except Exception as exc:
                return route, None, (time.perf_counter() - started) * 1000.0, \
                    f"{type(exc).__name__}: {exc}"
            return route, identifier, (time.perf_counter() - started) * 1000.0, ""

        tasks = [asyncio.ensure_future(attempt(route)) for route in routes]
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=max(0.0, deadline - (time.perf_counter() - started)),
                    return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    break
                for task in done:
                    route, identifier, elapsed_ms, error = task.result()
                    stats = self._stats[route.name]
                    if error or not identifier:
                        stats.failures += 1
                        stats.last_error = error or "route returned no identifier"
                        outcome.errors[route.name] = stats.last_error
                        continue
                    stats.accepted += 1
                    stats.total_ack_ms += elapsed_ms
                    outcome.accepted[route.name] = identifier
                    if not outcome.winner:
                        stats.first_receipts += 1
                        outcome.winner = route.name
                        outcome.winner_kind = route.kind
                        outcome.identifier = identifier
                        outcome.ack_ms = round(elapsed_ms, 2)
                if outcome.winner:
                    # The winner is known; the rest keep going as redundancy.
                    # Their results are folded in by the callback below rather
                    # than being waited for, because the caller needs to start
                    # confirming now.
                    for task in pending:
                        task.add_done_callback(self._fold_late)
                    pending = set()
        finally:
            pass
        if not outcome.winner:
            self.races_with_no_acceptance += 1
            for task in tasks:
                task.cancel()
        elif outcome.identifier:
            self._pending[outcome.identifier] = outcome
        return outcome

    def _fold_late(self, task: "asyncio.Task") -> None:
        """Record a route that acknowledged after the race was decided.

        Not a first receipt, but still evidence: a route that always accepts
        second is a route that is working, and dropping its result would make
        it look like it never answers.
        """
        if task.cancelled():
            return
        try:
            route, identifier, elapsed_ms, error = task.result()
        except Exception:  # pragma: no cover - defensive
            return
        stats = self._stats.get(route.name)
        if stats is None:
            return
        if error or not identifier:
            stats.failures += 1
            stats.last_error = error or "route returned no identifier"
            return
        stats.accepted += 1
        stats.total_ack_ms += elapsed_ms

    # --- outcomes --------------------------------------------------------

    def record_landing(self, identifier: str, *, landed: bool,
                       net_usd: float = 0.0) -> Optional[RaceOutcome]:
        """Attribute a resolved transaction back to the routes that carried it.

        Credited to EVERY route that accepted it, not only the winner. All of
        them delivered the same bytes; which relay the leader happened to take
        it from is not observable from here, and inventing an attribution
        would produce a routing table built on a guess.
        """
        outcome = self._pending.pop(identifier, None)
        if outcome is None:
            return None
        outcome.landed = bool(landed)
        for name in outcome.accepted:
            stats = self._stats.get(name)
            if stats is None:
                continue
            stats.resolved += 1
            if landed:
                stats.landed += 1
                stats.total_net_usd += float(net_usd)
        return outcome

    def forget(self, identifier: str) -> None:
        """Drop an unresolved race. Bounded memory, and never a false landing."""
        self._pending.pop(identifier, None)

    # --- ranking ---------------------------------------------------------

    def ranked(self) -> List[str]:
        """Routes best-first by MEASURED landing rate, unmeasured ones kept.

        An unmeasured route sorts on declaration order rather than last: a
        route ranked worst never gets attempted, so it is never measured, so
        it stays ranked worst. That is how a routing table freezes on whatever
        happened to work on the first day.
        """
        declared = list(self._routes)
        def key(name: str) -> Tuple[int, float, int]:
            stats = self._stats[name]
            if not stats.measured:
                return (1, 0.0, declared.index(name))
            return (0, -(stats.landed / max(1, stats.resolved)), declared.index(name))
        return sorted(declared, key=key)

    # --- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Which mechanisms are landing, and whether any of it is measured yet.

        `mechanisms` is the number that matters for redundancy: seven Jito
        regions is one mechanism, and a router with one mechanism has the
        redundancy of having none.
        """
        rows = {name: self._stats[name].to_dict() for name in self._routes}
        enabled = [route for route in self._routes.values() if route.enabled]
        kinds = sorted({route.kind for route in enabled})
        measured = [name for name, row in rows.items() if row["data_status"] == "OK"]
        if not enabled:
            status, detail = "DATA_BLOCKED", "no landing route is enabled"
        elif len(kinds) < 2:
            status, detail = "DEGRADED", (
                f"only one landing mechanism ({kinds[0] if kinds else 'none'}); "
                "regions of one mechanism fail together, which is the case "
                "racing was supposed to cover")
        elif not measured:
            status, detail = "OK", (
                "racing " + str(len(kinds)) + " mechanisms; no route has enough "
                "resolved attempts for its landing rate to mean anything yet")
        else:
            status, detail = "OK", ""
        return {
            "schema": LANDING_ROUTER_SCHEMA_VERSION,
            "status": status, "detail": detail,
            "routes": len(self._routes),
            "enabled": len(enabled),
            "mechanisms": kinds,
            "races": self.races,
            "races_with_no_acceptance": self.races_with_no_acceptance,
            "awaiting_resolution": len(self._pending),
            "ranked": self.ranked(),
            "measured_routes": sorted(measured),
            "by_route": rows,
        }
