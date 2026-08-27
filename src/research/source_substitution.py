"""When a source goes dark, take the next one -- and say which one you took.

The fact ladder in ``src/research/fallback.py`` answers a different question
from this module, and the two are easy to confuse. That one asks *how well do
we know this fact*: measured, corroborated, reconstructed, proxy, prior. This
one asks *who is still answering the phone*. A holder distribution read from a
Korean mirror of the same chain state is still MEASURED; it is simply measured
somewhere else. Substituting a venue does not degrade a rung, and a design
that conflated the two would shrink every position the moment a rate limit
moved us to a second endpoint.

So: a DOMAIN is a question the desk needs answered continuously -- the token
universe, new pools, a mint's price, what landing costs, whether supply is
still controlled. Each domain declares a LADDER of endpoints that answer it
interchangeably, ordered by preference, and tagged by region. The rotator
holds which rung is live.

Failure is expected and is not an incident. A public endpoint refuses a
datacentre address, changes a path, rate limits, or is simply down for an
hour. What must never happen is the desk reporting an unmeasured quantity as
a measured one, or sitting on a dead primary while four working equivalents
go unused. So:

**Rotation is automatic and recovery is automatic.** Consecutive failures
quarantine a rung with a doubling penalty and advance to the next. When the
quarantine lapses the rung becomes eligible again, and because the ladder is
ordered and ``current()`` always returns the best eligible rung, the desk
climbs back to its preferred source by itself. Nobody has to notice.

**Regional breadth is coverage, not decoration.** The Asian venues here are
not a politeness. Korean and Chinese-language flow leads a meaningful share of
memecoin attention, and a desk whose entire market context comes from two US
aggregators is a desk that is blind for the hours those two are the quiet
ones. Every rung carries a region, and ``coverage()`` reports how many regions
are actually answering rather than how many are declared.

**A rung nobody has probed is a claim, not a fact.** Every URL here is a
public, documented, keyless interface as far as we know, but "as far as we
know" is not a measurement. ``tools/verify_substitution.py`` probes the whole
set from the node that will use it and reports which rungs are live from that
address. A rung that never answers is reported by name, permanently, rather
than quietly padding a coverage number.

Nothing here reaches a private venue, an account-gated feed, or anything
behind an access control. Every endpoint is one a browser can open.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SUBSTITUTION_SCHEMA_VERSION = "v1"

#: Consecutive failures before a rung is stood down. One failure is a blip;
#: three in a row from the same endpoint is that endpoint.
FAILURES_BEFORE_ROTATE = 3

#: First quarantine, doubling per repeat, capped. Short enough that an
#: endpoint recovering from a five-minute outage is used again quickly;
#: long enough that a permanently dead one is not retried every minute.
QUARANTINE_BASE_S = 300.0
QUARANTINE_MAX_S = 7_200.0


@dataclass(frozen=True)
class Endpoint:
    """One way of answering one domain."""

    name: str
    url: str
    region: str = "global"
    requires_env: Tuple[str, ...] = ()
    detail: str = ""
    #: Free-form tag naming the payload shape, so a caller can pick a parser
    #: without the rotator having to understand any of them.
    shape: str = ""

    def missing_credentials(self) -> List[str]:
        """Which required variables are absent. Presence only, never a value."""
        return [name for name in self.requires_env if not os.getenv(name)]

    def format(self, **kwargs: Any) -> str:
        try:
            return self.url.format(**kwargs)
        except (KeyError, IndexError):
            return self.url


@dataclass
class RungState:
    """What this rung has actually done."""

    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    quarantined_until: float = 0.0
    quarantine_seconds: float = 0.0
    rotations: int = 0
    last_error: str = ""
    last_ok_at: float = 0.0

    def eligible(self, now: float) -> bool:
        return self.quarantined_until <= now


class SubstitutionRegistry:
    """Every domain's ladder, and which rung each is running on right now."""

    def __init__(self, *, failures_before_rotate: int = FAILURES_BEFORE_ROTATE,
                 quarantine_base_s: float = QUARANTINE_BASE_S,
                 quarantine_max_s: float = QUARANTINE_MAX_S):
        self.failures_before_rotate = max(1, int(failures_before_rotate))
        self.quarantine_base_s = float(quarantine_base_s)
        self.quarantine_max_s = float(quarantine_max_s)
        self._ladders: Dict[str, List[Endpoint]] = {}
        self._state: Dict[Tuple[str, str], RungState] = {}
        self.substitutions = 0
        self.recoveries = 0

    # --- declaration -----------------------------------------------------

    def declare(self, domain: str, endpoints: Sequence[Endpoint]) -> None:
        """Declare a domain's equivalents, best first. Order is load-bearing."""
        self._ladders[domain] = list(endpoints)
        for endpoint in endpoints:
            self._state.setdefault((domain, endpoint.name), RungState())

    def domains(self) -> List[str]:
        return sorted(self._ladders)

    def ladder(self, domain: str) -> List[Endpoint]:
        return list(self._ladders.get(domain, ()))

    # --- selection -------------------------------------------------------

    def current(self, domain: str, now: Optional[float] = None) -> Optional[Endpoint]:
        """The best rung that is not quarantined and not missing a credential.

        Always evaluated from the top, which is what makes recovery automatic:
        the moment a quarantine lapses the primary is preferred again without
        anything having to notice that it came back.
        """
        moment = time.time() if now is None else now
        for endpoint in self._ladders.get(domain, ()):
            if endpoint.missing_credentials():
                continue
            if self._state[(domain, endpoint.name)].eligible(moment):
                return endpoint
        return None

    def eligible(self, domain: str, now: Optional[float] = None) -> List[Endpoint]:
        moment = time.time() if now is None else now
        return [endpoint for endpoint in self._ladders.get(domain, ())
                if not endpoint.missing_credentials()
                and self._state[(domain, endpoint.name)].eligible(moment)]

    def dark_domains(self, now: Optional[float] = None) -> List[str]:
        """Domains with nothing left to ask. The list that deserves an alert."""
        moment = time.time() if now is None else now
        return [domain for domain in sorted(self._ladders)
                if not self.eligible(domain, moment)]

    # --- outcomes --------------------------------------------------------

    def note_success(self, domain: str, endpoint_name: str,
                     now: Optional[float] = None) -> None:
        """A pass that worked clears the penalty on that rung entirely.

        Not decayed, cleared. Keeping a partial penalty punishes an endpoint
        for an outage it has already recovered from, and the next single blip
        then rotates it away again.
        """
        moment = time.time() if now is None else now
        state = self._state.get((domain, endpoint_name))
        if state is None:
            return
        recovering = state.quarantine_seconds > 0 or state.consecutive_failures > 0
        state.successes += 1
        state.consecutive_failures = 0
        state.quarantined_until = 0.0
        state.quarantine_seconds = 0.0
        state.last_error = ""
        state.last_ok_at = moment
        if recovering:
            self.recoveries += 1

    def note_failure(self, domain: str, endpoint_name: str, reason: str = "",
                     now: Optional[float] = None) -> Optional[Endpoint]:
        """Record a failure; quarantine and rotate once it is clearly the rung.

        Returns the endpoint the domain has moved to, or None if the ladder is
        now exhausted -- which is the one case a human should hear about,
        because it means a question the desk asks continuously has no answer.
        """
        moment = time.time() if now is None else now
        state = self._state.get((domain, endpoint_name))
        if state is None:
            return self.current(domain, moment)
        state.failures += 1
        state.consecutive_failures += 1
        state.last_error = reason[:400]
        if state.consecutive_failures >= self.failures_before_rotate:
            state.quarantine_seconds = min(
                self.quarantine_max_s,
                (state.quarantine_seconds * 2) if state.quarantine_seconds
                else self.quarantine_base_s)
            state.quarantined_until = moment + state.quarantine_seconds
            state.consecutive_failures = 0
            state.rotations += 1
            self.substitutions += 1
            replacement = self.current(domain, moment)
            logger.info(
                "SUBSTITUTION %s: %s stood down for %.0fs (%s); now on %s",
                domain, endpoint_name, state.quarantine_seconds, reason or "no reason given",
                replacement.name if replacement else "NOTHING -- ladder exhausted")
            return replacement
        return self.current(domain, moment)

    def release(self, domain: str = "", now: Optional[float] = None) -> List[str]:
        """Lift quarantines early. The lever a fixer pulls when a domain is dark.

        A dark domain usually means the whole ladder was quarantined by one
        shared cause -- our address rate limited everywhere, the node's DNS
        wobbling, an outbound proxy blip -- rather than by every endpoint
        independently dying. Waiting out four separate penalties for a cause
        that has already passed is unmeasured data nobody needed to lose.
        """
        moment = time.time() if now is None else now
        released: List[str] = []
        for (this_domain, name), state in self._state.items():
            if domain and this_domain != domain:
                continue
            if state.quarantined_until > moment:
                state.quarantined_until = 0.0
                state.consecutive_failures = 0
                released.append(f"{this_domain}:{name}")
        return sorted(released)

    # --- reporting -------------------------------------------------------

    def coverage(self, now: Optional[float] = None) -> Dict[str, Any]:
        """How many regions are ANSWERING, as opposed to being declared.

        A declared region that has never returned a record is a coverage hole
        wearing the appearance of breadth, and it is the flattering direction
        of the error, so it is reported separately by name.
        """
        moment = time.time() if now is None else now
        declared: Dict[str, int] = {}
        live: Dict[str, int] = {}
        proven: Dict[str, int] = {}
        for domain, endpoints in self._ladders.items():
            for endpoint in endpoints:
                declared[endpoint.region] = declared.get(endpoint.region, 0) + 1
                state = self._state[(domain, endpoint.name)]
                if not endpoint.missing_credentials() and state.eligible(moment):
                    live[endpoint.region] = live.get(endpoint.region, 0) + 1
                if state.successes > 0:
                    proven[endpoint.region] = proven.get(endpoint.region, 0) + 1
        unproven = sorted(set(declared) - set(proven))
        return {
            "regions_declared": len(declared),
            "regions_proven": len(proven),
            "declared": dict(sorted(declared.items())),
            "live": dict(sorted(live.items())),
            "proven": dict(sorted(proven.items())),
            "unproven_regions": unproven,
        }

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Which domain is running on which rung, and what is dark.

        The line that matters is `dark`: a domain with no eligible endpoint is
        a question the desk asks continuously and currently cannot answer, and
        it is the only state here that is worth waking somebody for.
        """
        moment = time.time() if now is None else now
        rows = []
        degraded = []
        for domain in sorted(self._ladders):
            endpoints = self._ladders[domain]
            active = self.current(domain, moment)
            primary = endpoints[0].name if endpoints else ""
            depth = len(self.eligible(domain, moment))
            rows.append({
                "domain": domain,
                "active": active.name if active else None,
                "active_region": active.region if active else None,
                "primary": primary,
                "on_primary": bool(active and active.name == primary),
                "declared_rungs": len(endpoints),
                "eligible_rungs": depth,
                "rungs": [self._rung_row(domain, endpoint, moment)
                          for endpoint in endpoints],
            })
            if active is not None and active.name != primary:
                degraded.append(f"{domain} -> {active.name}")
        dark = self.dark_domains(moment)
        if dark:
            status = "DATA_BLOCKED"
            detail = ("no endpoint left for: " + ", ".join(dark)
                      + "; these questions currently have no answer")
        elif degraded:
            status = "SUBSTITUTED"
            detail = "running on a substitute for: " + ", ".join(degraded)
        else:
            status = "OK"
            detail = ""
        return {
            "schema": SUBSTITUTION_SCHEMA_VERSION,
            "status": status,
            "detail": detail,
            "domains": len(self._ladders),
            "rungs": len(self._state),
            "substitutions": self.substitutions,
            "recoveries": self.recoveries,
            "dark": dark,
            "substituted": degraded,
            "coverage": self.coverage(moment),
            "ladders": rows,
        }

    def _rung_row(self, domain: str, endpoint: Endpoint, now: float) -> Dict[str, Any]:
        state = self._state[(domain, endpoint.name)]
        missing = endpoint.missing_credentials()
        return {
            "name": endpoint.name,
            "region": endpoint.region,
            "successes": state.successes,
            "failures": state.failures,
            "rotations": state.rotations,
            "quarantined_for": (round(state.quarantined_until - now, 1)
                                if state.quarantined_until > now else None),
            "missing_credentials": missing or None,
            "last_error": state.last_error,
            "never_answered": state.successes == 0,
        }
