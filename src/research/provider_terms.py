"""Watching the terms, not the uptime.

The mesh is already good at noticing a provider that stopped answering: 403s,
429s, silent endpoints, quarantine, substitution, recovery. All of that is
availability surveillance, and it detects a change only once it has already
broken something.

The changes that actually end free tiers do not look like outages. A provider
moves Yellowstone behind a paid plan; a monthly allowance drops from ten
million requests to one; "no credit card required" quietly stops being on the
pricing page; a method the desk depends on is added to a paid-only list; the
terms add a clause about automated trading. Every one of those is announced on
a web page weeks before it is enforced, and none of them produce a single
failed request until the day the desk is already down.

So this module reads the pages.

**It does not diff hashes and call that a finding.** A pricing page changes its
hash on every marketing deploy, and a watcher that cries wolf on every deploy
is a watcher that gets ignored, which is worse than not having one. Instead a
snapshot extracts a small set of POLICY SIGNALS -- quota-shaped numbers, plan
words, card requirements, named methods, rate limits -- and the diff is
computed over those. A page whose hash changed but whose signals did not is
reported as exactly that: changed, no policy signal, no action.

**It classifies direction.** A quota that went up is not a finding worth
waking anyone for. A quota that went down, a free plan that gained a price, a
card requirement that appeared, a method that vanished from the free list --
those are `Severity.BREAKING`, and they carry the before and after values so
the reader does not have to go and look.

**It knows what it has never seen.** A provider the desk has heard of but never
verified sits in the registry as a CANDIDATE with a stated blocking reason,
rather than being added to a live ladder on the strength of a marketing page.
NodeFlare is the worked example: it is a real provider with a real free tier,
it serves twenty-odd EVM chains, and it does not serve Solana -- so adding it
to the Solana ladder would have installed a rung that can never answer, and
the failure would have looked like an outage rather than a category error.

Nothing here fetches on its own. A fetcher is injected, exactly as elsewhere in
this package, so the module is testable without the network and so the desk's
one HTTP policy stays in one place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROVIDER_TERMS_SCHEMA_VERSION = "v1"


class Severity(Enum):
    #: Nothing extracted changed.
    NONE = "NONE"
    #: The page changed but no policy signal did. Recorded, not escalated.
    COSMETIC = "COSMETIC"
    #: A signal moved in the desk's favour, or a new one appeared.
    INFORMATIONAL = "INFORMATIONAL"
    #: A signal moved against the desk. This is the one worth waking for.
    BREAKING = "BREAKING"


class Support(Enum):
    #: Measured working from this box, with a date.
    VERIFIED = "VERIFIED"
    #: Claimed by the provider, never measured here.
    CANDIDATE = "CANDIDATE"
    #: Measured and does not work, or structurally cannot.
    UNSUPPORTED = "UNSUPPORTED"


#: Quota-shaped numbers. Deliberately narrow: matching every number on a
#: pricing page produces a diff on the copyright year.
_QUOTA_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("requests_per_month", re.compile(
        r"([\d,.]+)\s*(k|m|million|thousand)?\s*(?:requests?|reqs?|calls?)"
        r"[^.\n]{0,30}?(?:per|/|a)\s*month", re.I)),
    ("compute_units_per_month", re.compile(
        r"([\d,.]+)\s*(k|m|million|thousand)?\s*compute\s*units?"
        r"[^.\n]{0,30}?(?:per|/|a)\s*month", re.I)),
    ("requests_per_second", re.compile(
        r"([\d,.]+)\s*(k|m|million|thousand)?\s*(?:requests?|reqs?|calls?)"
        r"[^.\n]{0,30}?(?:per|/|a)\s*(?:second|sec\b)", re.I)),
    ("credits_per_month", re.compile(
        r"([\d,.]+)\s*(k|m|million|thousand)?\s*credits?"
        r"[^.\n]{0,30}?(?:per|/|a)\s*month", re.I)),
)

_MULTIPLIERS = {"": 1.0, "k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}

#: Phrases whose PRESENCE is a policy fact. Each maps to whether its presence
#: is good for the desk, which is what lets the diff assign a direction rather
#: than just noting that a string appeared.
_FLAG_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]", bool], ...] = (
    ("no_card_required",
     re.compile(r"no\s+credit\s+card", re.I), True),
    # The lookbehind matters more than it looks: without it, "no credit card
    # required" sets card_required as well as no_card_required, and the
    # friendliest sentence on the page reads as the hostile one.
    ("card_required",
     re.compile(r"(?<!no )(?:credit\s+card|payment\s+method)\s+(?:is\s+)?"
                r"required|requires?\s+a\s+credit\s+card", re.I), False),
    ("free_tier",
     re.compile(r"\bfree\s+(tier|plan|forever|to\s+use)\b", re.I), True),
    ("no_rate_limit",
     re.compile(r"no\s+rate\s+limits?", re.I), True),
    ("trial_only",
     re.compile(r"\b(free\s+trial|trial\s+period|\d+[- ]day\s+trial)\b",
                re.I), False),
    ("paid_only",
     re.compile(r"\b(paid\s+plans?\s+only|available\s+on\s+paid|"
                r"upgrade\s+to\s+access)\b", re.I), False),
    ("deprecated",
     re.compile(r"\b(deprecat\w+|sunset|end[- ]of[- ]life|discontinu\w+)\b",
                re.I), False),
)


def _to_number(raw: str, unit: str) -> Optional[float]:
    cleaned = raw.replace(",", "").strip().rstrip(".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value * _MULTIPLIERS.get((unit or "").lower(), 1.0)


def extract_signals(text: str, *, methods: Sequence[str] = ()
                    ) -> Dict[str, Any]:
    """Reduce a page to the handful of facts a desk's plan depends on.

    Everything else -- layout, testimonials, the blog carousel -- is dropped
    on purpose. The output of this function IS the thing that gets diffed, so
    anything it keeps is something a marketing deploy can wake someone over.
    """
    signals: Dict[str, Any] = {"quotas": {}, "flags": {}, "methods": {}}
    body = text or ""
    for name, pattern in _QUOTA_PATTERNS:
        best: Optional[float] = None
        for match in pattern.finditer(body):
            value = _to_number(match.group(1), match.group(2) or "")
            if value is None:
                continue
            # The largest stated allowance is the free tier's headline, which
            # is the number that gets cut. Taking the first match instead
            # picks up whatever the page happens to mention earliest.
            best = value if best is None else max(best, value)
        if best is not None:
            signals["quotas"][name] = best
    for name, pattern, _ in _FLAG_PATTERNS:
        signals["flags"][name] = bool(pattern.search(body))
    for method in methods:
        signals["methods"][method] = bool(
            re.search(re.escape(method), body, re.I))
    return signals


@dataclass
class TermsSnapshot:
    """One dated read of one page, plus what was extracted from it."""

    provider: str
    url: str
    fetched_at: float
    content_hash: str
    signals: Dict[str, Any] = field(default_factory=dict)
    status: int = 200
    error: str = ""
    bytes_read: int = 0

    @property
    def usable(self) -> bool:
        return not self.error and self.status == 200 and bool(self.content_hash)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TermsSnapshot":
        return cls(
            provider=str(data.get("provider", "")),
            url=str(data.get("url", "")),
            fetched_at=float(data.get("fetched_at", 0.0)),
            content_hash=str(data.get("content_hash", "")),
            signals=dict(data.get("signals") or {}),
            status=int(data.get("status", 0)),
            error=str(data.get("error", "")),
            bytes_read=int(data.get("bytes_read", 0)))


@dataclass
class TermsChange:
    """One difference, with its direction already decided."""

    provider: str
    field: str
    before: Any
    after: Any
    severity: Severity
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "field": self.field,
                "before": self.before, "after": self.after,
                "severity": self.severity.value, "detail": self.detail}


@dataclass
class Provider:
    """A source of chain access, and what the desk has actually established.

    `chains` is what the provider serves, `support` is what the desk measured.
    Keeping them apart is the whole point: a provider can be entirely real,
    entirely free, and entirely incapable of answering the chain we trade.
    """

    name: str
    terms_url: str = ""
    rpc_url: str = ""
    chains: Tuple[str, ...] = ()
    support: Support = Support.CANDIDATE
    verified_at: float = 0.0
    blocking_reason: str = ""
    #: RPC methods whose free-tier availability is worth watching by name.
    watch_methods: Tuple[str, ...] = ()
    note: str = ""

    def serves(self, chain: str) -> bool:
        return chain.lower() in {item.lower() for item in self.chains}

    def usable_for(self, chain: str) -> Tuple[bool, str]:
        """Fail closed. An unmeasured provider is not a rung."""
        if not self.serves(chain):
            served = ", ".join(self.chains) or "nothing recorded"
            detail = f" {self.blocking_reason}" if self.blocking_reason else ""
            return False, (f"{self.name} does not serve {chain}; it serves "
                           f"{served}.{detail}")
        if self.support is Support.UNSUPPORTED:
            return False, self.blocking_reason or f"{self.name} is unsupported"
        if self.support is not Support.VERIFIED:
            return False, (self.blocking_reason
                           or f"{self.name} has never been measured from this "
                              "box; a marketing page is not a measurement")
        if not self.rpc_url:
            return False, f"{self.name} has no resolved endpoint URL"
        return True, ""


#: What the desk knows about providers it does not already run. Every entry is
#: either measured or explicitly marked as not measured; nothing here is
#: promoted into a live ladder by existing.
KNOWN_PROVIDERS: Tuple[Provider, ...] = (
    Provider(
        name="nodeflare",
        terms_url="https://nodeflare.app/",
        rpc_url="",
        chains=("ethereum", "base", "bnb", "arbitrum", "optimism",
                "avalanche", "polygon"),
        support=Support.UNSUPPORTED,
        blocking_reason=(
            "NodeFlare serves EVM chains only (its own documentation says 22-23 "
            "EVM chains). Solana is not EVM, so there is no Solana endpoint to "
            "add. Recorded here so this does not get re-raised as a missing "
            "rung: adding it to the Solana ladder would install a rung that can "
            "never answer, and every failure would read as an outage rather "
            "than a category error. It remains a genuine candidate for the EVM "
            "ladders, blocked on one thing -- the public endpoint URL is on a "
            "page this box cannot reach, and an RPC URL that has not been read "
            "is not an RPC URL."),
        note="EVM-only; free public endpoint, free key raises limits."),
    Provider(
        name="solarchive",
        terms_url="https://huggingface.co/datasets/solarchive/solarchive",
        chains=("solana",),
        support=Support.CANDIDATE,
        blocking_reason=(
            "Bulk parquet archive, not an RPC. Reachability from the desk's "
            "box is the thing to measure; see src/research/solarchive.py, "
            "whose verify() is that measurement."),
        note="Daily parquet partitions from Oct 2020, BigQuery-derived."),
    Provider(
        name="solana_tracker_public_rpc",
        terms_url="https://docs.solanatracker.io/",
        rpc_url="https://rpc.solanatracker.io/public",
        chains=("solana",),
        support=Support.CANDIDATE,
        blocking_reason=(
            "Reported no-signup public Solana RPC -- no account, key or card, "
            "and it accepts sendTransaction as well as reads. Recorded as a "
            "candidate rather than VERIFIED because the build box's egress "
            "proxy refuses the host, so nobody here has spoken to it; the "
            "per-method capability learner measures it on the first box that "
            "can. It belongs in the repair and failover pool, never on the "
            "T0 path."),
        watch_methods=("sendTransaction", "getAccountInfo",
                       "getLatestBlockhash", "getSignaturesForAddress"),
        note=("Keyed free plan is separate: ~500k credits/month, 5 RPS, 2 "
              "websockets, archival calls 10 credits. Their Yellowstone gRPC "
              "is a PAID product (EUR 200/month, 2026-09) and is not a free "
              "Geyser source -- the watcher exists partly so that stops being "
              "a thing anyone has to remember.")),
    Provider(
        name="raptor",
        terms_url="https://docs.solanatracker.io/raptor/overview",
        rpc_url="http://127.0.0.1:8080",
        chains=("solana",),
        support=Support.CANDIDATE,
        blocking_reason=(
            "Challenger execution route. Held in SHADOW by "
            "src/execution/raptor.py until paired realised fills promote it; "
            "'currently free with no rate limits' is a claim on a page, and "
            "the free-tier watcher exists to notice when it stops being one."),
        watch_methods=("/quote", "/swap", "/send-transaction"),
        note="Self-hostable DEX aggregator; hosted API also available."),
)


class ProviderTermsWatcher:
    """Dated snapshots of provider policy, and diffs that mean something.

    The fetcher is injected and must return ``(status, text)``. This module
    performs no I/O of its own beyond reading and writing its own state file.
    """

    def __init__(self, providers: Iterable[Provider] = KNOWN_PROVIDERS, *,
                 fetcher: Optional[Callable[[str], Tuple[int, str]]] = None,
                 state_path: Optional[Path] = None):
        self.providers: List[Provider] = list(providers)
        self.fetcher = fetcher
        self.state_path = Path(state_path) if state_path else None
        self.snapshots: Dict[str, TermsSnapshot] = {}
        self.history: List[TermsChange] = []
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("provider terms state unreadable: %s", exc)
            return
        for name, payload in (state.get("snapshots") or {}).items():
            try:
                self.snapshots[name] = TermsSnapshot.from_dict(payload)
            except (TypeError, ValueError) as exc:
                logger.warning("dropping unreadable snapshot %s: %s", name, exc)

    def save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "schema": PROVIDER_TERMS_SCHEMA_VERSION,
                "snapshots": {name: snap.to_dict()
                              for name, snap in self.snapshots.items()},
            }, indent=2, sort_keys=True))
        except OSError as exc:
            logger.warning("provider terms state unwritable: %s", exc)

    # -- capture ---------------------------------------------------------

    def capture(self, provider: Provider, *, now: Optional[float] = None
                ) -> TermsSnapshot:
        stamp = time.time() if now is None else now
        if not provider.terms_url:
            return TermsSnapshot(provider=provider.name, url="",
                                 fetched_at=stamp, content_hash="",
                                 status=0, error="no terms url recorded")
        if self.fetcher is None:
            return TermsSnapshot(provider=provider.name, url=provider.terms_url,
                                 fetched_at=stamp, content_hash="", status=0,
                                 error="no fetcher configured")
        try:
            status, text = self.fetcher(provider.terms_url)
        except Exception as exc:
            return TermsSnapshot(
                provider=provider.name, url=provider.terms_url,
                fetched_at=stamp, content_hash="", status=0,
                error=f"{type(exc).__name__}: {exc}")
        body = text or ""
        if status != 200:
            return TermsSnapshot(
                provider=provider.name, url=provider.terms_url,
                fetched_at=stamp, content_hash="", status=int(status),
                error=f"HTTP {status}", bytes_read=len(body))
        return TermsSnapshot(
            provider=provider.name, url=provider.terms_url, fetched_at=stamp,
            content_hash=hashlib.sha256(body.encode("utf-8", "replace")
                                        ).hexdigest(),
            signals=extract_signals(body, methods=provider.watch_methods),
            status=200, bytes_read=len(body))

    # -- diffing ---------------------------------------------------------

    @staticmethod
    def diff(previous: TermsSnapshot, current: TermsSnapshot
             ) -> List[TermsChange]:
        """What changed, and whether it changed against us."""
        changes: List[TermsChange] = []
        name = current.provider
        before_q = dict((previous.signals or {}).get("quotas") or {})
        after_q = dict((current.signals or {}).get("quotas") or {})
        for field_name in sorted(set(before_q) | set(after_q)):
            was = before_q.get(field_name)
            now = after_q.get(field_name)
            if was == now:
                continue
            if was is None:
                severity, detail = Severity.INFORMATIONAL, "newly stated"
            elif now is None:
                severity, detail = Severity.BREAKING, "allowance no longer stated"
            elif now < was:
                severity, detail = Severity.BREAKING, (
                    f"cut to {now / was:.0%} of the previous allowance")
            else:
                severity, detail = Severity.INFORMATIONAL, "raised"
            changes.append(TermsChange(provider=name, field=field_name,
                                       before=was, after=now,
                                       severity=severity, detail=detail))

        good_flag = {name_: good for name_, _, good in _FLAG_PATTERNS}
        before_f = dict((previous.signals or {}).get("flags") or {})
        after_f = dict((current.signals or {}).get("flags") or {})
        for field_name in sorted(set(before_f) | set(after_f)):
            was = before_f.get(field_name)
            now = after_f.get(field_name)
            if was == now:
                continue
            favourable = good_flag.get(field_name, True)
            # Gaining a good phrase or losing a bad one is fine; the reverse
            # is the free tier ending.
            against_us = (favourable and was and not now) or (
                not favourable and now and not was)
            changes.append(TermsChange(
                provider=name, field=f"flag:{field_name}", before=was,
                after=now,
                severity=Severity.BREAKING if against_us
                else Severity.INFORMATIONAL,
                detail="appeared" if now else "disappeared"))

        before_m = dict((previous.signals or {}).get("methods") or {})
        after_m = dict((current.signals or {}).get("methods") or {})
        for field_name in sorted(set(before_m) | set(after_m)):
            was = before_m.get(field_name)
            now = after_m.get(field_name)
            if was == now:
                continue
            changes.append(TermsChange(
                provider=name, field=f"method:{field_name}", before=was,
                after=now,
                severity=Severity.BREAKING if was and not now
                else Severity.INFORMATIONAL,
                detail="no longer listed" if was else "newly listed"))

        if not changes and previous.content_hash != current.content_hash:
            changes.append(TermsChange(
                provider=name, field="content_hash",
                before=previous.content_hash[:12],
                after=current.content_hash[:12], severity=Severity.COSMETIC,
                detail=("page changed but no policy signal did; recorded so a "
                        "later breaking change has a baseline, not escalated")))
        return changes

    # -- the pass --------------------------------------------------------

    def poll(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        """One surveillance pass over every provider with a terms URL."""
        stamp = time.time() if now is None else now
        changes: List[TermsChange] = []
        blocked: List[Dict[str, str]] = []
        checked = 0
        for provider in self.providers:
            snapshot = self.capture(provider, now=stamp)
            if not snapshot.usable:
                blocked.append({"provider": provider.name,
                                "reason": snapshot.error or "unusable"})
                # A failed read must not overwrite a good baseline; otherwise
                # one 503 erases the history the next diff depends on.
                continue
            checked += 1
            previous = self.snapshots.get(provider.name)
            if previous is not None and previous.usable:
                changes.extend(self.diff(previous, snapshot))
            self.snapshots[provider.name] = snapshot
        self.history.extend(changes)
        self.save()
        breaking = [item for item in changes
                    if item.severity is Severity.BREAKING]
        return {
            "checked": checked,
            "providers": len(self.providers),
            "data_blocked": blocked,
            "changes": [item.to_dict() for item in changes],
            "breaking": [item.to_dict() for item in breaking],
            "status": "BREAKING_CHANGE" if breaking else (
                "DATA_BLOCKED" if checked == 0 else "OK"),
            "checked_at": stamp,
        }

    # -- questions the runtime asks --------------------------------------

    def provider(self, name: str) -> Optional[Provider]:
        for item in self.providers:
            if item.name == name:
                return item
        return None

    def usable_rungs(self, chain: str) -> List[Provider]:
        return [item for item in self.providers if item.usable_for(chain)[0]]

    def status(self, chain: str = "solana") -> Dict[str, Any]:
        rows = []
        for item in self.providers:
            ok, reason = item.usable_for(chain)
            rows.append({"provider": item.name, "usable": ok,
                         "support": item.support.value, "reason": reason,
                         "last_checked": (
                             self.snapshots[item.name].fetched_at
                             if item.name in self.snapshots else None)})
        return {"chain": chain, "providers": rows,
                "usable": sum(1 for row in rows if row["usable"])}
