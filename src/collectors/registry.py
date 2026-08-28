"""Declaring hundreds of sources as configuration, and discovering more.

Adapters are not coverage. Twelve working adapters and four configured feeds
is a mesh that is 100% healthy and blind to most of the world, and the gap
between "we can read Telegram" and "we read the 300 channels that matter" is
the entire remaining job on the information side.

That gap is closed with configuration, not code. Sources are declared in YAML
-- id, kind, language, region, and whatever the adapter needs -- and this
module instantiates the mesh from it. Adding the 47th Korean regional outlet
is then an edit, not a pull request, which is the only way a source universe
this size stays maintainable.

Two things keep the registry honest.

Credentials are named, never embedded. A declaration says which environment
variable holds its secret; the registry reads presence, never value, and a
source whose credential is absent is reported as UNCONFIGURED rather than
silently skipped. A source that vanishes from the mesh because its key was
missing is a coverage hole nobody sees.

Discovery is evidence-gated. A source that repeatedly appears upstream of
sources we already trust is promoted automatically -- that is the whole point
of tracking repeaters -- but only after enough independent observations that
the ordering is not chance. Promoting on two coincidences fills the mesh with
noise that then has to be scored, and scoring costs more than the source was
ever worth.
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import yaml

from src.collectors.event_source import EventSource, SourceClass
from src.collectors import adapters

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = "v1"

# Successful-poll cadence by transport. Push/stream adapters merely drain a
# local queue and stay sub-second; remote HTTP/RSS endpoints are not hammered
# once per second. Per-source YAML can override these measurements.
DEFAULT_POLL_INTERVALS: Dict[str, float] = {
    "telegram": 0.10,
    "bluesky": 0.05,
    "nostr": 0.05,
    "discord": 0.10,
    "youtube": 0.25,
    "twitch": 0.25,
    "farcaster": 10.0,
    "mastodon": 15.0,
    "rss": 60.0,
    "official_site": 60.0,
    "metadata": 30.0,
    "code_repo": 300.0,
}

#: Declaration kind -> adapter factory. Every factory takes (source_id, fetch)
#: plus its own keywords, so a new kind is one entry here.
ADAPTER_KINDS: Dict[str, Callable[..., EventSource]] = {
    "telegram": adapters.telegram_source,
    "bluesky": adapters.bluesky_source,
    "nostr": adapters.nostr_source,
    "farcaster": adapters.farcaster_source,
    "youtube": adapters.youtube_websub_source,
    "twitch": adapters.twitch_eventsub_source,
    "mastodon": adapters.mastodon_source,
    "discord": adapters.discord_gateway_source,
    "rss": adapters.rss_source,
    "official_site": adapters.official_site_source,
    "code_repo": adapters.code_repository_source,
    "metadata": adapters.metadata_artifact_source,
}


def _adapter_options(declaration: "SourceDeclaration") -> Dict[str, Any]:
    """Translate declaration metadata to normaliser options.

    A declaration's ``options`` primarily configure its network transport
    (URL, relay, repository, instance).  Adapter factories normalise records
    already fetched by that transport and intentionally do not accept those
    connection parameters.  Passing the whole mapping made every real RSS,
    Mastodon, Nostr and repository transport look like ``NO_FETCHER`` even
    while it was running.  Keep the two interfaces explicit.
    """
    options = declaration.options
    if declaration.kind == "rss":
        # Top-level language is canonical. Keep accepting the older
        # declaration shape so verified overlays produced before the schema
        # cleanup do not silently lose multilingual classification.
        return {"language": declaration.language or str(options.get("language", ""))}
    if declaration.kind == "telegram":
        return {"channel": str(options.get("channel", ""))}
    if declaration.kind == "discord":
        return {"guild": str(options.get("guild", ""))}
    if declaration.kind == "official_site":
        domain = str(options.get("domain", "")).strip()
        if not domain:
            domain = (urlsplit(str(options.get("url", ""))).hostname or "").lower()
        return {"domain": domain}
    return {}


class DeclarationState(Enum):
    READY = "READY"
    UNCONFIGURED = "UNCONFIGURED"
    UNKNOWN_KIND = "UNKNOWN_KIND"
    NO_FETCHER = "NO_FETCHER"


@dataclass
class SourceDeclaration:
    """One line of the source universe."""

    source_id: str
    kind: str
    language: str = ""
    region: str = ""
    tier: int = 3
    requires_env: Sequence[str] = ()
    options: Dict[str, Any] = field(default_factory=dict)
    # Expected cadence. A push channel and a regional daily are not the same
    # kind of silence, and one universal clock either calls the healthy feed
    # dead or lets the dead one look healthy.
    degraded_after_seconds: Optional[float] = None
    dead_after_seconds: Optional[float] = None
    # How often this source is worth ASKING, which is a different number from
    # how long its silence means something. A regional newspaper polled every
    # second is four thousand pointless requests an hour and an eventual
    # block; a chat channel polled hourly is a chat channel we are not
    # reading. Declared per source because only the declaration knows which
    # this is.
    poll_interval_seconds: Optional[float] = None

    def missing_credentials(self) -> List[str]:
        """Which required environment variables are absent.

        Presence only. The registry never reads a secret's value, and nothing
        here logs one.
        """
        return [name for name in self.requires_env if not os.getenv(name)]


@dataclass
class RegistryReport:
    declared: int = 0
    ready: int = 0
    by_state: Dict[str, int] = field(default_factory=dict)
    by_kind: Dict[str, int] = field(default_factory=dict)
    by_language: Dict[str, int] = field(default_factory=dict)
    by_region: Dict[str, int] = field(default_factory=dict)
    by_cadence: Dict[str, int] = field(default_factory=dict)
    unconfigured: List[str] = field(default_factory=list)
    problems: List[Tuple[str, str]] = field(default_factory=list)

    def ready_share_or_zero(self) -> float:
        return (self.ready / self.declared) if self.declared else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "declared": self.declared, "ready": self.ready,
            "ready_share": (self.ready / self.declared) if self.declared else None,
            "by_state": dict(sorted(self.by_state.items())),
            "by_kind": dict(sorted(self.by_kind.items())),
            "by_language": dict(sorted(self.by_language.items())),
            "by_region": dict(sorted(self.by_region.items())),
            # How the declared universe is spread across cadences. A registry
            # of four hundred sources all polled every second is a registry
            # that will be rate-limited into uselessness by lunchtime.
            "by_cadence": dict(sorted(self.by_cadence.items())),
            # Named, because a source dropped for a missing key is a coverage
            # hole that no health check will ever mention.
            "unconfigured": sorted(self.unconfigured),
            "problems": [{"source": source, "reason": reason}
                         for source, reason in self.problems],
        }


def expand_env_channels(declarations: Sequence[SourceDeclaration],
                        ) -> List[SourceDeclaration]:
    """Turn `TELEGRAM_CHANNELS` into mesh declarations.

    The social collector already reads that variable, and asking an operator
    to list their channels twice -- once in an env var and once in YAML -- is
    asking for two lists that disagree. One list, two consumers.

    Channels already named by a declaration are left alone, so a channel that
    needs its own cadence or tier can still be declared explicitly and is not
    duplicated by this.
    """
    raw = os.getenv("TELEGRAM_CHANNELS", "")
    wanted = [item.strip().lstrip("@") for item in raw.split(",") if item.strip()]
    if not wanted:
        return list(declarations)

    existing = list(declarations)
    already = {str(item.options.get("channel", "")).lstrip("@")
               for item in existing if item.kind == "telegram"}
    # The declared telegram entries carry POLLING POLICY -- tier, cadence,
    # what silence means -- which is the same for any crypto chat channel and
    # is worth inheriting. What they do NOT lend is language or region: those
    # are claims about a channel's content, and assigning one by list position
    # would invent an attribute nobody supplied. A channel whose language
    # matters gets its own declaration in YAML.
    template = next((item for item in existing
                     if item.kind == "telegram" and not item.options.get("channel")),
                    None)
    added: List[SourceDeclaration] = []
    for channel in wanted:
        if channel in already:
            continue
        added.append(SourceDeclaration(
            source_id=f"telegram:{channel}", kind="telegram",
            tier=(template.tier if template else 1),
            requires_env=("TELEGRAM_API_ID", "TELEGRAM_API_HASH"),
            degraded_after_seconds=(template.degraded_after_seconds
                                    if template else 60.0) or 60.0,
            dead_after_seconds=(template.dead_after_seconds
                                if template else 300.0) or 300.0,
            poll_interval_seconds=(template.poll_interval_seconds
                                   if template else 1.0) or 1.0,
            options={"channel": channel}))
    logger.info("TELEGRAM_CHANNELS supplied %d channel(s); %d added to the mesh",
                len(wanted), len(added))
    return existing + added


def _cadence_bucket(seconds: float) -> str:
    """Human bucket for a poll interval, for the coverage report."""
    value = float(seconds)
    if value <= 5:
        return "realtime"
    if value <= 60:
        return "minute"
    if value <= 900:
        return "quarter_hour"
    if value <= 3_600:
        return "hourly"
    if value <= 21_600:
        return "six_hourly"
    return "daily"


def load_declarations(path: str) -> List[SourceDeclaration]:
    """Parse the source universe from YAML.

    ``path`` may name several files, comma separated. Later files OVERRIDE
    earlier ones by source_id, which is what lets an operator keep the seed
    registry under version control and layer a machine-verified overlay --
    `tools/verify_sources.py --out config/sources.verified.yaml` -- on top of
    it without editing the seed. A file that does not exist is skipped
    silently only when it is one of several: a single missing registry is an
    error worth reporting.

    A malformed entry is skipped with its id reported rather than raising:
    one bad line in a 400-source file must not take the whole mesh offline.
    """
    paths = [item.strip() for item in str(path).split(",") if item.strip()]
    if len(paths) > 1:
        merged: Dict[str, SourceDeclaration] = {}
        for one in paths:
            if not os.path.exists(one):
                logger.info("source registry overlay %s absent; skipping", one)
                continue
            for declaration in load_declarations(one):
                merged[declaration.source_id] = declaration
        return list(merged.values())

    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("source registry unreadable at %s: %s", path, exc)
        return []
    declarations: List[SourceDeclaration] = []
    for entry in raw.get("sources") or []:
        try:
            declarations.append(SourceDeclaration(
                source_id=str(entry["id"]), kind=str(entry["kind"]),
                language=str(entry.get("language", "")),
                region=str(entry.get("region", "")),
                tier=int(entry.get("tier", 3)),
                requires_env=tuple(entry.get("requires_env", ()) or ()),
                options=dict(entry.get("options", {}) or {}),
                degraded_after_seconds=(float(entry["degraded_after_seconds"])
                                        if entry.get("degraded_after_seconds") is not None
                                        else None),
                dead_after_seconds=(float(entry["dead_after_seconds"])
                                    if entry.get("dead_after_seconds") is not None
                                    else None),
                poll_interval_seconds=(float(entry["poll_interval_seconds"])
                                       if entry.get("poll_interval_seconds") is not None
                                       else None),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("skipping malformed source declaration %r: %s", entry, exc)
    return declarations


def build_sources(
    declarations: Sequence[SourceDeclaration],
    fetchers: Dict[str, Callable[[], Any]],
) -> Tuple[List[EventSource], RegistryReport]:
    """Instantiate every declaration that is ready, and report the rest.

    ``fetchers`` maps source_id to the async callable that actually talks to
    the network. Injecting them keeps the registry testable without
    credentials and stops this module from owning transport concerns.
    """
    report = RegistryReport(declared=len(declarations))
    sources: List[EventSource] = []

    for declaration in declarations:
        report.by_kind[declaration.kind] = report.by_kind.get(declaration.kind, 0) + 1
        if declaration.poll_interval_seconds is not None:
            bucket = _cadence_bucket(declaration.poll_interval_seconds)
            report.by_cadence[bucket] = report.by_cadence.get(bucket, 0) + 1
        if declaration.language:
            report.by_language[declaration.language] = (
                report.by_language.get(declaration.language, 0) + 1)
        if declaration.region:
            report.by_region[declaration.region] = (
                report.by_region.get(declaration.region, 0) + 1)

        def mark(state: DeclarationState, reason: str = "") -> None:
            report.by_state[state.value] = report.by_state.get(state.value, 0) + 1
            if state is DeclarationState.UNCONFIGURED:
                report.unconfigured.append(declaration.source_id)
            elif state is not DeclarationState.READY:
                report.problems.append((declaration.source_id, reason))

        factory = ADAPTER_KINDS.get(declaration.kind)
        if factory is None:
            mark(DeclarationState.UNKNOWN_KIND, f"no adapter for kind {declaration.kind}")
            continue
        missing = declaration.missing_credentials()
        if missing:
            mark(DeclarationState.UNCONFIGURED, f"missing {', '.join(missing)}")
            continue
        fetch = fetchers.get(declaration.source_id)
        if fetch is None:
            mark(DeclarationState.NO_FETCHER, "no fetcher supplied for this source")
            continue

        try:
            # Only the options this adapter actually declares. A source's
            # options describe the SOURCE, and its two consumers want
            # different parts: the transport needs the endpoint, the adapter
            # needs the language. Passing the whole block to both meant every
            # declaration that named a URL was rejected by its own adapter as
            # an unexpected keyword -- and reported NO_FETCHER, which reads as
            # a missing transport rather than as a misrouted option.
            source = factory(declaration.source_id, fetch,
                             **_adapter_options(declaration))
            if declaration.degraded_after_seconds is not None:
                source.degraded_after_seconds = declaration.degraded_after_seconds
            if declaration.dead_after_seconds is not None:
                source.dead_after_seconds = declaration.dead_after_seconds
            source.poll_interval_seconds = max(
                0.01, float(
                    declaration.poll_interval_seconds
                    if declaration.poll_interval_seconds is not None
                    else DEFAULT_POLL_INTERVALS.get(declaration.kind, 30.0)))
            sources.append(source)
        except TypeError as exc:
            mark(DeclarationState.NO_FETCHER, f"adapter rejected its options: {exc}")
            continue
        mark(DeclarationState.READY)
        report.ready += 1

    return sources, report


@dataclass
class Candidate:
    """A source seen leading known sources, not yet promoted."""

    source_id: str
    led_observations: int = 0
    total_observations: int = 0
    led_sources: Dict[str, int] = field(default_factory=dict)

    @property
    def lead_rate(self) -> Optional[float]:
        if self.total_observations == 0:
            return None
        return self.led_observations / self.total_observations


class SourceDiscovery:
    """Promotes sources that repeatedly appear upstream of ones we trust.

    Fed from the mesh's repeater ordering: when a piece of content arrives
    from several sources, whoever was first led the others. A source that is
    consistently first is worth watching directly.

    Gated on evidence. Promoting on two coincidences fills the mesh with noise
    that then has to be scored, and scoring costs more than the source was
    ever worth. Both a minimum observation count and a minimum lead rate must
    hold, and the observations have to be across DISTINCT known sources --
    leading the same channel twenty times is one relationship, not twenty.
    """

    def __init__(self, min_observations: int = 8, min_lead_rate: float = 0.6,
                 min_distinct_followers: int = 2):
        self.min_observations = max(2, min_observations)
        self.min_lead_rate = min_lead_rate
        self.min_distinct_followers = max(1, min_distinct_followers)
        self._candidates: Dict[str, Candidate] = {}

    def observe(self, ordered_sources: Sequence[str], known: Iterable[str]) -> None:
        """Record one content item's arrival order across sources."""
        known_set = set(known)
        if len(ordered_sources) < 2:
            return
        leader, *followers = ordered_sources
        if leader in known_set:
            # Already in the mesh; nothing to discover.
            return
        candidate = self._candidates.setdefault(leader, Candidate(leader))
        candidate.total_observations += 1
        known_followers = [name for name in followers if name in known_set]
        if known_followers:
            candidate.led_observations += 1
            for name in known_followers:
                candidate.led_sources[name] = candidate.led_sources.get(name, 0) + 1

    def promotable(self) -> List[Candidate]:
        ready = [
            candidate for candidate in self._candidates.values()
            if candidate.total_observations >= self.min_observations
            and (candidate.lead_rate or 0) >= self.min_lead_rate
            and len(candidate.led_sources) >= self.min_distinct_followers
        ]
        ready.sort(key=lambda item: (item.lead_rate or 0, item.led_observations),
                   reverse=True)
        return ready

    def report(self) -> Dict[str, Any]:
        return {
            "tracked": len(self._candidates),
            "promotable": [
                {"source": candidate.source_id, "lead_rate": candidate.lead_rate,
                 "observations": candidate.total_observations,
                 "distinct_followers": len(candidate.led_sources)}
                for candidate in self.promotable()
            ],
            "gate": {"min_observations": self.min_observations,
                     "min_lead_rate": self.min_lead_rate,
                     "min_distinct_followers": self.min_distinct_followers},
        }
