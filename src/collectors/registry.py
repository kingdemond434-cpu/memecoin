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

import yaml

from src.collectors.event_source import EventSource, SourceClass
from src.collectors import adapters

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = "v1"

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
    unconfigured: List[str] = field(default_factory=list)
    problems: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "declared": self.declared, "ready": self.ready,
            "ready_share": (self.ready / self.declared) if self.declared else None,
            "by_state": dict(sorted(self.by_state.items())),
            "by_kind": dict(sorted(self.by_kind.items())),
            "by_language": dict(sorted(self.by_language.items())),
            "by_region": dict(sorted(self.by_region.items())),
            # Named, because a source dropped for a missing key is a coverage
            # hole that no health check will ever mention.
            "unconfigured": sorted(self.unconfigured),
            "problems": [{"source": source, "reason": reason}
                         for source, reason in self.problems],
        }


def load_declarations(path: str) -> List[SourceDeclaration]:
    """Parse the source universe from YAML.

    A malformed entry is skipped with its id reported rather than raising:
    one bad line in a 400-source file must not take the whole mesh offline.
    """
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
            sources.append(factory(declaration.source_id, fetch, **declaration.options))
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
