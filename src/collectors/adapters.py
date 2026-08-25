"""Adapters for the lawful public source universe.

Each one does exactly two things: fetch, and normalise into `Event`. No
adapter judges relevance -- a source that filters on its own idea of what
matters silently becomes the model, and its filtering is invisible to
everything downstream that tries to score it.

Every adapter is constructed from configuration the operator supplies, and
none of them reaches anything behind an access control. Where a network offers
no lawful public interface, there is deliberately no adapter rather than a
scraper: the gap is real and pretending otherwise puts the whole mesh's
provenance in doubt.

The extraction helpers are shared and strict. A token address must decode to
32 bytes before it is treated as one, for the same reason the authenticity
resolver requires it: a candidate mint is one step from a purchase, and any
40-character word will otherwise become one.
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.collectors.event_source import Event, EventSource, SourceClass
from src.strategies.authenticity import extract_mints, normalise_host

logger = logging.getLogger(__name__)

_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def extract_urls(text: str) -> List[str]:
    return _URL.findall(text or "")


def normalise(
    source_id: str,
    source_class: SourceClass,
    text: str,
    source_at: float,
    observed_at: Optional[float] = None,
    author_id: str = "",
    language: str = "",
    entity_ids: Sequence[str] = (),
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """Build one canonical event, extracting tokens and links from the text."""
    return Event(
        source_id=source_id,
        source_class=source_class,
        source_at=float(source_at),
        observed_at=float(observed_at if observed_at is not None else time.time()),
        text=text or "",
        language=language,
        entity_ids=tuple(entity_ids),
        token_addresses=tuple(extract_mints(text or "")),
        urls=tuple(extract_urls(text or "")),
        author_id=author_id,
        metadata=dict(metadata or {}),
    )


class CallableSource(EventSource):
    """Adapter over any async callable returning raw records.

    The seam every concrete adapter shares. A network client is injected
    rather than constructed here, so the mesh is testable without a network
    and an adapter's normalisation can be verified independently of whatever
    library fetches for it.
    """

    def __init__(
        self,
        source_id: str,
        source_class: SourceClass,
        fetch: Callable[[], Any],
        to_event: Callable[[Dict[str, Any], float], Optional[Event]],
        language: str = "",
    ):
        super().__init__(source_id, source_class)
        self._fetch = fetch
        self._to_event = to_event
        self.language = language

    async def poll(self) -> List[Event]:
        records = await self._fetch()
        now = time.time()
        events: List[Event] = []
        for record in records or ():
            try:
                event = self._to_event(record, now)
            except (KeyError, TypeError, ValueError) as exc:
                # One malformed record must not lose the rest of the batch.
                logger.debug("source %s skipped a record: %s", self.source_id, exc)
                continue
            if event is not None:
                events.append(event)
        return events


def telegram_source(source_id: str, fetch, channel: str = "") -> CallableSource:
    """Public Telegram channel. MTProto pushes, so lag is usually sub-second."""
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = record.get("message") or record.get("text") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.CHAT, text,
            source_at=float(record.get("date") or now), observed_at=now,
            author_id=str(record.get("sender_id") or channel),
            metadata={"channel": channel, "message_id": record.get("id")})
    return CallableSource(source_id, SourceClass.CHAT, fetch, to_event)


def bluesky_source(source_id: str, fetch) -> CallableSource:
    """Bluesky firehose/Jetstream records, filtered locally against the entity set."""
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = ((record.get("commit") or {}).get("record") or {}).get("text") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.SOCIAL, text,
            source_at=float(record.get("time_us", now * 1e6)) / 1e6, observed_at=now,
            author_id=str(record.get("did") or ""))
    return CallableSource(source_id, SourceClass.SOCIAL, fetch, to_event)


def nostr_source(source_id: str, fetch) -> CallableSource:
    """Nostr relay events, subscribable by author public key."""
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = record.get("content") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.SOCIAL, text,
            source_at=float(record.get("created_at") or now), observed_at=now,
            author_id=str(record.get("pubkey") or ""))
    return CallableSource(source_id, SourceClass.SOCIAL, fetch, to_event)


def farcaster_source(source_id: str, fetch) -> CallableSource:
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = record.get("text") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.SOCIAL, text,
            source_at=float(record.get("timestamp") or now), observed_at=now,
            author_id=str((record.get("author") or {}).get("fid") or ""))
    return CallableSource(source_id, SourceClass.SOCIAL, fetch, to_event)


def youtube_websub_source(source_id: str, fetch) -> CallableSource:
    """YouTube WebSub push: title and description arrive before any transcript.

    Title and description are used deliberately. Waiting for a transcript to
    confirm what a title already states is spending the entire lead the push
    notification bought.
    """
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = " ".join(filter(None, [record.get("title"), record.get("description")]))
        if not text:
            return None
        return normalise(
            source_id, SourceClass.VIDEO, text,
            source_at=float(record.get("published") or now), observed_at=now,
            author_id=str(record.get("channel_id") or ""),
            metadata={"video_id": record.get("video_id")})
    return CallableSource(source_id, SourceClass.VIDEO, fetch, to_event)


def twitch_eventsub_source(source_id: str, fetch) -> CallableSource:
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        event = record.get("event") or {}
        text = " ".join(filter(None, [event.get("title"), event.get("category_name")]))
        if not text:
            return None
        return normalise(
            source_id, SourceClass.VIDEO, text,
            source_at=float(event.get("started_at_epoch") or now), observed_at=now,
            author_id=str(event.get("broadcaster_user_id") or ""))
    return CallableSource(source_id, SourceClass.VIDEO, fetch, to_event)


def mastodon_source(source_id: str, fetch) -> CallableSource:
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = record.get("content") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.SOCIAL, text,
            source_at=float(record.get("created_at_epoch") or now), observed_at=now,
            author_id=str((record.get("account") or {}).get("acct") or ""),
            language=record.get("language") or "")
    return CallableSource(source_id, SourceClass.SOCIAL, fetch, to_event)


def discord_gateway_source(source_id: str, fetch, guild: str = "") -> CallableSource:
    """Only servers the bot has been legitimately granted access to."""
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = record.get("content") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.CHAT, text,
            source_at=float(record.get("timestamp_epoch") or now), observed_at=now,
            author_id=str((record.get("author") or {}).get("id") or ""),
            metadata={"guild": guild, "channel_id": record.get("channel_id")})
    return CallableSource(source_id, SourceClass.CHAT, fetch, to_event)


def rss_source(source_id: str, fetch, language: str = "",
               source_class: SourceClass = SourceClass.FEED) -> CallableSource:
    """RSS/Atom/WebSub. Covers government, corporate, regional and niche press.

    One adapter for the whole long tail, because the tail is where the lead
    usually is: a local outlet breaking a story before global media is the
    case worth catching, and it is served by exactly the same parser.
    """
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = " ".join(filter(None, [record.get("title"), record.get("summary")]))
        if not text:
            return None
        return normalise(
            source_id, source_class, text,
            source_at=float(record.get("published_epoch") or now), observed_at=now,
            language=language or record.get("language") or "",
            metadata={"link": record.get("link"),
                      "host": normalise_host(record.get("link") or "")})
    return CallableSource(source_id, source_class, fetch, to_event, language=language)


def official_site_source(source_id: str, fetch, domain: str) -> CallableSource:
    """Change detection over an official page.

    An official domain is the strongest non-chain proof the authenticity
    resolver accepts, so this feeds Level B directly.
    """
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = record.get("changed_text") or ""
        if not text:
            return None
        return normalise(
            source_id, SourceClass.OFFICIAL, text,
            source_at=float(record.get("changed_at") or now), observed_at=now,
            metadata={"domain": domain, "path": record.get("path"),
                      "content_hash": record.get("content_hash")})
    return CallableSource(source_id, SourceClass.OFFICIAL, fetch, to_event)


def code_repository_source(source_id: str, fetch) -> CallableSource:
    """Public repository activity, which occasionally names a mint before marketing."""
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = " ".join(filter(None, [
            record.get("message"), record.get("body"), record.get("tag")]))
        if not text:
            return None
        return normalise(
            source_id, SourceClass.CODE, text,
            source_at=float(record.get("committed_at") or now), observed_at=now,
            author_id=str(record.get("author") or ""),
            metadata={"repo": record.get("repo"), "sha": record.get("sha")})
    return CallableSource(source_id, SourceClass.CODE, fetch, to_event)


def metadata_artifact_source(source_id: str, fetch) -> CallableSource:
    """Public token metadata (IPFS/Arweave), which can land before or with the mint."""
    def to_event(record: Dict[str, Any], now: float) -> Optional[Event]:
        text = " ".join(filter(None, [
            record.get("name"), record.get("symbol"), record.get("description")]))
        if not text and not record.get("mint"):
            return None
        event = normalise(
            source_id, SourceClass.WEB, text,
            source_at=float(record.get("uploaded_at") or now), observed_at=now,
            metadata={"uri": record.get("uri"), "image_hash": record.get("image_hash"),
                      "host": normalise_host(record.get("uri") or "")})
        mint = record.get("mint")
        if mint and mint not in event.token_addresses:
            # The mint may be a structured field rather than in the text.
            return Event(**{**event.__dict__,
                            "token_addresses": tuple([*event.token_addresses, mint])})
        return event
    return CallableSource(source_id, SourceClass.WEB, fetch, to_event)


#: Networks with no lawful public firehose. Named so the gap is a stated
#: limitation of the mesh rather than an unnoticed hole in its coverage.
UNCOVERED_NETWORKS = (
    "x_twitter",      # paid API only
    "instagram",      # no public firehose
    "tiktok",         # no public firehose
    "private_groups", # access-controlled by definition
)


def coverage_report(mesh_health: Dict[str, Any]) -> Dict[str, Any]:
    """Mesh health plus an explicit statement of what is not covered at all.

    Coverage over connected sources answers the wrong question on its own: a
    mesh can be 100% healthy across four adapters and still be blind to most
    of the world. Naming the uncovered networks keeps that visible.
    """
    return {**mesh_health, "uncovered_networks": list(UNCOVERED_NETWORKS),
            "coverage_is_over_connected_sources_only": True}
