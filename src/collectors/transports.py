"""Production transports for the source mesh.

The adapters in ``adapters.py`` normalise raw records into ``Event``. This is
the other half: what actually talks to the network and produces those records.
Without it ``build_sources`` runs from an empty fetcher map, every declaration
reports NO_FETCHER, and the mesh is an adapter library rather than a source of
signal -- which is exactly what "we have adapters" hides.

Three rules hold throughout.

**Lawful public interfaces only.** Every transport here reads something
published for anonymous readers, or something the operator has been granted
access to with their own credential. Nothing bypasses an access control,
nothing uses a borrowed account, and where a network offers no lawful public
interface there is deliberately no transport rather than a scraper.

**Credentials by name.** A transport is told the NAME of the environment
variable holding its credential and reads it at connect time. No value is
logged, stored, echoed into a report, or included in an exception message.

**A transport that cannot reach its endpoint says so by raising.** It never
returns an empty batch to paper over a failure: ``EventSource.collect``
counts consecutive failures and the mesh health surface reports them, and a
silent empty list turns a dead feed into a quiet one.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ElementTree
from collections import deque
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

TRANSPORTS_SCHEMA_VERSION = "v1"

USER_AGENT = "memecoin-source-mesh/1.0 (+public endpoint reader)"
DEFAULT_TIMEOUT_S = 10.0

# How many records one poll may return. A feed that suddenly serves ten
# thousand entries is either broken or backfilling, and either way handing all
# of them to a decision path is worse than dropping the tail.
MAX_BATCH = 200

# How many pushed records a streaming transport buffers between polls. Bounded
# and oldest-dropping, because on a burst the newest records are the ones a
# latency-sensitive desk wants and an unbounded buffer is a memory leak with a
# schedule.
STREAM_BUFFER = 2_000

_TAG = re.compile(r"<[^>]+>")


def strip_html(value: str) -> str:
    """Tags out, entities decoded. Mastodon and RSS both serve HTML bodies."""
    return html.unescape(_TAG.sub(" ", value or "")).strip()


def parse_timestamp(value: Any) -> Optional[float]:
    """Epoch seconds from the several shapes feeds actually use.

    Returns None rather than "now" when it cannot tell. A publication time
    defaulted to the moment we read it makes every stale item look fresh,
    which is the direction that manufactures lead time we did not have.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # Milliseconds and microseconds both appear in the wild.
        if number > 1e14:
            return number / 1e6
        if number > 1e11:
            return number / 1e3
        return number
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            from datetime import datetime

            return datetime.fromisoformat(candidate).timestamp()
        except ValueError:
            continue
    try:
        return parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError):
        return None


class TransportError(RuntimeError):
    """A transport could not reach or could not read its endpoint."""


@dataclass
class HttpClient:
    """Shared aiohttp session with a timeout and a stated user agent.

    One session across every HTTP transport: hundreds of sources each holding
    their own connection pool is hundreds of idle sockets, and the connector's
    per-host limit is what stops one slow host from starving the rest.
    """

    timeout_s: float = DEFAULT_TIMEOUT_S
    limit: int = 100
    limit_per_host: int = 4
    _session: Any = field(default=None, repr=False)

    async def session(self):
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                connector=aiohttp.TCPConnector(limit=self.limit,
                                               limit_per_host=self.limit_per_host),
                headers={"User-Agent": USER_AGENT})
        return self._session

    async def get(self, url: str, *, headers: Optional[Dict[str, str]] = None
                  ) -> Tuple[int, str, Dict[str, str]]:
        """Status, body and response headers. Raises only on transport failure."""
        session = await self.session()
        try:
            async with session.get(url, headers=headers or {}) as response:
                body = await response.text()
                return response.status, body, dict(response.headers)
        except asyncio.TimeoutError as exc:
            raise TransportError(f"timeout after {self.timeout_s}s") from exc
        except Exception as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class Transport:
    """Base for anything that produces raw records for an adapter."""

    #: Reported by ``transport_report`` so an operator can see what is wired
    #: without reading the code.
    kind = "transport"

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.polls = 0
        self.records = 0
        self.failures = 0
        self.last_error = ""

    async def __call__(self) -> List[Dict[str, Any]]:
        self.polls += 1
        try:
            records = await self.fetch()
        except TransportError as exc:
            self.failures += 1
            # The message, never the credential. Transports raise with the
            # endpoint and the failure mode and nothing else.
            self.last_error = str(exc)
            raise
        self.records += len(records)
        return records

    async def fetch(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def report(self) -> Dict[str, Any]:
        return {"source_id": self.source_id, "kind": self.kind,
                "polls": self.polls, "records": self.records,
                "failures": self.failures, "last_error": self.last_error}


class SeenSet:
    """Bounded 'have I already returned this' memory.

    A feed re-serves the same items on every request, so without this every
    poll re-emits the whole page and the mesh's dedupe window becomes the only
    thing between a story and being counted fifty times.
    """

    def __init__(self, capacity: int = 5_000):
        self.capacity = capacity
        self._order: Deque[str] = deque()
        self._members: set = set()

    def add_if_new(self, key: str) -> bool:
        if not key or key in self._members:
            return False
        self._members.add(key)
        self._order.append(key)
        while len(self._order) > self.capacity:
            self._members.discard(self._order.popleft())
        return True

    def __len__(self) -> int:
        return len(self._members)


class HttpTransport(Transport):
    """Conditional GET against one URL.

    ETag and Last-Modified are honoured, so a feed that has not changed costs
    a 304 rather than a body. At a few hundred sources on a one-second cadence
    that is the difference between polite and blocked.
    """

    kind = "http"

    def __init__(self, source_id: str, url: str, client: HttpClient):
        super().__init__(source_id)
        self.url = url
        self.client = client
        self._etag = ""
        self._last_modified = ""

    async def body(self) -> Optional[str]:
        """The response body, or None when the endpoint says nothing changed."""
        headers: Dict[str, str] = {}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._last_modified:
            headers["If-Modified-Since"] = self._last_modified
        status, text, response_headers = await self.client.get(self.url, headers=headers)
        if status == 304:
            return None
        if status == 429:
            raise TransportError(f"rate limited by {urlparse(self.url).netloc}")
        if status >= 400:
            raise TransportError(f"HTTP {status} from {urlparse(self.url).netloc}")
        self._etag = response_headers.get("ETag", "") or self._etag
        self._last_modified = response_headers.get("Last-Modified", "") or self._last_modified
        return text

    async def json_body(self) -> Optional[Any]:
        text = await self.body()
        if text is None:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise TransportError(f"response was not JSON: {exc}") from exc


class RssTransport(HttpTransport):
    """RSS 2.0 and Atom, parsed with the standard library.

    One transport for the whole long tail -- government notices, regional
    press, corporate newsrooms, per-channel video feeds -- because the tail is
    where the lead usually is and it is all served by the same two formats.
    """

    kind = "rss"

    _ATOM = "{http://www.w3.org/2005/Atom}"

    def __init__(self, source_id: str, url: str, client: HttpClient, language: str = ""):
        super().__init__(source_id, url, client)
        self.language = language
        self.seen = SeenSet()

    async def fetch(self) -> List[Dict[str, Any]]:
        text = await self.body()
        if text is None:
            return []
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            raise TransportError(f"feed did not parse as XML: {exc}") from exc
        records = [self._rss_item(item) for item in root.iter("item")]
        records += [self._atom_entry(entry) for entry in root.iter(f"{self._ATOM}entry")]
        fresh = [record for record in records
                 if record and self.seen.add_if_new(record["_key"])]
        return fresh[:MAX_BATCH]

    def _rss_item(self, item) -> Optional[Dict[str, Any]]:
        def text_of(tag: str) -> str:
            found = item.find(tag)
            return (found.text or "") if found is not None else ""

        link = text_of("link")
        guid = text_of("guid") or link
        title = strip_html(text_of("title"))
        summary = strip_html(text_of("description"))
        if not (title or summary):
            return None
        return {"_key": guid or title,
                "title": title, "summary": summary, "link": link,
                "published_epoch": parse_timestamp(text_of("pubDate")),
                "language": self.language}

    def _atom_entry(self, entry) -> Optional[Dict[str, Any]]:
        def text_of(tag: str) -> str:
            found = entry.find(f"{self._ATOM}{tag}")
            return (found.text or "") if found is not None else ""

        link_element = entry.find(f"{self._ATOM}link")
        link = link_element.get("href", "") if link_element is not None else ""
        title = strip_html(text_of("title"))
        summary = strip_html(text_of("summary") or text_of("content"))
        if not (title or summary):
            return None
        return {"_key": text_of("id") or link or title,
                "title": title, "summary": summary, "link": link,
                "published_epoch": parse_timestamp(
                    text_of("published") or text_of("updated")),
                "language": self.language}


class YouTubeChannelTransport(RssTransport):
    """A channel's public feed, which needs no key and no WebSub receiver.

    WebSub is faster when an operator has a public callback to receive on.
    This is the transport for when they do not, and it is the difference
    between a declared YouTube source and a working one.
    """

    kind = "youtube_feed"

    URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    _MEDIA = "{http://search.yahoo.com/mrss/}"
    _YT = "{http://www.youtube.com/xml/schemas/2015}"

    def __init__(self, source_id: str, channel_id: str, client: HttpClient):
        super().__init__(source_id, self.URL.format(channel_id=quote(channel_id)), client)
        self.channel_id = channel_id

    async def fetch(self) -> List[Dict[str, Any]]:
        text = await self.body()
        if text is None:
            return []
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            raise TransportError(f"channel feed did not parse: {exc}") from exc
        records: List[Dict[str, Any]] = []
        for entry in root.iter(f"{self._ATOM}entry"):
            video = entry.find(f"{self._YT}videoId")
            video_id = (video.text or "") if video is not None else ""
            title_element = entry.find(f"{self._ATOM}title")
            title = strip_html((title_element.text or "") if title_element is not None else "")
            group = entry.find(f"{self._MEDIA}group")
            description = ""
            if group is not None:
                description_element = group.find(f"{self._MEDIA}description")
                description = strip_html(
                    (description_element.text or "") if description_element is not None else "")
            published = entry.find(f"{self._ATOM}published")
            if not self.seen.add_if_new(video_id or title):
                continue
            records.append({
                "title": title, "description": description, "video_id": video_id,
                "channel_id": self.channel_id,
                "published": parse_timestamp(
                    (published.text or "") if published is not None else "")})
        return records[:MAX_BATCH]


class MastodonTimelineTransport(HttpTransport):
    """A Mastodon instance's PUBLIC timeline.

    Public timelines are exactly that: served to anonymous readers by design.
    Nothing here reads a follower-only post, a direct message, or anything an
    account has not published openly.
    """

    kind = "mastodon"

    def __init__(self, source_id: str, instance: str, client: HttpClient, limit: int = 40):
        self.instance = instance.rstrip("/")
        self.limit = limit
        super().__init__(source_id, f"{self.instance}/api/v1/timelines/public?limit={limit}",
                         client)
        self._since_id = ""

    async def fetch(self) -> List[Dict[str, Any]]:
        url = f"{self.instance}/api/v1/timelines/public?limit={self.limit}"
        if self._since_id:
            url = f"{url}&since_id={quote(self._since_id)}"
        self.url = url
        payload = await self.json_body()
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise TransportError("public timeline did not return a list")
        records: List[Dict[str, Any]] = []
        for status in payload[:MAX_BATCH]:
            if not isinstance(status, dict):
                continue
            identifier = str(status.get("id") or "")
            if identifier > self._since_id:
                self._since_id = identifier
            records.append({
                "content": strip_html(status.get("content") or ""),
                "created_at_epoch": parse_timestamp(status.get("created_at")),
                "account": status.get("account") or {},
                "language": status.get("language") or ""})
        return records


class JsonPollTransport(HttpTransport):
    """Any JSON endpoint that serves a list of records.

    The generic case: a public hub, an open data API, an aggregator. The
    caller supplies the path to the list and the field that identifies a
    record, because guessing either produces silent duplicates or an empty
    feed that looks like a quiet source.
    """

    kind = "json"

    def __init__(self, source_id: str, url: str, client: HttpClient, *,
                 list_path: Sequence[str] = (), id_field: str = "id",
                 mapping: Optional[Dict[str, str]] = None):
        super().__init__(source_id, url, client)
        self.list_path = tuple(list_path)
        self.id_field = id_field
        self.mapping = dict(mapping or {})
        self.seen = SeenSet()

    async def fetch(self) -> List[Dict[str, Any]]:
        payload = await self.json_body()
        if payload is None:
            return []
        for key in self.list_path:
            if not isinstance(payload, dict):
                raise TransportError(f"payload has no path segment {key!r}")
            payload = payload.get(key)
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise TransportError("endpoint did not return a list of records")
        records: List[Dict[str, Any]] = []
        for entry in payload[:MAX_BATCH]:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get(self.id_field) or json.dumps(entry, sort_keys=True)[:200])
            if not self.seen.add_if_new(key):
                continue
            records.append({target: entry.get(source)
                            for target, source in self.mapping.items()} if self.mapping
                           else entry)
        return records


class GithubRepoTransport(HttpTransport):
    """Public commit activity on a public repository.

    Unauthenticated and rate limited, which is why the cadence for a code
    source belongs in the declaration rather than on the mesh's clock.
    """

    kind = "code_repo"

    def __init__(self, source_id: str, repo: str, client: HttpClient, branch: str = ""):
        url = f"https://api.github.com/repos/{repo}/commits"
        if branch:
            url = f"{url}?sha={quote(branch)}"
        super().__init__(source_id, url, client)
        self.repo = repo
        self.seen = SeenSet()

    async def fetch(self) -> List[Dict[str, Any]]:
        payload = await self.json_body()
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise TransportError("commits endpoint did not return a list")
        records: List[Dict[str, Any]] = []
        for entry in payload[:MAX_BATCH]:
            commit = (entry or {}).get("commit") or {}
            sha = str(entry.get("sha") or "")
            if not self.seen.add_if_new(sha):
                continue
            author = commit.get("author") or {}
            records.append({
                "message": commit.get("message") or "", "sha": sha, "repo": self.repo,
                "author": author.get("name") or "",
                "committed_at": parse_timestamp(author.get("date"))})
        return records


class OfficialSiteTransport(HttpTransport):
    """Change detection over one official page.

    An official domain is the strongest non-chain proof the authenticity
    resolver accepts, so this feeds Level B directly -- and it emits ONLY on
    change: a page that re-serves the same bytes is not news, and treating it
    as news is how a static site becomes the loudest source in the mesh.
    """

    kind = "official_site"

    def __init__(self, source_id: str, url: str, client: HttpClient, domain: str = ""):
        super().__init__(source_id, url, client)
        self.domain = domain or urlparse(url).netloc
        self._hash = ""

    async def fetch(self) -> List[Dict[str, Any]]:
        text = await self.body()
        if text is None:
            return []
        stripped = strip_html(text)
        digest = hashlib.sha256(stripped.encode("utf-8", "replace")).hexdigest()
        if digest == self._hash:
            return []
        first = not self._hash
        self._hash = digest
        if first:
            # The first read establishes the baseline. Emitting it would
            # report the entire existing page as a change that just happened.
            return []
        return [{"changed_text": stripped[:4_000], "changed_at": time.time(),
                 "path": urlparse(self.url).path, "content_hash": digest}]


class QueueTransport(Transport):
    """A transport fed by something else in the process.

    Push channels invert the direction: a WebSub callback, an EventSub
    webhook, a gateway client and the desk's own metadata resolver all PRODUCE
    records rather than answering a poll. They push here, and the mesh drains
    it on its own cadence -- which keeps one polling model across every source
    instead of two schedulers that can disagree about what has been seen.
    """

    kind = "queue"

    def __init__(self, source_id: str, capacity: int = STREAM_BUFFER):
        super().__init__(source_id)
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self.dropped = 0

    def push(self, record: Dict[str, Any]) -> None:
        if len(self.buffer) == self.buffer.maxlen:
            # Oldest out. On a burst the newest records are the ones a
            # latency-sensitive desk wants.
            self.dropped += 1
        self.buffer.append(dict(record))

    async def fetch(self) -> List[Dict[str, Any]]:
        drained: List[Dict[str, Any]] = []
        while self.buffer and len(drained) < MAX_BATCH:
            drained.append(self.buffer.popleft())
        return drained

    def report(self) -> Dict[str, Any]:
        return {**super().report(), "buffered": len(self.buffer), "dropped": self.dropped}


class WebSocketTransport(QueueTransport):
    """A push stream drained through the same polling contract.

    The connection runs in its own task and reconnects with backoff. It is a
    QueueTransport underneath because a socket that has been up for an hour
    and a socket that reconnected four seconds ago should not look different
    to the mesh's cadence -- but they do to its health surface, which is why
    the connection state is reported.
    """

    kind = "websocket"

    def __init__(self, source_id: str, url: str, *, capacity: int = STREAM_BUFFER):
        super().__init__(source_id, capacity)
        self.url = url
        self.connected = False
        self.connects = 0
        self.disconnects = 0
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    def on_open(self) -> Sequence[str]:
        """Frames to send once connected. Subscriptions go here."""
        return ()

    def on_message(self, payload: Any) -> Optional[Dict[str, Any]]:
        """One decoded frame to one record, or None to ignore it."""
        raise NotImplementedError

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop = False
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        import aiohttp

        backoff = 1.0
        while not self._stop:
            try:
                async with aiohttp.ClientSession(
                        headers={"User-Agent": USER_AGENT}) as session:
                    async with session.ws_connect(self.url, heartbeat=30) as socket:
                        self.connected = True
                        self.connects += 1
                        backoff = 1.0
                        for frame in self.on_open():
                            await socket.send_str(frame)
                        async for message in socket:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            try:
                                record = self.on_message(json.loads(message.data))
                            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                                continue
                            if record is not None:
                                self.push(record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                if self.connected:
                    self.disconnects += 1
                self.connected = False
            if self._stop:
                break
            await asyncio.sleep(backoff)
            # Capped, because a stream that has been down for an hour should
            # still be retried within the minute.
            backoff = min(backoff * 2, 60.0)

    def report(self) -> Dict[str, Any]:
        return {**super().report(), "connected": self.connected,
                "connects": self.connects, "disconnects": self.disconnects,
                "url_host": urlparse(self.url).netloc}


class BlueskyJetstreamTransport(WebSocketTransport):
    """Bluesky's public Jetstream firehose. No account, no key.

    The whole network's posts arrive; filtering is local, and deliberately not
    done here -- an adapter or transport that decides what matters silently
    becomes the model.
    """

    kind = "bluesky"

    DEFAULT_URL = ("wss://jetstream2.us-east.bsky.network/subscribe"
                   "?wantedCollections=app.bsky.feed.post")

    def __init__(self, source_id: str, url: str = "", capacity: int = STREAM_BUFFER):
        super().__init__(source_id, url or self.DEFAULT_URL, capacity=capacity)

    def on_message(self, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        commit = payload.get("commit") or {}
        if commit.get("operation") not in (None, "create"):
            return None
        if not ((commit.get("record") or {}).get("text")):
            return None
        return payload


class NostrRelayTransport(WebSocketTransport):
    """A Nostr relay, subscribed to public notes.

    Relays are open by protocol. ``authors`` narrows the subscription to
    specific public keys when the operator knows which ones matter; without it
    the relay serves its whole public stream.
    """

    kind = "nostr"

    def __init__(self, source_id: str, url: str, authors: Sequence[str] = (),
                 kinds: Sequence[int] = (1,), limit: int = 100,
                 capacity: int = STREAM_BUFFER):
        super().__init__(source_id, url, capacity=capacity)
        self.authors = tuple(authors)
        self.kinds = tuple(kinds)
        self.limit = limit

    def on_open(self) -> Sequence[str]:
        subscription: Dict[str, Any] = {"kinds": list(self.kinds), "limit": self.limit}
        if self.authors:
            subscription["authors"] = list(self.authors)
        return (json.dumps(["REQ", f"mesh-{self.source_id}", subscription]),)

    def on_message(self, payload: Any) -> Optional[Dict[str, Any]]:
        # Relay frames are arrays: ["EVENT", <sub_id>, {...}].
        if not isinstance(payload, list) or len(payload) < 3 or payload[0] != "EVENT":
            return None
        event = payload[2]
        return event if isinstance(event, dict) and event.get("content") else None


class TelegramChannelTransport(Transport):
    """A public Telegram channel over MTProto, with the operator's own API key.

    Public channels only, read with a credential the operator registered
    themselves. Telethon is imported lazily so a desk that declares no
    Telegram source does not need it installed.
    """

    kind = "telegram"

    #: The session the operator authorised with
    #: `python -m src.research.telegram_authorize`. Pointed at the same path
    #: that tool writes: a transport looking somewhere else finds no session,
    #: and Telethon's response to no session is to ask for a phone number.
    SESSION_PATH = "data/telegram/collector"

    def __init__(self, source_id: str, channel: str, *,
                 api_id_env: str = "TELEGRAM_API_ID",
                 api_hash_env: str = "TELEGRAM_API_HASH",
                 session_name: str = "",
                 capacity: int = STREAM_BUFFER):
        super().__init__(source_id)
        self.channel = channel
        self.api_id_env = api_id_env
        self.api_hash_env = api_hash_env
        self.session_name = session_name or self.SESSION_PATH
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self.client: Any = None
        self.connected = False
        self._owns_client = True
        self._event_handler: Any = None

    def attach_client(self, client: Any) -> None:
        """Use an already connected, authorised client owned by the desk.

        Telethon's SQLite session is single-writer.  Opening one client per
        channel against the same session file makes the clients contend on
        that database and can keep the entire source mesh in startup for
        minutes.  One client can subscribe to every public channel, so the
        desk shares its social-intelligence client with these transports.
        """
        self.client = client
        self._owns_client = False

    async def start(self) -> None:
        """Connect to an ALREADY AUTHORISED session. Never prompts.

        Telethon's `start()` falls back to asking for a phone number and a
        login code on stdin when it finds no session. Under systemd there is
        no stdin, so that is a unit that hangs at start or dies with an
        unhelpful EOF -- and on a desk it is a unit that appears to be
        starting for ever. So the session file is checked first and its
        absence is refused with the command that creates it.
        """
        try:
            from telethon import events
        except ImportError as exc:
            raise TransportError(f"Telethon is not installed: {exc}") from exc

        if self.client is None:
            api_id = os.getenv(self.api_id_env, "")
            api_hash = os.getenv(self.api_hash_env, "")
            if not api_id or not api_hash:
                # Names, never values.
                raise TransportError(
                    f"{self.api_id_env} and {self.api_hash_env} must both be set")
            session_file = f"{self.session_name}.session"
            if not os.path.exists(session_file):
                raise TransportError(
                    f"no authorised Telegram session at {session_file}; run "
                    "`.venv/bin/python -m src.research.telegram_authorize` once, "
                    "interactively, before starting the desk")
            try:
                from telethon import TelegramClient
            except ImportError as exc:
                raise TransportError(f"Telethon is not installed: {exc}") from exc
            self.client = TelegramClient(self.session_name, int(api_id), api_hash)
            await self.client.connect()
        elif not bool(self.client.is_connected()):
            raise TransportError("shared Telegram client is not connected")

        async def _handler(event):  # pragma: no cover - needs a live connection
            message = event.message
            self.buffer.append({
                "message": message.message or "", "id": message.id,
                "date": message.date.timestamp() if message.date else time.time(),
                "sender_id": str(getattr(message, "sender_id", "") or self.channel)})

        # Resolve now, while startup failures are captured and named. Passing
        # an unresolved string defers this lookup into Telethon's update
        # dispatcher; an invalid channel then becomes an unhandled task
        # exception on every incoming update instead of one failed source.
        lookup: Any = self.channel
        if str(lookup).lstrip("-").isdigit():
            lookup = int(lookup)
        try:
            target = await self.client.get_input_entity(lookup)
        except Exception as exc:
            raise TransportError(
                f"Telegram channel {self.channel!r} is unavailable: "
                f"{type(exc).__name__}: {exc}") from exc
        self._event_handler = _handler
        self.client.add_event_handler(
            self._event_handler, events.NewMessage(chats=target))

        # connect(), not start(): start() is the one that prompts. A session
        # that exists but is no longer authorised is refused here rather than
        # silently connecting as nobody.
        if not await self.client.is_user_authorized():
            if self._owns_client:
                await self.client.disconnect()
            raise TransportError(
                "the Telegram session is no longer authorised; "
                "re-run src.research.telegram_authorize")
        self.connected = True

    async def stop(self) -> None:
        if self.client is not None:
            if self._event_handler is not None:
                self.client.remove_event_handler(self._event_handler)
                self._event_handler = None
            if self._owns_client:
                await self.client.disconnect()
            self.connected = False

    async def fetch(self) -> List[Dict[str, Any]]:
        if not self.connected:
            raise TransportError("telegram client is not connected")
        drained: List[Dict[str, Any]] = []
        while self.buffer and len(drained) < MAX_BATCH:
            drained.append(self.buffer.popleft())
        return drained

    def report(self) -> Dict[str, Any]:
        return {**super().report(), "connected": self.connected,
                "channel": self.channel, "buffered": len(self.buffer),
                "shared_client": not self._owns_client}


@dataclass
class TransportReport:
    """Which declarations got a transport, and precisely why the rest did not."""

    built: int = 0
    declared: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    # (source_id, reason). Named rather than counted: a source with no
    # transport is a coverage hole, and a hole nobody can name does not get
    # filled.
    #
    # Split three ways, because the three are different problems with
    # different owners. A declared region whose endpoint nobody has chosen yet
    # is a research task; a source missing a key is an operator task; a kind
    # with no transport is ours. One undifferentiated "unbuilt" list makes all
    # three look like the same neglected pile.
    pending_endpoint: List[Tuple[str, str]] = field(default_factory=list)
    unconfigured: List[Tuple[str, str]] = field(default_factory=list)
    unsupported: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def unbuilt(self) -> List[Tuple[str, str]]:
        return [*self.pending_endpoint, *self.unconfigured, *self.unsupported]

    def to_dict(self) -> Dict[str, Any]:
        def rows(items: List[Tuple[str, str]]) -> List[Dict[str, str]]:
            return [{"source": source, "reason": reason} for source, reason in sorted(items)]

        return {
            "schema": TRANSPORTS_SCHEMA_VERSION,
            "declared": self.declared, "built": self.built,
            "built_share": (self.built / self.declared) if self.declared else None,
            "by_kind": dict(sorted(self.by_kind.items())),
            "pending_endpoint": rows(self.pending_endpoint),
            "unconfigured": rows(self.unconfigured),
            "unsupported": rows(self.unsupported),
            "unbuilt": rows(self.unbuilt),
        }


#: What each declaration kind needs in ``options`` before a transport can be
#: built for it. Stated as data so a missing option is reported by name rather
#: than discovered as a TypeError at construction.
REQUIRED_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "rss": ("url",),
    "official_site": ("url",),
    "mastodon": ("instance",),
    "nostr": ("relay",),
    "code_repo": ("repo",),
    "telegram": ("channel",),
    "youtube": ("channel_id",),
    "farcaster": ("hub_url",),
    "bluesky": (),
    "metadata": (),
    "twitch": (),
    "discord": (),
}


def build_transport(declaration: Any, client: HttpClient) -> Any:
    """One transport for one declaration, or raise with the reason.

    Push-only kinds (metadata, twitch, discord) get a QueueTransport: they are
    produced by something else in the process rather than polled, and the
    queue is what lets them share one cadence and one health surface with
    everything else.
    """
    kind = declaration.kind
    options = dict(declaration.options or {})
    missing = [name for name in REQUIRED_OPTIONS.get(kind, ()) if not options.get(name)]
    if missing:
        raise TransportError(f"declaration is missing options: {', '.join(missing)}")
    source_id = declaration.source_id

    if kind == "rss":
        return RssTransport(source_id, str(options["url"]), client,
                            language=declaration.language)
    if kind == "official_site":
        return OfficialSiteTransport(source_id, str(options["url"]), client,
                                     domain=str(options.get("domain", "")))
    if kind == "mastodon":
        return MastodonTimelineTransport(source_id, str(options["instance"]), client,
                                         limit=int(options.get("limit", 40)))
    if kind == "code_repo":
        return GithubRepoTransport(source_id, str(options["repo"]), client,
                                   branch=str(options.get("branch", "")))
    if kind == "youtube":
        return YouTubeChannelTransport(source_id, str(options["channel_id"]), client)
    if kind == "bluesky":
        return BlueskyJetstreamTransport(source_id, str(options.get("url", "")))
    if kind == "nostr":
        return NostrRelayTransport(
            source_id, str(options["relay"]),
            authors=tuple(options.get("authors", ()) or ()),
            kinds=tuple(int(value) for value in options.get("kinds", (1,))))
    if kind == "farcaster":
        return JsonPollTransport(
            source_id, str(options["hub_url"]), client,
            list_path=tuple(options.get("list_path", ("messages",))),
            id_field=str(options.get("id_field", "hash")))
    if kind == "telegram":
        return TelegramChannelTransport(source_id, str(options["channel"]))
    if kind in ("metadata", "twitch", "discord"):
        return QueueTransport(source_id)
    raise TransportError(f"no transport for kind {kind!r}")


def build_transports(declarations: Sequence[Any], client: Optional[HttpClient] = None,
                     ) -> Tuple[Dict[str, Any], TransportReport, HttpClient]:
    """Every transport the declarations can support, plus what could not be built.

    Credentials are checked by PRESENCE before a transport is built, so a
    source missing a key is reported by name instead of failing on its first
    poll -- and no value is read.
    """
    client = client or HttpClient()
    report = TransportReport(declared=len(declarations))
    transports: Dict[str, Any] = {}
    for declaration in declarations:
        missing = declaration.missing_credentials()
        if missing:
            report.unconfigured.append(
                (declaration.source_id, f"missing credentials: {', '.join(missing)}"))
            continue
        try:
            transport = build_transport(declaration, client)
        except TransportError as exc:
            reason = str(exc)
            bucket = (report.pending_endpoint if "missing options" in reason
                      else report.unsupported)
            bucket.append((declaration.source_id, reason))
            continue
        transports[declaration.source_id] = transport
        report.built += 1
        report.by_kind[transport.kind] = report.by_kind.get(transport.kind, 0) + 1
    return transports, report, client


async def start_transports(transports: Dict[str, Any], *,
                           timeout_s: float = 15.0,
                           concurrency: int = 16) -> Dict[str, str]:
    """Connect everything that holds a connection. Returns what failed, by name.

    Failures are returned rather than raised: one relay refusing a connection
    must not stop the other three hundred sources from starting.
    """
    failures: Dict[str, str] = {}
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def start_one(source_id: str, transport: Any) -> None:
        starter = getattr(transport, "start", None)
        if starter is None:
            return
        try:
            async with semaphore:
                await asyncio.wait_for(starter(), timeout=max(0.1, timeout_s))
        except Exception as exc:
            failures[source_id] = f"{type(exc).__name__}: {exc}"
            logger.warning("transport %s failed to start: %s", source_id, exc)
    await asyncio.gather(*(start_one(source_id, transport)
                           for source_id, transport in transports.items()))
    return failures


async def stop_transports(transports: Dict[str, Any], client: Optional[HttpClient] = None
                          ) -> None:
    async def stop_one(transport: Any) -> None:
        stopper = getattr(transport, "stop", None)
        if stopper is None:
            return
        try:
            await asyncio.wait_for(stopper(), timeout=10.0)
        except Exception as exc:
            logger.debug("transport %s did not stop cleanly: %s",
                         getattr(transport, "source_id", "?"), exc)
    await asyncio.gather(*(stop_one(transport) for transport in transports.values()))
    if client is not None:
        await client.close()


def transport_report(transports: Dict[str, Any]) -> Dict[str, Any]:
    """Per-transport counters, so a wired-but-silent source is visible."""
    rows = [transport.report() for transport in transports.values()]
    answering = sum(1 for row in rows if row.get("records", 0) > 0)
    return {
        "schema": TRANSPORTS_SCHEMA_VERSION,
        "status": "OK" if answering else "DATA_BLOCKED",
        "transports": len(rows), "answering": answering,
        "failing": sum(1 for row in rows if row.get("failures", 0) > 0),
        "by_kind": {kind: sum(1 for row in rows if row["kind"] == kind)
                    for kind in sorted({row["kind"] for row in rows})},
        "sources": sorted(rows, key=lambda row: row["source_id"]),
    }
