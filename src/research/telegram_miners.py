"""Public Telegram, read the way a browser reads it -- and read at scale.

Telegram is where a contract address usually appears first, and the desk has
had a Telegram transport for a while. What it has not had is CHANNELS. Seven
declarations existed in ``config/sources.yaml`` naming seven languages and no
actual channel, because a made-up handle reads as a chosen one: the mesh
reports it UNCONFIGURED, an operator assumes it wants a key, and the coverage
number stays wrong in the flattering direction. That is a real problem and
guessing handles would not have solved it.

So this module reads Telegram two ways, and the cheap way is what fills the
expensive one.

**The preview reader.** ``https://t.me/s/<channel>`` is the page Telegram
serves to a browser with no account for any PUBLIC channel. No session, no API
key, no login. It is a few seconds behind MTProto and it is worth having
anyway, because it covers hundreds of channels at the cost of an HTTP GET
each, and because it works on a node whose Telegram session is not authorised
-- which is the state most desks are actually in most of the time.

**Discovery, then verification, then promotion.** Channels are not guessed.
They are harvested as ``t.me/...`` links out of text the desk has already
mined -- posts, repo READMEs, token metadata, other channels' forwards -- and
every candidate is then VERIFIED by fetching its own preview. A handle that
serves a preview with messages is real and public. A handle that 404s, or
serves a page with no messages, is neither, and it is recorded as rejected so
the same dead handle is not re-checked for ever. Only verified handles are
promoted into the mined set, and the promoted list is written to disk so the
work survives a restart.

The access boundary is in the mechanism, not in a policy note. This module
holds no credential and performs no login, so there is no configuration of it
that could open a private channel, a members-only group, or anything behind an
invite. If a channel is not readable by an anonymous browser, nothing here can
read it. Sketchy public channels, paid-promotion channels, claimed-insider
channels and pump groups are all fair game and are mined as SIGNALS -- what
they say is evidence about what a crowd is about to do, not advice -- and the
private ones stay shut.

One discipline throughout: a channel that returns no messages is reported as
silent, never as quiet. An unmeasured mention rate is not a low one, and
treating a parse failure as an absence of calls is how a desk concludes the
market has gone still while its scraper is broken.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.research.data_miners import (
    CADENCE_FAST, CADENCE_HOURLY, CADENCE_MINUTE, DataMinerPool, Enriches,
    MinerSpec, RateLimited,
)

logger = logging.getLogger(__name__)

TELEGRAM_MINER_SCHEMA_VERSION = "v1"

PREVIEW_URL = "https://t.me/s/{channel}"

#: A Solana address in text. Base58 excludes 0, O, I and l, which is most of
#: what stops this matching ordinary words; the length bound does the rest.
#: Deliberately permissive -- a false positive costs one metadata lookup that
#: returns nothing, a false negative costs the call.
CA_PATTERN = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")

#: A Telegram handle inside a link. Telegram's own rule: 5-32 characters,
#: letters, digits and underscore.
HANDLE_PATTERN = re.compile(
    r"(?:https?://)?t\.me/(?:s/)?(?:joinchat/)?([A-Za-z][A-Za-z0-9_]{4,31})\b")

#: Handles that are Telegram's own surfaces rather than channels. Fetching a
#: preview for these succeeds and returns nothing useful for ever.
RESERVED_HANDLES = frozenset({
    "telegram", "durov", "share", "addstickers", "joinchat", "proxy",
    "socks", "setlanguage", "addtheme", "iv", "s", "c", "contact",
    "telegramtips", "previews", "addlist", "boost", "login",
})

#: Words that make a public channel worth mining at all. A t.me link in a
#: README is usually a support channel; the ones that matter announce calls.
CALL_KEYWORDS = (
    "pump", "solana", "sol", "call", "gem", "ape", "snipe", "degen",
    "memecoin", "meme", "launch", "alpha", "signal", "moon", "trending",
    "insider", "whale", "dev", "presale", "listing",
)


def _strip_tags(fragment: str) -> str:
    """Text content of an HTML fragment, with <br> as a newline.

    Written against stdlib rather than a parser dependency because this runs
    on a node where adding a package means an operator step, and because the
    only structure that matters here is "text, and where the line breaks are".
    """
    class Collector(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: List[str] = []

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

        def handle_startendtag(self, tag: str, attrs) -> None:
            if tag == "br":
                self.parts.append("\n")

        def handle_starttag(self, tag: str, attrs) -> None:
            if tag == "br":
                self.parts.append("\n")

    collector = Collector()
    try:
        collector.feed(fragment)
        collector.close()
    except Exception:  # pragma: no cover - malformed fragment
        return re.sub(r"<[^>]+>", " ", fragment)
    return html.unescape("".join(collector.parts)).strip()


#: The preview page's own markup. Telegram has served these class names for
#: years, but a page that returns 200 and parses to zero messages is reported
#: as a PARSE failure rather than as an empty channel -- if Telegram renames a
#: class, the desk must hear "we cannot read this any more", not "nobody is
#: posting".
_MESSAGE_BLOCK = re.compile(
    r'data-post="(?P<channel>[^/"]+)/(?P<post>\d+)"(?P<body>.*?)'
    r'(?=data-post="|\Z)', re.DOTALL)
_TEXT_BLOCK = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(?P<text>.*?)</div>',
    re.DOTALL)
_TIME_BLOCK = re.compile(r'<time datetime="(?P<when>[^"]+)"')
_VIEWS_BLOCK = re.compile(
    r'<span class="tgme_widget_message_views">(?P<views>[^<]+)</span>')


def parse_preview(body: str, channel: str) -> List[Dict[str, Any]]:
    """Messages out of a t.me/s preview page.

    Views are parsed because they are the closest thing to a MEASURED crowd
    reading Telegram exposes anonymously: a call posted to a channel nobody
    reads and a call posted to one with forty thousand readers are different
    events, and a message count cannot tell them apart.
    """
    records: List[Dict[str, Any]] = []
    for match in _MESSAGE_BLOCK.finditer(body):
        block = match.group("body")
        text_match = _TEXT_BLOCK.search(block)
        text = _strip_tags(text_match.group("text")) if text_match else ""
        if not text:
            continue
        time_match = _TIME_BLOCK.search(block)
        views_match = _VIEWS_BLOCK.search(block)
        records.append({
            "channel": match.group("channel") or channel,
            "message_id": int(match.group("post")),
            "text": text[:4000],
            "posted_at": time_match.group("when") if time_match else "",
            "views": _parse_views(views_match.group("views")) if views_match else None,
            "mints": extract_mints(text),
            "handles": extract_handles(text),
        })
    return records


def _parse_views(raw: str) -> Optional[float]:
    """`12.4K` and `1.1M` are how the page writes a view count."""
    token = (raw or "").strip().upper().replace(" ", "")
    multiplier = 1.0
    if token.endswith("K"):
        multiplier, token = 1_000.0, token[:-1]
    elif token.endswith("M"):
        multiplier, token = 1_000_000.0, token[:-1]
    try:
        return float(token) * multiplier
    except ValueError:
        return None


def extract_mints(text: str) -> List[str]:
    """Candidate Solana addresses in a message, deduplicated, order kept."""
    seen: Set[str] = set()
    out: List[str] = []
    for candidate in CA_PATTERN.findall(text or ""):
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out[:12]


def extract_handles(text: str) -> List[str]:
    """Public channel handles linked from a message. The discovery input."""
    seen: Set[str] = set()
    out: List[str] = []
    for candidate in HANDLE_PATTERN.findall(text or ""):
        lowered = candidate.lower()
        if lowered in RESERVED_HANDLES or lowered in seen:
            continue
        seen.add(lowered)
        out.append(candidate)
    return out[:20]


@dataclass
class ChannelRecord:
    """One public channel and what reading it has actually produced."""

    handle: str
    state: str = "CANDIDATE"          # CANDIDATE | VERIFIED | REJECTED
    discovered_from: str = ""
    first_seen: float = 0.0
    verified_at: float = 0.0
    checks: int = 0
    messages: int = 0
    mints_seen: int = 0
    last_message_id: int = 0
    last_ok_at: float = 0.0
    last_error: str = ""
    #: How many independent places linked this channel. Discovery on one
    #: mention is how a mesh fills with noise that then has to be scored, and
    #: scoring costs more than the channel was ever worth.
    mentions: int = 0

    def to_dict(self, now: float) -> Dict[str, Any]:
        return {
            "handle": self.handle, "state": self.state,
            "mentions": self.mentions, "checks": self.checks,
            "messages": self.messages, "mints_seen": self.mints_seen,
            "discovered_from": self.discovered_from,
            "seconds_since_ok": (round(now - self.last_ok_at, 1)
                                 if self.last_ok_at else None),
            "last_error": self.last_error,
        }


class ChannelBook:
    """Which public channels we mine, how each got there, and which are dead.

    Persisted, because discovery is slow and a restart that forgets four
    hundred verified handles has thrown away days of it. Bounded, because an
    unbounded candidate set is a slow memory leak fed by every link in every
    message the desk has ever read.
    """

    def __init__(self, path: str = "data/telegram/channels.json", *,
                 max_verified: int = 600, max_candidates: int = 4_000,
                 promote_after_mentions: int = 1):
        self.path = path
        self.max_verified = int(max_verified)
        self.max_candidates = int(max_candidates)
        self.promote_after_mentions = max(1, int(promote_after_mentions))
        self.channels: Dict[str, ChannelRecord] = {}
        self._cursor = 0
        self.load()

    # --- membership ------------------------------------------------------

    def seed(self, handles: Iterable[str], source: str = "config") -> int:
        """Declare handles an operator chose. Still verified before use."""
        added = 0
        for handle in handles:
            if self.observe(handle, source):
                added += 1
        return added

    def observe(self, handle: str, source: str = "", now: Optional[float] = None) -> bool:
        """Record that something linked this handle. Returns True if new."""
        moment = time.time() if now is None else now
        key = (handle or "").strip().lstrip("@")
        if not key or key.lower() in RESERVED_HANDLES:
            return False
        if not HANDLE_PATTERN.fullmatch(f"t.me/{key}"):
            return False
        existing = self.channels.get(key.lower())
        if existing is not None:
            existing.mentions += 1
            return False
        if self._count("CANDIDATE") >= self.max_candidates:
            return False
        self.channels[key.lower()] = ChannelRecord(
            handle=key, discovered_from=source, first_seen=moment, mentions=1)
        return True

    def harvest(self, text: str, source: str = "") -> List[str]:
        """Every handle linked from a blob of text, added as candidates."""
        found = []
        for handle in extract_handles(text):
            if self.observe(handle, source):
                found.append(handle)
        return found

    # --- verification ----------------------------------------------------

    def mark_verified(self, handle: str, messages: int,
                      now: Optional[float] = None) -> None:
        moment = time.time() if now is None else now
        record = self.channels.get(handle.lower())
        if record is None:
            return
        record.state = "VERIFIED"
        record.verified_at = moment
        record.last_ok_at = moment
        record.checks += 1
        record.messages += messages
        record.last_error = ""
        self._evict_verified()

    def mark_rejected(self, handle: str, reason: str,
                      now: Optional[float] = None) -> None:
        """A handle that does not serve a public preview. Kept, not deleted.

        Deleting it means the next message linking it re-queues the same dead
        handle, for ever. Remembering the rejection is what makes discovery
        converge instead of cycling.
        """
        record = self.channels.get(handle.lower())
        if record is None:
            return
        record.state = "REJECTED"
        record.checks += 1
        record.last_error = reason[:200]

    def note_read(self, handle: str, messages: int, mints: int,
                  last_message_id: int, now: Optional[float] = None) -> None:
        moment = time.time() if now is None else now
        record = self.channels.get(handle.lower())
        if record is None:
            return
        record.checks += 1
        record.messages += messages
        record.mints_seen += mints
        record.last_message_id = max(record.last_message_id, int(last_message_id or 0))
        record.last_ok_at = moment
        record.last_error = ""

    def note_failure(self, handle: str, reason: str) -> None:
        record = self.channels.get(handle.lower())
        if record is None:
            return
        record.checks += 1
        record.last_error = reason[:200]

    # --- selection -------------------------------------------------------

    def verified(self) -> List[str]:
        return [record.handle for record in self.channels.values()
                if record.state == "VERIFIED"]

    def pending(self) -> List[str]:
        """Candidates with enough independent mentions to be worth a check."""
        return [record.handle for record in self.channels.values()
                if record.state == "CANDIDATE"
                and record.mentions >= self.promote_after_mentions]

    def next_batch(self, size: int) -> List[str]:
        """Verified handles, rotated, so no single channel is hammered."""
        handles = sorted(self.verified())
        if not handles:
            return []
        chosen = [handles[(self._cursor + offset) % len(handles)]
                  for offset in range(min(size, len(handles)))]
        self._cursor = (self._cursor + len(chosen)) % len(handles)
        return chosen

    def _count(self, state: str) -> int:
        return sum(1 for record in self.channels.values() if record.state == state)

    def _evict_verified(self) -> None:
        """Keep the channels that produce addresses, drop the ones that do not.

        Ranked by mints seen rather than by message count on purpose: a busy
        channel full of chat is worth less than a quiet one that only posts
        calls, and message volume would rank them the other way round.
        """
        verified = [record for record in self.channels.values()
                    if record.state == "VERIFIED"]
        if len(verified) <= self.max_verified:
            return
        verified.sort(key=lambda item: (item.mints_seen, item.messages), reverse=True)
        for record in verified[self.max_verified:]:
            record.state = "CANDIDATE"

    # --- persistence -----------------------------------------------------

    def load(self) -> int:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return 0
        for row in payload.get("channels") or []:
            try:
                record = ChannelRecord(**row)
            except TypeError:
                continue
            self.channels[record.handle.lower()] = record
        return len(self.channels)

    def save(self) -> bool:
        directory = os.path.dirname(self.path)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump({"schema": TELEGRAM_MINER_SCHEMA_VERSION,
                           "saved_at": time.time(),
                           "channels": [vars(record) for record
                                        in self.channels.values()]}, handle)
            return True
        except OSError as exc:
            logger.warning("channel book save failed: %s", exc)
            return False

    # --- reporting -------------------------------------------------------

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        """What is being read, what is queued, and what is silent.

        `silent` is the line that matters: a verified channel that has stopped
        answering is a coverage hole, and it looks exactly like a quiet market
        from inside a decision.
        """
        moment = time.time() if now is None else now
        rows = [record for record in self.channels.values()]
        verified = [row for row in rows if row.state == "VERIFIED"]
        silent = [row.handle for row in verified
                  if row.last_ok_at and moment - row.last_ok_at > 3_600]
        never = [row.handle for row in verified if not row.last_ok_at]
        producing = [row for row in verified if row.mints_seen > 0]
        return {
            "schema": TELEGRAM_MINER_SCHEMA_VERSION,
            "status": ("DATA_BLOCKED" if not verified
                       else "DEGRADED" if silent or never else "OK"),
            "detail": ("no verified public channel yet; discovery has not "
                       "converged or every candidate was rejected"
                       if not verified else
                       f"{len(silent)} verified channel(s) silent for over an hour"
                       if silent else ""),
            "verified": len(verified),
            "candidates": self._count("CANDIDATE"),
            "rejected": self._count("REJECTED"),
            "producing_addresses": len(producing),
            "messages": sum(row.messages for row in rows),
            "mints_seen": sum(row.mints_seen for row in rows),
            "silent": sorted(silent)[:40],
            "never_read": sorted(never)[:40],
            "top": [row.to_dict(moment) for row in
                    sorted(producing, key=lambda item: item.mints_seen,
                           reverse=True)[:25]],
        }


async def _get_preview(client: Any, handle: str) -> str:
    status, body, _headers = await client.get(PREVIEW_URL.format(channel=handle))
    if status == 429:
        raise RateLimited(f"t.me/s/{handle}")
    if status == 404:
        raise LookupError(f"t.me/s/{handle} does not exist or is not public")
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from t.me/s/{handle}")
    return body


def preview_miner(client: Any, book: ChannelBook,
                  on_message: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                  *, per_pass: int = 12,
                  ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Read a rotating batch of verified public channels.

    Every message is scanned for addresses AND for further handles, so reading
    channels is itself the discovery mechanism: the set grows toward whatever
    the crowd is actually linking rather than toward whatever was configured
    a month ago.
    """
    async def fetch() -> List[Dict[str, Any]]:
        handles = book.next_batch(per_pass)
        if not handles:
            return []
        records: List[Dict[str, Any]] = []
        for handle in handles:
            try:
                body = await _get_preview(client, handle)
            except RateLimited:
                raise
            except LookupError as exc:
                book.mark_rejected(handle, str(exc))
                continue
            except Exception as exc:
                book.note_failure(handle, f"{type(exc).__name__}: {exc}")
                continue
            messages = parse_preview(body, handle)
            if not messages and "tgme_widget_message" not in body:
                # 200 with no recognisable message markup. That is our parser
                # or their markup, not an empty channel, and calling it an
                # empty channel is how a broken scraper reads as a calm market.
                book.note_failure(
                    handle, "preview served but no message markup found; "
                            "parser or page layout changed")
                continue
            fresh = [row for row in messages
                     if row["message_id"] > book.channels[handle.lower()].last_message_id]
            mints = sum(len(row["mints"]) for row in fresh)
            book.note_read(handle, len(fresh), mints,
                           max((row["message_id"] for row in messages), default=0))
            for row in fresh:
                for linked in row["handles"]:
                    book.observe(linked, f"telegram:{handle}")
            records.extend({**row, "data_status": "OK"} for row in fresh)
        if records and on_message is not None:
            try:
                on_message(records)
            except Exception as exc:
                logger.warning("telegram consumer raised: %s", exc)
        return records

    return fetch


def verification_miner(client: Any, book: ChannelBook, *, per_pass: int = 6,
                       ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Check candidate handles and promote the ones that are real and public.

    Separated from reading on purpose and run on a slower clock. Verification
    is speculative work against handles that are mostly junk, and letting it
    share a budget with reading means a burst of links in one message delays
    every call the desk was actually there for.
    """
    async def fetch() -> List[Dict[str, Any]]:
        pending = book.pending()[:per_pass]
        if not pending:
            return []
        results: List[Dict[str, Any]] = []
        for handle in pending:
            try:
                body = await _get_preview(client, handle)
            except RateLimited:
                raise
            except LookupError as exc:
                book.mark_rejected(handle, str(exc))
                results.append({"handle": handle, "verified": False,
                                "reason": "not public", "data_status": "OK"})
                continue
            except Exception as exc:
                book.note_failure(handle, f"{type(exc).__name__}: {exc}")
                continue
            messages = parse_preview(body, handle)
            if not messages:
                book.mark_rejected(handle, "public page with no readable messages")
                results.append({"handle": handle, "verified": False,
                                "reason": "no messages", "data_status": "OK"})
                continue
            book.mark_verified(handle, len(messages))
            results.append({"handle": handle, "verified": True,
                            "messages": len(messages),
                            "mints": sum(len(extract_mints(row["text"]))
                                         for row in messages),
                            "data_status": "OK"})
        book.save()
        return results

    return fetch


def register_telegram_miners(pool: DataMinerPool, *, http: Any, book: ChannelBook,
                             on_message: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                             ) -> Dict[str, bool]:
    """Declare the public-Telegram set.

    Reading runs on the fast clock because a call is worth seconds, not
    minutes. Verification runs hourly because a handle that is public now will
    still be public in an hour, and spending the fast budget on speculative
    checks is spending it on the wrong thing.
    """
    registrations = (
        (MinerSpec(
            miner_id="telegram:public_preview", enriches=Enriches.SOCIAL_ATTENTION,
            cadence_seconds=CADENCE_FAST, endpoint=PREVIEW_URL,
            max_records=800,
            detail="public channel previews; no account, no session, no key"),
         preview_miner(http, book, on_message)),
        (MinerSpec(
            miner_id="telegram:verify_channels", enriches=Enriches.SOCIAL_ATTENTION,
            cadence_seconds=CADENCE_HOURLY, endpoint=PREVIEW_URL,
            detail="promote discovered handles that serve a public preview"),
         verification_miner(http, book)),
    )
    return {spec.miner_id: pool.register(spec, fetch)
            for spec, fetch in registrations}
