"""Bounded public onion research through SOCKS5 with remote name resolution.

Sources must be explicitly declared. Discovered links are candidates, never
permission to fetch a new host. Content has WEB provenance, not authenticity
or execution authority. No browser, script execution, login or download path.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import copy
import json
import os
import re
from pathlib import Path
import tempfile
import time
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

from src.collectors.transports import HttpClient, Transport, TransportError

MAX_BODY = 512 * 1024
PROXY_ENV = "MEMECOIN_TOR_PROXY_URL"


def onion_host(host: str) -> bool:
    """Validate a v3 address, including its version and checksum (Tor spec)."""
    host = host.lower()
    if not host.endswith(".onion") or len(host) != 62:
        return False
    label = host[:-6]
    try:
        raw = base64.b32decode(label.upper())
    except (ValueError, base64.binascii.Error):
        return False
    return (len(raw) == 35 and raw[-1] == 3 and
            raw[32:34] == hashlib.sha3_256(b".onion checksum" + raw[:32] + raw[-1:]).digest()[:2])


def onion_url(url: str, allowed_hosts=None) -> str:
    if not isinstance(url, str) or any(ord(c) <= 32 for c in url) or "\\" in url:
        raise TransportError("invalid public onion URL")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if (parts.scheme not in {"http", "https"} or not onion_host(host)
                or parts.username is not None or parts.password is not None
                or parts.port not in {None, 80, 443}
                or (allowed_hosts is not None and host not in allowed_hosts)):
            raise ValueError
    except ValueError:
        raise TransportError("valid declared v3 onion host and HTTP(S) URL required") from None
    # Query strings may carry cursors. Fragments never affect a server request.
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


class TorHttpClient(HttpClient):
    """A separate session: onion requests can never fall back to TCP/DNS."""

    def __init__(self, proxy_url: str, host: str, *, timeout_s=20., max_bytes=MAX_BODY):
        super().__init__(timeout_s=timeout_s, limit=2, limit_per_host=1)
        try:
            proxy = urlsplit(proxy_url)
            if (proxy.scheme != "socks5" or not proxy.hostname or not proxy.port
                    or proxy.username is not None or proxy.password is not None
                    or proxy.path or proxy.query or proxy.fragment):
                raise ValueError
        except ValueError:
            raise TransportError("Tor requires a credential-free socks5 proxy endpoint") from None
        if not onion_host(host):
            raise TransportError("invalid declared onion host")
        self.proxy_host, self.proxy_port = proxy.hostname, proxy.port
        self.allowed_hosts = {host}
        self.max_bytes = max(1024, min(MAX_BODY, int(max_bytes)))

    async def session(self):
        loop = asyncio.get_running_loop()
        existing = self._sessions.get(loop)
        if existing is not None and not existing.closed:
            return existing
        import aiohttp
        try:
            from aiohttp_socks import ProxyConnector, ProxyType
        except ImportError:
            raise TransportError("Tor connector unavailable; install aiohttp-socks") from None
        connector = ProxyConnector(proxy_type=ProxyType.SOCKS5, host=self.proxy_host,
            port=self.proxy_port, rdns=True, limit=2, limit_per_host=1)
        created = aiohttp.ClientSession(connector=connector, trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            headers={"User-Agent": "memecoin-onion-research/1.0"})
        self._sessions[loop] = created
        return created

    async def get(self, url, *, headers=None):
        url = onion_url(url, self.allowed_hosts)
        # Automatic redirects could leave the declared host, even before any
        # response is inspected. Validate every hop before opening it.
        try:
            async with asyncio.timeout(self.timeout_s):
                session = await self.session()
                for hop in range(4):
                    async with session.get(url, headers=headers or {}, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location or hop == 3:
                                raise TransportError("onion redirect missing or hop budget exhausted")
                            url = onion_url(urljoin(url, location), self.allowed_hosts)
                            continue
                        mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                        if response.status != 304 and mime not in {"text/html", "text/plain", "application/xhtml+xml"}:
                            raise TransportError("onion response is not a supported text document")
                        body = bytearray()
                        async for chunk in response.content.iter_chunked(8192):
                            body.extend(chunk)
                            if len(body) > self.max_bytes:
                                raise TransportError("onion document exceeds byte budget")
                        return response.status, body.decode("utf-8", "replace"), dict(response.headers)
        except TransportError:
            raise
        except Exception as exc:
            # Proxy URLs, headers and server bodies never enter error logs.
            raise TransportError("Tor fetch failed: " + type(exc).__name__) from None


class _Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.links, self.hidden = [], [], []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden.append(tag)
        if not self.hidden and tag == "a" and len(self.links) < 128:
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if self.hidden and self.hidden[-1] == tag:
            self.hidden.pop()

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


class OnionResearchTransport(Transport):
    kind = "onion"

    def __init__(self, source_id, url, *, source_ref, client=None,
                 max_pages=32, pages_per_poll=1, max_depth=3, state_path=None):
        super().__init__(source_id)
        self.url = onion_url(url)
        self.host = urlsplit(self.url).hostname
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise TransportError("public onion source provenance required")
        self.source_ref = source_ref
        self.max_pages = max(1, min(256, int(max_pages)))
        self.pages_per_poll = max(1, min(2, int(pages_per_poll)))
        self.max_depth = max(0, min(8, int(max_depth)))
        if client is None:
            proxy = os.environ.get(PROXY_ENV, "")
            if not proxy:
                raise TransportError("missing Tor proxy configuration: " + PROXY_ENV)
            client = TorHttpClient(proxy, self.host)
        self.client = client
        self.pages = {self.url: 0}
        self.queue = deque([self.url])
        self.hashes = {}
        self.last_observed_at = None
        self.candidate_links = set()
        self.state_path = Path(state_path) if state_path else None
        self.pending = []
        if self.state_path and self.state_path.exists():
            self._load()

    def _load(self):
        try:
            if self.state_path.stat().st_size > 2_000_000:
                raise ValueError("checkpoint exceeds byte budget")
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (saved.get("schema_version") != 1 or saved["source_id"] != self.source_id
                    or saved["url"] != self.url or saved["source_ref"] != self.source_ref):
                raise ValueError("checkpoint source binding mismatch")
            pages = saved["pages"]
            if not isinstance(pages, dict) or not 1 <= len(pages) <= self.max_pages:
                raise ValueError("checkpoint page budget mismatch")
            for url, depth in pages.items():
                onion_url(url, {self.host})
                if type(depth) is not int or not 0 <= depth <= self.max_depth:
                    raise ValueError("checkpoint crawl depth mismatch")
            queue = saved["queue"]
            if not isinstance(queue, list) or set(queue) != set(pages) or len(queue) != len(pages):
                raise ValueError("checkpoint crawl queue mismatch")
            pending = saved.get("pending", [])
            if not isinstance(pending, list) or len(pending) > 2:
                raise ValueError("checkpoint outbox exceeds budget")
            for record in pending:
                onion_url(record["url"], {self.host})
                if (not isinstance(record["text"], str) or len(record["text"]) > 16000
                        or record["source_ref"] != self.source_ref
                        or type(record["observed_at"]) not in {int, float}
                        or not 0 < record["observed_at"] <= time.time()
                        or not isinstance(record["observation_id"], str)):
                    raise ValueError("invalid checkpoint observation")
            hashes = saved["hashes"]
            if (not isinstance(hashes, dict) or not set(hashes) <= set(pages)
                    or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in hashes.values())):
                raise ValueError("invalid checkpoint content hashes")
            observed = saved.get("last_observed_at")
            if observed is not None and (type(observed) not in {int, float} or not 0 < observed <= time.time()):
                raise ValueError("invalid checkpoint observation time")
            self.pages, self.queue = pages, deque(queue)
            self.hashes, self.pending = saved["hashes"], pending
            self.last_observed_at = saved.get("last_observed_at")
            self.candidate_links = set(saved.get("candidate_links", [])[:128])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise TransportError("onion checkpoint invalid: " + type(exc).__name__) from None

    def _save(self):
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "source_id": self.source_id, "url": self.url, "source_ref": self.source_ref,
            "pages": self.pages, "queue": list(self.queue), "hashes": self.hashes,
            "pending": self.pending, "last_observed_at": self.last_observed_at,
            "candidate_links": sorted(self.candidate_links)}
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.state_path.parent,
                                             prefix=".onion-", delete=False) as handle:
                name = handle.name
                handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.state_path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    def acknowledge(self, observation_id, canonical_job_id):
        if not canonical_job_id:
            raise TransportError("canonical research queue acknowledgement required")
        previous = self.pending
        self.pending = [row for row in previous if row["observation_id"] != observation_id]
        try:
            self._save()
        except Exception:
            self.pending = previous
            raise

    async def fetch(self):
        if self.pending:
            return copy.deepcopy(self.pending)
        before = (dict(self.pages), deque(self.queue), dict(self.hashes),
                  set(self.candidate_links), self.last_observed_at)
        try:
            records = await self._fetch_pages()
            if self.state_path:
                self.pending = records
                self._save()
            return copy.deepcopy(records)
        except BaseException:
            # A failed second page must not consume the first page's change
            # before the caller has received any records from the batch.
            (self.pages, self.queue, self.hashes,
             self.candidate_links, self.last_observed_at) = before
            self.pending = []
            raise

    async def _fetch_pages(self):
        records = []
        for _ in range(min(self.pages_per_poll, len(self.queue))):
            url = self.queue[0]
            status, text, _headers = await self.client.get(url)
            if status != 200:
                raise TransportError("onion HTTP status " + str(status))
            doc = _Document()
            doc.feed(text)
            content = " ".join(" ".join(doc.text).split())
            if not content or (len(content) < 2000 and re.search(
                    r"(?:complete|solve|enter|verify).{0,40}captcha|proof.of.work required|"
                    r"(?:enable|requires?) javascript|sign in to (?:continue|view)", content, re.I)):
                raise TransportError("onion public content unavailable or access challenge present")
            digest = hashlib.sha256(content.encode()).hexdigest()
            observed = time.time()
            links = []
            for href in doc.links:
                try:
                    target = onion_url(urljoin(url, href))
                except TransportError:
                    continue
                links.append(target)
                if urlsplit(target).hostname != self.host:
                    if len(self.candidate_links) < 128:
                        self.candidate_links.add(target)
                elif (target not in self.pages and len(self.pages) < self.max_pages
                        and self.pages[url] < self.max_depth):
                    self.pages[target] = self.pages[url] + 1
                    self.queue.append(target)
            first = url not in self.hashes
            if digest != self.hashes.get(url) and content:
                records.append({"text": content[:16000], "url": url,
                    "observed_at": observed, "published_at": None,
                    "content_hash": digest, "source_ref": self.source_ref,
                    "observation_id": hashlib.sha256(json.dumps([url, digest, observed]).encode()).hexdigest(),
                    "text_truncated": len(content) > 16000,
                    "discovered_onion_links": links[:128],
                    "observation_kind": "initial_snapshot" if first else "content_change",
                    "transport": "tor_socks5_remote_dns"})
            self.hashes[url] = digest
            self.last_observed_at = observed
            self.queue.rotate(-1)
        return records

    async def stop(self):
        await self.client.close()

    def report(self):
        return {**super().report(), "scope": "PUBLIC_ONION_RESEARCH_ONLY",
            "execution_authority": False, "hostname_resolution": "SOCKS5_REMOTE",
            "declared_host": self.host, "pages": len(self.pages),
            "pending_canonical_delivery": len(self.pending),
            "last_observed_at": self.last_observed_at,
            "candidate_links": sorted(self.candidate_links),
            "status": "OBSERVED" if self.last_observed_at is not None else "NOT_OBSERVED"}
