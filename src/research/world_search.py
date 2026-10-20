"""The backends the world crawler actually searches with.

`PublicInternetEventCrawler` generates queries and decides what to do with
results; it deliberately owns no transport. This module is the transport, and
it is built from endpoints that are keyless, public, and documented -- because
a discovery layer gated behind an API key is a discovery layer that stops the
day the key expires, and this desk has already been through that with
Cointelegraph's regional feeds.

Four backends, each answering a different question:

    HN ALGOLIA      developer and infrastructure chatter. Where a new
                    launchpad, bundler or bot framework surfaces before the
                    trading crowd hears of it. Keyless, generous, no quota.
    DEXSCREENER     the chain's own answer. A name searched here returns the
                    pairs actually carrying it, which is how a rumour picked
                    up on the web becomes a mint address.
    GITHUB          code. A new Pump IDL, a scraper, a copy-trading repo, a
                    fork of somebody's sniper. Keyless at a lower rate; the
                    token, if present, only raises the limit.
    REDDIT          public listings on public subreddits.

**Every backend fails alone.** One 403 must not end a cycle: the fan-out
gathers with exceptions captured, and a backend that raised contributes no
findings and one recorded failure. A cycle that lost GitHub still learned
whatever the other three knew.

**Nothing here authenticates.** Reddit's public JSON listings, HN's public
index, DexScreener's public search, GitHub's unauthenticated search. Where a
token exists it is read from the environment and used only to raise a rate
limit -- never to reach something the anonymous request could not.

**Result text is untrusted.** Titles and snippets come from strangers; they are
carried as data into `Finding`, hashed, and scored by forward outcome. Nothing
in this module interprets them as instructions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.research.world_crawler import Finding, Query

logger = logging.getLogger(__name__)

HN_SEARCH = ("https://hn.algolia.com/api/v1/search"
             "?query={query}&tags=(story,comment)&hitsPerPage=20")
DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search?q={query}"
GITHUB_SEARCH = ("https://api.github.com/search/repositories"
                 "?q={query}&sort=updated&order=desc&per_page=20")
REDDIT_SEARCH = ("https://www.reddit.com/search.json"
                 "?q={query}&sort=new&limit=25&raw_json=1")

#: Per-backend ceiling on findings kept from one query. A search that returns
#: a thousand results is a search that matched nothing in particular, and
#: absorbing all of it buries the specific hits in noise.
MAX_RESULTS_PER_BACKEND = 20

#: How long one backend gets before the cycle moves on without it.
BACKEND_TIMEOUT_S = 12.0


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


async def _get_json(client: Any, url: str,
                    headers: Optional[Dict[str, str]] = None) -> Any:
    status, body, _ = await client.get(url, headers=headers)
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from {url.split('?')[0]}")
    return json.loads(body)


async def hn_search(client: Any, query: Query) -> List[Finding]:
    """Hacker News, where infrastructure surfaces before it is a product."""
    payload = await _get_json(client, HN_SEARCH.format(
        query=urllib.parse.quote(query.text)))
    findings: List[Finding] = []
    for hit in (payload or {}).get("hits", [])[:MAX_RESULTS_PER_BACKEND]:
        if not isinstance(hit, dict):
            continue
        url = _text(hit.get("url")) or (
            f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}")
        findings.append(Finding(
            url=url,
            title=_text(hit.get("title")) or _text(hit.get("story_title")),
            snippet=_text(hit.get("comment_text"))
            or _text(hit.get("story_text")),
            published_at=hit.get("created_at_i"),
            query=query))
    return findings


async def dexscreener_search(client: Any, query: Query) -> List[Finding]:
    """The chain's answer to a name: which pairs actually carry it.

    This is the join that turns an internet-side rumour into a mint, which is
    the crawler's second direction and the reason this backend is here rather
    than only in the price miners.
    """
    payload = await _get_json(client, DEXSCREENER_SEARCH.format(
        query=urllib.parse.quote(query.text)))
    findings: List[Finding] = []
    for pair in (payload or {}).get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        base = pair.get("baseToken") or {}
        mint = _text(base.get("address"))
        if not mint:
            continue
        findings.append(Finding(
            url=_text(pair.get("url")) or f"https://dexscreener.com/{mint}",
            title=f"{_text(base.get('name'))} ({_text(base.get('symbol'))})",
            # The mint goes in the snippet so the crawler's own extractor
            # finds it by the same path it finds one quoted in prose -- one
            # code path, so one set of bugs.
            snippet=f"{mint} {_text(pair.get('dexId'))} "
                    f"{_text(pair.get('chainId'))}",
            published_at=(float(pair["pairCreatedAt"]) / 1000.0
                          if isinstance(pair.get("pairCreatedAt"),
                                        (int, float)) else None),
            query=query))
        if len(findings) >= MAX_RESULTS_PER_BACKEND:
            break
    return findings


async def github_search(client: Any, query: Query) -> List[Finding]:
    """Repositories. A token, if set, raises the limit and nothing else."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = await _get_json(client, GITHUB_SEARCH.format(
        query=urllib.parse.quote(query.text)), headers=headers)
    findings: List[Finding] = []
    for item in (payload or {}).get("items", [])[:MAX_RESULTS_PER_BACKEND]:
        if not isinstance(item, dict):
            continue
        findings.append(Finding(
            url=_text(item.get("html_url")),
            title=_text(item.get("full_name")),
            snippet=" ".join(filter(None, [
                _text(item.get("description")),
                _text(item.get("homepage"))])),
            query=query))
    return findings


async def reddit_search(client: Any, query: Query) -> List[Finding]:
    """Public listings only. No login, no restricted community."""
    payload = await _get_json(
        client, REDDIT_SEARCH.format(query=urllib.parse.quote(query.text)),
        headers={"User-Agent": "memecoin-research/1.0 (public listings)"})
    children = ((payload or {}).get("data") or {}).get("children") or []
    findings: List[Finding] = []
    for child in children[:MAX_RESULTS_PER_BACKEND]:
        data = (child or {}).get("data") or {}
        if not isinstance(data, dict):
            continue
        permalink = _text(data.get("permalink"))
        findings.append(Finding(
            url=f"https://www.reddit.com{permalink}" if permalink
            else _text(data.get("url")),
            title=_text(data.get("title")),
            snippet=" ".join(filter(None, [_text(data.get("selftext"))[:2000],
                                           _text(data.get("url"))])),
            published_at=data.get("created_utc"),
            query=query))
    return findings


DEFAULT_BACKENDS: Tuple[Tuple[str, Callable[..., Any]], ...] = (
    ("hn", hn_search),
    ("dexscreener", dexscreener_search),
    ("github", github_search),
    ("reddit", reddit_search),
)


class PublicWebSearcher:
    """Fan out one query across every backend, and survive any of them.

    Callable with a `Query`, returning `Finding` rows -- which is exactly the
    interface `PublicInternetEventCrawler.acycle` expects, so the crawler
    never learns that HTTP exists.
    """

    def __init__(self, client: Any, *,
                 backends: Sequence[Tuple[str, Callable[..., Any]]] = (),
                 timeout_s: float = BACKEND_TIMEOUT_S):
        self.client = client
        self.backends = tuple(backends or DEFAULT_BACKENDS)
        self.timeout_s = float(timeout_s)
        #: Per-backend health, so a backend that has quietly started refusing
        #: every request is visible in the health report rather than only in
        #: a shrinking finding count.
        self.stats: Dict[str, Dict[str, Any]] = {
            name: {"calls": 0, "findings": 0, "failures": 0, "last_error": ""}
            for name, _ in self.backends}

    async def __call__(self, query: Query) -> List[Finding]:
        async def _run(name: str, backend: Callable[..., Any]
                       ) -> List[Finding]:
            self.stats[name]["calls"] += 1
            try:
                results = await asyncio.wait_for(
                    backend(self.client, query), timeout=self.timeout_s)
            except Exception as exc:
                self.stats[name]["failures"] += 1
                self.stats[name]["last_error"] = f"{type(exc).__name__}: {exc}"
                logger.debug("world search backend %s DATA_BLOCKED: %s",
                             name, exc)
                return []
            rows = [row for row in (results or []) if row.url]
            self.stats[name]["findings"] += len(rows)
            return rows

        gathered = await asyncio.gather(
            *(_run(name, backend) for name, backend in self.backends))
        findings: List[Finding] = []
        seen: set = set()
        for rows in gathered:
            for row in rows:
                if row.url in seen:
                    continue
                seen.add(row.url)
                findings.append(row)
        return findings

    def report(self) -> Dict[str, Any]:
        healthy = [name for name, row in self.stats.items()
                   if row["calls"] and row["failures"] < row["calls"]]
        return {"backends": len(self.backends), "healthy": len(healthy),
                "detail": {name: dict(row)
                           for name, row in self.stats.items()}}
