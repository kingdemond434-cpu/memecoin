"""Public-web miners: measured attention, and the corpus that gives it meaning.

The source mesh answers "did anyone mention this token". These answer the two
harder questions around that:

  HOW MANY PEOPLE WENT LOOKING.   A mention is a touch. Attention is traffic.
                                  The ignition model needs the second to tell
                                  an originator's post that nobody read from
                                  one that started a wave, and a touch count
                                  cannot distinguish them.

  WHAT ELSE IS CALLED THIS.       Copycat detection needs a corpus. A ticker
                                  that is unique today and one that is the
                                  ninth reuse of a name that rugged twice are
                                  different bets, and only a search across the
                                  wider venue set can tell them apart.

Everything here reads endpoints that are public and documented, without an
account or with an account of our own. Nothing here reaches into a private
group, a members-only channel, or anything behind an access control. Sketchy
PUBLIC channels are fair game and are mined as signals; private ones are not
touched, and that boundary is in the code rather than in a policy document:
there is no credential here that could open one.

Two of these need a key we already hold (YouTube, GitHub). The rest need
none. A keyless source may still refuse a datacentre address -- Reddit in
particular does, intermittently -- and when it does the pool reports that
miner as silent rather than reporting a zero. An unmeasured attention level is
not a low one.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from src.research.data_miners import (
    CADENCE_DAILY, CADENCE_HOURLY, CADENCE_MINUTE, CADENCE_QUARTER,
    DataMinerPool, Enriches, MinerSpec, RateLimited,
)

logger = logging.getLogger(__name__)

REDDIT_NEW_URL = "https://www.reddit.com/r/{sub}/new.json?limit=50"
HN_SEARCH_URL = ("https://hn.algolia.com/api/v1/search_by_date"
                 "?query={query}&tags=story&hitsPerPage=50")
WIKI_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")
COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?q={query}"
YOUTUBE_SEARCH_URL = (
    "https://www.googleapis.com/youtube/v3/search?part=snippet&type=video"
    "&order=date&maxResults=25&q={query}&publishedAfter={after}&key={key}")
GITHUB_SEARCH_URL = ("https://api.github.com/search/repositories"
                     "?q={query}&sort=updated&order=desc&per_page=30")

#: Public subreddits where Solana launches are discussed. Public listings
#: only; nothing here reads a private or restricted community.
DEFAULT_SUBREDDITS = (
    "solana", "SolanaMemeCoins", "CryptoMoonShots", "memecoins",
    "SatoshiStreetBets", "CryptoCurrency", "altcoin", "defi",
)

#: Wikipedia articles whose traffic proxies broad retail attention on the
#: theme rather than on any one coin. Daily granularity, which is all the
#: endpoint offers and all this is good for.
DEFAULT_ATTENTION_ARTICLES = ("Solana_(blockchain_platform)", "Meme_coin",
                              "Cryptocurrency", "Dogecoin")

#: Searches that surface tooling and infrastructure changes before they are
#: announced. A new Pump program IDL landing in a public repo is a schema
#: change we would otherwise discover by decoding failures in production.
DEFAULT_REPO_QUERIES = ("pump.fun+solana", "pumpswap", "solana+sniper+bot")


async def _get_json(client: Any, url: str,
                    headers: Optional[Dict[str, str]] = None) -> Any:
    """One JSON fetch, with a rate limit distinguished from a failure.

    A 429 means the miner works and the source wants us to wait; counting it
    as a failure makes a healthy miner look broken and spends its backoff on
    the wrong problem. A 403 from a public endpoint usually means the address
    is refused rather than the request being wrong, and it is reported with
    that reading attached so nobody debugs the query.
    """
    status, body, _headers = await client.get(url, headers=headers)
    if status == 429:
        raise RateLimited(url)
    if status == 403:
        raise RuntimeError(
            f"HTTP 403 from {url.split('?')[0]} -- public endpoint refusing "
            "this address, not a malformed request")
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from {url.split('?')[0]}")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"response was not JSON: {exc}") from exc


# --- measured attention --------------------------------------------------

def reddit_new_miner(client: Any, subs: Sequence[str] = DEFAULT_SUBREDDITS,
                     *, per_pass: int = 3) -> Callable[[], Awaitable[List[Dict]]]:
    """New posts from public subreddits, rotated so no single one is hammered.

    Score and comment count on a fresh post are the closest thing to a
    measured crowd reading available without an account: they say how many
    people reacted, not merely that something was said.
    """
    cursor = {"at": 0}

    async def fetch() -> List[Dict[str, Any]]:
        chosen = [subs[(cursor["at"] + offset) % len(subs)]
                  for offset in range(min(per_pass, len(subs)))]
        cursor["at"] = (cursor["at"] + len(chosen)) % len(subs)
        records: List[Dict[str, Any]] = []
        for sub in chosen:
            try:
                payload = await _get_json(client, REDDIT_NEW_URL.format(sub=sub))
            except RateLimited:
                raise
            except Exception as exc:
                logger.debug("reddit mine failed for r/%s: %s", sub, exc)
                continue
            children = ((payload or {}).get("data") or {}).get("children") or []
            for child in children:
                post = (child or {}).get("data") or {}
                if not post.get("id"):
                    continue
                records.append({
                    "venue": "reddit",
                    "community": sub,
                    "id": post.get("id"),
                    "title": post.get("title", ""),
                    "text": (post.get("selftext") or "")[:2000],
                    "author": post.get("author", ""),
                    "created_utc": float(post.get("created_utc", 0) or 0),
                    "score": int(post.get("score", 0) or 0),
                    "comments": int(post.get("num_comments", 0) or 0),
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "data_status": "OK",
                })
        return records

    return fetch


def hackernews_miner(client: Any, query: str = "solana"
                     ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Recent Hacker News stories matching a query.

    Low volume and high signal: when a Solana story reaches this audience the
    attention is broad rather than crypto-native, which is the transition the
    ignition model calls mass FOMO.
    """
    async def fetch() -> List[Dict[str, Any]]:
        payload = await _get_json(
            client, HN_SEARCH_URL.format(query=urllib.parse.quote(query)))
        hits = (payload or {}).get("hits") or []
        return [{
            "venue": "hackernews",
            "id": hit.get("objectID"),
            "title": hit.get("title") or hit.get("story_title") or "",
            "url": hit.get("url") or "",
            "author": hit.get("author", ""),
            "created_utc": float(hit.get("created_at_i", 0) or 0),
            "score": int(hit.get("points", 0) or 0),
            "comments": int(hit.get("num_comments", 0) or 0),
            "data_status": "OK",
        } for hit in hits if hit.get("objectID")]

    return fetch


def wikipedia_attention_miner(client: Any,
                              articles: Sequence[str] = DEFAULT_ATTENTION_ARTICLES,
                              *, days: int = 14,
                              ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Daily pageviews for theme articles: retail attention, measured.

    This is the broadest attention series available for free, and it moves
    days before a retail wave rather than during it. Daily granularity means
    it is a regime input, not a trade trigger -- which is exactly how the
    market-context consumer uses it.
    """
    async def fetch() -> List[Dict[str, Any]]:
        now = time.time()
        end = time.strftime("%Y%m%d", time.gmtime(now - 86_400))
        start = time.strftime("%Y%m%d", time.gmtime(now - days * 86_400))
        records: List[Dict[str, Any]] = []
        for article in articles:
            try:
                payload = await _get_json(client, WIKI_PAGEVIEWS_URL.format(
                    article=urllib.parse.quote(article, safe=""),
                    start=start, end=end))
            except RateLimited:
                raise
            except Exception as exc:
                logger.debug("pageviews mine failed for %s: %s", article, exc)
                continue
            items = (payload or {}).get("items") or []
            series = [int(item.get("views", 0) or 0) for item in items]
            if not series:
                continue
            baseline = sum(series[:-1]) / max(1, len(series) - 1)
            records.append({
                "venue": "wikipedia",
                "article": article,
                "days": len(series),
                "views_latest": series[-1],
                "views_mean": sum(series) / len(series),
                # Latest against its own trailing mean. A ratio, because the
                # absolute level differs by three orders of magnitude between
                # these articles and only the deviation is comparable.
                "attention_ratio": (series[-1] / baseline) if baseline > 0 else None,
                "data_status": "OK",
            })
        return records

    return fetch


def youtube_recent_miner(client: Any, key_provider: Callable[[], str],
                         queries: Sequence[str] = ("pump.fun", "solana memecoin"),
                         *, lookback_hours: int = 2,
                         ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Videos published in the last couple of hours for our standing queries.

    YouTube is where a retail wave is manufactured rather than reported: the
    video goes up, and the buying follows it by minutes. Publication time is
    the signal, so this asks for recency rather than relevance and pays the
    quota for a narrow window instead of a broad one.
    """
    async def fetch() -> List[Dict[str, Any]]:
        key = (key_provider() or "").strip()
        if not key:
            # Registered with requires_env, so this should be unreachable; if
            # the key is pulled at runtime the miner goes silent rather than
            # emitting an empty result that reads as "nobody posted".
            raise RuntimeError("YOUTUBE_API_KEY absent at call time")
        after = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                              time.gmtime(time.time() - lookback_hours * 3600))
        records: List[Dict[str, Any]] = []
        for query in queries:
            payload = await _get_json(client, YOUTUBE_SEARCH_URL.format(
                query=urllib.parse.quote(query), after=after, key=key))
            for item in (payload or {}).get("items") or []:
                snippet = item.get("snippet") or {}
                video_id = ((item.get("id") or {}).get("videoId"))
                if not video_id:
                    continue
                records.append({
                    "venue": "youtube",
                    "query": query,
                    "id": video_id,
                    "title": snippet.get("title", ""),
                    "text": (snippet.get("description") or "")[:2000],
                    "author": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "data_status": "OK",
                })
        return records

    return fetch


# --- the corpus that gives a name meaning --------------------------------

def dexscreener_search_miner(client: Any, terms: Callable[[], Sequence[str]],
                             *, per_pass: int = 6,
                             ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Every pair matching a watched token's name or symbol, across all chains.

    This is the copycat corpus. A symbol appearing on nine pairs, six of them
    dead, is a reused name; the same symbol appearing once is a new one. The
    authenticity resolver cannot make that call without the comparison set,
    and the comparison set is not something the chain stream can supply.
    """
    async def fetch() -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for term in list(terms())[:per_pass]:
            if not term or len(term) < 2:
                continue
            payload = await _get_json(client, DEXSCREENER_SEARCH_URL.format(
                query=urllib.parse.quote(term)))
            pairs = (payload or {}).get("pairs") or []
            live = 0
            for pair in pairs:
                liquidity = float(((pair.get("liquidity") or {}).get("usd") or 0))
                if liquidity > 0:
                    live += 1
            records.append({
                "term": term,
                "pairs_matching": len(pairs),
                "pairs_with_liquidity": live,
                # Reuse is the point: many matches, few alive, means the name
                # has been run before and abandoned.
                "abandoned_share": ((len(pairs) - live) / len(pairs)
                                    if pairs else None),
                "chains": sorted({pair.get("chainId", "") for pair in pairs
                                  if pair.get("chainId")}),
                "data_status": "OK",
            })
        return records

    return fetch


def coingecko_trending_miner(client: Any
                             ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """What retail is searching for right now.

    The trending list is a search-volume ranking, not a price ranking, which
    makes it a genuine attention reading rather than a restatement of returns.
    """
    async def fetch() -> List[Dict[str, Any]]:
        payload = await _get_json(client, COINGECKO_TRENDING_URL)
        records: List[Dict[str, Any]] = []
        for rank, entry in enumerate((payload or {}).get("coins") or []):
            item = (entry or {}).get("item") or {}
            if not item.get("id"):
                continue
            records.append({
                "venue": "coingecko_trending",
                "rank": rank,
                "id": item.get("id"),
                "symbol": (item.get("symbol") or "").upper(),
                "name": item.get("name", ""),
                "market_cap_rank": item.get("market_cap_rank"),
                "data_status": "OK",
            })
        return records

    return fetch


def github_activity_miner(client: Any, token_provider: Callable[[], str],
                          queries: Sequence[str] = DEFAULT_REPO_QUERIES,
                          ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Public repositories touching the programs we decode.

    A program upgrade shows up in a public IDL before it shows up as decoding
    failures in production. Watching for it is the difference between changing
    a discriminator deliberately and discovering it during a launch.
    """
    async def fetch() -> List[Dict[str, Any]]:
        token = (token_provider() or "").strip()
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        records: List[Dict[str, Any]] = []
        for query in queries:
            payload = await _get_json(
                client, GITHUB_SEARCH_URL.format(query=urllib.parse.quote(query)),
                headers=headers)
            for repo in (payload or {}).get("items") or []:
                if not repo.get("full_name"):
                    continue
                records.append({
                    "venue": "github",
                    "query": query,
                    "id": repo.get("full_name"),
                    "title": repo.get("full_name"),
                    "text": (repo.get("description") or "")[:1000],
                    "url": repo.get("html_url", ""),
                    "stars": int(repo.get("stargazers_count", 0) or 0),
                    "pushed_at": repo.get("pushed_at", ""),
                    "data_status": "OK",
                })
        return records

    return fetch


def register_web_miners(pool: DataMinerPool, *, http: Any,
                        search_terms: Callable[[], Sequence[str]],
                        youtube_key: Callable[[], str] = lambda: "",
                        github_token: Callable[[], str] = lambda: "",
                        ) -> Dict[str, bool]:
    """Declare the public-web set.

    Cadence here is set by how fast the underlying thing can change and by the
    politeness the endpoint expects, whichever is slower. Wikipedia publishes
    daily and is mined daily; a YouTube upload matters within minutes of going
    live and is mined on the minute against a two-hour window, which keeps the
    quota cost flat regardless of how long the desk has been running.
    """
    registrations = (
        (MinerSpec(
            miner_id="web:reddit_new", enriches=Enriches.SOCIAL_ATTENTION,
            cadence_seconds=CADENCE_MINUTE, endpoint="reddit public listings",
            detail="new posts with score and comment counts, subs rotated"),
         reddit_new_miner(http)),
        (MinerSpec(
            miner_id="web:hackernews", enriches=Enriches.SOCIAL_ATTENTION,
            cadence_seconds=CADENCE_QUARTER, endpoint=HN_SEARCH_URL,
            detail="broad-audience attention; the mass-FOMO transition"),
         hackernews_miner(http)),
        (MinerSpec(
            miner_id="web:wikipedia_attention", enriches=Enriches.MARKET_CONTEXT,
            cadence_seconds=CADENCE_DAILY, endpoint="wikimedia pageviews",
            detail="daily theme pageviews; a regime input, not a trigger"),
         wikipedia_attention_miner(http)),
        (MinerSpec(
            miner_id="web:youtube_recent", enriches=Enriches.NARRATIVE,
            cadence_seconds=CADENCE_MINUTE, endpoint="youtube data api search",
            requires_env=("YOUTUBE_API_KEY",),
            detail="uploads in the last two hours for standing queries"),
         youtube_recent_miner(http, youtube_key)),
        (MinerSpec(
            miner_id="web:name_corpus", enriches=Enriches.TOKEN_METADATA,
            cadence_seconds=CADENCE_MINUTE, endpoint="dexscreener search",
            detail="every pair sharing a watched token's name; copycat corpus"),
         dexscreener_search_miner(http, search_terms)),
        (MinerSpec(
            miner_id="web:coingecko_trending", enriches=Enriches.SOCIAL_ATTENTION,
            cadence_seconds=CADENCE_QUARTER, endpoint=COINGECKO_TRENDING_URL,
            detail="search-volume ranking; attention, not returns"),
         coingecko_trending_miner(http)),
        (MinerSpec(
            miner_id="web:program_repos", enriches=Enriches.NARRATIVE,
            cadence_seconds=CADENCE_HOURLY, endpoint=GITHUB_SEARCH_URL,
            requires_env=("GITHUB_TOKEN",),
            detail="public repos touching the programs we decode"),
         github_activity_miner(http, github_token)),
    )
    return {spec.miner_id: pool.register(spec, fetch)
            for spec, fetch in registrations}
