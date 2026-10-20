"""The hourly world crawler: hunting events, not articles.

`GlobalResearchMiner` already runs hourly and searches GitHub, arXiv and
multilingual RSS for research. `world_miners` already pulls GDELT's fifteen
minute global files. Both are narrative discovery over sources someone already
wrote down. Neither of them hunts *token launch events* across the open web,
and neither of them discovers a source it was not already told about.

This does. It runs on two clocks, because one clock is wrong in both
directions:

    CONTINUOUS   the chain, wallets, pool events, and every source already in
                 the watch mesh. Sub-second where it matters. Not this module.
    HOURLY       discovery -- new sites, forums, accounts, communities, repos,
                 launchpads, callers, terminology, languages, whole regional
                 ecosystems. This module.

Scraping the entire web every second is neither possible nor useful. Waiting an
hour to re-read something hot is negligent. So discovery is hourly, and its
OUTPUT is a promotion: a verified source moves into the continuous mesh, where
future posts arrive seconds after publication instead of up to an hour later.

**It works in both directions, and the second one is the interesting half.**

    CHAIN -> INTERNET   a mint appears. Expand around it: mint address, ticker,
                        name, creator, funding addresses, website, Telegram
                        handle, X account, developer aliases, prior token
                        names, launchpad -- in eleven languages -- and find who
                        was discussing it, especially BEFORE it existed.

    INTERNET -> CHAIN   a rumour, project, person or domain appears with no
                        token yet. Register it as a watch term. When a mint
                        later matches its name, ticker, domain or handle, that
                        match is a pre-launch hit with a measured lead time.

The first direction explains launches. The second one anticipates them, and
it is the direction almost nothing does, because it requires holding a claim
for days before it can be scored.

**The adversarial mandate.** A crawler that only searches around what the
model already likes confirms the model. A fixed fraction of every cycle's
budget is therefore reserved for material that could embarrass it: tokens the
model rejected that later exploded, sources previously scored badly that have
improved, unknown launchpads, unfamiliar terminology, new bundling and funding
patterns, new social platforms, new geographies. The reserve is spent even when
the ordinary queue is full -- otherwise it is spent never.

**The boundary.** Public pages, public chain data, public posts, communities
the desk is a member of and reads within their terms, and material volunteered
to it. Nothing here logs in, bypasses an access control, uses someone else's
credentials, or reads private messages. `AccessPolicy` enforces that at the
point a target is admitted, not as a comment, and a target that requires
authentication is refused with the reason recorded.

**Zero authority.** Findings become `InsiderEvent` candidates with no
confidence attached. They earn weight from forward outcomes through
`SourceEdgeLedger`, and nothing they contain can move capital on its own.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple)

from src.research.insider_event import (ClaimType, InsiderEvent, LawfulAccess,
                                        Mechanism)

logger = logging.getLogger(__name__)

WORLD_CRAWLER_SCHEMA_VERSION = "v1"

#: One discovery cycle per hour. The continuous mesh is elsewhere; this cadence
#: is for finding things that did not exist an hour ago.
DEFAULT_CYCLE_SECONDS = 3_600.0

#: Queries per cycle. A ceiling rather than a target: an exhausted queue ends
#: the cycle early rather than inventing work.
DEFAULT_QUERY_BUDGET = 120

#: Share of every cycle spent on material that could contradict the model.
#: Spent first, so a busy queue cannot crowd it out -- which is exactly how a
#: research budget silently becomes a confirmation budget.
DEFAULT_ADVERSARIAL_RESERVE = 0.25

#: A source must produce this many verified observations before it is promoted
#: into the continuous mesh. One lucky parse is not a feed.
DEFAULT_PROMOTION_OBSERVATIONS = 3


class Direction(Enum):
    CHAIN_TO_INTERNET = "chain_to_internet"
    INTERNET_TO_CHAIN = "internet_to_chain"
    ADVERSARIAL = "adversarial"


class SeedKind(Enum):
    MINT = "mint"
    TICKER = "ticker"
    TOKEN_NAME = "token_name"
    CREATOR = "creator"
    FUNDER = "funder"
    DOMAIN = "domain"
    HANDLE = "handle"
    ALIAS = "alias"
    LAUNCHPAD = "launchpad"
    NARRATIVE = "narrative"


#: Event words, per language, that turn a name into a launch query. Written out
#: rather than machine-translated at runtime: a wrong translation produces a
#: query that finds nothing and a coverage count that looks fine.
EVENT_TERMS: Dict[str, Tuple[str, ...]] = {
    "en": ("launch", "listing", "presale", "insider", "alpha", "call",
           "contract address", "buy before", "announcement", "airdrop"),
    "zh": ("上线", "上币", "预售", "内幕", "合约地址", "喊单", "公告", "空投"),
    "ko": ("상장", "프리세일", "내부자", "계약 주소", "공지", "에어드랍", "떡상"),
    "ja": ("上場", "プレセール", "インサイダー", "コントラクトアドレス", "発表",
           "エアドロップ"),
    "ru": ("листинг", "пресейл", "инсайдер", "адрес контракта", "анонс",
           "эирдроп"),
    "tr": ("listeleme", "ön satış", "içeriden", "sözleşme adresi", "duyuru",
           "airdrop"),
    "ar": ("إدراج", "بيع مسبق", "عنوان العقد", "إعلان"),
    "es": ("listado", "preventa", "información privilegiada",
           "dirección del contrato", "anuncio"),
    "pt": ("listagem", "pré-venda", "informação privilegiada",
           "endereço do contrato", "anúncio"),
    "id": ("listing", "prapenjualan", "orang dalam", "alamat kontrak",
           "pengumuman"),
    "vi": ("niêm yết", "bán trước", "nội gián", "địa chỉ hợp đồng",
           "thông báo"),
}

#: Seeds that carry their own identity -- an address is an address in every
#: language -- and gain nothing from being decorated with translated nouns.
_LANGUAGE_FREE_SEEDS = {SeedKind.MINT, SeedKind.CREATOR, SeedKind.FUNDER}

#: Where a discovered link might live. Used to classify, never to authorise:
#: classification says what parser to use, `AccessPolicy` says whether to fetch.
_PLATFORM_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("telegram", re.compile(r"(?:t\.me|telegram\.me)/(?:s/)?([\w_]{4,64})", re.I)),
    ("x", re.compile(r"(?:twitter\.com|x\.com)/([A-Za-z0-9_]{2,15})", re.I)),
    ("discord", re.compile(r"discord\.(?:gg|com/invite)/([\w-]{4,32})", re.I)),
    ("github", re.compile(r"github\.com/([\w.-]{1,39})(?:/([\w.-]+))?", re.I)),
    ("youtube", re.compile(r"(?:youtube\.com|youtu\.be)/\S+", re.I)),
    ("reddit", re.compile(r"reddit\.com/r/(\w{2,32})", re.I)),
)

_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.I)
# Single-character labels are real and matter here: x.com and t.me are
# two of the most common hosts the crawler will ever see, and a minimum
# label length of two silently drops both.
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]{0,62}\.[a-z]{2,24})\b", re.I)
#: Base58, 32-44 chars: a Solana address as it appears in prose.
_MINT_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")


@dataclass(frozen=True)
class AccessPolicy:
    """What the crawler is allowed to touch. Enforced, not documented.

    The desk's stated boundary is public and lawfully obtainable information.
    That is a property of the TARGET, so it is checked where targets are
    admitted rather than trusted to the caller.
    """

    allow_authenticated: bool = False
    #: Host substrings that are refused outright -- places whose useful content
    #: exists only behind someone else's login.
    denied_hosts: Tuple[str, ...] = (
        "mail.", "webmail.", "/dm/", "discord.com/channels/")
    denied_schemes: Tuple[str, ...] = ("file", "ftp", "data", "javascript")

    def admits(self, url: str) -> Tuple[bool, str]:
        target = (url or "").strip()
        if not target:
            return False, "empty target"
        lowered = target.lower()
        scheme = lowered.split(":", 1)[0] if ":" in lowered else ""
        if scheme in self.denied_schemes:
            return False, f"scheme {scheme} is not a public web fetch"
        if not lowered.startswith(("http://", "https://")):
            return False, "not an http(s) target"
        if "@" in lowered.split("//", 1)[-1].split("/", 1)[0]:
            return False, ("target embeds credentials; the crawler does not "
                           "authenticate")
        for marker in self.denied_hosts:
            if marker in lowered:
                return False, (f"target matches {marker!r}: its content sits "
                               "behind an access control, and reading it would "
                               "require bypassing one")
        return True, ""


@dataclass(frozen=True)
class Seed:
    """One thing to expand queries around."""

    kind: SeedKind
    value: str
    #: Where the seed came from, so a discovery chain can be walked backwards.
    origin: str = ""
    first_seen: float = 0.0

    def key(self) -> str:
        return f"{self.kind.value}:{self.value.lower()}"


@dataclass
class Query:
    text: str
    language: str
    seed: Seed
    direction: Direction

    def key(self) -> str:
        return hashlib.sha256(
            f"{self.language}|{self.text}".encode("utf-8")).hexdigest()[:16]


@dataclass
class Finding:
    """One result a search returned, before anything is believed about it."""

    url: str
    title: str = ""
    snippet: str = ""
    language: str = ""
    published_at: Optional[float] = None
    query: Optional[Query] = None
    observed_at: float = field(default_factory=time.time)

    def text(self) -> str:
        return f"{self.title} {self.snippet}"


@dataclass
class SourceCandidate:
    """A place that produced something, and how often it has done so."""

    identifier: str
    platform: str
    url: str
    observations: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    languages: Set[str] = field(default_factory=set)
    seeds: Set[str] = field(default_factory=set)
    promoted: bool = False
    refused_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["languages"] = sorted(self.languages)
        data["seeds"] = sorted(self.seeds)
        return data


@dataclass
class WatchTerm:
    """An internet-side entity with no token yet. The pre-launch half."""

    term: str
    kind: SeedKind
    registered_at: float
    origin_url: str = ""
    language: str = ""
    matched_mint: str = ""
    matched_at: Optional[float] = None

    @property
    def lead_seconds(self) -> Optional[float]:
        if self.matched_at is None:
            return None
        return max(0.0, self.matched_at - self.registered_at)


def classify_platform(url: str) -> Tuple[str, str]:
    """Which platform a link belongs to, and its identifier there."""
    for name, pattern in _PLATFORM_PATTERNS:
        match = pattern.search(url or "")
        if match:
            groups = [item for item in match.groups() if item]
            return name, (groups[0] if groups else url)
    match = _DOMAIN_RE.search((url or "").split("//", 1)[-1])
    return ("web", match.group(1).lower() if match else (url or ""))


def expand_queries(seed: Seed, *, languages: Sequence[str] = (),
                   direction: Direction = Direction.CHAIN_TO_INTERNET,
                   ) -> List[Query]:
    """Turn one seed into the searches that could find who knew about it.

    An address is expanded bare and in English only: it is the same string in
    every language, and pairing it with translated nouns multiplies the query
    count without widening coverage at all.
    """
    wanted = [code for code in (languages or EVENT_TERMS.keys())
              if code in EVENT_TERMS]
    if not seed.value.strip():
        return []
    queries: List[Query] = []
    if seed.kind in _LANGUAGE_FREE_SEEDS:
        queries.append(Query(text=seed.value, language="en", seed=seed,
                             direction=direction))
        for term in EVENT_TERMS["en"][:4]:
            queries.append(Query(text=f"{seed.value} {term}", language="en",
                                 seed=seed, direction=direction))
        return queries
    for code in wanted:
        for term in EVENT_TERMS[code]:
            queries.append(Query(text=f"{seed.value} {term}", language=code,
                                 seed=seed, direction=direction))
    return queries


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Pull the things worth chasing out of a page's text.

    Deliberately over-inclusive on links and under-inclusive on mints: a link
    that turns out to be nothing costs one refused fetch, while a base58 string
    that is not a mint pollutes the chain-side watch list.
    """
    body = text or ""
    urls = _URL_RE.findall(body)
    mints = [value for value in _MINT_RE.findall(body)
             # An address embedded in a URL is already carried by that URL, and
             # counting it twice inflates every discovery statistic.
             if not any(value in url for url in urls)]
    handles: List[str] = []
    for _, pattern in _PLATFORM_PATTERNS:
        for match in pattern.finditer(body):
            groups = [item for item in match.groups() if item]
            if groups:
                handles.append(groups[0])
    domains = [match.group(1).lower()
               for url in urls
               for match in [_DOMAIN_RE.search(url.split("//", 1)[-1])]
               if match]
    return {"urls": urls, "mints": sorted(set(mints)),
            "handles": sorted(set(handles)), "domains": sorted(set(domains))}


class PublicInternetEventCrawler:
    """Hourly, world-scale, bidirectional discovery with no trading authority.

    The searcher is injected: a callable taking a `Query` and returning
    `Finding` rows. This module performs no I/O, so it is testable without the
    network and the desk's HTTP policy stays in one place.
    """

    def __init__(self, *,
                 searcher: Optional[Callable[[Query], Sequence[Finding]]] = None,
                 policy: AccessPolicy = AccessPolicy(),
                 query_budget: int = DEFAULT_QUERY_BUDGET,
                 adversarial_reserve: float = DEFAULT_ADVERSARIAL_RESERVE,
                 promotion_observations: int = DEFAULT_PROMOTION_OBSERVATIONS,
                 languages: Sequence[str] = ()):
        self.searcher = searcher
        self.policy = policy
        self.query_budget = int(query_budget)
        self.adversarial_reserve = float(adversarial_reserve)
        self.promotion_observations = int(promotion_observations)
        self.languages = tuple(languages) or tuple(EVENT_TERMS)

        self.queue: List[Seed] = []
        self.adversarial_queue: List[Seed] = []
        self.seen_seeds: Set[str] = set()
        self.seen_queries: Set[str] = set()
        self.candidates: Dict[str, SourceCandidate] = {}
        self.watch_terms: Dict[str, WatchTerm] = {}
        self.refused: List[Dict[str, str]] = []
        self.cycles = 0
        #: Findings from the most recent cycle, so the caller that holds the
        #: market context can turn them into canonical events. Cleared at the
        #: start of every cycle, and bounded, because this is a working buffer
        #: and not a second corpus.
        self.recent_findings: List[Finding] = []
        self.max_recent_findings = 5_000

    # -- seeding ---------------------------------------------------------

    def seed(self, seed: Seed, *, adversarial: bool = False) -> bool:
        """Queue a seed once. Returns whether it was new."""
        if not seed.value.strip():
            return False
        key = seed.key()
        if key in self.seen_seeds:
            return False
        self.seen_seeds.add(key)
        (self.adversarial_queue if adversarial else self.queue).append(seed)
        return True

    def seed_from_chain(self, *, mint: str = "", ticker: str = "",
                        name: str = "", creator: str = "",
                        funders: Sequence[str] = (), domain: str = "",
                        handles: Sequence[str] = (),
                        launchpad: str = "", now: Optional[float] = None
                        ) -> int:
        """CHAIN -> INTERNET. Everything a launch gives us to search around."""
        stamp = time.time() if now is None else now
        added = 0
        pairs: List[Tuple[SeedKind, str]] = [
            (SeedKind.MINT, mint), (SeedKind.TICKER, ticker),
            (SeedKind.TOKEN_NAME, name), (SeedKind.CREATOR, creator),
            (SeedKind.DOMAIN, domain), (SeedKind.LAUNCHPAD, launchpad)]
        pairs.extend((SeedKind.FUNDER, value) for value in funders)
        pairs.extend((SeedKind.HANDLE, value) for value in handles)
        for kind, value in pairs:
            if value:
                added += int(self.seed(Seed(kind=kind, value=str(value),
                                            origin="chain", first_seen=stamp)))
        return added

    def seed_adversarial(self, seeds: Iterable[Seed]) -> int:
        """Material chosen because it could contradict the current model."""
        return sum(int(self.seed(item, adversarial=True)) for item in seeds)

    # -- the internet -> chain half --------------------------------------

    def register_watch_term(self, term: str, kind: SeedKind, *,
                            origin_url: str = "", language: str = "",
                            now: Optional[float] = None) -> bool:
        """Hold an entity that has no token yet, so a later mint can match it.

        This is the half that anticipates rather than explains, and it is the
        half that costs patience: the payoff is days later, when a mint appears
        carrying a name registered before it existed.
        """
        cleaned = (term or "").strip().lower()
        if len(cleaned) < 3 or cleaned in self.watch_terms:
            return False
        self.watch_terms[cleaned] = WatchTerm(
            term=cleaned, kind=kind,
            registered_at=time.time() if now is None else now,
            origin_url=origin_url, language=language)
        return True

    def match_new_mint(self, *, mint: str, ticker: str = "", name: str = "",
                       domain: str = "", handles: Sequence[str] = (),
                       now: Optional[float] = None) -> List[WatchTerm]:
        """Did anything we were already watching just become a token?

        Only unmatched terms can hit, and a hit is recorded once. A term that
        matched three mints would otherwise report three leads for one piece of
        information.
        """
        stamp = time.time() if now is None else now
        haystack = {value.strip().lower()
                    for value in [ticker, name, domain, *handles] if value}
        hits: List[WatchTerm] = []
        for term in self.watch_terms.values():
            if term.matched_at is not None:
                continue
            if term.term in haystack:
                term.matched_mint = mint
                term.matched_at = stamp
                hits.append(term)
        return hits

    def pre_launch_leads(self) -> List[Dict[str, Any]]:
        return [{"term": term.term, "kind": term.kind.value,
                 "mint": term.matched_mint,
                 "lead_seconds": term.lead_seconds,
                 "origin_url": term.origin_url, "language": term.language}
                for term in self.watch_terms.values()
                if term.matched_at is not None]

    # -- the cycle -------------------------------------------------------

    def _plan(self) -> List[Query]:
        """Build this cycle's queries, adversarial reserve spent first."""
        reserve = int(self.query_budget * max(0.0, min(
            1.0, self.adversarial_reserve)))
        planned: List[Query] = []

        def _drain(queue: List[Seed], limit: int,
                   direction: Direction) -> None:
            while queue and len(planned) < limit:
                seed = queue.pop(0)
                for query in expand_queries(seed, languages=self.languages,
                                            direction=direction):
                    if len(planned) >= limit:
                        # Unspent expansions are not lost: the seed goes back
                        # so the next cycle continues it rather than dropping
                        # the languages that happened to sort last.
                        queue.insert(0, seed)
                        return
                    if query.key() in self.seen_queries:
                        continue
                    self.seen_queries.add(query.key())
                    planned.append(query)

        _drain(self.adversarial_queue, reserve, Direction.ADVERSARIAL)
        _drain(self.queue, self.query_budget, Direction.CHAIN_TO_INTERNET)
        # Any reserve the adversarial queue could not spend is returned to the
        # ordinary queue rather than burned, but the reserve is always OFFERED
        # first, which is the property that matters.
        return planned

    def _absorb(self, finding: Finding, *, now: float) -> Dict[str, int]:
        """Turn one result into candidates, watch terms and follow-on seeds."""
        counts = {"candidates": 0, "watch_terms": 0, "seeds": 0,
                  "refused": 0}
        admitted, reason = self.policy.admits(finding.url)
        if not admitted:
            self.refused.append({"url": finding.url, "reason": reason})
            counts["refused"] = 1
            return counts

        platform, identifier = classify_platform(finding.url)
        key = f"{platform}:{identifier}"
        candidate = self.candidates.get(key)
        if candidate is None:
            candidate = SourceCandidate(identifier=identifier,
                                        platform=platform, url=finding.url,
                                        first_seen=now)
            self.candidates[key] = candidate
            counts["candidates"] = 1
        candidate.observations += 1
        candidate.last_seen = now
        if finding.language:
            candidate.languages.add(finding.language)
        if finding.query is not None:
            candidate.seeds.add(finding.query.seed.key())

        entities = extract_entities(finding.text())
        for mint in entities["mints"]:
            counts["seeds"] += int(self.seed(Seed(
                kind=SeedKind.MINT, value=mint, origin=finding.url,
                first_seen=now)))
        for handle in entities["handles"]:
            counts["seeds"] += int(self.seed(Seed(
                kind=SeedKind.HANDLE, value=handle, origin=finding.url,
                first_seen=now)))
            counts["watch_terms"] += int(self.register_watch_term(
                handle, SeedKind.HANDLE, origin_url=finding.url,
                language=finding.language, now=now))
        for domain in entities["domains"]:
            counts["watch_terms"] += int(self.register_watch_term(
                domain, SeedKind.DOMAIN, origin_url=finding.url,
                language=finding.language, now=now))
        return counts

    def _begin(self) -> Tuple[Optional[Dict[str, Any]], List[Query]]:
        """Shared preamble for the sync and async passes."""
        self.cycles += 1
        if self.searcher is None:
            return ({"status": "DATA_BLOCKED",
                     "reason": "no searcher configured",
                     "cycle": self.cycles}, [])
        planned = self._plan()
        if not planned:
            return ({"status": "IDLE", "cycle": self.cycles, "queries": 0,
                     "reason": "no unseen seeds queued"}, [])
        self.recent_findings = []
        return None, planned

    def _consume(self, query: Query, results: Sequence[Finding], *,
                 now: float, totals: Dict[str, int]) -> int:
        seen = 0
        for finding in results or []:
            if finding.query is None:
                finding.query = query
            if not finding.language:
                finding.language = query.language
            seen += 1
            if len(self.recent_findings) < self.max_recent_findings:
                self.recent_findings.append(finding)
            counts = self._absorb(finding, now=now)
            for name, value in counts.items():
                totals[name] += value
        return seen

    def _report(self, planned: Sequence[Query], *, findings: int,
                totals: Dict[str, int], failures: List[Dict[str, str]],
                by_direction: Dict[str, int], stamp: float) -> Dict[str, Any]:
        promoted = self.promotable()
        return {
            "status": "OK",
            "schema": WORLD_CRAWLER_SCHEMA_VERSION,
            "cycle": self.cycles,
            "queries": len(planned),
            "queries_by_direction": by_direction,
            "adversarial_queries": by_direction.get(
                Direction.ADVERSARIAL.value, 0),
            "findings": findings,
            "new_candidates": totals["candidates"],
            "new_watch_terms": totals["watch_terms"],
            "new_seeds": totals["seeds"],
            "refused": totals["refused"],
            "failures": failures,
            "promotable": [item.identifier for item in promoted],
            "queue_depth": len(self.queue),
            "adversarial_queue_depth": len(self.adversarial_queue),
            "watch_terms": len(self.watch_terms),
            "ran_at": stamp,
        }

    def cycle(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        """One hourly discovery pass. Returns what it did, honestly."""
        stamp = time.time() if now is None else now
        early, planned = self._begin()
        if early is not None:
            return early
        totals = {"candidates": 0, "watch_terms": 0, "seeds": 0, "refused": 0}
        findings = 0
        failures: List[Dict[str, str]] = []
        by_direction: Dict[str, int] = {}
        for query in planned:
            by_direction[query.direction.value] = by_direction.get(
                query.direction.value, 0) + 1
            try:
                results = self.searcher(query)
            except Exception as exc:
                failures.append({"query": query.text,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            if inspect.isawaitable(results):
                results.close()
                failures.append({
                    "query": query.text,
                    "error": ("searcher is async and cycle() is not; "
                              "await acycle() instead")})
                continue
            findings += self._consume(query, results, now=stamp, totals=totals)
        return self._report(planned, findings=findings, totals=totals,
                            failures=failures, by_direction=by_direction,
                            stamp=stamp)

    async def acycle(self, *, now: Optional[float] = None) -> Dict[str, Any]:
        """The same pass driven by an async searcher.

        The real searcher fans out over HTTP, so this is the form the runtime
        uses; `cycle` stays for offline replay and for tests that would rather
        not own an event loop.
        """
        stamp = time.time() if now is None else now
        early, planned = self._begin()
        if early is not None:
            return early
        totals = {"candidates": 0, "watch_terms": 0, "seeds": 0, "refused": 0}
        findings = 0
        failures: List[Dict[str, str]] = []
        by_direction: Dict[str, int] = {}
        for query in planned:
            by_direction[query.direction.value] = by_direction.get(
                query.direction.value, 0) + 1
            try:
                results = self.searcher(query)
                if inspect.isawaitable(results):
                    results = await results
            except Exception as exc:
                failures.append({"query": query.text,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            findings += self._consume(query, results, now=stamp, totals=totals)
        return self._report(planned, findings=findings, totals=totals,
                            failures=failures, by_direction=by_direction,
                            stamp=stamp)

    # -- promotion -------------------------------------------------------

    def promotable(self) -> List[SourceCandidate]:
        """Candidates that have produced enough to join the continuous mesh.

        The promotion is the crawler's whole point: an hourly search finds a
        channel once, and moving it into the mesh means its next post arrives
        in seconds. Sources are promoted for having produced material, never
        for what they claim about themselves.
        """
        return [candidate for candidate in self.candidates.values()
                if not candidate.promoted
                and candidate.observations >= self.promotion_observations]

    def mark_promoted(self, identifiers: Iterable[str]) -> int:
        wanted = set(identifiers)
        count = 0
        for candidate in self.candidates.values():
            if candidate.identifier in wanted and not candidate.promoted:
                candidate.promoted = True
                count += 1
        return count

    # -- output ----------------------------------------------------------

    def to_events(self, findings: Sequence[Finding], *,
                  access: LawfulAccess = LawfulAccess.PUBLIC
                  ) -> List[InsiderEvent]:
        """Findings as canonical events with no confidence attached.

        `confidence_at_time` is deliberately left None. A crawler result is an
        observation; the ledger decides what it is worth from what happened
        next, and a number invented here would be laundered into weight
        downstream.
        """
        events: List[InsiderEvent] = []
        for finding in findings:
            platform, identifier = classify_platform(finding.url)
            entities = extract_entities(finding.text())
            mint = entities["mints"][0] if entities["mints"] else ""
            events.append(InsiderEvent(
                event_id=hashlib.sha256(
                    f"{finding.url}|{finding.title}".encode("utf-8")
                ).hexdigest()[:24],
                source_id=f"{platform}:{identifier}",
                source_at=float(finding.published_at
                                if finding.published_at is not None
                                else finding.observed_at),
                observed_at=float(finding.observed_at),
                lawful_access=access,
                provenance="public_internet_crawl",
                source_type=platform, source_url=finding.url,
                token=mint, mint=mint, claim=finding.title,
                claim_type=ClaimType.UNSPECIFIED,
                mechanism=Mechanism.UNCLASSIFIED,
                language=finding.language,
                linked_domains=entities["domains"],
                linked_entities=entities["handles"],
                content_hash=hashlib.sha256(
                    finding.text().encode("utf-8")).hexdigest(),
                confidence_at_time=None))
        return events

    def status(self) -> Dict[str, Any]:
        promoted = sum(1 for item in self.candidates.values() if item.promoted)
        matched = sum(1 for item in self.watch_terms.values()
                      if item.matched_at is not None)
        return {
            "cycles": self.cycles,
            "queue_depth": len(self.queue),
            "adversarial_queue_depth": len(self.adversarial_queue),
            "candidates": len(self.candidates),
            "promoted": promoted,
            "watch_terms": len(self.watch_terms),
            "watch_terms_matched": matched,
            "refused_targets": len(self.refused),
            "languages": list(self.languages),
        }
