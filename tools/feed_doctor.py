"""Decide whether a feed URL is really a feed, and find a live one when it is not.

A dead feed rarely announces itself. Three of this desk's regional sources
answered 410, 404 and 403 -- honest failures -- but two answered **HTTP 200**
with the publisher's single-page-app shell: a soft-404 that a status check
counts as success and a tolerant XML parser would happily ingest as news. So
"reachable" is not the test. The test is whether the body parses as RSS or
Atom and carries dated items.

Discovery is deliberately conservative. It probes the publisher's own domain
first -- conventional feed paths, then `<link rel=alternate>` autodiscovery --
because a replacement from the same publisher preserves the editorial voice
and the language the declaration promises. A curated regional fallback is
offered only as a labelled suggestion, never silently substituted: swapping a
Chinese source for an English one because both were reachable would quietly
change what the mesh covers while reporting full health.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

USER_AGENT = "memecoin-shadow-feed-doctor/1.0 (+research; contact via operator)"

#: Paths publishers actually use, cheapest and most conventional first.
CANDIDATE_PATHS: Tuple[str, ...] = (
    "/feed", "/rss", "/rss.xml", "/feed.xml", "/atom.xml", "/index.xml",
    "/feed/", "/rss/", "/en/rss", "/zh/rss", "/ja/rss", "/ko/rss",
    "/rss/all.xml", "/feeds/posts/default", "/api/rss", "/news/rss",
    "/rss/news.xml", "/feed/rss", "/blog/feed", "/?feed=rss2",
)

#: Regional sources kept as *suggestions* when a publisher has no live feed at
#: all. Same language and beat as the sources this desk already declares, so a
#: swap does not silently narrow coverage to English.
#: Keyed by the language codes the declarations actually use (`zh`, not
#: `zh-cn`) -- a fallback table keyed on a code no declaration carries returns
#: nothing and looks like "no alternative exists".
#: Every entry below was probed and returned a parsing feed with items. Ones
#: that did not are recorded as comments so the next person does not re-probe
#: them: jinse.cn and 8btc.com no longer resolve, theblockbeats and
#: beincrypto/es answer 403, es.cointelegraph answers 410, and both
#: foresightnews.pro and techflowpost serve an app shell or a 404.
REGIONAL_FALLBACKS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "zh": (
        ("https://rss.odaily.news/rss/newsflash", "Odaily newsflash"),
        ("https://rss.odaily.news/rss/post", "Odaily posts"),
        ("https://www.panewslab.com/rss.xml", "PANews"),
        ("https://www.chaincatcher.com/rss/clist", "ChainCatcher (clist)"),
        ("https://www.chaincatcher.com/rss.xml", "ChainCatcher"),
        ("https://www.blocktempo.com/feed/", "BlockTempo"),
        ("https://www.abmedia.io/feed", "ABMedia"),
        ("https://zombit.info/feed/", "Zombit"),
    ),
    "ja": (
        ("https://coinpost.jp/?feed=rss2", "CoinPost"),
        ("https://www.neweconomy.jp/feed", "NeweConomy"),
        ("https://cointelegraph.com/rss/tag/japan", "Cointelegraph Japan tag"),
        ("https://bittimes.net/feed", "BitTimes"),
    ),
    "ko": (
        ("https://www.tokenpost.kr/rss", "TokenPost KR"),
        ("https://www.blockmedia.co.kr/feed", "BlockMedia"),
    ),
    "es": (
        ("https://diariobitcoin.com/feed/", "DiarioBitcoin"),
    ),
    "en": (
        ("https://cointelegraph.com/rss", "Cointelegraph"),
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
        ("https://decrypt.co/feed", "Decrypt"),
    ),
}

FEED_ROOTS = {"rss", "feed", "rdf"}
ITEM_TAGS = {"item", "entry"}


@dataclass
class Verdict:
    url: str
    ok: bool
    reason: str
    status: Optional[int] = None
    items: int = 0
    title: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "ok": self.ok, "reason": self.reason,
                "status": self.status, "items": self.items, "title": self.title}


@dataclass
class Diagnosis:
    source_id: str
    declared_url: str
    verdict: Verdict
    replacements: List[Verdict] = field(default_factory=list)
    suggestions: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "declared_url": self.declared_url,
            "verdict": self.verdict.as_dict(),
            "replacements": [item.as_dict() for item in self.replacements],
            "suggestions": [{"url": url, "name": name}
                            for url, name in self.suggestions],
        }


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def classify(body: bytes, status: Optional[int]) -> Verdict:
    """Is this body a feed with items? The soft-404 check lives here.

    Called with the URL blank; the caller fills it in. Kept separate from the
    fetch so it is testable without a network.
    """
    if status is not None and status != 200:
        return Verdict("", False, f"HTTP {status}", status)
    if not body:
        return Verdict("", False, "empty body", status)

    head = body[:400].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        # The soft-404: 200 OK, and an app shell where the feed used to be.
        return Verdict("", False, "HTTP 200 but body is HTML, not a feed", status)

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return Verdict("", False, f"does not parse as XML: {exc}", status)

    if _localname(root.tag) not in FEED_ROOTS:
        return Verdict("", False, f"XML root is <{_localname(root.tag)}>, not a feed",
                       status)

    items = [node for node in root.iter() if _localname(node.tag) in ITEM_TAGS]
    if not items:
        return Verdict("", False, "feed parses but carries no items", status)

    title = ""
    for node in root.iter():
        if _localname(node.tag) == "title" and (node.text or "").strip():
            title = (node.text or "").strip()[:80]
            break
    return Verdict("", True, "ok", status, len(items), title)


async def probe(session: Any, url: str, timeout: float = 12.0) -> Verdict:
    try:
        async with session.get(
            url, timeout=timeout, allow_redirects=True,
            headers={"User-Agent": USER_AGENT,
                     "Accept": "application/rss+xml, application/atom+xml,"
                               " application/xml;q=0.9, */*;q=0.5"},
        ) as resp:
            body = await resp.read()
            verdict = classify(body, resp.status)
    except asyncio.TimeoutError:
        verdict = Verdict("", False, "timed out")
    except Exception as exc:  # a probe must never raise into the caller
        verdict = Verdict("", False, f"{type(exc).__name__}: {exc}")
    verdict.url = url
    return verdict


async def autodiscover(session: Any, page_url: str) -> List[str]:
    """Read <link rel=alternate type=...rss|atom> off the publisher's page."""
    try:
        async with session.get(
            page_url, timeout=12.0, allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as resp:
            if resp.status != 200:
                return []
            body = (await resp.read())[:400_000].decode("utf-8", "replace")
    except Exception:
        return []

    found: List[str] = []
    for match in re.finditer(r"<link\b[^>]*>", body, re.IGNORECASE):
        tag = match.group(0)
        if "alternate" not in tag.lower():
            continue
        if not re.search(r"type=[\"']application/(rss|atom)\+xml", tag, re.IGNORECASE):
            continue
        href = re.search(r"href=[\"']([^\"']+)[\"']", tag, re.IGNORECASE)
        if href:
            found.append(urljoin(page_url, href.group(1)))
    return list(dict.fromkeys(found))


async def find_replacement(session: Any, url: str,
                           limit: int = 6) -> List[Verdict]:
    """Live feeds on the SAME publisher, best first. Never leaves the domain."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return []
    origin = f"{parts.scheme}://{parts.netloc}"

    candidates: List[str] = []
    for path in CANDIDATE_PATHS:
        candidate = origin + path
        if candidate.rstrip("/") != url.rstrip("/"):
            candidates.append(candidate)
    candidates.extend(await autodiscover(session, origin))

    seen: set = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]

    working: List[Verdict] = []
    # Probed in small waves so a doctor run cannot itself look like a scrape.
    for start in range(0, len(ordered), 6):
        wave = ordered[start:start + 6]
        for verdict in await asyncio.gather(*(probe(session, c) for c in wave)):
            if verdict.ok:
                working.append(verdict)
                if len(working) >= limit:
                    return working
    return working


async def verify_suggestions(session: Any,
                             suggestions: Sequence[Tuple[str, str]],
                             ) -> List[Tuple[str, str]]:
    """Keep only fallbacks that are live right now.

    A suggestion list that has not been probed is a list of guesses. Offering
    a dead alternative to replace a dead source wastes the operator's next
    twenty minutes and teaches them not to trust the tool.
    """
    if not suggestions:
        return []
    verdicts = await asyncio.gather(
        *(probe(session, url) for url, _ in suggestions))
    return [(url, name) for (url, name), verdict in zip(suggestions, verdicts)
            if verdict.ok]


def unhealthy_from_status(payload: Dict[str, Any]) -> List[str]:
    """Source ids the running desk currently reports as DEAD or DEGRADED."""
    mesh = payload.get("source_mesh") or {}
    ids: List[str] = []
    for row in mesh.get("unhealthy") or ():
        if isinstance(row, dict) and row.get("source_id"):
            ids.append(str(row["source_id"]))
    return ids


def render_overlay(repairs: Sequence[Tuple[str, str, str, str]]) -> str:
    """A source overlay carrying only same-publisher URL corrections.

    Emitted under the ORIGINAL source_id so the registry's id-keyed merge
    replaces the stale declaration rather than adding a second one alongside
    it. Cross-publisher candidates are deliberately absent: substituting a
    different outlet changes what the mesh covers, and that is an editorial
    decision, not a repair.
    """
    lines = [
        "# GENERATED by tools/feed_doctor.py --apply. Do not hand-edit.",
        "# Same-publisher URL corrections only: each entry keeps its original",
        "# source_id, language and region, and changes nothing but a path the",
        "# publisher moved. Regenerate rather than editing.",
        "sources:",
    ]
    for source_id, old_url, new_url, language in repairs:
        lines.append(f"  # was: {old_url}")
        lines.append(f"  - id: {source_id}")
        lines.append("    kind: rss")
        if language:
            lines.append(f"    language: {language}")
        lines.append(f'    options: {{url: "{new_url}"'
                     + (f", language: {language}" if language else "") + "}")
    return "\n".join(lines) + "\n"


def load_sources(paths: Sequence[str]) -> List[Tuple[str, str, str]]:
    """(source_id, url, language) for every rss declaration, overlay last."""
    import yaml
    merged: Dict[str, Tuple[str, str, str]] = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except OSError:
            continue
        for entry in raw.get("sources") or []:
            if not isinstance(entry, dict) or entry.get("kind") != "rss":
                continue
            url = (entry.get("options") or {}).get("url", "")
            if url:
                merged[str(entry["id"])] = (
                    str(entry["id"]), str(url), str(entry.get("language", "")))
    return list(merged.values())


async def diagnose(targets: Sequence[Tuple[str, str, str]], *,
                   repair: bool, verify: bool = True) -> List[Diagnosis]:
    import aiohttp
    results: List[Diagnosis] = []
    connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        for source_id, url, language in targets:
            verdict = await probe(session, url)
            diagnosis = Diagnosis(source_id, url, verdict)
            if not verdict.ok and repair:
                diagnosis.replacements = await find_replacement(session, url)
                if not diagnosis.replacements:
                    declared = {t[1].rstrip("/") for t in targets}
                    candidates = [
                        item for item in REGIONAL_FALLBACKS.get(language, ())
                        # Never propose a feed this mesh already polls: it
                        # would read as new coverage while adding none.
                        if item[0].rstrip("/") not in declared]
                    diagnosis.suggestions = (
                        await verify_suggestions(session, candidates)
                        if verify else candidates)
            results.append(diagnosis)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="check declared RSS sources and find live replacements")
    parser.add_argument("--sources",
                        default="config/sources.yaml,config/sources.verified.yaml")
    parser.add_argument("--url", action="append", default=[],
                        help="check one URL instead of the declared registry")
    parser.add_argument("--language", default="",
                        help="language hint for --url, used for suggestions")
    parser.add_argument("--only-broken", action="store_true",
                        help="report only sources that failed")
    parser.add_argument("--repair", action="store_true",
                        help="search the publisher for a live feed")
    parser.add_argument("--from-status", default="",
                        help="check only what the running desk calls unhealthy, "
                             "e.g. http://127.0.0.1:18080/status")
    parser.add_argument("--apply", default="",
                        help="write same-publisher corrections to this overlay "
                             "path; cross-publisher candidates are never applied")
    parser.add_argument("--no-verify-suggestions", action="store_true",
                        help="skip probing fallbacks before offering them")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.url:
        targets = [(url, url, args.language) for url in args.url]
    else:
        targets = load_sources([p.strip() for p in args.sources.split(",") if p.strip()])

    if args.from_status:
        import urllib.request
        try:
            with urllib.request.urlopen(args.from_status, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"could not read {args.from_status}: {exc}", file=sys.stderr)
            return 2
        wanted = set(unhealthy_from_status(payload))
        # An empty unhealthy list is the good case, not a reason to sweep the
        # whole registry: a full probe on every tick is what gets a doctor
        # rate-limited by the publishers it exists to protect.
        targets = [t for t in targets if t[0] in wanted]
        if not targets:
            print("the desk reports no unhealthy rss sources")
            return 0

    if not targets:
        print("no rss sources to check", file=sys.stderr)
        return 2

    results = asyncio.run(diagnose(targets, repair=args.repair,
                                   verify=not args.no_verify_suggestions))

    if args.apply:
        repairs = [(d.source_id, d.declared_url, d.replacements[0].url,
                    dict((t[0], t[2]) for t in targets).get(d.source_id, ""))
                   for d in results if not d.verdict.ok and d.replacements]
        if repairs:
            with open(args.apply, "w", encoding="utf-8") as handle:
                handle.write(render_overlay(repairs))
            print(f"wrote {len(repairs)} same-publisher correction(s) "
                  f"to {args.apply}")
            for source_id, old, new, _ in repairs:
                print(f"  {source_id}: {old} -> {new}")
        else:
            print("no same-publisher corrections to apply")
    broken = [d for d in results if not d.verdict.ok]
    shown = broken if args.only_broken else results

    if args.json:
        print(json.dumps([d.as_dict() for d in shown], indent=1, ensure_ascii=False))
        return 1 if broken else 0

    for diagnosis in shown:
        mark = "ok  " if diagnosis.verdict.ok else "DEAD"
        detail = (f"{diagnosis.verdict.items} items" if diagnosis.verdict.ok
                  else diagnosis.verdict.reason)
        print(f"[{mark}] {diagnosis.source_id}\n       {diagnosis.declared_url}"
              f"\n       {detail}")
        for replacement in diagnosis.replacements:
            print(f"       -> LIVE on same publisher: {replacement.url} "
                  f"({replacement.items} items) {replacement.title}")
        for url, name in diagnosis.suggestions:
            print(f"       ?  same-language candidate: {url}  [{name}]")
        print()
    print(f"{len(results) - len(broken)}/{len(results)} declared feeds are live")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
