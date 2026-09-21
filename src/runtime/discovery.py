"""The desk's discovery organ: the hourly clock, wired to the live desk.

`PublicInternetEventCrawler` knows how to discover. `PublicWebSearcher` knows
how to fetch. Neither of them knows what the desk is currently looking at, and
neither runs unless something calls them -- which, in this repository's
history, is the failure mode that has cost more than every algorithmic mistake
combined. Four trainers, an export tool and an entire source architecture have
each been written correctly, wired to nothing, and discovered months later.

So this module is the wire. It is a mixin, for the same reason the other
runtime services are: the loop reads and writes the desk's own subsystems --
the launch stream, the source mesh, the edge ledger, the provider registry --
and injecting all of them into a collaborator would be a behaviour-changing
refactor wearing a tidy-up's clothes.

It runs three things on one long clock:

    SEED     every launch the desk has seen since the last pass becomes a
             seed: mint, ticker, name, creator, funders, domain, handles,
             launchpad. This is the chain -> internet direction, and it is
             driven by what the desk actually traded rather than by a static
             watchlist.

    CRAWL    one discovery cycle. Its output is a promotion: candidates that
             have produced enough material are handed to the continuous
             source mesh, where their next post arrives in seconds instead of
             up to an hour.

    WATCH    the provider terms pass, on a much slower sub-clock, because a
             pricing page does not change hourly and reading it hourly is how
             a polite crawler becomes an impolite one.

The adversarial reserve is seeded from the desk's own embarrassments: tokens
it declined that went on to run, and sources it scored badly. Those are
generated here rather than in the crawler because only the desk knows what it
got wrong.

Nothing in this loop can trade. It produces `InsiderEvent` rows with no
confidence attached, and they earn weight only through forward outcomes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.research.insider_event import LawfulAccess, SourceEdgeLedger
from src.research.provider_terms import ProviderTermsWatcher
from src.research.world_crawler import (PublicInternetEventCrawler, Seed,
                                        SeedKind)
from src.research.world_search import PublicWebSearcher

logger = logging.getLogger(__name__)

#: One discovery pass an hour. The continuous mesh is a different clock; this
#: one is for finding what did not exist an hour ago.
DEFAULT_DISCOVERY_INTERVAL_S = 3_600.0

#: Provider terms are read once a day. Reading a pricing page every hour finds
#: nothing new and looks like abuse from the other end.
DEFAULT_TERMS_INTERVAL_S = 86_400.0

#: How many of the desk's recent misses to feed the adversarial reserve per
#: pass. Small on purpose: the reserve is a budget, not a backlog drain.
DEFAULT_ADVERSARIAL_PER_PASS = 5


class DeskDiscovery:
    """The desk's side of world discovery: what to seed, and when to run."""

    def _setup_discovery(self) -> None:
        """Construct the crawler, its searcher, and the ledger it feeds.

        Called from subsystem wiring. Offline desks get the crawler with no
        searcher, which reports DATA_BLOCKED per pass rather than pretending
        to have looked.
        """
        config = getattr(self, "global_config", {}) or {}
        self.world_searcher = (
            None if getattr(self, "offline", False)
            or getattr(self, "http_client", None) is None
            else PublicWebSearcher(self.http_client))
        self.world_crawler = PublicInternetEventCrawler(
            searcher=self.world_searcher,
            query_budget=int(config.get("discovery_query_budget", 120)),
            adversarial_reserve=float(
                config.get("discovery_adversarial_reserve", 0.25)),
            promotion_observations=int(
                config.get("discovery_promotion_observations", 3)),
            languages=tuple(config.get("discovery_languages", ()) or ()))
        self.source_edges = SourceEdgeLedger(
            min_cell_events=int(config.get("source_edge_min_events", 12)))
        self.provider_watcher = ProviderTermsWatcher(
            fetcher=None if getattr(self, "offline", False)
            else self._fetch_provider_page,
            state_path=self._discovery_state_path())
        self._last_terms_pass = 0.0
        self._discovery_seeded_tokens: set = set()
        self._discovery_last: Dict[str, Any] = {"status": "NOT_RUN"}

    def _discovery_state_path(self):
        base = getattr(self, "data_dir", None)
        if base is None:
            return None
        from pathlib import Path
        return Path(base) / "provider_terms.json"

    def _fetch_provider_page(self, url: str):
        """Synchronous adapter over the desk's one HTTP client.

        The terms watcher is synchronous by design -- four requests a day have
        no reason to be concurrent -- and it is therefore called from a worker
        thread, never from the event loop. From that thread the coroutine is
        submitted back to the loop and waited on.

        The obvious version of this (`run_until_complete` on the running loop)
        raises "this event loop is already running" every single time, and the
        watcher would have reported DATA_BLOCKED forever while looking wired.
        """
        client = getattr(self, "http_client", None)
        if client is None:
            raise RuntimeError("no http client")
        loop = getattr(self, "_discovery_loop_ref", None)
        if loop is None or not loop.is_running():
            raise RuntimeError("no running loop to submit the fetch to")
        future = asyncio.run_coroutine_threadsafe(client.get(url), loop)
        status, body, _ = future.result(timeout=30.0)
        return int(status), body if isinstance(body, str) else str(body or "")

    # -- promotion targets ------------------------------------------------

    #: Which discovered platforms the transport layer can actually serve, and
    #: with what declaration. A platform absent here is refused BY NAME rather
    #: than registered into a mesh that would report it NO_FETCHER forever.
    _PROMOTABLE_PLATFORMS = {
        "telegram": ("telegram", "channel", 60.0),
        "github": ("code_repo", "repo", 900.0),
    }

    def _register_discovered_source(self, candidate: Any) -> Tuple[bool, str]:
        """Build a real transport and source for a discovered candidate.

        Everything a declared source gets, a discovered one gets: a
        declaration, a built transport, a constructed adapter, and a place in
        the same mesh. A "promotion" that only set a flag would leave the
        channel exactly as unread as it was.
        """
        from src.collectors.registry import SourceDeclaration, build_sources
        from src.collectors.transports import build_transports

        mesh = getattr(self, "source_mesh", None)
        if mesh is None:
            return False, "no source mesh"
        mapping = self._PROMOTABLE_PLATFORMS.get(candidate.platform)
        if mapping is None:
            return False, (f"platform {candidate.platform!r} has no keyless "
                           "transport; recorded but not promoted")
        kind, option, cadence = mapping
        source_id = f"{kind}:discovered-{candidate.identifier}"
        if any(getattr(item, "source_id", "") == source_id
               for item in mesh.sources):
            return False, "already in the mesh"
        declaration = SourceDeclaration(
            source_id=source_id, kind=kind, tier=3,
            options={option: candidate.identifier},
            poll_interval_seconds=cadence,
            poll_timeout_seconds=float(getattr(self, "global_config", {}).get(
                "discovered_source_timeout", 20.0)))
        transports, _report, self.http_client = build_transports(
            [declaration], getattr(self, "http_client", None))
        if source_id not in transports:
            return False, "transport could not be built"
        sources, _registry = build_sources([declaration], transports)
        if not sources:
            return False, "source could not be constructed"
        for source in sources:
            mesh.add(source)
        self.source_fetchers[source_id] = transports[source_id]
        logger.info("DISCOVERY promoted %s into the continuous mesh",
                    source_id)
        return True, ""

    # -- seeding from the desk's own experience --------------------------

    def _discovery_seed_launches(self, limit: int = 200) -> int:
        """Everything the desk has seen since the last pass becomes a query."""
        crawler = getattr(self, "world_crawler", None)
        if crawler is None:
            return 0
        added = 0
        for token, record in list(self._discovery_launch_source())[:limit]:
            if token in self._discovery_seeded_tokens:
                continue
            self._discovery_seeded_tokens.add(token)
            added += crawler.seed_from_chain(
                mint=token,
                ticker=str(record.get("symbol", "") or ""),
                name=str(record.get("name", "") or ""),
                creator=str(record.get("creator", record.get("deployer", ""))
                            or ""),
                funders=[str(item) for item in
                         (record.get("funders") or [])][:4],
                domain=str(record.get("website", "") or ""),
                handles=[str(item) for item in
                         (record.get("socials") or [])][:4],
                launchpad=str(record.get("factory", "") or ""))
            # And the other direction: a mint appearing now may match an
            # entity the crawler registered days ago from a web page, which
            # is the pre-launch lead the whole second direction exists for.
            hits = crawler.match_new_mint(
                mint=token,
                ticker=str(record.get("symbol", "") or ""),
                name=str(record.get("name", "") or ""),
                domain=str(record.get("website", "") or ""),
                handles=[str(item) for item in (record.get("socials") or [])])
            for hit in hits:
                logger.info(
                    "PRE-LAUNCH LEAD %s seen on the web %.0fs before mint %s "
                    "(%s)", hit.term, hit.lead_seconds or 0.0, token,
                    hit.origin_url)
        return added

    def _discovery_launch_source(self):
        """Recent launches as (token, record) pairs, from whatever exists.

        Read defensively: this loop must not be the reason a desk fails to
        start because a collaborator changed shape.
        """
        builder = getattr(self, "dataset_builder", None)
        episodes = getattr(builder, "episodes", None) if builder else None
        if isinstance(episodes, dict):
            for token, episode in list(episodes.items()):
                record = episode if isinstance(episode, dict) else {}
                yield token, record
            return
        tracked = getattr(self, "tracked_tokens", None)
        if isinstance(tracked, dict):
            for token, record in list(tracked.items()):
                yield token, (record if isinstance(record, dict) else {})

    def _discovery_seed_adversarial(self) -> int:
        """The desk's own misses, fed back as things to go and look at.

        A crawler that only expands around what the model liked will confirm
        the model. These seeds exist to contradict it: launches it declined
        that ran anyway, and sources it scored badly.
        """
        crawler = getattr(self, "world_crawler", None)
        if crawler is None:
            return 0
        seeds: List[Seed] = []
        now = time.time()
        for token in self._discovery_missed_tokens()[
                :DEFAULT_ADVERSARIAL_PER_PASS]:
            seeds.append(Seed(kind=SeedKind.MINT, value=str(token),
                              origin="declined_then_ran", first_seen=now))
        for term in ("new solana launchpad", "new bundler solana",
                     "solana sniper bot github")[:2]:
            seeds.append(Seed(kind=SeedKind.NARRATIVE, value=term,
                              origin="adversarial_standing", first_seen=now))
        return crawler.seed_adversarial(seeds)

    def _discovery_missed_tokens(self) -> List[str]:
        """Tokens the desk declined that subsequently ran.

        Returns an empty list rather than raising when the counterfactual
        corpus is not available: a missing embarrassment record is a reason to
        have fewer adversarial seeds, not to skip the pass.
        """
        corpus = getattr(self, "counterfactual_corpus", None)
        finder = getattr(corpus, "declined_winners", None) if corpus else None
        if finder is None:
            return []
        try:
            return [str(item) for item in (finder() or [])]
        except Exception as exc:
            logger.debug("adversarial seed source unavailable: %s", exc)
            return []

    # -- promotion into the continuous mesh ------------------------------

    def _discovery_promote(self, report: Dict[str, Any]) -> int:
        """Hand produced-enough candidates to the mesh, then mark them.

        Marking happens only for the ones the mesh accepted. A candidate
        recorded as promoted but never registered would never be offered
        again, and would silently drop out of both systems.
        """
        crawler = getattr(self, "world_crawler", None)
        mesh = getattr(self, "source_mesh", None)
        if crawler is None:
            return 0
        candidates = crawler.promotable()
        if not candidates:
            return 0
        if mesh is None:
            # Nowhere to promote to yet. Leave them promotable so a later
            # pass, on a desk with a mesh, still gets them.
            report["promotion"] = "no mesh accepting discovered sources"
            return 0
        accepted: List[str] = []
        refused: List[str] = []
        for candidate in candidates:
            try:
                ok, reason = self._register_discovered_source(candidate)
            except Exception as exc:
                ok, reason = False, f"{type(exc).__name__}: {exc}"
            if ok:
                accepted.append(candidate.identifier)
            else:
                refused.append(f"{candidate.identifier}: {reason}")
                # Marked promoted anyway when the refusal is permanent, so a
                # platform with no transport is not re-offered every hour
                # forever; it stays in `candidates` with its reason recorded.
                if "no keyless transport" in reason or "already in" in reason:
                    candidate.refused_reason = reason
                    accepted.append(candidate.identifier)
        if refused:
            report["promotion_refused"] = refused[:20]
        return crawler.mark_promoted(accepted)

    # -- the loop --------------------------------------------------------

    async def _discovery_loop(self):
        """Hourly world discovery, on its own clock, deciding nothing.

        Deliberately started after the health server binds and never before:
        a discovery pass is dozens of HTTP requests and must not sit between
        process start and the port being answerable.
        """
        config = getattr(self, "global_config", {}) or {}
        # Captured here rather than in setup: setup runs before the loop
        # exists, and the terms fetcher needs a RUNNING loop to submit to.
        self._discovery_loop_ref = asyncio.get_running_loop()
        interval = float(config.get("discovery_interval_seconds",
                                    DEFAULT_DISCOVERY_INTERVAL_S))
        terms_interval = float(config.get("provider_terms_interval_seconds",
                                          DEFAULT_TERMS_INTERVAL_S))
        while self._running:
            await asyncio.sleep(interval)
            if getattr(self, "offline", False):
                continue
            try:
                report = await self._discovery_pass(
                    terms_interval=terms_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("discovery pass error: %s", exc)
                self._discovery_last = {"status": "ERROR", "error": str(exc)}
                continue
            self._discovery_last = report
            if report.get("status") == "OK":
                logger.info(
                    "DISCOVERY cycle %s: %d queries (%d adversarial), %d "
                    "findings, %d new candidates, %d watch terms, %d promoted",
                    report.get("cycle"), report.get("queries", 0),
                    report.get("adversarial_queries", 0),
                    report.get("findings", 0),
                    report.get("new_candidates", 0),
                    report.get("new_watch_terms", 0),
                    report.get("promoted", 0))

    async def _discovery_pass(self, *, terms_interval: float
                              ) -> Dict[str, Any]:
        """One pass: seed, crawl, promote, and occasionally read the terms."""
        crawler = getattr(self, "world_crawler", None)
        if crawler is None:
            return {"status": "DATA_BLOCKED", "reason": "crawler not wired"}

        seeded = self._discovery_seed_launches()
        adversarial = self._discovery_seed_adversarial()
        report = await crawler.acycle()
        report["seeded_from_chain"] = seeded
        report["seeded_adversarial"] = adversarial
        report["promoted"] = self._discovery_promote(report)
        report["pre_launch_leads"] = len(crawler.pre_launch_leads())

        # Every finding becomes a canonical event, immediately. They carry no
        # market context yet -- price, liquidity and forward returns are
        # attached later by the outcome path -- and that is the honest state:
        # `SourceEdgeLedger.cells()` groups only on horizons an event actually
        # has, so a contextless row is counted in the corpus and scores
        # nothing. Withholding them until they are complete would mean the
        # claim's ORIGINAL timestamp is lost, which is the one field that
        # cannot be reconstructed afterwards and the only reason to record a
        # claim at all.
        ledger = getattr(self, "source_edges", None)
        if ledger is not None and crawler.recent_findings:
            events = crawler.to_events(crawler.recent_findings,
                                       access=LawfulAccess.PUBLIC)
            outcome = ledger.extend(events)
            report["events_emitted"] = outcome["accepted"]
            report["events_refused"] = outcome["refused"]

        now = time.time()
        if now - self._last_terms_pass >= terms_interval:
            self._last_terms_pass = now
            watcher = getattr(self, "provider_watcher", None)
            if watcher is not None:
                # Off the loop: the watcher is synchronous and makes network
                # calls, and running it inline would block every other loop
                # for the duration of four page fetches.
                terms = await asyncio.to_thread(watcher.poll, now=now)
                report["provider_terms"] = terms
                for change in terms.get("breaking", []):
                    logger.warning(
                        "PROVIDER TERMS BREAKING %s %s: %s -> %s (%s)",
                        change["provider"], change["field"], change["before"],
                        change["after"], change["detail"])
        return report

    # -- reporting -------------------------------------------------------

    def world_discovery_report(self) -> Dict[str, Any]:
        """Named `world_` deliberately: `ReportingSurface.discovery_report`
        already exists for Telegram channel discovery, comes first in the
        desk's MRO, and would silently shadow this one -- the health surface
        would then have shown channel discovery twice and the world crawler
        never, with nothing failing anywhere."""
        crawler = getattr(self, "world_crawler", None)
        searcher = getattr(self, "world_searcher", None)
        ledger = getattr(self, "source_edges", None)
        watcher = getattr(self, "provider_watcher", None)
        return {
            "last_pass": dict(getattr(self, "_discovery_last", {})),
            "crawler": crawler.status() if crawler else None,
            "search": searcher.report() if searcher else
            {"status": "DATA_BLOCKED", "reason": "no searcher"},
            "source_edges": ledger.summary() if ledger else None,
            "providers": watcher.status("solana") if watcher else None,
            "pre_launch_leads": (crawler.pre_launch_leads()[:20]
                                 if crawler else []),
        }
