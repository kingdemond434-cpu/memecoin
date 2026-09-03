"""Proof that the discovery organ is actually connected.

This repository's most expensive recurring bug is not an algorithm; it is a
correct module wired to nothing. Four trainers, an export tool and an entire
source architecture each shipped, ran never, and were found months later. So
the crawler gets a test that fails if it stops being called, one that fails if
promotion stops registering a real transport, and one that fails if the terms
watcher goes back to deadlocking on the running loop.
"""

import asyncio
import inspect

import pytest

from src.collectors.event_source import SourceMesh
from src.research.world_crawler import (Finding, PublicInternetEventCrawler,
                                        Seed, SeedKind, SourceCandidate)
from src.runtime.discovery import DeskDiscovery


class FakeDesk(DeskDiscovery):
    """The smallest desk the discovery mixin can run against."""

    def __init__(self, *, offline=False, mesh=None, launches=None):
        self.offline = offline
        self.global_config = {}
        self.http_client = object() if not offline else None
        self.data_dir = None
        self.source_mesh = mesh
        self.source_fetchers = {}
        self.tracked_tokens = dict(launches or {})
        self._running = True
        self._setup_discovery()


# --- the loop exists and is started --------------------------------------

def test_the_desk_class_carries_the_discovery_mixin():
    from src.main import MemecoinQuantDesk
    assert issubclass(MemecoinQuantDesk, DeskDiscovery)
    assert hasattr(MemecoinQuantDesk, "_discovery_loop")


def test_main_actually_starts_the_discovery_loop():
    import pathlib
    source = pathlib.Path("src/main.py").read_text()
    assert "self._discovery_loop()" in source, (
        "the discovery loop is defined but never scheduled -- this is the "
        "exact orphaning failure the module exists to avoid")
    assert '"discovery"' in source


def test_wiring_constructs_the_crawler():
    import pathlib
    source = pathlib.Path("src/runtime/wiring.py").read_text()
    assert "_setup_discovery()" in source


def test_setup_builds_every_collaborator():
    desk = FakeDesk()
    assert desk.world_crawler is not None
    assert desk.source_edges is not None
    assert desk.provider_watcher is not None
    assert desk.world_searcher is not None


def test_an_offline_desk_has_no_searcher_and_says_so():
    desk = FakeDesk(offline=True)
    assert desk.world_searcher is None
    assert desk.world_crawler.cycle()["status"] == "DATA_BLOCKED"
    assert desk.world_discovery_report()["search"]["status"] == "DATA_BLOCKED"


# --- seeding from the desk's own experience ------------------------------

def test_launches_the_desk_saw_become_seeds():
    desk = FakeDesk(launches={
        "Mint1": {"symbol": "WIF", "name": "dogwifhat", "deployer": "Dep1",
                  "funders": ["F1"], "website": "wif.io",
                  "socials": ["wifcoin"], "factory": "pump"}})
    assert desk._discovery_seed_launches() >= 7
    values = {seed.value for seed in desk.world_crawler.queue}
    assert {"Mint1", "WIF", "dogwifhat", "Dep1"} <= values


def test_a_launch_is_seeded_once_however_often_it_is_seen():
    desk = FakeDesk(launches={"Mint1": {"symbol": "WIF"}})
    first = desk._discovery_seed_launches()
    assert first > 0
    assert desk._discovery_seed_launches() == 0


def test_seeding_survives_a_collaborator_with_the_wrong_shape():
    desk = FakeDesk()
    desk.tracked_tokens = {"Mint1": "not a dict"}
    assert desk._discovery_seed_launches() >= 0   # no raise


def test_the_adversarial_reserve_is_seeded_even_with_no_misses_recorded():
    desk = FakeDesk()
    # No counterfactual corpus attached, so no declined-winners exist; the
    # standing adversarial searches must still be queued, or the reserve
    # quietly becomes unspent on every real desk.
    assert desk._discovery_seed_adversarial() > 0
    assert desk.world_crawler.adversarial_queue


def test_a_raising_counterfactual_corpus_does_not_end_the_pass():
    desk = FakeDesk()

    class Corpus:
        def declined_winners(self):
            raise RuntimeError("no corpus")

    desk.counterfactual_corpus = Corpus()
    assert desk._discovery_missed_tokens() == []
    assert desk._discovery_seed_adversarial() > 0


# --- internet -> chain, end to end --------------------------------------

def test_a_web_entity_registered_first_produces_a_lead_when_the_mint_lands():
    desk = FakeDesk()
    desk.world_crawler.register_watch_term(
        "moonproject.io", SeedKind.DOMAIN,
        origin_url="https://forum.test/t", now=1_000.0)
    desk.tracked_tokens = {"MintX": {"website": "moonproject.io",
                                     "symbol": "MOON"}}
    desk._discovery_seed_launches()
    leads = desk.world_crawler.pre_launch_leads()
    assert len(leads) == 1
    assert leads[0]["mint"] == "MintX"
    assert leads[0]["lead_seconds"] > 0


# --- promotion is real --------------------------------------------------

def _candidate(platform, identifier, url):
    return SourceCandidate(identifier=identifier, platform=platform, url=url,
                           observations=5)


def test_promoting_a_telegram_channel_adds_a_real_source_to_the_mesh():
    mesh = SourceMesh([])
    desk = FakeDesk(mesh=mesh)
    ok, reason = desk._register_discovered_source(
        _candidate("telegram", "alphadesk", "https://t.me/s/alphadesk"))
    assert ok, reason
    ids = [source.source_id for source in mesh.sources]
    assert "telegram:discovered-alphadesk" in ids
    # And the transport exists, so the mesh will not report it NO_FETCHER.
    assert "telegram:discovered-alphadesk" in desk.source_fetchers


def test_promoting_the_same_channel_twice_is_refused_not_duplicated():
    mesh = SourceMesh([])
    desk = FakeDesk(mesh=mesh)
    candidate = _candidate("telegram", "alphadesk", "https://t.me/s/alphadesk")
    assert desk._register_discovered_source(candidate)[0]
    ok, reason = desk._register_discovered_source(candidate)
    assert not ok and "already in the mesh" in reason
    assert len(mesh.sources) == 1


def test_a_platform_with_no_keyless_transport_is_refused_by_name():
    desk = FakeDesk(mesh=SourceMesh([]))
    ok, reason = desk._register_discovered_source(
        _candidate("x", "some_dev", "https://x.com/some_dev"))
    assert not ok
    assert "no keyless transport" in reason


def test_promotion_without_a_mesh_leaves_candidates_promotable():
    desk = FakeDesk(mesh=None)
    desk.world_crawler.candidates["telegram:a"] = _candidate(
        "telegram", "a", "https://t.me/s/a")
    report = {}
    assert desk._discovery_promote(report) == 0
    assert desk.world_crawler.promotable()
    assert "no mesh" in report["promotion"]


def test_an_unpromotable_platform_is_not_re_offered_every_hour():
    desk = FakeDesk(mesh=SourceMesh([]))
    desk.world_crawler.candidates["x:dev"] = _candidate(
        "x", "dev", "https://x.com/dev")
    report = {}
    desk._discovery_promote(report)
    assert desk.world_crawler.promotable() == []
    assert report["promotion_refused"]


# --- the full pass ------------------------------------------------------

def test_a_pass_seeds_crawls_promotes_and_emits_events():
    mesh = SourceMesh([])
    desk = FakeDesk(mesh=mesh,
                    launches={"Mint1": {"symbol": "WIF", "name": "wif"}})

    async def searcher(query):
        return [Finding(url="https://t.me/s/alphadesk",
                        title="alpha", snippet="")]

    desk.world_crawler.searcher = searcher
    report = asyncio.run(desk._discovery_pass(terms_interval=10 ** 9))
    assert report["status"] == "OK"
    assert report["seeded_from_chain"] > 0
    assert report["seeded_adversarial"] > 0
    assert report["queries"] > 0
    assert report["events_emitted"] > 0
    assert report["promoted"] >= 1
    assert "telegram:discovered-alphadesk" in [
        source.source_id for source in mesh.sources]


def test_emitted_events_reach_the_ledger_and_score_nothing_yet():
    desk = FakeDesk()

    async def searcher(query):
        return [Finding(url="https://t.me/s/chan", title="call")]

    desk.world_crawler.searcher = searcher
    desk.world_crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    asyncio.run(desk._discovery_pass(terms_interval=10 ** 9))
    summary = desk.source_edges.summary()
    assert summary["events"] > 0
    # Discovery does not manufacture edge: no forward returns are attached
    # yet, so there is nothing to score.
    assert summary["cells_scored"] == 0


def test_a_pass_reports_data_blocked_rather_than_raising_without_a_crawler():
    desk = FakeDesk()
    desk.world_crawler = None
    report = asyncio.run(desk._discovery_pass(terms_interval=10 ** 9))
    assert report["status"] == "DATA_BLOCKED"


# --- the terms fetcher's loop discipline --------------------------------

def test_the_terms_fetcher_never_calls_run_until_complete():
    # The docstring names the trap on purpose, so check the body only.
    body = inspect.getsource(DeskDiscovery._fetch_provider_page)
    source = body.split('"""')[-1]
    assert "run_until_complete" not in source, (
        "run_until_complete on the desk's running loop raises every time; "
        "the watcher would report DATA_BLOCKED forever while looking wired")
    assert "run_coroutine_threadsafe" in source


def test_the_terms_fetcher_refuses_when_there_is_no_running_loop():
    desk = FakeDesk()
    with pytest.raises(RuntimeError) as caught:
        desk._fetch_provider_page("https://demo.test/pricing")
    assert "running loop" in str(caught.value)


def test_the_terms_pass_runs_off_the_event_loop():
    source = inspect.getsource(DeskDiscovery._discovery_pass)
    assert "asyncio.to_thread(watcher.poll" in source, (
        "a synchronous multi-page fetch inline on the loop blocks every "
        "other runtime loop for its duration")


def test_the_terms_fetcher_works_from_a_worker_thread():
    class Client:
        async def get(self, url, headers=None):
            return 200, "Free tier: 5,000,000 requests per month.", {}

    desk = FakeDesk()
    desk.http_client = Client()

    async def scenario():
        desk._discovery_loop_ref = asyncio.get_running_loop()
        return await asyncio.to_thread(
            desk._fetch_provider_page, "https://demo.test/pricing")

    status, body = asyncio.run(scenario())
    assert status == 200
    assert "5,000,000" in body


# --- reporting ----------------------------------------------------------

def test_the_world_report_does_not_collide_with_the_census_one():
    """`ReportingSurface.discovery_report` is the pool-census denominator and
    comes first in the desk's MRO. A world report sharing that name would be
    shadowed silently: the health surface would show channel discovery twice
    and the crawler never, and nothing anywhere would fail."""
    from src.main import MemecoinQuantDesk
    from src.runtime.reporting import ReportingSurface
    assert hasattr(MemecoinQuantDesk, "world_discovery_report")
    assert (MemecoinQuantDesk.discovery_report
            is ReportingSurface.discovery_report)
    assert (MemecoinQuantDesk.world_discovery_report
            is not ReportingSurface.discovery_report)


def test_the_discovery_report_is_shaped_for_the_health_surface():
    desk = FakeDesk()
    report = desk.world_discovery_report()
    assert set(report) >= {"last_pass", "crawler", "search", "source_edges",
                           "providers", "pre_launch_leads"}
    assert report["last_pass"]["status"] == "NOT_RUN"
    assert report["providers"]["usable"] == 0
