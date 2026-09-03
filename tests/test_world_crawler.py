"""The hourly world crawler.

Four things must hold or the crawler is theatre: it must search in the second
direction (internet -> chain), it must spend its adversarial budget even when
busy, it must refuse targets behind an access control, and it must attach no
confidence to anything it finds.
"""

import pytest

from src.research.insider_event import LawfulAccess, SourceEdgeLedger
from src.research.world_crawler import (
    AccessPolicy, Direction, EVENT_TERMS, Finding, PublicInternetEventCrawler,
    Query, Seed, SeedKind, classify_platform, expand_queries,
    extract_entities)

MINT = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


def _crawler(results=None, **kwargs):
    calls = []

    def search(query):
        calls.append(query)
        if callable(results):
            return results(query)
        return list(results or [])

    crawler = PublicInternetEventCrawler(searcher=search, **kwargs)
    crawler.calls = calls
    return crawler


# --- query expansion -----------------------------------------------------

def test_a_name_is_expanded_into_every_configured_language():
    seed = Seed(kind=SeedKind.TOKEN_NAME, value="Bonkinu")
    queries = expand_queries(seed)
    languages = {query.language for query in queries}
    assert languages == set(EVENT_TERMS)
    assert len(languages) >= 11
    assert any("上线" in query.text for query in queries)
    assert any("상장" in query.text for query in queries)
    assert any("листинг" in query.text for query in queries)


def test_an_address_is_not_decorated_with_translated_nouns():
    queries = expand_queries(Seed(kind=SeedKind.MINT, value=MINT))
    assert {query.language for query in queries} == {"en"}
    # The bare address is searched on its own -- the single most likely way to
    # find someone who posted it before the launch.
    assert any(query.text == MINT for query in queries)
    assert len(queries) < len(expand_queries(
        Seed(kind=SeedKind.TOKEN_NAME, value="Bonkinu")))


def test_an_empty_seed_expands_to_nothing():
    assert expand_queries(Seed(kind=SeedKind.TICKER, value="   ")) == []


def test_language_selection_narrows_the_plan():
    queries = expand_queries(Seed(kind=SeedKind.TICKER, value="WIF"),
                             languages=("ko", "ja"))
    assert {query.language for query in queries} == {"ko", "ja"}


# --- entity extraction ---------------------------------------------------

def test_handles_and_domains_come_out_of_a_page():
    found = extract_entities(
        "join https://t.me/alphacallers and https://x.com/dev_guy , "
        "site at https://moonproject.io/launch")
    assert "alphacallers" in found["handles"]
    assert "dev_guy" in found["handles"]
    assert "moonproject.io" in found["domains"]
    assert "t.me" in found["domains"]


def test_a_mint_inside_a_url_is_not_counted_twice():
    found = extract_entities(f"see https://solscan.io/token/{MINT}")
    assert found["mints"] == []
    assert any(MINT in url for url in found["urls"])


def test_a_bare_mint_in_prose_is_captured():
    found = extract_entities(f"CA: {MINT} buy now")
    assert found["mints"] == [MINT]


def test_platform_classification():
    assert classify_platform("https://t.me/s/whales") == ("telegram", "whales")
    assert classify_platform("https://x.com/some_dev") == ("x", "some_dev")
    assert classify_platform("https://reddit.com/r/solana") == (
        "reddit", "solana")
    assert classify_platform("https://blog.example.co/post")[0] == "web"


# --- the access boundary -------------------------------------------------

@pytest.mark.parametrize("url,fragment", [
    ("https://user:pw@site.test/x", "credentials"),
    ("https://discord.com/channels/1/2", "access control"),
    ("file:///etc/passwd", "public web fetch"),
    ("ftp://host/file", "public web fetch"),
    ("javascript:alert(1)", "public web fetch"),
    ("", "empty target"),
    ("not-a-url", "http(s)"),
])
def test_targets_behind_or_around_an_access_control_are_refused(url, fragment):
    admitted, reason = AccessPolicy().admits(url)
    assert not admitted
    assert fragment in reason


def test_a_public_page_is_admitted():
    assert AccessPolicy().admits("https://t.me/s/public_channel") == (True, "")


def test_a_refused_target_is_recorded_and_never_becomes_a_candidate():
    crawler = _crawler([Finding(url="https://discord.com/channels/9/9",
                                title="private")])
    crawler.seed_from_chain(ticker="WIF")
    report = crawler.cycle()
    assert report["refused"] >= 1
    assert crawler.candidates == {}
    assert "access control" in crawler.refused[0]["reason"]


# --- the adversarial reserve --------------------------------------------

def test_the_adversarial_reserve_is_spent_before_a_full_ordinary_queue():
    crawler = _crawler([], query_budget=20, adversarial_reserve=0.25)
    for index in range(50):
        crawler.seed(Seed(kind=SeedKind.TOKEN_NAME, value=f"tok{index}"))
    crawler.seed_adversarial(
        [Seed(kind=SeedKind.NARRATIVE, value="unknown launchpad")])
    report = crawler.cycle()
    assert report["adversarial_queries"] > 0
    assert report["queries_by_direction"][Direction.ADVERSARIAL.value] > 0


def test_a_zero_reserve_spends_nothing_adversarially():
    crawler = _crawler([], query_budget=20, adversarial_reserve=0.0)
    for index in range(10):
        crawler.seed(Seed(kind=SeedKind.TOKEN_NAME, value=f"tok{index}"))
    crawler.seed_adversarial([Seed(kind=SeedKind.NARRATIVE, value="x")])
    assert crawler.cycle()["adversarial_queries"] == 0


def test_the_budget_is_a_ceiling():
    crawler = _crawler([], query_budget=7)
    for index in range(50):
        crawler.seed(Seed(kind=SeedKind.TOKEN_NAME, value=f"tok{index}"))
    assert crawler.cycle()["queries"] <= 7


def test_an_unfinished_seed_is_continued_next_cycle_not_dropped():
    crawler = _crawler([], query_budget=5, adversarial_reserve=0.0)
    crawler.seed(Seed(kind=SeedKind.TOKEN_NAME, value="Bonkinu"))
    first = crawler.cycle()
    assert first["queries"] == 5
    assert crawler.queue, "the partially expanded seed was thrown away"
    second = crawler.cycle()
    assert second["queries"] > 0
    # And no query is issued twice across the two cycles.
    assert len({query.key() for query in crawler.calls}) == len(crawler.calls)


def test_a_query_is_never_repeated():
    crawler = _crawler([], query_budget=200)
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    first = crawler.cycle()["queries"]
    crawler.seen_seeds.clear()
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    assert first > 0
    assert crawler.cycle()["queries"] == 0


# --- chain -> internet ---------------------------------------------------

def test_a_launch_seeds_every_handle_the_chain_gave_us():
    crawler = _crawler([])
    added = crawler.seed_from_chain(
        mint=MINT, ticker="BONKINU", name="Bonk Inu", creator="Dep111",
        funders=["F1", "F2"], domain="bonkinu.io",
        handles=["bonkinu_official"], launchpad="pump")
    assert added == 9
    kinds = {seed.kind for seed in crawler.queue}
    assert SeedKind.MINT in kinds and SeedKind.FUNDER in kinds
    assert SeedKind.LAUNCHPAD in kinds


def test_a_finding_seeds_the_next_round_of_search():
    crawler = _crawler([Finding(
        url="https://blog.test/post",
        snippet=f"the dev also runs https://t.me/insiderdesk and {MINT}")])
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    report = crawler.cycle()
    assert report["new_seeds"] >= 2
    values = {seed.value for seed in crawler.queue}
    assert MINT in values and "insiderdesk" in values


# --- internet -> chain ---------------------------------------------------

def test_a_domain_seen_before_any_token_becomes_a_watch_term():
    crawler = _crawler([Finding(url="https://forum.test/thread",
                                snippet="watch https://moonproject.io soon")])
    crawler.seed(Seed(kind=SeedKind.NARRATIVE, value="new launches"))
    crawler.cycle(now=1_000.0)
    assert "moonproject.io" in crawler.watch_terms


def test_a_later_mint_matching_a_watch_term_is_a_measured_pre_launch_lead():
    crawler = _crawler([])
    crawler.register_watch_term("moonproject.io", SeedKind.DOMAIN,
                                origin_url="https://forum.test/t", now=1_000.0)
    hits = crawler.match_new_mint(mint=MINT, ticker="MOON",
                                  domain="moonproject.io", now=173_800.0)
    assert len(hits) == 1
    assert hits[0].matched_mint == MINT
    assert hits[0].lead_seconds == pytest.approx(172_800.0)   # two days
    leads = crawler.pre_launch_leads()
    assert leads[0]["origin_url"] == "https://forum.test/t"


def test_a_watch_term_hits_once_rather_than_on_every_later_mint():
    crawler = _crawler([])
    crawler.register_watch_term("moonproject.io", SeedKind.DOMAIN, now=0.0)
    assert len(crawler.match_new_mint(mint="A", domain="moonproject.io",
                                      now=10.0)) == 1
    assert crawler.match_new_mint(mint="B", domain="moonproject.io",
                                  now=20.0) == []
    assert len(crawler.pre_launch_leads()) == 1


def test_an_unrelated_mint_matches_nothing():
    crawler = _crawler([])
    crawler.register_watch_term("moonproject.io", SeedKind.DOMAIN, now=0.0)
    assert crawler.match_new_mint(mint="A", ticker="WIF", now=10.0) == []


def test_a_trivially_short_term_is_not_registered():
    crawler = _crawler([])
    assert not crawler.register_watch_term("io", SeedKind.DOMAIN)


# --- promotion into the continuous mesh ---------------------------------

def test_a_source_is_promoted_for_producing_not_for_claiming():
    finding = Finding(url="https://t.me/s/alphadesk", title="alpha insider")
    crawler = _crawler([finding], promotion_observations=3, query_budget=200)
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    report = crawler.cycle()
    # One cycle issues many queries, each returning the same channel: three
    # separate observations is the bar, and it is met by production.
    assert "alphadesk" in report["promotable"]
    assert crawler.candidates["telegram:alphadesk"].observations >= 3


def test_a_source_seen_once_is_not_promoted():
    seen = {"n": 0}

    def results(query):
        seen["n"] += 1
        return [Finding(url="https://t.me/s/oneoff")] if seen["n"] == 1 else []

    crawler = _crawler(results, promotion_observations=3, query_budget=50)
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    assert crawler.cycle()["promotable"] == []


def test_marking_promoted_removes_it_from_the_promotable_set():
    crawler = _crawler([Finding(url="https://t.me/s/alphadesk")],
                       promotion_observations=2, query_budget=200)
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    crawler.cycle()
    assert crawler.mark_promoted(["alphadesk"]) == 1
    assert crawler.promotable() == []
    assert crawler.status()["promoted"] == 1


# --- authority ----------------------------------------------------------

def test_findings_become_events_with_no_confidence_attached():
    crawler = _crawler([])
    events = crawler.to_events([Finding(
        url="https://t.me/s/alphadesk", title=f"buy {MINT} now",
        snippet="site https://scamcoin.io", language="en",
        published_at=100.0)])
    assert len(events) == 1
    event = events[0]
    assert event.confidence_at_time is None
    assert event.source_id == "telegram:alphadesk"
    assert event.mint == MINT
    assert event.lawful_access is LawfulAccess.PUBLIC
    assert "scamcoin.io" in event.linked_domains


def test_crawler_events_are_admissible_but_score_nothing_on_their_own():
    crawler = _crawler([])
    events = crawler.to_events(
        [Finding(url=f"https://t.me/s/chan{index}", title=f"call {MINT}")
         for index in range(40)])
    ledger = SourceEdgeLedger()
    assert ledger.extend(events)["accepted"] == 40
    # Accepted, and still worth nothing: no forward returns were observed, so
    # there is no cell to score. Discovery does not manufacture edge.
    assert ledger.scored_edges() == []


def test_no_searcher_is_data_blocked_rather_than_a_quiet_success():
    crawler = PublicInternetEventCrawler()
    crawler.seed(Seed(kind=SeedKind.TICKER, value="WIF"))
    assert crawler.cycle()["status"] == "DATA_BLOCKED"


def test_an_empty_queue_is_idle_rather_than_inventing_work():
    assert _crawler([]).cycle()["status"] == "IDLE"


def test_a_searcher_that_raises_costs_one_query_not_the_cycle():
    def results(query):
        if "上线" in query.text:
            raise RuntimeError("blocked")
        return [Finding(url="https://t.me/s/ok")]

    crawler = _crawler(results, query_budget=200)
    crawler.seed(Seed(kind=SeedKind.TOKEN_NAME, value="Bonkinu"))
    report = crawler.cycle()
    assert report["status"] == "OK"
    assert report["failures"]
    assert report["findings"] > 0
