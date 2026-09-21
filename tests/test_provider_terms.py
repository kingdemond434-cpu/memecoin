"""The free-tier watcher, and the two ways a watcher becomes useless.

It becomes useless by crying wolf on every marketing deploy, and by staying
silent through the change that actually matters. Both get tests.
"""

import json

import pytest

from src.research.provider_terms import (
    KNOWN_PROVIDERS, Provider, ProviderTermsWatcher, Severity, Support,
    TermsSnapshot, extract_signals)

FREE_PAGE = """
    Solana RPC for builders. Free tier: 10,000,000 requests per month.
    No credit card required. 50 requests per second on the free plan.
    Supported methods include getAccountInfo, getSignaturesForAddress,
    and Yellowstone gRPC streaming.
"""

CUT_PAGE = """
    Solana RPC for builders. Free tier: 1,000,000 requests per month.
    No credit card required. 50 requests per second on the free plan.
    Supported methods include getAccountInfo, getSignaturesForAddress.
"""

PAID_PAGE = """
    Solana RPC for builders. Start your 14-day trial: 1,000,000 requests
    per month. A credit card is required to begin. Yellowstone gRPC
    streaming is available on paid plans only.
"""


def _watcher(pages, providers=None, tmp_path=None):
    calls = []

    def fetch(url):
        calls.append(url)
        entry = pages.get(url, (404, ""))
        if isinstance(entry, Exception):
            raise entry
        return entry

    watcher = ProviderTermsWatcher(
        providers=providers or [Provider(
            name="demo", terms_url="https://demo.test/pricing",
            chains=("solana",),
            watch_methods=("Yellowstone", "getSignaturesForAddress"))],
        fetcher=fetch,
        state_path=(tmp_path / "terms.json") if tmp_path else None)
    watcher.calls = calls
    return watcher


# --- signal extraction ---------------------------------------------------

def test_quota_extraction_reads_the_headline_allowance():
    signals = extract_signals(FREE_PAGE)
    assert signals["quotas"]["requests_per_month"] == 10_000_000
    assert signals["quotas"]["requests_per_second"] == 50


def test_quota_extraction_understands_abbreviated_units():
    signals = extract_signals("Free plan: 2M compute units per month.")
    assert signals["quotas"]["compute_units_per_month"] == 2_000_000
    signals = extract_signals("Free plan: 500k requests per month.")
    assert signals["quotas"]["requests_per_month"] == 500_000


def test_flags_are_read_as_facts_not_sentiment():
    free = extract_signals(FREE_PAGE)["flags"]
    assert free["no_card_required"] and free["free_tier"]
    assert not free["card_required"] and not free["trial_only"]
    paid = extract_signals(PAID_PAGE)["flags"]
    assert paid["card_required"] and paid["trial_only"] and paid["paid_only"]


def test_a_copyright_year_is_not_a_quota():
    signals = extract_signals("(c) 2026 Provider Inc. All rights reserved.")
    assert signals["quotas"] == {}


# --- diffing -------------------------------------------------------------

def test_a_cosmetic_rewrite_is_recorded_but_not_escalated(tmp_path):
    pages = {"https://demo.test/pricing": (200, FREE_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    assert watcher.poll()["status"] == "OK"
    pages["https://demo.test/pricing"] = (
        200, FREE_PAGE + "\n<!-- new hero image, build 4471 -->")
    result = watcher.poll()
    assert result["status"] == "OK"
    assert result["breaking"] == []
    assert [item["severity"] for item in result["changes"]] == ["COSMETIC"]


def test_a_halved_allowance_is_breaking_and_says_by_how_much(tmp_path):
    pages = {"https://demo.test/pricing": (200, FREE_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    watcher.poll()
    pages["https://demo.test/pricing"] = (200, CUT_PAGE)
    result = watcher.poll()
    assert result["status"] == "BREAKING_CHANGE"
    quota = [item for item in result["breaking"]
             if item["field"] == "requests_per_month"][0]
    assert quota["before"] == 10_000_000 and quota["after"] == 1_000_000
    assert "10%" in quota["detail"]


def test_free_becoming_a_trial_with_a_card_is_breaking(tmp_path):
    pages = {"https://demo.test/pricing": (200, FREE_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    watcher.poll()
    pages["https://demo.test/pricing"] = (200, PAID_PAGE)
    result = watcher.poll()
    fields = {item["field"] for item in result["breaking"]}
    assert "flag:no_card_required" in fields   # lost a good phrase
    assert "flag:card_required" in fields      # gained a bad one
    assert "flag:trial_only" in fields
    assert "method:Yellowstone" not in fields  # still mentioned, on paid


def test_a_watched_method_disappearing_is_breaking(tmp_path):
    pages = {"https://demo.test/pricing": (200, FREE_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    watcher.poll()
    pages["https://demo.test/pricing"] = (
        200, FREE_PAGE.replace("Yellowstone gRPC streaming", "websockets"))
    result = watcher.poll()
    fields = {item["field"] for item in result["breaking"]}
    assert "method:Yellowstone" in fields


def test_a_raised_allowance_is_not_an_alarm(tmp_path):
    pages = {"https://demo.test/pricing": (200, CUT_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    watcher.poll()
    pages["https://demo.test/pricing"] = (200, FREE_PAGE)
    result = watcher.poll()
    assert result["status"] == "OK"
    assert all(item["severity"] != "BREAKING" for item in result["changes"])


# --- failures ------------------------------------------------------------

def test_a_failed_read_does_not_erase_the_baseline(tmp_path):
    pages = {"https://demo.test/pricing": (200, FREE_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    watcher.poll()
    baseline = watcher.snapshots["demo"].content_hash

    pages["https://demo.test/pricing"] = (503, "")
    blocked = watcher.poll()
    assert blocked["status"] == "DATA_BLOCKED"
    assert watcher.snapshots["demo"].content_hash == baseline

    # The cut still gets caught after the outage, because the baseline
    # survived it. This is the whole reason a failed read is not stored.
    pages["https://demo.test/pricing"] = (200, CUT_PAGE)
    assert watcher.poll()["status"] == "BREAKING_CHANGE"


def test_a_raising_fetcher_is_data_blocked_not_an_exception(tmp_path):
    watcher = _watcher({"https://demo.test/pricing": OSError("dns")},
                       tmp_path=tmp_path)
    result = watcher.poll()
    assert result["status"] == "DATA_BLOCKED"
    assert "OSError" in result["data_blocked"][0]["reason"]


def test_no_fetcher_is_data_blocked_rather_than_silently_clean():
    watcher = ProviderTermsWatcher(providers=[Provider(
        name="demo", terms_url="https://demo.test/pricing")])
    result = watcher.poll()
    assert result["status"] == "DATA_BLOCKED"
    assert "no fetcher" in result["data_blocked"][0]["reason"]


def test_state_survives_a_restart(tmp_path):
    pages = {"https://demo.test/pricing": (200, FREE_PAGE)}
    watcher = _watcher(pages, tmp_path=tmp_path)
    watcher.poll()

    pages["https://demo.test/pricing"] = (200, CUT_PAGE)
    reborn = _watcher(pages, tmp_path=tmp_path)
    assert "demo" in reborn.snapshots
    assert reborn.poll()["status"] == "BREAKING_CHANGE"


def test_corrupt_state_does_not_prevent_a_first_pass(tmp_path):
    (tmp_path / "terms.json").write_text("{not json")
    watcher = _watcher({"https://demo.test/pricing": (200, FREE_PAGE)},
                       tmp_path=tmp_path)
    assert watcher.poll()["status"] == "OK"


# --- the registry --------------------------------------------------------

def test_an_unmeasured_provider_is_not_a_rung():
    provider = Provider(name="p", chains=("solana",), support=Support.CANDIDATE,
                        rpc_url="https://p.test")
    usable, reason = provider.usable_for("solana")
    assert not usable
    assert "never been measured" in reason


def test_a_verified_provider_without_a_url_is_still_not_a_rung():
    provider = Provider(name="p", chains=("solana",), support=Support.VERIFIED)
    usable, reason = provider.usable_for("solana")
    assert not usable
    assert "no resolved endpoint" in reason


def test_a_verified_provider_with_a_url_is_usable():
    provider = Provider(name="p", chains=("solana",), support=Support.VERIFIED,
                        rpc_url="https://p.test")
    assert provider.usable_for("solana") == (True, "")


def test_nodeflare_is_recorded_as_evm_only_rather_than_added_to_solana():
    watcher = ProviderTermsWatcher()
    nodeflare = watcher.provider("nodeflare")
    assert nodeflare is not None
    usable, reason = nodeflare.usable_for("solana")
    assert not usable
    assert "EVM" in reason
    assert not nodeflare.serves("solana")
    assert nodeflare.serves("ethereum")
    # And it is not silently sitting in the usable set for the chain we trade.
    assert nodeflare not in watcher.usable_rungs("solana")


def test_no_known_provider_is_usable_for_solana_without_measurement():
    watcher = ProviderTermsWatcher()
    assert watcher.usable_rungs("solana") == []
    status = watcher.status("solana")
    assert status["usable"] == 0
    assert {row["provider"] for row in status["providers"]} >= {
        "nodeflare", "solarchive", "raptor"}


def test_every_known_provider_states_why_it_is_not_yet_a_rung():
    for provider in KNOWN_PROVIDERS:
        if provider.support is not Support.VERIFIED:
            assert provider.blocking_reason, provider.name


def test_the_solana_tracker_public_rpc_is_recorded_but_not_yet_a_rung():
    watcher = ProviderTermsWatcher()
    provider = watcher.provider("solana_tracker_public_rpc")
    assert provider is not None
    assert provider.serves("solana")
    usable, reason = provider.usable_for("solana")
    assert not usable, "nobody here has spoken to it"
    assert "never been measured" in reason or "candidate" in reason.lower()
    # And the paid-Yellowstone distinction is written down, so it does not get
    # re-raised as a free Geyser source.
    assert "PAID" in provider.note and "Yellowstone" in provider.note


def test_the_config_ladder_appends_it_last_with_no_paired_websocket():
    """rpc_urls and ws_urls are paired BY POSITION in this config.

    Inserting anywhere but the end silently hands one provider another
    provider's websocket, which is the failure the file's own comment warns
    about.
    """
    import yaml
    config = yaml.safe_load(open("config/chains.yaml"))
    solana = config["chains"]["solana"]
    assert solana["rpc_urls"][-1] == "https://rpc.solanatracker.io/public"
    assert len(solana["ws_urls"]) < len(solana["rpc_urls"])
