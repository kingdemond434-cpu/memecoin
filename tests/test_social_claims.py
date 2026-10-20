"""The wallet-to-social gap, closed by asking a different question.

`_find_linked_social` refused to link a wallet to a person, and it was right
to: there is no verified public source for it, and an edge written on a
display name is an identity claim the desk cannot support.

But a follower does not need to know which human owns a wallet. What moves a
memecoin is whether a named account with reach is TALKING ABOUT THIS MINT --
a link between a token and an account, which is directly observable:

    the deployer DECLARES handles in the token's own metadata
    the public source stream shows whether those handles POSTED

A declaration is free. A post is evidence, because the account controls what
it publishes and the desk observed it. And silence after a fair window is the
third state, which is the valuable one and the one the desk could never see.
"""

import time
from types import SimpleNamespace

import pytest

from src.research.social_claims import (
    CORROBORATED, DECLARED, UNCORROBORATED, SocialClaimLedger,
    claims_from_metadata, normalise_handle)

T0 = 1_700_000_000.0


def _event(author, tokens, at=T0, source="x"):
    return SimpleNamespace(author_id=author, token_addresses=tuple(tokens),
                           source_at=at, source_id=source)


class TestAHandleIsAHandle:
    @pytest.mark.parametrize("raw", [
        "https://x.com/SomeOne", "https://twitter.com/SomeOne/", "@SomeOne",
        "SomeOne", "someone"])
    def test_every_form_normalises_to_one(self, raw):
        assert normalise_handle(raw) == "someone"

    def test_case_is_dropped_because_a_miss_reports_silence(self):
        """A corroboration that misses on capitalisation reports silence from
        an account that spoke."""
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "https://x.com/BigAccount"}, T0)
        assert ledger.corroborate("mint", "@bigaccount", T0 + 10) is True

    def test_nothing_is_not_a_handle(self):
        assert normalise_handle("") == ""
        assert normalise_handle(None) == ""


class TestDeclarationsAreClaimsNotFacts:
    def test_metadata_fields_become_claims(self):
        claims = claims_from_metadata({
            "twitter": "https://x.com/Famous", "telegram": "@chat",
            "website": "https://site.example", "name": "Coin"})
        assert ("twitter", "famous") in claims
        assert ("telegram", "chat") in claims
        # A website has no handle in it, so it is kept verbatim rather than
        # having one invented from the last path segment.
        assert ("website", "https://site.example") in claims

    def test_a_launch_declaring_nothing_records_nothing(self):
        ledger = SocialClaimLedger()
        assert ledger.declare("mint", {"name": "Coin"}, T0) == 0
        assert ledger.reading("mint")["status"] == "DATA_BLOCKED"

    def test_a_fresh_claim_starts_unchecked(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.claims("mint")[0].status == DECLARED
        assert ledger.corroborated_handles("mint") == []

    def test_the_same_claim_twice_is_one_claim(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.declare("mint", {"twitter": "https://x.com/Famous"}, T0) == 0


class TestOnlyAPostIsEvidence:
    def test_a_post_from_the_declared_account_corroborates(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.observe_event(_event("@famous", ["mint"], T0 + 60)) == 1
        claim = ledger.claims("mint")[0]
        assert claim.status == CORROBORATED
        assert claim.corroborated_at == T0 + 60
        assert ledger.corroborated_handles("mint") == ["famous"]

    def test_a_post_about_the_token_by_somebody_else_does_not(self):
        """Otherwise every mention corroborates every claim, and the whole
        measurement collapses into 'somebody talked about it'."""
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.observe_event(_event("@randomguy", ["mint"], T0 + 60)) == 0
        assert ledger.claims("mint")[0].status == DECLARED

    def test_a_post_by_the_account_about_a_different_token_does_not(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.observe_event(_event("@famous", ["other"], T0 + 60)) == 0

    def test_an_anonymous_post_corroborates_nothing(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.observe_event(_event("", ["mint"], T0 + 60)) == 0


class TestSilenceIsTheValuableState:
    def test_a_named_account_that_stays_silent_is_flagged(self):
        """The impersonation signature. Under the old arrangement this looked
        identical to a genuine influencer launch, because neither produced any
        social evidence at all."""
        ledger = SocialClaimLedger(silence_window_s=3600.0)
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        reading = ledger.reading("mint", now=T0 + 7200)
        assert reading["silent"] == 1
        assert reading["corroborated"] == 0
        assert reading["impersonation_suspected"] is True

    def test_inside_the_window_the_question_is_still_open(self):
        """Silence at sixty seconds is not evidence of anything."""
        ledger = SocialClaimLedger(silence_window_s=3600.0)
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        reading = ledger.reading("mint", now=T0 + 60)
        assert reading["pending"] == 1
        assert reading["impersonation_suspected"] is False

    def test_a_corroborated_launch_is_never_flagged(self):
        ledger = SocialClaimLedger(silence_window_s=3600.0)
        ledger.declare("mint", {"twitter": "@famous", "telegram": "@chat"}, T0)
        ledger.observe_event(_event("@famous", ["mint"], T0 + 60))
        reading = ledger.reading("mint", now=T0 + 7200)
        assert reading["corroborated"] == 1
        assert reading["silent"] == 1
        assert reading["impersonation_suspected"] is False

    def test_settle_closes_the_window_permanently(self):
        ledger = SocialClaimLedger(silence_window_s=3600.0)
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        assert ledger.settle(now=T0 + 7200) == 1
        assert ledger.claims("mint")[0].status == UNCORROBORATED

    def test_a_late_post_still_corroborates_a_settled_claim(self):
        """Settling records that the window passed; it does not make the desk
        deaf afterwards."""
        ledger = SocialClaimLedger(silence_window_s=3600.0)
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        ledger.settle(now=T0 + 7200)
        assert ledger.corroborate("mint", "@famous", T0 + 8000) is True
        assert ledger.claims("mint")[0].status == CORROBORATED


class TestOneHandleAcrossManyLaunches:
    def test_reuse_is_counted_because_it_is_the_giveaway(self):
        """A handle claimed by nine mints this hour is endorsing none of
        them."""
        ledger = SocialClaimLedger()
        for index in range(9):
            ledger.declare(f"mint{index}", {"twitter": "@famous"}, T0 + index)
        assert ledger.reading("mint0")["handle_reuse"]["famous"] == 9
        assert ledger.report()["handles_claimed_by_several_launches"]["famous"] == 9

    def test_a_single_use_handle_is_not_in_the_reuse_report(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@quiet"}, T0)
        assert ledger.report()["handles_claimed_by_several_launches"] == {}


class TestItSaysWhatItDoesNotKnow:
    def test_the_reading_carries_no_authority(self):
        ledger = SocialClaimLedger()
        ledger.declare("mint", {"twitter": "@famous"}, T0)
        reading = ledger.reading("mint")
        assert reading["authority"] == "none"
        assert "never that it owns the wallet" in reading["detail"]

    def test_an_empty_ledger_is_blocked_rather_than_clean(self):
        report = SocialClaimLedger().report()
        assert report["status"] == "DATA_BLOCKED"
        assert report["corroboration_rate"] is None

    def test_tokens_are_evicted_oldest_first(self):
        ledger = SocialClaimLedger(max_tokens=4)
        for index in range(20):
            ledger.declare(f"mint{index}", {"twitter": f"@a{index}"}, T0 + index)
        assert ledger.report()["tokens_with_claims"] <= 4

    def test_a_round_trip_through_disk_keeps_the_statuses(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = SocialClaimLedger(str(path))
        ledger.declare("mint", {"twitter": "@famous", "telegram": "@chat"}, T0)
        ledger.observe_event(_event("@famous", ["mint"], T0 + 60))
        assert ledger.save()
        restored = SocialClaimLedger(str(path))
        assert restored.load()
        assert restored.corroborated_handles("mint") == ["famous"]


class TestTheDeskIsWiredToIt:
    def test_the_source_stream_corroborates(self):
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "src" / "runtime"
                  / "source_intelligence.py").read_text(encoding="utf-8")
        assert "observe_event" in source

    def test_the_fetch_is_off_the_hot_path(self):
        """A T0 decision may never wait on an IPFS gateway."""
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "src" / "runtime"
                  / "actor_intelligence.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        starter = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "start_social_claim_fetch")
        assert "create_task" in ast.unparse(starter)

    def test_a_candidate_without_a_uri_starts_nothing(self):
        from src.runtime.actor_intelligence import start_social_claim_fetch
        assert start_social_claim_fetch(
            SimpleNamespace(), SimpleNamespace(metadata={})) is False

    def test_the_manifest_declares_the_slot(self):
        from src.runtime.intelligence_manifest import ENTRY_CONTRIBUTORS
        assert "social_claims" in {c.key for c in ENTRY_CONTRIBUTORS}
