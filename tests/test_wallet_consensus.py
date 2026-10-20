"""Agreement is only agreement between people who decided separately.

Every failure mode here is the same shape: something that looks like several
wallets agreeing and is not. One operator's ring entering over four slots.
One wallet scaling in three times. Three wallets that merely score well on a
formula nobody has validated. Each of those produces exactly the pattern a
naive consensus rule is hunting for, which is why a naive consensus rule finds
so much of it.
"""

import pytest

from src.strategies.wallet_consensus import (
    ConsensusRule, WalletConsensus, default_rule_grid)

T0 = 1_700_000_000.0


def _followable(**wallets):
    """wallet -> confidence. >= 0.5 means measured, below means scored only."""
    return lambda: dict(wallets)


def _independent(count, detail=None):
    return lambda wallets: (count, detail or {})


def _one_per_wallet(wallets):
    """Every address is its own decider -- no ring detected."""
    return len(set(wallets)), {"reason": "all_distinct"}


def _consensus(followable, independence=_one_per_wallet, **kwargs):
    return WalletConsensus(followable_provider=followable,
                           independence_provider=independence, **kwargs)


class TestItCountsDecidersNotAddresses:
    def test_three_measured_wallets_in_a_tight_window_fire_the_k3_rules(self):
        engine = _consensus(_followable(a=0.8, b=0.7, c=0.6))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        signal = engine.evaluate("mint", now=T0 + 5)
        assert signal.ok
        assert "consensus:k3/w10s/measured" in signal.fired
        assert signal.independent_deciders == 3

    def test_a_ring_of_ten_addresses_is_one_decider_and_fires_nothing(self):
        engine = _consensus(_followable(**{f"w{i}": 0.8 for i in range(10)}),
                            independence=_independent(1, {"reason": "one_funder"}))
        for index in range(10):
            engine.observe_buy("mint", f"w{index}", at=T0 + index * 0.5)
        signal = engine.evaluate("mint", now=T0 + 20)
        assert signal.followable_wallets == 10
        assert signal.independent_deciders == 1
        assert signal.fired == ()

    def test_one_wallet_scaling_in_is_still_one_wallet(self):
        engine = _consensus(_followable(a=0.9))
        for index in range(5):
            engine.observe_buy("mint", "a", at=T0 + index)
        signal = engine.evaluate("mint", now=T0 + 10)
        assert signal.followable_wallets == 1
        assert signal.fired == ()

    def test_wallets_nobody_follows_are_not_stored_at_all(self):
        engine = _consensus(_followable(a=0.8))
        assert engine.observe_buy("mint", "a", at=T0) is True
        assert engine.observe_buy("mint", "stranger", at=T0) is False
        assert engine.evaluate("mint", now=T0 + 1).followable_wallets == 1


class TestUnmeasuredIndependenceBlocks:
    def test_without_an_independence_provider_nothing_fires(self):
        """Assuming independence here would invent the consensus."""
        engine = WalletConsensus(followable_provider=_followable(a=0.8, b=0.8, c=0.8))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        signal = engine.evaluate("mint", now=T0 + 5)
        assert signal.status == "DATA_BLOCKED"
        assert "independence" in signal.blocked
        assert signal.fired == ()

    def test_a_provider_that_cannot_tell_also_blocks(self):
        engine = _consensus(_followable(a=0.8, b=0.8, c=0.8),
                            independence=_independent(None, {"reason": "too_few_launches"}))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        assert engine.evaluate("mint", now=T0 + 5).status == "DATA_BLOCKED"

    def test_a_provider_that_raises_blocks_rather_than_defaulting(self):
        def boom(wallets):
            raise RuntimeError("graph unavailable")
        engine = _consensus(_followable(a=0.8, b=0.8, c=0.8), independence=boom)
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        assert engine.evaluate("mint", now=T0 + 5).status == "DATA_BLOCKED"


class TestMeasuredVersusMerelyScored:
    def test_scored_only_wallets_do_not_satisfy_the_measured_rules(self):
        engine = _consensus(_followable(a=0.45, b=0.40, c=0.30))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        signal = engine.evaluate("mint", now=T0 + 5)
        assert signal.measured_deciders == 0
        assert not any("measured" in name for name in signal.fired)
        assert "consensus:k3/w10s/any" in signal.fired

    def test_the_grid_separates_the_two_so_the_gauntlet_can_price_them(self):
        names = {rule.mechanism for rule in default_rule_grid()}
        assert "consensus:k3/w20s/measured" in names
        assert "consensus:k3/w20s/any" in names


class TestWindowsAndTheDeployer:
    def test_entries_spread_beyond_the_window_do_not_fire_the_tight_rules(self):
        engine = _consensus(_followable(a=0.8, b=0.8, c=0.8))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index * 30)
        fired = engine.evaluate("mint", now=T0 + 100).fired
        assert "consensus:k3/w10s/measured" not in fired
        assert "consensus:k3/w60s/measured" in fired

    def test_the_deployer_distributing_suppresses_the_rules_that_forbid_it(self):
        engine = _consensus(_followable(a=0.8, b=0.8, c=0.8))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        engine.observe_deployer_sell("mint", at=T0 + 2)
        signal = engine.evaluate("mint", now=T0 + 5)
        assert signal.deployer_selling is True
        assert "consensus:k3/w10s/measured/nodevsell" not in signal.fired
        assert "consensus:k3/w10s/measured" in signal.fired

    def test_stale_buys_fall_out_of_the_horizon(self):
        engine = _consensus(_followable(a=0.8, b=0.8, c=0.8), horizon_s=10.0)
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        assert engine.evaluate("mint", now=T0 + 500).followable_wallets == 0


class TestItReportsWithoutClaiming:
    def test_the_report_separates_never_fired_from_killed(self):
        engine = _consensus(_followable(a=0.8, b=0.8, c=0.8))
        for index, wallet in enumerate("abc"):
            engine.observe_buy("mint", wallet, at=T0 + index)
        engine.evaluate("mint", now=T0 + 5)
        report = engine.report()
        assert report["fired_counts"]
        assert report["never_fired"]
        assert set(report["fired_counts"]) & set(report["never_fired"]) == set()

    def test_nothing_observed_reports_data_blocked(self):
        assert _consensus(_followable()).report()["status"] == "DATA_BLOCKED"

    def test_a_rule_name_is_a_function_of_the_rule_alone(self):
        """The gauntlet keys rows on this, so it must not drift."""
        rule = ConsensusRule(3, 20.0, require_measured=True,
                             forbid_deployer_selling=True)
        assert rule.mechanism == "consensus:k3/w20s/measured/nodevsell"


class TestTheDeskActuallyFeedsIt:
    """A consensus engine nobody tells about buys is a report of zero.

    This codebase's recurring defect is a subsystem that is built, tested and
    never called, so the handler path is tested here rather than only the
    engine.
    """

    def _desk(self, consensus, creator="dev"):
        from types import SimpleNamespace
        from src.runtime.depth import FollowableTrades

        class Desk(FollowableTrades):
            pass

        desk = Desk()
        desk.wallet_consensus = consensus
        desk._latest_curve_state = {"mint": SimpleNamespace(creator=creator)}
        desk.info_graph = SimpleNamespace(events=[],
                                          record_event=lambda *a, **k: None)
        recorded = []
        desk.info_graph.record_event = lambda *a, **k: recorded.append(a)
        desk._recorded = recorded
        return desk

    def test_a_buy_from_a_followable_wallet_reaches_the_engine(self):
        engine = _consensus(_followable(a=0.8))
        desk = self._desk(engine)
        desk._record_followable_trade(
            "mint", {"wallet": "a", "side": "buy", "timestamp": T0})
        assert engine.evaluate("mint", now=T0 + 1).followable_wallets == 1

    def test_a_buy_from_an_unfollowed_wallet_raises_nothing(self):
        engine = _consensus(_followable(a=0.8))
        desk = self._desk(engine)
        desk._record_followable_trade(
            "mint", {"wallet": "stranger", "side": "buy", "timestamp": T0})
        assert engine.evaluate("mint", now=T0 + 1).followable_wallets == 0
        assert desk._recorded == []

    def test_a_merely_scored_wallet_does_not_raise_an_elite_buy_event(self):
        """The old gate was `overall_score >= 0.7` on a hand-weighted sum."""
        engine = _consensus(_followable(scored=0.45))
        desk = self._desk(engine)
        desk._record_followable_trade(
            "mint", {"wallet": "scored", "side": "buy", "timestamp": T0})
        assert desk._recorded == []

    def test_a_measured_wallet_does_raise_one(self):
        engine = _consensus(_followable(measured=0.9))
        desk = self._desk(engine)
        desk._record_followable_trade(
            "mint", {"wallet": "measured", "side": "buy", "timestamp": T0})
        assert len(desk._recorded) == 1

    def test_the_deployer_selling_is_recorded_as_such(self):
        engine = _consensus(_followable(dev=0.9))
        desk = self._desk(engine, creator="dev")
        desk._record_followable_trade(
            "mint", {"wallet": "dev", "side": "sell", "timestamp": T0})
        assert engine.evaluate("mint", now=T0 + 1).deployer_selling is True

    def test_an_ordinary_wallet_selling_is_not_the_deployer(self):
        engine = _consensus(_followable(a=0.9))
        desk = self._desk(engine, creator="dev")
        desk._record_followable_trade(
            "mint", {"wallet": "a", "side": "sell", "timestamp": T0})
        assert engine.evaluate("mint", now=T0 + 1).deployer_selling is False


def test_deployer_sells_for_untouched_tokens_do_not_accumulate_forever():
    """That map is not kept in step by the buy record, which does the evicting."""
    engine = _consensus(_followable(), max_tokens=8, horizon_s=10.0)
    for index in range(200):
        engine.observe_deployer_sell(f"mint{index}", at=T0 + index)
    assert len(engine._deployer_sold) <= 16
