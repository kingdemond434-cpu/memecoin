"""A fixed 1 SOL per copied wallet makes every wallet equally right forever.

The failure this guards against is quiet: a wallet that carried the book for a
month and stopped working looks, to a fixed-size copier, exactly like one that
never stopped. Allocation here moves on the agreement between two independent
estimates -- what the gauntlet said following the wallet was worth, and what
following it has actually paid since -- and the tests are written as the
specific ways that agreement can be faked.
"""

import pytest

from src.strategies.wallet_allocator import WalletAllocator, WalletBudget


def _allocator(**kwargs):
    return WalletAllocator(**kwargs)


def _survivor(allocator, wallet, bound=0.10):
    return allocator.register(wallet, lower_bound=bound, verdict="SURVIVOR")


class TestHistoryAloneFundsNothing:
    def test_a_survivor_enters_on_probation_not_at_its_ceiling(self):
        allocator = _allocator()
        budget = _survivor(allocator, "a", bound=0.20)
        assert budget.earned_weight == pytest.approx(1.0)
        assert budget.weight == pytest.approx(allocator.probation_weight)
        assert budget.on_probation

    def test_probation_does_not_lift_before_enough_live_outcomes(self):
        allocator = _allocator()
        _survivor(allocator, "a")
        for _ in range(allocator.min_live_samples - 1):
            allocator.record_live("a", 0.2)
        assert allocator.weight("a") == pytest.approx(allocator.probation_weight)

    def test_an_unregistered_wallet_gets_zero_not_a_small_default(self):
        assert _allocator().weight("never-heard-of-it") == 0.0

    def test_a_killed_wallet_is_registered_and_funded_at_zero(self):
        """Different from never having heard of it, and worth saying so."""
        allocator = _allocator()
        budget = allocator.register("a", lower_bound=0.3, verdict="KILL")
        assert budget.status == "OK"
        assert allocator.weight("a") == 0.0
        assert "KILL" in budget.detail

    def test_a_wallet_with_no_measured_bound_is_data_blocked(self):
        allocator = _allocator()
        budget = allocator.register("a", lower_bound=None, verdict="DATA_BLOCKED")
        assert budget.status == "DATA_BLOCKED"
        assert allocator.weight("a") == 0.0


class TestWeightFollowsAgreement:
    def test_agreeing_live_evidence_grows_the_weight_toward_the_ceiling(self):
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.10)
        for _ in range(40):
            allocator.record_live("a", 0.30)
        assert allocator.weight("a") > allocator.probation_weight

    def test_the_weight_never_exceeds_what_the_history_earned(self):
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.02)
        for _ in range(200):
            allocator.record_live("a", 5.0)
        budget = allocator._budgets["a"]
        assert allocator.weight("a") <= budget.earned_weight + 1e-9

    def test_falling_below_the_historical_bound_decays_the_weight(self):
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.10)
        for _ in range(40):
            allocator.record_live("a", 0.30)
        grown = allocator.weight("a")
        for _ in range(40):
            allocator.record_live("a", -0.50)
        assert allocator.weight("a") < grown

    def test_decay_is_faster_than_growth(self):
        """An edge that stopped working costs money every day it stays funded."""
        allocator = _allocator()
        assert allocator.decay_rate > allocator.growth_rate

    def test_a_wallet_that_stops_working_is_defunded_not_merely_trimmed(self):
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.10)
        for _ in range(60):
            allocator.record_live("a", 0.30)
        for _ in range(200):
            allocator.record_live("a", -1.0)
        assert allocator.weight("a") < 0.001

    def test_agreement_is_judged_against_the_bound_not_the_mean(self):
        """A live mean above the bound agrees even if it is unremarkable."""
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.05)
        for _ in range(40):
            allocator.record_live("a", 0.06)
        budget = allocator._budgets["a"]
        assert budget.diverging == 0
        assert budget.agreeing == 40

    def test_live_outcomes_for_an_unregistered_wallet_are_refused(self):
        assert _allocator().record_live("stranger", 0.5) is None


class TestClustersShareOneBudget:
    def test_ten_addresses_of_one_operator_cannot_hold_ten_budgets(self):
        allocator = _allocator(cluster_provider=lambda wallet: "ring",
                               cluster_cap=0.30)
        for index in range(10):
            _survivor(allocator, f"w{index}", bound=0.50)
            for _ in range(60):
                allocator.record_live(f"w{index}", 1.0)
        total = sum(allocator.weight(f"w{i}") for i in range(10))
        assert total <= 0.30 * 10  # each is individually capped by headroom
        assert allocator.weight("w0") < 0.30

    def test_unknown_clustering_keeps_wallets_separate(self):
        """Inventing a cluster would halve diversification on no evidence."""
        allocator = _allocator()
        _survivor(allocator, "a")
        _survivor(allocator, "b")
        assert allocator._budgets["a"].cluster == "a"
        assert allocator._budgets["b"].cluster == "b"

    def test_a_cluster_provider_that_raises_falls_back_to_the_address(self):
        def boom(wallet):
            raise RuntimeError("graph down")
        allocator = _allocator(cluster_provider=boom)
        assert _survivor(allocator, "a").cluster == "a"


class TestAllocationRefusesToSpreadThin:
    def test_nothing_earned_means_nothing_allocated(self):
        allocator = _allocator()
        allocator.register("a", lower_bound=None, verdict="DATA_BLOCKED")
        assert allocator.allocate(10_000.0) == {}

    def test_the_budget_splits_in_proportion_to_weight(self):
        allocator = _allocator()
        _survivor(allocator, "big", bound=0.10)
        _survivor(allocator, "small", bound=0.10)
        for _ in range(60):
            allocator.record_live("big", 0.50)
        for _ in range(allocator.min_live_samples):
            allocator.record_live("small", 0.50)
        split = allocator.allocate(1_000.0)
        assert split["big"] > split["small"]
        assert sum(split.values()) == pytest.approx(1_000.0)

    def test_a_zero_budget_allocates_nothing(self):
        allocator = _allocator()
        _survivor(allocator, "a")
        assert allocator.allocate(0.0) == {}


class TestTheReportSaysWhatIsHappening:
    def test_an_empty_book_says_it_is_funding_nothing(self):
        report = _allocator().report()
        assert report["status"] == "DATA_BLOCKED"
        assert "funding nothing" in report["detail"]

    def test_probation_and_decay_are_counted_separately(self):
        allocator = _allocator()
        _survivor(allocator, "new")
        _survivor(allocator, "failing", bound=0.10)
        for _ in range(60):
            allocator.record_live("failing", -1.0)
        report = allocator.report()
        assert report["on_probation"] == 1
        assert report["decaying"] == 1

    def test_concentration_is_reported_so_one_source_cannot_hide(self):
        allocator = _allocator(cluster_provider=lambda wallet: "ring")
        _survivor(allocator, "a")
        _survivor(allocator, "b")
        assert allocator.report()["concentration"] == pytest.approx(1.0)


class TestTheTwoEstimatesStayDisjoint:
    """The agreement test is meaningless if a wallet becomes its own benchmark.

    Historical value and live performance have to come from disjoint data. A
    re-registration that folded the live outcomes back into the bound would
    compare the wallet against itself, and a wallet compared against itself
    always agrees -- which is exactly the failure mode that makes a decayed
    edge look fine right up until the money is gone.
    """

    def test_the_bound_freezes_once_live_evidence_arrives(self):
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.10)
        allocator.record_live("a", -1.0)
        allocator.register("a", lower_bound=-5.0, verdict="KILL")
        assert allocator._budgets["a"].historical_lower_bound == pytest.approx(0.10)

    def test_re_registering_before_any_live_evidence_still_updates(self):
        """A wallet whose history is still accumulating is not yet a benchmark."""
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.05)
        _survivor(allocator, "a", bound=0.15)
        assert allocator._budgets["a"].historical_lower_bound == pytest.approx(0.15)

    def test_a_deliberate_rebaseline_is_possible_but_never_the_default(self):
        allocator = _allocator()
        _survivor(allocator, "a", bound=0.10)
        allocator.record_live("a", 0.2)
        allocator.register("a", lower_bound=0.40, verdict="SURVIVOR", refreeze=True)
        assert allocator._budgets["a"].historical_lower_bound == pytest.approx(0.40)


class TestTheFollowLoopActuallyFundsIt:
    """An allocator nobody tells about resolutions reports an empty book."""

    def _desk(self, value):
        from types import SimpleNamespace
        return SimpleNamespace(
            wallet_allocator=_allocator(),
            wallet_intel=SimpleNamespace(get_wallet_value=lambda wallet: value))

    def test_a_resolution_registers_and_funds_the_wallet(self):
        from src.runtime.depth import update_copy_budget
        from src.strategies.wallet_value import WalletValue
        value = WalletValue(status="OK", wallet="a", samples=30, lower_bound=0.10)
        desk = self._desk(value)
        update_copy_budget(desk, "a", 1.5, accepted=True)
        budget = desk.wallet_allocator._budgets["a"]
        assert budget.historical_verdict == "SURVIVOR"
        assert budget.live_samples == 1

    def test_an_unmeasurable_wallet_is_registered_but_not_funded(self):
        from src.runtime.depth import update_copy_budget
        from src.strategies.wallet_value import WalletValue
        value = WalletValue(status="DATA_BLOCKED", wallet="a")
        desk = self._desk(value)
        update_copy_budget(desk, "a", 1.5, accepted=True)
        assert desk.wallet_allocator.weight("a") == 0.0

    def test_a_desk_without_an_allocator_is_a_no_op_not_a_crash(self):
        from types import SimpleNamespace
        from src.runtime.depth import update_copy_budget
        update_copy_budget(SimpleNamespace(), "a", 1.5, accepted=True)

    def test_the_resolution_loop_calls_it(self):
        """The whole point. Testing the function alone would have passed
        every day this codebase's other subsystems sat unwired."""
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_resolve_follow_candidates")
        assert "update_copy_budget" in ast.unparse(loop)
