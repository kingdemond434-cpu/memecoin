"""An exit slower than its entry, and one policy for two distributions."""

from __future__ import annotations

import unittest

from src.execution.exit_readiness import (
    MAX_ACCEPTABLE_READY_MS, MIN_EXCURSION_SAMPLES, MODE_MONSTER, MODE_RECYCLER,
    MODE_UNDECIDED, ExcursionLedger, ExitReadinessLedger, SellTemplate,
    choose_exit_mode)
from src.strategies.actor_graph import Entry
from src.strategies.cohort_lifecycle import evaluate_cohorts


def _template(token="T", built_at=100.0, blockhash_at=100.0):
    return SellTemplate(token=token, built_at=built_at, accounts=("a", "b"),
                        program_id="PROG", blockhash="bh",
                        blockhash_at=blockhash_at, ready=True)


class TheSellIsBuiltAtFillTimeNotAtPanicTime(unittest.TestCase):

    def test_a_prompt_template_is_clean(self):
        ledger = ExitReadinessLedger()
        elapsed = ledger.on_fill("T", filled_at=100.0, template=_template())
        self.assertEqual(0.0, elapsed)
        self.assertEqual(0, ledger.late)
        self.assertEqual("OK", ledger.report()["status"])

    def test_a_late_template_is_a_recorded_defect(self):
        ledger = ExitReadinessLedger()
        late_by = (MAX_ACCEPTABLE_READY_MS + 500.0) / 1000.0
        ledger.on_fill("T", filled_at=100.0,
                       template=_template(built_at=100.0 + late_by))
        self.assertEqual(1, ledger.late)
        self.assertEqual("DEGRADED", ledger.report()["status"])

    def test_a_stale_blockhash_makes_the_template_unusable(self):
        ledger = ExitReadinessLedger()
        ledger.on_fill("T", 100.0, _template(blockhash_at=100.0))
        self.assertIsNone(ledger.template_for("T", now=100.0 + 3600))
        self.assertIsNotNone(ledger.template_for("T", now=101.0))

    def test_a_missing_template_at_exit_is_counted(self):
        ledger = ExitReadinessLedger()
        self.assertIsNone(ledger.template_for("NOPE"))
        self.assertEqual(1, ledger.missing)

    def test_no_fills_reports_blocked_rather_than_perfect(self):
        self.assertEqual("DATA_BLOCKED", ExitReadinessLedger().report()["status"])


class ExcursionsSeeWhatWinRateCannot(unittest.TestCase):

    def test_a_high_win_rate_entry_with_awful_drawdown_is_refused(self):
        ledger = ExcursionLedger()
        for _ in range(MIN_EXCURSION_SAMPLES):
            # Wins often, but risks 40% to make 10%.
            ledger.record("grindy", mfe=0.10, mae=-0.40)
        profile = ledger.profile("grindy")
        self.assertEqual("OK", profile.status)
        self.assertLess(profile.ratio, 1.0)
        self.assertFalse(profile.worth_repeating)

    def test_a_lower_win_rate_entry_with_clean_excursions_passes(self):
        ledger = ExcursionLedger()
        for _ in range(MIN_EXCURSION_SAMPLES):
            ledger.record("clean", mfe=0.60, mae=-0.05)
        self.assertTrue(ledger.profile("clean").worth_repeating)

    def test_a_thin_state_yields_no_profile(self):
        ledger = ExcursionLedger()
        ledger.record("thin", 5.0, -0.01)
        profile = ledger.profile("thin")
        self.assertEqual("DATA_BLOCKED", profile.status)
        self.assertIsNone(profile.worth_repeating)


def _cohort(sold=True, absorbers_independent=True, retained=1.0, chasers=False):
    # Entries at 160 so the distribution at 205/210 falls INSIDE the 60s
    # retention window. With entries at 100 the sells land past the last
    # mark, retention reads 1.0, and the fixture silently tests the
    # opposite of what it claims.
    wallets = [f"w{i}" for i in range(25)]
    entries = [Entry("T", wallet, 160.0) for wallet in wallets]
    flows = [{"wallet": w, "timestamp": 160.0, "units": 100.0} for w in wallets]
    if sold:
        gone = 100.0 * (1.0 - retained)
        if gone > 0:
            flows += [{"wallet": w, "timestamp": 210.0, "units": -gone}
                      for w in wallets]
        flows += [{"wallet": w, "timestamp": 205.0, "units": -5.0} for w in wallets]
        # The absorbers must actually take the supply the cohort released,
        # or the reading is FAILED for a good reason and the fixture is
        # testing nothing it claims to.
        released = 25 * (5.0 + max(0.0, gone))
        flows += [{"wallet": f"b{i}", "timestamp": 220.0, "units": released / 5.0}
                  for i in range(5)]
    independence = {f"b{i}": (0.9 if absorbers_independent else 0.05)
                    for i in range(5)}
    skills = {}
    if chasers:
        entries += [Entry("T", f"c{i}", 215.0, capital_usd=100.0) for i in range(10)]
        skills = {f"c{i}": 0.05 for i in range(10)}
        skills.update({w: 0.9 for w in wallets})
    return evaluate_cohorts(entries, flows, independence, skills, as_of=400.0,
                            absorption_window=(200.0, 260.0))


class ModesAreChosenFromEvidenceNotBlended(unittest.TestCase):

    def test_nothing_observed_means_undecided(self):
        choice = choose_exit_mode(None)
        self.assertEqual(MODE_UNDECIDED, choice.mode)
        self.assertIn("neither mode is chosen from nothing", choice.detail)

    def test_absorbed_supply_with_the_cohort_in_earns_monster_hold(self):
        choice = choose_exit_mode(_cohort(retained=0.9), monster_probability=0.3)
        self.assertEqual(MODE_MONSTER, choice.mode)

    def test_captured_supply_does_not_earn_it(self):
        # The buyers were related to the sellers: inventory, not demand.
        choice = choose_exit_mode(
            _cohort(absorbers_independent=False, retained=0.9),
            monster_probability=0.9)
        self.assertEqual(MODE_RECYCLER, choice.mode)
        self.assertTrue(any("inventory moving" in r for r in choice.reasons))

    def test_late_chasers_into_skilled_selling_bank_the_position(self):
        choice = choose_exit_mode(_cohort(retained=0.3, chasers=True))
        self.assertEqual(MODE_RECYCLER, choice.mode)

    def test_holding_is_the_expensive_default_so_it_needs_positive_evidence(self):
        # Absorption reads fine but the cohort has largely gone: not enough.
        choice = choose_exit_mode(_cohort(retained=0.2), monster_probability=0.9)
        self.assertEqual(MODE_RECYCLER, choice.mode)


if __name__ == "__main__":
    unittest.main()
