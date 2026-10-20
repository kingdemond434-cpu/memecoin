"""T0 must not wait on the network. This is the test that keeps it that way.

The desk optimises signer IPC from 505 to 235 microseconds and decodes the
chain in Rust to save nanoseconds, and all of it sat behind
`await self.rug_detector.analyze(...)` -- three to five sequential JSON-RPC
round trips -- placed directly in front of the decision, plus a Jupiter
quote per candidate to re-learn the SOL price. On a remote endpoint that is
one to five hundred milliseconds of waiting on the one launch the desk is
trying to be first to: six orders of magnitude more than everything the
whole fast path saves.

Removing the checks would be the wrong fix. Unmeasured safety is not safe,
and the screen already prices that -- a DATA_BLOCKED report enters at 35% of
size. What changes here is only WHEN the network is waited on: never in
front of the decision, always beside it, with the completed audit landing in
time for the next checkpoint on the ladder.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from src.detection.rug_detector import RiskLevel
from src.detection.t0_risk import (
    INVARIANT_FREEZE_AUTHORITY, INVARIANT_MINT_AUTHORITY, MIN_OBSERVATIONS,
    LaunchInvariantLedger, T0RiskView)


class _Report:
    """The shape of a completed audit, as far as the ledger reads it."""

    def __init__(self, mint_authority=False, freeze_authority=False,
                 token="Mint1", blocked=(), extensions=()):
        self.token_address = token
        self.blocked_checks = list(blocked)
        self.token_program = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        self.token_extensions = list(extensions)
        self.checks = {"mint": {
            "mint_authority_present": mint_authority,
            "freeze_authority_present": freeze_authority,
        }}


class AClaimIsEarnedNotAssumed(unittest.TestCase):

    def test_one_observation_is_not_enough_to_lean_on(self):
        ledger = LaunchInvariantLedger()
        ledger.observe_report("pump", _Report())
        self.assertFalse(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))

    def test_enough_agreeing_observations_earn_it(self):
        ledger = LaunchInvariantLedger()
        for index in range(MIN_OBSERVATIONS):
            ledger.observe_report("pump", _Report(token=f"M{index}"))
        self.assertTrue(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))
        self.assertTrue(ledger.holds("pump", INVARIANT_FREEZE_AUTHORITY))

    def test_one_counterexample_withdraws_it_permanently(self):
        # These are claims about what a PROGRAM does, not statistical
        # regularities. One counterexample means the claim was wrong or the
        # program changed, and a rate-based threshold would let an upgrade
        # that starts leaving mint authority live look like noise for hours.
        ledger = LaunchInvariantLedger()
        for index in range(MIN_OBSERVATIONS):
            ledger.observe_report("pump", _Report(token=f"M{index}"))
        self.assertTrue(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))

        ledger.observe_report("pump", _Report(mint_authority=True, token="Bad"))
        self.assertFalse(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))
        self.assertEqual("Bad",
                         ledger.state("pump", INVARIANT_MINT_AUTHORITY).withdrawn_example)

        # And no amount of subsequent agreement wins it back.
        for index in range(MIN_OBSERVATIONS * 3):
            ledger.observe_report("pump", _Report(token=f"L{index}"))
        self.assertFalse(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))
        # The freeze claim is untouched: they are separate claims.
        self.assertTrue(ledger.holds("pump", INVARIANT_FREEZE_AUTHORITY))

    def test_a_blocked_report_settles_nothing(self):
        # A check that could not run is not a counterexample. Counting it as
        # one would withdraw every claim the first time an endpoint
        # rate-limited us -- exactly when the local view matters most.
        ledger = LaunchInvariantLedger()
        for index in range(MIN_OBSERVATIONS):
            ledger.observe_report("pump", _Report(token=f"M{index}"))
        ledger.observe_report("pump", _Report(mint_authority=True,
                                              blocked=("mint_account",)))
        self.assertTrue(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))

    def test_claims_do_not_leak_between_programs(self):
        ledger = LaunchInvariantLedger()
        for index in range(MIN_OBSERVATIONS):
            ledger.observe_report("pump", _Report(token=f"M{index}"))
        self.assertTrue(ledger.holds("pump", INVARIANT_MINT_AUTHORITY))
        self.assertFalse(ledger.holds("raydium", INVARIANT_MINT_AUTHORITY))

    def test_the_ledger_survives_a_restart(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invariants.json"
            ledger = LaunchInvariantLedger(path)
            for index in range(MIN_OBSERVATIONS):
                ledger.observe_report("pump", _Report(token=f"M{index}"))
            self.assertTrue(ledger.save())
            # Two hundred observations thrown away on every restart is two
            # hundred that never accumulate.
            reloaded = LaunchInvariantLedger(path)
            self.assertTrue(reloaded.holds("pump", INVARIANT_MINT_AUTHORITY))


class _Curve:
    virtual_sol_reserves = 30_000_000_000
    virtual_token_reserves = 1_073_000_000_000_000
    complete = False


class TheLocalViewSaysWhatItDoesNotKnow(unittest.TestCase):

    def _view(self, ledger=None, curve=True):
        return T0RiskView(
            ledger or LaunchInvariantLedger(),
            curve_state_provider=(lambda _t: _Curve()) if curve else (lambda _t: None),
            risk_level_enum=RiskLevel)

    def test_with_no_invariants_everything_is_blocked_by_name(self):
        risk = self._view().assess("Mint1", "pump")
        self.assertEqual("DATA_BLOCKED", risk.data_status)
        for name in ("mint_authority", "freeze_authority", "holders"):
            self.assertIn(name, risk.blocked_checks)
        # Unmeasured is not safe: it must not claim the authorities are gone.
        self.assertTrue(risk.can_mint)
        self.assertTrue(risk.can_freeze)

    def test_an_earned_invariant_answers_without_a_round_trip(self):
        ledger = LaunchInvariantLedger()
        for index in range(MIN_OBSERVATIONS):
            ledger.observe_report("pump", _Report(token=f"M{index}"))
        risk = self._view(ledger).assess("Mint1", "pump")
        self.assertFalse(risk.can_mint)
        self.assertFalse(risk.can_freeze)
        self.assertEqual("program_invariant", risk.checks["mint_authority"]["source"])
        self.assertNotIn("mint_authority", risk.blocked_checks)

    def test_a_token_on_its_curve_has_a_sell_route_by_construction(self):
        # The curve is the counterparty and it cannot refuse. The router has
        # never indexed a mint seconds old, and its ignorance was previously
        # recorded as a confident "no route".
        risk = self._view().assess("Mint1", "pump")
        self.assertTrue(risk.sell_route_feasible)
        self.assertNotIn("sell_route", risk.blocked_checks)

    def test_without_a_curve_the_sell_route_is_unknown_not_false(self):
        risk = self._view(curve=False).assess("Mint1", "pump")
        self.assertIsNone(risk.sell_route_feasible)
        self.assertIn("sell_route", risk.blocked_checks)

    def test_it_never_invents_a_score(self):
        # A fabricated score would flow into the dataset as though it had
        # been measured, and every model downstream would train on it.
        risk = self._view().assess("Mint1", "pump")
        self.assertEqual(0.0, risk.score)
        self.assertTrue(risk.provisional)


class TheDecisionPathDoesNotWaitOnTheNetwork(unittest.IsolatedAsyncioTestCase):
    """The assertion that matters, made against the real desk.

    Every outbound call is made to raise. If anything on the T0 path still
    awaits the network, this test sees the exception rather than a decision.
    """

    MINT = "So11111111111111111111111111111111111111112"

    async def test_a_candidate_is_decided_with_every_rpc_poisoned(self):
        from src.detection.token_detector import DetectionSource, TokenCandidate
        from src.main import MemecoinQuantDesk, WSOL_MINT

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()

        calls = []

        async def poisoned(method, params):
            calls.append(method)
            raise AssertionError(
                f"T0 awaited the network: {method}. The full audit belongs "
                "beside the decision, never in front of it.")

        desk.solana_rpc.request = poisoned
        desk.rug_detector.rpc.request = poisoned

        candidate = TokenCandidate(
            address=self.MINT, chain="solana", source=DetectionSource.FACTORY,
            block_number=0, deployer="Deployer111111111111111111111111111111",
            factory="pump", pair="Pair1111111111111111111111111111111111",
            base_token=WSOL_MINT, timestamp=time.time())

        # The decision itself. It may decline -- almost everything does, and
        # DATA_BLOCKED is a legitimate outcome -- but it must REACH a verdict
        # without a round trip.
        await desk._evaluate_candidate(candidate)

        # The enrichment task the T0 path scheduled is where the network call
        # belongs. Draining it here proves it was scheduled, and that its
        # failure cannot propagate back into the decision.
        pending = [task for task in list(desk._risk_enrichment.values())]
        self.assertTrue(pending, "no enrichment was scheduled beside the decision")
        await asyncio.gather(*pending, return_exceptions=True)
        self.assertTrue(calls, "the audit never ran at all, even off the path")

    async def test_the_sol_price_is_read_not_fetched(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        desk._portfolio_refreshed_at = time.time()
        # Fresh: nothing is scheduled, and nothing is awaited.
        self.assertFalse(desk._ensure_portfolio_fresh())
        self.assertLess(desk.sol_price_age_s, 1.0)
        # Stale: a refresh is started BESIDE the decision, not awaited by it.
        desk._portfolio_refreshed_at = time.time() - 10_000
        self.assertTrue(desk._ensure_portfolio_fresh())
        self.assertGreater(desk.sol_price_age_s, 1_000)
        task = desk._portfolio_refresh_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
