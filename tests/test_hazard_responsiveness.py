import asyncio
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.wallet_value import FollowOutcome, WalletValueModel


class RankingWorkTests(unittest.TestCase):
    def test_rank_and_report_compute_the_same_prior_only_once(self):
        model = WalletValueModel(min_samples=2)
        for wallet in range(50):
            for _ in range(3):
                model.record(FollowOutcome(wallet=str(wallet), token='t', observed_at=1, executable_multiple=1.2 + wallet/100))
        expected = sorted((model.value(str(i)) for i in range(50)), key=lambda v: v.lower_bound, reverse=True)
        with patch.object(model, 'population_mean', wraps=model.population_mean) as prior:
            ranked = model.rank(limit=50, followable_only=False)
            self.assertEqual(1, prior.call_count)
            self.assertEqual([asdict(v) for v in expected], [asdict(v) for v in ranked])
            prior.reset_mock()
            model.report()
            self.assertEqual(1, prior.call_count)


class HazardFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_sweep_yields_and_does_not_recreate_an_evicted_token(self):
        wallet_intel = SimpleNamespace(followable_wallets=Mock(return_value={}))
        model = ContinuousRugHazardModel(SimpleNamespace(name='solana'), None, None, wallet_intel, None)
        for i in range(100): model.register_token(str(i))
        seen = []
        ready = asyncio.Event()
        async def compute(token, **kwargs):
            seen.append(token)
            if len(seen) == 1: ready.set()
            if len(seen) == 99: model._running = False
        model._compute_hazard = compute
        async def other_task():
            await ready.wait()
            self.assertLess(len(seen), 100)
            model.hazard_states.pop('1')
        model._running = True
        with patch('src.strategies.rug_hazard.asyncio.sleep', wraps=asyncio.sleep):
            monitor = asyncio.create_task(model._monitor_loop())
            try:
                await asyncio.wait_for(other_task(), 1)
                await asyncio.wait_for(monitor, 3)
            finally:
                model._running = False
                if not monitor.done(): monitor.cancel()
        self.assertNotIn('1', seen)
        self.assertEqual(99, len(seen))
        self.assertEqual(1, wallet_intel.followable_wallets.call_count)
