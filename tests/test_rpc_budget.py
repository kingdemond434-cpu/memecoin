import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.chains.rpc_budget import RPCBudget, RPCBudgetExhausted
from src.chains.rpc_manager import RPCManager, RPCEndpointConfig, _host_of


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'quota.sqlite'
        self.quota = RPCBudget(self.path, 'fixture-key', rps=5, daily=20, monthly=30)
        self.addCleanup(self.quota.close)

    def test_restart_and_separate_connection_share_rate_and_allocation(self):
        other = RPCBudget(self.path, 'fixture-key', rps=5, daily=20, monthly=30)
        try:
            self.assertEqual(0, self.quota.reserve(['getAccountInfo'], now=1000))
            self.assertGreater(other.reserve(['getAccountInfo'], now=1000), 0)
            self.assertEqual(0, other.reserve(['getAccountInfo'], now=1001))
            self.assertEqual(2, self.quota.db.execute('SELECT daily FROM rpc_allocation').fetchone()[0])
        finally: other.close()

    def test_archival_cost_and_daily_and_monthly_limits_are_distinct(self):
        self.quota.reserve(['getTransaction'], now=86400)
        self.quota.reserve(['getTransaction'], now=86401)
        with self.assertRaises(RPCBudgetExhausted): self.quota.reserve(['getAccountInfo'], now=86402)
        self.quota.reserve(['getTransaction'], now=172800)
        with self.assertRaises(RPCBudgetExhausted): self.quota.reserve(['getAccountInfo'], now=172801)
        self.assertEqual(0, self.quota.reserve(['getAccountInfo'], now=32*86400))

    def test_send_rate_is_one_per_second_and_cancelled_waits_do_not_spend(self):
        self.quota.reserve(['sendTransaction'], now=1000)
        self.assertAlmostEqual(.8, self.quota.reserve(['sendTransaction'], now=1000.2))
        self.assertEqual(1, self.quota.db.execute('SELECT daily FROM rpc_allocation').fetchone()[0])
        self.assertNotIn('fixture-key', self.path.read_bytes().decode('latin1'))


class RPCIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        manager = RPCManager(SimpleNamespace(name='solana', rpc_endpoints=[RPCEndpointConfig('https://user:fixture-password@rpc.example/path?api-key=fixture-key')]))
        return manager

    async def test_all_cooling_does_not_make_another_network_request(self):
        manager = self.manager()
        manager.endpoints[0].cooldown_until = 99999999999
        with self.assertRaisesRegex(RuntimeError, 'No healthy RPC'):
            await manager.request('getAccountInfo', [])
        self.assertIsNone(manager._session)
        self.assertEqual('rpc.example', _host_of(manager.endpoints[0].endpoint.url))

    async def test_network_exception_never_exports_key_url_or_userinfo(self):
        manager = self.manager()
        class Session:
            calls = 0
            def post(self, *args, **kwargs): raise ValueError('secret URL '+manager.endpoints[0].endpoint.url)
        session = Session()
        with patch.object(manager, '_session_for_loop', return_value=session):
            with self.assertRaises(Exception) as raised:
                await manager.request('getAccountInfo', [])
        message = str(raised.exception)
        self.assertNotIn('fixture-key', message)
        self.assertNotIn('fixture-password', message)
        self.assertNotIn('/path', message)
