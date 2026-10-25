import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from src.research.phantom_predictions import PhantomPredictions, PhantomResearchStore, CATALOG, LOADER, native_program_observation, documented_phantom_fee, _configured_world_rpc
from src.research.prediction_markets import PaperBroker
from src.runtime.market_gateway import make_app


def program():
    return {'context': {'slot': 123}, 'value': {'owner': LOADER, 'executable': True,
             'data': [base64.b64encode((2).to_bytes(4,'little')+bytes(range(32))).decode(), 'base64']}}


class NativeEvidenceTests(unittest.TestCase):
    def test_keyless_rpc_survives_absent_helius_and_demo_credentials_are_excluded(self):
        from urllib.parse import urlsplit
        with patch.dict('os.environ',{},clear=True):
            rpc = _configured_world_rpc()
        hosts = {urlsplit(ep.endpoint.url).hostname for ep in rpc.endpoints}
        self.assertIn('solana-rpc.publicnode.com',hosts)
        self.assertIn('api.mainnet-beta.solana.com',hosts)
        self.assertNotIn('mainnet.helius-rpc.com',hosts)
        self.assertNotIn('solana.leorpc.com',hosts)
        self.assertTrue(all('${' not in ep.endpoint.url for ep in rpc.endpoints))

    def test_owner_rpc_remains_in_same_pool_with_keyless_fallback(self):
        from urllib.parse import urlsplit
        with patch.dict('os.environ',{'HELIUS_API_KEY':'test-only-owner-value'},clear=True):
            rpc = _configured_world_rpc()
        hosts = {urlsplit(ep.endpoint.url).hostname for ep in rpc.endpoints}
        self.assertIn('mainnet.helius-rpc.com',hosts)
        self.assertIn('solana-rpc.publicnode.com',hosts)

    def test_program_presence_never_claims_verified_market_or_trade_authority(self):
        proof = native_program_observation(program())
        self.assertTrue(proof['executable_observed'])
        self.assertEqual('UNVERIFIED', proof['operator_market_identity'])
        self.assertFalse(proof['trade_authority'])
        for changes in ({'owner':'wrong'}, {'executable':False}, {'data':['AA==','base64']}):
            row = program();row['value'].update(changes)
            with self.assertRaises(ValueError): native_program_observation(row)
        row = program();row['context']['slot'] = True
        with self.assertRaises(ValueError): native_program_observation(row)

    def test_documented_fee_is_a_scenario_not_measured_execution(self):
        result = documented_phantom_fee(100, '.75')
        self.assertEqual('1.87500', result['usd'])
        self.assertFalse(result['trade_authority'])
        for n,p in ((1.5,.5),(1,'NaN'),(0,.5),(1,'bad'),(1,2)):
            with self.assertRaises(ValueError): documented_phantom_fee(n,p)


class NativeGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_scope_has_no_legacy_accounts_or_wallet_mining(self):
        with tempfile.TemporaryDirectory() as temp:
            store = PhantomResearchStore(Path(temp)/'native.sqlite')
            async with TestClient(TestServer(make_app(store, token='x'*32, start_runner=False))) as client:
                result = await (await client.get('/status', headers={'Host':'127.0.0.1:18082'})).json()
                self.assertEqual('phantom_world', result['prediction_markets']['venue'])
                self.assertEqual([], result['prediction_markets']['accounts'])
                self.assertEqual(0, result['prediction_markets']['simulated_fills'])
                self.assertIsNone(result['public_wallet_research'])
                names = {r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertNotIn('accounts', names)
                self.assertNotIn('public_trades', names)

    async def test_legacy_ledger_cannot_be_relabeled_as_phantom(self):
        with tempfile.TemporaryDirectory() as temp:
            broker = PaperBroker(Path(temp)/'legacy.sqlite')
            try:
                with self.assertRaisesRegex(ValueError,'cannot be relabeled'): make_app(broker, token='x'*32)
            finally: broker.close()

    async def test_unauthorized_metadata_is_backed_off_while_public_rpc_still_records(self):
        class Response:
            status = 401
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
        class Session:
            def __init__(self): self.requests=[]
            def get(self,url,**kwargs): self.requests.append(url);return Response()
        class RPC:
            async def request(self,method,params): return program()
        with tempfile.TemporaryDirectory() as temp:
            store = PhantomResearchStore(Path(temp)/'native.sqlite')
            try:
                observer,session = PhantomPredictions(store),Session()
                await observer.cycle(session,RPC())
                await observer.cycle(session,RPC())
                self.assertEqual([CATALOG],session.requests)
                result = observer.report()
                self.assertEqual('HTTP_401',result['catalog_status'])
                self.assertEqual(0,result['simulated_fills'])
                self.assertFalse(result['trade_authority'])
                self.assertEqual(2,store.db.execute("SELECT count(*) FROM events WHERE kind='world_candidate_program'").fetchone()[0])
            finally: store.close()

    async def test_study_restart_preserves_start_and_changed_spec_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'native.sqlite'
            store=PhantomResearchStore(path)
            first=PhantomPredictions(store).report()['started_at']
            store.close()
            store=PhantomResearchStore(path)
            self.assertEqual(first,PhantomPredictions(store).report()['started_at'])
            with store.db: store.db.execute("UPDATE phantom_meta SET value='changed' WHERE key='spec_hash'")
            store.close()
            with self.assertRaisesRegex(ValueError,'changed'): PhantomResearchStore(path)
