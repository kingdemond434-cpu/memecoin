"""Phantom's native World scope: public research, never substitute venues.

Phantom currently publishes no external API for this mobile feature. Native
discovery continues independently when World's HTTP metadata denies access.
On-chain observations remain research candidates, not verified market
identity, executable quotes, paper fills or permission to sign transactions.
"""
import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import aiohttp

from src.chains.rpc_manager import ChainConfig, ChainType, RPCEndpointConfig, RPCManager
from src.research import world_native

SUPPORT = 'https://help.phantom.com/articles/trade-predictions-in-phantom-47187740851731'
DISCLOSURE = 'https://phantom.com/prediction-market-disclosures'
FRONTEND = 'https://world.xyz/assets/api-CcCaG8Rc.js'
CATALOG = 'https://markets-api-proxy.world-xyz.workers.dev/api/v1/events'
# Independent public-chain research identifies this candidate. Operator docs
# are access-restricted, so its relationship to a user's Phantom market must
# still be verified against that market's exact public transaction/contracts.
PROGRAM_CANDIDATE = world_native.PROGRAM_CANDIDATE
PROGRAM_SOURCE = world_native.PROGRAM_SOURCE
LOADER = 'BPFLoaderUpgradeab1e11111111111111111111111'


class PhantomResearchStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS phantom_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,at REAL NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL);
        ''')
        spec = {'scope': 'phantom_world_only', 'live_authority': False, 'executable_paper_fills': False,
                'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'native_implementation': hashlib.sha256(Path(world_native.__file__).read_bytes()).hexdigest()}
        self.spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        previous = self.db.execute("SELECT value FROM phantom_meta WHERE key='spec_hash'").fetchone()
        if previous and previous[0] != self.spec_hash:
            self.db.close()
            raise ValueError('Phantom research specification changed; retain this study and use a new directory')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO phantom_meta VALUES('spec_hash',?)", (self.spec_hash,))
            self.db.execute("INSERT OR IGNORE INTO phantom_meta VALUES('started_at',?)", (str(time.time()),))

    def event(self, kind, payload):
        with self.db:
            self.db.execute('INSERT INTO events(at,kind,payload) VALUES(?,?,?)',
                            (time.time(), kind, json.dumps(payload, sort_keys=True, allow_nan=False)))

    def close(self):
        self.db.close()


def documented_phantom_fee(contracts, price):
    """A documented fee scenario, not a quote or measured executable cost."""
    try: n, p = Decimal(str(contracts)), Decimal(str(price))
    except ArithmeticError: raise ValueError('numeric contracts and price required') from None
    if not n.is_finite() or not p.is_finite() or n <= 0 or n != n.to_integral_value() or not 0 <= p <= 1:
        raise ValueError('positive whole contracts and a finite binary price required')
    return {'usd': str(Decimal('.1')*p*(1-p)*n), 'classification': 'DOCUMENTED_FORMULA_NOT_OBSERVED_QUOTE',
            'source': SUPPORT, 'trade_authority': False}


def native_program_observation(response):
    if not isinstance(response, dict) or type(response.get('context', {}).get('slot')) is not int or response['context']['slot'] < 0:
        raise ValueError('native RPC context slot required')
    value = response.get('value')
    if not isinstance(value, dict) or value.get('owner') != LOADER or value.get('executable') is not True:
        raise ValueError('candidate is not an observed upgradeable executable program')
    data = value.get('data')
    if not isinstance(data, list) or len(data) != 2 or data[1] != 'base64': raise ValueError('native account encoding required')
    raw = base64.b64decode(data[0], validate=True)
    if len(raw) != 36 or int.from_bytes(raw[:4], 'little') != 2 or not any(raw[4:]): raise ValueError('unexpected executable program account layout')
    return {'slot': response['context']['slot'], 'program': PROGRAM_CANDIDATE, 'owner': LOADER,
            'account_sha256': hashlib.sha256(raw).hexdigest(), 'executable_observed': True,
            'operator_market_identity': 'UNVERIFIED', 'independent_source': PROGRAM_SOURCE,
            'trade_authority': False}


class PhantomPredictions:
    def __init__(self, store):
        self.store = store
        self.cycles, self.errors = 0, {}
        self.catalog_next = 0
        self.catalog_status = 'UNOBSERVED'
        self.program_status = 'UNOBSERVED'
        self.last_success = None
        self.native = world_native.NativeWorldCensus(store)

    async def cycle(self, session, rpc=None):
        if time.time() >= self.catalog_next:
            try:
                async with session.get(CATALOG, params={'withNestedMarkets': 'true', 'limit': 200,
                        'status': 'active', 'sortBy': 'volume24h', 'sortOrder': 'desc'}, allow_redirects=False) as response:
                    self.catalog_status = 'PUBLIC_METADATA_OBSERVED' if response.status == 200 else f'HTTP_{response.status}'
                    self.catalog_next = time.time() + (3600 if response.status in {401, 403, 429} else 300)
                    if response.status != 200:
                        self.errors['metadata'] = self.catalog_status
                        self.store.event('world_catalog_access', {'source': CATALOG, 'http_status': response.status,
                                                                  'no_authentication_bypass': True})
                    else:
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            raw.extend(chunk)
                            if len(raw) > 4_000_000: raise ValueError('bounded World metadata page required')
                        payload = json.loads(raw)
                        if not isinstance(payload, dict): raise ValueError('World catalog object required')
                        # Raw public page is retained, not translated into a
                        # phantom account, fee model, fill or settlement.
                        self.store.event('world_public_catalog', {'source': CATALOG, 'payload': payload,
                            'sha256': hashlib.sha256(raw).hexdigest(), 'market_contract_identity': 'UNVERIFIED',
                            'executable_quote': False})
                        self.errors.pop('metadata', None)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                self.errors['metadata'] = type(exc).__name__
                self.catalog_next = time.time()+300
        if rpc is not None:
            try:
                raw = await rpc.request('getAccountInfo', [PROGRAM_CANDIDATE, {'encoding': 'base64', 'commitment': 'finalized'}])
                proof = native_program_observation(raw)
                self.store.event('world_candidate_program', {'observation': proof, 'native_response': raw})
                self.program_status, self.last_success = 'EXECUTABLE_ACCOUNT_OBSERVED_IDENTITY_UNVERIFIED', time.time()
                self.errors.pop('native_rpc', None)
                await self.native.poll(rpc)
            except Exception as exc:
                self.program_status = 'DATA_BLOCKED'
                self.errors['native_rpc'] = type(exc).__name__
        else:
            self.program_status = 'DATA_BLOCKED_RPC_CONNECTION_REQUIRED'
        self.cycles += 1

    async def run(self):
        rpc = None
        key = os.getenv('HELIUS_API_KEY', '').strip()
        if key:
            endpoint = RPCEndpointConfig('https://mainnet.helius-rpc.com/?api-key='+quote(key, safe=''))
            rpc = RPCManager(ChainConfig(name='Solana World public research', chain_id='solana', chain_type=ChainType.SOLANA,
                rpc_endpoints=[endpoint], explorer_api='', explorer_key='', native_token='SOL', decimals=9,
                block_time=.4, factories={}, routers={}, base_tokens=[], min_liquidity_usd=0, max_tax=0, honeypot_check=False))
        try:
            if rpc: await rpc.start()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12), headers={'User-Agent': 'MemecoinPhantomResearch/1.0'}) as session:
                while True:
                    await self.cycle(session, rpc)
                    await asyncio.sleep(120)
        finally:
            if rpc: await rpc.stop()

    def report(self):
        started = float(self.store.db.execute("SELECT value FROM phantom_meta WHERE key='started_at'").fetchone()[0])
        return {'venue': 'phantom_world', 'mode': 'RESEARCH_ONLY', 'live_submission_locked': True,
                'scope': 'Phantom native World prediction markets only; no substitute venues',
                'started_at': started, 'spec_hash': self.store.spec_hash, 'cycles': self.cycles,
                'catalog_status': self.catalog_status, 'program_observation': self.program_status,
                'last_native_observation': self.last_success, 'source_errors': self.errors,
                'phantom_external_api': 'NOT_PUBLICLY_SUPPORTED', 'support_source': SUPPORT,
                'provider_disclosure': DISCLOSURE, 'metadata_host_provenance': FRONTEND,
                'native_market_discovery': self.native.report(),
                'authorized_agent_connector': {'url': 'https://api.paybox.sh/mcp?app=world',
                    'status': 'OWNER_OAUTH_CONNECTION_REQUIRED', 'source': 'https://world.xyz/',
                    'grants_or_wallet_access_observed': False},
                'execution_blockers': ['AUTHORIZED_PHANTOM_MARKET_MAPPING', 'FIRM_EXECUTABLE_QUOTES',
                    'EXACT_MARKET_RULES_AND_SETTLEMENT', 'REGIONAL_ELIGIBILITY'],
                'real_fills': 0, 'simulated_fills': 0, 'production_performance': 'UNMEASURED',
                'executable_paper_status': 'DATA_BLOCKED_AUTHORIZED_MARKET_MAPPING_QUOTES_RULES_AND_SETTLEMENT_REQUIRED',
                'markets': 0, 'snapshots': 0, 'order_counts': {}, 'accounts': [], 'settlement_records': 0,
                'regional_eligibility': 'UNVERIFIED', 'trade_authority': False}
