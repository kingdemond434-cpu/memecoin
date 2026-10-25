import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from solders.pubkey import Pubkey
from src.research.phantom_predictions import PhantomResearchStore
from src.research.world_native import (NativeWorldCensus, CASH_MINT, PROGRAM_CANDIDATE,
    MARKET_DISCRIMINATOR, TOKEN_2022_PROGRAM, decode_market_candidate, verify_market_relations)
from src.runtime.market_gateway import make_app
from src.chains.rpc_manager import RPCError


def key(n):
    return str(Pubkey.from_bytes(n.to_bytes(32, 'little')))


def account(raw, owner=TOKEN_2022_PROGRAM):
    return {'owner': owner, 'executable': False, 'data': [base64.b64encode(raw).decode(), 'base64']}


def fixture(n=1):
    market, operator, yes, no, vault = [key(n*10+i) for i in range(1,6)]
    raw = bytearray(320);raw[:8] = MARKET_DISCRIMINATOR
    for offset, value in zip((8,40,72,104,136), (operator,CASH_MINT,yes,no,vault)):
        raw[offset:offset+32] = bytes(Pubkey.from_string(value))
    row = {'pubkey': market, 'account': account(raw, PROGRAM_CANDIDATE)}
    def mint(address, authority, label):
        raw = bytearray(166)
        struct.pack_into('<I',raw,0,1);raw[4:36]=bytes(Pubkey.from_string(authority))
        raw[44:46]=bytes((6,1));raw[165]=1
        payload = bytes(32)+bytes(Pubkey.from_string(address))
        for value in (label,'YES','https://m.world.xyz/'+address):
            value=value.encode();payload+=struct.pack('<I',len(value))+value
        payload+=struct.pack('<I',0)
        return account(raw+struct.pack('<HH',19,len(payload))+payload)
    cash=bytearray(82);cash[44:46]=bytes((6,1))
    vault_raw=bytearray(165);vault_raw[:32]=bytes(Pubkey.from_string(CASH_MINT))
    vault_raw[32:64]=bytes(Pubkey.from_string(market));vault_raw[108]=1
    return row, {yes:mint(yes,market,'Outcome YES'),no:mint(no,market,'Outcome NO'),
                 CASH_MINT:account(cash),vault:account(vault_raw)}


class NativeLayoutTests(unittest.TestCase):
    def test_native_relationships_do_not_claim_rules_or_prices(self):
        row, accounts = fixture()
        candidate=decode_market_candidate(row,123)
        proof=verify_market_relations(candidate,accounts,124)
        self.assertEqual('Outcome YES',proof['outcomes']['yes_mint']['metadata']['name'])
        for flag in ('firm_quote_available','market_rules_verified','settlement_verified','trade_authority'):
            self.assertFalse(proof[flag])
        bad=dict(accounts);bad[candidate['cash_vault']]=None
        with self.assertRaises(ValueError): verify_market_relations(candidate,bad,124)
        bad=dict(accounts);bad[candidate['yes_mint']]=accounts[candidate['no_mint']]
        with self.assertRaisesRegex(ValueError,'binding'): verify_market_relations(candidate,bad,124)
        bad=dict(accounts);raw=bytearray(base64.b64decode(bad[candidate['cash_vault']]['data'][0]))
        raw[32:64]=bytes(Pubkey.from_string(key(999)))
        bad[candidate['cash_vault']]=account(raw)
        mismatch=verify_market_relations(candidate,bad,124)
        self.assertFalse(mismatch['reference_relationships_consistent'])
        self.assertEqual('VAULT_AUTHORITY_DIFFERS_FROM_REFERENCE',mismatch['relationship_status'])
        self.assertFalse(mismatch['trade_authority'])

    def test_layout_owner_collateral_and_authority_fail_closed(self):
        row, accounts=fixture()
        for offset in (0,40,72):
            altered=json.loads(json.dumps(row));raw=bytearray(base64.b64decode(altered['account']['data'][0]))
            raw[offset:offset+32]=bytes(32)
            altered['account']=account(raw,PROGRAM_CANDIDATE)
            with self.assertRaises(ValueError): decode_market_candidate(altered,123)
        row['account']['owner']=key(999)
        with self.assertRaises(ValueError): decode_market_candidate(row,123)
        row, accounts=fixture();candidate=decode_market_candidate(row,123)
        raw=bytearray(base64.b64decode(accounts[candidate['yes_mint']]['data'][0]));raw[4:36]=bytes(32)
        accounts[candidate['yes_mint']]=account(raw)
        with self.assertRaisesRegex(ValueError,'authority'): verify_market_relations(candidate,accounts,124)


class NativeCensusTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_small_provider_limit_keeps_complete_relationship_frames(self):
        fixtures=[fixture(n) for n in range(1,12)]
        accounts={k:v for _,group in fixtures for k,v in group.items()}
        class RPC:
            sizes=[]
            async def request(self,method,params):
                if method=='getProgramAccounts': return {'context':{'slot':123},'value':[row for row,_ in fixtures]}
                self.sizes.append(len(params[0]))
                if len(params[0])>10: raise RPCError({'code':-32602,'message':'maximum of 10 accounts'})
                return {'context':{'slot':124},'value':[accounts.get(k) for k in params[0]]}
        with tempfile.TemporaryDirectory() as temp:
            store=PhantomResearchStore(Path(temp)/'native.sqlite')
            try:
                rpc=RPC();census=NativeWorldCensus(store)
                await census.poll(rpc)
                self.assertEqual(11,census.report()['by_relationship_status']['CONSISTENT_NATIVE_RELATIONSHIPS'])
                self.assertEqual([34,10,10,10,7],rpc.sizes)
                restored=NativeWorldCensus(store)
                self.assertEqual(10,restored.state['account_batch_limit'])
                self.assertEqual(1,store.db.execute("SELECT count(*) FROM events WHERE kind='world_rpc_account_capacity'").fetchone()[0])
                self.assertFalse(restored.report()['firm_quotes_available'])
            finally: store.close()

    async def test_rate_limit_cannot_trigger_split_retry_or_remove_pending_candidates(self):
        row,accounts=fixture()
        class RPC:
            calls=0
            async def request(self,method,params):
                if method=='getProgramAccounts': return {'context':{'slot':123},'value':[row]}
                self.calls+=1
                raise RPCError({'code':429,'message':'maximum of 10 accounts, rate limited'})
        with tempfile.TemporaryDirectory() as temp:
            store=PhantomResearchStore(Path(temp)/'native.sqlite')
            try:
                rpc=RPC();census=NativeWorldCensus(store);await census.poll(rpc)
                self.assertEqual(1,rpc.calls)
                self.assertEqual({'PENDING':1},census.report()['by_relationship_status'])
                self.assertNotIn('account_batch_limit',census.state)
            finally: store.close()

    async def test_all_batches_resume_after_failure_without_resetting_pending(self):
        fixtures=[fixture(n) for n in range(1,71)]
        accounts={k:v for _,group in fixtures for k,v in group.items()}
        class RPC:
            calls=0
            failed=False
            async def request(self,method,params):
                if method=='getProgramAccounts': return {'context':{'slot':123},'value':[row for row,_ in fixtures]}
                self.calls+=1
                if self.calls==2 and not self.failed:
                    self.failed=True;raise TimeoutError()
                return {'context':{'slot':124},'value':[accounts.get(k) for k in params[0]]}
        with tempfile.TemporaryDirectory() as temp:
            store=PhantomResearchStore(Path(temp)/'native.sqlite');rpc=RPC();census=NativeWorldCensus(store)
            try:
                await census.poll(rpc)
                self.assertEqual(33,census.report()['by_relationship_status']['CONSISTENT_NATIVE_RELATIONSHIPS'])
                census.state['next_refresh']=0;census._save()
                census=NativeWorldCensus(store)
                await census.poll(rpc)
                self.assertEqual(70,census.report()['by_relationship_status']['CONSISTENT_NATIVE_RELATIONSHIPS'])
                self.assertEqual(4,rpc.calls)
                first=census.page(limit=40);second=census.page(first['next_cursor'],40)
                self.assertEqual(70,len(first['items'])+len(second['items']))
                self.assertFalse(second['has_more'])
                self.assertFalse(census.report()['firm_quotes_available'])
                async with TestClient(TestServer(make_app(store,token='x'*32,start_runner=False))) as client:
                    url='/api/phantom-predictions/markets?limit=40'
                    self.assertEqual(401,(await client.get(url,headers={'Host':'127.0.0.1:18082'})).status)
                    response=await client.get(url,headers={'Host':'127.0.0.1:18082','Authorization':'Bearer '+'x'*32})
                    self.assertEqual(40,len((await response.json())['items']))
            finally:
                # TestClient owns the connection once the app is created.
                try: store.close()
                except Exception: pass

    async def test_stale_or_misaligned_native_response_retains_pending(self):
        row,accounts=fixture()
        class RPC:
            async def request(self,method,params):
                if method=='getProgramAccounts': return {'context':{'slot':123},'value':[row]}
                return {'context':{'slot':122},'value':[accounts.get(k) for k in params[0]]}
        with tempfile.TemporaryDirectory() as temp:
            store=PhantomResearchStore(Path(temp)/'native.sqlite')
            try:
                census=NativeWorldCensus(store);await census.poll(RPC())
                self.assertEqual({'PENDING':1},census.report()['by_relationship_status'])
                self.assertEqual('NATIVE_SNAPSHOT_PARTIAL_OR_BLOCKED',census.report()['status'])
            finally: store.close()
