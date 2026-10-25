import base64
import json
import time
import zlib
from types import SimpleNamespace

import pytest

from src.research.perps_snapshots import _marks, _fetch_snapshots
from src.research.phantom_perps import PhantomPerpsResearch
from tests.test_phantom_perps import catalog


def frame(marks):
    compressor=zlib.compressobj(wbits=-15)
    raw=json.dumps(marks).encode()
    return {'channel':'fastAssetCtxs','data':base64.b64encode(compressor.compress(raw)+compressor.flush()).decode()}


def test_protocol_example_decodes_exactly_and_incomplete_fields_never_refresh_marks():
    example={'channel':'fastAssetCtxs','data':'q1ZyCnFWsqpWyk0syg6oULJSsjQ3NTDQM1Wq1VFyDfFAkTI2MzXQMwJLVVRWWfmFuTiiyBuamOoZKdXWAgA='}
    assert _marks(example)=={'BTC':'97500.5','ETH':'3650.25','xyz:NVDA':'145.2'}
    assert _marks(frame({'BTC':{'midPx':'100'},'ETH':{'markPx':'0'}}))=={}
    for bad in ({'channel':'wrong','data':example['data']},{'channel':'fastAssetCtxs','data':'AA=='},
                {'channel':'fastAssetCtxs','data':example['data'][:-5]}):
        with pytest.raises((ValueError,zlib.error)): _marks(bad)


@pytest.mark.asyncio
async def test_partial_snapshot_timeout_keeps_received_data_and_rotates_unserved_targets(tmp_path):
    state=PhantomPerpsResearch(tmp_path/'snapshot.sqlite')
    for coin in ('AAA','BBB','CCC'):
        state.ingest(catalog(dex=coin,coin=coin))
    sent=[]
    class Socket:
        count=0
        async def send_json(self,message): sent.append(message)
        async def receive_json(self):
            self.count+=1
            if self.count==1: return frame({'AAA':{'markPx':'100'}})
            raise TimeoutError('public socket interrupted')
    class Context:
        async def __aenter__(self): return Socket()
        async def __aexit__(self,*_): return False
    class Session:
        def ws_connect(self,*args,**kwargs): return Context()
    class HTTP:
        async def session(self): return Session()
    first=await _fetch_snapshots(state,HTTP())
    assert first[0]['perps_role']=='mark_snapshot' and first[1]['perps_role']=='request_error'
    assert sent[1]['subscription']['coin']=='AAA'
    state.ingest_batch(first,'canonical:ws-first')
    assert state.report()['derived_record_errors']==0
    sent.clear()
    await _fetch_snapshots(state,HTTP())
    assert sent[1]['subscription']['coin']=='BBB'
    state.close()


@pytest.mark.asyncio
async def test_reconnect_and_subscription_quota_survive_restart_without_network(tmp_path):
    path=tmp_path/'snapshot.sqlite'
    state=PhantomPerpsResearch(path)
    with state.db:
        state.db.executemany('INSERT INTO pp_ws_attempts VALUES(?,?)',[(time.time(),1)]*24)
    state.close()
    state=PhantomPerpsResearch(path)
    assert await _fetch_snapshots(state,None)==[]
    state.close()


def test_keyed_context_cannot_refresh_another_contract_or_invent_oracle_from_mark(tmp_path):
    state=PhantomPerpsResearch(tmp_path/'snapshot.sqlite')
    now=time.time()
    state.ingest(catalog(at=now))
    mark={'provider':'hyperliquid','product':'perpetual','perps_role':'mark_snapshot',
          'observed_at':now,'frame':frame({'BTC':{'markPx':'101'},'foreign':{'markPx':'200'}})}
    state.ingest_batch([mark],'canonical:marks')
    assert state.db.execute('SELECT coin FROM pp_marks').fetchall()==[('BTC',)]
    assert state.db.execute('SELECT count(*) FROM pp_oracles').fetchone()[0]==0
    ctx={**mark,'perps_role':'asset_context','frame':{'channel':'activeAssetCtx',
         'data':{'coin':'foreign','ctx':{'markPx':'999','oraclePx':'998'}}}}
    state.ingest_batch([ctx],'canonical:foreign-context')
    assert state.db.execute('SELECT count(*) FROM pp_oracles').fetchone()[0]==0
    state.close()
