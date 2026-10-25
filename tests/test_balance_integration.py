import pytest

from src.chains.rpc_manager import RPCError
from src.research.chain_miners import account_balance_miner


class BatchRPC:
    def __init__(self): self.requests = []
    async def request(self,method,params):
        self.requests.append(list(params[0]))
        if len(params[0]) > 10:
            raise RPCError({'code':-32602,'message':'maximum of 10 accounts'})
        return {'context':{'slot':100+len(self.requests)}, 'value':[
            {'lamports':9_007_199_254_740_993+int(a), 'owner':'system', 'executable':False} for a in params[0]]}


@pytest.mark.asyncio
async def test_explicit_provider_bound_splits_without_omitting_or_misaligning_wallets():
    rpc = BatchRPC()
    fetch = account_balance_miner(rpc,lambda:[str(i) for i in range(25)])
    rows = await fetch()
    assert [r['address'] for r in rows] == [str(i) for i in range(25)]
    assert rows[0]['lamports'] == 9_007_199_254_740_993
    assert [len(r) for r in rpc.requests] == [25,10,10,5]
    assert rows[0]['rpc_context_slot'] != rows[-1]['rpc_context_slot']
    await fetch()
    assert [len(r) for r in rpc.requests[4:]] == [10,10,5]


@pytest.mark.asyncio
async def test_rate_limit_is_never_amplified_into_split_requests():
    class Limited:
        calls = 0
        async def request(self,*_):
            self.calls += 1
            raise RPCError({'code':429,'message':'maximum of 10 accounts; rate limited'})
    rpc = Limited()
    with pytest.raises(RPCError):
        await account_balance_miner(rpc,lambda:['a','b'])()
    assert rpc.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('values', [[{'lamports':1}], [{'lamports':None},None], [{'lamports':-1},None]])
async def test_missing_misaligned_or_invalid_balance_cannot_become_observed_zero(values):
    class BadRPC:
        async def request(self,*_): return {'value':values}
    with pytest.raises(ValueError):
        await account_balance_miner(BadRPC(),lambda:['a','b'])()
