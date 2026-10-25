import asyncio
from types import SimpleNamespace

import pytest
from src.chains.yellowstone_grpc import SolanaRpcProgramStream

@pytest.mark.asyncio
async def test_duplicate_launch_notifications_share_one_fetch_and_stop_drains_it():
    stream=SolanaRpcProgramStream(SimpleNamespace(),[])
    started=asyncio.Event()
    stopped=asyncio.Event()
    calls=[]
    async def fetch(signature):
        calls.append(signature)
        started.set()
        try:await asyncio.Future()
        finally:stopped.set()
    stream._fetch_confirmed_transaction=fetch
    assert stream._schedule_confirmed_transaction('one-launch')
    assert not stream._schedule_confirmed_transaction('one-launch')
    await started.wait()
    assert calls==['one-launch']
    assert stream.get_status()['duplicate_confirmation_fetches_avoided']==1
    await stream.stop()
    assert stopped.is_set() and not stream._pending_fetches

@pytest.mark.asyncio
async def test_a_temporarily_unavailable_polled_transaction_is_not_marked_seen():
    replies=[[],[{'signature':'new','err':None}],[{'signature':'new','err':None}]]
    transactions=[None,{'captured':'transaction'}]
    async def request(method,params):
        return replies.pop(0) if method=='getSignaturesForAddress' else transactions.pop(0)
    stream=SolanaRpcProgramStream(SimpleNamespace(request=request),['venue'])
    seen=[]
    stream.on('transaction',seen.append)
    await stream.poll_once()
    await stream.poll_once()
    assert 'new' not in stream._seen and not seen
    await stream.poll_once()
    assert 'new' in stream._seen and seen==[{'captured':'transaction'}]
