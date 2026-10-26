import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from src.chains.rpc_budget import SharedRPCCooldowns, RPCProviderCooling
from src.chains.rpc_manager import RPCManager, RPCEndpointConfig, ChainType


URL='https://user:fixture-password@mainnet.helius-rpc.com/path?api-key=fixture-secret'


def manager(monkeypatch,tmp_path,url=URL):
    monkeypatch.setenv('MEMECOIN_RPC_BUDGET_DB',str(tmp_path/'shared.sqlite'))
    monkeypatch.delenv('HELIUS_API_KEY',raising=False)
    return RPCManager(SimpleNamespace(name='solana',chain_type=ChainType.SOLANA,
        rpc_endpoints=[RPCEndpointConfig(url,ws_url='wss://rpc.example/fixture')]))


@pytest.mark.asyncio
async def test_restart_and_separate_managers_do_not_retry_into_refusal(monkeypatch,tmp_path):
    first=manager(monkeypatch,tmp_path)
    first._penalise(first.endpoints[0],429,3600)
    until=first.endpoints[0].cooldown_until
    await first.stop()
    second=manager(monkeypatch,tmp_path)
    try:
        assert second._select_endpoint() is None
        assert second.get_ws_urls()==[] and second.get_ws_url() is None
        assert second.endpoints[0].cooldown_until == until
        with pytest.raises(RuntimeError,match='No healthy RPC'):
            await second.request('getAccountInfo',[])
        with pytest.raises(RuntimeError,match='No healthy RPC'):
            await second.batch_request([{'method':'getSlot','id':1}])
        await second._check_endpoint(second.endpoints[0])
        assert await second.broadcast_request('sendTransaction',['fixture']) is None
        assert not second._sessions and second._session is None
        report=json.dumps(second.get_stats())
        assert 'fixture-secret' not in report and 'fixture-password' not in report and '/path' not in report
        assert not second.get_stats()['endpoints'][0]['available']
        # A late success in this process cannot erase the other process' refusal.
        second.endpoints[0].cooldown_until=0
        assert second._select_endpoint() is None
    finally:
        await second.stop()
    for path in tmp_path.iterdir():
        if path.is_file():
            assert b'fixture-secret' not in path.read_bytes() and b'fixture-password' not in path.read_bytes()


@pytest.mark.asyncio
async def test_refusal_during_budget_wait_is_rechecked_before_socket(monkeypatch,tmp_path):
    value=manager(monkeypatch,tmp_path)
    ep=value.endpoints[0]
    class Budget:
        async def acquire(self,methods):
            value._shared_cooldowns.defer(ep.endpoint,time.time()+600)
        def close(self): pass
    value._helius_budget=Budget()
    try:
        with pytest.raises(RPCProviderCooling) as found:
            await value._allocate(ep,['getAccountInfo'])
        assert found.value.retry_at > time.time() and not value._sessions
    finally:
        await value.stop()


def test_different_processes_extend_but_cannot_shorten_shared_deadline(tmp_path):
    path=tmp_path/'shared.sqlite'
    endpoint=RPCEndpointConfig(URL,headers={'Authorization':'fixture-header-secret'})
    state=SharedRPCCooldowns(path)
    state.defer(endpoint,1000)
    code=("from src.chains.rpc_budget import SharedRPCCooldowns; "
        "from src.chains.rpc_manager import RPCEndpointConfig; import sys; "
        "s=SharedRPCCooldowns(sys.argv[1]); "
        f"e=RPCEndpointConfig({URL!r},headers={{'Authorization':'fixture-header-secret'}}); "
        "s.defer(e,2000);s.defer(e,100);s.close()")
    subprocess.run([sys.executable,'-c',code,str(path)],check=True,timeout=20)
    try:
        assert state.until(endpoint)==2000
        assert state.until(RPCEndpointConfig('https://independent.example'))==0
    finally:
        state.close()
    assert b'fixture-header-secret' not in path.read_bytes()


@pytest.mark.asyncio
async def test_expired_refusal_releases_endpoint_without_resetting_other_refusals(monkeypatch,tmp_path):
    value=manager(monkeypatch,tmp_path)
    ep=value.endpoints[0]
    value._shared_cooldowns.defer(ep.endpoint,time.time()-1)
    try:
        assert value._select_endpoint() is ep
        value._shared_cooldowns.defer(ep.endpoint,time.time()+600)
        # Same host with an independently configured endpoint is not treated
        # as the same credential allocation; no automatic key rotation occurs.
        independent=RPCEndpointConfig('https://independent.example')
        assert value._shared_cooldowns.until(independent)==0
    finally:
        await value.stop()


@pytest.mark.parametrize('header,expected',[
    ('86400',86400),('0',0),('0.5',.5),('NaN',30),('inf',30),('-3',30),
    ('not-a-date',30),('',30),('Thu, 01 Jan 1970 01:00:00 GMT',2600),
    ('Thu, 01 Jan 1970 00:00:00 GMT',0)])
def test_retry_after_delays_and_dates_are_finite_and_not_clipped(monkeypatch,header,expected):
    monkeypatch.setattr('src.chains.rpc_manager.time.time',lambda:1000.)
    assert RPCManager._retry_after_seconds(SimpleNamespace(headers={'Retry-After':header}),30)==expected


@pytest.mark.parametrize('deadline',[float('nan'),float('inf'),-1])
def test_invalid_shared_deadlines_are_refused(tmp_path,deadline):
    state=SharedRPCCooldowns(tmp_path/'shared.sqlite')
    try:
        with pytest.raises(ValueError): state.defer(RPCEndpointConfig(URL),deadline)
    finally:
        state.close()
