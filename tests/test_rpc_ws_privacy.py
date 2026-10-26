from types import SimpleNamespace
import logging

import pytest

from src.chains.yellowstone_grpc import SolanaRpcProgramStream

SECRET='fixture-credential-not-real'


@pytest.mark.asyncio
async def test_reconnect_logging_does_not_echo_transport_urls(monkeypatch, caplog):
    stream=SolanaRpcProgramStream(SimpleNamespace(get_ws_urls=lambda:['wss://provider.invalid/?apikey='+SECRET]), [])
    class Connection:
        async def __aenter__(self):
            raise RuntimeError('provider echoed wss://provider.invalid/?apikey='+SECRET)
        async def __aexit__(self,*args):
            pass
    monkeypatch.setattr('websockets.connect', lambda *args,**kwargs:Connection())
    async def stop_after_failure(seconds):
        stream._running=False
    monkeypatch.setattr('asyncio.sleep',stop_after_failure)
    caplog.set_level(logging.WARNING)
    stream._running=True
    await stream._ws_loop()
    assert 'RuntimeError' in caplog.text and SECRET not in caplog.text


@pytest.mark.asyncio
async def test_poll_failure_status_keeps_only_error_type(monkeypatch):
    stream=SolanaRpcProgramStream(SimpleNamespace(), [])
    async def fail():
        raise ValueError('https://provider.invalid/?apikey='+SECRET)
    async def stop_after_failure(seconds):
        stream._running=False
    monkeypatch.setattr(stream,'poll_once',fail)
    monkeypatch.setattr('asyncio.sleep',stop_after_failure)
    stream._running=True
    await stream._loop()
    status=stream.get_status()
    assert status['status']=='DEGRADED' and 'ValueError' in status['detail']
    assert SECRET not in str(status)
