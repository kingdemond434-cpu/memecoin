from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.runtime import wiring


@pytest.mark.asyncio
async def test_missing_grpc_starts_rpc_fallback_after_real_decoders_register(monkeypatch):
    monkeypatch.delenv("YELLOWSTONE_GRPC_URL", raising=False)
    created = []
    class Stream:
        def __init__(self, rpc, programs):
            self.handlers = {}
            self.started = False
            created.append(self)
        def on(self, kind, handler):
            self.handlers.setdefault(kind, []).append(handler)
        async def start(self):
            assert self.handlers.get("program_data")
            assert self.handlers.get("transaction")
            self.started = True
    monkeypatch.setattr(wiring, "SolanaRpcProgramStream", Stream)
    desk = SimpleNamespace(offline=False, solana_rpc=object(), global_config={},
        _on_pump_event=AsyncMock(), _on_raydium_event=AsyncMock(), _on_launchpad_event=AsyncMock())
    await wiring.SubsystemWiring._setup_yellowstone(desk)
    assert len(created) == 1 and created[0].started
    assert desk.rpc_program_stream is created[0]
    assert desk.yellowstone.status == "DATA_BLOCKED"


@pytest.mark.asyncio
async def test_offline_smoke_never_starts_network_fallback(monkeypatch):
    monkeypatch.delenv("YELLOWSTONE_GRPC_URL", raising=False)
    def forbidden(*args):
        raise AssertionError("offline smoke attempted network stream")
    monkeypatch.setattr(wiring, "SolanaRpcProgramStream", forbidden)
    desk = SimpleNamespace(offline=True)
    await wiring.SubsystemWiring._setup_yellowstone(desk)
    assert desk.yellowstone.status_detail == "offline smoke-test mode"
