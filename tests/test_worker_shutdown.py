import asyncio
import threading
from types import SimpleNamespace

import pytest

from src.chains.rpc_manager import RPCManager
from src.collectors.transports import HttpClient
from src.runtime.offload import OffloadedPool
from src.runtime.research_runtime import ResearchRuntime


@pytest.mark.asyncio
async def test_worker_closes_its_http_and_rpc_sessions_before_loop_dies():
    http = HttpClient()
    rpc = RPCManager(SimpleNamespace(rpc_endpoints=[]))
    main_http = await http.session()
    main_rpc = rpc.session
    ready = threading.Event()
    sessions, cleanup_loops = [], []

    class Pool:
        async def start(self):
            sessions.extend([await http.session(), rpc.session])
            ready.set()
        async def stop(self):
            pass

    desk = SimpleNamespace(http_client=http, solana_rpc=rpc)
    async def cleanup():
        cleanup_loops.append(asyncio.get_running_loop())
        await ResearchRuntime._close_research_worker_resources(desk)

    worker = OffloadedPool(Pool(), sink=lambda *args: None, on_loop_stop=cleanup)
    try:
        await worker.start()
        assert await asyncio.to_thread(ready.wait, 5)
        await worker.stop()
        assert len(sessions) == 2 and all(session.closed for session in sessions)
        assert cleanup_loops[0].is_closed()
        assert not main_http.closed and not main_rpc.closed
    finally:
        await worker.stop()
        await http.close()
        await rpc.stop()
