"""Startup concurrency/lifecycle controls; no external connections."""
import asyncio
from unittest.mock import patch

import pytest

from src.chains.rpc_manager import ChainRegistry


def registry(*names):
    result = ChainRegistry.__new__(ChainRegistry)
    result.chains = dict.fromkeys(names)
    result.rpc_managers = {}
    result._start_locks = {}
    return result


def test_independent_chains_start_concurrently_but_repeated_chain_is_shared():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        made = []

        class Manager:
            def __init__(self, config):
                made.append(self)
            async def start(self):
                if len(made) == 2:
                    entered.set()
                await release.wait()
            async def stop(self):
                pass

        value = registry("a", "b")
        with patch("src.chains.rpc_manager.RPCManager", Manager):
            tasks = [asyncio.create_task(value.start_chain(name)) for name in ("a", "a", "b")]
            await asyncio.wait_for(entered.wait(), 2)
            assert not value.rpc_managers  # Partially initialized clients stay private.
            release.set()
            results = await asyncio.gather(*tasks)
            assert results[0] is results[1]
            assert results[0] is not results[2]
            await value.start_all()
            assert len(made) == 2
            await value.stop_all()
            assert not value.rpc_managers
            await value.start_all()
            assert len(made) == 4
    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_start_closes_unpublished_manager_and_can_retry(cancel):
    async def scenario():
        entered = asyncio.Event()
        made = []

        class Manager:
            def __init__(self, config):
                self.closed = False
                made.append(self)
            async def start(self):
                entered.set()
                if cancel:
                    await asyncio.Event().wait()
                raise ValueError("fixture startup failure")
            async def stop(self):
                self.closed = True

        value = registry("a")
        with patch("src.chains.rpc_manager.RPCManager", Manager):
            task = asyncio.create_task(value.start_chain("a"))
            await entered.wait()
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else ValueError):
                await task
            assert made[0].closed
            assert not value.rpc_managers
            # A failed attempt must not leave a locked chain behind.
            second = asyncio.create_task(value.start_chain("a"))
            await asyncio.sleep(0)
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
            assert len(made) == 2 and made[1].closed
    asyncio.run(scenario())


def test_all_declared_chains_are_started_in_bounded_batches_and_empty_means_empty():
    async def scenario():
        active = peak = 0
        started = []

        class Manager:
            def __init__(self, config):
                pass
            async def start(self):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0)
                started.append(self)
                active -= 1
            async def stop(self):
                pass

        value = registry(*(str(i) for i in range(13)))
        with patch("src.chains.rpc_manager.RPCManager", Manager):
            await value.start_all([])
            assert not started
            await value.start_all()
            assert len(started) == 13 and len(value.rpc_managers) == 13
            assert 1 < peak <= 4
    asyncio.run(scenario())
