"""Factories the process-offload tests import BY PATH in a child process.

Module level, because the child is spawned rather than forked: it re-imports
this module and looks the factory up by name. A closure or a local class
would not survive that, which is exactly why ProcessOffloadedPool takes an
import path instead of an object.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List


class _TickingPool:
    """The smallest thing shaped like a miner pool: it emits records."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.on_records: Callable[[str, List[Dict[str, Any]]], None] = (
            lambda miner_id, records: None)
        self._task = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.ensure_future(self._tick())

    async def _tick(self) -> None:
        index = 0
        while self._running:
            self.on_records("test:ticker", [{
                "index": index,
                # The point of the test: this is the CHILD's pid.
                "pid": os.getpid(),
                "label": self.config.get("label", ""),
            }])
            index += 1
            await asyncio.sleep(float(self.config.get("interval_s", 0.02)))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()


def build_ticking_pool(config: Dict[str, Any]):
    return _TickingPool(config)


def build_exploding_pool(config: Dict[str, Any]):
    class _Exploding(_TickingPool):
        async def start(self) -> None:
            raise RuntimeError("the child could not build its pool")

    return _Exploding(config)


def build_dying_pool(config: Dict[str, Any]):
    """Emits a few records and then takes the process down."""

    class _Dying(_TickingPool):
        async def _tick(self) -> None:
            for index in range(3):
                self.on_records("test:dying", [{"index": index, "pid": os.getpid()}])
                await asyncio.sleep(0.01)
            os._exit(9)

    return _Dying(config)
