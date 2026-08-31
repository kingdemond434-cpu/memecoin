"""Supervision of the desk's indispensable runtime loops.

A process whose health loop is alive and whose source consumer is dead looks
healthy to systemd while collecting nothing. That is the failure this module
exists to make impossible: every loop the desk cannot do without is started
by name, its exit is recorded, and an unexpected exit is fatal.

Failing the process is the correct recovery boundary. A loop cannot restart
itself into a consistent graph -- its peers hold half-built state that was
valid only while it was running -- but systemd can restart the whole desk
from a clean snapshot, and does.

Background tasks are the other half. They are not indispensable, so their
failure is counted and logged rather than fatal; a swallowed exception there
is how a subsystem stops working without anything saying so.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

#: A watchdog that has not completed a pass within this many seconds is not
#: watching anything, whatever its last recorded verdict says.
WATCHDOG_STALE_S = 180.0


class TaskSupervision:
    """Starts, watches, and reports on the loops the desk cannot do without."""

    def _spawn_background(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self._background_tasks.discard(done)
            if done.cancelled():
                return
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                self._background_failures += 1
                logger.error("background task failed", exc_info=(
                    type(exc), exc, exc.__traceback__))

        task.add_done_callback(completed)

    def _start_runtime_task(self, name: str, coroutine) -> asyncio.Task:
        """Start one indispensable loop and make an unexpected exit fatal.

        A process with a living health loop and a dead source consumer looks
        healthy to systemd while collecting nothing.  Failing the process is
        the correct recovery boundary because systemd can restart the complete
        graph from a consistent snapshot.
        """
        task = asyncio.create_task(coroutine, name=f"memecoin:{name}")
        self._task_health[name] = {
            "status": "RUNNING", "started_at": time.time(), "failures": 0}

        def completed(done: asyncio.Task) -> None:
            state = self._task_health.setdefault(name, {})
            if done.cancelled() or not self._running:
                state.update({"status": "STOPPED", "stopped_at": time.time()})
                return
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                exc = None
            detail = (f"{type(exc).__name__}: {exc}" if exc is not None
                      else "loop returned while the desk was running")
            state.update({"status": "FAILED", "failed_at": time.time(),
                          "detail": detail,
                          "failures": int(state.get("failures", 0)) + 1})
            self._fatal_task_detail = f"{name}: {detail}"
            logger.critical("critical runtime task failed: %s", self._fatal_task_detail)
            self._fatal_task_event.set()

        task.add_done_callback(completed)
        return task

    def runtime_task_report(self) -> Dict[str, Any]:
        states = {name: dict(value) for name, value in self._task_health.items()}
        failed = [name for name, value in states.items()
                  if value.get("status") == "FAILED"]
        return {
            "status": "CRITICAL" if failed else "OK" if states else "DATA_BLOCKED",
            "failed": failed,
            "fatal_detail": self._fatal_task_detail,
            "background_failures": self._background_failures,
            "tasks": states,
        }

    def watchdog_report(self) -> Dict[str, Any]:
        path = Path(self.global_config.get("ops_state_dir", "data/state")) \
            / "watchdog_state.json"
        try:
            payload = json.loads(path.read_text())
            age = max(0.0, time.time() - float(payload.get("last_run_at", 0.0)))
        except (OSError, ValueError, TypeError):
            return {"status": "DATA_BLOCKED",
                    "detail": "watchdog has not completed a run"}
        return {
            "status": "OK" if age <= WATCHDOG_STALE_S else "DEGRADED",
            "seconds_since_run": round(age, 1),
            "restart_count_window": len(payload.get("restart_times") or ()),
            "last_plan": payload.get("last_plan") or {},
            "last_failures": payload.get("last_failures") or [],
        }