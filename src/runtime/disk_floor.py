"""A free-space floor below which this desk stops growing its own files.

The memecoin desk shares a volume with a separate live trading system. On
2026-11-10 that volume had about 16 GB free while this desk's evidence ledger
alone was 18.4 GB and growing. A research desk filling the disk would take the
other system down with it, so the desk's durable writers consult this floor
and refuse to grow past it. Refusal uses each writer's existing failure path
(a counted evidence write failure; producer backpressure on the source
journal), so nothing already accepted is deleted, rewritten or acknowledged
early. What is lost while the floor holds is new intake, and that is reported.

The floor is `MEMECOIN_DISK_FLOOR_BYTES` (default 20 GiB). Zero disables it.
The free-space reading is cached briefly so the hot path pays one syscall
every few seconds, not one per row. An unreadable volume is treated as
unknown and does not block: the writer's own OSError handling still applies.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_FLOOR_BYTES = 20 * 1024 ** 3
CACHE_SECONDS = 5.0


def configured_floor() -> int:
    raw = os.getenv("MEMECOIN_DISK_FLOOR_BYTES", "")
    try:
        value = int(raw) if raw.strip() else DEFAULT_FLOOR_BYTES
    except ValueError:
        value = DEFAULT_FLOOR_BYTES
    return max(0, value)


class DiskFloor:
    """Cached below-floor check for one directory's volume."""

    def __init__(self, directory, *, floor_bytes: Optional[int] = None,
                 cache_seconds: float = CACHE_SECONDS, usage=shutil.disk_usage,
                 clock=time.monotonic):
        self.directory = Path(directory)
        self.floor_bytes = configured_floor() if floor_bytes is None else max(0, int(floor_bytes))
        self.cache_seconds = float(cache_seconds)
        self._usage, self._clock = usage, clock
        self._lock = threading.Lock()
        self._checked_at: Optional[float] = None
        self._free: Optional[int] = None
        self.refusals = 0
        self.read_failures = 0

    def free_bytes(self) -> Optional[int]:
        with self._lock:
            now = self._clock()
            if self._checked_at is None or now - self._checked_at >= self.cache_seconds:
                self._checked_at = now
                try:
                    target = self.directory if self.directory.exists() else self.directory.parent
                    self._free = int(self._usage(target).free)
                except OSError:
                    self._free = None
                    self.read_failures += 1
            return self._free

    def blocked(self) -> bool:
        """True when growth must be refused. Counts each refusal."""
        if not self.floor_bytes:
            return False
        free = self.free_bytes()
        if free is None or free >= self.floor_bytes:
            return False
        with self._lock:
            self.refusals += 1
        return True

    def report(self) -> Dict[str, Any]:
        free = self._free
        return {"floor_bytes": self.floor_bytes, "free_bytes": free,
                "below_floor": bool(self.floor_bytes and free is not None and free < self.floor_bytes),
                "refusals": self.refusals, "read_failures": self.read_failures}
