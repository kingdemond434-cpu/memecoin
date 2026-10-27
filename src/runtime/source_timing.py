"""Bounded process-local source service times, never end-to-end latency."""
from contextlib import contextmanager
import threading
import time

PHASES = ("canonical", "neural_intake", "claim_intake", "owner_hook",
          "delivery_ack", "onion_delivery", "canonical_ref", "journal_ack", "owner_index")

class SourceTiming:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}

    @contextmanager
    def measure(self, phase):
        if phase not in PHASES:
            raise ValueError("unknown source timing phase")
        started = time.perf_counter()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                row = self._values.setdefault(phase, {"attempts": 0, "completed": 0,
                    "interrupted": 0, "total_seconds": 0.0, "max_seconds": 0.0,
                    "last_seconds": 0.0})
                row["attempts"] += 1
                row["completed" if succeeded else "interrupted"] += 1
                row["total_seconds"] += elapsed
                row["max_seconds"] = max(row["max_seconds"], elapsed)
                row["last_seconds"] = elapsed

    def report(self):
        with self._lock:
            return {"scope": "process_local_service_time_not_source_latency",
                    "phases": {name: dict(row) for name, row in self._values.items()}}


def source_timing(owner):
    # Called on the owner loop before dispatching any worker using this object.
    value = getattr(owner, "_source_phase_timing", None)
    if value is None:
        value = SourceTiming()
        owner._source_phase_timing = value
    return value
