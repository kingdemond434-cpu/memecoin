"""Bounded process-local source service times, never end-to-end latency."""
from contextlib import contextmanager
import threading
import time

PHASES = ("canonical", "neural_intake", "claim_intake", "owner_hook",
          "delivery_ack", "onion_delivery", "canonical_ref", "journal_ack", "owner_index",
          "canonical_ref_dispatch_wait", "canonical_ref_worker",
          "journal_ack_dispatch_wait", "journal_ack_worker",
          "canonical_ack", "canonical_ack_dispatch_wait", "canonical_ack_worker",
          "canonical_ref_inline", "journal_ack_inline",
          "canonical_ref_owner_resume", "journal_ack_owner_resume",
          "canonical_ack_owner_resume")

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
            self._record(phase, elapsed, succeeded)

    def _record(self, phase, elapsed, succeeded):
        with self._lock:
            row = self._values.setdefault(phase, {"attempts": 0, "completed": 0,
                "interrupted": 0, "total_seconds": 0.0, "max_seconds": 0.0,
                "last_seconds": 0.0})
            row["attempts"] += 1
            row["completed" if succeeded else "interrupted"] += 1
            row["total_seconds"] += elapsed
            row["max_seconds"] = max(row["max_seconds"], elapsed)
            row["last_seconds"] = elapsed

    def worker(self, phase, call, *args):
        """Wrap only the reviewed disk calls, before scheduling to_thread.

        Dispatch wait includes owner scheduling and executor wait until worker
        entry. Worker time includes locks/SQLite/commit, not pure CPU time.
        """
        if phase not in ("canonical_ref", "journal_ack", "canonical_ack"):
            raise ValueError("unknown source dispatch phase")
        queued = time.perf_counter()
        def run():
            self._record(phase + "_dispatch_wait", max(0.0, time.perf_counter() - queued), True)
            try:
                with self.measure(phase + "_worker"):
                    return call(*args)
            finally:
                run.finished_at = time.perf_counter()
        run.finished_at = None
        return run

    def resumed(self, phase, worker, succeeded):
        """Record worker completion to caller return, including cancellation drain.

        This includes completion notification and owner scheduling, not pure
        loop lag. Missing completion (e.g. cancelled before dispatch) is omitted.
        Outcome describes caller return, never durable transaction success.
        """
        if phase not in ("canonical_ref", "journal_ack", "canonical_ack"):
            raise ValueError("unknown source dispatch phase")
        finished = worker.finished_at
        if finished is not None:
            self._record(phase + "_owner_resume",
                         max(0.0, time.perf_counter() - finished), succeeded)

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
