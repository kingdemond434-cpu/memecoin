"""The native receiver, running BESIDE the Python client until it earns trust.

`grpc.aio` put the desk's earliest information through the interpreter that
has to decide: a Python object per update, a WhichOneof, a dispatch and a
dict for every transaction on the chain -- thousands a second, almost all
discarded. The discarding was itself being done in the hot interpreter.

`solana_fastpath.NativeIngress` does the whole receive in Rust -- socket,
HTTP/2, prost decode, program filter, discriminator match, signature dedupe
-- and hands over only what survived, in batches, as tuples of primitives.

This module is the discipline around that, not the speed. The Rust
transaction builder had to reach byte parity before anything depended on it,
and this is the same bargain: a faster receiver that silently misses one
launch in a thousand is worse than the slower one that misses none, and the
only way to know which you have is to run both and compare.

    SHADOW    both run; Python decides; the native side is only counted
    AUTO      promoted after sustained agreement, demoted permanently on
              any miss the Python client saw and it did not

Demotion is a latch. A receiver that has once been shown to drop launches
does not get a second promotion on the strength of a quiet hour.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

NATIVE_INGRESS_SCHEMA_VERSION = "v1"

#: Matching events seen by BOTH before the native side may be promoted.
DEFAULT_PROMOTE_AFTER = 5_000

#: How long a signature stays in the parity window. The two receivers do not
#: see an event at the same instant, so a comparison made too early reports a
#: miss that is only a difference in arrival time.
PARITY_WINDOW_S = 30.0

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
MODE_AUTO = "AUTO"


@dataclass
class IngressEvent:
    """One matched instruction, still binary where binary is cheaper."""

    signature: bytes
    program: bytes
    discriminator: bytes
    fee_payer: bytes
    data: bytes
    accounts: Tuple[bytes, ...]
    slot: int
    received_ns: int
    is_vote: bool

    @classmethod
    def from_tuple(cls, row: Sequence[Any]) -> "IngressEvent":
        return cls(
            signature=row[0], program=row[1], discriminator=row[2],
            fee_payer=row[3], data=row[4], accounts=tuple(row[5]),
            slot=int(row[6]), received_ns=int(row[7]), is_vote=bool(row[8]))

    @property
    def signature_key(self) -> bytes:
        """The first eight bytes, which is what the Rust side dedupes on."""
        return self.signature[:8]


class NativeIngress:
    """Runs the Rust receiver and accounts for whether it can be trusted."""

    def __init__(self, endpoint: str, *, token: str = "",
                 programs: Sequence[str] = (),
                 mode: str = MODE_SHADOW,
                 promote_after: int = DEFAULT_PROMOTE_AFTER,
                 capacity: int = 8192):
        self.endpoint = str(endpoint or "")
        self.token = str(token or "")
        self.programs = tuple(programs)
        self.requested_mode = str(mode or MODE_OFF).upper()
        self.promote_after = int(promote_after)
        self.capacity = int(capacity)
        self._native: Optional[Any] = None
        self.available = False
        # Never blank. A report that says OFF with no reason is the shape of
        # a component nobody called -- which is exactly what happened the
        # first time this was wired, and the empty string hid it.
        self.unavailable_reason = "start() has not been called"
        self.mode = MODE_OFF
        self.demoted_reason = ""
        # Parity accounting. Keyed on the 8-byte signature prefix, with the
        # time it was seen, so a comparison is only made once both sides have
        # had the same chance.
        self._native_seen: Dict[bytes, float] = {}
        self._python_seen: Dict[bytes, float] = {}
        self.agreements = 0
        self.native_only = 0
        self.python_only = 0
        self.drained = 0
        self.started_at = 0.0

    # --- lifecycle -------------------------------------------------------

    def start(self) -> bool:
        """Begin receiving, or record precisely why not."""
        if self.requested_mode == MODE_OFF:
            self.unavailable_reason = "disabled by configuration"
            return False
        if not self.endpoint:
            self.unavailable_reason = "no endpoint configured"
            return False
        if not self.programs:
            self.unavailable_reason = "no programs to subscribe to"
            return False
        try:
            from solana_fastpath import NativeIngress as _Native
        except ImportError as exc:
            self.unavailable_reason = (
                f"the extension was built without the ingress feature ({exc}); "
                "rebuild with --features python,ingress")
            return False
        try:
            self._native = _Native(self.endpoint, None, self.capacity, 65_536)
            self._native.start(list(self.programs), self.token or None, 1)
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            self._native = None
            return False
        self.available = True
        # Cleared, because it is no longer true. Leaving a construction-time
        # reason in place after a successful start makes the report say
        # "start() has not been called" beside "available: True" -- two
        # fields of one document contradicting each other, which is worse
        # than either being wrong alone.
        self.unavailable_reason = ""
        self.mode = MODE_SHADOW
        self.started_at = time.time()
        logger.info(
            "NATIVE INGRESS running in SHADOW against %d program(s); it "
            "decides nothing until it has agreed with the Python client on "
            "%d events", len(self.programs), self.promote_after)
        return True

    def stop(self) -> None:
        if self._native is not None:
            try:
                self._native.stop()
            except Exception:  # pragma: no cover - teardown only
                pass
        self._native = None
        self.available = False
        self.mode = MODE_OFF

    # --- the drain -------------------------------------------------------

    def drain(self, max_events: int = 512) -> List[IngressEvent]:
        """Everything the Rust side has matched since the last call."""
        if self._native is None:
            return []
        try:
            rows = self._native.drain(int(max_events))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("native ingress drain failed: %s", exc)
            return []
        now = time.time()
        events = []
        for row in rows:
            event = IngressEvent.from_tuple(row)
            events.append(event)
            self._native_seen[event.signature_key] = now
        self.drained += len(events)
        self._reconcile(now)
        return events

    # --- parity ----------------------------------------------------------

    def note_python_event(self, signature: bytes) -> None:
        """Record that the REFERENCE client saw this signature.

        The Python client is the reference, not the competitor. Anything it
        saw and the native side did not is a miss, and misses are the only
        thing that matters here -- a receiver that is faster on the events it
        catches and blind to one launch in a thousand is not an improvement.
        """
        if not signature:
            return
        self._python_seen[bytes(signature[:8])] = time.time()

    def _reconcile(self, now: float) -> None:
        """Compare the two, but only on events old enough to be comparable."""
        cutoff = now - PARITY_WINDOW_S
        for key in [k for k, seen in self._python_seen.items() if seen < cutoff]:
            self._python_seen.pop(key, None)
            if key in self._native_seen:
                self._native_seen.pop(key, None)
                self.agreements += 1
                continue
            # The reference saw it and the native side did not, after a full
            # window. That is a miss.
            self.python_only += 1
            self._demote(f"missed {self.python_only} event(s) the Python "
                         f"client received")
        for key in [k for k, seen in self._native_seen.items() if seen < cutoff]:
            self._native_seen.pop(key, None)
            # Seen only natively is not a fault: the two subscribe with
            # different filters at different instants, and being EARLY looks
            # exactly like this. Counted, never punished.
            self.native_only += 1
        self._maybe_promote()

    def _maybe_promote(self) -> None:
        if self.mode != MODE_SHADOW or self.demoted_reason:
            return
        if self.agreements < self.promote_after:
            return
        self.mode = MODE_AUTO
        logger.info(
            "NATIVE INGRESS promoted to AUTO after %d agreements and %d "
            "misses", self.agreements, self.python_only)

    def _demote(self, reason: str) -> None:
        """Permanent. A receiver shown to drop launches does not get a rerun."""
        if self.demoted_reason:
            return
        self.demoted_reason = reason
        if self.mode == MODE_AUTO:
            logger.error("NATIVE INGRESS demoted from AUTO: %s", reason)
        self.mode = MODE_SHADOW

    # --- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        native = {}
        if self._native is not None:
            try:
                native = dict(self._native.report())
            except Exception:  # pragma: no cover - defensive
                native = {}
        total = self.agreements + self.python_only
        return {
            "schema": NATIVE_INGRESS_SCHEMA_VERSION,
            "status": ("OK" if self.mode == MODE_AUTO else
                       "SHADOW" if self.mode == MODE_SHADOW else
                       "OFF"),
            "mode": self.mode,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "promote_after": self.promote_after,
            "agreements": self.agreements,
            "missed_by_native": self.python_only,
            "native_only": self.native_only,
            "agreement_rate": (self.agreements / total) if total else None,
            "drained": self.drained,
            "demoted_reason": self.demoted_reason,
            "detail": ("the Python client remains the reference; this decides "
                       "nothing until it has matched it, and a single miss "
                       "demotes it permanently"),
            "native": native,
        }
