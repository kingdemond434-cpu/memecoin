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

    OFF       never started
    SHADOW    both run; Python decides; the native side is only counted, and
              it stays a shadow FOR EVER -- a configured SHADOW that quietly
              promotes itself is not a shadow
    AUTO      starts in shadow and may promote itself after sustained
              agreement; demoted permanently on any miss the Python client
              saw and it did not
    RUST      authoritative from the first event, for an operator who has
              already read a parity ledger and decided. Announced loudly,
              because nothing has been proven on this run.

Demotion is a latch. A receiver that has once been shown to drop launches
does not get a second promotion on the strength of a quiet hour.

Both sides must agree on WHAT a signature is. The Rust side dedupes on the
first eight bytes of the raw 64-byte Ed25519 signature; the Python decoder
converts that same signature to base58 text before it ever reaches this
module. Those are not the same eight bytes, and feeding one into a ledger
keyed on the other produces a parity ledger in which nothing ever matches:
every event the reference client saw reads as a miss, the latch fires inside
the first window, and the native path is permanently demoted before it has
been given a chance to be wrong. `canonical_signature_key` is the single
place where both representations are reduced to the same identity.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

NATIVE_INGRESS_SCHEMA_VERSION = "v2"

#: Matching events seen by BOTH before the native side may be promoted.
DEFAULT_PROMOTE_AFTER = 5_000

#: How long a signature stays in the parity window. The two receivers do not
#: see an event at the same instant, so a comparison made too early reports a
#: miss that is only a difference in arrival time.
PARITY_WINDOW_S = 30.0

#: The reference client is already streaming when the native side subscribes,
#: and a gRPC subscription takes a moment to be served. Events in that gap are
#: genuinely invisible to the native receiver through no fault of its own, and
#: punishing them with a permanent latch means AUTO is unreachable by
#: construction -- the same "looks wired, concludes nothing" failure the
#: parity ledger exists to prevent.
PARITY_WARMUP_S = 60.0

#: Misses that coincide with a stream reconnect are attributed to the
#: reconnect rather than the decoder -- but only up to this share of the
#: agreements, so "it was reconnecting" cannot become a permanent excuse.
MAX_RECONNECT_MISS_SHARE = 0.01

#: How many lead-time samples to keep. The point of the whole exercise is a
#: NUMBER for how much earlier the native path sees a launch; without it the
#: shadow can prove correctness and still not justify promotion.
LEAD_SAMPLES = 4096

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
MODE_AUTO = "AUTO"
MODE_RUST = "RUST"

VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_AUTO, MODE_RUST)

#: Modes in which the native side is allowed to become authoritative.
PROMOTABLE_MODES = (MODE_AUTO, MODE_RUST)

_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {char: value for value, char in enumerate(_B58_ALPHABET)}

#: A raw Ed25519 signature. Anything of exactly this length in bytes is taken
#: as raw and never as text -- no base58 encoding of a 64-byte signature is
#: 64 characters long (they are 86 to 88), so the two cannot be confused.
RAW_SIGNATURE_LEN = 64

#: The key both sides are reduced to: the first eight bytes of the raw
#: signature, which is exactly what the Rust sink dedupes on.
KEY_LEN = 8


def _b58decode(text: str) -> bytes:
    """Base58 to raw bytes, or ValueError. Deliberately not silent.

    A signature that cannot be decoded is not a signature, and swallowing it
    would put a zero-length key in the ledger where it would match every
    other undecodable one and manufacture agreements out of failures.
    """
    number = 0
    for char in text.encode("ascii", "strict"):
        try:
            number = number * 58 + _B58_INDEX[char]
        except KeyError:
            raise ValueError(f"not base58: {char!r}") from None
    raw = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    zeros = len(text) - len(text.lstrip("1"))
    return b"\0" * zeros + raw


def canonical_signature_key(signature: Any) -> bytes:
    """Reduce any representation of a signature to the SAME eight bytes.

    Three things arrive here and they look nothing alike:

      * ``bytes`` of length 64 -- the raw signature, as Rust delivers it
      * ``str`` -- base58, as the Python Yellowstone decoder produces it
      * ``bytes`` holding base58 ASCII -- the same string after a caller
        helpfully called ``.encode()`` on it

    The third is the one that caused the bug this function exists to close:
    ``b"5Kj7..."[:8]`` and ``raw[:8]`` are different key spaces, so the
    parity ledger matched nothing and the latch fired on the first launch.

    Returns b"" when the input cannot be resolved, so callers can COUNT the
    failure rather than record a key that is quietly wrong.
    """
    if signature is None:
        return b""
    if isinstance(signature, str):
        try:
            raw = _b58decode(signature)
        except (ValueError, UnicodeEncodeError):
            return b""
        return raw[:KEY_LEN] if len(raw) >= KEY_LEN else b""
    if isinstance(signature, (bytes, bytearray, memoryview)):
        raw = bytes(signature)
        if not raw:
            return b""
        if len(raw) == RAW_SIGNATURE_LEN:
            return raw[:KEY_LEN]
        # Not 64 bytes. Either base58 text that was encoded, or a truncated
        # raw signature. Text is decidable: every byte is in the alphabet and
        # the length is in the range base58 of 64 bytes can occupy.
        if 40 <= len(raw) <= 96 and all(char in _B58_INDEX for char in raw):
            try:
                decoded = _b58decode(raw.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                decoded = b""
            if len(decoded) >= KEY_LEN:
                return decoded[:KEY_LEN]
            return b""
        if len(raw) >= KEY_LEN:
            # A raw signature that was already truncated somewhere upstream.
            return raw[:KEY_LEN]
        return b""
    return b""


def normalise_mode(mode: Any) -> str:
    """An unrecognised mode is OFF, loudly -- never a silent SHADOW."""
    text = str(mode or "").strip().upper()
    if text in VALID_MODES:
        return text
    if text:
        logger.warning(
            "native ingress mode %r is not one of %s; treating as OFF",
            mode, ", ".join(VALID_MODES))
    return MODE_OFF


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
    #: The event already DECODED by the Rust side, when it recognised one.
    #: None is the common case and a real answer: most instructions on the
    #: program are the outer half of a buy or sell whose inner CPI event is
    #: the one that carries the fill. Its presence is what lets a consumer
    #: tell a launch from a trade WITHOUT decoding -- which is the point,
    #: because the interpreter was otherwise decoding events it was about to
    #: discard, the exact work the native path exists to stop doing.
    decoded: Optional[Dict[str, Any]] = None

    @classmethod
    def from_tuple(cls, row: Sequence[Any]) -> "IngressEvent":
        return cls(
            signature=row[0], program=row[1], discriminator=row[2],
            fee_payer=row[3], data=row[4], accounts=tuple(row[5]),
            slot=int(row[6]), received_ns=int(row[7]), is_vote=bool(row[8]),
            # Tolerated absent, so a desk running against an extension built
            # before the semantic decoder existed still works -- the two are
            # deployed separately and a version skew must degrade, not fail.
            decoded=(row[9] if len(row) > 9 else None))

    @property
    def kind(self) -> str:
        """What this event IS, or "unknown" when nothing decoded it."""
        if not self.decoded:
            return "unknown"
        return str(self.decoded.get("type", "") or "unknown")

    @property
    def signature_key(self) -> bytes:
        """The canonical identity, shared with whatever the Python side sends."""
        return canonical_signature_key(self.signature)

    @property
    def received_at(self) -> float:
        """Wall-clock seconds, comparable with `time.time()` on this box."""
        return self.received_ns / 1e9


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
        self.requested_mode = normalise_mode(mode)
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
        self.promoted_at = 0.0
        # Parity accounting, keyed on the CANONICAL signature identity with
        # the wall-clock second each side saw it, so a comparison is made
        # only once both sides have had the same chance.
        # (wire timestamp, local insertion time). Two clocks on purpose: the
        # wire timestamp is what lead time is measured from, but expiry has to
        # run on OUR clock, or a stream whose timestamps are skewed or absent
        # evicts every event the instant it arrives and reports the other side
        # as having missed nothing it ever saw.
        self._native_seen: Dict[bytes, Tuple[float, float]] = {}
        self._python_seen: Dict[bytes, float] = {}
        self.agreements = 0
        self.native_only = 0
        self.python_only = 0
        self.missed_during_reconnect = 0
        self.unresolvable_signatures = 0
        self.drained = 0
        self.drain_calls = 0
        #: How many drained events the Rust side had already decoded. The
        #: ratio to `drained` is what says whether the semantic decoder is
        #: actually carrying its weight or whether Python is still doing the
        #: work on everything that matters.
        self.decoded_natively = 0
        self.by_kind: Dict[str, int] = {}
        self.started_at = 0.0
        self.last_drain_at = 0.0
        # Lead time: how much EARLIER the native path saw an event the
        # reference client also saw. Signed, in milliseconds, so a native
        # path that is behind reads as negative rather than as zero.
        self._lead_ms: List[float] = []
        self._reconnects = 0
        self._last_reconnect_at = 0.0

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
        if self._native is not None:
            # Idempotent. Two subscriptions against one endpoint is a way to
            # get rate-limited off the feed the desk depends on.
            return True
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
        self.started_at = time.time()
        if self.requested_mode == MODE_RUST:
            self.mode = MODE_AUTO
            self.promoted_at = self.started_at
            logger.warning(
                "NATIVE INGRESS is AUTHORITATIVE from its first event because "
                "the configured mode is RUST. Nothing has been proven on this "
                "run; the parity ledger is still kept and a single miss still "
                "demotes it permanently")
        else:
            self.mode = MODE_SHADOW
            if self.requested_mode == MODE_SHADOW:
                logger.info(
                    "NATIVE INGRESS running in SHADOW against %d program(s). "
                    "Configured SHADOW: it will NOT promote itself however "
                    "well it agrees -- set native_ingress_mode=AUTO for that",
                    len(self.programs))
            else:
                logger.info(
                    "NATIVE INGRESS running in SHADOW against %d program(s); "
                    "it decides nothing until it has agreed with the Python "
                    "client on %d events", len(self.programs),
                    self.promote_after)
        return True

    def stop(self) -> None:
        """Release the stream. Safe to call when it never started."""
        if self._native is not None:
            try:
                self._native.stop()
            except Exception as exc:  # pragma: no cover - teardown only
                logger.debug("native ingress stop: %s", exc)
        self._native = None
        self.available = False
        self.mode = MODE_OFF
        if not self.unavailable_reason:
            self.unavailable_reason = "stopped"
        self._native_seen.clear()
        self._python_seen.clear()

    @property
    def running(self) -> bool:
        return self._native is not None and self.available

    @property
    def authoritative(self) -> bool:
        """True only when this path is allowed to decide anything."""
        return self.mode == MODE_AUTO and not self.demoted_reason

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
        self.drain_calls += 1
        self.last_drain_at = now
        events = []
        for row in rows:
            event = IngressEvent.from_tuple(row)
            events.append(event)
            if event.decoded:
                self.decoded_natively += 1
                kind = event.kind
                self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
            key = event.signature_key
            if not key:
                self.unresolvable_signatures += 1
                continue
            # The native timestamp is taken in Rust as the update leaves the
            # network stack, not here -- the whole point is to measure the
            # wire and not our own scheduling, and using `now` would report
            # the drain interval as though it were latency.
            seen_at = event.received_at or now
            python_at = self._python_seen.pop(key, None)
            if python_at is not None:
                self._agree(lead_ms=(python_at - seen_at) * 1000.0)
            else:
                self._native_seen[key] = (seen_at, now)
        self.drained += len(events)
        self._note_reconnects(now)
        self._sweep(now)
        return events

    # --- parity ----------------------------------------------------------

    def note_python_event(self, signature: Any) -> None:
        """Record that the REFERENCE client saw this signature.

        The Python client is the reference, not the competitor. Anything it
        saw and the native side did not is a miss, and misses are the only
        thing that matters here -- a receiver that is faster on the events it
        catches and blind to one launch in a thousand is not an improvement.

        Accepts base58 text, base58 bytes or a raw 64-byte signature; all
        three are reduced to the same key before they touch the ledger.
        """
        if self._native is None:
            return
        key = canonical_signature_key(signature)
        if not key:
            if signature:
                self.unresolvable_signatures += 1
            return
        now = time.time()
        native = self._native_seen.pop(key, None)
        if native is not None:
            self._agree(lead_ms=(now - native[0]) * 1000.0)
            return
        self._python_seen[key] = now

    def _agree(self, *, lead_ms: float) -> None:
        self.agreements += 1
        self._lead_ms.append(lead_ms)
        if len(self._lead_ms) > LEAD_SAMPLES:
            del self._lead_ms[:len(self._lead_ms) - LEAD_SAMPLES]
        self._maybe_promote()

    def _note_reconnects(self, now: float) -> None:
        """Track when the stream last dropped, so misses can be attributed."""
        if self._native is None:
            return
        try:
            reconnects = int(dict(self._native.report()).get("reconnects", 0))
        except Exception:  # pragma: no cover - defensive
            return
        if reconnects > self._reconnects:
            self._reconnects = reconnects
            self._last_reconnect_at = now

    def _sweep(self, now: float) -> None:
        """Resolve everything old enough that a match is no longer possible."""
        cutoff = now - PARITY_WINDOW_S
        for key in [k for k, seen in self._python_seen.items() if seen < cutoff]:
            seen_at = self._python_seen.pop(key)
            self.python_only += 1
            self._account_for_miss(seen_at)
        for key in [k for k, seen in self._native_seen.items()
                    if seen[1] < cutoff]:
            self._native_seen.pop(key, None)
            # Seen only natively is not a fault: the two subscribe with
            # different filters at different instants, and being EARLY looks
            # exactly like this. Counted, never punished.
            self.native_only += 1

    def _account_for_miss(self, seen_at: float) -> None:
        """One event the reference saw and the native side did not."""
        if self.started_at and seen_at < self.started_at + PARITY_WARMUP_S:
            # The reference client was already streaming when this subscribed.
            # Holding the gap against it makes promotion unreachable by
            # construction, which is indistinguishable from never wiring it.
            return
        if (self._last_reconnect_at
                and seen_at >= self._last_reconnect_at - PARITY_WINDOW_S):
            self.missed_during_reconnect += 1
            allowed = max(1.0, self.agreements * MAX_RECONNECT_MISS_SHARE)
            if self.missed_during_reconnect > allowed:
                self._demote(
                    f"lost {self.missed_during_reconnect} event(s) across "
                    f"stream reconnects, above {MAX_RECONNECT_MISS_SHARE:.1%} "
                    f"of {self.agreements} agreements")
            return
        self._demote(
            f"missed an event the Python client received while the stream "
            f"was continuously connected ({self.python_only} total)")

    def _maybe_promote(self) -> None:
        if self.mode != MODE_SHADOW or self.demoted_reason:
            return
        if self.requested_mode not in PROMOTABLE_MODES:
            # A configured SHADOW is a shadow for ever. The previous code
            # promoted out of it regardless of what was asked for, which made
            # the setting a lie in the one direction that matters.
            return
        if self.agreements < self.promote_after:
            return
        self.mode = MODE_AUTO
        self.promoted_at = time.time()
        logger.info(
            "NATIVE INGRESS promoted to AUTO after %d agreements, %d misses, "
            "median lead %s", self.agreements, self.python_only,
            f"{self.median_lead_ms:.1f}ms" if self._lead_ms else "unmeasured")

    def _demote(self, reason: str) -> None:
        """Permanent. A receiver shown to drop launches does not get a rerun."""
        if self.demoted_reason:
            return
        self.demoted_reason = reason
        if self.mode == MODE_AUTO:
            logger.error("NATIVE INGRESS demoted from AUTO: %s", reason)
        else:
            logger.warning(
                "NATIVE INGRESS will never be promoted on this run: %s", reason)
        self.mode = MODE_SHADOW

    # --- reporting -------------------------------------------------------

    @property
    def median_lead_ms(self) -> Optional[float]:
        """How much earlier the native path saw events BOTH sides saw."""
        if not self._lead_ms:
            return None
        return float(statistics.median(self._lead_ms))

    def _lead_percentile(self, fraction: float) -> Optional[float]:
        if not self._lead_ms:
            return None
        ordered = sorted(self._lead_ms)
        index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
        return float(ordered[index])

    def report(self) -> Dict[str, Any]:
        native = {}
        if self._native is not None:
            try:
                native = dict(self._native.report())
            except Exception:  # pragma: no cover - defensive
                native = {}
        total = self.agreements + self.python_only
        lead = self.median_lead_ms
        return {
            "schema": NATIVE_INGRESS_SCHEMA_VERSION,
            "status": ("OK" if self.mode == MODE_AUTO else
                       "SHADOW" if self.mode == MODE_SHADOW else
                       "OFF"),
            "mode": self.mode,
            "requested_mode": self.requested_mode,
            "authoritative": self.authoritative,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "promote_after": self.promote_after,
            "promotable": self.requested_mode in PROMOTABLE_MODES,
            "agreements": self.agreements,
            "missed_by_native": self.python_only,
            "missed_during_reconnect": self.missed_during_reconnect,
            "native_only": self.native_only,
            "unresolvable_signatures": self.unresolvable_signatures,
            "agreement_rate": (self.agreements / total) if total else None,
            "drained": self.drained,
            "decoded_natively": self.decoded_natively,
            "decoded_share": (self.decoded_natively / self.drained
                              if self.drained else None),
            "by_kind": dict(self.by_kind),
            "drain_calls": self.drain_calls,
            "seconds_since_drain": (
                round(time.time() - self.last_drain_at, 1)
                if self.last_drain_at else None),
            "median_lead_ms": round(lead, 3) if lead is not None else None,
            "lead_p10_ms": self._lead_percentile(0.10),
            "lead_p90_ms": self._lead_percentile(0.90),
            "lead_samples": len(self._lead_ms),
            "demoted_reason": self.demoted_reason,
            "detail": ("the Python client remains the reference; this decides "
                       "nothing until it has matched it, and a single miss on "
                       "a connected stream demotes it permanently"),
            "native": native,
        }
