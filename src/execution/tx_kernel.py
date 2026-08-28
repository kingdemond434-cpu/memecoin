"""Promoting the Rust transaction builder onto the money path, on evidence.

`native/solana_fastpath` can compile a v0 message and assemble a signed
transaction, and it has been proven byte-identical to solders across fifteen
hundred randomised transactions. That is a good reason to believe it and not a
reason to swap it in. A transaction builder that is wrong produces bytes that
look right, sign cleanly, and fail against accounts that do not exist -- and
it fails with real capital rather than in a test.

So it is promoted exactly the way the T0 kernel is: shadowed, compared, and
demoted permanently on the first disagreement.

    OFF      solders builds. Rust is not consulted.
    SHADOW   solders builds. Rust builds too, and the BYTES are compared.
    AUTO     shadow until `promote_after` consecutive byte-identical builds,
             then Rust builds and solders is not called.
    RUST     Rust from the first call. For tests and benchmarks.

Two things make this safer than the equivalent for the decision kernel.

**The comparison is exact.** Two decisions can differ by a rounding error and
argue about whether that matters. Two transactions either are the same bytes
or they are not, so there is no tolerance to tune and no judgement call about
what counts as agreement.

**The key never moves.** This is the constraint that shaped the design. The
desk's signer is deliberately isolated and may be another process entirely, so
the Rust path here compiles the message and assembles the result, and the
SIGNING happens exactly where it happened before, over exactly the same bytes.
`build_signed_transaction` exists in the extension and is deliberately not
used: it takes a secret key, and reaching for it would undo the isolation to
save a few microseconds.

What is actually saved is small and honestly stated: about forty microseconds
on a twenty-seven account Pump buy, against a four hundred millisecond slot.
This is wired for one-implementation cleanliness, not for speed. Anyone
reading it expecting a latency win should look at the wire instead.

**A promoted build is still checked, cheaply.** Recompiling in solders after
promotion would give back the whole saving, so both halves are verified by
invariants that cost microseconds instead.

A compiled MESSAGE is checked for four facts a garbage compile cannot fake:
the v0 prefix, the fee payer at account zero, the blockhash sitting exactly
where the account count says it should, and the instruction count we asked
for. An ASSEMBLED transaction is checked for its signature count, its total
length, and -- the one that matters -- that it ends with the exact message
bytes the signer signed, because a transaction whose tail is not the signed
message is a signature over something else.

These run on EVERY build, promoted or not. Sampling them would mean most wrong
messages are signed and submitted before anything notices, and a wrong message
is not a slow trade, it is a trade against accounts that do not exist. Full
byte-for-byte parity against solders continues on a small sample on top, so
the comparison never stops entirely and a dependency upgrade that changes
either side is caught.
"""

from __future__ import annotations

import base64
import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TX_KERNEL_SCHEMA_VERSION = "v1"

#: Consecutive byte-identical builds before Rust is allowed to build alone.
#: Higher than the decision kernel's bar relative to what it buys, because
#: what it buys is small and what it risks is a rejected transaction.
DEFAULT_PROMOTE_AFTER = 200

#: Share of promoted builds that are still fully compared against solders.
#: Not zero: a build path that stops being checked the moment it is trusted
#: is a build path that will drift silently through a dependency upgrade.
DEFAULT_AUDIT_RATE = 0.02

SIGNATURE_LEN = 64


class TxMode(Enum):
    OFF = "off"
    SHADOW = "shadow"
    AUTO = "auto"
    RUST = "rust"


def _load_native() -> Tuple[Optional[Any], str]:
    try:
        import solana_fastpath  # type: ignore
    except ImportError as exc:
        return None, f"native extension unavailable: {exc}"
    for required in ("compile_v0_message", "assemble_transaction"):
        if not hasattr(solana_fastpath, required):
            return None, f"extension present but carries no {required}"
    return solana_fastpath, "OK"


class TxKernel:
    """Compiles and assembles through whichever implementation is authoritative."""

    def __init__(self, *, mode: str = "auto",
                 promote_after: int = DEFAULT_PROMOTE_AFTER,
                 audit_rate: float = DEFAULT_AUDIT_RATE):
        try:
            self.mode = TxMode(str(mode).lower())
        except ValueError:
            logger.warning("unknown tx kernel mode %r; falling back to shadow", mode)
            self.mode = TxMode.SHADOW
        self.promote_after = max(1, int(promote_after))
        self.audit_rate = min(1.0, max(0.0, float(audit_rate)))
        self.native, self.native_status = _load_native()
        self.compared = 0
        self.agreements = 0
        self.divergences = 0
        self.consecutive_agreements = 0
        self.rust_builds = 0
        self.python_builds = 0
        self.rust_errors = 0
        self.audits = 0
        self.invariant_failures = 0
        # Set on the first byte mismatch while Rust was building alone, and
        # never cleared. A builder that produced one wrong transaction has to
        # be looked at, not quietly re-promoted on the next two hundred.
        self.demoted_reason = ""
        self.divergence_example: Dict[str, Any] = {}

    @property
    def rust_available(self) -> bool:
        return self.native is not None

    @property
    def rust_authoritative(self) -> bool:
        if not self.rust_available or self.demoted_reason:
            return False
        if self.mode is TxMode.RUST:
            return True
        if self.mode is TxMode.AUTO:
            return self.consecutive_agreements >= self.promote_after
        return False

    # --- compiling -------------------------------------------------------

    def compile_message(self, payer: bytes, instructions: Sequence[Any],
                        blockhash: bytes, python_bytes_fn) -> bytes:
        """The message bytes to sign, from whichever implementation is live.

        `python_bytes_fn` is a zero-argument callable returning solders' own
        answer. Passed rather than imported so this module holds no dependency
        on solders and can be tested without one.
        """
        if self.mode is TxMode.OFF or not self.rust_available:
            self.python_builds += 1
            return python_bytes_fn()

        authoritative = self.rust_authoritative
        try:
            rust_bytes = bytes(self.native.compile_v0_message(
                payer, self._raw(instructions), blockhash))
        except Exception as exc:
            self.rust_errors += 1
            self.consecutive_agreements = 0
            logger.warning("rust message compile raised; using solders: %s", exc)
            self.python_builds += 1
            return python_bytes_fn()

        # Structural check on every build, promoted or not. Without it a
        # promoted path only catches a bad compile on the audit sample, which
        # means most wrong messages get signed and submitted before anything
        # notices -- and a wrong message is not a slow trade, it is a trade
        # against accounts that do not exist. These checks are memcmp and
        # arithmetic; they cost microseconds and they cannot pass on garbage.
        failure = self._message_invariant_failure(
            rust_bytes, payer, blockhash, len(instructions))
        if failure:
            self.invariant_failures += 1
            self.consecutive_agreements = 0
            if not self.demoted_reason:
                self.demoted_reason = f"compiled message failed its invariant: {failure}"
                logger.error("TX KERNEL DEMOTED: %s", self.demoted_reason)
            self.python_builds += 1
            return python_bytes_fn()

        if authoritative and not self._should_audit():
            self.rust_builds += 1
            return rust_bytes

        python_bytes = python_bytes_fn()
        if authoritative:
            self.audits += 1
        self._record(python_bytes, rust_bytes, was_authoritative=authoritative)
        if authoritative and not self.demoted_reason:
            self.rust_builds += 1
            return rust_bytes
        self.python_builds += 1
        return python_bytes

    @staticmethod
    def _raw(instructions: Sequence[Any]) -> List[Tuple[bytes, List[Tuple[bytes, bool, bool]], bytes]]:
        """solders instructions as the plain tuples the extension takes."""
        raw = []
        for instruction in instructions:
            raw.append((
                bytes(instruction.program_id),
                [(bytes(meta.pubkey), bool(meta.is_signer), bool(meta.is_writable))
                 for meta in instruction.accounts],
                bytes(instruction.data),
            ))
        return raw

    @staticmethod
    def _message_invariant_failure(message: bytes, payer: bytes, blockhash: bytes,
                                   instruction_count: int) -> str:
        """What can be checked about a v0 message without recompiling it.

        Four facts, each of which a garbage compile fails immediately: it is
        a versioned message, the fee payer is account zero (the runtime debits
        that account, so getting it wrong spends the wrong wallet), the
        blockhash sits where the account list says it should, and the
        instruction count is the one we asked for.
        """
        if len(message) < 5 + 32 + 32:
            return f"too short to be a v0 message ({len(message)} bytes)"
        if message[0] != 0x80:
            return f"first byte is 0x{message[0]:02x}, not the v0 prefix"
        if message[1] < 1:
            return "no required signatures"
        # Short-vec account count. Anything above one byte here is more than
        # 127 accounts, which no transaction of ours has.
        count = message[4]
        if count & 0x80:
            return "account count is multi-byte; not a shape this desk builds"
        if count < 1:
            return "no account keys"
        keys_end = 5 + count * 32
        if len(message) < keys_end + 32 + 1:
            return f"truncated: {count} keys do not fit in {len(message)} bytes"
        if message[5:37] != payer:
            return "account zero is not the fee payer"
        if message[keys_end:keys_end + 32] != blockhash:
            return "the blockhash is not where the account list says it is"
        if message[keys_end + 32] != instruction_count:
            return (f"instruction count is {message[keys_end + 32]}, "
                    f"expected {instruction_count}")
        return ""

    def _should_audit(self) -> bool:
        return self.audit_rate > 0.0 and random.random() < self.audit_rate

    def _record(self, python_bytes: bytes, rust_bytes: bytes, *,
                was_authoritative: bool) -> None:
        self.compared += 1
        if python_bytes == rust_bytes:
            self.agreements += 1
            self.consecutive_agreements += 1
            return
        self.divergences += 1
        self.consecutive_agreements = 0
        if not self.demoted_reason:
            self.demoted_reason = (
                f"rust and solders produced different bytes "
                f"({len(rust_bytes)} vs {len(python_bytes)})")
            self.divergence_example = {
                "python_prefix": python_bytes[:64].hex(),
                "rust_prefix": rust_bytes[:64].hex(),
                "python_len": len(python_bytes), "rust_len": len(rust_bytes),
            }
            logger.error(
                "TX KERNEL DEMOTED: %s. solders builds for the rest of this "
                "session.%s", self.demoted_reason,
                " A transaction may already have been submitted from the Rust "
                "path." if was_authoritative else "")

    # --- assembling ------------------------------------------------------

    def assemble(self, message_bytes: bytes, signatures: Sequence[bytes],
                 python_assemble_fn) -> str:
        """The base64 transaction, from whichever implementation is live.

        The promoted path does not recompile in solders -- that would give
        back the whole saving -- but it does verify what can be checked in
        microseconds: the right number of signatures, and the transaction
        ending in exactly the message bytes that were signed.
        """
        if self.mode is TxMode.OFF or not self.rust_available:
            return python_assemble_fn()
        try:
            encoded = self.native.assemble_transaction(
                message_bytes, [bytes(signature) for signature in signatures])
        except Exception as exc:
            self.rust_errors += 1
            logger.warning("rust assemble raised; using solders: %s", exc)
            return python_assemble_fn()

        failure = self._invariant_failure(encoded, message_bytes, len(signatures))
        if failure:
            self.invariant_failures += 1
            if not self.demoted_reason:
                self.demoted_reason = f"assembled transaction failed its invariant: {failure}"
                logger.error("TX KERNEL DEMOTED: %s", self.demoted_reason)
            return python_assemble_fn()
        if not self.rust_authoritative:
            # Still shadowing: solders' answer is the one that goes out, and
            # the comparison is what the shadow phase is for.
            expected = python_assemble_fn()
            self._record(expected.encode(), encoded.encode(), was_authoritative=False)
            return expected
        return encoded

    @staticmethod
    def _invariant_failure(encoded: str, message_bytes: bytes,
                           signature_count: int) -> str:
        """Cheap structural checks. Microseconds, and they catch assembly faults."""
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            return f"not valid base64: {exc}"
        if not raw:
            return "empty transaction"
        if raw[0] != signature_count:
            return (f"signature count byte is {raw[0]}, expected {signature_count}")
        expected_len = 1 + signature_count * SIGNATURE_LEN + len(message_bytes)
        if len(raw) != expected_len:
            return f"length {len(raw)}, expected {expected_len}"
        if not raw.endswith(message_bytes):
            # The one that matters: a transaction whose trailing bytes are not
            # the message we signed is a signature over something else.
            return "the transaction does not end with the signed message"
        return ""

    # --- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Whether the build path is on Rust, and on what evidence."""
        total = self.rust_builds + self.python_builds
        return {
            "schema": TX_KERNEL_SCHEMA_VERSION,
            "mode": self.mode.value,
            "native": self.native_status,
            "rust_available": self.rust_available,
            "rust_authoritative": self.rust_authoritative,
            "promote_after": self.promote_after,
            "consecutive_agreements": self.consecutive_agreements,
            "compared": self.compared,
            "agreements": self.agreements,
            "divergences": self.divergences,
            "audits_since_promotion": self.audits,
            "invariant_failures": self.invariant_failures,
            "rust_errors": self.rust_errors,
            "builds_by_rust": self.rust_builds,
            "builds_by_solders": self.python_builds,
            "rust_share": (self.rust_builds / total) if total else None,
            "demoted_reason": self.demoted_reason,
            "divergence_example": dict(self.divergence_example),
            "status": ("OK" if self.rust_available and not self.demoted_reason
                       else "DATA_BLOCKED"),
            "detail": (self.demoted_reason or
                       ("" if self.rust_available else self.native_status)),
        }
