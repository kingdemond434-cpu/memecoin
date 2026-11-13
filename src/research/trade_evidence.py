"""Structured, hash-chained evidence records for every candidate and action."""

from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from src.research.evidence_certificate import validate_certificate


SCHEMA_VERSION = "v1"
GENESIS = "0" * 64


def _canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _verified_row(line, previous, expected_sequence):
    if isinstance(line, bytes): line = line.decode('utf-8')
    row = json.loads(line)
    digest = row.pop('hash')
    if int(row.get('sequence', -1)) != expected_sequence:
        raise ValueError('sequence gap')
    if row.get('previous_hash') != previous:
        raise ValueError('previous hash mismatch')
    if hashlib.sha256(_canonical(row)).hexdigest() != digest:
        raise ValueError('row hash mismatch')
    row['hash'] = digest
    return row, digest


class TradeEvidenceLedger:
    """Append-only JSONL whose rows authenticate their predecessor.

    Hash chaining makes accidental rewrites and truncation detectable.  It is
    not a remote notarization system and is not described as one.
    """

    def __init__(self, path: str | Path, *, recovery_certificate: str | Path | None = None):
        self.path = Path(path)
        self.sequence = 0
        self.last_hash = GENESIS
        self.write_failures = 0
        self._append_lock = threading.RLock()
        self.recovery_certificate = Path(recovery_certificate) if recovery_certificate is not None else None
        self.recovery_mode = 'LEGACY_FULL_PARSE'
        self.certified_prefix_records = 0
        self.recovery_error_type = None
        # Tail checkpoint (see docs/EVIDENCE_TAIL_CHECKPOINT.md): a running
        # SHA-256 over every byte after the verified base, so a restart
        # re-hashes bytes instead of JSON-decoding the whole history.
        self.checkpoint_path = Path(str(self.path) + '.tail.json')
        self.checkpoint_interval_s = 30.0
        self.checkpoint_used = False
        self.checkpoint_rejected_type = None
        self.checkpoint_failures = 0
        self._seg_base = None
        self._seg_hash = None
        self._extent = None
        self._checkpoint_at = 0.0
        started = time.perf_counter()
        self._recover_tail()
        self.recovery_seconds = time.perf_counter()-started
        if self.last_hash and self._seg_hash is not None:
            self._write_checkpoint(force=True)

    def _recover_tail(self) -> None:
        if not self.path.exists() and self.recovery_certificate is None:
            self._seg_base, self._seg_hash, self._extent = 0, hashlib.sha256(), 0
            return
        try:
            if self.recovery_certificate is not None:
                self.recovery_mode = 'CERTIFIED_RECOVERY_PENDING'
                self._recover_certified()
                return
            with self.path.open("rb") as handle:
                end = os.fstat(handle.fileno()).st_size
                resumed = self._resume_checkpoint(handle, 0, end)
                if resumed is not None:
                    count, previous, segment = resumed
                    self.sequence, self.last_hash = self._verify_rows(
                        handle, end, count, previous, segment)
                    self.recovery_mode = 'TAIL_CHECKPOINT_AND_VERIFIED_TAIL'
                else:
                    handle.seek(0)
                    segment = hashlib.sha256()
                    for line in handle:
                        segment.update(line)
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        self.sequence = int(row["sequence"])
                        self.last_hash = str(row["hash"])
                    end = handle.tell()
                self._seg_base, self._seg_hash, self._extent = 0, segment, end
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            # Do not append to a chain whose tail cannot be trusted.
            self.write_failures += 1
            self.last_hash = ""
            self.recovery_error_type = type(exc).__name__
            if self.recovery_certificate is not None:
                self.recovery_mode = 'CERTIFIED_RECOVERY_FAILED'

    @staticmethod
    def _verify_rows(handle, end, count, previous, segment):
        """Chain-verify complete rows from the handle position to `end`."""
        while handle.tell() < end:
            line = handle.readline(end-handle.tell())
            if not line or not line.endswith(b'\n'):
                raise ValueError('incomplete canonical tail')
            segment.update(line)
            if line.strip():
                _, previous = _verified_row(line, previous, count+1)
                count += 1
        return count, previous

    def _resume_checkpoint(self, handle, base, end):
        """Return (sequence, last_hash, running segment hash) at the
        checkpoint, with the handle positioned there, or None.

        Accepted only when the checkpoint starts at the same verified base,
        the SHA-256 of every byte from base to its extent matches, the extent
        is a row boundary, and that boundary row re-hashes to the recorded
        sequence and hash. Bytes past the base are protected only by the
        service-written hash chain either way; streaming SHA-256 over them
        detects the same accidental damage that per-row JSON verification
        does, at disk speed instead of decode speed. Rows after the extent
        are still chain-verified by the caller. Any doubt returns None and
        the caller runs the full path; it never trusts an unchecked tail.
        """
        try:
            if not self.checkpoint_path.exists():
                return None
            if self.checkpoint_path.stat().st_size > 64*1024:
                raise ValueError('tail checkpoint exceeds metadata envelope')
            mark = json.loads(self.checkpoint_path.read_text(encoding='utf-8'))
            if mark.get('schema') != 'trade_evidence_tail/2':
                raise ValueError('tail checkpoint schema')
            extent, count = int(mark['extent']), int(mark['sequence'])
            previous, digest = str(mark['last_hash']), str(mark['segment_sha256'])
            if int(mark['base_extent']) != base:
                raise ValueError('tail checkpoint base differs')
            if not base < extent <= end or count <= 0:
                raise ValueError('tail checkpoint extent')
            handle.seek(base)
            segment, remaining = hashlib.sha256(), extent-base
            while remaining:
                chunk = handle.read(min(remaining, 4*1024*1024))
                if not chunk:
                    raise ValueError('ledger truncated under checkpoint')
                segment.update(chunk)
                remaining -= len(chunk)
            if segment.hexdigest() != digest:
                raise ValueError('tail checkpoint segment hash differs')
            handle.seek(extent-1)
            if handle.read(1) != b'\n':
                raise ValueError('tail checkpoint is not a row boundary')
            last = self._prefix_last_row(handle, extent)
            if last is None:
                raise ValueError('tail checkpoint row missing')
            _, actual = _verified_row(last, json.loads(last)['previous_hash'], count)
            if actual != previous:
                raise ValueError('tail checkpoint row differs')
            handle.seek(extent)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.checkpoint_rejected_type = type(exc).__name__
            return None
        self.checkpoint_used = True
        return count, previous, segment

    def _write_checkpoint(self, *, force: bool = False) -> None:
        """Atomically persist the running tail state. Telemetry-grade: a
        failure is counted and never blocks or fails an append."""
        now = time.monotonic()
        if not force and now-self._checkpoint_at < self.checkpoint_interval_s:
            return
        if self._seg_hash is None or not self.last_hash or self.sequence <= 0:
            return
        self._checkpoint_at = now
        mark = {'schema': 'trade_evidence_tail/2', 'base_extent': self._seg_base,
                'extent': self._extent, 'sequence': self.sequence,
                'last_hash': self.last_hash,
                'segment_sha256': self._seg_hash.copy().hexdigest(),
                'written_at': time.time()}
        temporary = Path(str(self.checkpoint_path)+'.tmp')
        try:
            with temporary.open('w', encoding='utf-8') as handle:
                handle.write(json.dumps(mark, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.checkpoint_path)
        except OSError:
            self.checkpoint_failures += 1

    def flush_checkpoint(self) -> None:
        with self._append_lock:
            self._write_checkpoint(force=True)

    @staticmethod
    def _prefix_last_row(handle, extent):
        """Find the last nonblank committed row without loading its prefix."""
        position, carry = extent, b''
        while position:
            width = min(position, 64*1024)
            position -= width
            handle.seek(position)
            pieces = (handle.read(width)+carry).split(b'\n')
            carry = pieces[0]
            for line in reversed(pieces[1:]):
                if line.strip():
                    return line
        return carry if carry.strip() else None

    def _recover_certified(self):
        # The operator must protect this certificate from the service account's
        # writes. Its checksum is corruption detection, not a signature.
        if self.recovery_certificate.stat().st_size > 64*1024:
            raise ValueError('recovery certificate exceeds metadata envelope')
        certificate = json.loads(self.recovery_certificate.read_text(encoding='utf-8-sig'))
        validate_certificate(certificate)
        extent = certificate['canonical_bytes']
        count, previous = certificate['canonical_records'], certificate['last_hash']
        with self.path.open('rb') as handle:
            before = os.fstat(handle.fileno())
            if extent > before.st_size:
                raise ValueError('canonical file shorter than certified prefix')
            prefix_hash = hashlib.sha256()
            remaining = extent
            while remaining:
                chunk = handle.read(min(remaining, 4*1024*1024))
                if not chunk:
                    raise ValueError('certified prefix truncated during recovery')
                prefix_hash.update(chunk)
                remaining -= len(chunk)
            if prefix_hash.hexdigest() != certificate['canonical_sha256']:
                raise ValueError('live canonical prefix differs from certificate')
            if extent:
                handle.seek(extent-1)
                if handle.read(1) != b'\n':
                    raise ValueError('certified extent is not a row boundary')
            last = self._prefix_last_row(handle, extent)
            if count:
                row = json.loads(last) if last else {}
                _, actual = _verified_row(last, row['previous_hash'], count)
                if actual != previous:
                    raise ValueError('certificate tail metadata differs from prefix')
            elif last is not None or previous != GENESIS:
                raise ValueError('empty certificate has nonempty tail metadata')
            handle.seek(extent)
            resumed = self._resume_checkpoint(handle, extent, before.st_size)
            if resumed is not None:
                count, previous, segment = resumed
            else:
                handle.seek(extent)
                segment = hashlib.sha256()
            count, previous = self._verify_rows(handle, before.st_size, count, previous, segment)
            after = os.fstat(handle.fileno())
            named = self.path.stat()
            if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    or (after.st_size, after.st_mtime_ns) != (named.st_size, named.st_mtime_ns)
                    or not os.path.samestat(after, named)):
                raise ValueError('canonical file changed during recovery')
        self.sequence, self.last_hash = count, previous
        self.certified_prefix_records = certificate['canonical_records']
        self.recovery_mode = ('CERTIFIED_PREFIX_SHA256_TAIL_CHECKPOINT_AND_VERIFIED_TAIL'
                              if self.checkpoint_used else
                              'CERTIFIED_PREFIX_SHA256_AND_VERIFIED_TAIL')
        self._seg_base, self._seg_hash, self._extent = extent, segment, before.st_size

    def record(self, event_type: str, payload: Dict[str, Any], *,
               timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        with self._append_lock:
            return self._record_unlocked(event_type, payload, timestamp=timestamp)

    def record_reference(self, event_type: str, payload: Dict[str, Any], *,
                         timestamp: Optional[float] = None) -> Optional[str]:
        """Append and locate a row atomically across this desk's threads."""
        with self._append_lock:
            offset = self.path.stat().st_size if self.path.exists() else 0
            row = self.record(event_type, payload, timestamp=timestamp)
            return f"evidence:{row['hash']}:{offset}" if row is not None else None

    def _record_unlocked(self, event_type: str, payload: Dict[str, Any], *,
                         timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not event_type or not self.last_hash:
            self.write_failures += 1
            return None
        body = {
            "schema": SCHEMA_VERSION, "sequence": self.sequence + 1,
            "timestamp": float(timestamp or time.time()),
            "event_type": event_type, "previous_hash": self.last_hash,
            "payload": payload,
        }
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        row = {**body, "hash": digest}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Binary append with the platform line ending: identical bytes to
            # the former text-mode write, and known exactly for the running
            # segment hash.
            data = (json.dumps(row, separators=(",", ":"), default=str) + os.linesep).encode("utf-8")
            with self.path.open("ab") as handle:
                start = handle.seek(0, os.SEEK_END)
                handle.write(data)
        except OSError:
            self.write_failures += 1
            self._seg_hash = None  # bytes on disk are now unknown; stop checkpointing
            return None
        self.sequence += 1
        self.last_hash = digest
        if self._seg_hash is not None:
            if start != self._extent:
                self._seg_hash = None
            else:
                self._seg_hash.update(data)
                self._extent = start + len(data)
                self._write_checkpoint()
        return row

    @staticmethod
    def verify(path: str | Path) -> Tuple[bool, str, int]:
        previous = GENESIS
        expected_sequence = 1
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    _, previous = _verified_row(line, previous, expected_sequence)
                    expected_sequence += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return False, str(exc), expected_sequence - 1
        return True, "OK", expected_sequence - 1

    def matching_snapshot(self, payload_key: str, payload_value: Any) -> list[dict]:
        """Verify one committed prefix, streaming only matching rows to memory.

        Capture EOF under the writer lock, then release it before scanning.
        Appends can continue without extending or partially entering this read.
        Call this disk/CPU work outside the collection event loop.
        """
        with self._append_lock:
            if not self.path.exists(): return []
            handle = self.path.open('rb')
            end = os.fstat(handle.fileno()).st_size
        previous, sequence, matches = GENESIS, 1, []
        with handle:
            while handle.tell() < end:
                line = handle.readline(end-handle.tell())
                if not line or not line.endswith(b'\n'):
                    raise ValueError('incomplete committed evidence prefix')
                if not line.strip(): continue
                row, previous = _verified_row(line, previous, sequence)
                sequence += 1
                if row.get('payload', {}).get(payload_key) == payload_value:
                    matches.append(row)
        return matches

    def report(self) -> Dict[str, Any]:
        return {"status": "OK" if self.last_hash else "DATA_BLOCKED",
                "path": str(self.path), "records": self.sequence,
                "write_failures": self.write_failures,
                "hash_chain": bool(self.last_hash),
                "recovery_mode": self.recovery_mode,
                "certified_prefix_records": self.certified_prefix_records,
                "recovery_seconds": self.recovery_seconds,
                "recovery_error_type": self.recovery_error_type,
                "checkpoint_used": self.checkpoint_used,
                "checkpoint_rejected_type": self.checkpoint_rejected_type,
                "checkpoint_failures": self.checkpoint_failures}


def evidence_packet(*, mint: str, timestamp: float,
                    bonding_curve: Dict[str, Any], liquidity: Dict[str, Any],
                    sellability: Dict[str, Any], authorities: Dict[str, Any],
                    holder_distribution: Dict[str, Any], wallet_clusters: Dict[str, Any],
                    dev_wallet: Dict[str, Any], smart_wallet_flow: Dict[str, Any],
                    social_velocity: Dict[str, Any], entry_cost: Dict[str, Any],
                    exit_liquidity: Dict[str, Any], risk_vetoes: Iterable[str],
                    expected_edge: Optional[float], position_size: Optional[float],
                    exit_plan: Dict[str, Any], decision: str) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION, "mint": mint, "timestamp": timestamp,
        "bonding_curve": bonding_curve, "liquidity": liquidity,
        "sellability": sellability, "authorities": authorities,
        "holder_distribution": holder_distribution,
        "wallet_clusters": wallet_clusters, "dev_wallet": dev_wallet,
        "smart_wallet_flow": smart_wallet_flow,
        "social_velocity": social_velocity, "entry_cost": entry_cost,
        "exit_liquidity": exit_liquidity,
        "risk_vetoes": sorted(set(risk_vetoes)),
        "expected_edge": expected_edge, "position_size": position_size,
        "exit_plan": exit_plan, "decision": decision,
    }
