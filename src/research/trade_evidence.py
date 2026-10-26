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
        started = time.perf_counter()
        self._recover_tail()
        self.recovery_seconds = time.perf_counter()-started

    def _recover_tail(self) -> None:
        if not self.path.exists() and self.recovery_certificate is None:
            return
        try:
            if self.recovery_certificate is not None:
                self.recovery_mode = 'CERTIFIED_RECOVERY_PENDING'
                self._recover_certified()
                return
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.sequence = int(row["sequence"])
                    self.last_hash = str(row["hash"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            # Do not append to a chain whose tail cannot be trusted.
            self.write_failures += 1
            self.last_hash = ""
            self.recovery_error_type = type(exc).__name__
            if self.recovery_certificate is not None:
                self.recovery_mode = 'CERTIFIED_RECOVERY_FAILED'

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
            while handle.tell() < before.st_size:
                line = handle.readline(before.st_size-handle.tell())
                if not line or not line.endswith(b'\n'):
                    raise ValueError('incomplete canonical tail')
                if line.strip():
                    _, previous = _verified_row(line, previous, count+1)
                    count += 1
            after = os.fstat(handle.fileno())
            named = self.path.stat()
            if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    or (after.st_size, after.st_mtime_ns) != (named.st_size, named.st_mtime_ns)
                    or not os.path.samestat(after, named)):
                raise ValueError('canonical file changed during recovery')
        self.sequence, self.last_hash = count, previous
        self.certified_prefix_records = certificate['canonical_records']
        self.recovery_mode = 'CERTIFIED_PREFIX_SHA256_AND_VERIFIED_TAIL'

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
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
        except OSError:
            self.write_failures += 1
            return None
        self.sequence += 1
        self.last_hash = digest
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
                "recovery_error_type": self.recovery_error_type}


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
