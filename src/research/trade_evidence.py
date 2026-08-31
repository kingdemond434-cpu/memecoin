"""Structured, hash-chained evidence records for every candidate and action."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


SCHEMA_VERSION = "v1"
GENESIS = "0" * 64


def _canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


class TradeEvidenceLedger:
    """Append-only JSONL whose rows authenticate their predecessor.

    Hash chaining makes accidental rewrites and truncation detectable.  It is
    not a remote notarization system and is not described as one.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.sequence = 0
        self.last_hash = GENESIS
        self.write_failures = 0
        self._recover_tail()

    def _recover_tail(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.sequence = int(row["sequence"])
                    self.last_hash = str(row["hash"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # Do not append to a chain whose tail cannot be trusted.
            self.write_failures += 1
            self.last_hash = ""

    def record(self, event_type: str, payload: Dict[str, Any], *,
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
                    row = json.loads(line)
                    digest = row.pop("hash")
                    if int(row.get("sequence", -1)) != expected_sequence:
                        return False, "sequence gap", expected_sequence - 1
                    if row.get("previous_hash") != previous:
                        return False, "previous hash mismatch", expected_sequence - 1
                    if hashlib.sha256(_canonical(row)).hexdigest() != digest:
                        return False, "row hash mismatch", expected_sequence - 1
                    previous = digest
                    expected_sequence += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return False, str(exc), expected_sequence - 1
        return True, "OK", expected_sequence - 1

    def report(self) -> Dict[str, Any]:
        return {"status": "OK" if self.last_hash else "DATA_BLOCKED",
                "path": str(self.path), "records": self.sequence,
                "write_failures": self.write_failures,
                "hash_chain": bool(self.last_hash)}


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
