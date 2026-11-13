"""Contracted proof that the paper desk fills, exits and accounts correctly.

The desk writes one hash-chained `paper_fill` row to the trade-evidence
ledger for every simulated buy and every simulated sell. This module reads
those rows back, verifies the chain they live in, and turns them into one
small artifact whose verdict answers a single question a reviewer can check
without trusting the desk's own logs: is paper trading running, filling,
exiting and keeping its books straight?

Contract (schema `memecoin_paper_fill_proof/1`). Verdict PROVEN needs all of:
  * chain_ok             every scanned evidence row verifies against its predecessor
  * scan_complete        the scan reached the end of the committed ledger
  * round_trip           at least one paper position bought and fully sold
  * accounting_identity  per sell, pnl == proceeds + native delta - allocated
                         cost; per closed position, allocated costs sum to the
                         buy cost (both within 1e-6 relative)
  * fresh                at least one paper fill in the last `fresh_seconds`
Otherwise the verdict names the first failed condition in that order.

The scan is incremental: a checkpoint keeps the byte offset, sequence and
hash it reached, so each tick verifies only the new rows, and a tick never
reads more than `max_bytes`. A ledger that shrank or whose next row does not
chain onto the checkpoint is CHAIN_BROKEN, never silently rescanned.

Read-only with respect to the ledger. Paper only; no authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SCHEMA = "memecoin_paper_fill_proof/1"
STATE_SCHEMA = "memecoin_paper_fill_proof_state/1"
EVENT = "paper_fill"
GENESIS = "0" * 64
TOLERANCE = 1e-6
CONDITIONS = ("chain_ok", "scan_complete", "round_trip", "accounting_identity", "fresh")


def _canonical(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _empty_state() -> Dict[str, Any]:
    return {
        "schema": STATE_SCHEMA, "offset": 0, "sequence": 0, "last_hash": GENESIS,
        "chain_ok": True, "chain_error": "", "rows_scanned": 0,
        "open": {},           # token -> {cost, allocated, tokens, opened_at, buys}
        "totals": {"buys": 0, "sells": 0, "round_trips": 0, "buy_cost_usd": 0.0,
                   "proceeds_usd": 0.0, "native_delta_usd": 0.0,
                   "realized_pnl_usd": 0.0, "identity_failures": 0,
                   "malformed_rows": 0, "wins": 0, "log_wealth": 0.0},
        "days": {},           # YYYY-MM-DD -> {buys, sells, pnl}
        "recent": [],         # last 25 fills, newest last
        "identity_examples": [],
        "first_fill_at": None, "last_fill_at": None,
    }


class PaperFillProof:
    """Incremental verifier and aggregator over `paper_fill` evidence rows."""

    def __init__(self, evidence_path: str | Path, state_path: str | Path, *,
                 max_bytes: int = 256 * 1024 * 1024, fresh_seconds: float = 86400.0,
                 duty: float = 0.25, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time):
        self.evidence_path = Path(evidence_path)
        self.state_path = Path(state_path)
        self.max_bytes = int(max_bytes)
        self.fresh_seconds = float(fresh_seconds)
        self.duty = min(max(float(duty), 0.01), 1.0)
        self._sleep, self._clock = sleep, clock
        self.state = self._load()
        self.scan_complete = False

    # -- persistence -------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _empty_state()
        if state.get("schema") != STATE_SCHEMA:
            return _empty_state()
        base = _empty_state()
        base.update(state)
        base["totals"] = {**_empty_state()["totals"], **state.get("totals", {})}
        return base

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.state_path)

    # -- scanning ----------------------------------------------------
    def _broken(self, why: str) -> None:
        self.state["chain_ok"] = False
        self.state["chain_error"] = why

    def scan(self) -> Dict[str, Any]:
        """Verify and aggregate committed rows past the checkpoint."""
        state = self.state
        self.scan_complete = False
        if not state["chain_ok"]:
            return state
        try:
            size = self.evidence_path.stat().st_size
        except OSError:
            # No ledger yet is not a broken one: nothing has been committed.
            self.scan_complete = state["offset"] == 0
            if not self.scan_complete:
                self._broken("evidence ledger missing after rows were verified")
            return state
        if size < state["offset"]:
            self._broken("evidence ledger shrank below the verified offset")
            self._save()
            return state
        budget = self.max_bytes
        started = time.perf_counter()
        with self.evidence_path.open("rb") as handle:
            handle.seek(state["offset"])
            while handle.tell() < size and budget > 0:
                line = handle.readline(size - handle.tell())
                if not line.endswith(b"\n"):
                    break                       # writer mid-append; next tick
                budget -= len(line)
                if line.strip():
                    try:
                        row = json.loads(line)
                        digest = row.pop("hash")
                        if int(row.get("sequence", -1)) != state["sequence"] + 1:
                            raise ValueError("sequence gap")
                        if row.get("previous_hash") != state["last_hash"]:
                            raise ValueError("previous hash mismatch")
                        if hashlib.sha256(_canonical(row)).hexdigest() != digest:
                            raise ValueError("row hash mismatch")
                    except (ValueError, KeyError, TypeError) as exc:
                        self._broken(f"row {state['sequence'] + 1}: {exc}")
                        break
                    state["sequence"] += 1
                    state["last_hash"] = digest
                    state["rows_scanned"] += 1
                    if row.get("event_type") == EVENT:
                        self._apply(row)
                state["offset"] = handle.tell()
                # Pace: hold the scan to `duty` of one core.
                worked = time.perf_counter() - started
                if worked > 0.05:
                    self._sleep(worked * (1 - self.duty) / self.duty)
                    started = time.perf_counter()
        # Complete means every committed byte was verified; an unterminated
        # tail (writer mid-append) is left for the next tick, not claimed.
        self.scan_complete = state["chain_ok"] and state["offset"] >= size
        self._save()
        return state

    def _apply(self, row: Dict[str, Any]) -> None:
        state, totals = self.state, self.state["totals"]
        fill = row.get("payload") or {}
        at = _num(row.get("timestamp"))
        token, side = str(fill.get("token", "")), fill.get("side")
        if not token or side not in ("buy", "sell") or not math.isfinite(at):
            totals["malformed_rows"] += 1
            return
        day = time.strftime("%Y-%m-%d", time.gmtime(at))
        bucket = state["days"].setdefault(day, {"buys": 0, "sells": 0, "pnl": 0.0})
        if side == "buy":
            cost = _num(fill.get("cost_usd"))
            if not math.isfinite(cost) or cost <= 0:
                totals["malformed_rows"] += 1
                return
            book = state["open"].setdefault(token, {"cost": 0.0, "allocated": 0.0,
                                                    "pnl": 0.0, "opened_at": at, "buys": 0})
            book["cost"] += cost
            book["buys"] += 1
            totals["buys"] += 1
            totals["buy_cost_usd"] += cost
            bucket["buys"] += 1
        else:
            proceeds = _num(fill.get("proceeds_usd"))
            native = _num(fill.get("native_delta_usd"))
            allocated = _num(fill.get("allocated_cost_usd"))
            pnl = _num(fill.get("realized_pnl_usd"))
            if not all(map(math.isfinite, (proceeds, native, allocated, pnl))):
                totals["malformed_rows"] += 1
                return
            book = state["open"].get(token)
            ok = _close(pnl, proceeds + native - allocated) and book is not None
            totals["sells"] += 1
            totals["proceeds_usd"] += proceeds
            totals["native_delta_usd"] += native
            totals["realized_pnl_usd"] += pnl
            bucket["sells"] += 1
            bucket["pnl"] += pnl
            if book is not None:
                book["allocated"] += allocated
                book["pnl"] += pnl
                if fill.get("closed"):
                    ok = ok and _close(book["allocated"], book["cost"])
                    totals["round_trips"] += 1
                    totals["wins"] += book["pnl"] > 0
                    equity = _num(fill.get("equity_usd"))
                    if math.isfinite(equity) and equity > 0:
                        totals["log_wealth"] += math.log(max(1e-9, 1 + book["pnl"] / equity))
                    state["open"].pop(token, None)
            if not ok:
                totals["identity_failures"] += 1
                if len(state["identity_examples"]) < 10:
                    state["identity_examples"].append({
                        "sequence": state["sequence"], "token": token,
                        "pnl": pnl, "proceeds": proceeds, "native": native,
                        "allocated": allocated,
                        "book_cost": book["cost"] if book else None,
                        "book_allocated": book["allocated"] if book else None})
        state["first_fill_at"] = state["first_fill_at"] or at
        state["last_fill_at"] = max(at, state["last_fill_at"] or at)
        keep = {k: fill.get(k) for k in ("token", "side", "cost_usd", "proceeds_usd",
                                          "realized_pnl_usd", "reason", "closed", "sleeve")}
        state["recent"] = (state["recent"] + [{"at": at, "sequence": state["sequence"], **keep}])[-25:]

    # -- artifact ----------------------------------------------------
    def artifact(self) -> Dict[str, Any]:
        state, totals, now = self.state, self.state["totals"], self._clock()
        last = state["last_fill_at"]
        checks = {
            "chain_ok": bool(state["chain_ok"]),
            "scan_complete": bool(self.scan_complete),
            "round_trip": totals["round_trips"] > 0,
            "accounting_identity": totals["identity_failures"] == 0 and totals["malformed_rows"] == 0,
            "fresh": last is not None and now - last <= self.fresh_seconds,
        }
        failed = next((name for name in CONDITIONS if not checks[name]), None)
        verdict = "PROVEN" if failed is None else {
            "chain_ok": "CHAIN_BROKEN", "scan_complete": "CATCHING_UP",
            "round_trip": "NO_FILLS" if totals["buys"] == 0 else "NO_ROUND_TRIP",
            "accounting_identity": "ACCOUNTING_MISMATCH", "fresh": "STALE"}[failed]
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        closed = totals["round_trips"]
        return {
            "schema": SCHEMA, "generated_at": now, "verdict": verdict, "checks": checks,
            "contract": list(CONDITIONS), "fresh_seconds": self.fresh_seconds,
            "scope": "paper fills only: simulated buys and sells recorded by the desk; not live orders",
            "evidence": {"path": str(self.evidence_path), "rows_verified": state["sequence"],
                         "verified_bytes": state["offset"], "last_hash": state["last_hash"],
                         "chain_error": state["chain_error"] or None},
            "totals": {**totals,
                       "win_rate": totals["wins"] / closed if closed else None,
                       "mean_log_wealth_per_round_trip": totals["log_wealth"] / closed if closed else None},
            "open_positions": len(state["open"]),
            "open_cost_usd": sum(b["cost"] - b["allocated"] for b in state["open"].values()),
            "today": state["days"].get(today, {"buys": 0, "sells": 0, "pnl": 0.0}),
            "days": dict(sorted(state["days"].items())[-14:]),
            "first_fill_at": state["first_fill_at"], "last_fill_at": last,
            "seconds_since_last_fill": None if last is None else now - last,
            "recent_fills": state["recent"],
            "identity_examples": state["identity_examples"],
        }


def write_artifact(path: str | Path, artifact: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run(state_dir: str | Path, out: Optional[str | Path] = None, *,
        name: str = "paper_fill_proof", **kwargs: Any) -> Dict[str, Any]:
    """One scheduled tick. `name` keeps independent readers off each other's
    checkpoint: the desk uses the default, an ad hoc audit run uses its own."""
    root = Path(state_dir)
    proof = PaperFillProof(root / "trade_evidence.jsonl", root / f"{name}.state.json", **kwargs)
    proof.scan()
    artifact = proof.artifact()
    write_artifact(out or root / f"{name}.json", artifact)
    return artifact


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-dir", default="data/state")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-mb", type=int, default=256)
    parser.add_argument("--duty", type=float, default=0.25)
    parser.add_argument("--name", default="paper_fill_proof.audit",
                        help="checkpoint/artifact name; never the desk's own")
    args = parser.parse_args(argv)
    if args.name == "paper_fill_proof":
        parser.error("the desk owns that checkpoint; choose another --name")
    artifact = run(args.state_dir, args.out, name=args.name,
                   max_bytes=args.max_mb * 1024 * 1024, duty=args.duty)
    print(json.dumps({k: artifact[k] for k in ("verdict", "checks", "open_positions")}
                     | {"round_trips": artifact["totals"]["round_trips"]}))
    return 0 if artifact["verdict"] == "PROVEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
