"""Actually run the evidence ledger's own integrity check. Nobody had.

TradeEvidenceLedger.verify() walks the whole hash chain and confirms every
row's hash matches its content and its predecessor -- a complete, correct
implementation of exactly the audit-proof property this ledger exists to
provide. It had never been called outside the test suite. The live status
endpoint's "hash_chain": true means only "this process holds a hash in
memory right now"; it says nothing about whether the file ON DISK is
actually intact, and a crash mid-write, a disk fault, or a manual edit
could silently break the chain while that field kept reporting true.

This is the proprietary moat: 72,000+ append-only records of every
candidate and decision this desk has ever made, structured so tampering or
truncation is mathematically detectable. An auditor that has never been run
is not evidence the audit would pass.

Measured 2026-08-29 on the same 72,469-record file, minutes apart: 4 seconds
at load average ~4, then 42 seconds at load average ~11 -- this shared box's
contention, not the ledger's size, dominates the cost. It is also O(n) in
record count and this file only grows. Both are why this runs on its own
periodic timer rather than inside the every-60-second watchdog loop, and why
the exact sequence number is reported: so a slow run reads as "the box was
busy" and a growing one reads as "the ledger is bigger," never as ambiguity
about which.

Deliberately does nothing about a broken chain beyond naming where it broke.
"Repairing" a corrupted audit trail is not a category of automated action --
truncating or rewriting it IS tampering with the evidence, and the honest
response to a real break is a human decision, not a script's guess.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from src.research.trade_evidence import TradeEvidenceLedger

LEDGER_DOCTOR_SCHEMA_VERSION = "v1"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="verify the trade-evidence hash chain end to end")
    parser.add_argument("--path", default="data/state/trade_evidence.jsonl")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--alert", action="store_true",
                        help="send a Telegram alert (via ops.watchdog's channel) "
                             "if the chain is broken")
    args = parser.parse_args(argv)

    path = Path(args.path)
    started = time.time()
    if not path.exists():
        # Absent before the first record is ever written is not a fault --
        # matches the same convention as the backfill checkpoint.
        report = {"schema": LEDGER_DOCTOR_SCHEMA_VERSION, "path": str(path),
                  "status": "ABSENT", "verified_through_sequence": 0,
                  "elapsed_seconds": 0.0}
        print(json.dumps(report, indent=1) if args.json else "ledger absent (no records yet)")
        return 0

    ok, reason, sequence = TradeEvidenceLedger.verify(path)
    elapsed = time.time() - started
    report = {
        "schema": LEDGER_DOCTOR_SCHEMA_VERSION, "path": str(path),
        "status": "OK" if ok else "BROKEN", "reason": reason,
        "verified_through_sequence": sequence, "elapsed_seconds": round(elapsed, 3),
    }

    if args.json:
        print(json.dumps(report, indent=1))
    elif ok:
        print(f"OK: {sequence} records verified in {elapsed:.1f}s")
    else:
        print(f"BROKEN at sequence {sequence}: {reason}", file=sys.stderr)

    if not ok and args.alert:
        from ops.watchdog import _send_telegram_alert
        _send_telegram_alert(
            f"memecoin ledger doctor: trade-evidence hash chain BROKEN at "
            f"sequence {sequence} ({reason}). This is the audit trail; "
            f"do not truncate or rewrite it without understanding why first.")

    return 1 if not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
