"""Read-only bounded exit provenance/conservation diagnostics; no SVM or RPC execution."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from types import SimpleNamespace

from src.research.fork_probe import ForkProbe
from src.research.fork_simulation import digest
from src.research.paper_bank import inventory_keys


def safe_hash(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else digest(value)


def sealed(value):
    return value.get("seal") == digest({k: v for k, v in value.items() if k != "seal"})


def integer(value):
    return type(value) is int and value >= 0


def local_code(receipt):
    # Parse local VM metadata only, never forward logs/provider error strings.
    match = re.search(r"Custom\((\d+)\)", str(receipt.get("error", "")))
    return int(match.group(1)) if match else None


def diagnose(db, job, probe_id, body, stored_inventory):
    row = json.loads(body)
    control = row["paper_control"]
    execution = control["execution"]
    current = json.loads(stored_inventory)
    checks = {"result_hash": digest({k: v for k, v in row.items() if k != "probe_id"}) == probe_id,
              "inventory_matches_result": current == execution["inventory"],
              "inventory_seal": sealed(current),
              "paper_only": execution.get("real_fill") is False and execution.get("promotion_eligible") is False}
    venue = row["target"].get("venue")
    result = {"probe_sha256": safe_hash(probe_id), "venue": venue if venue in
              {"pump_curve", "pumpswap", "meteora_dbc", "raydium_launchlab"} else "UNSUPPORTED",
              "local_error_code": local_code(execution.get("exit", {})), "checks": checks}
    frame_id = control["source_frame_sha256"]
    frame = ForkProbe._read_frame(SimpleNamespace(db=db, lock=threading.RLock()), frame_id)
    checks["frame_binding"] = (execution["exit_frame_sha256"] == frame_id == row["input_sha256"]
                                and frame["target"] == current["target"])
    result["exit_frame_sha256"] = frame_id
    previous = db.execute("SELECT body,available FROM results WHERE job=? AND ref=? AND id!=? ORDER BY created DESC LIMIT 2",
                          (job, control.get("previous_canonical_ref"), probe_id)).fetchall()
    checks["unique_prior_canonical_inventory"] = len(previous) == 1
    if len(previous) != 1:
        return result
    prior = json.loads(previous[0][0])["paper_control"]["execution"]["inventory"]
    checks["prior_seal"] = sealed(prior)
    available = previous[0][1]
    checks["prior_canonical_pit"] = (type(available) in (int, float) and math.isfinite(available)
        and 0 < available <= frame["observed_at"])
    checks["inventory_identity"] = prior["target"] == current["target"] and prior["token_program"] == current["token_program"]
    checks["exit_clock_binding"] = (execution["slot"] == current["last_slot"] == frame["slot"]
        and execution["observed_at"] == current["last_observed_at"] == frame["observed_at"])
    expected_input = {**frame, "_paper_request": {"phase": "exit", "inventory": prior}}
    checks["exact_input_binding"] = digest(expected_input) == execution["input_sha256"]
    checks["strictly_later_frame"] = (frame["slot"] > prior["last_slot"] and
                                        frame["observed_at"] > prior["last_observed_at"])
    allowed = inventory_keys(frame, prior["token_program"])
    checks["reviewed_overlay_only"] = (set(prior["accounts"]) == set(current["accounts"]) == allowed
        and all(frame["accounts"].get(key) is None for key in allowed))
    checks["units_bound"] = (integer(prior["token_units"]) and prior["token_units"] > 0
        and prior["token_units"] == execution["sell_input_token_units"]
        and current["token_units"] == execution["remaining_token_units"])
    fee = execution["exit"]["vm_fee_lamports"]
    checks["fee_accumulation"] = (integer(fee) and integer(prior["vm_fees_lamports"])
        and current["vm_fees_lamports"] == prior["vm_fees_lamports"] + fee
        and execution["vm_fees_lamports"] == current["vm_fees_lamports"])
    checks["entry_provenance_retained"] = (current["entry_frame_sha256"] == prior["entry_frame_sha256"]
        and current["last_frame_sha256"] == frame_id)
    if execution["exit"].get("succeeded") is False:
        checks["failed_exit_units_retained"] = current["token_units"] == prior["token_units"]
        before, after = prior["accounts"], current["accounts"]
        checks["failed_exit_fee_debit"] = (sum(v["lamports"] for v in before.values() if v)
            - sum(v["lamports"] for v in after.values() if v) == fee)
        def without_lamports(accounts):
            return {k: None if v is None else {x: y for x, y in v.items() if x != "lamports"}
                    for k, v in accounts.items()}
        checks["failed_exit_account_data_retained"] = without_lamports(before) == without_lamports(after)
    for name, value in (("sell_input_token_units", execution["sell_input_token_units"]),
                        ("retained_token_units", current["token_units"]), ("exit_fee_lamports", fee),
                        ("cumulative_vm_fees_lamports", current["vm_fees_lamports"])):
        result[name] = value if integer(value) else None
    return result


def audit(path, *, sample_limit=20):
    if type(sample_limit) is not int or not 1 <= sample_limit <= 100:
        raise ValueError("bounded sample_limit must be 1..100")
    db = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=5)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        counts = dict(db.execute("SELECT json_extract(r.body,'$.paper_control.execution.status'),count(*) "
            "FROM paper_bank_inventory p JOIN results r ON r.id=p.probe_id WHERE r.ref IS NOT NULL GROUP BY 1"))
        known_statuses={"PAPER_BANK_OPENED", "PAPER_BANK_CLOSED", "PAPER_BANK_ENTRY_FAILED", "PAPER_BANK_EXIT_UNRESOLVED"}
        clean_counts=Counter()
        for status, count in counts.items():
            clean_counts[status if status in known_statuses else "UNKNOWN"] += count
        counts=dict(clean_counts)
        pending = db.execute("SELECT count(*) FROM paper_bank_inventory p JOIN results r ON r.id=p.probe_id "
                             "WHERE r.ref IS NULL").fetchone()[0]
        rows = db.execute("SELECT r.job,r.id,r.body,p.body FROM paper_bank_inventory p JOIN results r ON r.id=p.probe_id "
            "WHERE r.ref IS NOT NULL AND json_extract(r.body,'$.paper_control.execution.status')="
            "'PAPER_BANK_EXIT_UNRESOLVED' ORDER BY r.created DESC,r.id LIMIT ?", (sample_limit,)).fetchall()
        samples=[]
        for job, probe_id, body, inventory in rows:
            try:
                samples.append(diagnose(db, job, probe_id, body, inventory))
            except Exception as exc:
                # Hashes and exception type only; no account contents or private paths/errors.
                samples.append({"probe_sha256": safe_hash(probe_id), "checks": {"diagnostic_complete": False},
                                "error_type": type(exc).__name__})
        failures = Counter(key for row in samples for key, passed in row["checks"].items() if not passed)
        return {"status": "FAIL" if failures else "PASS" if samples else "NO_UNRESOLVED_SAMPLE",
                "canonical_execution_statuses": counts, "pending_canonical_inventory_results": pending,
                "sampled_unresolved": len(samples), "failure_categories": dict(failures), "samples": samples,
                "scope": "sampled archived frame/input/overlay/unit/VM-fee integrity, not external canonical-ledger verification",
                "protocol_failure_resolved": False, "honeypot_proven": False, "real_fill": False,
                "profitable_edge_established": False}
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--sample-limit", type=int, default=20)
    args=parser.parse_args()
    try: result=audit(args.database, sample_limit=args.sample_limit)
    except Exception as exc: result={"status":"ERROR", "error_type":type(exc).__name__}
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
