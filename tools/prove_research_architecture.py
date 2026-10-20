"""Retain executable offline architecture proof, source seals and canonical receipts.

Run against a fresh --output-dir. This exercises actual producers and consumers
with declared synthetic native/public inputs. It grants no live capability and
does not certify production performance or unobserved forward time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from src.research.trade_evidence import TradeEvidenceLedger
from tools.prove_unknown_engine import SOURCE_FILES, isolated_runtime

ROOT = Path(__file__).resolve().parents[1]
TESTS = ("tests/test_research_workflows.py", "tests/test_mechanism_labs.py",
    "tests/test_population_hunter.py", "tests/test_population_callbacks.py",
    "tests/test_research_validation.py", "tests/test_native_forward.py")


def seal(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(output):
    args = [str(ROOT / path) for path in TESTS] + ["-q", "--tb=short",
        "--rootdir", str(ROOT), "--basetemp", str(output / "pytest"),
        "-o", "cache_dir=" + str(output / "pytest-cache"), "--junitxml", str(output / "junit.xml")]
    with isolated_runtime(output):
        import pytest
        return int(pytest.main(args))


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = tuple(dict.fromkeys((*SOURCE_FILES, *TESTS, "src/research/actor_store.py",
        "src/research/descendant_generator.py", "tests/test_candidate_replay.py",
        "tests/test_scientific_core.py", "tools/prove_research_architecture.py")))
    before = {name: seal(ROOT / name) for name in paths}
    command = [sys.executable, "-B", "-m", "tools.prove_research_architecture", "--child-output", str(output)]
    allowed = {"path", "systemroot", "windir", "comspec", "temp", "tmp", "userprofile", "home",
               "localappdata", "appdata", "programfiles", "programfiles(x86)", "pathext", "lang", "lc_all"}
    environment = {key: value for key, value in os.environ.items() if key.lower() in allowed}
    environment.update(ALLOW_LIVE_TRADING="", SOLANA_PRIVATE_KEY="", MEMECOIN_SIGNER_SOCKET="",
        PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    started = time.time()
    with (output / "pytest.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=output, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, timeout=600, check=False)
    after = {name: seal(ROOT / name) for name in paths}
    drift = [name for name in paths if before[name] != after[name]]
    ledger = TradeEvidenceLedger(output / "acceptance-evidence.jsonl")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    if (output / "junit.xml").exists():
        for case in ET.parse(output / "junit.xml").iter("testcase"):
            status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else "SKIP" if case.find("skipped") is not None else "PASS"
            counts[status] += 1
            ref = ledger.record_reference("research_architecture_acceptance", {
                "test": case.get("classname", "") + "." + case.get("name", ""),
                "status": status, "duration_seconds": float(case.get("time", 0)),
                "scope": "ISOLATED_SYNTHETIC_PRODUCER_CONSUMER_EXPERIMENT",
                "production_edge": "UNMEASURED", "live_capability": False})
            if ref is None:
                raise RuntimeError("acceptance evidence append failed")
    verified, detail, records = TradeEvidenceLedger.verify(ledger.path)
    manifest = {"schema": "research-architecture-proof-v1", "started_at": started,
        "completed_at": time.time(), "command": command, "exit_code": result.returncode,
        "status": "PASS" if result.returncode == 0 and counts["PASS"] and not counts["SKIP"] and not drift and verified else "FAIL",
        "counts": counts, "scope": "OFFLINE_SYNTHETIC_RUNTIME_ONLY", "production_performance": "UNMEASURED",
        "live_capability": False, "source_sha256": before, "source_sha256_at_end": after,
        "sources_changed_during_run": drift,
        "canonical_evidence_chain": {"verified": verified, "detail": detail, "records": records},
        "files": {str(path.relative_to(output)): {"sha256": seal(path), "bytes": path.stat().st_size}
                  for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "counts", "sources_changed_during_run")}, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    destinations = parser.add_mutually_exclusive_group(required=True)
    destinations.add_argument("--output-dir", type=Path)
    destinations.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child(args.child_output) if args.child_output else run(args.output_dir))
