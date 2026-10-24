"""Run a pinned test release in a fresh checkout with no production state.

Example: python -m tools.validate_checkout --revision <40-character SHA>
  --output C:/validation/new-run
Keep output outside the running repository. Untracked state and credentials
are never copied; receipts, logs and the checkout are retained for inspection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def validate(source, revision, output, *, python=sys.executable, tests=("tests",)):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("exact immutable commit SHA required")
    if source == output or source in output.parents:
        raise ValueError("validation output must be outside the production checkout")
    for target in tests:
        path = Path(target)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "tests":
            raise ValueError("only relative tests/ paths are accepted")
    output.mkdir(parents=True, exist_ok=False)
    checkout = output / "repo"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--local", str(source), str(checkout)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "config", "core.hooksPath", "deploy/hooks"], check=True)
    subprocess.run(["git", "-C", str(checkout), "switch", "--quiet", "--detach", revision],
                   check=True, capture_output=True)
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise RuntimeError("validation checkout differs from pinned release")
    for relative in (".env", "data/state/trade_evidence.jsonl", "data/research/champion_state.json"):
        path = checkout / relative
        if path.is_file() and path.stat().st_size:
            raise RuntimeError("committed credentials or mutable state present; refusing validation")
    allowed = {"path", "systemroot", "windir", "comspec", "temp", "tmp", "userprofile", "home",
               "localappdata", "appdata", "programfiles", "programfiles(x86)", "pathext", "lang", "lc_all"}
    environment = {key: value for key, value in os.environ.items() if key.lower() in allowed}
    environment.update(ALLOW_LIVE_TRADING="", SOLANA_PRIVATE_KEY="", MEMECOIN_SIGNER_SOCKET="",
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=str(checkout))
    junit = output / "junit.xml"
    command = [str(python), "-m", "pytest", *tests, "-q", "-p", "pytest_asyncio.plugin",
               "--junitxml", str(junit)]
    started = time.time()
    with (output / "pytest.log").open("w", encoding="utf-8") as log:
        try:
            exit_code = subprocess.run(command, cwd=checkout, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=900, check=False).returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    if junit.is_file():
        root = ET.parse(junit).getroot()
        for case in root.iter("testcase"):
            kind = ("failed" if case.find("failure") is not None else
                    "errors" if case.find("error") is not None else
                    "skipped" if case.find("skipped") is not None else "passed")
            counts[kind] += 1
    report = {"status": "PASS" if exit_code == 0 else "FAIL", "revision": head,
              "checkout": str(checkout), "exit_code": exit_code, "counts": counts,
              "started_at": started, "elapsed_seconds": time.time() - started,
              "production_state_copied": False, "credential_environment": "REMOVED",
              "scope": "software tests; no forward or profitability certification"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", action="append", dest="tests")
    args = parser.parse_args(argv)
    report = validate(args.source, args.revision, args.output, tests=args.tests or ("tests",))
    print(json.dumps(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
