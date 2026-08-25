"""Entry point for the weekly evidence pack.

Runs on the node, on a timer, and writes a single capped JSON file. The Claude
audit reads that file. It does not read the repository's history, the research
lake, or the logs -- that is the entire point of the pack existing.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.audit_pack import DEFAULT_MAX_BYTES, build_audit_pack
from ops.health import HealthThresholds, run_health_checks

logger = logging.getLogger("ops.build_audit_pack")


def _load_jsonl(path: Path, since: float) -> Optional[List[Dict[str, Any]]]:
    """Rows newer than ``since``. None (not []) when the file is missing.

    The distinction matters downstream: an empty list means "nothing happened",
    None means "we could not look", and the pack reports those differently.
    """
    if not path.exists():
        return None
    rows: List[Dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if float(row.get("timestamp", 0) or 0) >= since:
                    rows.append(row)
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None
    return rows


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="build the weekly Claude audit pack")
    parser.add_argument("--root", default=".")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", default="data/audit/latest_pack.json")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    root = Path(args.root).resolve()
    since = time.time() - args.days * 86_400
    state = root / "data" / "state"

    health = run_health_checks(
        readiness_path=state / "readiness.json",
        model_dir=root / "models",
        execution_log=state / "execution_attempts.jsonl",
        root=root, thresholds=HealthThresholds(),
    )

    trades = _load_jsonl(state / "trade_outcomes.jsonl", since)
    leak_report = None
    tail_report = None
    premature_report = None
    ledger = None
    if trades:
        # Imported here so the pack builder still runs on a node where the
        # research package is unavailable; a missing analysis is a
        # DATA_BLOCKED section, not a crashed timer.
        try:
            from src.research.attribution import (
                alpha_ledger, find_leaks, rank_research, tail_contribution)
            from src.strategies.monster import premature_exit_rates

            equity = float((_load_json(state / "readiness.json") or {})
                           .get("equity", {}).get("wallet_equity_usd", 0) or 0)
            if equity > 0:
                leak_report = rank_research(find_leaks(trades, equity))
            tail_report = tail_contribution(trades)
            premature_report = premature_exit_rates(trades)
            ledger = alpha_ledger(trades)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("attribution unavailable: %s", exc)

    pack = build_audit_pack(
        repo_root=root,
        health=health,
        leak_report=leak_report,
        trades=trades,
        rug_events=trades,
        execution_stats=_load_json(state / "execution_stats.json"),
        ledger=ledger,
        decay=_load_json(state / "edge_decay.json"),
        tail_report=tail_report,
        premature_report=premature_report,
        moat_stats=_load_json(state / "moat_stats.json"),
        period_days=args.days,
        run_tests=not args.skip_tests,
    )

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = pack.serialise(max_bytes=args.max_bytes)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(out_path)
    logger.info("audit pack written to %s (%d bytes)", out_path, len(text.encode()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
