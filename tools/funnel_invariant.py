"""Is the money path connected? One command, two state files, no network.

    PYTHONPATH=. python3 tools/funnel_invariant.py

Exits 0 when every exercised link carries, 3 when one does not, and 4 when
there is not enough evidence to say. Built to be the acceptance test after a
deploy, because "service healthy" was true for 8.76 days while the desk
entered nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.funnel_invariant import DEFAULT_BREAK_THRESHOLD, check

EXIT_OK = 0
EXIT_BROKEN = 3
EXIT_DATA_BLOCKED = 4


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _census_report(state_dir: Path) -> Optional[Dict[str, Any]]:
    """Rebuild the census report from its persisted state."""
    raw = _load(state_dir / "launch_census.json")
    if raw is None:
        return None
    from src.research.launch_census import LaunchCensus
    census = LaunchCensus(path=state_dir / "launch_census.json")
    return census.report()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="data/state")
    parser.add_argument("--threshold", type=int, default=DEFAULT_BREAK_THRESHOLD)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir)
    census = _census_report(state_dir)
    evidence_raw = _load(state_dir / "forward_evidence.json") or {}
    entered = int(evidence_raw.get("entered", 0) or 0)
    evidence = {
        "entered": entered,
        "net_log_growth": (evidence_raw.get("net_log_growth")
                           if entered else None),
    }
    verdict = check(census, evidence, threshold=args.threshold)

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        # Blocked is not broken. A path nobody has walked has not failed.
        label = ("BROKEN" if verdict.breaks else
                 "" if verdict.status == "OK" else "not yet exercised")
        print(f"funnel: {verdict.status}" + (f"  {label}" if label else ""))
        width = max(len(name) for name in verdict.counts) if verdict.counts else 8
        previous = None
        for name, count in verdict.counts.items():
            arrow = "" if previous is None else "  |"
            if arrow:
                print(f"  {'':>{width}}  {arrow}")
            print(f"  {name:>{width}}  {count:>10,}")
            previous = name
        if verdict.breaks:
            print()
            for item in verdict.breaks:
                print(f"  BREAK {item.upstream} -> {item.downstream}")
                print(f"        {item.detail}")
        print()
        print(f"  {verdict.detail}")

    if verdict.status != "OK":
        return EXIT_DATA_BLOCKED
    return EXIT_BROKEN if verdict.breaks else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
