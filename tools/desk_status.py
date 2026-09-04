"""Read the desk's own health snapshot off disk, without asking it anything.

`curl localhost:18080/status` needs the desk to be up, listening, and past
startup. That is three ways to get an empty answer at exactly the moment
something is wrong, and it is what happened on 2026-09-04: the port answered
nothing while the diagnosis was the reason for looking.

The desk already writes the entire readiness snapshot to disk every health
tick, atomically, for the out-of-process monitor. That file answers the same
questions and answers them while the desk is down, restarting, or wedged --
and its mtime says which of those it is, which the HTTP endpoint cannot say
at all because a dead desk does not reply.

    python -m tools.desk_status                 the sections that matter most
    python -m tools.desk_status --key sell_route  one section, in full
    python -m tools.desk_status --list          what sections exist

Reads one file and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

EXIT_OK = 0
EXIT_STALE = 3
EXIT_DATA_BLOCKED = 4

#: How old a snapshot may be before it describes a desk that has stopped
#: writing rather than a desk that is quiet. The health loop writes every
#: minute; five gives room for a slow tick without hiding a dead one.
STALE_AFTER_S = 300.0

#: The sections worth showing by default, in the order a diagnosis reads
#: them. Everything else is one `--key` away.
DEFAULT_KEYS = ("mode", "live_submission_locked", "prediction", "continuation",
                "sell_route", "capacity", "gauntlet", "exit_policy",
                "equity", "promotion_blocked_at")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="data/state")
    parser.add_argument("--key", action="append", default=[],
                        help="show this section in full; repeatable")
    parser.add_argument("--list", action="store_true",
                        help="list the sections the snapshot carries")
    args = parser.parse_args(argv)

    path = Path(args.state_dir) / "readiness.json"
    if not path.exists():
        print(f"no readiness snapshot at {path}. The desk writes this every "
              "health tick, so its absence means it has not completed one "
              "since the file was last removed.")
        return EXIT_DATA_BLOCKED
    try:
        snapshot: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"readiness snapshot unreadable: {exc}")
        return EXIT_DATA_BLOCKED

    age = time.time() - path.stat().st_mtime
    stale = age > STALE_AFTER_S
    print(f"snapshot age: {age:.0f}s"
          + ("  STALE -- the desk has stopped writing; what follows describes "
             "the last tick it managed, not now" if stale else ""))
    print()

    if args.list:
        for key in sorted(snapshot):
            print(f"  {key}")
        return EXIT_STALE if stale else EXIT_OK

    keys = args.key or [key for key in DEFAULT_KEYS if key in snapshot]
    for key in keys:
        if key not in snapshot:
            print(f"{key}: not in this snapshot")
            continue
        value = snapshot[key]
        if isinstance(value, (dict, list)):
            print(f"{key}:")
            print("  " + json.dumps(value, indent=2, default=str
                                    ).replace("\n", "\n  "))
        else:
            print(f"{key}: {value}")
        print()
    return EXIT_STALE if stale else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
