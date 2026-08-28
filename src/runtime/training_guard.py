"""Keep offline model training from starving the always-on collector.

The VPS also runs other independent research workloads. A chronological
trainer is useful only if it leaves the forward evidence collector alive, so
the systemd unit treats insufficient currently available memory as a clean
skip and tries again on the next hourly tick.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def mem_available_mib(path: Path = Path("/proc/meminfo")) -> Optional[float]:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for row in rows:
        if not row.startswith("MemAvailable:"):
            continue
        fields = row.split()
        try:
            return float(fields[1]) / 1024.0
        except (IndexError, ValueError):
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-available-mib", type=float, default=1400.0)
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"))
    args = parser.parse_args()
    available = mem_available_mib(args.meminfo)
    if available is None:
        print("TRAINING_GUARD=SKIP reason=MemAvailable_unreadable")
        return 1
    if available < max(0.0, args.min_available_mib):
        print("TRAINING_GUARD=SKIP "
              f"available_mib={available:.0f} required_mib={args.min_available_mib:.0f}")
        return 1
    print("TRAINING_GUARD=OK "
          f"available_mib={available:.0f} required_mib={args.min_available_mib:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
