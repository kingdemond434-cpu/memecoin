#!/usr/bin/env python3
"""Inbox watcher: runs the full pipeline whenever a discovery JSON lands in
data/discoveries/inbox/. Polling, stdlib only — suitable for a systemd
service or a tmux pane.

    python watch.py [--interval 10]
"""

from __future__ import annotations

import argparse
import time

from pipeline.util import PLATFORM_DIR
from run_pipeline import run

INBOX = PLATFORM_DIR / "data" / "discoveries" / "inbox"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=10, help="poll seconds")
    args = parser.parse_args()

    print(f"Watching {INBOX} every {args.interval}s — Ctrl+C to stop.")
    while True:
        if any(INBOX.glob("*.json")):
            print("[watch] new discovery detected, running pipeline…")
            run(PLATFORM_DIR / "data", PLATFORM_DIR / "site",
                PLATFORM_DIR / "config" / "platform.json")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
