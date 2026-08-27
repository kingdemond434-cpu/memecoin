"""Entry point for the 24/7 monitor. One shot per invocation, driven by a timer.

A long-lived monitor process is one more thing that can die silently. A timer
firing a short process cannot: if the process stops running, its output stops
being fresh, and the freshness of that output is itself the first check the
next run makes. The failure mode is self-reporting rather than invisible.

Exit codes are meaningful so systemd and the escalation hook can act without
parsing anything:

    0  everything OK
    1  something WARN
    2  something CRITICAL, or an escalation fired
    3  the monitor itself could not run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ops.health import HealthThresholds, State, run_health_checks

logger = logging.getLogger("ops.monitor")

EXIT_OK, EXIT_WARN, EXIT_CRITICAL, EXIT_MONITOR_FAILED = 0, 1, 2, 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="local health checks for the trading node")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--readiness", default="data/state/readiness.json")
    parser.add_argument("--models", default="models")
    parser.add_argument("--execution-log", default="data/state/execution_attempts.jsonl")
    parser.add_argument("--out", default="data/state/health.json")
    parser.add_argument("--history", default="data/state/health_history.jsonl")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    root = Path(args.root).resolve()

    try:
        report = run_health_checks(
            readiness_path=root / args.readiness,
            model_dir=root / args.models,
            execution_log=root / args.execution_log,
            root=root,
            thresholds=HealthThresholds(),
        )
    except Exception as exc:  # pragma: no cover - the monitor must not take the node down
        logger.exception("health checks failed to run: %s", exc)
        return EXIT_MONITOR_FAILED

    payload = report.to_dict()
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a reader never sees a half-written snapshot and
    # concludes the node is broken.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(out_path)

    history = root / args.history
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a") as handle:
        handle.write(json.dumps({
            "at": payload["generated_at"], "worst": payload["worst_state"],
            "counts": payload["counts"], "escalations": payload["escalations"],
        }, default=str) + "\n")

    for check in report.checks:
        if check.state is State.CRITICAL:
            logger.error("CRITICAL %s: %s", check.name, check.detail)
        elif check.state is State.WARN:
            logger.warning("WARN %s: %s", check.name, check.detail)
        elif check.state is State.DATA_BLOCKED and not args.quiet:
            logger.info("DATA_BLOCKED %s: %s", check.name, check.detail)

    if report.escalations or report.worst is State.CRITICAL:
        return EXIT_CRITICAL
    if report.worst is State.WARN:
        return EXIT_WARN
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
