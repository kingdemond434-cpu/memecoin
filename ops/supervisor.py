"""One pass of: check health, correct what can be corrected, deploy what verifies.

Run on a timer. Everything it does is bounded, reversible, and reported.

    python -m ops.supervisor --root ~/.local/opt/memecoin-shadow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from ops.alert import Alerter
from ops.autodeploy import AutoDeployer
from ops.autofix import AutoFixer, standard_remedies

logger = logging.getLogger("supervisor")


def read_status(url: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """The desk's own /status, or None if it cannot be reached.

    None is a finding, not an absence: a desk that does not answer is the
    single loudest thing this supervisor can observe.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def build_health(status: Optional[Dict[str, Any]], previous: Optional[Dict[str, Any]],
                 now: float, root: Path) -> Dict[str, Any]:
    """Every check this supervisor knows how to make."""
    from ops.health import HealthThresholds, check_pipeline, check_subsystems

    if status is None:
        return {"checks": [{
            "name": "readiness_freshness", "state": "CRITICAL",
            "detail": "the desk did not answer /status", "evidence": {},
            "escalate": True}]}
    thresholds = HealthThresholds()
    checks = (check_pipeline(status, now, thresholds, previous)
              + check_subsystems(status, root, now, thresholds))
    return {"checks": [{"name": check.name, "state": check.state.value,
                        "detail": check.detail, "evidence": check.evidence,
                        "escalate": check.escalate}
                       for check in checks]}


def _fast_pass(args, root: Path, state: Path, now: float) -> int:
    """Is the desk answering at all. Nothing else.

    Deliberately minimal: one HTTP call and, at most, one restart. The full
    pass reads a large status document, evaluates thirty checks and may run
    the test suite -- none of which should stand between a wedged desk and a
    restart. Splitting them is what makes a thirty-second cadence affordable.
    """
    alive = read_status(args.status_url.replace("/status", "/health"),
                        timeout=5.0) is not None
    if alive:
        return 0
    health = {"checks": [{"name": "readiness_freshness", "state": "CRITICAL",
                          "detail": "the desk did not answer /health",
                          "evidence": {}, "escalate": True}]}
    fixer = AutoFixer(state_path=state / "autofix.json")
    for remedy in standard_remedies(args.service, root=root):
        fixer.register(remedy)
    acted = [] if args.no_fix else fixer.run(health, now)
    print(json.dumps({"at": now, "fast": True, "alive": False,
                      "acted": [item.to_dict() for item in acted]}))
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.home() / ".local/opt/memecoin-shadow"))
    parser.add_argument("--status-url", default="http://127.0.0.1:18080/status")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--service", default="memecoin-shadow.service")
    parser.add_argument("--no-deploy", action="store_true",
                        help="check and correct, but do not pull")
    parser.add_argument("--no-fix", action="store_true",
                        help="report only; take no corrective action")
    parser.add_argument("--no-alert", action="store_true",
                        help="do not notify; still writes the paper trail")
    parser.add_argument("--fast", action="store_true",
                        help=("liveness only: is the desk answering. Cheap "
                              "enough to run every thirty seconds, so the "
                              "worst case for a wedged desk is half a minute "
                              "rather than a full pass"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s supervisor %(message)s")

    root = Path(args.root)
    state = root / "data" / "state"
    now = time.time()

    if args.fast:
        return _fast_pass(args, root, state, now)

    # Deploy first. A fault the newest commit fixes should not be "corrected"
    # by restarting the code that has it.
    deployed = None
    if not args.no_deploy:
        deployed = AutoDeployer(root, branch=args.branch, service=args.service,
                                state_path=state / "autodeploy.json").run()
        if deployed.status in ("DEPLOYED",):
            # The desk was just restarted; its health means nothing yet.
            logger.info("deployed %s; skipping this health pass",
                        deployed.now[:8])
            print(json.dumps({"deploy": deployed.to_dict()}))
            return 0

    previous_path = state / "supervisor_last_status.json"
    previous = None
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text())
        except (OSError, json.JSONDecodeError):
            previous = None

    status = read_status(args.status_url)
    health = build_health(status, previous, now, root)

    fixer = AutoFixer(state_path=state / "autofix.json")
    for remedy in standard_remedies(args.service, root=root,
                                    status_base=args.status_url.rsplit("/", 1)[0]):
        fixer.register(remedy)
    acted = [] if args.no_fix else fixer.run(health, now)

    if status is not None:
        try:
            state.mkdir(parents=True, exist_ok=True)
            previous_path.write_text(json.dumps({**status, "_observed_at": now}))
        except OSError:
            pass

    critical = [check for check in health["checks"] if check["state"] == "CRITICAL"]

    # Escalate what a person has to know about: anything critical, and any
    # fault the fixer has given up on. Deduplicated, so a flapping fault does
    # not become a flapping notification.
    alerter = Alerter(log_path=state / "escalations.jsonl",
                      state_path=state / "alert_state.json")
    if not args.no_alert:
        for check in critical:
            alerter.escalate(check["name"],
                             f"{check['name']}: {check['detail']}", now)
        for message in fixer.report(now)["escalations"]:
            alerter.escalate("autofix_exhausted", message, now)
        # And close the loop on anything that recovered.
        still_bad = {check["name"] for check in critical}
        for key in list(alerter.sent):
            if key not in still_bad and key != "autofix_exhausted":
                alerter.clear(key, now)

    report = {
        "at": now,
        "deploy": deployed.to_dict() if deployed else None,
        "critical": [check["name"] for check in critical],
        "acted": [item.to_dict() for item in acted],
        "autofix": fixer.report(now),
        "alerts": alerter.report(),
        "checks": health["checks"],
    }
    print(json.dumps(report))
    # 0 healthy, 1 acted or degraded, 2 escalated and needs a person.
    if fixer.report(now)["escalations"]:
        return 2
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
