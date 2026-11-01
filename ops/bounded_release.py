"""Fail-closed, side-effect-free gates for CPU-bounded paper releases.

Callers retain private manifests; exceptions never include action arguments.
This module does not install tasks, modify manifests, or launch processes.
"""
from copy import deepcopy
import math

RATES = {"spot": 500, "prediction": 100, "capture": 150,
         "terminal": 50, "development": 200}
ACTION_KEYS = {"execute", "arguments", "working_directory"}


def plan_role_action(manifest, role, action, *, tested_head, release_head,
                     backup_verified, task_before, task_after):
    """Return a private candidate only when budgets and task identity stay fixed."""
    def require(ok):
        if not ok:
            raise ValueError("Bounded release precondition failed")
    require(role in RATES and role != "development")
    require(isinstance(tested_head, str) and len(tested_head) == 40
            and all(c in "0123456789abcdef" for c in tested_head)
            and tested_head == release_head)
    require(backup_verified is True)
    require(task_before == task_after and bool(task_before))
    require(manifest.get("version") == 1)
    roles = manifest.get("roles", {})
    require(set(roles) == set(RATES))
    for name, rate in RATES.items():
        require(type(roles[name].get("cpu_rate")) is int
                and roles[name]["cpu_rate"] == rate)
    digest = manifest.get("assembly_sha256", "")
    require(isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdefABCDEF" for c in digest))
    require(set(action) == ACTION_KEYS)
    require(all(isinstance(v, str) and v.strip() and "\x00" not in v
                for v in action.values()))
    candidate = deepcopy(manifest)
    candidate["roles"][role].update(action)
    return candidate


def verify_kernel_receipt(receipt, role, *, now, started_at, expected_child):
    """Require fresh post-start CPU-job evidence, never task Running alone."""
    stamp = receipt.get("observed_at")
    valid_clock = (type(stamp) in (int, float) and math.isfinite(stamp)
                   and started_at <= stamp <= now and now - stamp <= 15)
    if not (role in RATES and valid_clock
            and receipt.get("role") == role and receipt.get("state") == "BOUNDED"
            and type(receipt.get("cpu_rate")) is int
            and receipt["cpu_rate"] == RATES[role]
            and receipt.get("control_flags") == 5
            and receipt.get("priority_class") == 16384
            and receipt.get("total_configured_percent") == 10
            and type(expected_child) is int and expected_child > 0
            and receipt.get("child_pid") == expected_child
            and expected_child in receipt.get("process_ids", [])
            and receipt.get("live_authority_changed") is False):
        raise ValueError("Fresh bounded kernel receipt required")
    return True
