from copy import deepcopy
import pytest
from ops.bounded_release import RATES, plan_role_action, verify_kernel_receipt


def inputs():
    manifest = {"version": 1, "assembly_sha256": "a" * 64,
                "roles": {k: {"cpu_rate": v, "task": k, "execute": "old",
                              "arguments": "private", "working_directory": "old"}
                          for k, v in RATES.items()}}
    kw = dict(tested_head="a" * 40, release_head="a" * 40,
              backup_verified=True, task_before={"wrapper": "fixed", "priority": 5},
              task_after={"wrapper": "fixed", "priority": 5})
    return manifest, kw


def test_candidate_preserves_original_other_roles_and_budgets():
    manifest, kw = inputs()
    original = deepcopy(manifest)
    result = plan_role_action(manifest, "spot", dict(execute="new", arguments="new", working_directory="new"), **kw)
    assert manifest == original
    assert result["roles"]["spot"]["cpu_rate"] == 500
    assert result["roles"]["prediction"] == manifest["roles"]["prediction"]
    assert result["assembly_sha256"] == manifest["assembly_sha256"]


@pytest.mark.parametrize("field,value", [("backup_verified", False), ("release_head", "b" * 40),
                                         ("task_after", {"wrapper": "unwrapped"})])
def test_release_gates(field, value):
    manifest, kw = inputs()
    kw[field] = value
    with pytest.raises(ValueError):
        plan_role_action(manifest, "spot", dict(execute="new", arguments="new", working_directory="new"), **kw)


@pytest.mark.parametrize("rate", [501, True, 1000])
def test_budget_changes_rejected(rate):
    manifest, kw = inputs()
    manifest["roles"]["spot"]["cpu_rate"] = rate
    with pytest.raises(ValueError):
        plan_role_action(manifest, "spot", dict(execute="new", arguments="new", working_directory="new"), **kw)


def receipt():
    return dict(observed_at=100, role="spot", state="BOUNDED", cpu_rate=500,
                control_flags=5, priority_class=16384, total_configured_percent=10,
                child_pid=42, process_ids=[41, 42], live_authority_changed=False)


def test_fresh_kernel_receipt():
    assert verify_kernel_receipt(receipt(), "spot", now=105, started_at=99, expected_child=42)


@pytest.mark.parametrize("field,value", [("observed_at", 98), ("observed_at", 106),
    ("observed_at", float("nan")), ("control_flags", 1), ("priority_class", 32),
    ("process_ids", [41]), ("state", "EXITED"), ("cpu_rate", 501)])
def test_kernel_fail_closed(field, value):
    row = receipt()
    row[field] = value
    with pytest.raises(ValueError):
        verify_kernel_receipt(row, "spot", now=105, started_at=99, expected_child=42)
