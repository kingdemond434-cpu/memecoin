"""Two training paths, one corpus, 4 GB of RAM.

The desk trains in-process on its own clock; a systemd timer trains every
fifteen minutes. Both invoke the same trainers over the same corpus and
neither knew the other existed, which on this box means two resident copies of
the labelled corpus and a kernel choosing a victim. Four OOM kills is what that
looked like from outside, and it looked identical to "the trainer is too big",
which is why the memory cap kept being adjusted at the wrong thing.

These tests pin the exclusion, the priority, and the drift detector that would
have caught the installed unit saying `MemoryMax=infinity` while the file in
this repository said `640M`.
"""

import asyncio
import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from src.runtime.training_lock import TrainingBusy, holder, lock_path, training_lock
from tools.unit_drift import audit, compare, parse_bytes, parse_unit_file

UNIT_DIR = Path("deploy/systemd")
TRAINER = UNIT_DIR / "memecoin-shadow-trainer.service"
DESK = UNIT_DIR / "memecoin-shadow.service"


# --- the lock ------------------------------------------------------------

def test_a_second_holder_is_refused_immediately(tmp_path):
    with training_lock(tmp_path, owner="first"):
        with pytest.raises(TrainingBusy) as caught:
            with training_lock(tmp_path, owner="second"):
                pytest.fail("two trainers held the lock at once")
    assert "already training" in str(caught.value)


def test_the_lock_is_released_when_the_holder_leaves(tmp_path):
    with training_lock(tmp_path, owner="first"):
        pass
    with training_lock(tmp_path, owner="second"):
        pass   # no raise


def test_the_lock_names_its_holder(tmp_path):
    with training_lock(tmp_path, owner="training_supervisor"):
        busy, who = holder(tmp_path)
        assert busy
        assert "training_supervisor" in who
        assert f"pid={os.getpid()}" in who


def test_holder_reports_free_when_nobody_is_training(tmp_path):
    with training_lock(tmp_path, owner="x"):
        pass
    assert holder(tmp_path) == (False, "")


def test_holder_is_free_before_the_file_exists(tmp_path):
    assert holder(tmp_path) == (False, "")
    assert not lock_path(tmp_path).exists()


def _child(directory, ready, done):
    from src.runtime.training_lock import training_lock as lock
    with lock(Path(directory), owner="child"):
        ready.set()
        done.wait(10)


def test_the_lock_excludes_a_genuinely_separate_process(tmp_path):
    """The whole point: this must hold across processes, not just threads.

    A threading lock, a module-level flag or a PID file would all pass a
    same-process test and fail the actual scenario, which is a systemd unit
    and a long-running desk that share nothing but a filesystem.
    """
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    done = context.Event()
    process = context.Process(target=_child, args=(str(tmp_path), ready, done))
    process.start()
    try:
        assert ready.wait(30), "child never acquired the lock"
        busy, who = holder(tmp_path)
        assert busy
        assert "child" in who
        with pytest.raises(TrainingBusy):
            with training_lock(tmp_path, owner="parent"):
                pytest.fail("acquired a lock another process holds")
    finally:
        done.set()
        process.join(30)
    # And once that process is gone the lock is free, without anyone
    # cleaning up a stale PID file.
    assert holder(tmp_path) == (False, "")


def test_a_killed_holder_does_not_leave_the_lock_stuck(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    done = context.Event()
    process = context.Process(target=_child, args=(str(tmp_path), ready, done))
    process.start()
    assert ready.wait(30)
    process.kill()
    process.join(30)
    # SIGKILL, no cleanup, no atexit. The kernel released it.
    assert holder(tmp_path) == (False, "")
    with training_lock(tmp_path, owner="after"):
        pass


# --- the supervisor takes it --------------------------------------------

def test_the_supervisor_skips_rather_than_running_a_second_pass(tmp_path):
    from src.runtime.training import TrainingSupervisor
    storage = tmp_path / "data" / "launch_episodes"
    storage.mkdir(parents=True)
    supervisor = TrainingSupervisor(storage, tmp_path / "models", jobs=[])
    with training_lock(tmp_path / "data", owner="systemd"):
        result = asyncio.run(supervisor.run_round())
    assert result["status"] == "SKIPPED_LOCKED"
    assert "systemd" in result["detail"]
    # A skip is not a round: the counters must not move, or the next
    # should_train() decision is made against evidence that was never used.
    assert supervisor.rounds == 0


def test_the_supervisor_runs_when_the_lock_is_free(tmp_path):
    from src.runtime.training import TrainingSupervisor
    storage = tmp_path / "data" / "launch_episodes"
    storage.mkdir(parents=True)
    supervisor = TrainingSupervisor(storage, tmp_path / "models", jobs=[])
    result = asyncio.run(supervisor.run_round())
    assert result.get("status") != "SKIPPED_LOCKED"
    assert supervisor.rounds == 1


# --- the systemd entry point --------------------------------------------

def test_train_once_reports_a_skip_with_exit_75(tmp_path, capsys):
    from src.runtime import train_once
    storage = tmp_path / "data" / "launch_episodes"
    storage.mkdir(parents=True)
    with training_lock(tmp_path / "data", owner="the desk"):
        code = train_once.main(["--storage", str(storage),
                                "--model-dir", str(tmp_path / "models")])
    assert code == train_once.EXIT_BUSY
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SKIPPED_LOCKED"
    assert "the desk" in payload["holder"]


def test_train_once_is_not_due_on_an_empty_corpus(tmp_path, capsys):
    from src.runtime import train_once
    storage = tmp_path / "data" / "launch_episodes"
    storage.mkdir(parents=True)
    code = train_once.main(["--storage", str(storage),
                            "--model-dir", str(tmp_path / "models")])
    assert code == train_once.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_DUE"
    assert "no resolved episodes" in payload["detail"]


# --- the unit files ------------------------------------------------------

def test_the_trainer_unit_runs_the_locked_entry_point_not_raw_trainers():
    directives = parse_unit_file(TRAINER)
    starts = directives.get("ExecStart") or []
    assert len(starts) == 1, (
        "three independent ExecStarts cannot take one lock between them")
    assert "src.runtime.train_once" in starts[0]
    for orphan in ("src.research.shadow_trainer", "src.research.hazard_trainer",
                   "src.research.exit_policy_trainer"):
        assert not any(orphan in line for line in starts), (
            f"{orphan} invoked directly; it would run unlocked")


def test_the_trainer_unit_treats_a_lock_skip_as_success():
    directives = parse_unit_file(TRAINER)
    assert "75" in (directives.get("SuccessExitStatus") or [""])[0], (
        "a skip would show up in `list-units --failed` as noise")


def test_the_trainer_is_capped_and_throttled_before_it_is_killed():
    directives = parse_unit_file(TRAINER)
    hard = parse_bytes((directives.get("MemoryMax") or [""])[0])
    soft = parse_bytes((directives.get("MemoryHigh") or [""])[0])
    assert hard is not None, "an uncapped trainer is what caused the incident"
    assert soft is not None and soft < hard, (
        "without a soft limit the first byte over the cap is fatal")


def test_the_kernel_is_told_to_take_the_trainer_and_not_the_desk():
    trainer = parse_unit_file(TRAINER)
    desk = parse_unit_file(DESK)
    trainer_score = int((trainer.get("OOMScoreAdjust") or ["0"])[0])
    desk_score = int((desk.get("OOMScoreAdjust") or ["0"])[0])
    assert trainer_score > 0, "a round can be repeated on the next tick"
    assert desk_score < 0, "forward evidence cannot be recreated at any price"
    assert trainer_score > desk_score


def test_the_desk_still_has_its_own_ladder():
    desk = parse_unit_file(DESK)
    hard = parse_bytes((desk.get("MemoryMax") or [""])[0])
    soft = parse_bytes((desk.get("MemoryHigh") or [""])[0])
    assert hard and soft and soft < hard


# --- drift ---------------------------------------------------------------

def test_an_uncapped_box_against_a_capped_repo_is_critical():
    drifts = compare("u", {"MemoryMax": ["1200M"]}, {"MemoryMax": "infinity"})
    assert [item.severity for item in drifts] == ["CRITICAL"]
    assert "unbounded" in drifts[0].detail


def test_the_same_cap_written_two_ways_is_not_drift():
    assert compare("u", {"MemoryMax": ["1200M"]},
                   {"MemoryMax": str(1200 * 1024 ** 2)}) == []


def test_a_missing_oom_priority_on_the_box_is_critical():
    drifts = compare("u", {"OOMScoreAdjust": ["500"]}, {"OOMScoreAdjust": ""})
    assert drifts and drifts[0].severity == "CRITICAL"


def test_a_box_running_a_different_command_is_critical():
    drifts = compare(
        "u", {"ExecStart": ["/x/python -m src.runtime.train_once --storage d"]},
        {"ExecStart": "{ path=/x/python ; argv[]=/x/python -m src.research.shadow_trainer }"})
    assert any(item.prop == "ExecStart" and item.severity == "CRITICAL"
               for item in drifts)


def test_a_matching_command_is_not_drift():
    drifts = compare(
        "u", {"ExecStart": ["/x/python -m src.runtime.train_once"]},
        {"ExecStart": "{ path=/x/python ; argv[]=/x/python -m src.runtime.train_once }"})
    assert not [item for item in drifts if item.prop == "ExecStart"]


def test_the_audit_reproduces_the_incident():
    """The exact shape of 2026-09-03: repo capped, box infinite."""
    def show(unit):
        if unit != "memecoin-shadow-trainer.service":
            return {}
        return {"MemoryMax": "infinity", "MemoryHigh": "infinity",
                "OOMScoreAdjust": "0", "CPUQuota": "",
                "ExecStart": "{ argv[]=python -m src.runtime.train_once }",
                "SuccessExitStatus": "", "Nice": "10"}

    report = audit(UNIT_DIR, show=show,
                   units=["memecoin-shadow-trainer.service"])
    assert report["status"] == "CRITICAL"
    fields = {item["property"] for item in report["critical"]}
    assert {"MemoryMax", "MemoryHigh", "OOMScoreAdjust"} <= fields
    assert "daemon-reload" in report["remedy"]


def test_a_unit_not_installed_is_data_blocked_not_clean():
    report = audit(UNIT_DIR, show=lambda unit: {},
                   units=["memecoin-shadow-trainer.service"])
    assert report["status"] == "DATA_BLOCKED"
    assert report["unreadable"][0]["reason"] == "not installed on this box"


def test_a_box_matching_the_repo_is_clean():
    directives = parse_unit_file(TRAINER)

    def show(unit):
        return {prop: (directives.get(prop) or [""])[-1]
                for prop in ("MemoryMax", "MemoryHigh", "OOMScoreAdjust",
                             "CPUQuota", "SuccessExitStatus", "Nice")} | {
            "ExecStart": "{ argv[]=" + (directives["ExecStart"][0]) + " }"}

    report = audit(UNIT_DIR, show=show,
                   units=["memecoin-shadow-trainer.service"])
    assert report["status"] == "OK", report["drift"]
