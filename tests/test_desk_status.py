"""Reading the desk's health without asking the desk.

`curl localhost:18080/status` needs the desk up, listening and past a
120-second Type=notify startup. That is three ways to get an empty answer at
exactly the moment something is wrong -- which is what happened on
2026-09-04, when the port returned nothing and the diagnosis was the reason
for looking. The snapshot on disk answers while the desk is down, and its
mtime distinguishes "quiet" from "stopped", which a dead port cannot.
"""

import json
import os
import time

from tools.desk_status import (
    EXIT_DATA_BLOCKED, EXIT_OK, EXIT_STALE, STALE_AFTER_S, main)


def _snapshot(tmp_path, payload=None, *, age_s=0.0):
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(payload if payload is not None else
                               {"mode": "DRY_RUN", "prediction": "OK"}))
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return tmp_path


def test_it_reads_the_snapshot_without_the_desk_running(tmp_path, capsys):
    main(["--state-dir", str(_snapshot(tmp_path))])
    assert "DRY_RUN" in capsys.readouterr().out


def test_a_stale_snapshot_says_so_rather_than_reading_as_current(tmp_path, capsys):
    """The difference a dead HTTP port cannot express: quiet versus stopped."""
    state = _snapshot(tmp_path, age_s=STALE_AFTER_S + 60)
    assert main(["--state-dir", str(state)]) == EXIT_STALE
    printed = capsys.readouterr().out
    assert "STALE" in printed
    assert "not now" in printed


def test_a_fresh_snapshot_is_not_flagged(tmp_path):
    assert main(["--state-dir", str(_snapshot(tmp_path))]) == EXIT_OK


def test_a_missing_snapshot_is_explained_not_just_absent(tmp_path, capsys):
    assert main(["--state-dir", str(tmp_path)]) == EXIT_DATA_BLOCKED
    assert "every health tick" in capsys.readouterr().out


def test_a_corrupt_snapshot_does_not_raise(tmp_path):
    (tmp_path / "readiness.json").write_text("{not json")
    assert main(["--state-dir", str(tmp_path)]) == EXIT_DATA_BLOCKED


def test_one_section_can_be_asked_for_in_full(tmp_path, capsys):
    state = _snapshot(tmp_path, {
        "mode": "DRY_RUN",
        "sell_route": {"status": "OK", "skipped_total": 301}})
    main(["--state-dir", str(state), "--key", "sell_route"])
    printed = capsys.readouterr().out
    assert "301" in printed
    assert "DRY_RUN" not in printed, "only the asked-for section"


def test_asking_for_a_missing_section_says_so(tmp_path, capsys):
    main(["--state-dir", str(_snapshot(tmp_path)), "--key", "nonexistent"])
    assert "not in this snapshot" in capsys.readouterr().out


def test_the_default_sections_are_the_ones_a_diagnosis_needs(tmp_path, capsys):
    """These are the keys this session's findings were all read out of."""
    from tools.desk_status import DEFAULT_KEYS
    for key in ("sell_route", "continuation", "capacity", "gauntlet",
                "prediction"):
        assert key in DEFAULT_KEYS, key


def test_the_default_keys_are_ones_the_desk_actually_writes():
    """A default key the snapshot never carries prints noise every run."""
    from pathlib import Path
    from tools.desk_status import DEFAULT_KEYS
    reporting = Path("src/runtime/reporting.py").read_text()
    for key in DEFAULT_KEYS:
        assert f'"{key}"' in reporting, key
