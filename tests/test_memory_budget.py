"""Memecoin runs inside one memory budget, and cannot leave it.

This box is shared with the quant platform, which is why it exists. Reported
2026-09-04: memecoin work was leaving quant processes stale. That was not one
unit misbehaving -- the desk at 1341M, the trainer at 1200M and the gauntlet
at 1200M are each individually defensible and jointly 3.7 GB on a 3814 MB
machine. Per-unit caps bound each process and bound the family at nothing.

The slice is the total. These tests exist because the failure mode is a NEW
unit added later that quietly sits outside it: nothing breaks, nothing warns,
and the budget silently stops being a budget.
"""

import re
from pathlib import Path

import pytest

UNIT_DIR = Path("deploy/systemd")
SLICE_FILE = UNIT_DIR / "memecoin.slice"

_SUFFIX = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def _bytes(value: str):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMG]?)", value.strip())
    if not match:
        return None
    return int(float(match.group(1)) * _SUFFIX.get(match.group(2), 1))


def _directive(text: str, name: str):
    match = re.search(rf"^{name}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def units():
    return sorted(UNIT_DIR.glob("*.service"))


def test_the_slice_exists_and_caps_the_family():
    assert SLICE_FILE.exists()
    text = SLICE_FILE.read_text()
    high = _bytes(_directive(text, "MemoryHigh") or "")
    hard = _bytes(_directive(text, "MemoryMax") or "")
    assert high and hard, "a slice with no cap is a comment"
    assert high < hard, "MemoryHigh throttles before MemoryMax kills"


@pytest.mark.parametrize("unit", units(), ids=lambda path: path.name)
def test_every_unit_is_inside_the_budget(unit):
    """A unit outside the slice is a unit outside the budget."""
    assert "Slice=memecoin.slice" in unit.read_text(), (
        f"{unit.name} runs outside memecoin.slice and can starve the quant "
        "platform regardless of its own caps")


def test_the_family_cap_is_a_minority_of_the_box():
    """3814 MB shared with quant. Memecoin does not get most of it."""
    hard = _bytes(_directive(SLICE_FILE.read_text(), "MemoryMax") or "")
    assert hard < 0.55 * 3814 * 1024 ** 2


def test_the_slice_yields_cpu_rather_than_capping_it():
    """A hard quota leaves the box idle while the desk waits; a weight does not."""
    text = SLICE_FILE.read_text()
    assert _directive(text, "CPUWeight")
    assert int(_directive(text, "CPUWeight")) < 100, "quant wins contention"
    assert _directive(text, "CPUQuota") is None


def test_the_batch_jobs_are_capped_tighter_than_the_desk():
    """The collector is the thing that must survive; batch work is repeatable."""
    desk = _bytes(_directive(
        (UNIT_DIR / "memecoin-shadow.service").read_text(), "MemoryMax") or "")
    gauntlet = _bytes(_directive(
        (UNIT_DIR / "memecoin-gauntlet.service").read_text(), "MemoryMax") or "")
    assert desk and gauntlet
    assert gauntlet < desk


def test_the_gauntlet_refuses_to_start_on_a_tight_box():
    text = (UNIT_DIR / "memecoin-gauntlet.service").read_text()
    assert "ExecCondition" in text and "training_guard" in text


def test_the_desk_is_the_last_process_the_kernel_takes():
    """Every batch unit should die before the collector does."""
    desk = _directive((UNIT_DIR / "memecoin-shadow.service").read_text(),
                      "OOMScoreAdjust")
    assert desk is not None and int(desk) < 0
    for name in ("memecoin-shadow-trainer.service", "memecoin-gauntlet.service"):
        adjust = _directive((UNIT_DIR / name).read_text(), "OOMScoreAdjust")
        assert adjust is not None and int(adjust) > int(desk), name


# --- why the router is being asked at all ---------------------------------

def test_the_curve_route_skip_is_counted_by_cause():
    """The 2026-08-29 fix recurred on 2026-09-04 because nothing counted it.

    `catastrophic_exit_price_impact` needs a NUMERIC price impact, and the
    bonding-curve branch returns None for that -- so every one of those
    vetoes came from the router, and the native branch had been skipped
    silently every time.
    """
    from src.detection.rug_detector import RugDetector
    detector = RugDetector.__new__(RugDetector)
    detector.curve_route_skips = {"no_cached_curve_state": 301,
                                  "curve_not_tradeable": 58}
    report = detector.sell_route_report()
    assert report["status"] == "OK"
    assert report["skipped_total"] == 359
    assert "ignorance, not a property of the token" in report["detail"]


def test_a_detector_with_no_counter_yet_does_not_raise():
    """This runs inside the safety report; bookkeeping must never break it."""
    from src.detection.rug_detector import RugDetector
    detector = RugDetector.__new__(RugDetector)
    assert detector.sell_route_report()["status"] == "DATA_BLOCKED"
