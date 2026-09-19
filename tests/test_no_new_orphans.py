"""A method written and never wired must not pass review again.

Three bugs on 2026-09-19 were the same shape, and the suite was green through
all three because every test covered the METHODS and none covered the WIRING:

  four `LaunchCensus` funnel transitions, dropped by the commit that split
  `main.py`, so hard safety rejects were filed as ordinary screens;

  `PromotionLedger.demote` -- the only control that can lower the desk's
  trading authority -- so a catastrophic loss at LIVE changed nothing;

  `PriorityFeeOptimizer.record_attempt`, which left `landing_rates` empty
  forever and made the fee optimiser a lookup table of magic numbers that
  set the priority fee on every entry.

The baseline is the accepted set. It can only shrink: a new orphan fails,
and so does a stale entry, so wiring one forces its line to be deleted.
"""

from pathlib import Path

import pytest

from tools.orphan_sweep import (
    BASELINE, ORPHAN_CLASSES, load_baseline, load_classified_baseline, sweep)


@pytest.fixture(scope="module")
def orphans():
    return sweep()


def test_no_new_orphans(orphans):
    """A public method with no caller is a feature that does not run."""
    new = sorted(set(orphans) - load_baseline())
    detail = "\n".join(f"  {name}: {orphans[name][0]}" for name in new)
    assert not new, (
        f"{len(new)} public method(s) with no production caller and not in "
        f"the baseline:\n{detail}\n\n"
        "Wire it, or add it to tests/orphan_baseline.txt with a reason.")


def test_the_baseline_has_no_stale_entries(orphans):
    """So the set shrinks as things get wired, instead of drifting."""
    stale = sorted(load_baseline() - set(orphans))
    assert not stale, (
        "these are wired now; delete their lines from "
        f"tests/orphan_baseline.txt:\n  " + "\n  ".join(stale))


def test_the_three_that_were_found_today_are_not_in_it():
    """Regression guard for the specific methods, not just the shape."""
    baseline = load_baseline()
    for name in ("demote", "record_attempt", "data_blocked", "reject",
                 "awaiting_state", "decision_ready", "elapsed_us"):
        assert name not in baseline, f"{name} was wired; it must not reappear"


def test_the_sweep_sees_through_getattr_dispatch():
    """`getattr(x, "method")` is a call. The first version of this missed it.

    `is_calibrated` and `head_positives` are reached that way from
    `continuation.py`, and a sweep that cannot see it reports live code as
    dead -- which is how a guard like this loses trust and gets ignored.
    """
    assert "is_calibrated" not in sweep()
    assert "head_positives" not in sweep()


def test_the_baseline_file_explains_itself():
    text = BASELINE.read_text(encoding="utf-8")
    assert text.startswith("#"), "a bare list of names teaches nobody anything"
    assert "never wired" in text


def test_every_accepted_orphan_is_classified():
    """An unclassified entry is how a parking lot forms.

    Everything lands in it, nothing is ever asked to leave, and the list stops
    meaning anything -- which is the state this file was in at 88 entries
    before the classes existed.
    """
    rows = load_classified_baseline()
    assert rows, "the baseline is empty; that is not the same as classified"
    for name, (klass, reason) in rows.items():
        assert klass in ORPHAN_CLASSES, f"{name}: unknown class {klass!r}"
        assert reason, f"{name}: a class without a reason teaches nobody"


def test_class_a_is_debt_and_is_counted():
    """Class A means the desk is believed to have a feature it does not have.

    Not an assertion that the number is small -- it is 28 -- but that it is
    VISIBLE. A capability list nobody counts is a capability list nobody
    fixes, and every one of the three bugs that made this file necessary sat
    in exactly this state.
    """
    rows = load_classified_baseline()
    debt = sorted(name for name, (klass, _) in rows.items() if klass == "A")
    assert len(debt) <= 28, (
        f"class A grew to {len(debt)}: a capability was written and left "
        "unwired. Wire it, or justify a new class:\n  " + "\n  ".join(debt))


def test_class_b_entries_are_deletions_not_residents():
    """B means something already wired supersedes it, so it gets deleted.

    The two that were here -- `set_quote_provider` and
    `set_curve_state_provider` on RugDetector -- were setters for providers
    the constructor already takes. They looked like an injection point that
    nobody used, which is indistinguishable at a glance from an injection
    point somebody forgot to use, and that ambiguity is expensive on a class
    whose unwired provider caused 678 launches to be vetoed.
    """
    rows = load_classified_baseline()
    superseded = {name for name, (klass, _) in rows.items() if klass == "B"}
    assert not superseded, (
        "class B is a deletion waiting for a hand, not an address:\n  "
        + "\n  ".join(sorted(superseded)))
