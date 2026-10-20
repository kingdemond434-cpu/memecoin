"""Every disposition the census can record must have something that records it.

`LaunchCensus` exposes seven funnel transitions. On 2026-08-28 two commits
wired them all -- the second was titled "Make launch funnel exhaustive" -- and
`e2deedc`, the commit that split main.py out of one 5,570-line file, silently
dropped four of them. `data_blocked`, `reject`, `awaiting_state` and
`decision_ready` were left with NO caller anywhere, so:

  * a launch nobody could price was filed as one the desk looked at and
    declined, which are opposite problems with opposite remedies;
  * a hard safety reject, which is a terminal DECISION, was filed as a
    pre-decision disappearance;
  * a launch in flight got no disposition at all until it finished, so any
    launch that died during enrichment was never filed.

Measured on the live desk 2026-09-04: of 8,757 launches seen, 5,739 (65.5%)
had reached no disposition at all.

Nothing failed. No test covered the wiring, only the methods, so a refactor
that dropped every call site kept a green suite. That is what this file is
for: it tests the WIRING, which is the thing that was lost.
"""

import inspect
import re
from pathlib import Path

import pytest

from src.research.launch_census import Disposition, LaunchCensus, Stage

SRC = Path("src")

#: The funnel transitions. Every one of these moves a launch from one
#: disposition to another, so a launch's fate is unexplainable without it.
FUNNEL = ("see", "screen", "data_blocked", "awaiting_state", "decision_ready",
          "decide", "reject", "enter")


def _callers(method: str) -> int:
    pattern = re.compile(rf"census\.{method}\(")
    return sum(len(pattern.findall(path.read_text(encoding="utf-8")))
               for path in SRC.rglob("*.py"))


@pytest.mark.parametrize("method", FUNNEL)
def test_every_funnel_transition_has_a_caller(method):
    """A transition nothing calls is a disposition no launch can reach."""
    assert _callers(method) >= 1, (
        f"LaunchCensus.{method}() has no caller in src/. Every launch that "
        "should reach that disposition now reaches none, and the funnel "
        "reports it as unaccounted.")


def test_the_funnel_list_covers_every_public_transition():
    """So a NEW transition cannot be added and left unwired either."""
    public = {name for name, _ in inspect.getmembers(
        LaunchCensus, predicate=inspect.isfunction)
        if not name.startswith("_")}
    # Everything that is not a transition: reads, persistence, maintenance.
    not_a_transition = {
        "report", "state", "save", "load", "resolve", "knows",
        "missed_monster_report", "mints_pending_death_classification",
    }
    assert public - not_a_transition - set(FUNNEL) == set(), (
        "a public LaunchCensus method is neither a known read nor in FUNNEL; "
        "if it is a transition, add it to FUNNEL so its wiring is tested")


# --- the dispatch itself --------------------------------------------------

def _desk_source() -> str:
    return (SRC / "main.py").read_text(encoding="utf-8")


def test_an_unpriceable_launch_is_not_filed_as_a_decline():
    """DATA_BLOCKED and screened are opposite problems."""
    source = _desk_source()
    assert 'if str(reason).upper().startswith("DATA_BLOCKED"):' in source
    assert "self.launch_census.data_blocked(token, reason)" in source


def test_a_hard_safety_veto_is_a_decision_not_a_disappearance():
    source = _desk_source()
    assert 'elif str(reason).startswith("safety_veto:"):' in source
    assert "self.launch_census.reject(token, reason)" in source


def test_a_launch_in_flight_is_filed_before_its_enrichment_finishes():
    """Otherwise one that dies mid-enrichment is never filed at all."""
    source = _desk_source()
    assert "self.launch_census.awaiting_state(token," in source
    assert "candidate_pipeline_running" in source


def test_a_decidable_launch_says_so():
    assert "self.launch_census.decision_ready(" in _desk_source()


# --- the census's own accounting -----------------------------------------

def _census():
    census = LaunchCensus()
    census.see("mint")
    return census


def test_the_three_reasons_land_in_three_different_buckets():
    blocked, vetoed, screened = _census(), _census(), _census()
    blocked.data_blocked("mint", "DATA_BLOCKED_prediction_model")
    vetoed.reject("mint", "safety_veto:sell_route_unavailable")
    screened.screen("mint", "p_2x_below_gate")

    assert blocked.report()["funnel"]["data_blocked"] == 1
    assert vetoed.report()["funnel"]["decided_reject"] == 1
    assert screened.report()["funnel"]["screened_out"] == 1
    # And none of them leaks into another's count.
    assert blocked.report()["funnel"]["screened_out"] == 0
    assert vetoed.report()["funnel"]["screened_out"] == 0
    assert screened.report()["funnel"]["data_blocked"] == 0


def test_a_filed_launch_is_never_unaccounted():
    """The 65.5% this whole file exists for."""
    for transition, args in (("screen", ("mint", "reason")),
                             ("data_blocked", ("mint", "reason")),
                             ("awaiting_state", ("mint", "reason")),
                             ("decision_ready", ("mint", "reason")),
                             ("reject", ("mint", "safety_veto:x"))):
        census = _census()
        getattr(census, transition)(*args)
        assert census.report()["funnel"]["unaccounted"] == 0, transition


def test_seeing_a_launch_already_files_it():
    """`see()` sets AWAITING_STATE, so nothing is unaccounted from birth.

    Worth pinning, because it is what makes `unaccounted` mean a genuine
    accounting fault rather than a launch that merely has not been decided
    yet -- and because a reader subtracting `screened + decided` from `seen`
    will get a large fake number, every launch in flight or data-blocked. I
    did exactly that on 2026-09-04 and reported two thirds of the funnel as
    unaccounted on arithmetic the census does not use.
    """
    funnel = _census().report()["funnel"]
    assert funnel["unaccounted"] == 0
    assert funnel["dispositions"][Disposition.AWAITING_STATE.value] == 1
    # And the honest denominator check: dispositions partition `seen`.
    assert sum(funnel["dispositions"].values()) == funnel["seen"]
