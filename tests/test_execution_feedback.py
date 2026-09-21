"""The fee optimiser could not learn, and the competition was a guess.

`PriorityFeeOptimizer.get_optimal_fee` sets the compute-unit price on every
entry and selects from `self.landing_rates`. Nothing ever wrote to that dict.
It was empty for the life of the desk, so every call fell through to a
hardcoded base -- and the `competition` argument that scales that base was
itself hardcoded to 0.5 at both call sites. Five magic numbers wearing an
optimiser's name.

Found by sweeping for public methods with no caller. `record_attempt` was
written to close this loop and never called.
"""

from types import SimpleNamespace

import pytest

from src.execution.jupiter_jito import PriorityFeeOptimizer
from src.runtime.execution_feedback import (
    ExecutionFeedback, fee_competition)


def _desk(outcomes=None, optimiser=None):
    desk = ExecutionFeedback()
    desk.latency = SimpleNamespace(outcomes=dict(outcomes or {}))
    desk.fee_optimizer = optimiser
    return desk


# --- the loop that was open ----------------------------------------------

def test_the_optimiser_learns_nothing_until_something_records():
    """The state the desk shipped in."""
    optimiser = PriorityFeeOptimizer()
    assert optimiser.landing_rates == {}
    # Two different values of `expected_value`, same answer as an untaught
    # optimiser: the selection has nothing to select from.
    assert optimiser.get_optimal_fee(50.0, 0.5) == 5_000


def test_recording_an_attempt_gives_it_something_to_select_from():
    desk = _desk(optimiser=PriorityFeeOptimizer())
    desk._record_fee_outcome(7_500, True, SimpleNamespace(
        elapsed_us=lambda: 42_000))
    assert desk.fee_optimizer.landing_rates == {7_500: 1.0}


def test_a_losing_bid_lowers_its_own_landing_rate():
    desk = _desk(optimiser=PriorityFeeOptimizer())
    for landed in (True, False, False, False):
        desk._record_fee_outcome(7_500, landed, None)
    assert desk.fee_optimizer.landing_rates[7_500] == pytest.approx(0.25)
    # And a rate that poor no longer qualifies as viable.
    assert desk.fee_optimizer.get_optimal_fee(50.0, 0.5) == 5_000


def test_latency_is_carried_from_the_trace_not_invented():
    desk = _desk(optimiser=PriorityFeeOptimizer())
    desk._record_fee_outcome(9_000, True,
                             SimpleNamespace(elapsed_us=lambda: 1_250_000))
    entry = desk.fee_optimizer.fee_history[-1]
    assert entry["latency"] == 1_250


def test_a_missing_trace_records_the_bid_anyway():
    """The bid's outcome is the point; the latency is a detail."""
    desk = _desk(optimiser=PriorityFeeOptimizer())
    desk._record_fee_outcome(9_000, True, None)
    assert 9_000 in desk.fee_optimizer.landing_rates


# --- competition, measured rather than asserted ---------------------------

def test_competition_is_the_share_of_submissions_that_lost():
    desk = _desk({"entered": 3, "submit_failed": 1})
    assert fee_competition(desk.latency) == pytest.approx(0.25)


def test_a_desk_losing_every_race_reports_full_competition():
    desk = _desk({"entered": 0, "submit_failed": 9})
    assert fee_competition(desk.latency) == 1.0


def test_with_nothing_submitted_it_falls_back_to_the_old_constant():
    """Stated rather than assumed, and it stops being used on attempt one."""
    assert fee_competition(_desk({}).latency) == 0.5


def test_other_latency_outcomes_do_not_count_as_races():
    """A screened launch was never submitted and never raced."""
    desk = _desk({"entered": 1, "submit_failed": 1, "screened": 500})
    assert fee_competition(desk.latency) == pytest.approx(0.5)


# --- it must never be able to break the entry path ------------------------

def test_a_broken_optimiser_does_not_take_down_an_entry():
    def explode(*_args):
        raise RuntimeError("bad optimiser")

    desk = _desk(optimiser=SimpleNamespace(record_attempt=explode))
    desk._record_fee_outcome(5_000, True, None)  # must not raise


def test_a_desk_with_no_optimiser_is_not_an_error():
    _desk(optimiser=None)._record_fee_outcome(5_000, True, None)


def test_the_call_sites_no_longer_hardcode_competition():
    """Both sites passed a literal 0.5 -- a claim about the market dressed
    as an argument."""
    from pathlib import Path
    source = Path("src/main.py").read_text(encoding="utf-8")
    calls = source.count("get_optimal_fee")
    assert calls >= 2
    assert source.count("fee_competition(getattr(self, \"latency\", None))") == calls
    assert "get_optimal_fee(trade_info[\"position_value_usd\"], 0.5)" not in source
