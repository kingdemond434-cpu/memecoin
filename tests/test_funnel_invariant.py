"""The observation that would have named the defect in one sentence.

The desk ran 8.76 days, took 150,278 decisions across 14,097 launches, and
entered nothing. Every dashboard was green, because each individual number
looked like a strict policy working. Nothing ever asked the one question that
separates a strict filter from a disconnected wire:

    did enough arrive upstream that downstream could not still be EXACTLY
    zero by chance?

A filter that passes 1 in 500 is a policy. A filter that has passed 0 of
100,000 is a wire.
"""

import pytest

from src.research.funnel_invariant import (
    DEFAULT_BREAK_THRESHOLD, check, check_desk, stage_counts)


def _census(seen=0, decision_ready=0, decided=0, entered=0):
    return {"funnel": {"seen": seen, "decision_ready": decision_ready,
                       "reached_a_decision": decided, "entered": entered},
            "outcomes": {}}


def _evidence(entered=0, net_log_growth=None):
    return {"entered": entered, "net_log_growth": net_log_growth}


class TestItFindsTheDefectThatActuallyHappened:
    def test_the_live_failure_is_named_at_the_right_link(self):
        """150,278 decisions, zero entries. The break is decided -> entered,
        not anywhere upstream of it."""
        verdict = check(_census(seen=14_097, decided=678), _evidence())
        assert not verdict.connected
        assert [(b.upstream, b.downstream) for b in verdict.breaks] == [
            ("decided", "entered")]

    def test_entering_and_never_closing_is_also_a_break(self):
        """Three promotion criteria are ratios over CLOSED positions, so a
        desk that enters and never closes is as stuck as one that never
        enters -- and looks healthier."""
        verdict = check(_census(seen=5_000, decided=900, entered=400),
                        _evidence(entered=0))
        assert [(b.upstream, b.downstream) for b in verdict.breaks] == [
            ("entered", "closed")]

    def test_closing_without_a_measurement_is_a_break(self):
        verdict = check(_census(seen=5_000, decided=900, entered=400),
                        _evidence(entered=300, net_log_growth=None))
        assert [(b.upstream, b.downstream) for b in verdict.breaks] == [
            ("closed", "measured")]

    def test_a_fully_connected_path_reports_connected(self):
        verdict = check(_census(seen=5_000, decided=900, entered=400),
                        _evidence(entered=300, net_log_growth=0.01))
        assert verdict.connected
        assert verdict.breaks == []


class TestStrictIsNotBroken:
    def test_a_harsh_but_working_filter_is_not_flagged(self):
        """One entry from 100,000 decisions is a policy. It passes."""
        verdict = check(_census(seen=100_000, decided=100_000, entered=1),
                        _evidence(entered=1, net_log_growth=-0.2))
        assert verdict.connected

    def test_below_the_threshold_nothing_is_concluded(self):
        """Ten launches and no entry is not evidence of anything."""
        verdict = check(_census(seen=10, decided=10), _evidence())
        assert verdict.connected

    def test_the_threshold_is_where_the_judgement_changes(self):
        under = check(_census(seen=99, decided=99), _evidence(), threshold=100)
        over = check(_census(seen=100, decided=100), _evidence(), threshold=100)
        assert under.connected
        assert not over.connected


class TestItRefusesRatherThanGuesses:
    def test_an_unexercised_path_is_blocked_not_broken(self):
        verdict = check(_census(), _evidence())
        assert verdict.status == "DATA_BLOCKED"
        assert verdict.breaks == []
        assert "nothing to conclude" in verdict.detail

    def test_a_missing_census_is_blocked(self):
        assert check(None, None).status == "DATA_BLOCKED"

    def test_a_growth_number_without_an_entry_is_not_a_measurement(self):
        """net_log_growth is None exactly when entered == 0, so a stored
        0.0 beside zero entries is a stale file, not an answer."""
        counts = stage_counts(_census(seen=1_000, decided=500),
                              {"entered": 0, "net_log_growth": 0.0})
        assert counts["measured"] == 1  # the raw report says so
        verdict = check_desk(type("D", (), {
            "launch_census": None,
            "forward_evidence": type("E", (), {"entered": 0,
                                               "net_log_growth": 0.0})()})())
        assert verdict.counts["measured"] == 0


class TestCountsAreCumulativeNotCurrent:
    def test_a_launch_past_decision_ready_still_counts_as_having_reached_it(self):
        """DECISION_READY is a disposition, so a launch that moved on is no
        longer sitting in it. Reading the live count alone would report the
        stage as empty while thousands had passed through."""
        counts = stage_counts(_census(seen=1_000, decision_ready=5,
                                      decided=700, entered=20))
        assert counts["decision_ready"] == 725
        assert counts["decided"] == 720

    def test_the_desk_reader_survives_a_desk_with_neither_subsystem(self):
        verdict = check_desk(type("D", (), {})())
        assert verdict.status == "DATA_BLOCKED"


def test_the_reporting_surface_asks_the_question():
    """It was never asked, which is the entire reason this module exists."""
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src" / "runtime"
              / "reporting.py").read_text(encoding="utf-8")
    assert "check_desk" in source
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "check_desk"]
    assert len(calls) == 1
