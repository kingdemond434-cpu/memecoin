"""The diagnostic, and the inference it rests on.

The live desk reported 150,278 decisions over 8.76 days with
`net_log_growth`, `rug_loss_share` and `monster_enrichment` all NOT MEASURED.
That is not three coincidences: all three are ratios over entered positions,
and the first is None exactly when nothing has been entered. The inference
"zero entries" is load-bearing enough to get its own test.
"""

import json

import pytest

from src.research.forward_evidence import ForwardEvidence, Outcome
from tools.why_no_entries import (
    EXIT_DATA_BLOCKED, EXIT_NO_ENTRIES, EXIT_OK, main)


def _census(tmp_path, **totals):
    payload = {"totals": {"seen": 100, "screened": 100, "decided": 0,
                          "entered": 0, **totals}}
    (tmp_path / "launch_census.json").write_text(json.dumps(payload))
    return tmp_path


# --- the inference --------------------------------------------------------

def test_net_log_growth_is_unmeasured_exactly_when_nothing_was_entered():
    """The step the whole diagnosis turns on."""
    ledger = ForwardEvidence()
    for index in range(500):
        ledger.record(Outcome(token=f"t{index}", entered=False, regime="bull",
                              max_multiple=1.2))
    assert ledger.entered == 0
    assert ledger.evidence().net_log_growth is None

    ledger.record(Outcome(token="first", entered=True, regime="bull",
                          realized_pnl_usd=-1.0,
                          equity_at_decision_usd=1_000.0))
    assert ledger.evidence().net_log_growth is not None


def test_the_other_two_blockers_have_the_same_empty_denominator():
    ledger = ForwardEvidence()
    for index in range(500):
        ledger.record(Outcome(token=f"t{index}", entered=False, regime="bull",
                              max_multiple=(12.0 if index % 50 == 0 else 1.2)))
    evidence = ledger.evidence()
    # Monsters were SEEN -- the base rate exists -- and still no enrichment,
    # because enrichment is about what was bought.
    assert ledger.monsters > 0
    assert evidence.monster_enrichment is None
    assert evidence.rug_loss_share is None


def test_a_hundred_thousand_declines_do_not_move_the_gate():
    """No number of further decisions changes an absent denominator."""
    ledger = ForwardEvidence()
    for index in range(5_000):
        ledger.record(Outcome(token=f"t{index}", entered=False, regime="bull",
                              max_multiple=1.2))
    evidence = ledger.evidence()
    assert evidence.decisions == 5_000
    assert evidence.net_log_growth is None


# --- the tool -------------------------------------------------------------

def test_zero_entries_is_reported_as_the_blocker(tmp_path, capsys):
    state = _census(tmp_path, screened_by_reason={"model_not_trained": 100})
    assert main(["--state-dir", str(state)]) == EXIT_NO_ENTRIES
    printed = capsys.readouterr().out
    assert "NOTHING HAS BEEN ENTERED" in printed
    assert "model_not_trained" in printed


def test_a_desk_that_enters_is_not_flagged(tmp_path):
    state = _census(tmp_path, entered=4, decided=9)
    assert main(["--state-dir", str(state)]) == EXIT_OK


def test_a_missing_census_is_data_blocked(tmp_path):
    assert main(["--state-dir", str(tmp_path)]) == EXIT_DATA_BLOCKED


def test_an_empty_census_is_data_blocked_not_zero_percent(tmp_path):
    (tmp_path / "launch_census.json").write_text(json.dumps({"totals": {"seen": 0}}))
    assert main(["--state-dir", str(tmp_path)]) == EXIT_DATA_BLOCKED


def test_could_not_answer_is_separated_from_answered_no(tmp_path, capsys):
    """One is fixed by making something run; the other is a policy choice."""
    state = _census(tmp_path, screened_by_reason={
        "prediction_data_blocked": 80, "p_2x_below_gate": 20})
    main(["--state-dir", str(state)])
    printed = capsys.readouterr().out
    assert "could not answer" in printed and "answered no" in printed
    lines = [line for line in printed.splitlines()
             if "could not answer" in line or "answered no" in line]
    assert "80" in lines[0] and "20" in lines[1]


def test_refused_monsters_are_attributed_to_the_screen_that_refused_them(
        tmp_path, capsys):
    state = _census(tmp_path, screened_by_reason={"model_not_trained": 100},
                    monsters_by_screen={"model_not_trained": 41})
    main(["--state-dir", str(state)])
    printed = capsys.readouterr().out
    assert "10x+ launches this desk refused" in printed
    assert "41" in printed


def test_the_top_refusal_is_named_with_what_to_run(tmp_path, capsys):
    """A histogram says what is happening; this says what to do about it."""
    state = _census(tmp_path, screened_by_reason={
        "DATA_BLOCKED_prediction_model": 98, "DATA_BLOCKED_liquidity": 2})
    main(["--state-dir", str(state)])
    printed = capsys.readouterr().out
    assert "top refusal: DATA_BLOCKED_prediction_model" in printed
    assert "src.runtime.train_once" in printed


def test_every_remedy_matches_a_reason_the_desk_actually_emits():
    """A remedy keyed to a string nothing produces never fires."""
    from pathlib import Path as _Path
    from tools.why_no_entries import _REMEDY, _remedy
    desk = _Path("src/main.py").read_text().lower()
    for marker, _ in _REMEDY:
        assert marker in desk, marker
    assert _remedy("DATA_BLOCKED_prediction_model")
    assert _remedy("something_nobody_emits") == ""
