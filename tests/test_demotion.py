"""The only control that lowers trading authority, and it had no caller.

`PromotionLedger.demote` was written and tested and never wired. A
catastrophic failure was COUNTED -- which blocks the next promotion -- and
left the authority the desk already had untouched. A desk at LIVE that lost
half its book to a rug stayed at LIVE.

That was survivable only because the desk had never entered a position. It
stops being survivable the moment it does.
"""

import time
from types import SimpleNamespace

import pytest

from src.research.forward_evidence import ForwardEvidence
from src.research.promotion_gate import PromotionLedger, Stage
from src.runtime.regime import RegimeAndEvidence


def _desk(tmp_path, stage, *, equity=1000.0):
    desk = RegimeAndEvidence()
    ledger = PromotionLedger(tmp_path / "promotion.jsonl")
    ledger._write_stage(stage, "test")
    desk.promotion_ledger = ledger
    desk.forward_evidence = ForwardEvidence(stage=stage)
    desk.wallet_equity_usd = equity
    desk.dry_run = False
    desk._evidence_saved_at = time.time()
    desk._census_saved_at = time.time()
    desk.launch_census = SimpleNamespace(save=lambda: None)
    desk.calibration = SimpleNamespace(save=lambda: None)
    return desk, ledger


def _catastrophe(desk, token="rugged"):
    desk._record_forward_evidence({
        "token": token, "entered": True, "attempted": True, "rugged": True,
        "realized_pnl_usd": -900.0, "regime": "bear"})


# --- the automatic path ---------------------------------------------------

def test_a_catastrophic_loss_lowers_live_authority(tmp_path):
    desk, ledger = _desk(tmp_path, Stage.CANARY)
    _catastrophe(desk)
    assert ledger.current_stage() is Stage.FORWARD_SHADOW


def test_the_ledger_and_the_evidence_agree_on_the_new_stage(tmp_path):
    """Otherwise the accumulator keeps clearing a bar it no longer sits at."""
    desk, ledger = _desk(tmp_path, Stage.LIVE)
    _catastrophe(desk)
    assert desk.forward_evidence.stage is ledger.current_stage()


def test_a_paper_loss_at_a_shadow_stage_changes_nothing(tmp_path):
    """The shadow rungs are where losses are free; reacting to one is noise."""
    desk, ledger = _desk(tmp_path, Stage.FORWARD_SHADOW)
    _catastrophe(desk)
    assert ledger.current_stage() is Stage.FORWARD_SHADOW


def test_an_ordinary_loss_is_not_a_catastrophe(tmp_path):
    desk, ledger = _desk(tmp_path, Stage.CANARY)
    desk._record_forward_evidence({
        "token": "t", "entered": True, "attempted": True, "rugged": True,
        "realized_pnl_usd": -50.0, "regime": "bear"})
    assert ledger.current_stage() is Stage.CANARY


def test_one_event_costs_one_stage_not_three(tmp_path):
    """A single rug can close several positions within seconds."""
    desk, ledger = _desk(tmp_path, Stage.LIVE)
    for index in range(3):
        _catastrophe(desk, f"same_rug_{index}")
    assert ledger.current_stage() is Stage.CANARY


def test_a_later_catastrophe_demotes_again(tmp_path):
    """The cooldown is not a softener on repeated failures."""
    desk, ledger = _desk(tmp_path, Stage.LIVE)
    _catastrophe(desk, "first")
    desk._last_demotion_at = time.time() - desk.DEMOTION_COOLDOWN_S - 1
    _catastrophe(desk, "second")
    assert ledger.current_stage() is Stage.FORWARD_SHADOW


def test_a_broken_ledger_never_takes_the_desk_down(tmp_path):
    desk, _ = _desk(tmp_path, Stage.CANARY)
    desk.promotion_ledger = SimpleNamespace(
        authorises_live_capital=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert desk._demote_on_catastrophe("t") is False


def test_a_desk_with_no_ledger_is_not_an_error(tmp_path):
    desk, _ = _desk(tmp_path, Stage.CANARY)
    desk.promotion_ledger = None
    assert desk._demote_on_catastrophe("t") is False


# --- the operator path ----------------------------------------------------

def test_the_operator_tool_lowers_one_stage(tmp_path):
    from tools.demote_desk import EXIT_OK, main
    ledger = PromotionLedger(tmp_path / "promotion.jsonl")
    ledger._write_stage(Stage.LIVE, "test")
    assert main(["--state-dir", str(tmp_path), "--yes",
                 "--reason", "exit latency doubled since Tuesday"]) == EXIT_OK
    assert ledger.current_stage() is Stage.CANARY


def test_the_operator_tool_demands_a_real_reason(tmp_path):
    """It is the permanent record of why authority was taken away."""
    from tools.demote_desk import EXIT_REFUSED, main
    ledger = PromotionLedger(tmp_path / "promotion.jsonl")
    ledger._write_stage(Stage.LIVE, "test")
    assert main(["--state-dir", str(tmp_path), "--yes", "--reason", "bad"]) == (
        EXIT_REFUSED)
    assert ledger.current_stage() is Stage.LIVE


def test_the_operator_tool_will_not_go_below_the_bottom(tmp_path):
    from tools.demote_desk import EXIT_REFUSED, main
    assert main(["--state-dir", str(tmp_path), "--yes",
                 "--reason", "nothing left to take away here"]) == EXIT_REFUSED


def test_the_reason_is_written_into_the_permanent_record(tmp_path):
    from tools.demote_desk import main
    ledger = PromotionLedger(tmp_path / "promotion.jsonl")
    ledger._write_stage(Stage.LIVE, "test")
    main(["--state-dir", str(tmp_path), "--yes",
          "--reason", "exit latency doubled since Tuesday"])
    assert "exit latency doubled" in ledger._stage_path.read_text()
