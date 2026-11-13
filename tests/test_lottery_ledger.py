"""Counterfactual lottery tickets on launches the core desk declined."""
import math

from src.research.lottery_ledger import LotteryLedger, LotteryPolicy

DAY = 86400.0


def _ledger(**kw):
    return LotteryLedger(LotteryPolicy(**kw))


def _file(ledger, token="m", at=0.0, entry=2.0, p10=0.2, reason="p_2x_below_gate",
          clear=True, equity=10_000.0):
    return ledger.consider(token, at=at, entry_price=entry, p_10x=p10, reason=reason,
                           safety_clear=clear, equity_usd=equity)


def test_only_safe_declines_with_a_priced_tail_get_a_ticket():
    ledger = _ledger()
    assert _file(ledger, "ok")
    assert not _file(ledger, "veto", reason="safety_veto:sell_route_unavailable")
    assert not _file(ledger, "blocked", reason="DATA_BLOCKED_prediction_model")
    assert not _file(ledger, "unsafe", clear=False)
    assert not _file(ledger, "thin", p10=0.05)
    assert not _file(ledger, "nomark", entry=None)
    assert not _file(ledger, "ok")
    assert ledger.report()["skipped"] == {
        "hard_reject": 2, "safety_not_clear": 1, "tail_too_thin": 1,
        "no_entry_mark": 1, "already_ticketed": 1}


def test_harvest_is_booked_at_the_threshold_and_the_runner_at_its_last_mark():
    ledger = _ledger(round_trip_cost=0.0)
    _file(ledger, entry=2.0)                       # ticket cost 10 USD
    ledger.mark("m", 2.0 * 50, 60)                 # 50x peak
    ledger.mark("m", 2.0 * 20, 120)                # fades to 20x
    ledger.expire(8 * DAY)
    report = ledger.report()
    # half sold at 3x (15) + runner half at 20x (100)
    assert math.isclose(report["closed_proceeds_usd"], 115.0)
    assert report["peak_at_least"] == {"3x": 1, "10x": 1, "100x": 0}
    assert report["closed_by_reason"] == {"horizon": 1}


def test_a_rug_zeroes_the_runner_but_keeps_a_harvest_already_taken():
    ledger = _ledger(round_trip_cost=0.0)
    _file(ledger, "a")
    _file(ledger, "b")
    ledger.mark("a", 2.0 * 4, 30)
    ledger.rugged("a", 60)
    ledger.rugged("b", 60)
    report = ledger.report()
    assert math.isclose(report["closed_proceeds_usd"], 15.0)
    assert math.isclose(report["closed_net_usd"], 15.0 - 20.0)


def test_costs_daily_cap_and_state_round_trip():
    ledger = _ledger(max_tickets_per_day=2, round_trip_cost=0.04)
    assert _file(ledger, "a") and _file(ledger, "b")
    assert not _file(ledger, "c")
    assert _file(ledger, "d", at=DAY + 1)
    ledger.rugged("a", 10)
    report = ledger.report()
    assert math.isclose(report["closed_net_usd"], -10.0 - 0.4)
    restored = _ledger(max_tickets_per_day=2)
    restored.load(ledger.state())
    assert set(restored.open) == {"b", "d"} and restored.report()["closed_tickets"] == 1


def test_a_mark_after_the_horizon_closes_at_the_previous_mark():
    ledger = _ledger(round_trip_cost=0.0)
    _file(ledger)
    ledger.mark("m", 2.0 * 2, 60)
    ledger.mark("m", 2.0 * 90, 8 * DAY)            # too late to count
    assert math.isclose(ledger.report()["closed_proceeds_usd"], 20.0)


def test_the_runtime_files_a_declined_launch_from_its_live_mark():
    from types import SimpleNamespace
    from src.runtime import research_runtime

    owner = next(value for value in vars(research_runtime).values()
                 if isinstance(value, type) and hasattr(value, "_lottery_consider_declined"))
    desk = owner.__new__(owner)
    desk.global_config = {}
    desk.champion_challenger = None
    desk.wallet_equity_usd = 5_000.0
    desk._latest_stream_mark = {"mint": {"multiple": 1.7}}
    desk._curve_entry_price = {"mint": 2e-8}
    desk._lottery_consider_declined("mint", SimpleNamespace(p_10x=0.3), "p_2x_below_gate",
                                    {"clear": True})
    ticket = desk.lottery_ledger.open["mint"]
    assert math.isclose(ticket.entry_price, 3.4e-8) and ticket.cost_usd == 5.0
    # The desk drops (or restarts without) its in-memory anchor; the ticket
    # does not depend on it, so a later absolute price reads the same multiple.
    desk._curve_entry_price.clear()
    desk.lottery_ledger.mark("mint", 3 * 3.4e-8, 10)
    assert math.isclose(ticket.last, 3.0) and ticket.harvested


def test_equity_level_growth_is_the_real_size_log_and_totals_survive_restart():
    ledger = _ledger(round_trip_cost=0.0)
    _file(ledger, "a"); _file(ledger, "b")
    ledger.rugged("a", 10)                          # money multiple 0
    ledger.mark("b", 2.0 * 5, 20); ledger.expire(8 * DAY)   # 0.5*3 + 0.5*5 = 4x
    report = ledger.report()
    expected = (math.log(1 - 0.001) + math.log(1 + 0.001 * 3)) / 2
    assert math.isclose(report["mean_log_equity_growth_per_ticket"], expected)
    restored = _ledger(round_trip_cost=0.0)
    state = ledger.state(); state["closed"] = []    # inspection list trimmed away
    restored.load(state)
    assert restored.report()["closed_tickets"] == 2
    assert math.isclose(restored.report()["closed_proceeds_usd"], 40.0)


def test_a_migrated_runner_is_counted_as_unmarked():
    ledger = _ledger()
    _file(ledger)
    ledger.migrated("m")
    ledger.expire(8 * DAY)
    assert ledger.report()["unmarked_after_migration"] == 1
