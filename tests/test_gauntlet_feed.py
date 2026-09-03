"""The feed that makes the gauntlet a measurement instead of an empty table.

Two failure modes get most of the tests. The first is a feed that manufactures
evidence -- reading a missing feature as False, a missing price as zero, a
missing regime as a regime -- because every one of those produces a fuller
scoreboard and a wronger one. The second is the specific mistake of comparing
mechanisms that decide at different instants on an absolute clock, which kills
every late-deciding rule for being late.
"""

import gzip
import json
import time
from pathlib import Path

import pytest

from src.research.forward_evidence import ForwardEvidence, Outcome
from src.research.gauntlet_feed import (
    CONTROL, DEFAULT_MECHANISMS, EpisodeView, GauntletFeed, Mechanism,
    SNAPSHOT_KEY_BY_OFFSET, iter_episodes, regime_of)
from src.research.promotion_gate import DEFAULT_CRITERIA, Stage, evaluate


def marks(count=40, start=1_700_000_000.0, step=0.5, peak=6.0, depth=5.0):
    """A launch that rises to `peak` and falls back, priced all the way."""
    out = []
    for index in range(count):
        fraction = index / max(count - 1, 1)
        multiple = 1.0 + (peak - 1.0) * (1.0 - abs(2.0 * fraction - 1.0))
        out.append({"timestamp": start + index * step,
                    "price_multiple": round(multiple, 6),
                    "executable_sol": depth, "feasible": True})
    return out


def episode(token="tok", *, created_at=1_700_000_000.0, t0=None, t10s=None,
            observations=None, status="OK", sol_change=5.0, launch_rate=10.0):
    market = {}
    if sol_change is not None and launch_rate is not None:
        market = {"status": "OK", "sol_change_24h": sol_change,
                  "meme_launch_rate_1h": launch_rate}
    snapshots = {"t0": {"market_features": market, **(t0 or {})}}
    if t10s is not None:
        snapshots["t10s"] = dict(t10s)
    return {
        "token": token, "chain": "solana", "created_at": created_at,
        "deployer": "dep", "factory": "pump", "pair": "", "base_token": "",
        "snapshots": snapshots,
        "market_observations": (marks(start=created_at)
                                if observations is None else observations),
        "final_outcome": {"status": status, "max_multiple": 6.0},
    }


# --- missing is missing ---------------------------------------------------

def test_an_unmeasured_feature_produces_no_observation_not_a_negative():
    """The deployer block is absent, so the rule has nothing to say."""
    feed = GauntletFeed(mechanisms=[DEFAULT_MECHANISMS[0]])
    feed.observations([episode()])
    bucket = feed.counters.by_mechanism["clean_deployer_history"]
    assert bucket["unmeasured"] == 1
    assert bucket["matched"] == 0 and bucket["unmatched"] == 0


def test_a_deployer_we_have_never_seen_is_not_a_dirty_deployer():
    """`has_profile: False` is a first-time deployer, not a failed clean one."""
    row = episode(t0={"deployer_features": {"has_profile": False}})
    feed = GauntletFeed()
    feed.observations([row])
    assert feed.counters.by_mechanism["clean_deployer_history"]["unmatched"] == 1
    assert feed.counters.by_mechanism["first_time_deployer"]["matched"] == 1


def test_a_profile_without_a_rug_rate_is_unmeasured_rather_than_clean():
    row = episode(t0={"deployer_features": {"has_profile": True,
                                            "prior_launches": 4}})
    feed = GauntletFeed()
    feed.observations([row])
    assert feed.counters.by_mechanism["clean_deployer_history"]["unmeasured"] == 1


def test_an_unpriceable_latency_is_absent_not_zero():
    """One mark only, at T0. Every later entry priced against nothing."""
    row = episode(observations=[{"timestamp": 1_700_000_000.0,
                                 "price_multiple": 1.0,
                                 "executable_sol": 1.0}])
    feed = GauntletFeed(mechanisms=[])
    built = feed.observations([row])
    if built:
        control = built[0]
        assert any(value is None
                   for value in control.net_return_by_latency.values())
        assert 0.0 not in {key for key, value
                           in control.net_return_by_latency.items()
                           if value == 0.0}


def test_a_launch_with_no_priced_marks_is_dropped_and_counted():
    row = episode(observations=[])
    feed = GauntletFeed()
    assert feed.observations([row]) == []
    assert feed.coverage()["dropped_no_lifecycle"] == 1


# --- regimes --------------------------------------------------------------

def test_an_unmeasured_market_is_not_a_regime():
    row = episode(sol_change=None, launch_rate=None)
    feed = GauntletFeed()
    assert feed.observations([row]) == []
    coverage = feed.coverage()
    assert coverage["dropped_no_measured_regime"] == 1
    assert coverage["regimes"] == {}


def test_the_unknown_bucket_is_available_but_never_the_default():
    row = episode(sol_change=None, launch_rate=None)
    strict = GauntletFeed()
    assert strict.require_measured_regime is True
    loose = GauntletFeed(require_measured_regime=False)
    assert loose.observations([row])
    assert "unknown" in loose.coverage()["regimes"]


def test_regime_has_two_axes_so_one_market_is_not_three_regimes():
    up_quiet = regime_of(EpisodeView(episode(sol_change=9.0, launch_rate=5.0)))
    up_busy = regime_of(EpisodeView(episode(sol_change=9.0, launch_rate=50.0)))
    down_quiet = regime_of(EpisodeView(episode(sol_change=-9.0, launch_rate=5.0)))
    assert up_quiet == "up/quiet"
    assert len({up_quiet, up_busy, down_quiet}) == 3


def test_a_flat_market_is_its_own_bucket_rather_than_rounding_to_a_trend():
    assert regime_of(EpisodeView(episode(sol_change=0.5,
                                         launch_rate=5.0))) == "flat/quiet"


# --- latency is lateness --------------------------------------------------

def test_latency_is_measured_from_the_decision_not_from_the_launch():
    """A rule reading the ten-second snapshot is not judged as a 100ms sniper.

    Keyed on the clock, its earliest column would be 10s, the gauntlet's
    reference latency is 1s, and it would be killed for being late by a test
    that was asking a different question.
    """
    row = episode(t10s={"wallet_features": {"smart_buyer_count": 3}})
    feed = GauntletFeed(mechanisms=[
        Mechanism("smart_money_first_10s",
                  DEFAULT_MECHANISMS[3].rule, 10.0, "wallet")])
    built = [item for item in feed.observations([row])
             if item.mechanism == "smart_money_first_10s"]
    assert built, "the mechanism matched and should have been priced"
    keys = set(built[0].net_return_by_latency)
    assert 0.0 in keys, "its own decision instant is latency zero"
    assert max(keys) <= 30.0


def test_two_mechanisms_at_different_instants_share_latency_columns():
    row = episode(t0={"deployer_features": {"has_profile": False}},
                  t10s={"wallet_features": {"smart_buyer_count": 1}})
    feed = GauntletFeed()
    built = {item.mechanism: item for item in feed.observations([row])}
    early = set(built["first_time_deployer"].net_return_by_latency)
    late = set(built["smart_money_first_10s"].net_return_by_latency)
    assert early == late


def test_a_later_decision_prices_a_later_entry():
    """Same launch, two instants, different fills -- the prices really move."""
    row = episode(t0={"deployer_features": {"has_profile": False}},
                  t10s={"wallet_features": {"smart_buyer_count": 1}})
    feed = GauntletFeed()
    built = {item.mechanism: item for item in feed.observations([row])}
    assert (built["first_time_deployer"].net_return_by_latency[0.0]
            != built["smart_money_first_10s"].net_return_by_latency[0.0])


# --- the control ----------------------------------------------------------

def test_the_control_arm_cannot_be_configured_away():
    feed = GauntletFeed(mechanisms=[])
    assert feed.mechanisms == (CONTROL,)


def test_passing_the_control_explicitly_does_not_duplicate_it():
    feed = GauntletFeed(mechanisms=[CONTROL, DEFAULT_MECHANISMS[0]])
    names = [mechanism.name for mechanism in feed.mechanisms]
    assert names.count(CONTROL.name) == 1


def test_the_control_selects_every_launch_it_can_price():
    feed = GauntletFeed(mechanisms=[])
    built = feed.observations([episode("a"), episode("b")])
    assert len(built) == 2
    assert all(item.is_control for item in built)


# --- robustness -----------------------------------------------------------

def test_a_rule_that_raises_does_not_take_the_run_down():
    def explode(view):
        raise RuntimeError("bad rule")

    feed = GauntletFeed(mechanisms=[Mechanism("boom", explode, 0.0)])
    built = feed.observations([episode()])
    assert [item.mechanism for item in built] == [CONTROL.name]
    assert feed.counters.by_mechanism["boom"]["unmeasured"] == 1


def test_the_cost_the_replay_charged_is_the_cost_the_gauntlet_re_charges():
    feed = GauntletFeed(mechanisms=[])
    built = feed.observations([episode()])
    assert built[0].cost_fraction == feed.round_trip_cost


def test_snapshot_offsets_match_the_dataset_builder():
    """This map is a copy, so it gets a test that it is still a true one."""
    builder = pytest.importorskip("src.research.dataset_builder")
    mirrored = {float(offset): timepoint.value
                for timepoint, offset in builder.SNAPSHOT_OFFSETS_S.items()
                if offset >= 0}
    assert mirrored == SNAPSHOT_KEY_BY_OFFSET


# --- loading --------------------------------------------------------------

def _write(storage, name, payload):
    directory = storage / name[:2]
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"{name}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def test_unresolved_launches_are_skipped_rather_than_scored_as_losses(tmp_path):
    _write(tmp_path, "aa1", episode("aa1"))
    _write(tmp_path, "bb2", episode("bb2", status="PENDING"))
    tokens = [item["token"] for item in iter_episodes(tmp_path)]
    assert tokens == ["aa1"]


def test_an_unreadable_file_does_not_end_the_scan(tmp_path):
    _write(tmp_path, "aa1", episode("aa1"))
    (tmp_path / "zz").mkdir(parents=True, exist_ok=True)
    (tmp_path / "zz" / "zz9.json.gz").write_bytes(b"not gzip at all")
    assert [item["token"] for item in iter_episodes(tmp_path)] == ["aa1"]


def test_the_limit_counts_resolved_episodes_not_files(tmp_path):
    _write(tmp_path, "aa1", episode("aa1", status="PENDING"))
    _write(tmp_path, "bb2", episode("bb2"))
    _write(tmp_path, "cc3", episode("cc3"))
    assert len(list(iter_episodes(tmp_path, limit=2))) == 2


# --- the report -----------------------------------------------------------

def test_an_empty_corpus_reports_no_measurement_rather_than_no_edge():
    feed = GauntletFeed()
    report = feed.run([])
    assert report["mechanisms"] == 0
    assert report["survivors"] == 0
    # And the ledger refuses it, so the gate stays blocked on "not measured"
    # rather than being told the gauntlet found nothing.
    assert ForwardEvidence().record_gauntlet(report) is None


def test_the_coverage_block_says_why_launches_were_dropped():
    feed = GauntletFeed()
    feed.run([episode("a", sol_change=None, launch_rate=None),
              episode("b", observations=[])])
    coverage = feed.coverage()
    assert coverage["episodes"] == 2
    assert coverage["dropped_no_lifecycle"] == 1
    assert coverage["dropped_no_measured_regime"] == 1


def test_a_real_corpus_produces_rows_for_the_mechanisms_it_could_read():
    rows = [episode(f"t{index}",
                    created_at=1_700_000_000.0 + index * 3600,
                    t0={"deployer_features": {"has_profile": True,
                                              "prior_launches": 3,
                                              "rug_rate": 0.1}},
                    sol_change=(9.0 if index % 3 == 0 else
                                -9.0 if index % 3 == 1 else 0.0),
                    launch_rate=(5.0 if index % 2 else 50.0))
            for index in range(40)]
    report = GauntletFeed().run(rows)
    names = {row["mechanism"] for row in report["rows"]}
    assert CONTROL.name in names
    assert "clean_deployer_history" in names
    # Nothing is asserted about the verdicts: forty synthetic launches is far
    # under the gauntlet's minimum, and a test that pinned a verdict here
    # would be pinning the sample size rather than the finding.


# --- the other half: the ledger the gate reads ----------------------------

def _traded(ledger, count, *, pnl=-3.0, equity=1000.0, win_every=20):
    for index in range(count):
        ledger.record(Outcome(
            token=f"t{index}", entered=True, regime="up/quiet",
            realized_pnl_usd=(90.0 if index % win_every == 0 else pnl),
            equity_at_decision_usd=equity, real_fill=True,
            execution_attempted=True, execution_succeeded=True))


def test_a_short_run_reports_no_lower_bound_rather_than_a_flattering_one():
    ledger = ForwardEvidence()
    _traded(ledger, ForwardEvidence.MIN_BOOTSTRAP_SAMPLE - 1)
    assert ledger.lower_bound() is None
    assert ledger.evidence().net_log_growth_lower_bound is None


def test_the_lower_bound_appears_once_there_is_enough_to_resample():
    ledger = ForwardEvidence()
    _traded(ledger, ForwardEvidence.MIN_BOOTSTRAP_SAMPLE)
    bound = ledger.lower_bound()
    assert bound is not None
    # It is a LOWER bound, so it sits under the realised mean.
    mean = ledger.net_log_growth / ledger.entered
    assert bound < mean


def test_a_ruin_stays_in_the_sample_at_the_floor():
    """Deleting the worst trade the desk ever had is not a lower bound."""
    ledger = ForwardEvidence()
    _traded(ledger, ForwardEvidence.MIN_BOOTSTRAP_SAMPLE)
    clean = ledger.lower_bound()
    ledger.record(Outcome(token="ruin", entered=True, regime="up/quiet",
                          realized_pnl_usd=-2_000.0,
                          equity_at_decision_usd=1_000.0))
    assert ledger.catastrophic_failures == 1
    assert ledger.lower_bound() < clean


def test_an_unrun_gauntlet_is_unmeasured_rather_than_zero():
    assert ForwardEvidence().gauntlet_survivors() is None


def test_a_stale_gauntlet_stops_counting():
    ledger = ForwardEvidence()
    ledger.record_gauntlet({"mechanisms": 6, "survivors": 2},
                           at=time.time() - ForwardEvidence.GAUNTLET_MAX_AGE_S - 60)
    assert ledger.gauntlet_survivors() is None


def test_a_fresh_gauntlet_counts():
    ledger = ForwardEvidence()
    assert ledger.record_gauntlet({"mechanisms": 6, "survivors": 2}) == 2
    assert ledger.evidence().gauntlet_survivors == 2


def test_a_bare_count_is_accepted_as_well_as_a_report():
    ledger = ForwardEvidence()
    assert ledger.record_gauntlet(1) == 1
    assert ledger.gauntlet_survivors() == 1


def test_a_run_that_scored_nothing_is_refused():
    ledger = ForwardEvidence()
    assert ledger.record_gauntlet({"mechanisms": 0, "survivors": 0}) is None
    assert ledger.gauntlet_survivors() is None


def test_zero_survivors_from_a_real_run_is_recorded_as_zero():
    """Measured-and-failed is a different state from never-measured."""
    ledger = ForwardEvidence()
    assert ledger.record_gauntlet({"mechanisms": 6, "survivors": 0}) == 0
    assert ledger.gauntlet_survivors() == 0


def test_the_sample_and_the_gauntlet_stamp_survive_a_restart(tmp_path):
    path = tmp_path / "forward_evidence.json"
    ledger = ForwardEvidence(path)
    _traded(ledger, ForwardEvidence.MIN_BOOTSTRAP_SAMPLE)
    ledger.record_gauntlet({"mechanisms": 4, "survivors": 1})
    assert ledger.save()
    reborn = ForwardEvidence(path)
    assert reborn.lower_bound() == ledger.lower_bound()
    assert reborn.gauntlet_survivors() == 1


def test_a_restart_does_not_refresh_a_stale_gauntlet(tmp_path):
    """Otherwise restarting the desk would be a way to re-pass the gate."""
    path = tmp_path / "forward_evidence.json"
    ledger = ForwardEvidence(path)
    ledger.record_gauntlet(
        {"mechanisms": 4, "survivors": 1},
        at=time.time() - ForwardEvidence.GAUNTLET_MAX_AGE_S - 60)
    ledger.save()
    assert ForwardEvidence(path).gauntlet_survivors() is None


def test_the_canary_rung_is_reachable_once_both_halves_are_fed():
    """The bug this whole module exists for: two fields nobody could fill.

    Before the feed, `net_log_growth_lower_bound` and `gauntlet_survivors`
    had no producer anywhere, so CANARY and LIVE failed on "not measured" for
    the life of the desk however well it traded.
    """
    ledger = ForwardEvidence(stage=Stage.FORWARD_SHADOW)
    _traded(ledger, ForwardEvidence.MIN_BOOTSTRAP_SAMPLE)
    ledger.record_gauntlet({"mechanisms": 6, "survivors": 2})
    evidence = ledger.evidence()
    assert evidence.net_log_growth_lower_bound is not None
    assert evidence.gauntlet_survivors == 2
    # Still short of CANARY on the counting requirements -- but no longer on
    # the two that could never be measured at all.
    verdict = evaluate(DEFAULT_CRITERIA[Stage.FORWARD_SHADOW], evidence)
    assert "net_log_growth_lower_bound" not in verdict.unmeasured
    assert "gauntlet_survivors" not in verdict.unmeasured


# --- the sidecar, and the two-writer bug it exists to avoid ---------------

def test_the_verdict_is_not_written_into_the_desks_own_ledger(tmp_path):
    """The desk rewrites forward_evidence.json whole on every save.

    A second process recording into it does not merge; whichever saved last
    wins, and the loser is either this verdict or the desk's entire decision
    count.
    """
    evidence_path = tmp_path / "forward_evidence.json"
    verdict_path = tmp_path / "gauntlet.json"
    desk = ForwardEvidence(evidence_path)
    _traded(desk, 5)
    desk.save()

    assert ForwardEvidence.write_gauntlet(
        verdict_path, {"mechanisms": 4, "survivors": 1})
    # The desk saves again, as it does every minute.
    desk.save()
    # And the verdict is still there, because it never lived in that file.
    reborn = ForwardEvidence(evidence_path)
    assert reborn.load_gauntlet(verdict_path) == 1
    assert reborn.decisions == 5


def test_a_verdict_with_no_timestamp_is_refused():
    """An unageable verdict is one that never goes stale."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gauntlet.json"
        path.write_text(json.dumps({"mechanisms": 4, "survivors": 3}))
        assert ForwardEvidence().load_gauntlet(path) is None


def test_a_missing_verdict_file_is_not_an_error(tmp_path):
    assert ForwardEvidence().load_gauntlet(tmp_path / "absent.json") is None


def test_a_corrupt_verdict_file_does_not_raise(tmp_path):
    path = tmp_path / "gauntlet.json"
    path.write_text("{not json")
    assert ForwardEvidence().load_gauntlet(path) is None


def test_the_verdicts_age_survives_the_round_trip(tmp_path):
    path = tmp_path / "gauntlet.json"
    stamped = time.time() - 3 * 86_400.0
    ForwardEvidence.write_gauntlet(
        path, {"mechanisms": 4, "survivors": 1}, at=stamped)
    ledger = ForwardEvidence()
    assert ledger.load_gauntlet(path) == 1
    assert 2.9 < ledger.gauntlet_age_s() / 86_400.0 < 3.1


def test_the_expiry_leaves_slack_over_the_weekly_timer():
    """Equal to the interval would retire the verdict as it is replaced."""
    week = 7 * 86_400.0
    assert ForwardEvidence.GAUNTLET_MAX_AGE_S > week
    unit = Path("deploy/systemd/memecoin-gauntlet.timer").read_text()
    assert "OnCalendar=Sun" in unit
