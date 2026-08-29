"""24/7 coverage added 2026-08-29: the two failure classes that already hit
this desk with zero automated detection.

A cross-project OOM event disabled the desk service and all five timers, and
a corrupted GITHUB_TOKEN 401'd silently for as long as nobody read the raw
value. Both were found by a human reading logs by hand. These tests pin the
coverage that makes that unnecessary next time.
"""

import unittest

from ops.watchdog import FLEET_UNITS, Policy, decide, observed_faults
from tools.credential_doctor import check_all, parse_env_file


def _healthy_readiness():
    return {
        "runtime_tasks": {"status": "OK", "failed": []},
        "source_mesh": {"sources": 2, "producers": 2, "streaming": True},
        "yellowstone": {"status": "STREAMING"},
        "rpc_program_stream": {"status": "RPC_WS"},
        "memory": {"band": "calm"},
        "stream_events": {"total": 10, "token_created": 1},
        "pump_decoder": {"status": "OK"},
        "event_loop": {}, "data_miners": {"status": "OK"},
        "prediction": "OK", "rug_hazard": {"model_trained": True},
        "credentials": {"absent": []},
    }


def _call(readiness=None, state=None, now=10_000.0, **overrides):
    values = dict(
        service_active=True, service_enabled=True,
        readiness=readiness or _healthy_readiness(), readiness_age=10.0,
        state=state if state is not None else {}, now=now,
        policy=Policy(), trainer_active=False, training_age=10.0)
    values.update(overrides)
    return decide(**values)


class DisabledFleetUnitsAreReenabledImmediately(unittest.TestCase):
    """Exactly 2026-08-29's incident: an external event disabled every
    memecoin timer, and nothing came back until a human noticed."""

    def test_a_disabled_unit_is_queued_for_reenable(self):
        plan = _call(disabled_units=["memecoin-shadow-trainer.timer"])
        self.assertIn("memecoin-shadow-trainer.timer", plan.reenable_units)

    def test_reenable_is_not_debounced_like_a_restart(self):
        """No cooldown, no two-observation wait: enabling an already-enabled
        unit is a no-op, so there is no flapping risk to guard against, and
        the whole point is closing this gap faster than a restart would."""
        plan = _call(disabled_units=["memecoin-watchdog.timer"], state={})
        self.assertIn("memecoin-watchdog.timer", plan.reenable_units)

    def test_nothing_disabled_means_nothing_queued(self):
        plan = _call(disabled_units=[])
        self.assertEqual(plan.reenable_units, [])

    def test_the_fleet_list_covers_every_unit_this_session_installed(self):
        expected = {
            "memecoin-shadow.service", "memecoin-watchdog.timer",
            "memecoin-health.timer", "memecoin-shadow-trainer.timer",
            "memecoin-feed-doctor.timer", "memecoin-audit-pack.timer",
            "memecoin-backfill.timer", "memecoin-credential-doctor.timer",
        }
        self.assertEqual(set(FLEET_UNITS), expected)


class LowMemoryIsAnnouncedBeforeTrainingStaleFires(unittest.TestCase):
    """Six hourly training skips went unremarked for hours on 2026-08-28
    because the only existing signal (training_stale_seconds) waits 26h."""

    def test_a_brief_dip_does_not_alert(self):
        state = {}
        plan = _call(available_mib=200.0, state=state, now=10_000.0)
        self.assertNotIn("training_guard_memory_starved", plan.alerts)

    def test_a_sustained_dip_alerts_before_the_26h_stale_threshold(self):
        state = {}
        _call(available_mib=200.0, state=state, now=10_000.0)
        plan = _call(available_mib=200.0, state=state, now=10_000.0 + 10_801)
        self.assertIn("training_guard_memory_starved", plan.alerts)

    def test_recovering_above_the_floor_resets_the_clock(self):
        state = {}
        _call(available_mib=200.0, state=state, now=10_000.0)
        _call(available_mib=900.0, state=state, now=15_000.0)  # recovers
        plan = _call(available_mib=200.0, state=state, now=15_100.0)
        self.assertNotIn("training_guard_memory_starved", plan.alerts)

    def test_plenty_of_memory_never_alerts(self):
        plan = _call(available_mib=2000.0)
        self.assertNotIn("training_guard_memory_starved", plan.alerts)


class DiskAndBackfillStalenessAreWatched(unittest.TestCase):
    def test_disk_past_the_floor_alerts(self):
        plan = _call(disk_used_fraction=0.95)
        self.assertIn("disk_nearly_full", plan.alerts)

    def test_disk_below_the_floor_is_quiet(self):
        plan = _call(disk_used_fraction=0.50)
        self.assertNotIn("disk_nearly_full", plan.alerts)

    def test_a_stale_backfill_checkpoint_alerts(self):
        plan = _call(backfill_checkpoint_age=70_000.0)
        self.assertIn("backfill_stalled", plan.alerts)

    def test_a_fresh_checkpoint_is_quiet(self):
        plan = _call(backfill_checkpoint_age=3_600.0)
        self.assertNotIn("backfill_stalled", plan.alerts)

    def test_no_checkpoint_yet_is_not_a_fault(self):
        """Absent before the first run ever completes -- too early to judge."""
        plan = _call(backfill_checkpoint_age=None)
        self.assertNotIn("backfill_stalled", plan.alerts)


class OwnMemoryCeilingCannotSilentlyRegress(unittest.TestCase):
    """The exact setup of 2026-08-28's OOM storm: a per-cgroup MemoryMax
    that cannot bind on this box hands victim selection to the global OOM
    killer, which is how this service died 62 times and took another
    project's units down with it."""

    def test_a_ceiling_above_physical_ram_alerts(self):
        plan = _call(own_memory_max_bytes=4 * 1024**3,      # 4G
                     total_physical_bytes=3814 * 1024**2)   # 3814 MB box
        self.assertIn("own_memory_ceiling_exceeds_physical_ram", plan.alerts)

    def test_a_ceiling_that_actually_binds_is_quiet(self):
        plan = _call(own_memory_max_bytes=900 * 1024**2,
                     total_physical_bytes=3814 * 1024**2)
        self.assertNotIn("own_memory_ceiling_exceeds_physical_ram", plan.alerts)

    def test_no_ceiling_at_all_is_the_same_hazard_on_a_zero_swap_box(self):
        plan = _call(own_memory_max_bytes=None,
                     total_physical_bytes=3814 * 1024**2)
        self.assertIn("own_memory_ceiling_exceeds_physical_ram", plan.alerts)

    def test_unmeasurable_total_memory_makes_no_claim(self):
        plan = _call(own_memory_max_bytes=None, total_physical_bytes=None)
        self.assertNotIn("own_memory_ceiling_exceeds_physical_ram", plan.alerts)


class CredentialDoctorCatchesShapeNotAuthenticity(unittest.TestCase):
    def test_the_actual_incident_value_is_caught(self):
        env = {"GITHUB_TOKEN": (
            'ssh -i "$env:USERPROFILE\\.ssh\\quant_vps" quant@95.216.191.70 '
            '"systemctl --user restart memecoin-shadow.service"')}
        verdicts = {v.name: v for v in check_all(env)}
        self.assertFalse(verdicts["GITHUB_TOKEN"].ok)

    def test_a_real_looking_key_passes(self):
        env = {"JUPITER_API_KEY":
               "jup_a688ab456d7bf79e2334d0d1de73e47b36f38b7a432e462b580ff0f7679e2451"}
        verdicts = {v.name: v for v in check_all(env)}
        self.assertTrue(verdicts["JUPITER_API_KEY"].ok)

    def test_an_unset_credential_is_not_a_defect(self):
        verdicts = {v.name: v for v in check_all({})}
        self.assertTrue(all(v.ok for v in verdicts.values()))

    def test_a_broken_verdict_never_carries_the_raw_value(self):
        """Names, never values -- even in the failure path."""
        secret = "ssh -i supersecretpath quant@1.2.3.4"
        env = {"GITHUB_TOKEN": secret}
        verdicts = check_all(env)
        broken = [v for v in verdicts if not v.ok]
        self.assertTrue(broken)
        for v in broken:
            self.assertNotIn(secret, v.reason)
            self.assertNotIn(secret, json_of(v))

    def test_env_file_parsing_strips_quotes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env"
            path.write_text('FOO="bar baz"\nEMPTY=\n# comment\nBARE=qux\n')
            parsed = parse_env_file(path)
        self.assertEqual(parsed["FOO"], "bar baz")
        self.assertEqual(parsed["BARE"], "qux")
        self.assertEqual(parsed["EMPTY"], "")


def json_of(verdict):
    return str(verdict.as_dict())


if __name__ == "__main__":
    unittest.main()


class OwnMemoryCeilingIsCorrectedNotJustAnnounced(unittest.TestCase):
    """The exact fix the other project's drop-in already applies once,
    reapplied automatically -- self-calibrated from measured RSS rather than
    a constant, since a hardcoded guess never checked against this box's
    physical RAM is the entire original bug."""

    def test_a_bad_ceiling_is_corrected_from_measured_rss(self):
        plan = _call(own_memory_max_bytes=4 * 1024**3,
                    total_physical_bytes=3814 * 1024**2,
                    desk_rss_bytes=500 * 1024**2)
        self.assertIsNotNone(plan.correct_memory_ceiling)
        high, ceiling = plan.correct_memory_ceiling
        self.assertLess(high, ceiling)
        self.assertLess(ceiling, 3814 * 1024**2)  # never proposes the bug itself

    def test_the_correction_never_exceeds_a_safe_share_of_total_ram(self):
        """Even a huge measured RSS must not eat the whole shared box."""
        plan = _call(own_memory_max_bytes=None,
                    total_physical_bytes=3814 * 1024**2,
                    desk_rss_bytes=3000 * 1024**2)
        high, ceiling = plan.correct_memory_ceiling
        self.assertLessEqual(ceiling, int(3814 * 1024**2 * 0.45))

    def test_without_a_measured_rss_no_guess_is_made(self):
        """No constant fallback: an uncalibrated number is the original bug."""
        plan = _call(own_memory_max_bytes=4 * 1024**3,
                    total_physical_bytes=3814 * 1024**2,
                    desk_rss_bytes=None)
        self.assertIsNone(plan.correct_memory_ceiling)
        self.assertIn("own_memory_ceiling_exceeds_physical_ram", plan.alerts)

    def test_a_healthy_ceiling_proposes_no_correction(self):
        plan = _call(own_memory_max_bytes=900 * 1024**2,
                    total_physical_bytes=3814 * 1024**2,
                    desk_rss_bytes=400 * 1024**2)
        self.assertIsNone(plan.correct_memory_ceiling)


class StalledBackfillIsRetriedNotJustAnnounced(unittest.TestCase):
    """Idempotent and checkpointed, so a retry is safe -- unlike guessing
    which process to kill for the memory/disk alerts."""

    def test_a_stall_triggers_a_retry(self):
        plan = _call(backfill_checkpoint_age=70_000.0, state={})
        self.assertTrue(plan.start_backfill)
        self.assertIn("backfill_stalled", plan.alerts)

    def test_a_recent_retry_is_not_repeated_immediately(self):
        state = {"last_backfill_start_at": 9_950.0}
        plan = _call(backfill_checkpoint_age=70_000.0, state=state, now=10_000.0)
        self.assertFalse(plan.start_backfill)

    def test_a_stale_enough_retry_fires_again(self):
        state = {"last_backfill_start_at": 5_000.0}
        plan = _call(backfill_checkpoint_age=70_000.0, state=state, now=10_000.0)
        self.assertTrue(plan.start_backfill)

    def test_no_stall_means_no_retry(self):
        plan = _call(backfill_checkpoint_age=3_600.0)
        self.assertFalse(plan.start_backfill)


class SomeAlertsHaveNoSafeAutomatedFix(unittest.TestCase):
    """Not everything alert-only was left that way by omission. These stay
    alerts because the only available "fix" is either destructive (kill an
    unidentified process to free memory or disk), needs a secret only a
    human holds, needs interactive auth, or requires a passing model --
    which is this desk's entire open research question, not an operational
    fault."""

    def test_low_memory_has_no_auto_kill(self):
        state = {}
        _call(available_mib=100.0, state=state, now=10_000.0)
        plan = _call(available_mib=100.0, state=state, now=25_000.0)
        self.assertIn("training_guard_memory_starved", plan.alerts)
        # No field on Plan proposes killing anything; nothing to assert
        # false on other than the alert existing without a paired action.

    def test_disk_pressure_has_no_auto_delete(self):
        plan = _call(disk_used_fraction=0.95)
        self.assertIn("disk_nearly_full", plan.alerts)

    def test_missing_credentials_has_no_auto_repair(self):
        readiness = _healthy_readiness()
        readiness["credentials"] = {"absent": [{"name": "X_BEARER_TOKEN"}]}
        repairs, alerts = observed_faults(readiness)
        self.assertEqual(repairs, [])
        self.assertIn("optional_credentials_absent", alerts)


class WalletTrackingDeathIsWatched(unittest.TestCase):
    """wallets_tracked read zero for the desk's ENTIRE history before the
    2026-08-29 fix (an unthrottled burst was exhausting the free RPC pool on
    every launch). Alert-only, not a fixer: that specific cause is fixed, but
    a future zero-tracking regression could have an entirely different one,
    and a blind restart is not a substitute for the kind of investigation
    that actually found it.
    """

    def _readiness(self, seen, tracked):
        readiness = _healthy_readiness()
        readiness["launch_census"] = {"funnel": {"seen": seen}}
        readiness["wallet_follow"] = {"model": {"wallets_tracked": tracked}}
        return readiness

    def test_too_few_launches_is_not_yet_evidence(self):
        """Nothing to track is not the same fault as failing to track."""
        state = {}
        readiness = self._readiness(seen=3, tracked=0)
        plan = _call(readiness=readiness, state=state, now=10_000.0)
        plan = _call(readiness=readiness, state=state, now=10_000.0 + 10_000)
        self.assertNotIn("wallet_tracking_dead", plan.alerts)

    def test_sustained_zero_despite_real_launch_flow_alerts(self):
        state = {}
        readiness = self._readiness(seen=200, tracked=0)
        _call(readiness=readiness, state=state, now=10_000.0)
        plan = _call(readiness=readiness, state=state, now=10_000.0 + 7_201)
        self.assertIn("wallet_tracking_dead", plan.alerts)

    def test_a_brief_zero_stretch_does_not_alert(self):
        """The pipeline's own 5-minute recalc cycle and 300s follow horizon
        must not be flagged as a fault during normal warm-up."""
        state = {}
        readiness = self._readiness(seen=200, tracked=0)
        _call(readiness=readiness, state=state, now=10_000.0)
        plan = _call(readiness=readiness, state=state, now=10_500.0)
        self.assertNotIn("wallet_tracking_dead", plan.alerts)

    def test_any_tracked_wallet_resets_the_clock(self):
        state = {}
        dead = self._readiness(seen=200, tracked=0)
        _call(readiness=dead, state=state, now=10_000.0)
        alive = self._readiness(seen=200, tracked=3)
        _call(readiness=alive, state=state, now=15_000.0)
        plan = _call(readiness=dead, state=state, now=15_100.0)
        self.assertNotIn("wallet_tracking_dead", plan.alerts)

    def test_no_fixer_is_proposed_for_this(self):
        """Alert-only by design: the cause could differ next time."""
        state = {}
        readiness = self._readiness(seen=200, tracked=0)
        _call(readiness=readiness, state=state, now=10_000.0)
        plan = _call(readiness=readiness, state=state, now=10_000.0 + 7_201)
        self.assertFalse(plan.restart_desk)
        self.assertEqual(plan.reenable_units, [])
