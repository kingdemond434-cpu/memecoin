"""Bounded, local self-healing for the always-on memecoin shadow desk.

This process is deliberately small and independent of the desk.  It repairs
only faults where a restart is a real remedy: a stopped service, a stale
heartbeat, a dead indispensable coroutine, or a fully stopped source mesh.
It never invents credentials, promotes a model, weakens a veto, deletes data,
or unlocks live trading.  Those are not operational repairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("ops.watchdog")

WATCHDOG_SCHEMA_VERSION = "v1"
EXIT_OK, EXIT_REPAIRED, EXIT_FAILED = 0, 1, 2

#: Every unit this desk owns, checked for the exact failure observed
#: 2026-08-29: a cross-project OOM event took down the desk service and all
#: five timers by disabling them, not merely stopping them, so nothing came
#: back on its own restart-always policy or its next scheduled tick.
#:
#: Limitation, stated rather than hidden: this check runs FROM
#: memecoin-watchdog.timer, so the one entry it cannot recover unaided is its
#: own timer being disabled -- nothing would be running to notice. The weekly
#: audit-pack timer is a second, independent trigger point for the same
#: check, which bounds the outage to at most a week even in that exact case,
#: rather than fixing it outright.
FLEET_UNITS: Tuple[str, ...] = (
    "memecoin-shadow.service",
    "memecoin-watchdog.timer",
    "memecoin-health.timer",
    "memecoin-shadow-trainer.timer",
    "memecoin-feed-doctor.timer",
    "memecoin-audit-pack.timer",
    "memecoin-backfill.timer",
    "memecoin-credential-doctor.timer",
)


@dataclass
class Policy:
    readiness_stale_seconds: float = 180.0
    restart_cooldown_seconds: float = 300.0
    restart_window_seconds: float = 3600.0
    max_restarts_per_window: int = 3
    persistent_fault_runs: int = 2
    training_stale_seconds: float = 93_600.0
    training_retry_seconds: float = 3_600.0
    #: How long the Telegram mention count may stand still before the signal
    #: path is called stalled. Across 38 channels the observed rate is about
    #: one mention a minute, but it is bursty and genuinely quiet stretches
    #: happen; half an hour is far outside them and still catches a real
    #: outage inside the hour it would otherwise cost.
    telegram_stall_seconds: float = 1_800.0
    #: Mirrors the live training_guard's own --min-available-mib (see
    #: src/runtime/training_guard.py). Declared here too, deliberately,
    #: rather than imported: this alert is about visibility into a condition
    #: that already cost six silent hourly skips on 2026-08-28, and a
    #: watchdog whose threshold constant depends on the exact module it is
    #: warning about is one refactor away from silently losing that warning.
    training_guard_min_mib: float = 640.0
    #: How long available memory may stay under that floor before the
    #: watchdog says so on its own, rather than waiting for
    #: training_stale_seconds (26h) to notice indirectly. Three hours is
    #: three missed hourly attempts -- a real pattern, not one bad tick.
    training_low_memory_alert_seconds: float = 10_800.0
    #: Disk full breaks everything downstream at once: episodes cannot be
    #: written, the trade-evidence hash chain cannot append, models cannot
    #: save. 90% leaves headroom to alert before the failure, not after it.
    disk_usage_alert_fraction: float = 0.90
    #: The backfill timer runs every 6h; three missed cycles (18h) is a
    #: pattern, not one contended run losing to the live desk for RPC quota
    #: -- which is expected and already handled by the tool's own retry.
    backfill_stale_seconds: float = 64_800.0
    #: How often the watchdog itself may retry a stalled backfill. Shorter
    #: than the timer's own 6h cadence, since the whole point is closing the
    #: gap faster; long enough that a persistently-contended RPC pool is not
    #: hammered every 60s by the watchdog on top of the timer's own attempts.
    backfill_retry_seconds: float = 3_600.0
    #: wallet_follow.model.wallets_tracked read zero for the desk's entire
    #: history before 2026-08-29's fix (an unthrottled burst was exhausting
    #: the free RPC pool on every single launch). How long it may stay at
    #: zero WHILE launches keep arriving before that is called an anomaly
    #: again rather than early pipeline warm-up: the deeper chain (wallet ->
    #: score -> watch list -> follow -> resolve) has a 5-minute recalc cycle
    #: and a 300s follow horizon built in, so anything shorter than this
    #: would flag the pipeline's own normal warm-up as a fault.
    wallet_tracking_dead_seconds: float = 7_200.0
    #: Below this many newly-seen launches, "wallets_tracked is still zero"
    #: is not yet evidence of anything -- there has been nothing to track.
    wallet_tracking_min_launches: int = 20


@dataclass
class Plan:
    restart_desk: bool = False
    start_trainer: bool = False
    enable_desk: bool = False
    #: Units found disabled and queued for `systemctl --user enable`.
    #: Unconditional, unlike restart_desk: enabling an already-enabled unit
    #: is a no-op, so there is no flapping risk this needs to be debounced
    #: against, and the whole point is to close this gap faster than the
    #: two-observation rule would.
    reenable_units: List[str] = field(default_factory=list)
    #: (MemoryHigh bytes, MemoryMax bytes) to apply live via `systemctl
    #: set-property` when the desk's own ceiling cannot bind on this box.
    #: Unconditional like reenable_units: set-property is idempotent and
    #: applies without a restart, so there is nothing to debounce.
    correct_memory_ceiling: Optional[Tuple[int, int]] = None
    start_backfill: bool = False
    repair_reasons: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True),
                         encoding="utf-8")
    temporary.replace(path)


def _good_stream(status: Any) -> bool:
    return str(status or "").upper() in {
        "OK", "STREAMING", "RPC_WS", "RPC_FALLBACK"}


def observed_faults(readiness: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (repairable, alert-only) faults from one fresh snapshot."""
    repairable: List[str] = []
    alerts: List[str] = []

    tasks = readiness.get("runtime_tasks") or {}
    failed_tasks = list(tasks.get("failed") or ())
    if str(tasks.get("status", "")).upper() == "CRITICAL" or failed_tasks:
        repairable.append("critical_runtime_task_failed")

    mesh = readiness.get("source_mesh") or {}
    sources = int(mesh.get("sources", 0) or 0)
    producers = int(mesh.get("producers", 0) or 0)
    if sources and (not mesh.get("streaming") or producers < sources):
        repairable.append("source_mesh_not_running")

    yellowstone = readiness.get("yellowstone") or {}
    rpc_stream = readiness.get("rpc_program_stream") or {}
    if (not _good_stream(yellowstone.get("status"))
            and not _good_stream(rpc_stream.get("status"))):
        repairable.append("all_chain_event_streams_down")

    memory = readiness.get("memory") or {}
    if str(memory.get("band", "")).lower() == "shed":
        repairable.append("memory_governor_persistently_shedding")

    # Repairable, not alert-only: a stream that delivers trades but no
    # creation events is a broken subscription on a healthy socket, and a
    # miner pool that is entirely dark while runnable is a wedged scheduler.
    # Both are exactly what a process restart re-establishes. They still pass
    # through the two-observation persistence rule, the cooldown and the
    # restart budget -- an immediate fixer without those is a flapping desk.
    stream_events = readiness.get("stream_events") or {}
    if (int(stream_events.get("total", 0) or 0) >= 100
            and int(stream_events.get("token_created", 0) or 0) == 0):
        repairable.append("chain_stream_delivers_no_creation_events")

    # Alert-only, and the reason is structural: restarting re-decodes the
    # same unknown bytes the same way. Only new decoder code fixes this.
    decoder = readiness.get("pump_decoder") or {}
    if str(decoder.get("status", "")).upper() in {"DEGRADED", "CRITICAL"}:
        alerts.append("pump_decoder_degraded")

    # Alert-only: the drops already happened. A restart destroys the queue
    # that is now keeping up and un-drops nothing.
    event_loop = readiness.get("event_loop") or {}
    if (int(event_loop.get("candidate_drops", 0) or 0)
            or int(event_loop.get("redecision_drops", 0) or 0)):
        alerts.append("decision_queue_dropped_work")

    miners = readiness.get("data_miners") or {}
    if (int(miners.get("runnable", 0) or 0) > 0
            and str(miners.get("status", "")).upper() == "DATA_BLOCKED"):
        repairable.append("all_runnable_data_miners_are_dark")

    # Telegram is the desk's fastest human-signal path and the one whose
    # failure is quietest: the session stays "connected", the handler stays
    # registered, and messages simply stop. Nothing else here would notice,
    # so the signal path is checked on its own terms rather than inferred
    # from the source mesh totals it does not contribute to.
    telegram = ((readiness.get("credentials") or {}).get("telegram") or {})
    social_status = ((readiness.get("social") or {}).get("data_status") or {})
    if telegram.get("keys_present"):
        if not telegram.get("session_authorised"):
            # Only an operator can fix this, and until they do the desk is
            # blind to every channel while looking healthy.
            alerts.append("telegram_session_unauthorised")
        elif str(social_status.get("telegram", "")).upper().startswith("DATA_BLOCKED"):
            # Alert-only on purpose: the common cause is a flood-wait, and
            # reconnecting during one extends the penalty. The repair would
            # be the injury.
            alerts.append("telegram_signal_path_blocked")
        elif not int(telegram.get("channels_listed", 0) or 0):
            alerts.append("telegram_no_channels_configured")

    # These require evidence or an operator/provider.  Naming them makes the
    # watchdog complete without pretending a restart can manufacture a fix.
    if readiness.get("prediction") != "OK":
        alerts.append("prediction_model_data_blocked")
    hazard = readiness.get("rug_hazard") or {}
    if not hazard.get("model_trained"):
        alerts.append("rug_hazard_model_data_blocked")
    credentials = readiness.get("credentials") or {}
    if credentials.get("absent"):
        alerts.append("optional_credentials_absent")
    return list(dict.fromkeys(repairable)), list(dict.fromkeys(alerts))


def decide(*, service_active: bool, service_enabled: bool,
           readiness: Dict[str, Any], readiness_age: Optional[float],
           state: Dict[str, Any], now: float, policy: Policy,
           trainer_active: bool, training_age: Optional[float],
           disabled_units: Sequence[str] = (),
           available_mib: Optional[float] = None,
           disk_used_fraction: Optional[float] = None,
           backfill_checkpoint_age: Optional[float] = None,
           own_memory_max_bytes: Optional[int] = None,
           total_physical_bytes: Optional[int] = None,
           desk_rss_bytes: Optional[int] = None) -> Plan:
    """Pure repair decision, separated so restart storms are testable.

    Every argument here is a value main() already read; decide() does no I/O
    of its own so the whole decision stays a plain function of its inputs,
    which is what makes restart storms and alert thresholds testable without
    a live systemd, a live desk, or a live filesystem behind them.
    """
    plan = Plan(enable_desk=not service_enabled)
    plan.reenable_units = [unit for unit in disabled_units]
    immediate: List[str] = []
    persistent: List[str] = []

    if available_mib is not None:
        if available_mib < policy.training_guard_min_mib:
            since = float(state.get("low_memory_since", 0.0) or 0.0)
            if not since:
                state["low_memory_since"] = now
            elif now - since > policy.training_low_memory_alert_seconds:
                plan.alerts.append("training_guard_memory_starved")
        else:
            state.pop("low_memory_since", None)

    if disk_used_fraction is not None and disk_used_fraction >= policy.disk_usage_alert_fraction:
        plan.alerts.append("disk_nearly_full")

    if backfill_checkpoint_age is not None and backfill_checkpoint_age > policy.backfill_stale_seconds:
        plan.alerts.append("backfill_stalled")
        # The tool is idempotent and checkpointed -- kicking it once more is
        # a safe repair, unlike killing whatever is consuming memory or disk
        # would be. Retried, not repeated every tick: if the stall is
        # persistent RPC contention rather than a one-off, spamming `start`
        # every 60s wastes the exact quota it is trying to use.
        last_attempt = float(state.get("last_backfill_start_at", 0.0) or 0.0)
        if now - last_attempt >= policy.backfill_retry_seconds:
            plan.start_backfill = True

    if total_physical_bytes is not None:
        # Exactly tonight's incident: a per-cgroup ceiling set above physical
        # RAM (or no ceiling at all, on a box with zero swap) cannot bind, so
        # the GLOBAL OOM killer picks a victim from the whole machine instead
        # -- which is how this service was killed 62 times in 7 days and took
        # another project's units down with it. A correct fix already exists
        # as a drop-in (memecoin-shadow.service.d/10-blast-radius.conf, owned
        # by the other project's automation); this exists to notice if that
        # drop-in is ever removed or reverted rather than silently
        # regressing back into the same failure.
        if own_memory_max_bytes is None or own_memory_max_bytes > total_physical_bytes:
            plan.alerts.append("own_memory_ceiling_exceeds_physical_ram")
            # Repaired live via `systemctl set-property` -- the exact
            # mechanism the other project's own fix used, applied without a
            # restart. Self-calibrating from the desk's OWN measured RSS
            # rather than a constant, because a hardcoded guess baked into
            # this file is exactly how the original bug (MemoryMax=4G,
            # never once checked against this box's 3814 MB) happened.
            # Capped at a fraction of total RAM so a correction cannot itself
            # starve everything else running on a shared box.
            if desk_rss_bytes and desk_rss_bytes > 0:
                high = min(int(desk_rss_bytes * 1.6), int(total_physical_bytes * 0.35))
                ceiling = min(int(desk_rss_bytes * 2.2), int(total_physical_bytes * 0.45))
                floor = 256 * 1024 * 1024
                high = max(high, floor)
                ceiling = max(ceiling, high + 128 * 1024 * 1024)
                plan.correct_memory_ceiling = (high, ceiling)
    if not service_active:
        immediate.append("desk_service_inactive")
    elif readiness_age is None:
        immediate.append("readiness_missing")
    elif readiness_age > policy.readiness_stale_seconds:
        immediate.append("readiness_stale")
    else:
        repairable, alerts = observed_faults(readiness)
        persistent.extend(repairable)
        plan.alerts.extend(alerts)
        # A stalled Telegram feed reports OK on every field it has, so the
        # only evidence it has stopped is that the mention count is not
        # moving -- which needs memory across runs, and observed_faults
        # deliberately has none. Repairable, not alert-only: a restart
        # reconnects the Telethon client, which is the fix. It joins the
        # persistent list so it obeys the same two-observation rule,
        # cooldown and restart budget as every other repair.
        telegram = ((readiness.get("credentials") or {}).get("telegram") or {})
        if telegram.get("keys_present") and telegram.get("session_authorised"):
            mentions = int(((readiness.get("social") or {}).get("total_mentions", 0)) or 0)
            previous = state.get("telegram_mentions")
            since = float(state.get("telegram_mentions_at", 0.0) or 0.0)
            if previous is None or mentions != int(previous):
                state["telegram_mentions"] = mentions
                state["telegram_mentions_at"] = now
            elif since and now - since > policy.telegram_stall_seconds:
                persistent.append("telegram_signals_stalled")

        # Alert-only, unlike the Telegram stall above: a restart is not the
        # fix here, because the last time this read zero for the desk's
        # entire history the cause was an unthrottled burst inside the
        # wallet-intelligence pipeline itself (fixed 2026-08-29), not a
        # connection that reconnecting repairs. A future zero-tracking
        # regression needs the same kind of investigation, not a blind
        # restart, so this names the anomaly rather than guessing at a
        # remedy for a failure mode that could recur for an entirely
        # different reason next time.
        launches_seen = int(((readiness.get("launch_census") or {}).get("funnel") or {})
                            .get("seen", 0) or 0)
        wallets_tracked = int(((readiness.get("wallet_follow") or {}).get("model") or {})
                              .get("wallets_tracked", 0) or 0)
        if launches_seen >= policy.wallet_tracking_min_launches:
            if wallets_tracked > 0:
                state.pop("wallet_tracking_dead_since", None)
            else:
                since = float(state.get("wallet_tracking_dead_since", 0.0) or 0.0)
                if not since:
                    state["wallet_tracking_dead_since"] = now
                elif now - since > policy.wallet_tracking_dead_seconds:
                    plan.alerts.append("wallet_tracking_dead")

    consecutive = dict(state.get("consecutive_faults") or {})
    seen = set(immediate + persistent)
    for reason in set(consecutive) | seen:
        consecutive[reason] = (
            int(consecutive.get(reason, 0)) + 1 if reason in seen else 0)
    state["consecutive_faults"] = consecutive

    reasons = list(immediate)
    reasons.extend(reason for reason in persistent
                   if consecutive.get(reason, 0) >= policy.persistent_fault_runs)
    restarts = [float(item) for item in state.get("restart_times", [])
                if now - float(item) <= policy.restart_window_seconds]
    state["restart_times"] = restarts
    last_restart = float(state.get("last_restart_at", 0.0) or 0.0)
    cooldown_ok = now - last_restart >= policy.restart_cooldown_seconds
    budget_ok = len(restarts) < policy.max_restarts_per_window
    if reasons and cooldown_ok and budget_ok:
        plan.restart_desk = True
        plan.repair_reasons.extend(reasons)
    elif reasons:
        plan.alerts.append(
            "restart_suppressed_by_cooldown" if not cooldown_ok
            else "restart_budget_exhausted")

    if (training_age is not None
            and training_age > policy.training_stale_seconds
            and not trainer_active
            and now - float(state.get("last_trainer_start_at", 0.0) or 0.0)
            >= policy.training_retry_seconds):
        plan.start_trainer = True
    return plan


class Systemctl:
    def __init__(self, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.runner = runner

    def call(self, *arguments: str) -> subprocess.CompletedProcess:
        return self.runner(
            ["systemctl", "--user", *arguments], text=True,
            capture_output=True, timeout=30, check=False)

    def active(self, unit: str) -> bool:
        return self.call("is-active", "--quiet", unit).returncode == 0

    def enabled(self, unit: str) -> bool:
        return self.call("is-enabled", "--quiet", unit).returncode == 0

    def show_property(self, unit: str, prop: str) -> str:
        result = self.call("show", unit, "-p", prop, "--value")
        return result.stdout.strip() if result.returncode == 0 else ""


def _read_meminfo_field(field_name: str) -> Optional[int]:
    """Bytes for one /proc/meminfo field, or None if it cannot be read.

    Duplicated from src.runtime.training_guard's approach rather than
    imported, for the same reason the memory threshold is duplicated in
    Policy: this module's only dependency should be the standard library, so
    a broken import elsewhere in the desk cannot also take out its watchdog.
    """
    try:
        for row in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if row.startswith(f"{field_name}:"):
                return int(row.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _own_memory_max_bytes(controller: "Systemctl", unit: str) -> Optional[int]:
    raw = controller.show_property(unit, "MemoryMax")
    if not raw or raw == "infinity":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _backfill_checkpoint_age(root: Path, now: float) -> Optional[float]:
    path = root / "data/state/backfill_checkpoint.json"
    data = _read_json(path)
    updated_at = data.get("updated_at")
    if updated_at is None:
        # Absent before the first run ever completes -- not a fault, just
        # too early to have an opinion.
        return None
    try:
        return now - float(updated_at)
    except (TypeError, ValueError):
        return None


def _latest_training_age(model_dir: Path, now: float) -> float:
    paths = [model_dir / name for name in (
        "last_training_report.json", "last_hazard_training_report.json",
        "last_exit_policy_report.json")]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return float("inf")
    return now - min(path.stat().st_mtime for path in existing)


def _append_event(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _send_telegram_alert(message: str) -> bool:
    token = (os.getenv("TELEGRAM_ALERT_BOT_TOKEN", "")
             or os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id = os.getenv("TELEGRAM_ALERT_CHAT_ID", "")
    if not token or not chat_id:
        # Say so. An unconfigured alert channel returning a quiet False means
        # every alert this watchdog has ever raised was written to a journal
        # nobody is watching, and the desk looks silent-because-healthy
        # instead of silent-because-nobody-is-listening. Names, never values.
        missing = [name for name, value in (
            ("TELEGRAM_ALERT_BOT_TOKEN or TELEGRAM_BOT_TOKEN", token),
            ("TELEGRAM_ALERT_CHAT_ID", chat_id)) if not value]
        logger.warning(
            "ALERT NOT DELIVERED (%s unset); alert was: %s",
            " and ".join(missing), message)
        return False
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return 200 <= int(response.status) < 300
    except Exception as exc:  # alerts must never break repair
        logger.warning("Telegram alert failed: %s", exc)
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="bounded memecoin desk self-healing")
    parser.add_argument("--root", default=".")
    parser.add_argument("--desk-unit", default="memecoin-shadow.service")
    parser.add_argument("--trainer-unit", default="memecoin-shadow-trainer.service")
    parser.add_argument("--backfill-unit", default="memecoin-backfill.service")
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and record without calling systemctl")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    root = Path(args.root).resolve()
    state_path = root / "data/state/watchdog_state.json"
    event_path = root / "data/state/watchdog_events.jsonl"
    maintenance = root / "data/state/maintenance.lock"
    now = time.time()
    if maintenance.exists():
        logger.info("maintenance lock present; no repair attempted")
        return EXIT_OK

    controller = Systemctl()
    service_active = controller.active(args.desk_unit)
    service_enabled = controller.enabled(args.desk_unit)
    trainer_active = controller.active(args.trainer_unit)
    readiness_path = root / "data/state/readiness.json"
    readiness_age = (now - readiness_path.stat().st_mtime
                     if readiness_path.exists() else None)
    state = _read_json(state_path)

    disabled_units = [unit for unit in FLEET_UNITS if not controller.enabled(unit)]
    try:
        usage = shutil.disk_usage(root)
        disk_used_fraction = 1.0 - (usage.free / max(usage.total, 1))
    except OSError:
        disk_used_fraction = None
    mem_available_bytes = _read_meminfo_field("MemAvailable")
    available_mib = (mem_available_bytes / (1024 * 1024)
                     if mem_available_bytes is not None else None)

    readiness = _read_json(readiness_path)
    desk_rss_bytes = (readiness.get("memory") or {}).get("rss_bytes")
    plan = decide(
        service_active=service_active, service_enabled=service_enabled,
        readiness=readiness, readiness_age=readiness_age,
        state=state, now=now, policy=Policy(), trainer_active=trainer_active,
        training_age=_latest_training_age(root / "models", now),
        disabled_units=disabled_units,
        available_mib=available_mib,
        disk_used_fraction=disk_used_fraction,
        backfill_checkpoint_age=_backfill_checkpoint_age(root, now),
        own_memory_max_bytes=_own_memory_max_bytes(controller, args.desk_unit),
        total_physical_bytes=_read_meminfo_field("MemTotal"),
        desk_rss_bytes=desk_rss_bytes)

    actions: List[str] = []
    failures: List[str] = []
    if plan.enable_desk and not args.dry_run:
        result = controller.call("enable", args.desk_unit)
        (actions if result.returncode == 0 else failures).append("enable_desk")
    for unit in plan.reenable_units:
        if args.dry_run:
            break
        result = controller.call("enable", unit)
        (actions if result.returncode == 0 else failures).append(f"reenable:{unit}")
    if plan.correct_memory_ceiling and not args.dry_run:
        high, ceiling = plan.correct_memory_ceiling
        result = controller.call(
            "set-property", args.desk_unit,
            f"MemoryHigh={high}", f"MemoryMax={ceiling}")
        (actions if result.returncode == 0 else failures).append("correct_memory_ceiling")
    if plan.start_backfill and not args.dry_run:
        result = controller.call("--no-block", "start", args.backfill_unit)
        if result.returncode == 0:
            actions.append("start_backfill")
            state["last_backfill_start_at"] = now
        else:
            failures.append("start_backfill")
    if plan.restart_desk and not args.dry_run:
        result = controller.call("restart", args.desk_unit)
        if result.returncode == 0:
            actions.append("restart_desk")
            state.setdefault("restart_times", []).append(now)
            state["last_restart_at"] = now
        else:
            failures.append("restart_desk")
    if plan.start_trainer and not args.dry_run:
        result = controller.call("--no-block", "start", args.trainer_unit)
        if result.returncode == 0:
            actions.append("start_trainer")
            state["last_trainer_start_at"] = now
        else:
            failures.append("start_trainer")

    material = failures or plan.repair_reasons or plan.alerts
    alert_delivered: Optional[bool] = None
    if material:
        summary = ("memecoin watchdog: "
                   f"actions={actions or ['none']} failures={failures or ['none']} "
                   f"repairs={plan.repair_reasons or ['none']} "
                   f"alerts={plan.alerts or ['none']}")
        logger.warning(summary)
        alert_delivered = _send_telegram_alert(summary)

    payload = {
        "schema": WATCHDOG_SCHEMA_VERSION, "at": now,
        "service_active": service_active, "service_enabled": service_enabled,
        "readiness_age_seconds": readiness_age,
        "plan": asdict(plan), "actions": actions, "failures": failures,
        "dry_run": bool(args.dry_run),
        # Whether anyone was actually told. Without this the ledger records
        # that an alert was raised and leaves whether it arrived unknowable.
        "alert_delivered": alert_delivered,
    }
    state["last_run_at"] = now
    state["last_plan"] = asdict(plan)
    state["last_failures"] = failures
    state["last_alert_delivered"] = alert_delivered
    _write_json_atomic(state_path, state)
    _append_event(event_path, payload)

    return EXIT_FAILED if failures else EXIT_REPAIRED if actions else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
