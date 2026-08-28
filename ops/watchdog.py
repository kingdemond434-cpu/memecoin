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


@dataclass
class Policy:
    readiness_stale_seconds: float = 180.0
    restart_cooldown_seconds: float = 300.0
    restart_window_seconds: float = 3600.0
    max_restarts_per_window: int = 3
    persistent_fault_runs: int = 2
    training_stale_seconds: float = 93_600.0
    training_retry_seconds: float = 3_600.0


@dataclass
class Plan:
    restart_desk: bool = False
    start_trainer: bool = False
    enable_desk: bool = False
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

    stream_events = readiness.get("stream_events") or {}
    if (int(stream_events.get("total", 0) or 0) >= 100
            and int(stream_events.get("token_created", 0) or 0) == 0):
        alerts.append("chain_stream_delivers_no_creation_events")

    decoder = readiness.get("pump_decoder") or {}
    if str(decoder.get("status", "")).upper() in {"DEGRADED", "CRITICAL"}:
        alerts.append("pump_decoder_degraded")

    event_loop = readiness.get("event_loop") or {}
    if (int(event_loop.get("candidate_drops", 0) or 0)
            or int(event_loop.get("redecision_drops", 0) or 0)):
        alerts.append("decision_queue_dropped_work")

    miners = readiness.get("data_miners") or {}
    if (int(miners.get("runnable", 0) or 0) > 0
            and str(miners.get("status", "")).upper() == "DATA_BLOCKED"):
        alerts.append("all_runnable_data_miners_are_dark")

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
           trainer_active: bool, training_age: Optional[float]) -> Plan:
    """Pure repair decision, separated so restart storms are testable."""
    plan = Plan(enable_desk=not service_enabled)
    immediate: List[str] = []
    persistent: List[str] = []
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
    plan = decide(
        service_active=service_active, service_enabled=service_enabled,
        readiness=_read_json(readiness_path), readiness_age=readiness_age,
        state=state, now=now, policy=Policy(), trainer_active=trainer_active,
        training_age=_latest_training_age(root / "models", now))

    actions: List[str] = []
    failures: List[str] = []
    if plan.enable_desk and not args.dry_run:
        result = controller.call("enable", args.desk_unit)
        (actions if result.returncode == 0 else failures).append("enable_desk")
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

    payload = {
        "schema": WATCHDOG_SCHEMA_VERSION, "at": now,
        "service_active": service_active, "service_enabled": service_enabled,
        "readiness_age_seconds": readiness_age,
        "plan": asdict(plan), "actions": actions, "failures": failures,
        "dry_run": bool(args.dry_run),
    }
    state["last_run_at"] = now
    state["last_plan"] = asdict(plan)
    state["last_failures"] = failures
    _write_json_atomic(state_path, state)
    _append_event(event_path, payload)

    material = failures or plan.repair_reasons or plan.alerts
    if material:
        summary = ("memecoin watchdog: "
                   f"actions={actions or ['none']} failures={failures or ['none']} "
                   f"repairs={plan.repair_reasons or ['none']} "
                   f"alerts={plan.alerts or ['none']}")
        logger.warning(summary)
        _send_telegram_alert(summary)
    return EXIT_FAILED if failures else EXIT_REPAIRED if actions else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
