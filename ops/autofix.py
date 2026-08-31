"""Corrective actions the node takes on itself, bounded so they cannot hide.

The monitor reports. Nothing acted on the reports, so every fault waited for a
person to notice -- which on a desk whose only job is to accumulate evidence
means the evidence stops accumulating until someone SSHes in.

So this acts. Carefully, because an auto-fixer is a liability before it is an
asset:

**Every action is bounded.** A restart is attempted at most a few times an
hour. Beyond that the fault is not transient and restarting is no longer a
fix, it is a way of hiding a fault while producing no evidence.

**Every action escalates when it stops working.** The failure mode of
supervision is a service that restarts forever and nobody is told. Past the
budget, this stops acting and says so loudly, which is the correct behaviour:
a human is needed and the machine should say why rather than keep flapping.

**No action invents a fix it does not understand.** There is a fixed
repertoire -- restart the desk, clear a wedged socket, roll back a bad deploy.
An unrecognised fault is reported and left alone. A supervisor that improvises
against a fault it cannot name will eventually improvise against a fault that
was not there.

**Nothing here can trade, sign, or touch capital.** It restarts processes and
moves files. That boundary is what makes it safe to run unattended.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("autofix")

AUTOFIX_SCHEMA_VERSION = "v1"

#: How many times one action may run in the window before it is abandoned as
#: not-a-fix. Deliberately small: three restarts that did not help are
#: evidence that a fourth will not either.
DEFAULT_BUDGET = 3
DEFAULT_WINDOW_S = 3_600.0

#: Minimum gap between two attempts at the same action. A desk needs a couple
#: of minutes to reach a steady state before its health means anything, and a
#: supervisor that re-checks sooner will restart a service that was starting.
DEFAULT_COOLDOWN_S = 240.0


class Outcome(Enum):
    NOT_NEEDED = "not_needed"
    ACTED = "acted"
    COOLING = "cooling"
    BUDGET_SPENT = "budget_spent"
    FAILED = "failed"


@dataclass
class Remedy:
    """One fault this supervisor knows how to act on."""

    name: str
    #: Reads the health report and says whether this remedy applies.
    applies: Callable[[Dict[str, Any]], bool]
    #: Performs the action. Returns True if it ran cleanly.
    act: Callable[[], bool]
    why: str = ""
    budget: int = DEFAULT_BUDGET
    cooldown_s: float = DEFAULT_COOLDOWN_S


@dataclass
class Attempt:
    remedy: str
    at: float
    outcome: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"remedy": self.remedy, "at": self.at,
                "outcome": self.outcome, "detail": self.detail}


def systemctl(*args: str, user: bool = True, timeout: float = 60.0) -> bool:
    """Run one systemctl command. Never raises; the caller decides."""
    command = ["systemctl"] + (["--user"] if user else []) + list(args)
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error("systemctl %s failed: %s", " ".join(args), exc)
        return False
    if result.returncode != 0:
        logger.error("systemctl %s exited %d: %s", " ".join(args),
                     result.returncode, result.stderr.strip()[:400])
        return False
    return True


class AutoFixer:
    """Applies a fixed repertoire of remedies, within a budget, and escalates."""

    def __init__(self, state_path: Optional[Path] = None, *,
                 window_s: float = DEFAULT_WINDOW_S):
        self.state_path = Path(state_path) if state_path else None
        self.window_s = float(window_s)
        self.remedies: List[Remedy] = []
        self.attempts: List[Attempt] = []
        self.escalations: List[str] = []
        self._load()

    def register(self, remedy: Remedy) -> None:
        self.remedies.append(remedy)

    # --- the loop --------------------------------------------------------

    def run(self, health: Dict[str, Any],
            now: Optional[float] = None) -> List[Attempt]:
        """One pass. At most one remedy acts, because two faults are usually
        one fault seen twice, and acting on both compounds the disturbance."""
        moment = time.time() if now is None else now
        self._expire(moment)
        acted: List[Attempt] = []
        for remedy in self.remedies:
            try:
                if not remedy.applies(health):
                    continue
            except Exception as exc:
                logger.warning("remedy %s could not evaluate: %s",
                               remedy.name, exc)
                continue
            attempt = self._attempt(remedy, moment)
            acted.append(attempt)
            self.attempts.append(attempt)
            if attempt.outcome == Outcome.ACTED.value:
                # One disturbance per pass.
                break
        self._save()
        return acted

    def _attempt(self, remedy: Remedy, now: float) -> Attempt:
        recent = [item for item in self.attempts
                  if item.remedy == remedy.name
                  and item.outcome == Outcome.ACTED.value]
        if recent and now - max(item.at for item in recent) < remedy.cooldown_s:
            return Attempt(remedy.name, now, Outcome.COOLING.value,
                           "acted recently; a restarting service has no health")
        if len(recent) >= remedy.budget:
            message = (f"{remedy.name} has run {len(recent)} times in the last "
                       f"{self.window_s / 3600:.0f}h and the fault persists; "
                       "this is not a transient fault and restarting is no "
                       "longer a fix")
            if message not in self.escalations:
                self.escalations.append(message)
                logger.error("AUTOFIX ESCALATION: %s", message)
            return Attempt(remedy.name, now, Outcome.BUDGET_SPENT.value, message)
        logger.warning("autofix acting: %s (%s)", remedy.name, remedy.why)
        try:
            ok = bool(remedy.act())
        except Exception as exc:
            return Attempt(remedy.name, now, Outcome.FAILED.value,
                           f"{type(exc).__name__}: {exc}")
        return Attempt(remedy.name, now,
                       Outcome.ACTED.value if ok else Outcome.FAILED.value,
                       remedy.why)

    def _expire(self, now: float) -> None:
        self.attempts = [item for item in self.attempts
                         if now - item.at <= self.window_s]

    # --- reporting and persistence ---------------------------------------

    def report(self, now: Optional[float] = None) -> Dict[str, Any]:
        moment = time.time() if now is None else now
        self._expire(moment)
        by_remedy: Dict[str, int] = {}
        for item in self.attempts:
            if item.outcome == Outcome.ACTED.value:
                by_remedy[item.remedy] = by_remedy.get(item.remedy, 0) + 1
        return {
            "schema": AUTOFIX_SCHEMA_VERSION,
            "status": ("CRITICAL" if self.escalations else
                       "WARN" if by_remedy else "OK"),
            "detail": ("; ".join(self.escalations) if self.escalations else
                       "acting on a fault the node can correct itself"
                       if by_remedy else ""),
            "registered": [remedy.name for remedy in self.remedies],
            "actions_in_window": by_remedy,
            "window_hours": round(self.window_s / 3600.0, 1),
            "escalations": list(self.escalations),
            "recent": [item.to_dict() for item in self.attempts[-20:]],
        }

    def _save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "attempts": [item.to_dict() for item in self.attempts],
                "escalations": self.escalations}))
        except OSError as exc:
            logger.warning("autofix state unwritable: %s", exc)

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.attempts = [Attempt(row.get("remedy", ""), float(row.get("at", 0)),
                                 row.get("outcome", ""), row.get("detail", ""))
                         for row in state.get("attempts") or []]
        self.escalations = list(state.get("escalations") or [])


def _in_state(health: Dict[str, Any], state: str, names: Sequence[str]) -> bool:
    for check in health.get("checks") or []:
        if check.get("name") in names and check.get("state") == state:
            return True
    return False


def _critical(health: Dict[str, Any], *names: str) -> bool:
    """True when any named check is CRITICAL in the monitor's output."""
    return _in_state(health, "CRITICAL", names)


def _warn(health: Dict[str, Any], *names: str) -> bool:
    """True when any named check is WARN.

    Kept separate on purpose. A warning is a reason to act slowly and rarely;
    treating it like a crisis is how a supervisor turns a degraded subsystem
    into a restart loop.
    """
    return _in_state(health, "WARN", names)


def post(url: str, timeout: float = 20.0) -> bool:
    """One POST to the desk's own loopback API. Never raises."""
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.error("POST %s failed: %s", url, exc)
        return False


def prune_spill(root: Path, *, keep_bytes: int = 512 * 1024**2) -> bool:
    """Trim the append-only spill logs when the disk is filling.

    These are the only files here that grow without bound and can be shortened
    without losing anything the totals depend on: the launch census counts
    before it spills, and the landing model replays only its most recent rows.
    Oldest lines go first, so the newest history -- the part a model would
    actually train on -- is what survives.

    Nothing else is touched. Deleting a model artefact or an evidence ledger
    to free space would trade the thing the desk is for the ability to keep
    running, which is not a trade a supervisor gets to make.
    """
    spills = [root / "data" / "state" / "launch_census.jsonl",
              root / "data" / "state" / "landing_attempts.jsonl"]
    trimmed = False
    for path in spills:
        try:
            if not path.exists() or path.stat().st_size <= keep_bytes:
                continue
            lines = path.read_text(errors="replace").splitlines()
            keep = lines[len(lines) // 2:]
            path.write_text("\n".join(keep) + "\n")
            logger.warning("pruned %s from %d to %d rows",
                           path.name, len(lines), len(keep))
            trimmed = True
        except OSError as exc:
            logger.error("could not prune %s: %s", path, exc)
    return trimmed


def standard_remedies(service: str = "memecoin-shadow.service",
                      *, root: Optional[Path] = None,
                      status_base: str = "http://127.0.0.1:18080",
                      trainer: str = "memecoin-shadow-trainer.service"
                      ) -> List[Remedy]:
    """The fixed repertoire.

    Each entry names a fault the node can genuinely correct by restarting
    something. Anything not listed here is reported and left alone -- a
    supervisor that improvises against a fault it cannot name will eventually
    improvise against a fault that was not there.

    Four checks are CRITICAL and deliberately have no entry here, because the
    only available action would destroy the evidence:

    `safety_kill_switch` -- a supervisor that clears the kill switch has
    removed the one thing standing between a losing day and a worse one.

    `kernel_decision` and `kernel_transaction` -- a demoted kernel is a real
    disagreement between two implementations. Restarting clears the demotion,
    re-shadows, and re-promotes on the next run of agreements, which is a
    supervisor laundering a correctness fault into a fresh start. The
    disagreement is the finding and a human has to read it.

    `execution_failure_rate` -- a high failure rate is the market telling you
    something, and restarting hides the message.

    `subsystem_latency` warns when nothing has been traced. There is no fixer
    because the ledger needs traffic, and no restart produces traffic.
    """
    return [
        Remedy(
            name="restart_on_dead_desk",
            applies=lambda health: _critical(health, "readiness_freshness"),
            act=lambda: systemctl("restart", service),
            why="the desk stopped writing readiness; it is wedged or gone"),
        Remedy(
            name="restart_on_silent_stream",
            applies=lambda health: _critical(
                health, "pipeline_stream_delivery", "pipeline_stream_events"),
            act=lambda: systemctl("restart", service),
            why=("the chain stream is connected and delivering nothing; a "
                 "reconnect is the only thing this node can do about it")),
        Remedy(
            name="restart_on_stalled_census",
            applies=lambda health: _critical(health, "pipeline_census"),
            act=lambda: systemctl("restart", service),
            why="launches stopped being counted; the denominator is frozen"),
        Remedy(
            name="restart_on_dead_feed",
            applies=lambda health: _critical(health, "feed_yellowstone"),
            act=lambda: systemctl("restart", service),
            why="the feed is reported dead by the monitor"),
        Remedy(
            name="flush_stale_ledgers",
            applies=lambda health: _critical(
                health, "persistence_evidence", "persistence_census"),
            act=lambda: post(f"{status_base}/flush"),
            why=("the evidence ledgers have gone stale; a restart here would "
                 "lose exactly what the check is worried about, so force a "
                 "save instead"),
            # Cheap and non-destructive, so it may run more often than a
            # restart and needs almost no cooldown.
            budget=10, cooldown_s=60.0),
        Remedy(
            name="restart_on_stale_market_data",
            applies=lambda health: _critical(health, "data_market_observations"),
            act=lambda: systemctl("restart", service),
            why=("market observations stopped arriving; episodes are being "
                 "built blind")),
        Remedy(
            name="restart_on_silent_miners",
            applies=lambda health: _warn(health, "subsystem_miners"),
            act=lambda: systemctl("restart", service),
            why=("the miner pool has gone quiet; the context that explains a "
                 "price path is not being collected"),
            # A warning, not a crisis. Acted on rarely and slowly, so a
            # briefly rate-limited miner never causes a restart.
            budget=1, cooldown_s=3_600.0),
        Remedy(
            name="free_disk_space",
            applies=lambda health: _critical(health, "resource_disk"),
            act=lambda: prune_spill(root or Path.cwd()),
            why=("the disk is nearly full; trim the append-only spill logs "
                 "before it stops the evidence ledgers being written"),
            budget=4, cooldown_s=600.0),
        Remedy(
            name="substitute_blocked_sources",
            applies=lambda health: _critical(health, "breadth_substitution"),
            act=lambda: post(f"{status_base}/release-sources"),
            why=("every rung of a data domain is quarantined at once, which "
                 "is almost always one shared cause -- this address rate "
                 "limited everywhere, DNS wobbling, an outbound proxy blip -- "
                 "rather than every operator independently dying; lift the "
                 "penalties and let the ladder re-sort itself rather than "
                 "waiting out four separate timers for a cause that has "
                 "already passed"),
            # Cheap, non-destructive and idempotent: it clears timers and
            # nothing else. If the cause has not passed the rungs simply fail
            # again and re-quarantine, so it can run often.
            budget=12, cooldown_s=120.0),
        Remedy(
            name="reseed_channel_discovery",
            applies=lambda health: _warn(health, "breadth_telegram"),
            act=lambda: post(f"{status_base}/verify-channels"),
            why=("the public-Telegram side has no verified channel or has "
                 "gone silent; run the verification pass now rather than "
                 "waiting for its hourly slot"),
            budget=4, cooldown_s=900.0),
        Remedy(
            name="restart_on_dead_miner_thread",
            applies=lambda health: _critical(health, "runtime_miner_thread"),
            act=lambda: systemctl("restart", service),
            why=("the miner thread died; the pool keeps reporting its last "
                 "numbers, so every other signal says the miners are healthy "
                 "while nothing is being collected"),
            budget=3, cooldown_s=600.0),
        Remedy(
            name="flush_decision_corpus",
            applies=lambda health: _critical(health, "persistence_corpus"),
            act=lambda: post(f"{status_base}/flush"),
            why=("corpus writes are failing and this is the one file that "
                 "cannot be rebuilt by re-running anything; force a write "
                 "rather than restarting into the same buffer"),
            budget=6, cooldown_s=120.0),
        Remedy(
            name="retrain_stale_models",
            applies=lambda health: _warn(
                health, "model_rug_hazard", "model_exit_policy",
                "model_prediction"),
            act=lambda: systemctl("start", trainer),
            why="a model artefact is stale; the trainer already exists to fix that",
            budget=2, cooldown_s=7_200.0),
    ]
