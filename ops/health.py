"""Continuous local health checks. No model, no network, no Claude.

The thing that keeps a trading node alive has to run every few minutes,
forever, on the node itself. Anything that depends on a person reading logs or
on a language model being invoked is not a health check -- it is a hope that
someone looks.

Each check returns one of four states, and the distinction between the last
two carries most of the value:

    OK            measured, and fine
    WARN          measured, and heading the wrong way
    CRITICAL      measured, and broken now
    DATA_BLOCKED  could not be measured at all

A monitor that reports a check it could not run as OK is worse than having no
monitor, because it manufactures confidence exactly where visibility was lost.
A stale readiness file is the canonical case: the desk may be fine, or the
process may have died twenty minutes ago, and those look identical from the
file alone. So freshness is checked first and separately, and every check that
reads a stale snapshot degrades to DATA_BLOCKED rather than reporting the last
known-good values as current.

Checks are ordered by what they protect. Safety state comes first: a node that
has quietly lost its live-capital lock is a more urgent finding than one whose
disk is filling, however much noisier the disk would be.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HEALTH_SCHEMA_VERSION = "v1"


class State(Enum):
    OK = "OK"
    WARN = "WARN"
    CRITICAL = "CRITICAL"
    DATA_BLOCKED = "DATA_BLOCKED"

    @property
    def rank(self) -> int:
        return {"OK": 0, "DATA_BLOCKED": 1, "WARN": 2, "CRITICAL": 3}[self.value]


@dataclass
class Check:
    name: str
    state: State
    detail: str
    # Anything a weekly audit would need to judge the finding without going
    # back to raw logs.
    evidence: Dict[str, Any] = field(default_factory=dict)
    # Set when this finding should wake a human or an audit immediately rather
    # than waiting for the weekly pack.
    escalate: bool = False


@dataclass
class HealthThresholds:
    """Every threshold in one place, so none of them is a magic number inline."""

    readiness_stale_seconds: float = 300.0
    feed_stale_seconds: float = 120.0
    market_observation_stale_seconds: float = 900.0
    disk_warn_pct: float = 80.0
    disk_critical_pct: float = 92.0
    memory_warn_pct: float = 80.0
    memory_critical_pct: float = 92.0
    max_execution_failure_rate: float = 0.35
    min_execution_attempts_for_verdict: int = 20
    training_stale_seconds: float = 172_800.0
    max_data_blocked_token_share: float = 0.50
    # A stream that opened and delivers nothing looked healthy for hours. So
    # did a router filing every update as the wrong type, and a decoder that
    # never saw the instruction carrying launches. Each of those is a check
    # now, because each produced an empty denominator behind a green status.
    stream_silent_seconds: float = 180.0
    census_stall_seconds: float = 900.0
    min_dry_build_success: float = 0.90
    miner_silent_seconds: float = 1_800.0
    min_miners_producing: int = 3
    evidence_stale_seconds: float = 600.0
    max_source_dead_share: float = 0.60
    landing_log_stale_seconds: float = 86_400.0


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, "file does not exist"
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"


def check_readiness_freshness(path: Path, now: float,
                              thresholds: HealthThresholds) -> Check:
    """Is the desk still writing? Everything downstream depends on this answer.

    Checked first and separately because a stale snapshot and a healthy one are
    indistinguishable from their contents alone.
    """
    if not path.exists():
        return Check("readiness_freshness", State.DATA_BLOCKED,
                     f"no readiness snapshot at {path}",
                     {"path": str(path)}, escalate=True)
    age = now - path.stat().st_mtime
    evidence = {"path": str(path), "age_seconds": round(age, 1),
                "threshold_seconds": thresholds.readiness_stale_seconds}
    if age > thresholds.readiness_stale_seconds:
        return Check("readiness_freshness", State.CRITICAL,
                     f"readiness snapshot is {age:.0f}s old; the desk may not be running",
                     evidence, escalate=True)
    return Check("readiness_freshness", State.OK,
                 f"readiness snapshot {age:.0f}s old", evidence)


def check_safety_state(readiness: Dict[str, Any]) -> List[Check]:
    """The controls that bound ruin. Reported before anything noisier."""
    checks: List[Check] = []
    locked = readiness.get("live_submission_locked")
    dry_run = (readiness.get("execution") or {}).get("dry_run")
    mode = readiness.get("mode")

    if locked is None or dry_run is None:
        checks.append(Check("safety_live_lock", State.DATA_BLOCKED,
                            "readiness does not report the live-capital lock",
                            {"mode": mode}, escalate=True))
    elif not locked and dry_run:
        # Not an error, but worth stating plainly: the acknowledgement is
        # present and only the dry-run flag is holding submission back.
        checks.append(Check("safety_live_lock", State.WARN,
                            "ALLOW_LIVE_TRADING is acknowledged; only dry_run blocks submission",
                            {"mode": mode, "dry_run": dry_run}, escalate=True))
    elif not locked and not dry_run:
        checks.append(Check("safety_live_lock", State.WARN,
                            "live submission is UNLOCKED and dry_run is off",
                            {"mode": mode}, escalate=True))
    else:
        checks.append(Check("safety_live_lock", State.OK,
                            "live submission locked", {"mode": mode}))

    portfolio = readiness.get("portfolio") or {}
    if "kill_switch_active" not in portfolio:
        checks.append(Check("safety_kill_switch", State.DATA_BLOCKED,
                            "portfolio state does not report the kill switch"))
    elif portfolio.get("kill_switch_active"):
        checks.append(Check(
            "safety_kill_switch", State.CRITICAL,
            "daily-loss or giveback kill switch is active; the book is halted",
            {"daily_pnl": portfolio.get("daily_pnl"),
             "max_daily_loss": portfolio.get("max_daily_loss"),
             "daily_giveback_floor": portfolio.get("daily_giveback_floor"),
             "daily_peak_pnl": portfolio.get("daily_peak_pnl")},
            escalate=True))
    else:
        checks.append(Check("safety_kill_switch", State.OK, "kill switch not active",
                            {"daily_pnl": portfolio.get("daily_pnl")}))
    return checks


def check_feeds(readiness: Dict[str, Any], thresholds: HealthThresholds) -> List[Check]:
    """Streams and their freshness. A silent feed is the most expensive outage."""
    checks: List[Check] = []
    yellowstone = readiness.get("yellowstone") or {}
    status = str(yellowstone.get("status", "UNKNOWN"))
    if status == "STREAMING":
        checks.append(Check("feed_yellowstone", State.OK, "streaming",
                            {"status": status}))
    elif status in {"NOT_STARTED", "UNKNOWN"}:
        checks.append(Check("feed_yellowstone", State.DATA_BLOCKED,
                            f"yellowstone status is {status}", {"status": status}))
    else:
        checks.append(Check("feed_yellowstone", State.CRITICAL,
                            f"yellowstone is not streaming: {status}",
                            {"status": status, "detail": yellowstone.get("detail")},
                            escalate=True))

    program_stream = readiness.get("rpc_program_stream")
    if program_stream is None:
        checks.append(Check("feed_rpc_program_stream", State.DATA_BLOCKED,
                            "no RPC program stream configured"))
    else:
        stream_status = str(program_stream.get("status", "UNKNOWN"))
        checks.append(Check(
            "feed_rpc_program_stream",
            State.OK if stream_status in {"STREAMING", "OK"} else State.WARN,
            f"rpc program stream {stream_status}", {"status": stream_status}))
    return checks


def check_data_freshness(readiness: Dict[str, Any], now: float,
                         thresholds: HealthThresholds) -> Check:
    """Are market observations still arriving into the research lake?"""
    dataset = readiness.get("dataset") or {}
    observed_at = dataset.get("market_observed_at")
    if not observed_at:
        return Check("data_market_observations", State.DATA_BLOCKED,
                     "dataset reports no market observation timestamp",
                     {"active_episodes": dataset.get("active_episodes")})
    age = now - float(observed_at)
    evidence = {"age_seconds": round(age, 1),
                "active_episodes": dataset.get("active_episodes"),
                "completed_episodes": dataset.get("completed_episodes"),
                "indexed_outcomes": dataset.get("indexed_outcomes"),
                "sources": dataset.get("market_sources")}
    if age > thresholds.market_observation_stale_seconds:
        return Check("data_market_observations", State.CRITICAL,
                     f"no market observation recorded for {age / 60:.0f} minutes; "
                     "the moat is not growing",
                     evidence, escalate=True)
    return Check("data_market_observations", State.OK,
                 f"last market observation {age:.0f}s ago", evidence)


def check_models(readiness: Dict[str, Any], model_dir: Path, now: float,
                 thresholds: HealthThresholds) -> List[Check]:
    """Model presence, training recency and artifact integrity.

    A model file that changed without a training run is the finding this
    exists to catch: it means an artifact was replaced by something other
    than the promotion path.
    """
    checks: List[Check] = []
    prediction = readiness.get("prediction")
    checks.append(Check(
        "model_prediction", State.OK if prediction == "OK" else State.DATA_BLOCKED,
        f"prediction model {prediction}", {"status": prediction}))

    hazard = readiness.get("rug_hazard") or {}
    checks.append(Check(
        "model_rug_hazard",
        State.OK if hazard.get("model_trained") else State.DATA_BLOCKED,
        f"rug hazard {hazard.get('model_status', 'UNKNOWN')}",
        {"status": hazard.get("model_status"),
         "detail": hazard.get("model_status_detail"),
         "tracked_tokens": hazard.get("tracked_tokens"),
         "data_blocked_tokens": hazard.get("data_blocked_tokens")}))

    exit_policy = readiness.get("exit_policy") or {}
    checks.append(Check(
        "model_exit_policy",
        State.OK if exit_policy.get("status") == "OK" else State.DATA_BLOCKED,
        f"exit policy {exit_policy.get('status', 'UNKNOWN')}",
        {"detail": exit_policy.get("detail")}))

    for label, filename in (("shadow", "last_training_report.json"),
                            ("hazard", "last_hazard_training_report.json"),
                            ("exit_policy", "last_exit_policy_report.json")):
        path = model_dir / filename
        report, error = _read_json(path)
        if report is None:
            checks.append(Check(f"training_{label}", State.DATA_BLOCKED,
                                f"{filename} {error}", {"path": str(path)}))
            continue
        age = now - path.stat().st_mtime
        state = State.OK
        if age > thresholds.training_stale_seconds:
            state = State.WARN
        checks.append(Check(
            f"training_{label}", state,
            f"last {label} training {age / 3600:.1f}h ago, status "
            f"{report.get('status', 'UNKNOWN')}",
            {"age_seconds": round(age, 1), "status": report.get("status"),
             "detail": report.get("detail") or report.get("reason")}))
    return checks


def check_pipeline(readiness: Dict[str, Any], now: float,
                   thresholds: HealthThresholds,
                   previous: Optional[Dict[str, Any]] = None) -> List[Check]:
    """The chain from stream to denominator, checked at every joint.

    Three separate defects produced an empty launch census behind a passing
    status: a stream reporting a connection as delivery, a router filing every
    update as the first field name in a list, and a decoder that never saw the
    CPI-wrapped instruction carrying launches. None of them was visible as a
    failure anywhere. Each is a check here now, at the joint where it broke.
    """
    checks: List[Check] = []

    stream = readiness.get("stream_events") or {}
    total = stream.get("total") or 0
    creations = stream.get("token_created") or 0
    if not total:
        checks.append(Check(
            "pipeline_stream_events", State.CRITICAL,
            "no chain event has reached the desk; nothing downstream can fill",
            evidence={"total": total}, escalate=True))
    elif not creations:
        checks.append(Check(
            "pipeline_stream_events", State.CRITICAL,
            f"{total} events delivered and not one creation; the decoder is "
            "naming trades and missing launches",
            evidence={"total": total, "by_type": stream.get("by_type")},
            escalate=True))
    else:
        checks.append(Check("pipeline_stream_events", State.OK,
                            f"{creations} creations of {total} events",
                            evidence={"token_created": creations}))

    yellow = readiness.get("yellowstone") or {}
    quiet = yellow.get("seconds_since_response")
    if yellow.get("status") == "DATA_BLOCKED":
        checks.append(Check(
            "pipeline_stream_delivery", State.CRITICAL,
            yellow.get("detail") or "the stream is connected and silent",
            evidence=dict(yellow), escalate=True))
    elif quiet is not None and quiet > thresholds.stream_silent_seconds:
        checks.append(Check(
            "pipeline_stream_delivery", State.CRITICAL,
            f"no stream response for {quiet:.0f}s",
            evidence={"seconds_since_response": quiet}, escalate=True))
    else:
        checks.append(Check("pipeline_stream_delivery", State.OK, "",
                            evidence={"responses": yellow.get("responses")}))

    decoder = readiness.get("pump_decoder") or {}
    if decoder.get("status") == "DEGRADED":
        checks.append(Check(
            "pipeline_decoder", State.CRITICAL, decoder.get("detail", ""),
            evidence={"unmatched": decoder.get("unmatched_prefixes"),
                      "matched": decoder.get("matched")}, escalate=True))
    elif decoder.get("status") == "OK":
        checks.append(Check("pipeline_decoder", State.OK, "",
                            evidence={"matched": decoder.get("matched")}))

    census = ((readiness.get("launch_census") or {}).get("funnel") or {})
    seen = census.get("seen")
    if seen is None:
        checks.append(Check("pipeline_census", State.DATA_BLOCKED,
                            "the census has not reported"))
    elif previous is not None:
        before = ((previous.get("launch_census") or {}).get("funnel") or {}).get("seen")
        elapsed = now - float(previous.get("_observed_at", now))
        if (before is not None and seen <= before
                and elapsed > thresholds.census_stall_seconds):
            checks.append(Check(
                "pipeline_census", State.CRITICAL,
                f"the denominator has not moved in {elapsed:.0f}s; launches "
                "are arriving and not being counted, or none are arriving",
                evidence={"seen": seen, "previous": before}, escalate=True))
        else:
            checks.append(Check("pipeline_census", State.OK, "",
                                evidence={"seen": seen}))
    else:
        checks.append(Check("pipeline_census", State.OK, "",
                            evidence={"seen": seen}))

    build = readiness.get("dry_build") or {}
    rate = build.get("success_rate")
    if rate is not None and rate < thresholds.min_dry_build_success:
        checks.append(Check(
            "pipeline_execution_build", State.CRITICAL,
            f"only {rate:.0%} of transactions build; with capital these are "
            "rejected trades",
            evidence={"failures": build.get("failures")}, escalate=True))
    elif rate is not None:
        checks.append(Check("pipeline_execution_build", State.OK, "",
                            evidence={"built": build.get("built")}))
    return checks


def check_subsystems(readiness: Dict[str, Any], root: Path, now: float,
                     thresholds: HealthThresholds) -> List[Check]:
    """Everything that can degrade without stopping the process.

    A desk does not usually fail by crashing. It fails by continuing to run
    with one subsystem quietly producing nothing, which is why every one of
    these is checked separately rather than inferred from the process being
    alive.
    """
    checks: List[Check] = []

    miners = readiness.get("data_miners") or {}
    producing = miners.get("producing")
    if producing is None:
        checks.append(Check("subsystem_miners", State.DATA_BLOCKED,
                            "the miner pool has not reported"))
    elif producing < thresholds.min_miners_producing:
        checks.append(Check(
            "subsystem_miners", State.WARN,
            f"only {producing} miner(s) producing; the context that explains "
            "a price path is not being collected",
            evidence={"silent": miners.get("silent"),
                      "awaiting": miners.get("awaiting_credentials")}))
    else:
        checks.append(Check("subsystem_miners", State.OK, "",
                            evidence={"producing": producing,
                                      "records": miners.get("total_records")}))

    memory = readiness.get("memory") or {}
    band = memory.get("band")
    if band == "shed":
        checks.append(Check(
            "subsystem_memory", State.CRITICAL,
            "the governor is shedding context to stay inside the ceiling; "
            "this host is undersized for this workload",
            evidence=dict(memory), escalate=True))
    elif band == "trim":
        checks.append(Check("subsystem_memory", State.WARN,
                            "trimming caches under memory pressure",
                            evidence=dict(memory)))
    elif band == "unmeasured":
        checks.append(Check("subsystem_memory", State.DATA_BLOCKED,
                            "the footprint cannot be read on this host"))
    else:
        checks.append(Check("subsystem_memory", State.OK, "",
                            evidence={"fraction": memory.get("fraction")}))

    signer = readiness.get("signer") or {}
    if signer.get("halted") and signer.get("isolated"):
        checks.append(Check(
            "subsystem_signer", State.CRITICAL,
            f"the isolated signer is halted: {signer.get('halt_reason', '')}",
            evidence=dict(signer), escalate=True))
    elif signer.get("mode") == "local":
        checks.append(Check(
            "subsystem_signer", State.WARN,
            "the private key is held in the trading process; correct for "
            "shadow, wrong for capital",
            evidence={"mode": "local"}))
    elif signer.get("mode"):
        checks.append(Check("subsystem_signer", State.OK, "",
                            evidence={"mode": signer.get("mode")}))

    facts = readiness.get("fact_ladder") or {}
    degraded = facts.get("degraded_facts") or []
    if degraded:
        checks.append(Check(
            "subsystem_fact_ladder", State.WARN,
            "these facts are usually inferred rather than read: "
            + ", ".join(degraded[:5]),
            evidence={"degraded": degraded}))

    calibration = readiness.get("calibration") or {}
    if calibration.get("models_miscalibrated"):
        checks.append(Check(
            "subsystem_calibration", State.WARN,
            calibration.get("detail", "a model is miscalibrated"),
            evidence={"count": calibration.get("models_miscalibrated")}))

    conditions = readiness.get("execution_conditions") or {}
    if conditions.get("status") == "DATA_BLOCKED":
        checks.append(Check(
            "subsystem_execution_conditions", State.WARN,
            "bids are being made against an unknown congestion bucket",
            evidence={}))

    for name, filename, stale in (
            ("evidence", "forward_evidence.json", thresholds.evidence_stale_seconds),
            ("census", "launch_census.json", thresholds.evidence_stale_seconds)):
        path = root / "data" / "state" / filename
        if not path.exists():
            checks.append(Check(f"persistence_{name}", State.DATA_BLOCKED,
                                f"{filename} has never been written"))
            continue
        age = now - path.stat().st_mtime
        if age > stale:
            checks.append(Check(
                f"persistence_{name}", State.CRITICAL,
                f"{filename} has not been written for {age:.0f}s; a restart "
                "now would lose everything since",
                evidence={"age_s": age}, escalate=True))
        else:
            checks.append(Check(f"persistence_{name}", State.OK, "",
                                evidence={"age_s": round(age, 1)}))
    return checks


def check_kernels(readiness: Dict[str, Any]) -> List[Check]:
    """The two implementations that can quietly stop being the fast one.

    A demoted kernel is the most dangerous silent state this desk has. It
    keeps trading, keeps reporting healthy, and has secretly moved back to
    the slow implementation because the fast one disagreed once. Nothing else
    surfaces that -- the trades still happen and the status still says OK --
    so it is checked here and it is CRITICAL.

    Deliberately no fixer. A restart clears the demotion, re-shadows, and
    re-promotes on the next run of agreements, which is a supervisor
    laundering a real disagreement into a fresh start. The disagreement is
    the finding; a human has to look at it.
    """
    checks: List[Check] = []
    for name, section, what in (
        ("kernel_decision", "t0_kernel", "the T0 decision kernel"),
        ("kernel_transaction", "tx_kernel", "the transaction builder"),
    ):
        report = readiness.get(section) or {}
        if not report:
            checks.append(Check(name, State.DATA_BLOCKED,
                                f"{what} has not reported"))
            continue
        demoted = report.get("demoted_reason") or ""
        if demoted:
            checks.append(Check(
                name, State.CRITICAL,
                f"{what} DEMOTED and is running on the slow implementation: "
                f"{demoted}. A restart would clear this and hide it.",
                evidence={"divergences": report.get("divergences"),
                          "example": report.get("divergence_example")
                                     or report.get("divergence_examples")},
                escalate=True))
        elif report.get("invariant_failures"):
            checks.append(Check(
                name, State.CRITICAL,
                f"{what} produced {report['invariant_failures']} structurally "
                "invalid output(s); these would have been signed",
                escalate=True))
        else:
            checks.append(Check(name, State.OK, "", evidence={
                "authoritative": ("rust" if report.get("rust_authoritative")
                                  else "python"),
                "agreements": report.get("consecutive_agreements")}))

    # A promoted decision whose Python check was dropped is never verified.
    # Not a crisis on its own -- it is a check that did not happen, not a
    # disagreement -- but a rising count means parity has stopped guarding.
    kernel = readiness.get("t0_kernel") or {}
    unverified = kernel.get("parity_unverified")
    if unverified is None:
        checks.append(Check("kernel_parity_coverage", State.DATA_BLOCKED,
                            "parity coverage has not reported"))
    elif unverified:
        checks.append(Check(
            "kernel_parity_coverage", State.WARN,
            f"{unverified} promoted decision(s) were never verified against "
            "python; the parity worker is not keeping up",
            evidence={"pending": kernel.get("parity_pending")}))
    else:
        checks.append(Check("kernel_parity_coverage", State.OK, "",
                            evidence={"checked": kernel.get("parity_checked")}))
    return checks


def check_runtime(readiness: Dict[str, Any]) -> List[Check]:
    """The moving parts that keep the desk collecting rather than deciding."""
    checks: List[Check] = []

    offload = readiness.get("miner_offload") or {}
    status = offload.get("status")
    if not offload:
        checks.append(Check("runtime_miner_thread", State.DATA_BLOCKED,
                            "the miner offload has not reported"))
    elif status == "CRITICAL":
        # The pool's own report keeps its last numbers, so a dead thread looks
        # like healthy miners from every other angle.
        checks.append(Check(
            "runtime_miner_thread", State.CRITICAL,
            offload.get("detail") or "the miner thread is not alive; miners "
                                     "are dark and the pool report is stale",
            escalate=True))
    elif status == "DEGRADED":
        checks.append(Check("runtime_miner_thread", State.WARN,
                            offload.get("detail", ""),
                            evidence={"dropped": offload.get("dropped")}))
    else:
        checks.append(Check("runtime_miner_thread", State.OK, "",
                            evidence={"mode": status,
                                      "delivered": offload.get("delivered")}))

    corpus = readiness.get("decision_corpus") or {}
    if not corpus:
        checks.append(Check("persistence_corpus", State.DATA_BLOCKED,
                            "the decision corpus has not reported"))
    elif corpus.get("write_failures"):
        # This is the one file that cannot be rebuilt by re-running anything.
        checks.append(Check(
            "persistence_corpus", State.CRITICAL,
            f"{corpus['write_failures']} corpus write(s) failed: "
            f"{corpus.get('last_error', '')}. Decisions are being lost and "
            "cannot be reconstructed from anything else.",
            escalate=True))
    elif corpus.get("status") == "DEGRADED":
        checks.append(Check("persistence_corpus", State.WARN,
                            corpus.get("detail", ""),
                            evidence={"ignore_share": corpus.get("ignore_share")}))
    else:
        checks.append(Check("persistence_corpus", State.OK, "",
                            evidence={"recorded": corpus.get("recorded"),
                                      "resolved": corpus.get("resolved")}))

    latency = readiness.get("latency") or {}
    if not latency:
        checks.append(Check("subsystem_latency", State.DATA_BLOCKED,
                            "the latency ledger has not reported"))
    elif latency.get("status") == "DATA_BLOCKED":
        # A warning rather than a fault, and deliberately unfixable: a ledger
        # with no traces needs traffic, and no restart produces traffic.
        checks.append(Check(
            "subsystem_latency", State.WARN,
            "nothing has been traced; every speed claim about this desk is "
            "currently unverified"))
    else:
        checks.append(Check("subsystem_latency", State.OK, "", evidence={
            "slowest_ours": latency.get("dominant_controllable_stage"),
            "unmeasured": latency.get("unmeasured_stages")}))
    return checks


def check_breadth(readiness: Dict[str, Any]) -> List[Check]:
    """Whether the global data universe is actually answering.

    Four separate questions, deliberately not merged. A substituted domain is
    healthy -- that is the ladder working. A DARK domain is a question the
    desk asks continuously and cannot answer. A silent channel book is a
    Telegram side that has stopped reading. A discovery gap is a hole in our
    own decoder, and it is the one that looks like a calm market from inside.
    """
    checks: List[Check] = []

    substitution = readiness.get("substitution") or {}
    dark = substitution.get("dark") or []
    substituted = substitution.get("substituted") or []
    if not substitution:
        checks.append(Check("breadth_substitution", State.DATA_BLOCKED,
                            "the substitution registry has not reported"))
    elif dark:
        checks.append(Check(
            "breadth_substitution", State.CRITICAL,
            "no endpoint left for: " + ", ".join(dark[:6])
            + "; these questions currently have no answer at all",
            evidence={"dark": dark, "substitutions": substitution.get("substitutions")},
            escalate=True))
    elif substituted:
        # Not a warning. This is the ladder doing exactly its job, and paging
        # on it would train an operator to ignore the page that matters.
        checks.append(Check("breadth_substitution", State.OK, "",
                            evidence={"running_on_substitute": substituted[:8]}))
    else:
        checks.append(Check("breadth_substitution", State.OK, "",
                            evidence={"domains": substitution.get("domains")}))

    coverage = (substitution.get("coverage") or {})
    declared = coverage.get("regions_declared")
    proven = coverage.get("regions_proven")
    if declared is None or proven is None:
        checks.append(Check("breadth_regions", State.DATA_BLOCKED,
                            "regional coverage has not been measured"))
    elif proven == 0 and declared:
        checks.append(Check(
            "breadth_regions", State.WARN,
            f"{declared} regions declared and none has returned a record; "
            "the breadth is configuration, not coverage",
            evidence={"unproven": coverage.get("unproven_regions")}))
    else:
        checks.append(Check("breadth_regions", State.OK, "",
                            evidence={"proven": proven, "declared": declared}))

    telegram = readiness.get("telegram_channels") or {}
    status = telegram.get("status")
    if status == "DATA_BLOCKED":
        checks.append(Check(
            "breadth_telegram", State.WARN,
            telegram.get("detail")
            or "no verified public Telegram channel; discovery has not converged",
            evidence={"candidates": telegram.get("candidates"),
                      "rejected": telegram.get("rejected")}))
    elif status == "DEGRADED":
        checks.append(Check("breadth_telegram", State.WARN,
                            telegram.get("detail") or "verified channels have gone silent",
                            evidence={"silent": (telegram.get("silent") or [])[:10]}))
    elif not telegram:
        checks.append(Check("breadth_telegram", State.DATA_BLOCKED,
                            "the channel book has not reported"))
    else:
        checks.append(Check("breadth_telegram", State.OK, "",
                            evidence={"verified": telegram.get("verified"),
                                      "mints_seen": telegram.get("mints_seen")}))

    discovery = readiness.get("discovery") or {}
    if not discovery or discovery.get("status") == "DATA_BLOCKED":
        checks.append(Check("breadth_discovery", State.DATA_BLOCKED,
                            "no external pool discovery yet this run"))
    elif discovery.get("status") == "DEGRADED":
        checks.append(Check(
            "breadth_discovery", State.WARN, discovery.get("detail", ""),
            evidence={"missed": discovery.get("missed_by_our_stream"),
                      "seen": discovery.get("external_pools_seen")}))
    else:
        checks.append(Check("breadth_discovery", State.OK, "",
                            evidence={"coverage": discovery.get("coverage")}))

    router = readiness.get("landing_router") or {}
    if not router:
        checks.append(Check("breadth_landing_routes", State.DATA_BLOCKED,
                            "the landing router has not reported"))
    elif router.get("status") == "DATA_BLOCKED":
        checks.append(Check(
            "breadth_landing_routes", State.CRITICAL,
            router.get("detail") or "no landing route is enabled; a signed "
                                    "transaction has nowhere to go",
            escalate=True))
    elif router.get("status") == "DEGRADED":
        checks.append(Check(
            "breadth_landing_routes", State.WARN, router.get("detail", ""),
            evidence={"mechanisms": router.get("mechanisms"),
                      "enabled": router.get("enabled")}))
    else:
        checks.append(Check("breadth_landing_routes", State.OK, "",
                            evidence={"mechanisms": router.get("mechanisms"),
                                      "measured": router.get("measured_routes")}))

    identity = readiness.get("identity_watch") or {}
    if not identity:
        checks.append(Check("breadth_identity", State.DATA_BLOCKED,
                            "the figure registry has not reported"))
    elif identity.get("status") == "DEGRADED":
        checks.append(Check(
            "breadth_identity", State.WARN, identity.get("detail", ""),
            evidence={"figures": identity.get("figures"),
                      "with_channels": identity.get("figures_with_channels")}))
    else:
        checks.append(Check("breadth_identity", State.OK, "",
                            evidence={"figures": identity.get("figures"),
                                      "claims_found": identity.get("claims_found")}))
    return checks


def check_resources(root: Path, thresholds: HealthThresholds) -> List[Check]:
    """Disk and memory. Measured, and DATA_BLOCKED where the platform hides them."""
    checks: List[Check] = []
    try:
        usage = shutil.disk_usage(root)
        used_pct = usage.used / usage.total * 100.0
        state = State.OK
        if used_pct >= thresholds.disk_critical_pct:
            state = State.CRITICAL
        elif used_pct >= thresholds.disk_warn_pct:
            state = State.WARN
        checks.append(Check("resource_disk", state, f"disk {used_pct:.1f}% used",
                            {"used_pct": round(used_pct, 1),
                             "free_gb": round(usage.free / 1024 ** 3, 2)},
                            escalate=state is State.CRITICAL))
    except OSError as exc:
        checks.append(Check("resource_disk", State.DATA_BLOCKED, f"disk unreadable: {exc}"))

    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        checks.append(Check("resource_memory", State.DATA_BLOCKED,
                            "/proc/meminfo not present on this platform"))
        return checks
    try:
        values = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key.strip()] = float(rest.strip().split()[0])
        total = values.get("MemTotal", 0.0)
        available = values.get("MemAvailable", 0.0)
        if total <= 0:
            raise ValueError("MemTotal missing")
        used_pct = (1.0 - available / total) * 100.0
        state = State.OK
        if used_pct >= thresholds.memory_critical_pct:
            state = State.CRITICAL
        elif used_pct >= thresholds.memory_warn_pct:
            state = State.WARN
        checks.append(Check("resource_memory", state, f"memory {used_pct:.1f}% used",
                            {"used_pct": round(used_pct, 1),
                             "available_mb": round(available / 1024, 1),
                             "total_mb": round(total / 1024, 1)},
                            escalate=state is State.CRITICAL))
    except (OSError, ValueError, IndexError) as exc:
        checks.append(Check("resource_memory", State.DATA_BLOCKED,
                            f"memory unreadable: {exc}"))
    return checks


def check_execution(execution_log: Path, now: float, window_seconds: float,
                    thresholds: HealthThresholds) -> Check:
    """Recent fill quality, from the desk's own execution attempts.

    A failure rate is only a verdict above a minimum attempt count. Three
    failures out of four is not a 75% failure rate, it is four observations,
    and acting on it retires a working route on noise.
    """
    if not execution_log.exists():
        return Check("execution_failure_rate", State.DATA_BLOCKED,
                     f"no execution log at {execution_log}")
    attempts: List[Dict[str, Any]] = []
    try:
        with execution_log.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if now - float(row.get("timestamp", 0) or 0) <= window_seconds:
                    attempts.append(row)
    except OSError as exc:
        return Check("execution_failure_rate", State.DATA_BLOCKED,
                     f"execution log unreadable: {exc}")

    if len(attempts) < thresholds.min_execution_attempts_for_verdict:
        return Check("execution_failure_rate", State.DATA_BLOCKED,
                     f"{len(attempts)} attempts in window, below the "
                     f"{thresholds.min_execution_attempts_for_verdict} needed for a verdict",
                     {"attempts": len(attempts)})
    failures = sum(1 for row in attempts if not row.get("success"))
    rate = failures / len(attempts)
    reasons: Dict[str, int] = {}
    for row in attempts:
        if not row.get("success"):
            key = str(row.get("error") or row.get("status") or "unknown")
            reasons[key] = reasons.get(key, 0) + 1
    state = State.CRITICAL if rate > thresholds.max_execution_failure_rate else State.OK
    return Check("execution_failure_rate", state,
                 f"{rate:.1%} of {len(attempts)} recent attempts failed",
                 {"attempts": len(attempts), "failures": failures,
                  "rate": round(rate, 4),
                  "top_reasons": dict(sorted(reasons.items(),
                                             key=lambda item: item[1], reverse=True)[:5])},
                 escalate=state is State.CRITICAL)


def check_sources(readiness: Dict[str, Any]) -> List[Check]:
    """Social and research source health, as the collectors themselves report it."""
    checks: List[Check] = []
    social = readiness.get("social") or {}
    statuses = social.get("data_status") or {}
    blocked = {name: status for name, status in statuses.items()
               if not str(status).startswith("OK")}
    if not statuses:
        checks.append(Check("source_social", State.DATA_BLOCKED,
                            "no social collector reported a status"))
    elif blocked:
        checks.append(Check("source_social", State.WARN,
                            f"{len(blocked)} of {len(statuses)} social sources degraded",
                            {"degraded": blocked}))
    else:
        checks.append(Check("source_social", State.OK,
                            f"all {len(statuses)} social sources OK",
                            {"tracked_accounts": social.get("tracked_accounts"),
                             "total_mentions": social.get("total_mentions")}))

    research = readiness.get("research") or {}
    checks.append(Check(
        "source_research", State.OK if research else State.DATA_BLOCKED,
        "research miner reporting" if research else "research miner reported nothing",
        {k: v for k, v in list(research.items())[:8]}))
    return checks


def check_intelligence_coverage(readiness: Dict[str, Any]) -> List[Check]:
    """Whether every declared module is still reaching the decision.

    This is the check that would have caught four components being reported
    wired while their imports had silently failed to apply. A disconnected
    module raises nothing and logs nothing; the only thing that changes is
    that its slot stops appearing in decisions. So an orphan rate above zero
    is a FAIL, not a warning -- capital is being committed by a brain that is
    missing a lobe it believes it has.
    """
    checks: List[Check] = []
    coverage = readiness.get("intelligence_coverage") or {}
    if not coverage:
        return [Check("intelligence_coverage", State.DATA_BLOCKED,
                      "desk reported no coverage tracking")]
    for stage in ("entry", "position"):
        report = coverage.get(stage) or {}
        name = f"intelligence_coverage_{stage}"
        decisions = int(report.get("decisions", 0) or 0)
        if not decisions:
            checks.append(Check(name, State.DATA_BLOCKED,
                                f"no {stage} decisions observed yet"))
            continue
        orphaned = list(report.get("orphaned") or [])
        if orphaned:
            checks.append(Check(name, State.CRITICAL,
                                f"{len(orphaned)} {stage} modules never reached a decision",
                                {"orphaned": orphaned, "decisions": decisions}))
        else:
            checks.append(Check(name, State.OK,
                                f"every declared {stage} module reached all "
                                f"{decisions} decisions",
                                {"decisions": decisions}))
    return checks


def check_champions(readiness: Dict[str, Any]) -> Check:
    """Promotion state. Champions decaying without replacement is a slow failure."""
    champions = readiness.get("champions") or {}
    if not champions:
        return Check("promotion_pipeline", State.DATA_BLOCKED,
                     "champion framework reported nothing")
    live = int(champions.get("live_champions", 0) or 0)
    decaying = int(champions.get("decaying_champions", 0) or 0)
    hibernated = int(champions.get("hibernated_champions", 0) or 0)
    shadow = int(champions.get("shadow_models", 0) or 0)
    evidence = {"live": live, "decaying": decaying, "hibernated": hibernated,
                "shadow": shadow, "canary": champions.get("canary_models"),
                "total_hypotheses": champions.get("total_hypotheses")}
    if decaying and not shadow:
        return Check("promotion_pipeline", State.WARN,
                     f"{decaying} champions decaying with no shadow candidates behind them",
                     evidence)
    return Check("promotion_pipeline", State.OK,
                 f"{live} live, {shadow} shadow, {decaying} decaying", evidence)


@dataclass
class HealthReport:
    generated_at: float
    checks: List[Check]

    @property
    def worst(self) -> State:
        return max((check.state for check in self.checks), key=lambda s: s.rank,
                   default=State.DATA_BLOCKED)

    @property
    def escalations(self) -> List[Check]:
        return [check for check in self.checks if check.escalate]

    def by_state(self, state: State) -> List[Check]:
        return [check for check in self.checks if check.state is state]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": HEALTH_SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "worst_state": self.worst.value,
            "counts": {state.value: len(self.by_state(state)) for state in State},
            "escalations": [check.name for check in self.escalations],
            "checks": [
                {"name": check.name, "state": check.state.value, "detail": check.detail,
                 "evidence": check.evidence, "escalate": check.escalate}
                for check in self.checks
            ],
        }


def run_health_checks(
    readiness_path: Path,
    model_dir: Path,
    execution_log: Path,
    root: Path,
    now: Optional[float] = None,
    thresholds: Optional[HealthThresholds] = None,
    execution_window_seconds: float = 3_600.0,
) -> HealthReport:
    """Every check, in the order of what it protects."""
    now = time.time() if now is None else now
    thresholds = thresholds or HealthThresholds()
    checks: List[Check] = []

    freshness = check_readiness_freshness(readiness_path, now, thresholds)
    checks.append(freshness)

    readiness: Dict[str, Any] = {}
    if freshness.state is State.OK:
        loaded, error = _read_json(readiness_path)
        if loaded is None:
            checks.append(Check("readiness_parse", State.CRITICAL,
                                f"readiness snapshot {error}", escalate=True))
        else:
            readiness = loaded

    if readiness:
        checks.extend(check_safety_state(readiness))
        checks.extend(check_feeds(readiness, thresholds))
        checks.append(check_data_freshness(readiness, now, thresholds))
        checks.extend(check_models(readiness, model_dir, now, thresholds))
        checks.extend(check_sources(readiness))
        checks.extend(check_intelligence_coverage(readiness))
        checks.append(check_champions(readiness))
        checks.extend(check_breadth(readiness))
        checks.extend(check_kernels(readiness))
        checks.extend(check_runtime(readiness))
    else:
        # Everything that reads the snapshot degrades together, and says so.
        # Reporting these as OK would manufacture confidence exactly where
        # visibility was lost.
        for name in ("safety_live_lock", "safety_kill_switch", "feed_yellowstone",
                     "data_market_observations", "model_prediction", "promotion_pipeline"):
            checks.append(Check(name, State.DATA_BLOCKED,
                                "no usable readiness snapshot to check against"))

    checks.extend(check_resources(root, thresholds))
    checks.append(check_execution(execution_log, now, execution_window_seconds, thresholds))
    return HealthReport(generated_at=now, checks=checks)
