"""The compact weekly evidence pack.

The point of this file is what it leaves out. A weekly audit that hands over
raw logs and the whole repository is an audit that spends its attention on
retrieval instead of judgement, and it gets worse as the moat grows -- exactly
backwards, since a larger moat should make the audit sharper, not slower.

So the pack is capped, and everything in it is either an anomaly, a
diff, or a ranked summary. Three rules keep it honest:

Nothing is included because it is interesting. A section earns its bytes by
being something a reviewer could act on: a leak with a size, a check that
failed, a mechanism whose forward performance moved, a file that changed.

Absent evidence is stated, never omitted. A section with no data says so
explicitly and says why. Silently dropping an empty section makes the pack
read as though everything in it is the whole picture, and a reviewer cannot
challenge a claim that was never made.

Truncation is visible. When a list is cut to fit the cap, the pack records how
many entries were dropped. An audit that unknowingly sees the top 10 of 4,000
findings will confidently rank the wrong work first.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ops.health import HealthReport, State

logger = logging.getLogger(__name__)

AUDIT_PACK_SCHEMA_VERSION = "v1"

# Hard cap. Beyond this the pack stops being an evidence bundle and becomes the
# log dump it exists to replace.
DEFAULT_MAX_BYTES = 220_000
DEFAULT_TOP_N = 10


@dataclass
class Section:
    name: str
    status: str
    summary: str
    entries: List[Dict[str, Any]] = field(default_factory=list)
    truncated: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"status": self.status, "summary": self.summary}
        if self.entries:
            payload["entries"] = self.entries
        if self.truncated:
            # An audit that unknowingly sees the top 10 of 4,000 findings will
            # confidently rank the wrong work first.
            payload["truncated_entries"] = self.truncated
        if self.detail:
            payload["detail"] = self.detail
        return payload


def _blocked(name: str, reason: str) -> Section:
    return Section(name=name, status="DATA_BLOCKED", summary=reason)


def _take(entries: Sequence[Dict[str, Any]], limit: int) -> tuple:
    kept = list(entries[:limit])
    return kept, max(0, len(entries) - limit)


def system_integrity_section(health: Optional[HealthReport]) -> Section:
    """Bucket 1: dead feeds, stale data, broken parsers, resources, failed jobs."""
    if health is None:
        return _blocked("system_integrity", "no health report was produced this period")
    problems = [check for check in health.checks if check.state is not State.OK]
    entries = [{"check": check.name, "state": check.state.value, "detail": check.detail,
                "evidence": check.evidence, "escalated": check.escalate}
               for check in sorted(problems, key=lambda c: -c.state.rank)]
    kept, truncated = _take(entries, 25)
    return Section(
        name="system_integrity",
        status=health.worst.value,
        summary=(f"{len(problems)} of {len(health.checks)} checks not OK; "
                 f"worst state {health.worst.value}"),
        entries=kept, truncated=truncated,
    )


def wealth_leaks_section(leak_report: Optional[Dict[str, Any]],
                         top_n: int = DEFAULT_TOP_N) -> Section:
    """Bucket 2 and output 1: the exact events that cost the most log growth."""
    if not leak_report:
        return _blocked("wealth_leaks",
                        "no trade outcomes were available to attribute this period")
    worst = leak_report.get("worst_tokens") or []
    kept, truncated = _take(worst, top_n)
    total = leak_report.get("total_forgone_log_growth", 0.0)
    return Section(
        name="wealth_leaks",
        status="OK" if worst else "DATA_BLOCKED",
        summary=f"{total:.4f} total forgone log growth across {len(worst)} findings",
        entries=kept, truncated=truncated,
        detail=json.dumps({"by_leak": leak_report.get("by_leak"),
                           "share_by_leak": leak_report.get("share_by_leak"),
                           "top_causes": leak_report.get("top_causes")}),
    )


def rug_defence_section(rug_events: Optional[Sequence[Dict[str, Any]]],
                        top_n: int = DEFAULT_TOP_N) -> Section:
    """Bucket 3: rug losses, near misses, FALSE ALARMS and failed escapes.

    False alarms are in here deliberately. An audit shown only the rugs that
    were entered will keep tightening the detector, and a detector tightened
    without a view of what it wrongly rejected converges on refusing to trade.
    """
    if rug_events is None:
        return _blocked("rug_defence", "no rug outcome records were available")
    entered = [row for row in rug_events if row.get("entered")]
    false_alarms = [row for row in rug_events
                    if not row.get("entered") and not row.get("rugged")
                    and str(row.get("rejection_reason", "")).startswith("rug")]
    escapes_failed = [row for row in rug_events if row.get("escape_failed")]
    entries = [
        {"token": row.get("token"), "kind": kind,
         "realized_multiple": row.get("realized_multiple"),
         "max_feasible_multiple": row.get("max_feasible_multiple"),
         "earliest_warning_seconds": row.get("earliest_warning_seconds"),
         "exit_reason": row.get("exit_reason"),
         "rejection_reason": row.get("rejection_reason")}
        for kind, rows in (("rug_entered", entered),
                           ("false_alarm", false_alarms),
                           ("escape_failed", escapes_failed))
        for row in rows
    ]
    kept, truncated = _take(entries, top_n * 2)
    return Section(
        name="rug_defence",
        status="OK" if rug_events else "DATA_BLOCKED",
        summary=(f"{len(entered)} rugs entered, {len(false_alarms)} likely false alarms, "
                 f"{len(escapes_failed)} failed escapes"),
        entries=kept, truncated=truncated,
    )


def monster_audit_section(trades: Optional[Sequence[Dict[str, Any]]],
                          tail_report: Optional[Dict[str, Any]],
                          premature_report: Optional[Dict[str, Any]],
                          top_n: int = DEFAULT_TOP_N) -> Section:
    """Bucket 2b: missed monsters, premature exits, and the tail capture behind them."""
    if trades is None:
        return _blocked("monster_audit", "no trade records were available")
    missed = sorted(
        (row for row in trades
         if not row.get("entered") and float(row.get("max_feasible_multiple", 0) or 0) >= 10.0),
        key=lambda row: -float(row.get("max_feasible_multiple", 0) or 0))
    early = sorted(
        (row for row in trades
         if row.get("entered")
         and float(row.get("max_feasible_multiple", 0) or 0) > 0
         and float(row.get("realized_multiple", 0) or 0)
         < 0.5 * float(row.get("max_feasible_multiple", 0) or 0)),
        key=lambda row: -(float(row.get("max_feasible_multiple", 0) or 0)
                          - float(row.get("realized_multiple", 0) or 0)))
    entries = ([{"token": row.get("token"), "kind": "missed_monster",
                 "max_feasible_multiple": row.get("max_feasible_multiple"),
                 "rejection_reason": row.get("rejection_reason"),
                 "decision_features": row.get("decision_features")}
                for row in missed[:top_n]]
               + [{"token": row.get("token"), "kind": "premature_exit",
                   "realized_multiple": row.get("realized_multiple"),
                   "max_feasible_multiple": row.get("max_feasible_multiple"),
                   "exit_reason": row.get("exit_reason")}
                  for row in early[:top_n]])
    return Section(
        name="monster_audit",
        status="OK",
        summary=(f"{len(missed)} missed 10x+ launches, {len(early)} exits below half "
                 "of what was feasible"),
        entries=entries,
        truncated=max(0, len(missed) - top_n) + max(0, len(early) - top_n),
        detail=json.dumps({"tail_contribution": tail_report,
                           "premature_exit_rates": premature_report}),
    )


def execution_section(execution_stats: Optional[Dict[str, Any]]) -> Section:
    """Bucket 5: landing, latency, slippage, route quality."""
    if not execution_stats:
        return _blocked("execution",
                        "no execution telemetry was recorded this period")
    return Section(
        name="execution", status="OK",
        summary=(f"{execution_stats.get('attempts', 0)} attempts, "
                 f"{execution_stats.get('land_rate', 0):.1%} landed, "
                 f"p50 submit->land {execution_stats.get('p50_land_ms', 'n/a')}ms"),
        entries=[execution_stats],
    )


def edge_health_section(ledger: Optional[Dict[str, Any]],
                        decay: Optional[Dict[str, Any]]) -> Section:
    """Bucket 4 and output 4: which mechanisms paid, and the keep/kill table."""
    if not ledger and not decay:
        return _blocked("edge_health",
                        "no alpha ledger or decay report was available")
    verdicts = []
    for mechanism, health in (decay or {}).items():
        status = health.get("status")
        verdict = {"HEALTHY": "EXPAND", "WEAKENING": "SHADOW",
                   "DEGRADED": "HIBERNATE", "MEASURING": "KEEP"}.get(status, "KEEP")
        verdicts.append({"mechanism": mechanism, "health": status,
                         "verdict": verdict, "sample": health.get("sample"),
                         "ratio": health.get("ratio"),
                         "recent_mean_log_growth": health.get("recent_mean_log_growth")})
    verdicts.sort(key=lambda row: row["mechanism"])
    return Section(
        name="edge_health", status="OK",
        summary=(f"{len(verdicts)} mechanisms scored; "
                 f"ledger reconciles={ (ledger or {}).get('reconciles') }"),
        entries=verdicts,
        detail=json.dumps({"ledger": ledger}),
    )


def moat_section(moat_stats: Optional[Dict[str, Any]]) -> Section:
    """Bucket 6: is the private dataset actually growing, and is the node using it."""
    if not moat_stats:
        return _blocked("moat_growth", "no dataset growth statistics were available")
    return Section(
        name="moat_growth", status="OK",
        summary=(f"{moat_stats.get('episodes_added', 0)} launches recorded this period, "
                 f"{moat_stats.get('indexed_outcomes', 0)} indexed outcomes total"),
        entries=[moat_stats],
    )


def changes_section(repo_root: Path, since_days: int = 7) -> Section:
    """Output 3: what changed since last week -- code, config, models."""
    try:
        result = subprocess.run(
            ["git", "log", f"--since={since_days} days ago", "--pretty=format:%h %ad %s",
             "--date=short", "--stat"],
            # `--stat` over a long period walks every tree in the range, and
            # the audit pack runs on a loaded box while the suite is running.
            # Thirty seconds was enough on an idle machine and reported
            # DATA_BLOCKED under load, which reads as "no history" rather than
            # "ask again".
            cwd=repo_root, capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _blocked("recent_changes", f"git log failed: {exc}")
    if result.returncode != 0:
        return _blocked("recent_changes",
                        f"git log exited {result.returncode}: {result.stderr.strip()[:200]}")
    text = result.stdout.strip()
    if not text:
        return Section(name="recent_changes", status="OK",
                       summary=f"no commits in the last {since_days} days")
    lines = text.splitlines()
    commits = [line for line in lines if line and not line.startswith(" ")]
    return Section(
        name="recent_changes", status="OK",
        summary=f"{len(commits)} commits in the last {since_days} days",
        entries=[{"log": "\n".join(lines[:400])}],
        truncated=max(0, len(lines) - 400),
    )


def tests_section(repo_root: Path, python: str = ".venv/bin/python") -> Section:
    """Output: did the suite actually pass on the machine that runs the bot."""
    try:
        result = subprocess.run(
            [python, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=repo_root, capture_output=True, text=True, timeout=900, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return _blocked("tests", f"could not run the suite: {exc}")
    tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
    return Section(
        name="tests",
        status="OK" if result.returncode == 0 else "CRITICAL",
        summary=("suite passed" if result.returncode == 0
                 else f"suite FAILED (exit {result.returncode})"),
        entries=[{"tail": "\n".join(tail)}],
    )


@dataclass
class AuditPack:
    generated_at: float
    period_days: int
    sections: List[Section]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": AUDIT_PACK_SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime(self.generated_at)),
            "period_days": self.period_days,
            "sections": {section.name: section.to_dict() for section in self.sections},
        }

    def serialise(self, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
        """JSON, trimmed to the cap by dropping the largest optional detail first.

        Trimming removes `detail` blobs before it removes entries, because a
        finding a reviewer cannot see at all is worse than one they can see
        without its supporting blob. Every trim is recorded in the pack.
        """
        payload = self.to_dict()
        text = json.dumps(payload, indent=1, default=str)
        if len(text.encode()) <= max_bytes:
            return text

        trimmed: List[str] = []
        sections = payload["sections"]
        for name in sorted(sections,
                           key=lambda key: -len(str(sections[key].get("detail", "")))):
            if len(json.dumps(payload, indent=1, default=str).encode()) <= max_bytes:
                break
            if sections[name].get("detail"):
                sections[name].pop("detail")
                trimmed.append(f"{name}.detail")
        for name in sorted(sections,
                           key=lambda key: -len(str(sections[key].get("entries", [])))):
            if len(json.dumps(payload, indent=1, default=str).encode()) <= max_bytes:
                break
            entries = sections[name].get("entries") or []
            if len(entries) > 3:
                sections[name]["entries"] = entries[:3]
                sections[name]["truncated_entries"] = (
                    sections[name].get("truncated_entries", 0) + len(entries) - 3)
                trimmed.append(f"{name}.entries")
        payload["trimmed_to_fit"] = trimmed
        payload["max_bytes"] = max_bytes
        return json.dumps(payload, indent=1, default=str)


def build_audit_pack(
    repo_root: Path,
    health: Optional[HealthReport] = None,
    leak_report: Optional[Dict[str, Any]] = None,
    trades: Optional[Sequence[Dict[str, Any]]] = None,
    rug_events: Optional[Sequence[Dict[str, Any]]] = None,
    execution_stats: Optional[Dict[str, Any]] = None,
    ledger: Optional[Dict[str, Any]] = None,
    decay: Optional[Dict[str, Any]] = None,
    tail_report: Optional[Dict[str, Any]] = None,
    premature_report: Optional[Dict[str, Any]] = None,
    moat_stats: Optional[Dict[str, Any]] = None,
    period_days: int = 7,
    run_tests: bool = True,
    now: Optional[float] = None,
) -> AuditPack:
    """Assemble every section. Missing inputs produce DATA_BLOCKED sections, not gaps."""
    sections = [
        system_integrity_section(health),
        wealth_leaks_section(leak_report),
        monster_audit_section(trades, tail_report, premature_report),
        rug_defence_section(rug_events),
        execution_section(execution_stats),
        edge_health_section(ledger, decay),
        moat_section(moat_stats),
        changes_section(repo_root, period_days),
    ]
    if run_tests:
        sections.append(tests_section(repo_root))
    else:
        sections.append(_blocked("tests", "test run was disabled for this pack"))
    return AuditPack(generated_at=time.time() if now is None else now,
                     period_days=period_days, sections=sections)
