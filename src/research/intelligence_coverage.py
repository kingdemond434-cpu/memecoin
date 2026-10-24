"""Declared global scope, observed coverage and bounded blind-spot proposals.

The scope catalog is a target denominator, not universal knowledge. A configured
URL never counts as an observation. This loop proposes research and keeps its
history; it cannot change code, signing policy, models or live authority.
"""
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import aiohttp
import yaml

from src.research.prediction_markets import PublicVenues, encode

ROOT = Path(__file__).resolve().parents[2]


class IntelligenceCoverage:
    def __init__(self, broker, *, sources=None, scope=None):
        self.db, self.broker = broker.db, broker
        self.sources = yaml.safe_load(Path(sources or ROOT / "config/sources.yaml").read_text(encoding="utf-8"))["sources"]
        self.scope = yaml.safe_load(Path(scope or ROOT / "config/intelligence_scope.yaml").read_text(encoding="utf-8"))
        if len(self.sources) > 2000 or len({s["id"] for s in self.sources}) != len(self.sources):
            raise ValueError("invalid bounded source catalog")
        self.scope_hash = hashlib.sha256(encode(self.scope).encode()).hexdigest()
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS intelligence_snapshots(id INTEGER PRIMARY KEY,at REAL NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS intelligence_debt(id TEXT PRIMARY KEY,kind TEXT NOT NULL,subject TEXT NOT NULL,
          evidence TEXT NOT NULL,first_seen REAL NOT NULL,last_seen REAL NOT NULL,observations INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS intelligence_rounds(id INTEGER PRIMARY KEY,at REAL NOT NULL,scope_hash TEXT NOT NULL,plan TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS intelligence_cursor(id INTEGER PRIMARY KEY CHECK(id=1),first_seen REAL NOT NULL,debt_id TEXT NOT NULL);
        """)
        if "active" not in {row[1] for row in self.db.execute("PRAGMA table_info(intelligence_debt)")}:
            self.db.execute("ALTER TABLE intelligence_debt ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        self.db.execute("CREATE INDEX IF NOT EXISTS intelligence_debt_work ON intelligence_debt(active,first_seen,id)")
        self.db.commit()
        self.error = "MAIN_OBSERVATIONS_NOT_YET_AVAILABLE"

    def finding(self, kind, subject, evidence):
        key = hashlib.sha256(encode([kind, subject]).encode()).hexdigest()
        now = time.time()
        self.db.execute("INSERT INTO intelligence_debt(id,kind,subject,evidence,first_seen,last_seen,observations,active) VALUES(?,?,?,?,?,?,1,1) ON CONFLICT(id) DO UPDATE SET evidence=excluded.evidence,last_seen=excluded.last_seen,observations=observations+1,active=1",
                        (key, kind, str(subject)[:200], encode(evidence), now, now))

    def ingest(self, snapshot):
        if not isinstance(snapshot, dict):
            raise ValueError("coverage snapshot must be an object")
        observed = snapshot.get("observed_at")
        if type(observed) not in (float, int) or not math.isfinite(observed) or not 0 < observed <= time.time() + 5:
            raise ValueError("native coverage snapshot observation time required")
        if not isinstance(snapshot.get("sources"), list) or len(snapshot["sources"]) > 2000:
            raise ValueError("invalid source observation list")
        # Keep only whitelisted measurements; no endpoints, credentials or text.
        sources, identifiers = [], set()
        for row in snapshot["sources"]:
            if not isinstance(row, dict) or not isinstance(row.get("source_id"), str) or not 0 < len(row["source_id"]) <= 200:
                raise ValueError("invalid source identity")
            if row["source_id"] in identifiers or row.get("state") not in {"OK", "DEGRADED", "DEAD", "NEVER_STARTED"}:
                raise ValueError("duplicate source or invalid state")
            identifiers.add(row["source_id"])
            clean = {"source_id": row["source_id"], "state": row["state"]}
            for key in ("last_poll_ok_at", "last_event_at"):
                value = row.get(key)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= observed + 5):
                    raise ValueError("invalid source observation timestamp")
                clean[key] = value
            sources.append(clean)
        misses = snapshot.get("misses", {})
        if not isinstance(misses, dict) or not isinstance(misses.get("review_candidates", []), list):
            raise ValueError("invalid missed opportunity measurements")
        candidates = []
        for row in misses.get("review_candidates", [])[:100]:
            if not isinstance(row, dict) or not isinstance(row.get("token"), str) or not 0 < len(row["token"]) <= 200:
                raise ValueError("invalid opportunity identity")
            peak = row.get("peak_multiple")
            if type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 1:
                raise ValueError("positive finite peak required")
            candidates.append({"token": row["token"], "peak_multiple": peak,
                               "classification": "HINDSIGHT_POSITIVE_REQUIRES_EXECUTABLE_REVIEW",
                               "executable_after_cost_profit": None, "trade_authority": False})
        counts = {}
        for key in ("retained_positive_peaks", "retained_positive_not_entered", "unresolved_retained"):
            value = misses.get(key)
            if value is not None:
                if type(value) is not int or value < 0: raise ValueError("invalid opportunity count")
                counts[key] = value
        retained = {"observed_at": observed, "sources": sources,
                    "misses": {**counts, "review_candidates": candidates},
                    "denominator": "supported_observed_streams_only"}
        with self.db:
            self.db.execute("INSERT INTO intelligence_snapshots(at,payload) VALUES(?,?)", (time.time(), encode(retained)))
            for row in sources:
                if row["state"] != "OK": self.finding("SOURCE_HEALTH", row["source_id"], row)
                else:
                    key = hashlib.sha256(encode(["SOURCE_HEALTH", row["source_id"]]).encode()).hexdigest()
                    self.db.execute("UPDATE intelligence_debt SET active=0 WHERE id=?", (key,))
            for row in retained["misses"].get("review_candidates", [])[:100]:
                self.finding("HINDSIGHT_POSITIVE_REVIEW", row["token"], row)
            # Keep complete durable debt/round history, bound repeated bulky snapshots.
            self.db.execute("DELETE FROM intelligence_snapshots WHERE id < (SELECT max(id)-100 FROM intelligence_snapshots)")
        self.error = ""

    def propose_round(self):
        count = self.db.execute("SELECT coalesce(max(id),0) FROM intelligence_rounds").fetchone()[0]
        slots = max(1, min(16, int(self.scope["policy"]["total_proposals_per_round"])))
        explore = max(1, min(slots, int(self.scope["policy"]["explore_slots_per_round"])))
        with self.db:
            for source in self.sources:
                if not source.get("options"):
                    self.finding("SOURCE_TARGET_WITHOUT_ENDPOINT", source["id"], {"kind": source["kind"],
                        "region": source.get("region", "UNKNOWN"), "language": source.get("language", "UNKNOWN")})
                else:
                    key = hashlib.sha256(encode(["SOURCE_TARGET_WITHOUT_ENDPOINT", source["id"]]).encode()).hexdigest()
                    self.db.execute("UPDATE intelligence_debt SET active=0 WHERE id=?", (key,))
            # Seek through an indexed bounded working set; retained history can
            # grow without an offset scan or loading every old launch into RAM.
            cursor = self.db.execute("SELECT first_seen,debt_id FROM intelligence_cursor WHERE id=1").fetchone()
            point = tuple(cursor) if cursor else (0, "")
            budget = slots - explore
            debts = self.db.execute("SELECT id,kind,subject,first_seen FROM intelligence_debt WHERE active=1 AND (first_seen,id)>(?,?) ORDER BY first_seen,id LIMIT ?", (*point, budget)).fetchall()
            if len(debts) < budget:
                debts += self.db.execute("SELECT id,kind,subject,first_seen FROM intelligence_debt WHERE active=1 AND (first_seen,id)<=(?,?) ORDER BY first_seen,id LIMIT ?", (*point, budget-len(debts))).fetchall()
            jobs = []
            if debts:
                jobs.extend({"kind": row[1], "subject": row[2], "debt_id": row[0],
                             "status": "PROPOSED_REQUIRES_VERIFIED_INPUTS", "promotion_authority": False}
                            for row in debts)
                self.db.execute("INSERT OR REPLACE INTO intelligence_cursor VALUES(1,?,?)", (debts[-1][3], debts[-1][0]))
            domains = self.scope["domains"]
            for index in range(explore):
                domain = domains[(count * explore + index) % len(domains)]
                jobs.append({"kind": "BLIND_SPOT_EXPLORATION", "subject": domain,
                             "status": "PROPOSED_NOT_OBSERVED", "promotion_authority": False})
            self.db.execute("INSERT INTO intelligence_rounds(at,scope_hash,plan) VALUES(?,?,?)", (time.time(), self.scope_hash, encode(jobs)))
        return jobs

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            venues = PublicVenues(session)
            while True:
                try:
                    self.ingest(await venues.get("http://127.0.0.1:18081", "/intelligence-coverage"))
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, RuntimeError) as exc:
                    self.error = type(exc).__name__
                self.propose_round()
                await asyncio.sleep(120)

    def report(self):
        last = self.db.execute("SELECT at,payload FROM intelligence_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        snapshot = json.loads(last[1]) if last else None
        fresh = bool(snapshot and not self.error and time.time() - snapshot["observed_at"] <= self.scope["policy"]["snapshot_fresh_seconds"])
        observed = {r["source_id"]: r for r in snapshot["sources"]} if fresh else {}
        rows = []
        for source in self.sources:
            measurement = observed.get(source["id"])
            rows.append({"source_id": source["id"], "kind": source["kind"], "region": source.get("region", "UNKNOWN"),
                         "language": source.get("language", "UNKNOWN"), "endpoint_declared": bool(source.get("options")),
                         "credentials_required": bool(source.get("requires_env")),
                         "runtime_state": measurement["state"] if measurement else "UNOBSERVED",
                         "last_poll_ok_at": measurement.get("last_poll_ok_at") if measurement else None,
                         "last_event_at": measurement.get("last_event_at") if measurement else None})
        declared = {s["id"] for s in self.sources}
        dynamic = [row for key, row in observed.items() if key not in declared]
        last_plan = self.db.execute("SELECT plan FROM intelligence_rounds ORDER BY id DESC LIMIT 1").fetchone()
        return {"scope_hash": self.scope_hash, "declared_sources": len(rows), "source_rows": rows,
                "target_domains": self.scope["domains"], "target_actor_classes": self.scope["actor_classes"],
                "target_chains": self.scope["chains"], "target_regions": self.scope["regions"], "target_languages": self.scope["languages"],
                "fresh_main_snapshot": fresh, "main_link_error": self.error,
                "main_observed_at": snapshot["observed_at"] if snapshot else None,
                "last_retained_opportunity_review": snapshot["misses"] if snapshot else None,
                "observed_ok_sources": sum(r["runtime_state"] == "OK" for r in rows),
                "observed_sources_with_events": sum(r["last_event_at"] is not None for r in rows),
                "dynamic_source_rows": dynamic,
                "debt_records": self.db.execute("SELECT count(*) FROM intelligence_debt").fetchone()[0],
                "active_debt_records": self.db.execute("SELECT count(*) FROM intelligence_debt WHERE active=1").fetchone()[0],
                "completed_planning_rounds": self.db.execute("SELECT count(*) FROM intelligence_rounds").fetchone()[0],
                "latest_plan": json.loads(last_plan[0]) if last_plan else [],
                "all_global_sources_observed": False, "global_launch_coverage_fraction": None,
                "automatic_changes": "research_proposals_only_existing_validation_gates_remain_required",
                "trade_authority": False}
