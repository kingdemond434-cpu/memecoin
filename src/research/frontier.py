"""Durable discovery/work index over the desk's canonical evidence.

This SQLite file contains coordinates, contracts and references, never raw
research evidence. The existing research loop calls bounded lease/ack APIs;
this module starts no scheduler and grants no signing or capital authority.
Unknown measurements stay None. Registered work is retained after completion,
failure and low-value hibernation so restart cannot erase coverage debt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence
import uuid
import weakref


SCHEMA_VERSION = 1
FRONTIER_AXES = (
    "chain", "chain_family", "protocol", "lifecycle_stage", "event_type",
    "public_actor_mechanism", "wallet_cohort", "funding_lineage",
    "contract_property", "data_source", "representation", "narrative",
    "region_language", "horizon", "entry_delay", "route", "fee_policy",
    "exit_policy", "risk_policy", "capital_size", "portfolio_state",
)
FRONTIER_STATES = frozenset((
    "UNSEEN", "DISCOVERED", "SOURCE_HUNT", "DATA_BLOCKED", "OBSERVED",
    "PARTIAL", "QUEUED", "TESTED", "DUPLICATE", "INVALID", "GRAVEYARDED",
    "CERTIFIED", "FORWARD", "CANARY", "LIVE", "DECAYED", "HIBERNATED_LOW_VOI",
))
CHAIN_STAGES = (
    "DISCOVERED", "OBSERVATION_ONLY", "HISTORICAL_RESEARCH", "FORWARD_SHADOW",
    "SIMULATED_EXECUTION", "CANARY", "LIVE",
)
CHAIN_METRICS = (
    "launches_per_day", "new_pool_rate", "memecoin_turnover",
    "executable_liquidity", "wallet_activity", "retail_social_activity",
    "public_stream_quality", "historical_data_depth", "protocol_observability",
    "fee_distribution", "block_slot_latency", "finality_reorg_risk",
    "mev_environment", "route_quality", "bridge_dependence", "integration_cost",
    "security_operational_risk",
)
PROTOCOL_FIELDS = (
    "verified_source", "abi_ref", "idl_ref", "addresses", "verification_provenance",
    "observed_native_events", "transaction_fixture_refs", "decoder_parity",
    "unknown_event_rate", "buy_simulation", "sell_simulation", "execution_support",
    "last_verified", "status",
)
DISPOSITIONS = frozenset((
    "EXACT_CANDIDATE", "MECHANISM_HYPOTHESIS", "DATA_REQUEST",
    "REPRESENTATION_REQUEST", "SOURCE_REQUEST", "DEEPENING_TASK",
    "VALIDATION_ATTACK", "FAILURE_DESCENDANT", "SURVIVOR_DESCENDANT",
    "DUPLICATE", "INVALID", "BLOCKED", "NEGATIVE_KNOWLEDGE",
))
EXPLORATION_TAGS = (
    "unknown_chains", "unknown_protocols", "unknown_actors", "unknown_mechanisms",
    "new_source_classes", "strange_falsifiable", "new_models_tools",
)
COST_COMPONENTS = ("inference", "data", "engineering", "conversion", "gauntlet")
_MAX_BATCH = 1000


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value: Any, name: str, *, optional: bool = True) -> Optional[float]:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite nonnegative measurement or None")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def normalize_axes(axes: Optional[Mapping[str, Any]] = None) -> Dict[str, Optional[str]]:
    axes = dict(axes or {})
    unknown = set(axes) - set(FRONTIER_AXES)
    if unknown:
        raise ValueError(f"unknown frontier axes: {sorted(unknown)}")
    result: Dict[str, Optional[str]] = {}
    for name in FRONTIER_AXES:
        value = axes.get(name)
        if value is not None and (isinstance(value, (dict, list, tuple, bool))
                                  or not str(value).strip()):
            raise ValueError(f"{name} must be a scalar coordinate or None")
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        result[name] = str(value).strip() if value is not None else None
    return result


def coordinate_id(axes: Optional[Mapping[str, Any]] = None) -> str:
    return "frontier-" + hashlib.sha256(_json(normalize_axes(axes)).encode()).hexdigest()


def stable_job_id(kind: str, payload_ref: str, axes: Optional[Mapping[str, Any]] = None) -> str:
    return "work-" + hashlib.sha256(
        _json([str(kind), str(payload_ref), coordinate_id(axes)]).encode()).hexdigest()


def _refs(refs: Sequence[str]) -> List[str]:
    if isinstance(refs, str):
        refs = (refs,)
    values = sorted(set(str(ref).strip() for ref in refs if str(ref).strip()))
    if not values:
        raise ValueError("a canonical evidence reference is required")
    return values


def _research_state(state: str) -> str:
    state = str(state).strip().upper()
    if state in ("CANARY", "LIVE"):
        raise PermissionError("research cannot grant CANARY/LIVE authority")
    if state not in FRONTIER_STATES:
        raise ValueError(f"unknown frontier state: {state}")
    return state


@dataclass(frozen=True)
class AuctionInputs:
    """Comparable resource units; missing costs/value never imply free/zero.

    Callers normalize costs to the same declared unit. A discovery job can
    have unknown value and still receive an exploration slot, but cannot be
    leased with an unknown acquisition/conversion/validation cost envelope.
    """

    probability_unique: Optional[float] = None
    expected_delta_logw: Optional[float] = None
    novelty: Optional[float] = None
    coverage_debt: Optional[float] = None
    urgency: Optional[float] = None
    costs: Mapping[str, Optional[float]] = field(default_factory=lambda: {
        name: None for name in COST_COMPONENTS})
    cost_unit: str = "resource_units"
    delay_value_per_second: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for name in ("probability_unique", "expected_delta_logw", "novelty",
                     "coverage_debt", "urgency", "delay_value_per_second"):
            result[name] = _number(result[name], name)
        if result["probability_unique"] is not None and result["probability_unique"] > 1:
            raise ValueError("probability_unique must be <= 1")
        if set(self.costs) - set(COST_COMPONENTS):
            raise ValueError("unknown auction cost component")
        result["costs"] = {name: _number(self.costs.get(name), f"costs.{name}")
                           for name in COST_COMPONENTS}
        if not str(self.cost_unit).strip():
            raise ValueError("cost_unit is required")
        result["cost_unit"] = str(self.cost_unit)
        return result

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "AuctionInputs":
        return cls(**dict(values or {}))

    @property
    def total_cost(self) -> Optional[float]:
        values = list(self.to_dict()["costs"].values())
        return sum(values) if all(v is not None for v in values) else None

    @property
    def priority(self) -> Optional[float]:
        row = self.to_dict()
        factors = [row[name] for name in ("probability_unique", "expected_delta_logw",
                                          "novelty", "coverage_debt", "urgency")]
        cost = self.total_cost
        if cost is None or any(v is None for v in factors):
            return None
        # A declared zero envelope must not create an infinite score.
        return math.prod(factors) / max(cost, 1e-12)


class FrontierIndex:
    """Transactional sparse index driven by existing research passes."""

    def __init__(self, path: Any = "data/research/frontier.sqlite", *,
                 exploration_floors: Optional[Mapping[str, float]] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.exploration_floors = dict(
            {tag: 0.1 for tag in EXPLORATION_TAGS}
            if exploration_floors is None else exploration_floors)
        for tag, share in self.exploration_floors.items():
            if not str(tag).strip() or _number(share, f"floor.{tag}", optional=False) > 1:
                raise ValueError("exploration floor shares must lie in [0, 1]")
        if sum(self.exploration_floors.values()) > 1 + 1e-9:
            raise ValueError("exploration floors must sum to <= 1")
        with self._db() as db:
            # The miner worker and main-loop intake share this reference
            # index. WAL permits readers during a durable writer commit;
            # rollback-journal flush/delete cycles stall Windows intake.
            mode = db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError("frontier requires WAL for concurrent research intake")
            had_active_tags = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='active_job_tags'").fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS coordinates(
                    coordinate_id TEXT PRIMARY KEY, axes TEXT NOT NULL, state TEXT NOT NULL,
                    evidence_refs TEXT NOT NULL, coverage_debt REAL,
                    expected_value REAL, enabling_change TEXT, hibernation_reason TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS chains(
                    chain_id TEXT NOT NULL, version TEXT NOT NULL, family TEXT NOT NULL,
                    stage TEXT NOT NULL, metrics TEXT NOT NULL, evidence_refs TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(chain_id, version));
                CREATE TABLE IF NOT EXISTS protocols(
                    protocol_id TEXT NOT NULL, chain_id TEXT NOT NULL, version TEXT NOT NULL,
                    contract TEXT NOT NULL, evidence_refs TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(protocol_id, chain_id, version));
                CREATE TABLE IF NOT EXISTS jobs(
                    job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_ref TEXT NOT NULL,
                    coordinate_id TEXT NOT NULL, state TEXT NOT NULL, disposition TEXT,
                    evidence_refs TEXT NOT NULL, auction TEXT NOT NULL, priority REAL,
                    total_cost REAL, cost_unit TEXT NOT NULL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, available_at REAL NOT NULL, attempts INTEGER NOT NULL,
                    lease_owner TEXT, lease_token TEXT, lease_until REAL, last_error TEXT,
                    completed_at REAL, enabling_change TEXT, hibernation_reason TEXT);
                CREATE TABLE IF NOT EXISTS job_tags(
                    job_id TEXT NOT NULL, tag TEXT NOT NULL, PRIMARY KEY(job_id, tag));
                CREATE INDEX IF NOT EXISTS jobs_due ON jobs(state, available_at, priority DESC, created_at);
                CREATE INDEX IF NOT EXISTS tag_jobs ON job_tags(tag, job_id);
                CREATE INDEX IF NOT EXISTS jobs_active_created ON jobs(cost_unit,created_at,job_id)
                    WHERE total_cost IS NOT NULL AND state IN ('READY','RETRY','LEASED');
                CREATE INDEX IF NOT EXISTS jobs_active_kind_created ON jobs(cost_unit,kind,created_at,job_id)
                    WHERE total_cost IS NOT NULL AND state IN ('READY','RETRY','LEASED');
                CREATE INDEX IF NOT EXISTS jobs_active_priority ON jobs(cost_unit,priority DESC,created_at,job_id)
                    WHERE total_cost IS NOT NULL AND state IN ('READY','RETRY','LEASED');
                CREATE TABLE IF NOT EXISTS active_job_tags(
                    tag TEXT NOT NULL, job_id TEXT NOT NULL, cost_unit TEXT NOT NULL,
                    kind TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY(tag,job_id));
                CREATE INDEX IF NOT EXISTS active_tags_created ON active_job_tags(tag,cost_unit,created_at,job_id);
                CREATE INDEX IF NOT EXISTS active_tags_kind_created ON active_job_tags(tag,cost_unit,kind,created_at,job_id);
                CREATE INDEX IF NOT EXISTS active_tags_job ON active_job_tags(job_id);
                CREATE TRIGGER IF NOT EXISTS active_tags_job_insert AFTER INSERT ON jobs BEGIN
                    INSERT OR REPLACE INTO active_job_tags
                    SELECT t.tag,NEW.job_id,NEW.cost_unit,NEW.kind,NEW.created_at FROM job_tags t
                    WHERE t.job_id=NEW.job_id AND NEW.total_cost IS NOT NULL
                      AND NEW.state IN ('READY','RETRY','LEASED');
                END;
                CREATE TRIGGER IF NOT EXISTS active_tags_job_update
                AFTER UPDATE OF state,total_cost,cost_unit,kind,created_at ON jobs BEGIN
                    DELETE FROM active_job_tags WHERE job_id=NEW.job_id;
                    INSERT OR REPLACE INTO active_job_tags
                    SELECT t.tag,NEW.job_id,NEW.cost_unit,NEW.kind,NEW.created_at FROM job_tags t
                    WHERE t.job_id=NEW.job_id AND NEW.total_cost IS NOT NULL
                      AND NEW.state IN ('READY','RETRY','LEASED');
                END;
                CREATE TRIGGER IF NOT EXISTS active_tags_job_delete AFTER DELETE ON jobs BEGIN
                    DELETE FROM active_job_tags WHERE job_id=OLD.job_id;
                END;
                CREATE TRIGGER IF NOT EXISTS active_tags_tag_insert AFTER INSERT ON job_tags BEGIN
                    INSERT OR REPLACE INTO active_job_tags
                    SELECT NEW.tag,j.job_id,j.cost_unit,j.kind,j.created_at FROM jobs j
                    WHERE j.job_id=NEW.job_id AND j.total_cost IS NOT NULL
                      AND j.state IN ('READY','RETRY','LEASED');
                END;
                CREATE TRIGGER IF NOT EXISTS active_tags_tag_update AFTER UPDATE ON job_tags BEGIN
                    DELETE FROM active_job_tags WHERE job_id=OLD.job_id AND tag=OLD.tag;
                    INSERT OR REPLACE INTO active_job_tags
                    SELECT NEW.tag,j.job_id,j.cost_unit,j.kind,j.created_at FROM jobs j
                    WHERE j.job_id=NEW.job_id AND j.total_cost IS NOT NULL
                      AND j.state IN ('READY','RETRY','LEASED');
                END;
                CREATE TRIGGER IF NOT EXISTS active_tags_tag_delete AFTER DELETE ON job_tags BEGIN
                    DELETE FROM active_job_tags WHERE job_id=OLD.job_id AND tag=OLD.tag;
                END;
                CREATE TABLE IF NOT EXISTS transitions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL,
                    action TEXT NOT NULL, previous_state TEXT, next_state TEXT,
                    reason TEXT, evidence_refs TEXT NOT NULL, at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS transitions_time ON transitions(at);
            """)
            current = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if current and int(current[0]) != SCHEMA_VERSION:
                raise ValueError("unsupported frontier schema version")
            db.execute("INSERT OR IGNORE INTO metadata VALUES('schema_version', ?)",
                       (str(SCHEMA_VERSION),))
            db.execute("INSERT OR IGNORE INTO metadata VALUES('exploration_cursor', '0')")
            if not had_active_tags or not db.execute("SELECT 1 FROM metadata WHERE key='active_job_tags_v1'").fetchone():
                # Derived queue only: old jobs/tags remain the source of truth.
                # Backfill and its completion marker commit in one transaction;
                # triggers keep every later transition in that same transaction.
                cache_size = db.execute("PRAGMA cache_size").fetchone()[0]
                try:
                    # Read the large legacy job table sequentially. Following
                    # its state index caused random reads across an 800 MB VPS
                    # file. A fixed 128 MiB cache holds tag/index working pages
                    # for this one migration, then returns to the prior bound.
                    db.execute("PRAGMA cache_size=-131072")
                    db.execute("DELETE FROM active_job_tags")
                    db.execute("INSERT INTO active_job_tags SELECT t.tag,j.job_id,j.cost_unit,j.kind,j.created_at "
                               "FROM jobs j NOT INDEXED CROSS JOIN job_tags t "
                               "WHERE t.job_id=j.job_id AND j.total_cost IS NOT NULL "
                               "AND j.state IN ('READY','RETRY','LEASED')")
                finally:
                    db.execute(f"PRAGMA cache_size={int(cache_size)}")
                db.execute("INSERT OR REPLACE INTO metadata VALUES('active_job_tags_v1','1')")
            # Keep WAL open across short transactions. Closing the last
            # connection after every event would checkpoint/delete it again,
            # turning each source record back into repeated database flushes.
            keeper = sqlite3.connect(str(self.path), timeout=15, check_same_thread=False)
            keeper.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            self._close_keeper = weakref.finalize(self, keeper.close)

    def close(self) -> None:
        """Release the idle WAL anchor after all research producers stop."""
        self._close_keeper()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        finalizer = getattr(self, "_close_keeper", None)
        if finalizer is not None and not finalizer.alive:
            raise RuntimeError("research frontier is closed")
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _event(db: sqlite3.Connection, subject: str, action: str,
               previous: Optional[str], following: Optional[str], reason: str,
               refs: Sequence[str], at: float) -> None:
        db.execute("INSERT INTO transitions(subject,action,previous_state,next_state,reason,evidence_refs,at) "
                   "VALUES(?,?,?,?,?,?,?)", (subject, action, previous, following, reason, _json(refs), at))

    @staticmethod
    def _coordinate_row(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["axes"] = json.loads(result["axes"])
        result["evidence_refs"] = json.loads(result["evidence_refs"])
        return result

    def coordinate(self, axes: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        identity = coordinate_id(axes)
        with self._db() as db:
            row = db.execute("SELECT * FROM coordinates WHERE coordinate_id=?", (identity,)).fetchone()
        return self._coordinate_row(row) if row else {
            "coordinate_id": identity, "axes": normalize_axes(axes), "state": "UNSEEN",
            "evidence_refs": [], "coverage_debt": None, "expected_value": None,
            "enabling_change": None, "hibernation_reason": None,
            "created_at": None, "updated_at": None,
        }

    def observe_coordinate(self, axes: Mapping[str, Any], *, state: str = "DISCOVERED",
                           evidence_refs: Sequence[str], coverage_debt: Optional[float] = None,
                           expected_value: Optional[float] = None,
                           enabling_change: Optional[str] = None,
                           hibernation_reason: Optional[str] = None,
                           now: Optional[float] = None) -> str:
        state = _research_state(state)
        refs = _refs(evidence_refs)
        debt = _number(coverage_debt, "coverage_debt")
        value = _number(expected_value, "expected_value")
        if state == "HIBERNATED_LOW_VOI" and (not enabling_change or not hibernation_reason):
            raise ValueError("hibernation requires a named enabling_change and reason")
        moment = time.time() if now is None else float(now)
        normalized = normalize_axes(axes)
        identity = coordinate_id(normalized)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM coordinates WHERE coordinate_id=?", (identity,)).fetchone()
            if old and old["state"] == "HIBERNATED_LOW_VOI" and state != old["state"]:
                raise ValueError("use wake_coordinate with the named enabling change")
            merged = sorted(set(refs + (json.loads(old["evidence_refs"]) if old else [])))
            db.execute("""INSERT INTO coordinates VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(coordinate_id) DO UPDATE SET state=excluded.state,
                evidence_refs=excluded.evidence_refs,
                coverage_debt=COALESCE(excluded.coverage_debt,coordinates.coverage_debt),
                expected_value=COALESCE(excluded.expected_value,coordinates.expected_value),
                enabling_change=COALESCE(excluded.enabling_change,coordinates.enabling_change),
                hibernation_reason=COALESCE(excluded.hibernation_reason,coordinates.hibernation_reason),
                updated_at=excluded.updated_at""",
                       (identity, _json(normalized), state, _json(merged), debt, value,
                        enabling_change, hibernation_reason, moment, moment))
            self._event(db, identity, "coordinate", old["state"] if old else "UNSEEN",
                        state, hibernation_reason or "observation", refs, moment)
        return identity

    def wake_coordinate(self, axes: Mapping[str, Any], *, enabling_change: str,
                        evidence_refs: Sequence[str], now: Optional[float] = None) -> str:
        refs = _refs(evidence_refs)
        identity = coordinate_id(axes)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM coordinates WHERE coordinate_id=?", (identity,)).fetchone()
            if not row or row["state"] != "HIBERNATED_LOW_VOI":
                raise ValueError("coordinate is not hibernated")
            if row["enabling_change"] != enabling_change:
                raise ValueError("enabling change does not match")
            merged = sorted(set(refs + json.loads(row["evidence_refs"])))
            db.execute("UPDATE coordinates SET state='DISCOVERED', evidence_refs=?, updated_at=? "
                       "WHERE coordinate_id=?", (_json(merged), moment, identity))
            self._event(db, identity, "wake", row["state"], "DISCOVERED", enabling_change, refs, moment)
        return identity

    def register_chain(self, chain_id: str, family: str, *, version: str = "1",
                       metrics: Optional[Mapping[str, Any]] = None,
                       evidence_refs: Sequence[str], now: Optional[float] = None) -> Dict[str, Any]:
        if not chain_id.strip() or not family.strip() or not version.strip():
            raise ValueError("chain identity, family and version are required")
        measured = dict(metrics or {})
        if set(measured) - set(CHAIN_METRICS):
            raise ValueError("unknown chain metric")
        moment = time.time() if now is None else float(now)
        refs = _refs(evidence_refs)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM chains WHERE chain_id=? AND version=?",
                             (chain_id, version)).fetchone()
            if old and old["family"] != family:
                raise ValueError("a chain version cannot change family")
            values = json.loads(old["metrics"]) if old else dict.fromkeys(CHAIN_METRICS)
            values.update(measured)
            merged = sorted(set(refs + (json.loads(old["evidence_refs"]) if old else [])))
            db.execute("""INSERT INTO chains VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(chain_id,version) DO UPDATE SET metrics=excluded.metrics,
                evidence_refs=excluded.evidence_refs, updated_at=excluded.updated_at""",
                       (chain_id, version, family, "DISCOVERED", _json(values),
                        _json(merged), moment, moment))
            self._event(db, f"chain:{chain_id}:{version}", "chain_observed", None,
                        old["stage"] if old else "DISCOVERED", "dynamic observation", refs, moment)
            self._discover_in_db(db, {"chain": chain_id, "chain_family": family}, refs, moment)
        return self.chain(chain_id, version=version)

    def chain(self, chain_id: str, *, version: str = "1") -> Optional[Dict[str, Any]]:
        with self._db() as db:
            row = db.execute("SELECT * FROM chains WHERE chain_id=? AND version=?",
                             (chain_id, version)).fetchone()
        if not row:
            return None
        result = dict(row)
        for name in ("metrics", "evidence_refs"):
            result[name] = json.loads(result[name])
        return result

    def advance_chain(self, chain_id: str, stage: str, *, version: str = "1",
                      evidence_refs: Sequence[str], now: Optional[float] = None) -> Dict[str, Any]:
        stage = str(stage).strip().upper()
        if stage in ("CANARY", "LIVE"):
            raise PermissionError("research cannot grant CANARY/LIVE authority")
        if stage not in CHAIN_STAGES:
            raise ValueError("unknown chain stage")
        refs = _refs(evidence_refs)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM chains WHERE chain_id=? AND version=?",
                             (chain_id, version)).fetchone()
            if not row:
                raise KeyError(chain_id)
            if CHAIN_STAGES.index(stage) != CHAIN_STAGES.index(row["stage"]) + 1:
                raise ValueError("advance exactly one research stage at a time")
            merged = sorted(set(refs + json.loads(row["evidence_refs"])))
            db.execute("UPDATE chains SET stage=?, evidence_refs=?, updated_at=? "
                       "WHERE chain_id=? AND version=?", (stage, _json(merged), moment, chain_id, version))
            self._event(db, f"chain:{chain_id}:{version}", "chain_stage", row["stage"],
                        stage, "external evidence referenced; no capital authority", refs, moment)
        return self.chain(chain_id, version=version)

    def register_protocol(self, protocol_id: str, chain_id: str, version: str, *,
                          contract: Optional[Mapping[str, Any]] = None,
                          evidence_refs: Sequence[str], now: Optional[float] = None) -> Dict[str, Any]:
        if not all(str(item).strip() for item in (protocol_id, chain_id, version)):
            raise ValueError("protocol, chain and version are required")
        proposed = dict(contract or {})
        if set(proposed) - set(PROTOCOL_FIELDS):
            raise ValueError("unknown protocol contract field")
        if "status" in proposed and proposed["status"] is not None:
            proposed["status"] = str(proposed["status"]).strip().upper()
        if proposed.get("status") in ("CANARY", "LIVE"):
            raise PermissionError("research cannot grant CANARY/LIVE authority")
        if proposed.get("execution_support") not in (None, False, "UNKNOWN", "UNSUPPORTED", "SHADOW_ONLY"):
            raise PermissionError("protocol discovery cannot grant execution support")
        refs = _refs(evidence_refs)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM protocols WHERE protocol_id=? AND chain_id=? AND version=?",
                             (protocol_id, chain_id, version)).fetchone()
            values = json.loads(old["contract"]) if old else dict.fromkeys(PROTOCOL_FIELDS)
            for name in ("verified_source", "abi_ref", "idl_ref", "addresses"):
                if name in proposed and values[name] is not None and proposed[name] != values[name]:
                    raise ValueError(f"{name} changed; register a new protocol version")
            values.update(proposed)
            if values["status"] is None:
                values["status"] = "DISCOVERED"
            # A syntactic decode never earns verification. Provenance, native
            # fixtures and measured parity are required even for research.
            if str(values["status"]).upper() in ("VERIFIED", "CERTIFIED"):
                if not (values["verified_source"] and (values["abi_ref"] or values["idl_ref"])
                        and values["addresses"] and values["verification_provenance"]
                        and values["transaction_fixture_refs"] and values["decoder_parity"] is True
                        and values["last_verified"] is not None):
                    raise ValueError("verified protocol requires source, layout, fixtures and parity evidence")
            rate = values["unknown_event_rate"]
            if rate is not None and (_number(rate, "unknown_event_rate", optional=False) > 1):
                raise ValueError("unknown_event_rate must lie in [0, 1]")
            merged = sorted(set(refs + (json.loads(old["evidence_refs"]) if old else [])))
            db.execute("""INSERT INTO protocols VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(protocol_id,chain_id,version) DO UPDATE SET contract=excluded.contract,
                evidence_refs=excluded.evidence_refs, updated_at=excluded.updated_at""",
                       (protocol_id, chain_id, version, _json(values), _json(merged), moment, moment))
            self._event(db, f"protocol:{chain_id}:{protocol_id}:{version}", "protocol_observed", None,
                        values["status"], "versioned observation; no execution authority", refs, moment)
            self._discover_in_db(db, {"chain": chain_id, "protocol": protocol_id}, refs, moment)
        return self.protocol(protocol_id, chain_id, version)

    def protocol(self, protocol_id: str, chain_id: str, version: str) -> Optional[Dict[str, Any]]:
        with self._db() as db:
            row = db.execute("SELECT * FROM protocols WHERE protocol_id=? AND chain_id=? AND version=?",
                             (protocol_id, chain_id, version)).fetchone()
        if not row:
            return None
        result = dict(row)
        for name in ("contract", "evidence_refs"):
            result[name] = json.loads(result[name])
        return result

    def _discover_in_db(self, db: sqlite3.Connection, axes: Mapping[str, Any],
                        refs: Sequence[str], at: float) -> str:
        """An observation and its registry/job index commit together.

        Read and update under the same write transaction: a racing new
        sighting must not turn another worker's TESTED cell into DISCOVERED.
        """
        normalized = normalize_axes(axes)
        identity = coordinate_id(normalized)
        old = db.execute("SELECT * FROM coordinates WHERE coordinate_id=?", (identity,)).fetchone()
        if old:
            merged = sorted(set(refs + json.loads(old["evidence_refs"])))
            db.execute("UPDATE coordinates SET evidence_refs=?,updated_at=? WHERE coordinate_id=?",
                       (_json(merged), at, identity))
        else:
            db.execute("INSERT INTO coordinates VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (identity, _json(normalized), "DISCOVERED", _json(refs), None,
                        None, None, None, at, at))
        self._event(db, identity, "discovery", old["state"] if old else "UNSEEN",
                    old["state"] if old else "DISCOVERED", "dynamic observation", refs, at)
        return identity

    def _discover_coordinate(self, axes: Mapping[str, Any], refs: Sequence[str], at: float) -> str:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._discover_in_db(db, axes, refs, at)

    def observe_discovery(self, kind: str, identity: str, *, evidence_refs: Sequence[str],
                          axes: Optional[Mapping[str, Any]] = None,
                          family: str = "UNKNOWN", version: str = "1",
                          metrics: Optional[Mapping[str, Any]] = None,
                          contract: Optional[Mapping[str, Any]] = None,
                          now: Optional[float] = None) -> str:
        """Admission from existing miners/crawler; discoveries remain research.

        The evidence reference identifies an already-persisted canonical row.
        Metadata is deliberately not copied into another raw evidence store.
        """
        if not str(identity).strip():
            raise ValueError("discovery identity is required")
        kind = kind.lower()
        mapped = dict(axes or {})
        if kind == "chain":
            self.register_chain(identity, family, version=version, metrics=metrics,
                                evidence_refs=evidence_refs, now=now)
            mapped.update(chain=identity, chain_family=family)
        elif kind == "protocol":
            if not mapped.get("chain"):
                raise ValueError("protocol discovery requires its native chain")
            self.register_protocol(identity, str(mapped["chain"]), version,
                                   contract=contract, evidence_refs=evidence_refs, now=now)
            mapped["protocol"] = identity
        else:
            axis = {"source": "data_source", "actor": "public_actor_mechanism",
                    "mechanism": "public_actor_mechanism", "representation": "representation",
                    "narrative": "narrative"}.get(kind)
            if axis is None:
                raise ValueError("unsupported discovery kind")
            mapped[axis] = identity
        return self._discover_coordinate(mapped, _refs(evidence_refs), time.time() if now is None else now)

    def enqueue(self, kind: str, payload_ref: str, *, axes: Optional[Mapping[str, Any]] = None,
                job_id: Optional[str] = None, evidence_refs: Sequence[str] = (),
                auction: Optional[AuctionInputs] = None, exploration_tags: Sequence[str] = (),
                disposition: Optional[str] = None, now: Optional[float] = None) -> str:
        if not str(kind).strip() or not str(payload_ref).strip():
            raise ValueError("kind and canonical payload_ref are required")
        if disposition is not None and disposition not in DISPOSITIONS:
            raise ValueError("unknown research disposition")
        refs = _refs(evidence_refs or (payload_ref,))
        normalized = normalize_axes(axes)
        identity = job_id or stable_job_id(kind, payload_ref, normalized)
        if not str(identity).strip():
            raise ValueError("job identity is required")
        moment = time.time() if now is None else float(now)
        inputs = auction or AuctionInputs()
        values = inputs.to_dict()
        tags = sorted(set(str(tag).strip() for tag in exploration_tags if str(tag).strip()))
        coord = coordinate_id(normalized)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE job_id=?", (identity,)).fetchone()
            if old:
                if (old["kind"], old["payload_ref"], old["coordinate_id"]) != (kind, payload_ref, coord):
                    raise ValueError("stable job identity collides with different work")
                return identity  # delivery retry must not reset lease, ack or costs
            self._discover_in_db(db, normalized, refs, moment)
            cell = db.execute("SELECT state,enabling_change,hibernation_reason FROM coordinates "
                              "WHERE coordinate_id=?", (coord,)).fetchone()
            hibernated = cell["state"] == "HIBERNATED_LOW_VOI"
            db.execute("""INSERT INTO jobs(job_id,kind,payload_ref,coordinate_id,state,disposition,
                evidence_refs,auction,priority,total_cost,cost_unit,created_at,updated_at,available_at,attempts,
                enabling_change,hibernation_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (identity, kind, payload_ref, coord, "HIBERNATED" if hibernated else "READY", disposition, _json(refs),
                        _json(values), inputs.priority, inputs.total_cost, values["cost_unit"],
                        moment, moment, moment, 0,
                        cell["enabling_change"] if hibernated else None,
                        cell["hibernation_reason"] if hibernated else None))
            db.executemany("INSERT INTO job_tags VALUES(?,?)", [(identity, tag) for tag in tags])
            self._event(db, identity, "enqueue", None, "HIBERNATED" if hibernated else "READY",
                        "canonical evidence reference", refs, moment)
        return identity

    @staticmethod
    def _job_row(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        for name in ("evidence_refs", "auction"):
            result[name] = json.loads(result[name])
        result["priority_status"] = "UNMEASURED" if result["priority"] is None else "MEASURED"
        return result

    def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                return None
            result = self._job_row(row)
            result["axes"] = json.loads(db.execute(
                "SELECT axes FROM coordinates WHERE coordinate_id=?", (row["coordinate_id"],)).fetchone()[0])
            result["exploration_tags"] = [r[0] for r in db.execute(
                "SELECT tag FROM job_tags WHERE job_id=? ORDER BY tag", (job_id,))]
        return result

    def lease_jobs(self, worker: str, *, limit: int = 8, lease_seconds: float = 60,
                   resource_budget: Optional[float] = None, cost_unit: str = "resource_units",
                   scan_limit: int = 256, kinds: Optional[Sequence[str]] = None,
                   now: Optional[float] = None) -> List[Dict[str, Any]]:
        """One bounded resource auction inside the caller's existing pass.

        Exploration floors rotate durably when a pass is smaller than the
        number of discovery dimensions. Unknown-cost work remains READY and
        visible; no job disappears when it cannot fit this pass's envelope.
        """
        if not str(worker).strip():
            raise ValueError("worker identity is required")
        if (isinstance(limit, bool) or isinstance(scan_limit, bool)
                or int(limit) != limit or int(scan_limit) != scan_limit
                or not 1 <= int(limit) <= _MAX_BATCH or not 1 <= int(scan_limit) <= _MAX_BATCH):
            raise ValueError("lease/scan limit must be bounded between 1 and 1000")
        duration = _number(lease_seconds, "lease_seconds", optional=False)
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        budget = _number(resource_budget, "resource_budget")
        remaining = math.inf if budget is None else budget
        moment = time.time() if now is None else float(now)
        selected: List[sqlite3.Row] = []
        selected_ids = set()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            # Expired leases are selectable without rewriting every expired
            # row: one bounded pass never scans/mutates an unbounded backlog.
            due = "((j.state IN ('READY','RETRY') AND j.available_at<=?) OR (j.state='LEASED' AND j.lease_until<=?))"
            eligible = due + " AND j.state IN ('READY','RETRY','LEASED') AND j.total_cost IS NOT NULL AND j.cost_unit=? AND EXISTS " \
                "(SELECT 1 FROM coordinates c WHERE c.coordinate_id=j.coordinate_id AND c.state!='HIBERNATED_LOW_VOI')"
            params = (moment, moment, cost_unit)
            if kinds is not None:
                names = sorted(set(kinds))
                if not names:
                    return []
                if len(names) > 64 or any(not isinstance(name, str) or not name for name in names):
                    raise ValueError("bounded named consumer kinds required")
                eligible += " AND j.kind IN (" + ",".join("?" for _ in names) + ")"
                params += tuple(names)
            tags = [tag for tag, fraction in self.exploration_floors.items() if fraction > 0]
            cursor = int(db.execute("SELECT value FROM metadata WHERE key='exploration_cursor'").fetchone()[0])
            ordered_tags = tags[cursor % len(tags):] + tags[:cursor % len(tags)] if tags else []

            def take(rows: Sequence[sqlite3.Row], maximum: int) -> int:
                nonlocal remaining
                added = 0
                for row in rows:
                    if len(selected) >= limit or added >= maximum:
                        break
                    if row["job_id"] in selected_ids or row["total_cost"] > remaining:
                        continue
                    selected.append(row)
                    selected_ids.add(row["job_id"])
                    remaining -= row["total_cost"]
                    added += 1
                return added

            visited = 0
            for tag in ordered_tags:
                if len(selected) >= limit:
                    break
                visited += 1
                count = max(1, int(math.floor(limit * self.exploration_floors[tag])))
                # A rare tag must not scan unrelated active work, while a
                # common tag must not scan completed tagged history either.
                # This transactionally maintained sparse index excludes both.
                one_kind = kinds is not None and len(names) == 1
                fifo_index = "active_tags_kind_created" if one_kind else "active_tags_created"
                tag_kind = " AND a.kind=?" if one_kind else ""
                tag_params = (tag, cost_unit, names[0]) if one_kind else (tag, cost_unit)
                rows = db.execute(f"SELECT j.* FROM active_job_tags a INDEXED BY {fifo_index} "
                                  "JOIN jobs j ON j.job_id=a.job_id "
                                  f"WHERE a.tag=? AND a.cost_unit=?{tag_kind} AND {eligible} "
                                  "ORDER BY a.created_at,a.job_id LIMIT ?",
                                  (*tag_params, *params, int(scan_limit))).fetchall()
                take(rows, count)
            if tags:
                db.execute("UPDATE metadata SET value=? WHERE key='exploration_cursor'",
                           (str((cursor + max(1, visited)) % len(tags)),))
            if len(selected) < limit:
                rows = db.execute(f"SELECT j.* FROM jobs j INDEXED BY jobs_active_priority WHERE {eligible} "
                                  "ORDER BY j.priority DESC,j.created_at,j.job_id LIMIT ?",
                                  (*params, int(scan_limit))).fetchall()
                take(rows, limit - len(selected))
            result = []
            for row in selected:
                token = uuid.uuid4().hex
                db.execute("UPDATE jobs SET state='LEASED',lease_owner=?,lease_token=?,lease_until=?,"
                           "attempts=attempts+1,updated_at=? WHERE job_id=?",
                           (worker, token, moment + duration, moment, row["job_id"]))
                self._event(db, row["job_id"], "lease", row["state"], "LEASED", worker,
                            json.loads(row["evidence_refs"]), moment)
                leased = self._job_row(db.execute("SELECT * FROM jobs WHERE job_id=?",
                                                  (row["job_id"],)).fetchone())
                leased["axes"] = json.loads(db.execute(
                    "SELECT axes FROM coordinates WHERE coordinate_id=?", (row["coordinate_id"],)).fetchone()[0])
                result.append(leased)
        return result

    def pending_work(self, *, kinds: Sequence[str] = (), states: Sequence[str] = ("READY", "RETRY"),
                     limit: int = 64, after_job_id: str = "", enabling_change: Optional[str] = None) -> List[Dict[str, Any]]:
        """A bounded cursor scan for consumer migration and evidence wakes."""
        if not 1 <= int(limit) <= _MAX_BATCH or not states or len(states) > 8 or len(kinds) > 64:
            raise ValueError("invalid bounded pending-work query")
        where = ["state IN (" + ",".join("?" for _ in states) + ")", "job_id>?"]
        params: list = [*states, after_job_id]
        if kinds:
            where.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            params.extend(kinds)
        if enabling_change is not None:
            where.append("enabling_change=?")
            params.append(enabling_change)
        with self._db() as db:
            rows = db.execute("SELECT job_id FROM jobs WHERE " + " AND ".join(where) +
                              " ORDER BY job_id LIMIT ?", (*params, int(limit))).fetchall()
        return [self.job(row[0]) for row in rows]

    def due_kinds(self, kinds: Sequence[str], *, resource_budget: float, now: float) -> List[str]:
        """Costed consumer groups due inside this pass's resource envelope."""
        names = sorted(set(kinds))
        if not names or len(names) > 64:
            raise ValueError("bounded registered consumer kinds required")
        budget = _number(resource_budget, "resource_budget", optional=False)
        with self._db() as db:
            # Ask whether one eligible row exists per bounded consumer kind.
            # DISTINCT over every retained row bought no additional evidence.
            found = []
            for name in names:
                row = db.execute("SELECT j.kind FROM jobs j INDEXED BY jobs_active_kind_created "
                "JOIN coordinates c USING(coordinate_id) WHERE j.kind=? "
                "AND j.state IN ('READY','RETRY','LEASED') "
                "AND ((j.state IN ('READY','RETRY') AND j.available_at<=?) OR (j.state='LEASED' AND j.lease_until<=?)) "
                "AND j.total_cost IS NOT NULL AND j.total_cost<=? AND j.cost_unit='resource_units' "
                "AND c.state!='HIBERNATED_LOW_VOI' LIMIT 1", (name, now, now, budget)).fetchone()
                if row:
                    found.append(row[0])
        return found

    def price_pending_job(self, job_id: str, *, auction: AuctionInputs,
                          evidence_refs: Sequence[str], reason: str,
                          now: Optional[float] = None) -> bool:
        """Declare a consumer's resource envelope without resetting work."""
        if auction.total_cost is None or auction.total_cost <= 0 or not reason:
            raise ValueError("explicit bounded positive resource cost and reason required")
        refs = _refs(evidence_refs)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["state"] not in {"READY", "RETRY"} or row["total_cost"] is not None:
                return False
            coordinate = db.execute("SELECT state FROM coordinates WHERE coordinate_id=?", (row["coordinate_id"],)).fetchone()
            if coordinate[0] == "HIBERNATED_LOW_VOI":
                return False
            merged = sorted(set(refs + json.loads(row["evidence_refs"])))
            values = auction.to_dict()
            db.execute("UPDATE jobs SET auction=?,priority=?,total_cost=?,cost_unit=?,evidence_refs=?,updated_at=? WHERE job_id=?",
                (_json(values), auction.priority, auction.total_cost, values["cost_unit"], _json(merged), moment, job_id))
            self._event(db, job_id, "resource_envelope_declared", row["state"], row["state"], reason, refs, moment)
        return True

    def block_leased_job(self, job_id: str, worker: str, lease_token: str, *, enabling_change: str,
                         reason: str, evidence_refs: Sequence[str], now: Optional[float] = None) -> None:
        """Keep incomplete work resumable; a blocked receipt is not an ACK."""
        if not enabling_change or not reason:
            raise ValueError("named enabling change and measured blocking reason required")
        refs = _refs(evidence_refs)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._leased(db, job_id, worker, lease_token, moment)
            merged = sorted(set(refs + json.loads(row["evidence_refs"])))
            db.execute("UPDATE jobs SET state='HIBERNATED',disposition='BLOCKED',enabling_change=?,hibernation_reason=?,"
                       "last_error=?,evidence_refs=?,updated_at=?,lease_owner=NULL,lease_token=NULL,lease_until=NULL WHERE job_id=?",
                       (enabling_change, reason, reason, _json(merged), moment, job_id))
            self._event(db, job_id, "input_blocked", "LEASED", "HIBERNATED", reason, refs, moment)

    @staticmethod
    def _leased(db: sqlite3.Connection, job_id: str, worker: str, token: str,
                at: float) -> sqlite3.Row:
        row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        if (row["state"] != "LEASED" or row["lease_owner"] != worker
                or row["lease_token"] != token or row["lease_until"] <= at):
            raise PermissionError("lease expired, revoked or belongs to another worker")
        return row

    def ack(self, job_id: str, worker: str, lease_token: str, *, disposition: str,
            evidence_refs: Sequence[str], now: Optional[float] = None) -> None:
        if disposition not in DISPOSITIONS:
            raise ValueError("unknown research disposition")
        refs = _refs(evidence_refs)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._leased(db, job_id, worker, lease_token, moment)
            merged = sorted(set(refs + json.loads(row["evidence_refs"])))
            db.execute("UPDATE jobs SET state='ACKED',disposition=?,evidence_refs=?,completed_at=?,"
                       "updated_at=?,lease_owner=NULL,lease_token=NULL,lease_until=NULL WHERE job_id=?",
                       (disposition, _json(merged), moment, moment, job_id))
            self._event(db, job_id, "ack", "LEASED", "ACKED", disposition, refs, moment)

    def retry(self, job_id: str, worker: str, lease_token: str, *, reason: str,
              delay_seconds: float = 60, now: Optional[float] = None) -> None:
        if not str(reason).strip():
            raise ValueError("retry reason is required")
        delay = _number(delay_seconds, "delay_seconds", optional=False)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._leased(db, job_id, worker, lease_token, moment)
            db.execute("UPDATE jobs SET state='RETRY',last_error=?,available_at=?,updated_at=?,"
                       "lease_owner=NULL,lease_token=NULL,lease_until=NULL WHERE job_id=?",
                       (reason, moment + delay, moment, job_id))
            self._event(db, job_id, "retry", "LEASED", "RETRY", reason,
                        json.loads(row["evidence_refs"]), moment)

    def hibernate_job(self, job_id: str, *, enabling_change: str, reason: str,
                      now: Optional[float] = None) -> None:
        if not str(enabling_change).strip() or not str(reason).strip():
            raise ValueError("hibernation requires enabling_change and reason")
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            if row["state"] in ("LEASED", "ACKED"):
                raise ValueError("active/completed work cannot be silently hibernated")
            db.execute("UPDATE jobs SET state='HIBERNATED',enabling_change=?,hibernation_reason=?,"
                       "updated_at=? WHERE job_id=?", (enabling_change, reason, moment, job_id))
            self._event(db, job_id, "hibernate", row["state"], "HIBERNATED", reason,
                        json.loads(row["evidence_refs"]), moment)

    def wake_job(self, job_id: str, *, enabling_change: str, evidence_refs: Sequence[str],
                 auction: Optional[AuctionInputs] = None, now: Optional[float] = None) -> None:
        refs = _refs(evidence_refs)
        moment = time.time() if now is None else float(now)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["state"] != "HIBERNATED" or row["enabling_change"] != enabling_change:
                raise ValueError("work is not hibernated on the named enabling change")
            coordinate = db.execute("SELECT state FROM coordinates WHERE coordinate_id=?",
                                    (row["coordinate_id"],)).fetchone()
            if coordinate["state"] == "HIBERNATED_LOW_VOI":
                raise ValueError("wake the coordinate with its named enabling change first")
            inputs = auction or AuctionInputs.from_mapping(json.loads(row["auction"]))
            merged = sorted(set(refs + json.loads(row["evidence_refs"])))
            db.execute("UPDATE jobs SET state='READY',available_at=?,updated_at=?,evidence_refs=?,"
                       "auction=?,priority=?,total_cost=?,cost_unit=? WHERE job_id=?",
                       (moment, moment, _json(merged), _json(inputs.to_dict()), inputs.priority,
                        inputs.total_cost, inputs.cost_unit, job_id))
            self._event(db, job_id, "wake", "HIBERNATED", "READY", enabling_change, refs, moment)

    def report(self, *, now: Optional[float] = None, window_seconds: float = 3600) -> Dict[str, Any]:
        moment = time.time() if now is None else float(now)
        window = _number(window_seconds, "window_seconds", optional=False)
        if window <= 0:
            raise ValueError("window_seconds must be positive")
        with self._db() as db:
            counts = {r[0]: r[1] for r in db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state")}
            states = {r[0]: r[1] for r in db.execute("SELECT state,COUNT(*) FROM coordinates GROUP BY state")}
            arrivals = db.execute("SELECT COUNT(*) FROM jobs WHERE created_at>=? AND created_at<=?",
                                  (moment - window, moment)).fetchone()[0]
            drains = db.execute("SELECT COUNT(*) FROM jobs WHERE completed_at>=? AND completed_at<=?",
                                (moment - window, moment)).fetchone()[0]
            pending = "state!='ACKED'"
            oldest = db.execute(f"SELECT MIN(created_at) FROM jobs WHERE {pending}").fetchone()[0]
            valuable = db.execute(f"SELECT job_id,created_at FROM jobs WHERE {pending} AND priority>0 "
                                  "ORDER BY created_at LIMIT 1").fetchone()
            blocked = db.execute(f"SELECT COUNT(*) FROM jobs WHERE {pending} AND total_cost IS NULL").fetchone()[0]
            debt = db.execute("SELECT SUM(coverage_debt),COUNT(coverage_debt),COUNT(*) FROM coordinates").fetchone()
            chain_count = db.execute("SELECT COUNT(DISTINCT chain_id) FROM chains").fetchone()[0]
            protocol_count = db.execute("SELECT COUNT(*) FROM protocols").fetchone()[0]
            # SQL aggregation avoids loading the growing index into RAM.
            delay = db.execute(f"SELECT SUM(json_extract(auction,'$.delay_value_per_second') * "
                               f"MAX(0,? - created_at)), COUNT(json_extract(auction,'$.delay_value_per_second')),"
                               f"COUNT(*) FROM jobs WHERE {pending}", (moment,)).fetchone()
            by_kind = [dict(row) for row in db.execute(
                "SELECT kind,state,COUNT(*) AS count,MIN(created_at) AS oldest_at FROM jobs GROUP BY kind,state")]
        pending_count = sum(n for state, n in counts.items() if state != "ACKED")
        return {
            "schema_version": SCHEMA_VERSION, "status": "INDEX_ONLY_RUNTIME_UNMEASURED",
            "path": str(self.path), "coordinates": sum(states.values()), "coordinate_states": states,
            "unregistered_coordinate_state": "UNSEEN", "coverage_debt": debt[0],
            "coverage_debt_measured_coordinates": debt[1], "coverage_debt_unmeasured_coordinates": debt[2] - debt[1],
            "chains": chain_count, "protocol_versions": protocol_count,
            "queue_states": counts, "pending": pending_count, "cost_blocked": blocked,
            "arrivals": arrivals, "drained": drains, "window_seconds": window,
            "growth_per_second": (arrivals - drains) / window,
            "drain_per_second": drains / window,
            "oldest_pending_age_seconds": max(0.0, moment - oldest) if oldest is not None else None,
            "oldest_high_value_job": valuable["job_id"] if valuable else None,
            "oldest_high_value_age_seconds": max(0.0, moment - valuable["created_at"]) if valuable else None,
            "cost_of_delay": delay[0], "cost_of_delay_measured_jobs": delay[1],
            "cost_of_delay_unmeasured_jobs": delay[2] - delay[1],
            "by_kind": by_kind, "exploration_floors": dict(self.exploration_floors),
            "authority": "RESEARCH_INDEX_ONLY_NO_CAPITAL_AUTHORITY",
        }
