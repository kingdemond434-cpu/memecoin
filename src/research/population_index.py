"""Incremental durable population rows; canonical source facts stay in evidence.

One transaction commits new observations, controls and their replay receipt.
Rows are immutable by identity. The small framework descriptor is written only
after migration commits, so a missing sidecar can never become an empty desk.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import weakref

STORAGE = "population_sqlite_v1"
BUCKETS = ("actors", "outcomes", "narratives", "source_outcomes", "controls",
           "receipts", "receipt_observed_at")


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class PopulationIndex:
    def __init__(self, path, *, required=False):
        self.path = Path(path)
        if required and not self.path.is_file():
            raise RuntimeError("referenced population index is missing; refusing empty replacement")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            if str(db.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
                raise RuntimeError("population index requires WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records(
                    bucket TEXT NOT NULL, identity TEXT NOT NULL, payload TEXT NOT NULL,
                    digest TEXT NOT NULL, PRIMARY KEY(bucket,identity)) WITHOUT ROWID;
            """)
            version = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            if required and not version:
                raise RuntimeError("referenced population index is uninitialized")
            if version and version[0] != "1":
                raise RuntimeError("unsupported population index schema")
        keeper = sqlite3.connect(str(self.path), timeout=15, check_same_thread=False)
        keeper.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        self._close_keeper = weakref.finalize(self, keeper.close)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(str(self.path), timeout=15)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @property
    def descriptor(self):
        return {"storage": STORAGE, "schema": 1, "file": self.path.name,
                "source_facts": "CANONICAL_EVIDENCE_REFERENCES"}

    @staticmethod
    def _put(db, changes):
        for bucket, rows in changes.items():
            if bucket not in BUCKETS or not isinstance(rows, dict):
                raise ValueError("known population row bucket required")
            for identity, value in rows.items():
                if not isinstance(identity, str):
                    raise ValueError("string population row identity required")
                payload = _encoded(value)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                old = db.execute("SELECT digest,payload FROM records WHERE bucket=? AND identity=?",
                                 (bucket, identity)).fetchone()
                if old:
                    if old != (digest, payload):
                        raise ValueError("immutable population row conflicts with committed identity")
                else:
                    db.execute("INSERT INTO records VALUES(?,?,?,?)", (bucket, identity, payload, digest))

    def initialize(self, legacy):
        if set(legacy) != {"schema", *BUCKETS} or legacy["schema"] != 1:
            raise ValueError("unsupported legacy population checkpoint shape")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            exists = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            if not exists:
                self._put(db, {name: legacy[name] for name in BUCKETS})
                db.execute("INSERT INTO metadata VALUES('schema','1')")
        if self.load() != legacy:
            raise RuntimeError("population migration differs from retained legacy checkpoint")

    def apply(self, changes):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._put(db, changes)

    def load(self):
        result = {"schema": 1, **{name: {} for name in BUCKETS}}
        with self._db() as db:
            if not db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone():
                raise RuntimeError("population index is uninitialized")
            for bucket, identity, payload, digest in db.execute("SELECT bucket,identity,payload,digest FROM records"):
                if bucket not in BUCKETS or hashlib.sha256(payload.encode()).hexdigest() != digest:
                    raise RuntimeError("population index row integrity failed")
                result[bucket][identity] = json.loads(payload)
        return result

    def close(self):
        self._close_keeper()
