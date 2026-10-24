"""Quant-derived durable exact-work recall and honest research yield reporting.

Retest avoidance requires identical native context AND inputs AND algorithm.
Descriptive findings and missing data are never recorded as trading failures.
Operational yield is reported separately from measured economic value.
"""
import hashlib
import json
import math
import time
from contextlib import contextmanager


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ResearchMemory:
    def __init__(self, db):
        self.db = db
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS research_memory(
          id TEXT PRIMARY KEY,family TEXT NOT NULL,context TEXT NOT NULL,inputs TEXT NOT NULL,
          algorithm TEXT NOT NULL,result TEXT NOT NULL,result_hash TEXT NOT NULL,
          seconds REAL NOT NULL,records INTEGER NOT NULL,created REAL NOT NULL,hits INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS research_memory_family ON research_memory(family,created);
        """)

    @contextmanager
    def transaction(self):
        if self.db.in_transaction: yield
        else:
            with self.db: yield

    @staticmethod
    def identity(family, context, inputs, algorithm):
        if not family or not algorithm or not isinstance(context, dict) or not context.get("venue"):
            raise ValueError("research family, algorithm and explicit native venue context required")
        return seal([family, context, inputs, algorithm])

    def recall(self, family, context, inputs, algorithm):
        key = self.identity(family, context, inputs, algorithm)
        row = self.db.execute("SELECT result,result_hash FROM research_memory WHERE id=?", (key,)).fetchone()
        if not row: return None
        result = json.loads(row[0])
        if seal(result) != row[1]: raise ValueError("research memory result integrity failed")
        with self.transaction(): self.db.execute("UPDATE research_memory SET hits=hits+1 WHERE id=?", (key,))
        return result

    def retain(self, family, context, inputs, algorithm, result, *, seconds, records):
        if not isinstance(result, dict) or not math.isfinite(seconds) or seconds < 0 or type(records) is not int or records < 0:
            raise ValueError("measured duration and nonnegative record count required")
        key = self.identity(family, context, inputs, algorithm)
        checksum = seal(result)
        old = self.db.execute("SELECT result_hash FROM research_memory WHERE id=?", (key,)).fetchone()
        if old and old[0] != checksum: raise ValueError("same frozen research inputs produced conflicting results")
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO research_memory(id,family,context,inputs,algorithm,result,result_hash,seconds,records,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (key, family, json.dumps(context, sort_keys=True), seal(inputs), algorithm,
                             json.dumps(result, sort_keys=True, allow_nan=False), checksum, seconds, records, time.time()))
        return key

    def report(self):
        families = []
        for row in self.db.execute("SELECT family,count(*),sum(seconds),sum(records),sum(hits) FROM research_memory GROUP BY family"):
            families.append({"family": row[0], "distinct_studies": row[1], "measured_seconds": row[2],
                             "records_analyzed": row[3], "exact_reuse_hits": row[4],
                             "economic_delta_log_wealth": None, "economic_roi": None})
        return {"families": families, "authority": "RESEARCH_ONLY", "economic_value": "UNMEASURED",
                "descriptive_reports_are_not_survivors": True, "no_cross_venue_negative_transfer": True}
