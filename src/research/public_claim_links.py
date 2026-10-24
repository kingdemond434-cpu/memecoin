"""Public source-to-address claim links; an address mention never proves a role.

No raw posts, credentials or keys are exported. The source's existing canonical
research job must exist before a link can be retained. This is an observation
index, independent of wallet skill and capital promotion.
"""
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from solders.pubkey import Pubkey


def public_address(value):
    if not isinstance(value, str): return None
    if re.fullmatch(r"0x[0-9a-fA-F]{40}", value): return ("evm", value.lower())
    if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", value): return None
    try:
        return ("solana", str(Pubkey.from_string(value)))
    except ValueError:
        return None


class PublicClaimLinks:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=.5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS links(
          seq INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,source_job TEXT NOT NULL,
          source_id TEXT NOT NULL,address TEXT NOT NULL,chain TEXT NOT NULL,observed_at REAL NOT NULL,
          topics TEXT NOT NULL,publication_known INTEGER NOT NULL)""")

    def record(self, *, source_job, source_id, addresses, topics, observed_at, publication_known=False):
        if not source_job or not source_id or not math.isfinite(observed_at) or observed_at <= 0:
            raise ValueError("canonical public source job and observation time required")
        labels = sorted({str(t)[:64] for t in topics})[:20]
        with self.db:
            for value in list(addresses)[:100]:
                address = public_address(value)
                if not address: continue
                chain, normalized = address
                identity = hashlib.sha256(json.dumps([source_job, chain, normalized]).encode()).hexdigest()
                self.db.execute("INSERT OR IGNORE INTO links(id,source_job,source_id,address,chain,observed_at,topics,publication_known) VALUES(?,?,?,?,?,?,?,?)",
                                (identity, str(source_job), str(source_id)[:200], normalized, chain,
                                 observed_at, json.dumps(labels), int(publication_known is True)))

    def page(self, after=0, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("invalid public claim cursor")
        rows = self.db.execute("SELECT * FROM links WHERE seq>? ORDER BY seq LIMIT ?", (after, limit)).fetchall()
        items = [{**dict(row), "topics": json.loads(row["topics"]), "claim_status": "UNVERIFIED_PUBLIC_MENTION",
                  "entity_role": "UNKNOWN", "trade_authority": False} for row in rows]
        return {"items": items, "next_cursor": rows[-1]["seq"] if rows else after,
                "has_more": bool(rows and self.db.execute("SELECT 1 FROM links WHERE seq>? LIMIT 1", (rows[-1]["seq"],)).fetchone()),
                "status": "OBSERVED" if items else "NO_NEW_OBSERVATIONS"}

    def report(self):
        return {"links": self.db.execute("SELECT count(*) FROM links").fetchone()[0],
                "by_chain": dict(self.db.execute("SELECT chain,count(*) FROM links GROUP BY chain")),
                "insider_identity_verified": False, "trade_authority": False}

    def close(self): self.db.close()
