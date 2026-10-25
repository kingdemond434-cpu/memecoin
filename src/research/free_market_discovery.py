"""Keyless, expanding GeckoTerminal discovery with canonical retention ACKs.

Provider network names are identifiers, not verified native chain mappings.
Latest pools, trending and the provider's accessible pagination are a sample,
never a complete launch census. No credentials, signing or execution methods.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.research.data_miners import Enriches, MinerSpec, RateLimited

ORIGIN = "https://api.geckoterminal.com"
BASE = ORIGIN + "/api/v2"
DOCUMENTATION = ORIGIN + "/docs/index.html"
MINER = "free:geckoterminal_pages"


def identifier(value):
    if (not isinstance(value, str) or value in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value)):
        raise ValueError("invalid provider identifier")
    return value


def page_url(role, network, page):
    if type(page) is not int or page < 1:
        raise ValueError("positive provider page required")
    if role == "networks" and network == "":
        return BASE + "/networks?page=" + str(page)
    if role not in {"new_pools", "trending_pools", "dexes"}:
        raise ValueError("unreviewed free information route")
    return BASE + "/networks/" + identifier(network) + "/" + role + "?page=" + str(page)


def next_page(payload, url, current):
    """Honor pagination only within the exact reviewed route; never fetch a link."""
    links = payload.get("links")
    if isinstance(links, dict) and "next" in links:
        if links["next"] is None:
            return None
        target, source = urlparse(str(links["next"])), urlparse(url)
        query = parse_qs(target.query)
        if (target.scheme != "https" or target.netloc != source.netloc
                or target.path != source.path or target.fragment
                or set(query) != {"page"} or len(query["page"]) != 1):
            raise ValueError("untrusted provider pagination link")
        value = query["page"][0]
        if not value.isdecimal() or int(value) != current + 1:
            raise ValueError("nonsequential provider pagination")
        return int(value)
    # Public pool pages sometimes omit links. An empty page closes only this
    # traversal; a provider refusal is recorded separately, never as completion.
    return current + 1 if payload["data"] else None


def pool_rows(page):
    """Preserve both token identities; no symbol or address suffix guessing."""
    if (page["role"] not in {"new_pools", "trending_pools"} or page["http_status"] != 200
            or page.get("parse_status", "OK") != "OK"):
        return []
    network, rows = page["network"], []
    for item in page["payload"]["data"]:
        if not isinstance(item, dict):
            continue
        attrs, rels = item.get("attributes") or {}, item.get("relationships") or {}
        if not isinstance(attrs, dict) or not isinstance(rels, dict):
            continue
        def relation(role):
            value = rels.get(role)
            data = value.get("data") if isinstance(value, dict) else None
            return data.get("id") if isinstance(data, dict) else None
        def token(role):
            value = relation(role)
            prefix = network + "_"
            return value[len(prefix):] if isinstance(value, str) and value.startswith(prefix) else None
        base, quote = token("base_token"), token("quote_token")
        pool = attrs.get("address")
        if not base or not quote or not isinstance(pool, str) or not pool:
            continue
        def number(key):
            value = attrs.get(key)
            if isinstance(value, bool):
                return None
            try:
                value = float(value)
                return value if math.isfinite(value) and value >= 0 else None
            except (ValueError, TypeError):
                return None
        dex = relation("dex")
        rows.append({"provider": "geckoterminal", "discovery_kind": "pool_observation",
            "network_id": network, "chain_id": "geckoterminal:" + network,
            "native_chain_mapping": "UNVERIFIED", "pool": pool, "token": base,
            "quote_token": quote, "dex": dex, "name": attrs.get("name"),
            "provider_created_at": attrs.get("pool_created_at"),
            "price_usd": number("base_token_price_usd"), "liquidity_usd": number("reserve_in_usd"),
            "fdv_usd": number("fdv_usd"), "volume_windows": attrs.get("volume_usd"),
            "transaction_windows": attrs.get("transactions"),
            "price_change_windows": attrs.get("price_change_percentage"),
            "observed_at": page["observed_at"], "source_url": page["source_url"],
            "page_digest": page["digest"], "quote_status": "INDICATIVE_PROVIDER_OBSERVATION",
            "execution_eligible": False, "native_execution": False})
    return rows


class FreeMarketDiscovery:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=3)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS jobs(role TEXT,network TEXT,page INTEGER NOT NULL,
            due REAL NOT NULL,last_served REAL NOT NULL DEFAULT 0,last_error TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(role,network));
          CREATE TABLE IF NOT EXISTS networks(id TEXT PRIMARY KEY,body TEXT,observed REAL,ref TEXT);
          CREATE TABLE IF NOT EXISTS dexes(network TEXT,id TEXT,body TEXT,observed REAL,ref TEXT,PRIMARY KEY(network,id));
          CREATE TABLE IF NOT EXISTS pools(network TEXT,address TEXT,body TEXT,observed REAL,ref TEXT,PRIMARY KEY(network,address));
          CREATE TABLE IF NOT EXISTS pages(digest TEXT PRIMARY KEY,role TEXT,network TEXT,page INTEGER,
            status INTEGER,parsed TEXT,observed REAL,rows INTEGER,ref TEXT);
          CREATE TABLE IF NOT EXISTS pending(id INTEGER PRIMARY KEY CHECK(id=1),body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS pacing(id INTEGER PRIMARY KEY CHECK(id=1),next_at REAL NOT NULL);
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO jobs(role,network,page,due) VALUES('networks','',1,0)")
            # A useful bootstrap, not an allowlist: every returned network is admitted.
            for role in ("new_pools", "trending_pools", "dexes"):
                self.db.execute("INSERT OR IGNORE INTO jobs(role,network,page,due) VALUES(?,'solana',1,0)", (role,))

    async def fetch(self, http):
        now = time.time()
        with self.lock:
            pending = self.db.execute("SELECT body FROM pending WHERE id=1").fetchone()
            if pending:
                page = json.loads(pending[0])
                return [page, *pool_rows(page)]
            pacing = self.db.execute("SELECT next_at FROM pacing WHERE id=1").fetchone()
            if pacing and pacing[0] > now:
                return []
            # Alternate source roles as well as networks. Hundreds of cold
            # DEX-directory jobs must not precede every fresh-pool page just
            # because their role sorts first. Preserve reviewed network-page
            # traversal priority and canonical ACK before cursor advancement.
            job = self.db.execute("SELECT j.role,j.network,j.page FROM jobs j JOIN (SELECT role,max(last_served) AS served FROM jobs GROUP BY role) r ON j.role=r.role WHERE j.due<=? ORDER BY CASE WHEN j.role='networks' THEN 0 ELSE 1 END,r.served,j.due,j.last_served,j.role,j.network LIMIT 1", (now,)).fetchone()
            if not job:
                return []
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO pacing VALUES(1,?)", (now + 15,))
        role, network, page_no = job
        url = page_url(role, network, page_no)
        status, body, headers = await http.get(url, headers={"Accept": "application/json;version=20230203"}, allow_redirects=False)
        observed = time.time()
        if status == 429:
            try:
                retry = max(60, float(headers.get("Retry-After", "60")))
                if not math.isfinite(retry):
                    retry = 60
            except (ValueError, TypeError):
                retry = 60
            with self.lock, self.db:
                self.db.execute("INSERT OR REPLACE INTO pacing VALUES(1,?)", (observed + retry,))
            raise RateLimited("GeckoTerminal public provider cooldown")
        parsed = "OK"
        if status == 200:
            try:
                def refuse_constant(_value):
                    raise ValueError("nonfinite JSON constant")
                payload = json.loads(body, parse_constant=refuse_constant)
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    raise ValueError("unrecognized free discovery response")
                following = next_page(payload, url, page_no)
            except (ValueError, TypeError):
                # Retain the original bytes and an explicit gap, then let
                # other network jobs progress instead of sticking at one row.
                payload, following, parsed = {"data": []}, None, "INVALID_SCHEMA_OR_PAGINATION"
        else:
            payload, following, parsed = {"data": []}, None, "HTTP_REFUSAL"
        digest = hashlib.sha256((url + "\n" + str(observed) + "\n" + body).encode()).hexdigest()
        page = {"provider": "geckoterminal", "discovery_kind": "source_page", "role": role,
                "network": network, "page": page_no, "http_status": status, "parse_status": parsed,
                "source_url": url, "raw_body": body, "payload": payload, "next_page": following,
                "observed_at": observed, "digest": digest, "documentation": DOCUMENTATION,
                "selection_bias": "provider indexed latest/trending pools; pagination and cache constrained",
                "execution_eligible": False}
        with self.lock, self.db:
            self.db.execute("INSERT INTO pending VALUES(1,?)", (json.dumps(page, allow_nan=False),))
        return [page, *pool_rows(page)]

    def discovery_records(self, page, ref):
        if page["http_status"] != 200 or page.get("parse_status", "OK") != "OK":
            return []
        rows = []
        for item in page["payload"]["data"]:
            if not isinstance(item, dict):
                continue
            kind = "chain" if page["role"] == "networks" else "protocol" if page["role"] == "dexes" else None
            if kind is None:
                continue
            try:
                key = identifier(item.get("id"))
            except ValueError:
                continue
            network = key if kind == "chain" else page["network"]
            chain = "geckoterminal:" + network
            rows.append({"kind": kind, "identity": chain if kind == "chain" else chain + ":" + key,
                         "chain_id": chain, "chain_family": "UNKNOWN", "data_source": "geckoterminal:public-api",
                         "source_response_ref": ref, "observed_at": page["observed_at"],
                         "contract": {"status": "PROVIDER_IDENTIFIER_NATIVE_MAPPING_UNVERIFIED"},
                         "execution_eligible": False})
        return rows

    def acknowledge(self, records, ref):
        """Called on the miner worker only AFTER durable canonical retention."""
        if not isinstance(ref, str) or not ref:
            raise ValueError("canonical source reference required")
        page = next(row for row in records if row.get("discovery_kind") == "source_page")
        with self.lock, self.db:
            pending = self.db.execute("SELECT body FROM pending WHERE id=1").fetchone()
            if not pending:
                if self.db.execute("SELECT 1 FROM pages WHERE digest=?", (page["digest"],)).fetchone():
                    return  # replayed ACK; never move an already advanced cursor
                raise ValueError("free discovery pending page missing")
            original = json.loads(pending[0])
            if original["digest"] != page["digest"]:
                raise ValueError("free discovery pending identity mismatch")
            role, network, at = original["role"], original["network"], original["observed_at"]
            data = original["payload"]["data"]
            if original["http_status"] == 200 and original["parse_status"] == "OK":
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    if role in {"networks", "dexes"}:
                        try:
                            key = identifier(item.get("id"))
                        except ValueError:
                            continue
                        body = json.dumps(item, allow_nan=False)
                        if role == "networks":
                            self.db.execute("INSERT OR REPLACE INTO networks VALUES(?,?,?,?)", (key, body, at, ref))
                            for child in ("new_pools", "trending_pools", "dexes"):
                                self.db.execute("INSERT OR IGNORE INTO jobs(role,network,page,due) VALUES(?,?,1,?)", (child, key, at))
                        else:
                            self.db.execute("INSERT OR REPLACE INTO dexes VALUES(?,?,?,?,?)", (network, key, body, at, ref))
                for row in pool_rows(original):
                    self.db.execute("INSERT OR REPLACE INTO pools VALUES(?,?,?,?,?)", (network, row["pool"], json.dumps(row, allow_nan=False), at, ref))
            following = original["next_page"]
            error = "" if original["parse_status"] == "OK" else original["parse_status"] + ":HTTP_" + str(original["http_status"])
            # No total page/network ceiling. Refused pages remain explicit
            # gaps; each cycle revisits latest pages rather than claiming EOF.
            due = at if following else at + (3600 if role in {"networks", "dexes"} else 900)
            self.db.execute("UPDATE jobs SET page=?,due=?,last_served=?,last_error=? WHERE role=? AND network=?",
                            (following or 1, due, at, error, role, network))
            self.db.execute("INSERT OR IGNORE INTO pages VALUES(?,?,?,?,?,?,?,?,?)",
                            (original["digest"], role, network, original["page"], original["http_status"], original["parse_status"], at, len(data), ref))
            self.db.execute("DELETE FROM pending WHERE id=1")

    def register(self, pool, http):
        spec = MinerSpec(MINER, Enriches.TOKEN_METADATA, 15, endpoint=BASE,
                         detail="keyless expanding network/DEX/pool pages; canonical ACK before cursor progress")
        async def fetch():
            return await self.fetch(http)
        return {MINER: pool.register(spec, fetch)}

    def report(self):
        with self.lock:
            counts = {name: self.db.execute("SELECT count(*) FROM " + name).fetchone()[0]
                      for name in ("networks", "dexes", "pools", "pages", "pending")}
            gaps = self.db.execute("SELECT count(*) FROM pages WHERE status<>200").fetchone()[0]
            schema_gaps = self.db.execute("SELECT count(*) FROM pages WHERE parsed='INVALID_SCHEMA_OR_PAGINATION'").fetchone()[0]
        return {"status": "KEYLESS_OBSERVATION", **counts, "provider_refused_pages": gaps,
                "unparsed_pages": schema_gaps,
                "coverage": "SAMPLED_PROVIDER_INDEX; NOT_A_LAUNCH_CENSUS", "native_chain_mapping": "UNVERIFIED",
                "proprietary_wallet_labels": False, "execution_eligible": False}

    def close(self):
        with self.lock:
            self.db.close()
