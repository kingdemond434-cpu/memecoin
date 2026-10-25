"""Continuously expanding public-onion frontier with durable canonical outbox.

Discovery is not limited by a source/page count or crawl depth. A single paced
worker walks a disk-backed queue; host fairness and retries constrain *work*,
not admission. No scripts, login, access-challenge bypass or direct DNS path.
"""
from __future__ import annotations

import hashlib
import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from src.collectors.onion import PROXY_ENV, TorHttpClient, _Document, onion_url
from src.collectors.transports import Transport, TransportError

SOURCE_ID = "onion:continuous-frontier"


class OnionFrontierTransport(Transport):
    kind = "onion"

    def __init__(self, path, *, client_factory=None):
        super().__init__(SOURCE_ID)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=3, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS hosts(host TEXT PRIMARY KEY,last_attempt REAL NOT NULL DEFAULT 0,next_at REAL NOT NULL DEFAULT 0,turn INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS urls(url TEXT PRIMARY KEY,host TEXT NOT NULL,source_ref TEXT NOT NULL,
            discovered_at REAL NOT NULL,depth INTEGER NOT NULL,next_at REAL NOT NULL DEFAULT 0,
            observed_at REAL,digest TEXT,failures INTEGER NOT NULL DEFAULT 0,last_error TEXT NOT NULL DEFAULT '');
          CREATE INDEX IF NOT EXISTS onion_due ON urls(host,next_at);
          CREATE TABLE IF NOT EXISTS origins(url TEXT NOT NULL,source_ref TEXT NOT NULL,parent TEXT NOT NULL,
            observed_at REAL NOT NULL,PRIMARY KEY(url,source_ref,parent));
          CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY,record TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS meta(name TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        self.lock = threading.RLock()
        self.client_factory = client_factory or (lambda host: TorHttpClient(os.environ[PROXY_ENV], host))

    def _enqueue(self, urls, source_ref, observed_at, parent="", depth=0):
        if not source_ref:
            raise ValueError("durable public source provenance required")
        admitted = []
        for value in urls:
            try:
                url = onion_url(value)
            except TransportError:
                continue
            host = urlsplit(url).hostname
            admitted.append((url, host))
        self.db.executemany("INSERT OR IGNORE INTO hosts(host) VALUES(?)", ((host,) for _, host in admitted))
        self.db.executemany("INSERT OR IGNORE INTO urls(url,host,source_ref,discovered_at,depth) VALUES(?,?,?,?,?)",
                            ((url, host, source_ref, observed_at, depth) for url, host in admitted))
        self.db.executemany("INSERT OR IGNORE INTO origins VALUES(?,?,?,?)",
                            ((url, source_ref, parent, observed_at) for url, _ in admitted))

    def enqueue(self, urls, source_ref, observed_at):
        with self.lock, self.db:
            self._enqueue(urls, source_ref, observed_at)

    def migrate(self, state, legacy_directory=None):
        """Retain legacy pending/admitted URLs, once, without rewriting its archive."""
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM meta WHERE name='legacy_migrated'").fetchone():
                return
            for bucket in ("admitted", "pending"):
                for entry in state.get(bucket, {}).values():
                    self._enqueue([entry["url"]], entry["source_ref"], entry["discovered_at"])
                    if legacy_directory and entry.get('source_id'):
                        saved_path = Path(legacy_directory) / (hashlib.sha256(entry['source_id'].encode()).hexdigest() + '.json')
                        if saved_path.exists():
                            from src.collectors.onion import OnionResearchTransport
                            # Use the original strict loader, with no network
                            # client, to retain old crawl pages and unacked
                            # observations instead of abandoning their outbox.
                            legacy = OnionResearchTransport(entry['source_id'], entry['url'], source_ref=entry['source_ref'],
                                client=object(), max_pages=256, max_depth=8, state_path=saved_path)
                            self._enqueue(legacy.pages, entry['source_ref'], entry['discovered_at'])
                            for observation in legacy.pending:
                                record = {**observation, 'crawl_depth': legacy.pages[observation['url']]}
                                self.db.execute('INSERT OR IGNORE INTO outbox VALUES(?,?)',
                                                (record['observation_id'], json.dumps(record, allow_nan=False)))
            self.db.execute("INSERT INTO meta VALUES('legacy_migrated',?)", (str(state.get("capacity_debt", 0)),))

    def acknowledge(self, observation_id, canonical_job_id):
        if not canonical_job_id:
            raise TransportError("canonical research queue acknowledgement required")
        with self.lock, self.db:
            saved = self.db.execute("SELECT record FROM outbox WHERE id=?", (observation_id,)).fetchone()
            if saved is None:
                return
            record = json.loads(saved[0])
            # New children cannot be fetched until their parent's observation
            # exists in the canonical ledger, including after a process crash.
            self._enqueue(record["discovered_onion_links"], "frontier_job:" + canonical_job_id,
                          record["observed_at"], record["url"], record["crawl_depth"] + 1)
            self.db.execute("DELETE FROM outbox WHERE id=?", (observation_id,))

    def _lease(self, now):
        with self.lock, self.db:
            pending = self.db.execute("SELECT record FROM outbox ORDER BY rowid LIMIT 1").fetchone()
            if pending:
                return [json.loads(pending[0])], None
            row = self.db.execute("""SELECT u.* FROM urls u JOIN hosts h ON h.host=u.host
              WHERE u.next_at<=? AND h.next_at<=?
              ORDER BY h.last_attempt,((u.observed_at IS NULL)=(h.turn % 2)) DESC,u.next_at,u.discovered_at,u.url LIMIT 1""", (now, now)).fetchone()
            if row is None:
                return [], None
            self.db.execute("UPDATE hosts SET last_attempt=?,next_at=?,turn=turn+1 WHERE host=?", (now, now + 2, row["host"]))
            # Lease survives cancellation/restart; timed-out pages are retried.
            self.db.execute("UPDATE urls SET next_at=? WHERE url=?", (now + 60, row["url"]))
            return [], row

    def _store_observation(self, row, observed, revision, record):
        with self.lock, self.db:
            self.db.execute("UPDATE urls SET observed_at=?,digest=?,failures=0,last_error='',next_at=? WHERE url=?",
                            (observed, revision, observed + 120, row["url"]))
            if revision != row["digest"]:
                self.db.execute("INSERT INTO outbox VALUES(?,?)", (record["observation_id"], json.dumps(record, allow_nan=False)))

    async def fetch(self):
        now = time.time()
        pending, row = await asyncio.to_thread(self._lease, now)
        if row is None:
            return pending
        client = self.client_factory(row["host"])
        try:
            status, body, headers = await client.get(row["url"])
            if status != 200:
                try:
                    retry = float(headers.get("Retry-After", 0))
                except (ValueError, TypeError):
                    retry = 0
                if not 0 <= retry < float('inf'):
                    retry = 0
                await asyncio.to_thread(self._defer, row, "HTTP_" + str(status), now,
                                        max(3600 if status in {401, 403, 429} else 120, retry))
                return []
            document = _Document(link_limit=None)
            document.feed(body)
            content = " ".join(" ".join(document.text).split())
            if not content or re.search(r"(?:complete|solve|enter|verify).{0,40}captcha|proof.of.work required|"
                                       r"(?:enable|requires?) javascript|sign in to (?:continue|view)", content[:2000], re.I):
                await asyncio.to_thread(self._defer, row, "PUBLIC_CONTENT_OR_ACCESS_CHALLENGE", now, 3600)
                return []
            from src.collectors.onion_discovery import PATTERN, candidate_url
            links = set()
            for href in document.links:
                try:
                    links.add(onion_url(urljoin(row["url"], href)))
                except TransportError:
                    pass
            for match in PATTERN.finditer(content):
                try:
                    links.add(candidate_url(match.group()))
                except TransportError:
                    pass
            digest = hashlib.sha256(content.encode()).hexdigest()
            observed = time.time()
            # Links participate in change detection: pagination links can
            # change while visible text stays the same.
            revision = hashlib.sha256(json.dumps([digest, sorted(links)]).encode()).hexdigest()
            record = {"text": content, "url": row["url"], "source_ref": row["source_ref"],
                "content_hash": digest, "observed_at": observed, "published_at": None,
                "observation_id": hashlib.sha256(json.dumps([row["url"], revision, observed]).encode()).hexdigest(),
                "observation_kind": "initial_snapshot" if row["observed_at"] is None else "content_change",
                "discovered_onion_links": sorted(links), "crawl_depth": row["depth"],
                "text_truncated": False, "transport": "tor_socks5_remote_dns"}
            await asyncio.to_thread(self._store_observation, row, observed, revision, record)
            return [record] if revision != row["digest"] else []
        except Exception as exc:
            await asyncio.to_thread(self._defer, row, type(exc).__name__, now, 120)
            raise TransportError("public onion frontier fetch failed: " + type(exc).__name__) from None
        finally:
            await client.close()

    def _defer(self, row, reason, now, delay):
        with self.lock, self.db:
            self.db.execute("UPDATE urls SET failures=failures+1,last_error=?,next_at=? WHERE url=?",
                            (reason, now + max(delay, min(86400, 120 * 2 ** min(row["failures"], 9))), row["url"]))
            self.db.execute("UPDATE hosts SET next_at=MAX(next_at,?) WHERE host=?", (now + delay, row["host"]))

    def report(self):
        with self.lock:
            counts = self.db.execute("SELECT count(*),count(observed_at),max(observed_at),sum(failures>0) FROM urls").fetchone()
            pending = self.db.execute("SELECT count(*) FROM outbox").fetchone()[0]
            hosts = self.db.execute("SELECT count(*) FROM hosts").fetchone()[0]
            oldest = self.db.execute("SELECT min(discovered_at) FROM urls WHERE observed_at IS NULL").fetchone()[0]
            legacy = self.db.execute("SELECT value FROM meta WHERE name='legacy_migrated'").fetchone()
        return {**super().report(), "scope": "CONTINUOUS_PUBLIC_ONION_DISCOVERY", "execution_authority": False,
            "hostname_resolution": "SOCKS5_REMOTE", "discovered_hosts": hosts, "pages": counts[0],
            "observed_pages": counts[1], "last_observed_at": counts[2], "deferred_pages": counts[3] or 0,
            "pending_pages": counts[0] - counts[1], "pending_canonical_delivery": pending,
            "oldest_unobserved_at": oldest, "legacy_candidate_capacity_debt": int(legacy[0]) if legacy else 0,
            "source_count_limit": None, "page_count_limit": None, "crawl_depth_limit": None,
            "workers": 1, "status": "OBSERVED" if counts[1] else "NOT_OBSERVED",
            "all_onion_coverage": False}

    async def stop(self):
        def close():
            with self.lock:
                self.db.close()
        await asyncio.to_thread(close)
