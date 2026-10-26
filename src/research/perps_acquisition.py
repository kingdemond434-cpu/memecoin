"""Durable, fair public perps polling; no account or exchange requests."""
from __future__ import annotations

import asyncio
import math
import time

from src.research.data_miners import RateLimited
from src.runtime.loop_local import loop_local_semaphore


class PublicPerpsAcquisition:
    # This collector's cooperative share of the documented 1200 weight/min/IP.
    # Other applications on the IP are unknown; a provider 429 always wins.
    WEIGHT_PER_MINUTE = 900

    def __init__(self, state, request):
        self.state, self.request = state, request
        self.inflight = loop_local_semaphore(4, "public-perps-http")
        with state.lock, state.db:
            state.db.executescript("""
              CREATE TABLE IF NOT EXISTS acquisition_service(
                role TEXT NOT NULL,target TEXT NOT NULL,attempted REAL NOT NULL,
                error TEXT NOT NULL,PRIMARY KEY(role,target));
              CREATE TABLE IF NOT EXISTS acquisition_weight(seq INTEGER PRIMARY KEY AUTOINCREMENT,at REAL NOT NULL,weight INTEGER NOT NULL);
              CREATE INDEX IF NOT EXISTS acquisition_weight_at ON acquisition_weight(at);
              CREATE TABLE IF NOT EXISTS acquisition_limits(key TEXT PRIMARY KEY,value REAL NOT NULL);
            """)

    def targets(self, role, quantum):
        with self.state.lock:
            cursor = self.state.db.execute("""
              SELECT DISTINCT m.coin,COALESCE(s.attempted,0) AS serviced
              FROM markets m LEFT JOIN acquisition_service s ON s.target=m.coin AND s.role=?
              WHERE m.active=1 ORDER BY serviced,m.coin LIMIT ?
            """, (role, quantum))
            try:
                rows = cursor.fetchall()
            finally:
                cursor.close()
            return [row[0] for row in rows]

    async def fetch(self, http, payload, *, role, target):
        async with self.inflight:
            now = time.time()
            # fundingHistory has response-dependent weight. Reserve headroom
            # before issuing it; charge the observed additional weight below.
            weight = 2 if payload["type"] == "l2Book" else 20
            reserve = 70 if payload["type"] == "fundingHistory" else weight
            with self.state.lock, self.state.db:
                db = self.state.db
                db.execute("DELETE FROM acquisition_weight WHERE at<=?", (now - 60,))
                cooling = db.execute("SELECT value FROM acquisition_limits WHERE key='cooldown'").fetchone()
                used = db.execute("SELECT COALESCE(sum(weight),0) FROM acquisition_weight").fetchone()[0]
                # Keep catalog/funding opportunities when a growing book
                # universe alone could consume the entire provider allowance.
                allowance = self.WEIGHT_PER_MINUTE - (100 if role == "book" else 0)
                if (cooling and cooling[0] > now) or used + reserve > allowance:
                    return None  # no HTTP, no target advancement, no invented provider failure
                db.execute("INSERT INTO acquisition_weight(at,weight) VALUES(?,?)", (now, reserve))
                charge_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                db.execute("INSERT INTO acquisition_service VALUES(?,?,?,'') ON CONFLICT(role,target) "
                           "DO UPDATE SET attempted=excluded.attempted", (role, target, now))
            try:
                body = await self.request(http, payload)
                actual = weight + (math.ceil(len(body) / 20) if payload["type"] == "fundingHistory" and isinstance(body, list) else 0)
                with self.state.lock, self.state.db:
                    self.state.db.execute("UPDATE acquisition_weight SET weight=? WHERE rowid=?", (actual, charge_id))
                    self.state.db.execute("UPDATE acquisition_service SET error='' WHERE role=? AND target=?", (role, target))
                return body
            except Exception as exc:
                with self.state.lock, self.state.db:
                    self.state.db.execute("UPDATE acquisition_service SET error=? WHERE role=? AND target=?",
                                          (type(exc).__name__, role, target))
                    if isinstance(exc, RateLimited):
                        delay = getattr(exc, "retry_after_seconds", 60)
                        delay = delay if isinstance(delay, (int,float)) and math.isfinite(delay) and delay >= 0 else 60
                        self.state.db.execute("INSERT INTO acquisition_limits VALUES('cooldown',?) "
                                              "ON CONFLICT(key) DO UPDATE SET value=MAX(value,excluded.value)", (time.time()+max(60,delay),))
                raise

    async def books(self, http, observation):
        # Aim to revisit the observed universe each minute. Larger universes
        # stretch that objective under the shared weight envelope, never vanish.
        quantum = max(1, math.ceil(len(self.state.market_names()) / 12))
        async def one(coin):
            try:
                body = await self.fetch(http, {"type": "l2Book", "coin": coin}, role="book", target=coin)
                return observation("book", coin=coin, book=body) if body is not None else None
            except Exception as exc:
                return observation("request_error", request_role="book", coin=coin,
                                   error=type(exc).__name__)
        # Tasks are bounded by the available request weight, not market count.
        # Keep the remaining universe in SQLite for the next service quantum.
        with self.state.lock:
            used = self.state.db.execute("SELECT COALESCE(sum(weight),0) FROM acquisition_weight WHERE at>?",
                                         (time.time()-60,)).fetchone()[0]
        available = max(0, (self.WEIGHT_PER_MINUTE-used)//2)
        rows = await asyncio.gather(*(one(coin) for coin in self.targets("book", min(quantum, available))))
        return [row for row in rows if row is not None]

    def report(self, now):
        with self.state.lock:
            db = self.state.db
            errors = [{"role":r[0], "target":r[1], "error":r[2], "attempted_at":r[3]} for r in db.execute(
                "SELECT role,target,error,attempted FROM acquisition_service WHERE error<>'' ORDER BY attempted DESC LIMIT 10")]
            cooling = db.execute("SELECT value FROM acquisition_limits WHERE key='cooldown'").fetchone()
            used = db.execute("SELECT COALESCE(sum(weight),0) FROM acquisition_weight WHERE at>?", (now-60,)).fetchone()[0]
        return {"weight_last_minute":used, "collector_weight_budget_per_minute":self.WEIGHT_PER_MINUTE,
                "provider_cooldown_until":cooling[0] if cooling and cooling[0]>now else None,
                "recent_request_errors":errors, "book_refresh_objective_seconds":60,
                "book_refresh_guaranteed":False, "other_ip_consumers_measured":False}
