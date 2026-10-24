"""Shared, durable reservations for a configured RPC credential.

These are conservative HTTP allocation units, not provider-billed credits.
WebSocket bytes and other applications using the same key remain unmeasured.
No credential or credential-bearing URL is stored in this database.
"""
import asyncio
import hashlib
import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


class RPCBudgetExhausted(RuntimeError):
    def __init__(self, retry_at):
        super().__init__("configured RPC allocation exhausted")
        self.retry_at = retry_at


class RPCBudget:
    def __init__(self, path, credential, *, rps=10, daily=None, monthly=None):
        if not credential or not math.isfinite(rps) or rps <= 0 or any(v is not None and (type(v) is not int or v <= 0) for v in (daily, monthly)):
            raise ValueError("valid RPC rate and optional operator allocations required")
        if daily is not None and monthly is not None and daily > monthly: raise ValueError('daily allocation exceeds monthly allocation')
        self.rps, self.daily, self.monthly = float(rps), daily, monthly
        self.identity = hashlib.sha256(credential.encode()).hexdigest()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=3, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS rpc_allocation(provider TEXT PRIMARY KEY,day TEXT,month TEXT,daily INTEGER,monthly INTEGER,next_at REAL,next_send REAL)")
        self.db.commit()
        self.lock = threading.RLock()

    @staticmethod
    def cost(method):
        # Reserve conservatively for archival/history-heavy methods; actual
        # plan-specific billing and provider-side credits are not inferred.
        return 10 if method in {"getProgramAccounts", "getTransaction", "getBlock", "getSignaturesForAddress", "getAsset", "getAssetsByOwner"} else 1

    def reserve(self, methods, now=None):
        now = time.time() if now is None else now
        if not methods or len(methods) > 100: raise ValueError("bounded RPC method list required")
        if methods.count('sendTransaction') > 1: raise ValueError('sendTransaction reservations require individual rate slots')
        clock = datetime.fromtimestamp(now, timezone.utc)
        day, month = clock.strftime("%Y-%m-%d"), clock.strftime("%Y-%m")
        with self.lock, self.db:
            # Serialize read/modify/write across the main desk and sidecar,
            # including across separate Python processes and restarts.
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute("SELECT day,month,daily,monthly,next_at,next_send FROM rpc_allocation WHERE provider=?", (self.identity,)).fetchone()
            used_day = row[2] if row and row[0] == day else 0
            used_month = row[3] if row and row[1] == month else 0
            cost = sum(self.cost(m) for m in methods)
            midnight = clock.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            if self.monthly is not None and used_month + cost > self.monthly:
                reset = (clock.replace(day=28, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=4)).replace(day=1)
                raise RPCBudgetExhausted(reset.timestamp())
            if self.daily is not None and used_day + cost > self.daily: raise RPCBudgetExhausted(midnight.timestamp())
            delay = max(0, (row[4] if row else 0) - now,
                        (row[5] if row and "sendTransaction" in methods else 0) - now)
            if delay > 0: return delay
            next_send = now + 1 if "sendTransaction" in methods else row[5] if row else 0
            self.db.execute("INSERT OR REPLACE INTO rpc_allocation VALUES(?,?,?,?,?,?,?)",
                            (self.identity, day, month, used_day+cost, used_month+cost,
                             now+len(methods)/self.rps, next_send))
        return 0

    async def acquire(self, methods):
        while True:
            delay = self.reserve(methods)
            if not delay: return
            await asyncio.sleep(min(delay, 1))

    def report(self):
        clock = datetime.now(timezone.utc)
        with self.lock:
            row = self.db.execute("SELECT day,month,daily,monthly FROM rpc_allocation WHERE provider=?", (self.identity,)).fetchone()
        return {"reserved_daily_units": row[2] if row and row[0] == clock.strftime("%Y-%m-%d") else 0,
                "reserved_monthly_units": row[3] if row and row[1] == clock.strftime("%Y-%m") else 0,
                "daily_limit": self.daily, "monthly_limit": self.monthly, "requests_per_second": self.rps,
                "measurement": "conservative_HTTP_reservations_not_provider_billing",
                "websocket_and_external_application_usage": "UNMEASURED"}

    def close(self):
        with self.lock: self.db.close()
