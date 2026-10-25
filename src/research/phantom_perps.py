"""Phantom perps' underlying Hyperliquid public market research.

No wallet queries, account credentials, exchange/order methods or signing.
The provider universe is broader than Phantom's verified UI listing universe.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from email.utils import parsedate_to_datetime
from decimal import Decimal, ROUND_UP, InvalidOperation
from pathlib import Path

from src.research.data_miners import Enriches, MinerSpec, RateLimited, rotating_selector
from src.research.perps_acquisition import PublicPerpsAcquisition
from src.research.perps_paper import PerpsPaperLedger
from src.research.perps_snapshots import _fetch_snapshots, _marks, WS

INFO = "https://api.hyperliquid.xyz/info"
PHANTOM_REFERENCE = "https://help.phantom.com/articles/42292249388307"


async def public_info(http, payload):
    allowed = {"perpDexs": {"type"}, "spotMeta": {"type"}, "metaAndAssetCtxs": {"type", "dex"},
               "l2Book": {"type", "coin"}, "fundingHistory": {"type", "coin", "startTime"}}
    if payload.get("type") not in allowed or set(payload) - allowed[payload["type"]]:
        raise ValueError("unreviewed public perps information request")
    session = await http.session()
    async with session.post(INFO, json=payload, allow_redirects=False) as response:
        if response.status == 429:
            error = RateLimited("Hyperliquid public information rate limited")
            try:
                delay = float(response.headers.get("Retry-After", 60))
                error.retry_after_seconds = delay if math.isfinite(delay) and delay >= 0 else 60
            except (TypeError, ValueError):
                try:
                    error.retry_after_seconds = max(0, parsedate_to_datetime(response.headers["Retry-After"]).timestamp()-time.time())
                except (ValueError, TypeError, KeyError, OverflowError):
                    error.retry_after_seconds = 60
            raise error
        if response.status != 200:
            raise RuntimeError("Hyperliquid public information HTTP " + str(response.status))
        return await response.json()


def _decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid perps measurement") from None
    if not number.is_finite():
        raise ValueError("non-finite perps measurement")
    return number


def book_cost_scenario(book, size_decimals, *, observed_at, now=None,
                       fee_bps="9.5", notional_usdc="10.25", collateral_verified=False):
    """Static displayed-depth scenario, not an order or a confirmed paper fill.

    Fee is Phantom's published baseline scenario, not a verified account tier
    or deployer-specific fee. Future funding and dynamic impact remain unknown.
    """
    now = time.time() if now is None else now
    if not collateral_verified:
        return {"status": "DATA_BLOCKED_COLLATERAL_MAPPING"}
    book_time = book.get("time")
    if type(book_time) is not int or not 0 <= now - book_time / 1000 <= 10 or not 0 <= now - observed_at <= 10:
        return {"status": "DATA_BLOCKED_STALE_BOOK"}
    if type(size_decimals) is not int or not 0 <= size_decimals <= 18:
        raise ValueError("invalid perps size precision")
    levels = book.get("levels")
    if not isinstance(levels, list) or len(levels) != 2 or not all(levels):
        return {"status": "DATA_BLOCKED_BOOK_DEPTH"}
    parsed = [[(_decimal(row["px"]), _decimal(row["sz"])) for row in side] for side in levels]
    bids, asks = parsed
    if any(px <= 0 or size < 0 for side in parsed for px, size in side):
        raise ValueError("invalid perps book level")
    if ([row[0] for row in bids] != sorted((row[0] for row in bids), reverse=True)
            or [row[0] for row in asks] != sorted(row[0] for row in asks) or bids[0][0] >= asks[0][0]):
        raise ValueError("unordered or crossed perps book")
    notional, fee = _decimal(notional_usdc), _decimal(fee_bps)
    if notional <= 0 or fee < 0:
        raise ValueError("positive probe notional and nonnegative fees required")
    quantity = (notional / asks[0][0]).quantize(Decimal(1).scaleb(-size_decimals), rounding=ROUND_UP)
    def walk(side):
        left, cash = quantity, Decimal(0)
        for price, available in side:
            take = min(left, available)
            cash += price * take
            left -= take
            if not left:
                break
        return cash if not left else None
    buy, sell = walk(asks), walk(bids)
    if buy is None or sell is None:
        return {"status": "DATA_BLOCKED_BOOK_DEPTH", "quantity": str(quantity)}
    fees = (buy + sell) * fee / 10000
    return {"status": "STATIC_OBSERVED_BOOK_SCENARIO", "quantity": str(quantity),
            "buy_notional_usdc": str(buy), "sell_notional_usdc": str(sell),
            "spread_and_depth_cost_usdc": str(buy - sell), "scenario_fees_usdc": str(fees),
            "roundtrip_cost_usdc": str(buy - sell + fees), "fee_bps_per_leg": str(fee),
            "fee_basis": "scenario input; published baseline 9.5bps; account/deployer fee unverified",
            "missing_costs": ["future funding", "latency/adverse selection", "dynamic market impact", "liquidation"],
            "order_submitted": False, "real_fill": False, "simulated_strategy_fill": False}


class PhantomPerpsResearch:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=3)
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS dexes(name TEXT PRIMARY KEY,observed REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS catalogs(dex TEXT PRIMARY KEY,metadata TEXT NOT NULL,observed REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS collateral_tokens(token_index INTEGER PRIMARY KEY,metadata TEXT NOT NULL,observed REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS markets(dex TEXT NOT NULL,coin TEXT NOT NULL,active INTEGER NOT NULL,
            spec TEXT NOT NULL,context TEXT NOT NULL,observed REAL NOT NULL,PRIMARY KEY(dex,coin));
          CREATE INDEX IF NOT EXISTS perps_market_identity ON markets(coin,active);
          CREATE TABLE IF NOT EXISTS books(coin TEXT PRIMARY KEY,body TEXT NOT NULL,observed REAL NOT NULL,scenario TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS funding(coin TEXT NOT NULL,time INTEGER NOT NULL,body TEXT NOT NULL,observed REAL NOT NULL,PRIMARY KEY(coin,time));
          CREATE TABLE IF NOT EXISTS cursors(coin TEXT PRIMARY KEY,next_time INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS retained_batches(ref TEXT PRIMARY KEY,observed REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS ingestion_errors(ref TEXT NOT NULL,record_index INTEGER NOT NULL,
            error TEXT NOT NULL,PRIMARY KEY(ref,record_index));
          CREATE TABLE IF NOT EXISTS pp_ws_attempts(at REAL NOT NULL,subscriptions INTEGER NOT NULL);
        """)
        self.acquisition = PublicPerpsAcquisition(self, lambda http, payload: public_info(http, payload))
        self.paper = PerpsPaperLedger(self)

    def dex_names(self):
        with self.lock:
            return ["native", *["dex:" + row[0] for row in self.db.execute("SELECT name FROM dexes WHERE name<>'' ORDER BY name")]]

    def market_names(self):
        with self.lock:
            return [row[0] for row in self.db.execute("SELECT DISTINCT coin FROM markets WHERE active=1 ORDER BY coin")]

    def next_funding(self, coin):
        with self.lock:
            row = self.db.execute("SELECT next_time FROM cursors WHERE coin=?", (coin,)).fetchone()
            return row[0] if row else 0

    def ingest_batch(self, records, source_ref):
        """Worker-side derived transaction AFTER canonical archive acknowledgement."""
        if not isinstance(source_ref, str) or not source_ref:
            raise ValueError("canonical perps source reference required")
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM retained_batches WHERE ref=?", (source_ref,)).fetchone():
                return
            for index, record in enumerate(records):
                if record.get("provider") != "hyperliquid" or record.get("product") != "perpetual":
                    raise ValueError("foreign record in perps batch")
                self.db.execute("SAVEPOINT perps_record")
                try:
                    accepted = self._ingest(record)
                    if accepted:
                        self.paper.on_record(record,source_ref)
                except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                    self.db.execute("ROLLBACK TO perps_record")
                    # Retain the canonical reference and explicit parse error;
                    # one malformed book must not discard the successful ones.
                    self.db.execute("INSERT INTO ingestion_errors VALUES(?,?,?)",
                                    (source_ref, index, type(exc).__name__))
                finally:
                    self.db.execute("RELEASE perps_record")
            self.db.execute("INSERT INTO retained_batches VALUES(?,?)", (source_ref, time.time()))

    def ingest(self, record):
        with self.lock, self.db:
            self._ingest(record)

    def _ingest(self, record):
        role, at = record["perps_role"], float(record["observed_at"])
        if not _decimal(at).is_finite() or at <= 0:
            raise ValueError("positive finite perps observation clock required")
        # The caller owns the single SQLite transaction, including batch ACK.
        if role == "dex_inventory":
            self.db.executemany("INSERT INTO dexes VALUES(?,?) ON CONFLICT(name) DO UPDATE SET observed=MAX(observed,excluded.observed)",
                                ((name, at) for name in record["dexes"]))
        elif role == "collateral_inventory":
            tokens = record["tokens"]
            if any(type(token.get("index")) is not int for token in tokens):
                raise ValueError("invalid collateral token index")
            self.db.executemany("INSERT INTO collateral_tokens VALUES(?,?,?) ON CONFLICT(token_index) DO UPDATE SET metadata=excluded.metadata,observed=excluded.observed WHERE excluded.observed>collateral_tokens.observed",
                                ((token["index"], json.dumps(token, allow_nan=False), at) for token in tokens))
        elif role == "catalog":
            dex, meta, contexts = record["dex"], record["meta"], record["contexts"]
            previous = self.db.execute("SELECT observed FROM catalogs WHERE dex=?", (dex,)).fetchone()
            if previous and previous[0] >= at:
                return
            universe = meta["universe"]
            if len(universe) != len(contexts) or len({m["name"] for m in universe}) != len(universe):
                raise ValueError("perps metadata/context alignment failed")
            self.db.execute("INSERT INTO catalogs VALUES(?,?,?) ON CONFLICT(dex) DO UPDATE SET metadata=excluded.metadata,observed=excluded.observed",
                            (dex, json.dumps({key:value for key,value in meta.items() if key != "universe"}, allow_nan=False), at))
            self.db.execute("UPDATE markets SET active=0 WHERE dex=?", (dex,))
            self.db.executemany("""INSERT INTO markets VALUES(?,?,?,?,?,?) ON CONFLICT(dex,coin) DO UPDATE SET
                active=excluded.active,spec=excluded.spec,context=excluded.context,observed=excluded.observed""",
                ((dex, market["name"], int(not market.get("isDelisted", False)),
                  json.dumps({"asset": market, "collateral_token":meta.get("collateralToken")}, allow_nan=False),
                  json.dumps(context, allow_nan=False), at) for market, context in zip(universe, contexts)))
        elif role == "book":
            coin, book = record["coin"], record["book"]
            if book.get("coin") != coin:
                raise ValueError("perps book identity mismatch")
            if type(book.get("time")) is not int or book["time"] < 0:
                raise ValueError("perps provider book clock required")
            if book["time"] > int(at*1000)+1000:
                raise ValueError("future provider book cannot replace the current perps snapshot")
            previous = self.db.execute("SELECT body,observed FROM books WHERE coin=?", (coin,)).fetchone()
            if previous and (previous[1] >= at or json.loads(previous[0]).get("time", -1) >= book.get("time", -1)):
                return
            row = self.db.execute("SELECT spec FROM markets WHERE coin=? AND active=1", (coin,)).fetchone()
            if row is None:
                raise ValueError("perps book has no observed contract")
            spec = json.loads(row[0])
            token = self.db.execute("SELECT metadata FROM collateral_tokens WHERE token_index=?", (spec["collateral_token"],)).fetchone()
            token_meta = json.loads(token[0]) if token else {}
            # Native canonical USDC is index zero; retain all other token
            # identities without manufacturing USD-to-collateral parity.
            verified = (type(spec["collateral_token"]) is int and spec["collateral_token"] == 0
                        and token_meta.get("name") == "USDC" and token_meta.get("isCanonical") is True)
            scenario = book_cost_scenario(book, spec["asset"]["szDecimals"], observed_at=at, collateral_verified=verified)
            scenario["collateral_token_index"] = spec["collateral_token"]
            self.db.execute("INSERT INTO books VALUES(?,?,?,?) ON CONFLICT(coin) DO UPDATE SET body=excluded.body,observed=excluded.observed,scenario=excluded.scenario",
                            (coin, json.dumps(book, allow_nan=False), at, json.dumps(scenario, allow_nan=False)))
        elif role == "funding":
            coin, start, rows = record["coin"], record["start_time"], record["rows"]
            if type(start) is not int or start < 0:
                raise ValueError("invalid funding history cursor")
            for row in rows:
                if row.get("coin") != coin or type(row.get("time")) is not int or not start <= row["time"] <= at*1000:
                    raise ValueError("perps funding identity/time mismatch")
                previous = self.db.execute("SELECT body FROM funding WHERE coin=? AND time=?", (coin, row["time"])).fetchone()
                if previous and json.loads(previous[0]) != row:
                    raise ValueError("conflicting immutable perps funding observation")
            self.db.executemany("INSERT OR IGNORE INTO funding VALUES(?,?,?,?)",
                                ((coin, row["time"], json.dumps(row, allow_nan=False), at) for row in rows))
            if rows:
                next_time = max(row["time"] for row in rows) + 1
                self.db.execute("INSERT INTO cursors VALUES(?,?) ON CONFLICT(coin) DO UPDATE SET next_time=MAX(next_time,excluded.next_time)",
                                (coin, next_time))
        elif role == "mark_snapshot":
            _marks(record['frame'])
        elif role == "asset_context":
            frame = record['frame']
            if frame.get('channel')!='activeAssetCtx' or not isinstance(frame.get('data',{}).get('coin'),str):
                raise ValueError('keyed active asset context required')
            if not isinstance(frame['data'].get('ctx'),dict): raise ValueError('asset context object required')
        elif role == "request_error":
            # An explicit retained HTTP failure is not a book or zero value.
            if record.get("request_role") not in {"book","snapshot"} or not record.get("error"):
                raise ValueError("invalid perps acquisition failure")
        else:
            raise ValueError("unknown perps observation")
        return True

    def report(self):
        with self.lock:
            dexes = self.db.execute("SELECT count(*) FROM dexes").fetchone()[0]
            markets = self.db.execute("SELECT count(*),max(observed),count(DISTINCT dex) FROM markets WHERE active=1").fetchone()
            books = self.db.execute("SELECT count(*),max(observed) FROM books").fetchone()
            funding = self.db.execute("SELECT count(*) FROM funding").fetchone()[0]
            examples = [{"coin":row[0], "observed_at":row[2], **json.loads(row[1])} for row in self.db.execute(
                "SELECT coin,scenario,observed FROM books ORDER BY observed DESC LIMIT 5")]
            now = time.time()
            fresh = self.db.execute("SELECT count(*) FROM books b WHERE b.observed>=? AND "
                "json_extract(b.body,'$.time') BETWEEN ? AND ? AND EXISTS "
                "(SELECT 1 FROM markets m WHERE m.coin=b.coin AND m.active=1)",
                (now-10, int((now-10)*1000), int(now*1000))).fetchone()[0]
            batches = self.db.execute("SELECT count(*) FROM retained_batches").fetchone()[0]
            errors = self.db.execute("SELECT count(*) FROM ingestion_errors").fetchone()[0]
            paper = self.paper.report()
        return {"status": "PUBLIC_MARKET_RESEARCH" if markets[0] else "NOT_OBSERVED",
            "underlying_venue": "Hyperliquid", "phantom_reference": PHANTOM_REFERENCE,
            "discovered_dexes": dexes, "observed_catalog_dexes": markets[2], "active_markets": markets[0],
            "last_catalog_observed_at": markets[1], "books_recorded": books[0], "last_book_observed_at": books[1],
            "funding_records": funding, "funding_backfill_complete": False,
            "latest_cost_examples": examples,
            "fresh_active_books": fresh, "book_freshness_window_seconds":10,
            "canonical_batches_indexed":batches, "acquisition":self.acquisition.report(now),
            "derived_record_errors":errors, "derived_error_archive":"ingestion_errors; original records in canonical batch",
            "phantom_listing_verified_by_market": False, "live_eligibility": "UNVERIFIED",
            "execution_authority": False, "real_fills": 0, "simulated_strategy_fills": 0,
            "simulated_control_fills":paper['simulated_fills'], "paper_ledger":paper,
            "paper_scope": "independent forward book controls with fees, positions, margin, funding and liquidation scenarios; no validated alpha",
            "scope": "all discovered underlying perp DEXes; individual Phantom listings not asserted"}

    def register(self, pool, http):
        select_dex = rotating_selector(self.dex_names, 1)
        inventory_due = 0.0
        collateral_due = 0.0
        def observation(role, **fields):
            return {"provider": "hyperliquid", "product": "perpetual", "perps_role": role,
                    "observed_at": time.time(), "execution_authority": False, **fields}
        async def catalog():
            nonlocal inventory_due, collateral_due
            if time.monotonic() >= inventory_due:
                raw = await self.acquisition.fetch(http, {"type": "perpDexs"}, role="catalog", target="dex_inventory")
                if raw is None: return []
                if not isinstance(raw, list) or any(row is not None and
                        (not isinstance(row, dict) or not isinstance(row.get("name"), str)) for row in raw):
                    raise ValueError("invalid perpetual DEX inventory")
                inventory_due = time.monotonic() + 900
                return [observation("dex_inventory", dexes=["", *[r["name"] for r in raw if r is not None]])]
            if time.monotonic() >= collateral_due:
                raw = await self.acquisition.fetch(http, {"type":"spotMeta"}, role="catalog", target="collateral_inventory")
                if raw is None: return []
                if not isinstance(raw, dict) or not isinstance(raw.get("tokens"), list):
                    raise ValueError("invalid collateral inventory")
                collateral_due = time.monotonic() + 900
                return [observation("collateral_inventory", tokens=raw["tokens"])]
            dex_key = select_dex()[0]
            dex = "" if dex_key == "native" else dex_key[4:]
            raw = await self.acquisition.fetch(http, {"type": "metaAndAssetCtxs", "dex": dex}, role="catalog", target=dex_key)
            if raw is None: return []
            meta, contexts = raw
            return [observation("catalog", dex=dex, meta=meta, contexts=contexts)]
        async def book():
            return await self.acquisition.books(http, observation)
        async def funding():
            rows = []
            for coin in self.acquisition.targets("funding", 1):
                start = self.next_funding(coin)
                values = await self.acquisition.fetch(http, {"type": "fundingHistory", "coin": coin, "startTime": start}, role="funding", target=coin)
                if values is None: continue
                rows.append(observation("funding", coin=coin, start_time=start, rows=values))
            return rows
        return {spec.miner_id: pool.register(spec, fetch) for spec, fetch in (
            (MinerSpec("perps:catalog", Enriches.MARKET_CONTEXT, 15, endpoint=INFO, detail="underlying full perp DEX inventory and rotating catalogs"), catalog),
            (MinerSpec("perps:books", Enriches.VENUE_LIQUIDITY, 5, endpoint=INFO, detail="fair adaptive public book batches; shared durable provider budget and cooldown"), book),
            (MinerSpec("perps:snapshots", Enriches.EXECUTION_CONDITIONS, 5, endpoint=WS, detail="keyed public mark and oracle snapshots; adaptive fair rotation, not continuous ticks"), lambda:_fetch_snapshots(self,http)),
            (MinerSpec("perps:funding", Enriches.EXECUTION_CONDITIONS, 60, endpoint=INFO, detail="resumable provider-available funding history; no invented cashflows"), funding))}

    def close(self):
        with self.lock:
            self.db.close()
