"""Public prediction venues and a durable, explicitly simulated paper broker.

Only GET requests leave this module. Paper cash is independent of every wallet
and of the memecoin portfolio. A book is evidence of displayed liquidity, not
of a real fill. Strategies are frozen before observations and are never
promoted into the existing real-capital ladder by this module.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import aiohttp

D = Decimal
ONE = D("1")
ZERO = D("0")
KALSHI_QUADRATIC_TYPES = {"quadratic", "quadratic_with_maker_fees", "quadratic_with_combo_maker_fees"}


def number(value) -> Decimal:
    result = D(str(value))
    if not result.is_finite():
        raise ValueError("non-finite market value")
    return result


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


@dataclass(frozen=True)
class Market:
    venue: str
    market_id: str
    question: str
    yes: str
    no: str
    status: str
    fee_rate: str | None
    fee_kind: str
    minimum: str = "1"
    result: str = ""

    @property
    def key(self):
        return self.venue + ":" + self.market_id


@dataclass(frozen=True)
class Book:
    venue: str
    asset: str
    bids: tuple[tuple[str, str], ...]
    asks: tuple[tuple[str, str], ...]
    observed_at: float
    source_at: float
    snapshot_id: str


def levels(rows, *, reverse=False):
    result = {}
    if not isinstance(rows, list) or len(rows) > 10000:
        raise ValueError("invalid or oversized book")
    for row in rows:
        p, q = (row["price"], row["size"]) if isinstance(row, dict) else row
        price, size = number(p), number(q)
        if not ZERO < price < ONE or size < ZERO:
            raise ValueError("invalid price or size")
        if size:
            result[price] = result.get(price, ZERO) + size
    return tuple((str(p), str(result[p])) for p in sorted(result, reverse=reverse))


def make_book(venue, asset, bids, asks, observed_at, source_at=None):
    bid, ask = levels(bids, reverse=True), levels(asks)
    if bid and ask and number(bid[0][0]) >= number(ask[0][0]):
        raise ValueError("crossed or locked book cannot support conservative paper fills")
    stamp = observed_at if source_at is None else source_at
    payload = encode([venue, asset, bid, ask, stamp, observed_at])
    return Book(venue, asset, bid, ask, observed_at, stamp,
                hashlib.sha256(payload.encode()).hexdigest())


def fee(market: Market, quantity: Decimal, price: Decimal) -> Decimal:
    if market.fee_rate is None:
        raise ValueError("market-specific fee is unknown")
    rate = number(market.fee_rate)
    if rate < ZERO or rate > ONE:
        raise ValueError("invalid fee rate")
    amount = quantity * rate * price * (ONE - price)
    if market.fee_kind == "kalshi_quadratic":
        return amount.quantize(D("0.01"), rounding=ROUND_CEILING)
    if market.fee_kind == "polymarket_quadratic":
        return amount.quantize(D("0.00001"), rounding=ROUND_HALF_UP)
    raise ValueError("unsupported fee model")


class PublicVenues:
    """Bounded, timed, non-authenticated official API reads. No proxy bypass."""
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.errors = {}
        self.series_fees = {}
        self.event_fees = {}
        self.fee_evidence = {}
        self.offset = 0
        self.cursor = ""
        self.search_index = 0
        self.condition_hints = []

    async def get(self, base, path, params=None):
        # All hosts and path templates are chosen here, never from market prose.
        async with self.session.get(base + path, params=params, allow_redirects=False) as r:
            if r.status != 200:
                raise RuntimeError(f"public API HTTP {r.status}")
            body = bytearray()
            async for chunk in r.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > 4_000_000:
                    raise ValueError("public API response exceeds size limit")
            return json.loads(body)

    @staticmethod
    def polymarket(row):
        tokens = row.get("clobTokenIds", [])
        outcomes = row.get("outcomes", [])
        if isinstance(tokens, str): tokens = json.loads(tokens)
        if isinstance(outcomes, str): outcomes = json.loads(outcomes)
        labels = [str(x).lower() for x in outcomes]
        if len(tokens) != 2 or set(labels) not in ({"yes", "no"}, {"up", "down"}):
            raise ValueError("only binary YES/NO or UP/DOWN CTF markets are currently supported")
        positive, negative = ("yes", "no") if "yes" in labels else ("up", "down")
        if not all(str(x).isdigit() for x in tokens):
            raise ValueError("unsupported position identifier")
        schedule = row.get("feeSchedule") or {}
        rate = None
        if row.get("feesEnabled") is False:
            rate = "0"
        elif schedule.get("rate") is not None and number(schedule.get("exponent", 1)) == ONE:
            rate = str(number(schedule["rate"]))
        result = ""
        # A closed book or a price near one is NOT a settlement receipt.
        statuses = row.get("umaResolutionStatuses") or []
        if isinstance(statuses, str): statuses = json.loads(statuses)
        if row.get("closed") and ("resolved" in statuses or row.get("umaResolutionStatus") == "resolved"):
            prices = row.get("outcomePrices") or []
            if isinstance(prices, str): prices = json.loads(prices)
            if len(prices) == 2:
                exact = [number(x) for x in prices]
                if sorted(exact) == [ZERO, ONE]:
                    result = "yes" if labels[exact.index(ONE)] == positive else "no"
        active = (row.get("active") is True and row.get("closed") is False
                  and row.get("acceptingOrders") is True and row.get("enableOrderBook") is True)
        return Market("polymarket", str(row["id"]), str(row.get("question", "")),
                      str(tokens[labels.index(positive)]), str(tokens[labels.index(negative)]),
                      "open" if active else "closed", rate, "polymarket_quadratic",
                      str(row.get("orderMinSize", "1")), result)

    async def kalshi(self, row):
        ticker = str(row["ticker"])
        series = str(row.get("event_ticker", "")).split("-")[0]
        cached = self.series_fees.get(series)
        if not cached or time.time() - cached[0] > 60:
            data = await self.get("https://external-api.kalshi.com/trade-api/v2", "/series/" + quote(series, safe=""))
            spec = data["series"]
            # Unknown overrides are never treated as free trading.
            rate = None
            if spec.get("fee_type") in KALSHI_QUADRATIC_TYPES and spec.get("fee_multiplier") is not None:
                rate = str(D("0.07") * number(spec["fee_multiplier"]))
            cached = (time.time(), rate, spec)
            self.series_fees[series] = cached
        rate = cached[1]
        event_ticker = str(row.get("event_ticker", ""))
        changes = self.event_fees.get(event_ticker)
        if not changes or time.time() - changes[0] > 60:
            data = await self.get("https://external-api.kalshi.com/trade-api/v2", "/events/fee_changes",
                                  {"event_ticker": event_ticker, "limit": 1000})
            changes = (time.time(), data)
            self.event_fees[event_ticker] = changes
        if changes[1].get("cursor"):
            rate = None  # An incomplete fee history cannot establish a rate.
        else:
            def stamp(item):
                value = item["scheduled_ts"]
                return float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            effective = [c for c in changes[1]["event_fee_changes"] if stamp(c) <= time.time()]
            if effective:
                change = max(effective, key=stamp)
                kind, multiplier = change.get("fee_type_override"), change.get("fee_multiplier_override")
                if kind is not None or multiplier is not None:
                    kind = kind or cached[2].get("fee_type")
                    multiplier = cached[2].get("fee_multiplier") if multiplier is None else multiplier
                    rate = str(D("0.07") * number(multiplier)) if kind in KALSHI_QUADRATIC_TYPES and multiplier is not None else None
        self.fee_evidence[ticker] = {"series": cached[2], "event_fee_changes": changes[1], "observed_at": changes[0]}
        if row.get("fee_type") or row.get("fee_multiplier") is not None:
            rate = (str(D("0.07") * number(row["fee_multiplier"]))
                    if row.get("fee_type") in KALSHI_QUADRATIC_TYPES and row.get("fee_multiplier") is not None else None)
        status = "open" if row.get("status") in {"open", "active"} else "closed"
        result = str(row.get("result", "")) if row.get("status") in {"settled", "finalized"} else ""
        if result not in {"yes", "no"}: result = ""
        return Market("kalshi", ticker, str(row.get("title", "")), ticker + ":yes", ticker + ":no",
                      status, rate, "kalshi_quadratic", "1", result)

    async def discover(self, venue, limit=12):
        if venue == "polymarket":
            rows = await self.get("https://gamma-api.polymarket.com", "/markets", {
                "closed": "false", "active": "true", "limit": 50, "offset": self.offset,
                "order": "volume24hr", "ascending": "false"})
            self.offset = (self.offset + 50) % 500 if rows else 0
            highlighted = []
            query = ("bitcoin up or down", "ethereum up or down", "solana up or down")[self.search_index % 3]
            self.search_index += 1
            try:
                search = await self.get("https://gamma-api.polymarket.com", "/public-search",
                                        {"q": query, "limit_per_type": 4, "events_status": "active"})
                highlighted.extend(m for e in search.get("events", []) for m in e.get("markets", []))
                self.errors.pop("crypto_search", None)
            except (RuntimeError, ValueError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self.errors["crypto_search"] = type(exc).__name__
            crypto_rows = list(highlighted)
            highlighted = []
            hints = [c for c in self.condition_hints[:8] if re.fullmatch(r"0x[0-9a-fA-F]{64}", c)]
            if hints:
                try:
                    recent = await self.get("https://gamma-api.polymarket.com", "/markets", {"condition_ids": hints, "limit": 8})
                    highlighted.extend(r for r in recent if r.get("conditionId") in hints)
                    self.errors.pop("recent_market_lookup", None)
                except (RuntimeError, ValueError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    self.errors["recent_market_lookup"] = type(exc).__name__
            result = []
            crypto = []
            broad = []
            for row in highlighted:
                try: result.append(self.polymarket(row))
                except (ValueError, KeyError, TypeError): continue
            for row in crypto_rows:
                try: crypto.append(self.polymarket(row))
                except (ValueError, KeyError, TypeError): continue
            for row in rows:
                try: broad.append(self.polymarket(row))
                except (ValueError, KeyError, TypeError): continue
            result = ([m for m in result if m.status == "open"][:max(1, limit // 4)]
                      + [m for m in crypto if m.status == "open"][:max(1, limit // 4)] + broad)
            return list({m.key: m for m in result if m.status == "open"}.values())[:limit]
        data = await self.get("https://external-api.kalshi.com/trade-api/v2", "/markets", {
            "status": "open", "limit": 100, "mve_filter": "exclude", "cursor": self.cursor})
        self.cursor = str(data.get("cursor", ""))
        rows = [r for r in data["markets"] if not r.get("mve_collection_ticker")]
        rows.sort(key=lambda r: number(r.get("volume_24h_fp", "0")), reverse=True)
        return [await self.kalshi(r) for r in rows[:limit]]

    async def market(self, venue, market_id):
        if venue == "polymarket":
            row = await self.get("https://gamma-api.polymarket.com", "/markets/" + quote(market_id, safe=""))
            return self.polymarket(row), row
        data = await self.get("https://external-api.kalshi.com/trade-api/v2", "/markets/" + quote(market_id, safe=""))
        market = await self.kalshi(data["market"])
        return market, {**data["market"], "_paper_fee_evidence": self.fee_evidence[market.market_id]}

    async def books(self, market):
        if market.venue == "polymarket":
            result = []
            for asset in (market.yes, market.no):
                row = await self.get("https://clob.polymarket.com", "/book", {"token_id": asset})
                now = time.time()
                stamp = float(row["timestamp"]) / 1000 if row.get("timestamp") else now
                result.append(make_book("polymarket", asset, row.get("bids", []), row.get("asks", []), now, stamp))
            return result
        row = await self.get("https://external-api.kalshi.com/trade-api/v2",
                             "/markets/" + quote(market.market_id, safe="") + "/orderbook", {"depth": 100})
        ob = row["orderbook_fp"]
        yes, no = ob.get("yes_dollars", []), ob.get("no_dollars", [])
        now = time.time()
        invert = lambda items: [[str(ONE - number(p)), q] for p, q in items]
        return [make_book("kalshi", market.yes, yes, invert(no), now),
                make_book("kalshi", market.no, no, invert(yes), now)]


class PaperBroker:
    """SQLite transactions bind simulated orders, fills, cash and evidence."""
    def __init__(self, path: Path, *, initial_cash="1000", depth_fraction="0.1", max_age=30,
                 max_positions=20, max_market_cost="25"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS markets(key TEXT PRIMARY KEY,payload TEXT NOT NULL,observed REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS books(id TEXT PRIMARY KEY,market TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,receipt TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS positions(account TEXT NOT NULL,market TEXT NOT NULL,asset TEXT NOT NULL,
          quantity TEXT NOT NULL,cost TEXT NOT NULL,opened REAL NOT NULL,PRIMARY KEY(account,market,asset));
        CREATE TABLE IF NOT EXISTS accounts(id TEXT PRIMARY KEY,cash TEXT NOT NULL,realized TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,at REAL NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS consumed(account TEXT NOT NULL,book TEXT NOT NULL,side TEXT NOT NULL,price TEXT NOT NULL,quantity TEXT NOT NULL,
          PRIMARY KEY(account,book,side,price));
        """)
        self.depth_fraction = number(depth_fraction)
        if not ZERO < self.depth_fraction <= ONE: raise ValueError("invalid depth fraction")
        self.max_age, self.max_positions, self.max_market_cost = max_age, max_positions, number(max_market_cost)
        spec = {"version": 1, "initial_cash": str(number(initial_cash)), "depth_fraction": str(self.depth_fraction),
                "max_age": max_age, "max_positions": max_positions, "max_market_cost": str(self.max_market_cost),
                "strategies": ["complete_set_v1", "yes_control_v1", "no_control_v1"],
                "hold_seconds": 300, "entry_latency_seconds": 1, "max_market_entries": 1,
                "fill_model": "visible_depth_haircut_taker", "live_execution": False,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.spec_hash = hashlib.sha256(encode(spec).encode()).hexdigest()
        old = self.db.execute("SELECT value FROM meta WHERE key='spec_hash'").fetchone()
        if old and old[0] != self.spec_hash:
            self.db.close()
            raise ValueError("paper experiment specification changed; use a new database")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('spec_hash',?)", (self.spec_hash,))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('spec',?)", (encode(spec),))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('started_at',?)", (str(time.time()),))
            for venue in ("polymarket", "kalshi"):
                for strategy in spec["strategies"]:
                    self.db.execute("INSERT OR IGNORE INTO accounts VALUES(?,?,?)",
                                    (venue + ":" + strategy, str(number(initial_cash)), "0"))

    def event(self, kind, payload):
        self.db.execute("INSERT INTO events(at,kind,payload) VALUES(?,?,?)", (time.time(), kind, encode(payload)))

    @contextmanager
    def transaction(self):
        # A complete-set attempt owns an outer SAVEPOINT. Do not commit a
        # single leg out from underneath it.
        if self.db.in_transaction:
            yield
        else:
            with self.db:
                yield

    def observe(self, market, books, raw=None):
        with self.db:
            self.db.execute("INSERT INTO markets VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,observed=excluded.observed",
                            (market.key, encode(asdict(market)), time.time()))
            for book in books:
                self.db.execute("INSERT OR IGNORE INTO books VALUES(?,?,?)", (book.snapshot_id, market.key, encode(asdict(book))))
            if raw is not None: self.event("market_response", {"market": market.key, "raw": raw})
            self.event("observation", {"market": market.key, "books": [b.snapshot_id for b in books]})

    def submit(self, market: Market, book: Book, *, account: str, order_id: str, side: str,
               quantity, limit, submitted_at: float, now=None, tif="FOK"):
        now = time.time() if now is None else now
        q, cap = number(quantity), number(limit)
        request = {"market": market.key, "asset": book.asset, "account": account,
                   "side": side, "quantity": str(q), "limit": str(cap), "submitted_at": submitted_at, "tif": tif}
        old = self.db.execute("SELECT payload,receipt FROM orders WHERE id=?", (order_id,)).fetchone()
        if old:
            if old[0] != encode(request): raise ValueError("order id reused with different intent")
            return json.loads(old[1])
        if side not in {"BUY", "SELL"} or tif not in {"FOK", "FAK"} or q <= ZERO or not ZERO < cap <= ONE:
            raise ValueError("invalid paper order")
        def reject(reason):
            receipt = {"status": "REJECTED", "reason": reason, "simulated": True, "filled": "0", "order_id": order_id}
            self.db.execute("INSERT INTO orders VALUES(?,?,?,?)", (order_id, encode(request), "REJECTED", encode(receipt)))
            self.event("paper_order", receipt)
            return receipt
        with self.transaction():
            stored = self.db.execute("SELECT payload FROM books WHERE id=? AND market=?", (book.snapshot_id, market.key)).fetchone()
            if not stored or stored[0] != encode(asdict(book)): return reject("unretained book")
            if book.venue != market.venue or book.asset not in {market.yes, market.no}: return reject("market/asset mismatch")
            if market.status != "open": return reject("market is closed")
            if not account.startswith(market.venue + ":"): return reject("venue account mismatch")
            if now - book.source_at > self.max_age or now - book.observed_at > self.max_age or min(book.source_at, book.observed_at) > now + 2:
                return reject("stale or future book")
            if book.observed_at < submitted_at + 1: return reject("awaiting fresh post-decision book")
            if market.fee_rate is None: return reject("unknown fees")
            cashrow = self.db.execute("SELECT cash,realized FROM accounts WHERE id=?", (account,)).fetchone()
            if not cashrow: return reject("unknown paper account")
            position = self.db.execute("SELECT * FROM positions WHERE account=? AND market=? AND asset=?",
                                       (account, market.key, book.asset)).fetchone()
            held, basis = (number(position["quantity"]), number(position["cost"])) if position else (ZERO, ZERO)
            if side == "SELL" and q > held: return reject("short selling prohibited")
            if side == "BUY":
                n = self.db.execute("SELECT count(*) FROM positions WHERE account=? AND CAST(quantity AS REAL)>0", (account,)).fetchone()[0]
                if n >= self.max_positions and not held: return reject("position cap")
            remaining, gross, fees = q, ZERO, ZERO
            used = []
            for p, size in (book.asks if side == "BUY" else book.bids):
                price = number(p)
                if side == "BUY" and price > cap or side == "SELL" and price < cap: break
                previous = self.db.execute("SELECT quantity FROM consumed WHERE account=? AND book=? AND side=? AND price=?", (account, book.snapshot_id, side, p)).fetchone()
                available = max(ZERO, number(size) * self.depth_fraction - (number(previous[0]) if previous else ZERO))
                take = min(remaining, available)
                if take:
                    gross += take * price
                    fees += fee(market, take, price)
                    used.append((p, take))
                    remaining -= take
                if remaining <= ZERO: break
            filled = q - remaining
            if not filled or remaining and tif == "FOK": return reject("insufficient displayed depth")
            cash, realized = number(cashrow[0]), number(cashrow[1])
            cost = gross + fees
            if side == "BUY":
                exposure = sum((number(r[0]) for r in self.db.execute("SELECT cost FROM positions WHERE account=? AND market=?", (account, market.key))), ZERO)
                if cash < cost: return reject("insufficient paper cash")
                if exposure + cost > self.max_market_cost: return reject("market exposure cap")
                cash -= cost
                held += filled
                basis += cost
                opened = position["opened"] if position and number(position["quantity"]) else now
            else:
                removed = basis * filled / held
                held -= filled
                basis -= removed
                cash += gross - fees
                realized += gross - fees - removed
                opened = position["opened"]
            self.db.execute("UPDATE accounts SET cash=?,realized=? WHERE id=?", (str(cash), str(realized), account))
            self.db.execute("INSERT INTO positions VALUES(?,?,?,?,?,?) ON CONFLICT(account,market,asset) DO UPDATE SET quantity=excluded.quantity,cost=excluded.cost,opened=excluded.opened",
                            (account, market.key, book.asset, str(held), str(basis), opened))
            for p, take in used:
                previous = self.db.execute("SELECT quantity FROM consumed WHERE account=? AND book=? AND side=? AND price=?", (account, book.snapshot_id, side, p)).fetchone()
                self.db.execute("INSERT OR REPLACE INTO consumed VALUES(?,?,?,?,?)", (account, book.snapshot_id, side, p, str(take + (number(previous[0]) if previous else ZERO))))
            receipt = {"status": "SIMULATED_FILLED" if not remaining else "SIMULATED_PARTIAL", "simulated": True,
                       "order_id": order_id, "filled": str(filled), "gross": str(gross), "fee": str(fees),
                       "book": book.snapshot_id, "spec_hash": self.spec_hash, "real_fill": False}
            self.db.execute("INSERT INTO orders VALUES(?,?,?,?)", (order_id, encode(request), receipt["status"], encode(receipt)))
            self.event("paper_order", receipt)
            return receipt

    def settle(self, market: Market):
        if market.result not in {"yes", "no"}: return 0
        count = 0
        with self.db:
            for row in self.db.execute("SELECT * FROM positions WHERE market=? AND CAST(quantity AS REAL)>0", (market.key,)).fetchall():
                won = row["asset"] == (market.yes if market.result == "yes" else market.no)
                payout = number(row["quantity"]) if won else ZERO
                account = self.db.execute("SELECT cash,realized FROM accounts WHERE id=?", (row["account"],)).fetchone()
                self.db.execute("UPDATE accounts SET cash=?,realized=? WHERE id=?",
                                (str(number(account[0]) + payout), str(number(account[1]) + payout - number(row["cost"])), row["account"]))
                self.db.execute("UPDATE positions SET quantity='0',cost='0' WHERE account=? AND market=? AND asset=?", (row["account"], market.key, row["asset"]))
                self.event("simulated_settlement", {"market": market.key, "account": row["account"], "result": market.result, "payout": str(payout)})
                count += 1
        return count

    def report(self):
        accounts = []
        for row in self.db.execute("SELECT * FROM accounts ORDER BY id"):
            positions = self.db.execute("SELECT * FROM positions WHERE account=? AND CAST(quantity AS REAL)>0", (row["id"],)).fetchall()
            accounts.append({**dict(row), "open_positions": len(positions),
                             "unliquidated_cost": str(sum((number(p["cost"]) for p in positions), ZERO))})
        counts = {r[0]: r[1] for r in self.db.execute("SELECT status,count(*) FROM orders GROUP BY status")}
        return {"mode": "PAPER", "live_submission_locked": True, "production_performance": "UNMEASURED",
                "spec_hash": self.spec_hash, "started_at": float(self.db.execute("SELECT value FROM meta WHERE key='started_at'").fetchone()[0]),
                "markets": self.db.execute("SELECT count(*) FROM markets").fetchone()[0],
                "snapshots": self.db.execute("SELECT count(*) FROM books").fetchone()[0],
                "order_counts": counts, "accounts": accounts,
                "settlement_records": self.db.execute("SELECT count(*) FROM events WHERE kind='simulated_settlement'").fetchone()[0],
                "real_fills": 0, "qualifying_promotion_receipts": 0,
                "caveat": "Displayed-depth simulation omits queue competition and guarantees no executable edge."}

    def close(self):
        self.db.close()


class PredictionPaperRunner:
    def __init__(self, broker: PaperBroker, *, interval=30, markets_per_venue=8):
        self.broker, self.interval = broker, max(10, interval)
        self.limit = min(20, max(1, markets_per_venue))
        self.errors, self.last_success, self.cycles = {}, {}, 0
        self.decisions = {}
        self.condition_hints = lambda: []
        self.seen = set(r[0] for r in broker.db.execute("SELECT key FROM meta WHERE key LIKE 'entered:%'"))

    def process(self, market, books):
        now = time.time()
        self.broker.observe(market, books)
        # Existing positions always get exit priority, including after restart.
        for position in self.broker.db.execute("SELECT * FROM positions WHERE market=? AND CAST(quantity AS REAL)>0", (market.key,)).fetchall():
            if now - position["opened"] < 300: continue
            book = next(b for b in books if b.asset == position["asset"])
            self.broker.submit(market, book, account=position["account"],
                               order_id=f"exit:{position['account']}:{market.key}:{book.snapshot_id}", side="SELL",
                               quantity=position["quantity"], limit="0.00001", submitted_at=position["opened"], tif="FAK")
        if market.fee_rate is None: return
        key = "entered:" + market.key
        if key in self.seen: return
        old = self.decisions.get(market.key)
        if not old:
            self.decisions[market.key] = now
            with self.broker.db: self.broker.event("frozen_entry_decision", {"market": market.key, "at": now, "spec_hash": self.broker.spec_hash})
            return
        if now - old < 1: return
        quantity = max(ONE, number(market.minimum))
        # Controls are explicitly benchmarks, not an asserted predictive edge.
        for strategy, book in zip(("yes_control_v1", "no_control_v1"), books):
            self.broker.submit(market, book, account=market.venue + ":" + strategy,
                               order_id=f"entry:{market.key}:{strategy}", side="BUY", quantity=quantity,
                               limit="0.99", submitted_at=old)
        # Exact same-market complete set only: never fuzzy-match cross-venue rules.
        if all(b.asks for b in books) and max(b.observed_at for b in books) - min(b.observed_at for b in books) <= 2:
            costs = [quantity * number(b.asks[0][0]) + fee(market, quantity, number(b.asks[0][0])) for b in books]
            if sum(costs, ZERO) < quantity * D("0.995"):
                # Multi-leg FOK simulation commits BOTH or NEITHER. Real atomic
                # execution is not inferred from this synthetic transaction.
                self.broker.db.execute("SAVEPOINT complete_set")
                try:
                    receipts = [self.broker.submit(market, b, account=market.venue + ":complete_set_v1",
                                                  order_id=f"pair:{market.key}:{b.asset}", side="BUY", quantity=quantity,
                                                  limit=b.asks[0][0], submitted_at=old) for b in books]
                    if any(r["status"] != "SIMULATED_FILLED" for r in receipts):
                        raise ValueError("both legs must fill")
                    self.broker.db.execute("RELEASE complete_set")
                except ValueError:
                    self.broker.db.execute("ROLLBACK TO complete_set")
                    self.broker.db.execute("RELEASE complete_set")
        with self.broker.db:
            self.broker.db.execute("INSERT OR IGNORE INTO meta VALUES(?,?)", (key, str(now)))
        self.seen.add(key)
        self.decisions.pop(market.key, None)

    async def cycle(self, venues):
        venues.condition_hints = self.condition_hints()
        for venue in ("polymarket", "kalshi"):
            try:
                discovered = await venues.discover(venue, self.limit)
                # Revisit held and pending markets even when catalog pagination moves.
                tracked = self.broker.db.execute("SELECT DISTINCT m.payload FROM markets m JOIN positions p ON p.market=m.key WHERE CAST(p.quantity AS REAL)>0 AND m.key LIKE ?", (venue + ":%",)).fetchall()
                markets = {m.key: m for m in discovered}
                for row in tracked:
                    m = Market(**json.loads(row[0])); markets[m.key] = m
                for key in list(self.decisions):
                    if key.startswith(venue + ":"):
                        row = self.broker.db.execute("SELECT payload FROM markets WHERE key=?", (key,)).fetchone()
                        if row: markets[key] = Market(**json.loads(row[0]))
                semaphore = asyncio.Semaphore(4)
                async def collect(m):
                  async with semaphore:
                    try:
                        m, raw = await venues.market(venue, m.market_id)
                        self.broker.observe(m, [], raw)
                        from src.research.local_decisions import annotate_public_text
                        with self.broker.db:
                            self.broker.event("local_decision_shadow", {"market": m.key,
                                "decision": annotate_public_text(m.question + "\n" + str(raw.get("description", raw.get("rules_primary", ""))))})
                        if self.broker.settle(m): return
                        if m.status != "open": return
                        books = await venues.books(m)
                        self.process(m, books)
                        self.errors.pop(m.key, None)
                    except (ValueError, KeyError, RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        self.errors[m.key] = type(exc).__name__ + ": " + str(exc)[:150]
                await asyncio.gather(*(collect(m) for m in list(markets.values())[:self.limit + 40]))
                self.last_success[venue] = time.time()
                self.errors.pop(venue, None)
            except (ValueError, KeyError, RuntimeError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self.errors[venue] = type(exc).__name__ + ": " + str(exc)[:150]
        self.errors = dict(list(self.errors.items())[-100:])
        for key in list(self.errors):
            if key.startswith("public_catalog:"): self.errors.pop(key)
        self.errors.update({"public_catalog:" + k: v for k, v in venues.errors.items()})
        self.cycles += 1

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12),
                                         headers={"User-Agent": "MemecoinResearchPaper/1.0"}) as session:
            venues = PublicVenues(session)
            while True:
                await self.cycle(venues)
                await asyncio.sleep(self.interval)

    def report(self):
        return {**self.broker.report(), "cycles": self.cycles, "last_success": self.last_success,
                "source_errors": self.errors, "pending_decisions": len(self.decisions)}
