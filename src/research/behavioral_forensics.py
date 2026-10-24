"""Infer falsifiable mechanisms from public trades, never hidden source code.

Adapted from the basket/escalation/ablation approach in the user's quant desk:
quant@921ed183a75b190e52e2eb1cdb67b4b16c7dac8b,
desks/mt5/research/provider_reverse.py. Binary outcome and Solana semantics
are handled here instead of copying MT5 lot sizes, clocks or capital policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import time
import asyncio
import shutil
from collections import defaultdict

from src.research.prediction_markets import encode, number, PublicVenues


def normalize_trade(row):
    wallet = str(row.get("proxy_wallet", row.get("wallet", ""))).lower()
    if not re.fullmatch(r"0x[0-9a-f]{40}", wallet): raise ValueError("invalid public wallet address")
    side = row.get("side")
    quantity, price = float(number(row["size"])), float(number(row["price"]))
    stamp = float(number(row["timestamp"]))
    precision = "milliseconds" if stamp > 1e12 else "seconds"
    if precision == "milliseconds": stamp /= 1000
    if (not math.isfinite(quantity + price + stamp) or side not in {"BUY", "SELL"}
            or quantity <= 0 or not 0 < price < 1 or not 0 < stamp <= time.time() + 60):
        raise ValueError("invalid public trade")
    condition, asset = str(row["condition_id"]), str(row["token_id"])
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}", condition) or not asset.isdigit(): raise ValueError("invalid market identity")
    return {"id": hashlib.sha256(encode(row).encode()).hexdigest(), "wallet": wallet,
            "market": condition, "asset": asset, "side": side, "quantity": quantity,
            "price": price, "at": stamp, "timestamp_precision": precision,
            "outcome": str(row.get("outcome", "")), "transaction": str(row.get("transaction_hash", "")),
            "row_identity": "content_hash_without_log_index", "fee": None, "source": "polymarket_data_v2"}


def structure(trades, *, basket_seconds=300):
    """Describe observations; missing sells/fees/inventory cannot become P&L."""
    grouped = defaultdict(list)
    for trade in trades: grouped[(trade["market"], trade["asset"], trade["side"])].append(trade)
    baskets = []
    adverse_adds = escalated_adds = adds = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda t: t["at"])
        basket = []
        for trade in ordered:
            if basket and trade["at"] - basket[-1]["at"] > basket_seconds:
                baskets.append(basket); basket = []
            if basket:
                adds += 1
                sign = 1 if trade["side"] == "BUY" else -1
                adverse_adds += (trade["price"] - basket[0]["price"]) * sign < 0
                escalated_adds += trade["quantity"] > basket[-1]["quantity"] * 1.15
            basket.append(trade)
        if basket: baskets.append(basket)
    buys = [t for t in trades if t["side"] == "BUY"]
    pairs = []
    by_market = defaultdict(list)
    for trade in buys: by_market[trade["market"]].append(trade)
    for market, rows in by_market.items():
        # Exact semantic pairs only; arbitrary tokens/headlines are not hedges.
        labels = {r["outcome"].lower() for r in rows}
        if labels not in ({"yes", "no"}, {"up", "down"}): continue
        amounts, costs = defaultdict(float), defaultdict(float)
        for row in rows:
            amounts[row["outcome"].lower()] += row["quantity"]
            costs[row["outcome"].lower()] += row["price"] * row["quantity"]
        total = sum(costs[label] / amounts[label] for label in labels)
        pairs.append({"market": market, "paired_bought_quantity": min(amounts.values()),
                      "combined_average_buy_price": total, "after_fee_edge": None,
                      "simultaneous_execution_established": False,
                      "evidence_ids": [r["id"] for r in rows]})
    mechanisms = []
    if pairs: mechanisms.append("COMPLEMENTARY_BUYS_OBSERVED")
    if adds >= 10 and adverse_adds / adds >= .8: mechanisms.append("ADVERSE_ADDS_OBSERVED")
    if adds >= 10 and escalated_adds / adds >= .8: mechanisms.append("SIZE_ESCALATION_OBSERVED")
    if buys and len(buys) == len(trades): mechanisms.append("BUY_ONLY_IN_OBSERVED_SLICE")
    high_price = sum(t["price"] >= .95 for t in buys) / len(buys) if buys else None
    return {"status": "STRUCTURE_ONLY" if trades else "DATA_BLOCKED", "trades": len(trades),
            "baskets": len(baskets), "max_basket_depth": max(map(len, baskets), default=0),
            "adverse_add_rate": adverse_adds / adds if adds else None,
            "escalation_rate": escalated_adds / adds if adds else None,
            "high_price_buy_rate": high_price, "mechanisms": mechanisms,
            "complementary_buys": pairs[:50], "realized_after_cost_pnl": None,
            "skill_or_luck": "DATA_BLOCKED_NATIVE_FOLLOW_AND_CONTROL_OUTCOMES_REQUIRED",
            "latency_edge": "UNMEASURED_REFERENCE_CLOCK_AND_DECISION_FEED_REQUIRED",
            "promotion_authority": False,
            "limitations": ["public trades omit hidden inventory, transfers and historical fees",
                            "leaderboard selection creates survivorship bias",
                            "identical rows lack a unique fill/log index and may be undercounted",
                            "structure is a hypothesis, not recovered proprietary code or proven edge"]}


def paired_growth_test(rows, *, declared_trials, block_days=7, iterations=1000, seed=0):
    """Quant-style cost/control falsification preserving adjacent daily dependence."""
    if not isinstance(declared_trials, int) or declared_trials < 1: raise ValueError("preregistered trial budget required")
    if not 1 <= block_days <= 30 or not 100 <= iterations <= 10000: raise ValueError("invalid bootstrap bounds")
    ordered = sorted(rows, key=lambda r: r["decision_at"])
    daily = defaultdict(list)
    for row in ordered:
        if (not row.get("evidence_ref") or row.get("costs_measured") is not True
                or row.get("paired_same_cohort") is not True or row.get("exposure_semantics") != "daily_portfolio"):
            raise ValueError("paired native cost/evidence contract missing")
        decision, available = float(number(row["decision_at"])), float(number(row["outcome_observed_at"]))
        if not 0 < decision < available <= time.time(): raise ValueError("invalid forward chronology")
        strategy, control = float(number(row["net_return"])), float(number(row["control_net_return"]))
        if min(strategy, control) <= -1: raise ValueError("ruin or invalid return requires explicit survival analysis")
        day = int(decision // 86400)
        if day in daily: raise ValueError("one complete portfolio observation per day is required")
        daily[day].append(math.log1p(strategy) - math.log1p(control))
    if len(daily) < 30:
        return {"status": "DATA_BLOCKED", "reason": "at least 30 independently observed days required", "days": len(daily), "promotion_authority": False}
    dates = sorted(daily)
    if any(b - a != 1 for a, b in zip(dates, dates[1:])):
        return {"status": "DATA_BLOCKED", "reason": "missing daily exposure/quiet-day coverage", "days": len(daily), "promotion_authority": False}
    alpha = .05 / declared_trials
    if iterations * alpha < 5:
        return {"status": "DATA_BLOCKED", "reason": "insufficient bootstrap tail resolution for declared multiplicity",
                "days": len(daily), "familywise_alpha": alpha, "promotion_authority": False}
    values = [sum(daily[d]) for d in dates]
    rng = random.Random(seed)
    boot = []
    for _ in range(iterations):
        sample = []
        while len(sample) < len(values):
            start = rng.randrange(len(values) - min(block_days, len(values)) + 1)
            sample.extend(values[start:start + block_days])
        boot.append(statistics.mean(sample[:len(values)]))
    boot.sort()
    lower = boot[max(0, min(len(boot) - 1, int(alpha * len(boot))))]
    return {"status": "POSITIVE_RESEARCH_EVIDENCE" if lower > 0 else "NO_ESTABLISHED_EDGE",
            "days": len(values), "mean_paired_daily_log_growth": statistics.mean(values),
            "lower_bound": lower, "familywise_alpha": alpha, "declared_trials": declared_trials,
            "method": "moving_block_bootstrap_of_paired_after_cost_daily_log_growth",
            "promotion_authority": False, "real_execution_authority": False}


class PublicWalletResearch:
    """Bounded, resumable public discovery and mechanism inference, no orders."""
    def __init__(self, broker):
        self.broker, self.db = broker, broker.db
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS public_wallets(address TEXT PRIMARY KEY,origin TEXT NOT NULL,last_scan REAL NOT NULL DEFAULT 0,backfill_cursor TEXT,reported_pnl TEXT);
        CREATE TABLE IF NOT EXISTS public_trades(id TEXT PRIMARY KEY,wallet TEXT NOT NULL,at REAL NOT NULL,payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS public_trades_wallet ON public_trades(wallet,at);
        CREATE INDEX IF NOT EXISTS public_trades_at ON public_trades(at DESC);
        CREATE TABLE IF NOT EXISTS public_forensics(wallet TEXT PRIMARY KEY,at REAL NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS public_pages(hash TEXT PRIMARY KEY,at REAL NOT NULL,source TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS public_claim_hints(id TEXT PRIMARY KEY,chain TEXT NOT NULL,address TEXT NOT NULL,payload TEXT NOT NULL);
        """)
        self.cycles, self.errors, self.last_success = 0, {}, 0

    def retain_page(self, path, response):
        checksum = hashlib.sha256(encode(response).encode()).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO public_pages VALUES(?,?,?,?)", (checksum, time.time(), path, encode(response)))
        return checksum

    def wallet(self, address, origin, pnl=None):
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", str(address)): raise ValueError("invalid wallet")
        cap = 128 if origin == "unverified_public_claim" else 96
        if self.db.execute("SELECT count(*) FROM public_wallets").fetchone()[0] >= cap: return
        self.db.execute("INSERT OR IGNORE INTO public_wallets(address,origin,reported_pnl) VALUES(?,?,?)", (address.lower(), origin, str(pnl) if pnl is not None else None))

    def retain_trades(self, response):
        for row in response["data"]:
            try: trade = normalize_trade(row)
            except (ValueError, KeyError, TypeError): continue
            self.db.execute("INSERT OR IGNORE INTO public_trades VALUES(?,?,?,?)", (trade["id"], trade["wallet"], trade["at"], encode(trade)))

    async def cycle(self, venues):
        base = "https://data-api.polymarket.com"
        try:
            filename = self.db.execute("PRAGMA database_list").fetchone()[2]
            if shutil.disk_usage(filename).free < 5_000_000_000:
                self.errors["discovery"] = "DATA_BLOCKED_DISK_RESERVE: preserve at least 5 GB free"
                return
            leaders = await venues.get(base, "/v2/leaderboard", {"limit": 20, "time_period": "ALL", "order_by": "PNL"})
            recent = await venues.get(base, "/v2/trades", {"limit": 200})
            with self.db:
                self.retain_page("/v2/leaderboard", leaders)
                self.retain_page("/v2/trades", recent)
                for row in leaders["data"]: self.wallet(row["user_id"], "public_leaderboard_selected", row.get("pnl"))
                # Include a contemporaneous activity cohort, not only famous winners.
                for row in recent["data"][:40]:
                    if row.get("proxy_wallet"): self.wallet(row["proxy_wallet"], "contemporaneous_public_activity")
                self.retain_trades(recent)
            for wallet in self.db.execute("SELECT * FROM public_wallets ORDER BY last_scan,address LIMIT 4").fetchall():
                address = wallet["address"]
                try:
                    response = await venues.get(base, "/v2/trades", {"user": address, "limit": 200})
                    with self.db:
                        self.retain_page("/v2/trades?user=" + address, response)
                        self.retain_trades(response)
                    cursor = wallet["backfill_cursor"]
                    if cursor:
                        older = await venues.get(base, "/v2/trades", {"user": address, "limit": 200, "cursor": cursor})
                        with self.db:
                            self.retain_page("/v2/trades?user=" + address + "&cursor=[opaque]", older)
                            self.retain_trades(older)
                        cursor = older.get("pagination", {}).get("next_cursor")
                    elif cursor is None:
                        cursor = response.get("pagination", {}).get("next_cursor")
                    rows = self.db.execute("SELECT payload FROM public_trades WHERE wallet=? ORDER BY at DESC LIMIT 5000", (address,)).fetchall()
                    report = structure([json.loads(r[0]) for r in rows])
                    total = self.db.execute("SELECT count(*) FROM public_trades WHERE wallet=?", (address,)).fetchone()[0]
                    report.update(wallet=address, selection_origin=wallet["origin"],
                                  history_complete=not cursor and total <= 5000, analysis_window_limit=5000,
                                  analysis_truncated=total > 5000, retained_trade_count=total,
                                  reported_pnl_is_unverified=True)
                    with self.db:
                        self.db.execute("UPDATE public_wallets SET last_scan=?,backfill_cursor=? WHERE address=?", (time.time(), cursor or "", address))
                        self.db.execute("INSERT OR REPLACE INTO public_forensics VALUES(?,?,?)", (address, time.time(), encode(report)))
                        self.broker.event("public_wallet_mechanism_hypothesis", report)
                    self.errors.pop(address, None)
                except Exception as exc:
                    self.errors[address] = type(exc).__name__ + ": " + str(exc)[:100]
            self.cycles += 1
            self.last_success = time.time()
            self.errors.pop("discovery", None)
        except Exception as exc:
            self.errors["discovery"] = type(exc).__name__ + ": " + str(exc)[:100]

    async def public_claims(self, venues):
        try:
            row = self.db.execute("SELECT value FROM meta WHERE key='public_claim_cursor'").fetchone()
            cursor = int(row[0]) if row else 0
            response = await venues.get("http://127.0.0.1:18081", "/public-claims", {"after": cursor, "limit": 100})
            with self.db:
                self.retain_page("local-public-claims", response)
                for item in response["items"]:
                    if item.get("claim_status") != "UNVERIFIED_PUBLIC_MENTION" or item.get("trade_authority") is not False:
                        raise ValueError("unexpected public claim authority")
                    self.db.execute("INSERT OR IGNORE INTO public_claim_hints VALUES(?,?,?,?)",
                                    (item["id"], item["chain"], item["address"], encode(item)))
                    if item["chain"] == "evm": self.wallet(item["address"], "unverified_public_claim")
                self.db.execute("INSERT OR REPLACE INTO meta VALUES('public_claim_cursor',?)", (str(response["next_cursor"]),))
            self.errors.pop("public_claims", None)
        except Exception as exc:
            self.errors["public_claims"] = type(exc).__name__ + ": " + str(exc)[:100]

    async def run(self):
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12),
                                         headers={"User-Agent": "MemecoinPublicForensics/1.0"}) as session:
            venues = PublicVenues(session)
            while True:
                await self.cycle(venues)
                await self.public_claims(venues)
                await asyncio.sleep(120)

    def report(self):
        return {"status": "OBSERVING" if self.last_success else "DATA_BLOCKED", "cycles": self.cycles,
                "last_success": self.last_success, "wallets": self.db.execute("SELECT count(*) FROM public_wallets").fetchone()[0],
                "trades": self.db.execute("SELECT count(*) FROM public_trades").fetchone()[0],
                "forensic_reports": self.db.execute("SELECT count(*) FROM public_forensics").fetchone()[0],
                "public_claim_mentions": self.db.execute("SELECT count(*) FROM public_claim_hints").fetchone()[0],
                "public_claim_chains": dict(self.db.execute("SELECT chain,count(*) FROM public_claim_hints GROUP BY chain")),
                "source_errors": self.errors, "promotion_authority": False, "edge": "UNMEASURED"}
