"""Global breadth: launches, prices, supply and flow, from wherever answers.

The existing miner set is good and narrow. It reads the chain directly, and it
reads two US aggregators for everything the chain does not carry. That is a
fine desk with two blind spots that matter, and this module closes both.

**Discovery outside our own stream.** The program stream sees Pump and
PumpSwap. A token that launches on Raydium, or migrates, or is created by a
program we have not decoded, is invisible to it -- and the launch census, the
denominator the whole promotion ladder rests on, is only as complete as
discovery is. So new pools are mined from operators who watch every Solana
program rather than the two we decode, and a launch appearing there and not in
our census is reported as exactly that: a hole in our coverage, not an absence
in the world.

**The world outside the US session.** Korean, Japanese, Chinese-language,
Indian, Turkish, Indonesian, Thai, Latin American and African venues price the
same assets in their own currencies against their own flow. Two things fall
out of reading them that cannot be read anywhere else: a regional risk
appetite that leads the US session by hours, and the moment a token gets a new
regional market -- which is a step change in its buyer base and is invisible
in USD pairs until after it has happened.

Every miner here goes through the substitution registry rather than holding a
URL. That is the whole point: when an endpoint refuses this address or moves a
path, the miner does not fail, it asks the next operator and records which one
answered. Provenance survives the substitution -- every record says which rung
produced it -- so a model trained on this can never confuse "the price from
our preferred oracle" with "the price from whoever was up".

Rate limits are respected per endpoint by the pool, and per rung by the
registry. A public venue that asks us to slow down is not a broken venue, and
a desk that retries into a limit is a desk that gets its address blocked and
then reports the whole region as dead.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from src.research.data_miners import (
    CADENCE_DAILY, CADENCE_FAST, CADENCE_HOURLY, CADENCE_MINUTE,
    CADENCE_QUARTER, DataMinerPool, Enriches, MinerSpec, RateLimited,
)
from src.research.source_substitution import Endpoint, SubstitutionRegistry

logger = logging.getLogger(__name__)

#: Records kept from one venue pass. A ticker endpoint returns every market it
#: lists -- thousands, most of them irrelevant -- and handing all of them
#: downstream costs more than the handful that matter are worth.
VENUE_RECORD_CAP = 400


async def _fetch_json(client: Any, url: str,
                      headers: Optional[Dict[str, str]] = None) -> Any:
    """One JSON fetch, with a rate limit distinguished from a failure.

    A 429 means the endpoint works and wants us to wait; counting it as a
    failure rotates away from a healthy source and spends the ladder on the
    wrong problem. A 403 from a public endpoint is almost always the address
    being refused rather than the request being wrong, and it is reported with
    that reading attached so nobody debugs a correct query.
    """
    status, body, _headers = await client.get(url, headers=headers)
    if status == 429:
        raise RateLimited(url)
    if status == 403:
        raise RuntimeError(
            f"HTTP 403 from {url.split('?')[0]} -- public endpoint refusing "
            "this address, not a malformed request")
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from {url.split('?')[0]}")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"response was not JSON: {exc}") from exc


#: Substrings that identify a failure originating INSIDE this process
#: rather than at the endpoint. Matched on the message because the
#: exception types are shared with genuine network faults -- aiohttp raises
#: RuntimeError for both "your session is on the wrong loop" and nothing
#: else useful, so the text is the only discriminator available.
LOCAL_FAULT_MARKERS = (
    # A session used from a loop other than the one that created it.
    "Timeout context manager should be used inside a task",
    "attached to a different loop",
    "Event loop is closed",
    "no running event loop",
    # The session was closed underneath a caller still using it.
    "Session is closed",
    "Connector is closed",
)


def _is_local_fault(exc: BaseException) -> bool:
    """True when the failure is ours, so no endpoint should be blamed for it."""
    message = f"{type(exc).__name__}: {exc}"
    return any(marker in message for marker in LOCAL_FAULT_MARKERS)


class LadderFetcher:
    """Ask a domain's live rung; on failure, rotate and try the next one.

    Bounded by `max_attempts` per pass rather than walking the whole ladder.
    A pass that tries six operators has spent six timeouts to answer a
    question whose whole value was being fast, and the miner is about to be
    called again anyway.
    """

    def __init__(self, client: Any, registry: SubstitutionRegistry,
                 *, max_attempts: int = 3):
        self.client = client
        self.registry = registry
        self.max_attempts = max(1, int(max_attempts))
        #: Faults that were ours, kept separate from anything an endpoint did.
        self.local_faults = 0
        self.last_local_fault = ""

    async def get(self, domain: str, **params: Any) -> tuple:
        """Returns (payload, endpoint). Raises only when the ladder is spent."""
        errors: List[str] = []
        rate_limited = False
        for _ in range(self.max_attempts):
            endpoint = self.registry.current(domain)
            if endpoint is None:
                break
            url = endpoint.format(**params)
            try:
                payload = await _fetch_json(self.client, url)
            except RateLimited:
                # A limit is the endpoint working. Do not quarantine it; the
                # pool's own per-miner backoff is the right instrument, and
                # rotating here would spend a healthy rung on a busy minute.
                rate_limited = True
                errors.append(f"{endpoint.name}: rate limited")
                break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                errors.append(f"{endpoint.name}: {reason}")
                if _is_local_fault(exc):
                    # Our bug, not their outage. Quarantining an operator
                    # for a fault that never left this process is how a
                    # cross-loop aiohttp error took down four ladders at
                    # once and got reported as the world declining to
                    # answer. The rung stays live and the fault is raised
                    # so it is fixed rather than absorbed.
                    self.local_faults += 1
                    self.last_local_fault = reason
                    logger.error("LADDER %s: local fault, not the endpoint's "
                                 "(%s): %s", domain, endpoint.name, reason)
                    raise
                self.registry.note_failure(domain, endpoint.name, reason)
                continue
            self.registry.note_success(domain, endpoint.name)
            return payload, endpoint
        if rate_limited:
            raise RateLimited("; ".join(errors))
        raise RuntimeError(
            f"every rung of {domain} declined this pass: " + "; ".join(errors)
            if errors else f"no endpoint available for {domain}")


def _stamp(endpoint: Endpoint, record: Dict[str, Any]) -> Dict[str, Any]:
    """Which rung produced this. Survives every substitution, on purpose.

    A model trained on a lake that cannot distinguish the preferred oracle
    from the fallback has learned the fallback's biases as though they were
    the market's.
    """
    return {**record, "_source": endpoint.name, "_region": endpoint.region,
            "data_status": "OK"}


# --- discovery -----------------------------------------------------------

def new_pools_miner(fetcher: LadderFetcher,
                    on_discovery: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                    ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Pools created recently, from operators who watch every Solana program.

    The census is the denominator for everything downstream, and it is only as
    complete as discovery is. A pool here that our own stream never reported
    is a decoder gap or a program we do not know about, which is precisely the
    failure that looks like a quiet market from the inside.
    """
    async def fetch() -> List[Dict[str, Any]]:
        payload, endpoint = await fetcher.get("new_pools")
        records = _parse_pools(payload, endpoint)
        if records and on_discovery is not None:
            try:
                on_discovery(records)
            except Exception as exc:
                logger.warning("new-pool consumer raised: %s", exc)
        return records

    return fetch


def _parse_pools(payload: Any, endpoint: Endpoint) -> List[Dict[str, Any]]:
    """One parser per payload shape, chosen by the rung's declared shape."""
    records: List[Dict[str, Any]] = []
    if endpoint.shape == "geckoterminal_pools":
        for item in (payload or {}).get("data") or []:
            attributes = item.get("attributes") or {}
            relationships = item.get("relationships") or {}
            base = ((relationships.get("base_token") or {}).get("data") or {})
            mint = str(base.get("id") or "").split("_")[-1]
            records.append(_stamp(endpoint, {
                "pool": attributes.get("address", ""),
                "mint": mint,
                "name": attributes.get("name", ""),
                "created_at": attributes.get("pool_created_at", ""),
                "liquidity_usd": _number(attributes.get("reserve_in_usd")),
                "fdv_usd": _number(attributes.get("fdv_usd")),
                "venue": "geckoterminal",
            }))
    elif endpoint.shape == "dexscreener_profiles":
        for item in payload or []:
            if str(item.get("chainId", "")).lower() != "solana":
                continue
            records.append(_stamp(endpoint, {
                "mint": item.get("tokenAddress", ""),
                "url": item.get("url", ""),
                "description": (item.get("description") or "")[:1000],
                "links": item.get("links") or [],
                "venue": "dexscreener",
            }))
    elif endpoint.shape == "raydium_pools":
        rows = ((payload or {}).get("data") or {}).get("data") or []
        for item in rows:
            records.append(_stamp(endpoint, {
                "pool": item.get("id", ""),
                "mint": ((item.get("mintA") or {}).get("address")
                         or (item.get("mintB") or {}).get("address") or ""),
                "liquidity_usd": _number(item.get("tvl")),
                "venue": "raydium",
            }))
    return [row for row in records if row.get("mint") or row.get("pool")]


def _number(value: Any) -> Optional[float]:
    """A number, or None. Never a zero standing in for an unparseable value.

    An unmeasured liquidity is not an empty pool, and the direction that error
    runs is the expensive one.
    """
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# --- supply control ------------------------------------------------------

def supply_control_miner(fetcher: LadderFetcher,
                         tokens: Callable[[], Sequence[str]],
                         *, per_pass: int = 8,
                         ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Mint authority, freeze authority, LP lock, top-holder concentration.

    A rug is a supply event before it is a price event. This reads the same
    facts the chain carries, from an operator who has already assembled them,
    which is worth a round trip when the alternative is four RPC calls per
    token at the exact moment latency matters.
    """
    async def fetch() -> List[Dict[str, Any]]:
        watched = [mint for mint in tokens() if mint][:per_pass]
        records: List[Dict[str, Any]] = []
        for mint in watched:
            try:
                payload, endpoint = await fetcher.get("supply_control", mint=mint)
            except RateLimited:
                raise
            except Exception as exc:
                # One token failing is not the miner failing. Record nothing
                # for it -- never a zero -- and carry on with the rest.
                logger.debug("supply control for %s: %s", mint, exc)
                continue
            parsed = _parse_supply(payload, endpoint, mint)
            if parsed:
                records.append(parsed)
        return records

    return fetch


def _parse_supply(payload: Any, endpoint: Endpoint, mint: str) -> Optional[Dict[str, Any]]:
    if endpoint.shape not in ("rugcheck", "rugcheck_summary"):
        return None
    if not isinstance(payload, dict):
        return None
    risks = payload.get("risks") or []
    return _stamp(endpoint, {
        "mint": mint,
        "mint_authority_present": payload.get("mintAuthority") not in (None, ""),
        "freeze_authority_present": payload.get("freezeAuthority") not in (None, ""),
        "lp_locked_pct": _number(payload.get("lpLockedPct")),
        "top_holder_pct": _number(
            ((payload.get("topHolders") or [{}])[0] or {}).get("pct")),
        "risk_count": len(risks),
        "risk_names": [str(risk.get("name", ""))[:80] for risk in risks[:12]],
        "score": _number(payload.get("score")),
    })


# --- regional flow -------------------------------------------------------

def venue_ticker_miner(fetcher: LadderFetcher, domain: str,
                       *, symbols: Sequence[str] = ("SOL", "BTC"),
                       ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """One venue's markets, normalised to (symbol, quote, last, volume).

    Read for two things, neither of which is a price we would trade on. The
    first is regional risk appetite: SOL priced in KRW moving while SOL in USD
    does not is Korean flow, and it leads. The second is market COUNT -- a
    venue listing a market it did not list yesterday is a step change in a
    token's buyer base, and it is invisible in USD pairs until afterwards.
    """
    wanted = tuple(symbol.upper() for symbol in symbols)

    async def fetch() -> List[Dict[str, Any]]:
        payload, endpoint = await fetcher.get(domain)
        rows = _parse_tickers(payload, endpoint)
        if wanted:
            focused = [row for row in rows
                       if any(symbol in str(row.get("symbol", "")).upper()
                              for symbol in wanted)]
        else:
            focused = rows
        # The market count is a measurement about the venue itself and is the
        # reason to read the whole payload even when only two symbols matter.
        summary = _stamp(endpoint, {
            "venue": endpoint.name, "kind": "venue_summary",
            "markets": len(rows), "symbol": "",
        })
        return [summary, *focused[:VENUE_RECORD_CAP]]

    return fetch


def _parse_tickers(payload: Any, endpoint: Endpoint) -> List[Dict[str, Any]]:
    """Normalise a dozen venue dialects into one row shape.

    Every venue names the same three numbers differently and nests them
    differently. Doing this once here is what lets everything downstream treat
    a Korean venue and a global one as the same kind of observation.
    """
    shape = endpoint.shape
    rows: List[Dict[str, Any]] = []

    def add(symbol: Any, last: Any, volume: Any = None, quote: str = "") -> None:
        price = _number(last)
        if not symbol or price is None:
            return
        rows.append(_stamp(endpoint, {
            "venue": endpoint.name, "kind": "ticker",
            "symbol": str(symbol), "quote": quote,
            "last": price, "volume": _number(volume),
        }))

    if shape == "binance_ticker":
        for item in payload or []:
            add(item.get("symbol"), item.get("lastPrice"), item.get("quoteVolume"))
    elif shape == "okx_ticker":
        for item in (payload or {}).get("data") or []:
            add(item.get("instId"), item.get("last"), item.get("volCcy24h"))
    elif shape == "bybit_ticker":
        for item in ((payload or {}).get("result") or {}).get("list") or []:
            add(item.get("symbol"), item.get("lastPrice"), item.get("turnover24h"))
    elif shape == "gate_ticker":
        for item in payload or []:
            add(item.get("currency_pair"), item.get("last"), item.get("quote_volume"))
    elif shape == "bitget_ticker":
        for item in (payload or {}).get("data") or []:
            add(item.get("symbol"), item.get("lastPr") or item.get("close"),
                item.get("quoteVolume"))
    elif shape == "kucoin_ticker":
        for item in ((payload or {}).get("data") or {}).get("ticker") or []:
            add(item.get("symbol"), item.get("last"), item.get("volValue"))
    elif shape == "htx_ticker":
        for item in (payload or {}).get("data") or []:
            add(item.get("symbol"), item.get("close"), item.get("vol"))
    elif shape == "upbit_markets":
        for item in payload or []:
            market = str(item.get("market", ""))
            if not market:
                continue
            quote = market.split("-")[0]
            rows.append(_stamp(endpoint, {
                "venue": endpoint.name, "kind": "market_listed",
                "symbol": market, "quote": quote,
                "name": item.get("korean_name") or item.get("english_name") or "",
                "last": None, "volume": None,
            }))
    elif shape == "bithumb_ticker":
        data = (payload or {}).get("data") or {}
        for symbol, item in data.items():
            if not isinstance(item, dict):
                continue
            add(symbol, item.get("closing_price"), item.get("acc_trade_value_24H"),
                quote="KRW")
    elif shape == "coinone_ticker":
        for item in (payload or {}).get("tickers") or []:
            add(item.get("target_currency"), item.get("last"),
                item.get("quote_volume"), quote="KRW")
    elif shape == "bitflyer_markets":
        for item in payload or []:
            code = item.get("product_code")
            if not code:
                continue
            rows.append(_stamp(endpoint, {
                "venue": endpoint.name, "kind": "market_listed",
                "symbol": str(code), "quote": "JPY", "last": None, "volume": None,
            }))
    elif shape == "gmo_ticker":
        for item in (payload or {}).get("data") or []:
            add(item.get("symbol"), item.get("last"), item.get("volume"), quote="JPY")
    elif shape == "bitbank_ticker":
        tickers = ((payload or {}).get("data") or {}).get("tickers") or []
        for item in tickers:
            add(item.get("pair"), item.get("last"), item.get("vol"), quote="JPY")
    elif shape == "indodax_ticker":
        for pair, item in ((payload or {}).get("tickers") or {}).items():
            if isinstance(item, dict):
                add(pair, item.get("last"), item.get("vol_idr"), quote="IDR")
    elif shape == "bitkub_ticker":
        for pair, item in (payload or {}).items():
            if isinstance(item, dict):
                add(pair, item.get("last"), item.get("quoteVolume"), quote="THB")
    elif shape == "coindcx_ticker":
        for item in payload or []:
            add(item.get("market"), item.get("last_price"), item.get("volume"))
    elif shape == "wazirx_ticker":
        for pair, item in (payload or {}).items():
            if isinstance(item, dict):
                add(pair, item.get("last"), item.get("volume"), quote="INR")
    elif shape == "btcturk_ticker":
        for item in (payload or {}).get("data") or []:
            add(item.get("pair"), item.get("last"), item.get("volume"), quote="TRY")
    elif shape == "paribu_ticker":
        for pair, item in (payload or {}).items():
            if isinstance(item, dict):
                add(pair, item.get("last"), item.get("volume"), quote="TRY")
    elif shape == "bitso_ticker":
        for item in (payload or {}).get("payload") or []:
            add(item.get("book"), item.get("last"), item.get("volume"))
    elif shape == "mercado_ticker":
        for item in payload or []:
            add(item.get("pair"), item.get("last"), item.get("vol"), quote="BRL")
    elif shape == "luno_ticker":
        for item in (payload or {}).get("tickers") or []:
            add(item.get("pair"), item.get("last_trade"), item.get("rolling_24_hour_volume"))
    elif shape == "valr_ticker":
        for item in payload or []:
            add(item.get("currencyPair"), item.get("lastTradedPrice"),
                item.get("quoteVolume"))
    return rows


def market_regime_miner(fetcher: LadderFetcher,
                        ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Total market state and crowd risk appetite. A regime input, not a trigger."""
    async def fetch() -> List[Dict[str, Any]]:
        payload, endpoint = await fetcher.get("market_context")
        shape = endpoint.shape
        if shape == "coingecko_global":
            data = (payload or {}).get("data") or {}
            return [_stamp(endpoint, {
                "kind": "global",
                "total_market_cap_usd": _number(
                    (data.get("total_market_cap") or {}).get("usd")),
                "total_volume_usd": _number(
                    (data.get("total_volume") or {}).get("usd")),
                "market_cap_change_24h": _number(
                    data.get("market_cap_change_percentage_24h_usd")),
            })]
        if shape == "paprika_global":
            return [_stamp(endpoint, {
                "kind": "global",
                "total_market_cap_usd": _number((payload or {}).get("market_cap_usd")),
                "total_volume_usd": _number((payload or {}).get("volume_24h_usd")),
                "market_cap_change_24h": _number(
                    (payload or {}).get("market_cap_change_24h")),
            })]
        if shape == "coinlore_global":
            first = (payload or [{}])[0] if isinstance(payload, list) else {}
            return [_stamp(endpoint, {
                "kind": "global",
                "total_market_cap_usd": _number(first.get("total_mcap")),
                "total_volume_usd": _number(first.get("total_volume")),
            })]
        if shape == "llama_overview":
            return [_stamp(endpoint, {
                "kind": "dex_volume",
                "total_volume_usd": _number((payload or {}).get("total24h")),
                "change_1d": _number((payload or {}).get("change_1d")),
            })]
        if shape == "fear_greed":
            rows = (payload or {}).get("data") or []
            if not rows:
                return []
            return [_stamp(endpoint, {
                "kind": "risk_appetite",
                "fear_greed": _number(rows[0].get("value")),
                "classification": rows[0].get("value_classification", ""),
            })]
        return []

    return fetch


# --- wallet breadth ------------------------------------------------------

def wallet_holdings_miner(rpc: Any, wallets: Callable[[], Sequence[str]],
                          *, per_pass: int = 6,
                          ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """What tracked wallets currently HOLD, not merely what they did.

    A signature history says a wallet bought something. A holdings snapshot
    says whether it still has it, which is the difference between a wallet
    that called a launch and one that called it and left. The second is the
    only one worth following.

    Rotated a few wallets per pass on purpose: the full elite set is hundreds
    of addresses, `getTokenAccountsByOwner` is one round trip each, and a
    miner that asks for all of them at once is a miner that rate limits the
    RPC the execution path also depends on.
    """
    cursor = {"at": 0}
    TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    async def fetch() -> List[Dict[str, Any]]:
        watched = [wallet for wallet in wallets() if wallet]
        if not watched:
            return []
        chosen = [watched[(cursor["at"] + offset) % len(watched)]
                  for offset in range(min(per_pass, len(watched)))]
        cursor["at"] = (cursor["at"] + len(chosen)) % len(watched)
        records: List[Dict[str, Any]] = []
        for wallet in chosen:
            try:
                result = await rpc.request(
                    "getTokenAccountsByOwner",
                    [wallet, {"programId": TOKEN_PROGRAM},
                     {"commitment": "confirmed", "encoding": "jsonParsed"}])
            except Exception as exc:
                logger.debug("holdings for %s: %s", wallet, exc)
                continue
            for account in (result or {}).get("value") or []:
                info = (((account.get("account") or {}).get("data") or {})
                        .get("parsed") or {}).get("info") or {}
                amount = (info.get("tokenAmount") or {})
                units = _number(amount.get("uiAmount"))
                if not units:
                    # A closed or emptied token account is not a holding.
                    continue
                records.append({
                    "wallet": wallet, "mint": info.get("mint", ""),
                    "units": units,
                    "decimals": amount.get("decimals"),
                    "data_status": "OK",
                })
        return records

    return fetch


def wallet_activity_miner(rpc: Any, wallets: Callable[[], Sequence[str]],
                          *, per_pass: int = 8, limit: int = 25,
                          ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Recent signatures per tracked wallet: the raw material for flow graphs.

    Deliberately shallow and wide rather than deep and narrow. Twenty-five
    signatures across many wallets finds the wallet that just woke up; a
    thousand signatures for one wallet finds nothing that a full history
    backfill would not find later and more cheaply.
    """
    cursor = {"at": 0}

    async def fetch() -> List[Dict[str, Any]]:
        watched = [wallet for wallet in wallets() if wallet]
        if not watched:
            return []
        chosen = [watched[(cursor["at"] + offset) % len(watched)]
                  for offset in range(min(per_pass, len(watched)))]
        cursor["at"] = (cursor["at"] + len(chosen)) % len(watched)
        records: List[Dict[str, Any]] = []
        for wallet in chosen:
            try:
                result = await rpc.request(
                    "getSignaturesForAddress",
                    [wallet, {"limit": int(limit), "commitment": "confirmed"}])
            except Exception as exc:
                logger.debug("activity for %s: %s", wallet, exc)
                continue
            for row in result or []:
                records.append({
                    "wallet": wallet,
                    "signature": row.get("signature", ""),
                    "slot": row.get("slot"),
                    "block_time": row.get("blockTime"),
                    "err": bool(row.get("err")),
                    "data_status": "OK",
                })
        return records

    return fetch


# --- registration --------------------------------------------------------

def register_regional_miners(pool: DataMinerPool, *, http: Any, rpc: Any,
                             registry: SubstitutionRegistry,
                             watched_tokens: Callable[[], Sequence[str]],
                             tracked_wallets: Callable[[], Sequence[str]],
                             on_discovery: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                             ) -> Dict[str, bool]:
    """Declare the breadth set.

    Cadence is set by how fast the underlying thing can change and by the
    politeness the endpoint expects, whichever is slower. New pools are mined
    on the minute because a launch is only interesting for minutes; venue
    tickers quarter-hourly because a regional risk appetite that moved in the
    last ninety seconds is noise; market regime hourly because it is a regime.
    """
    fetcher = LadderFetcher(http, registry)
    registrations = (
        (MinerSpec(
            miner_id="global:new_pools", enriches=Enriches.TOKEN_METADATA,
            cadence_seconds=CADENCE_MINUTE, endpoint="new_pools ladder",
            detail="launches outside our own program stream; census completeness"),
         new_pools_miner(fetcher, on_discovery)),
        (MinerSpec(
            miner_id="global:supply_control", enriches=Enriches.SUPPLY_CONTROL,
            cadence_seconds=CADENCE_MINUTE, endpoint="supply_control ladder",
            detail="mint and freeze authority, LP lock, top-holder concentration"),
         supply_control_miner(fetcher, watched_tokens)),
        (MinerSpec(
            miner_id="global:venue_tickers", enriches=Enriches.MARKET_CONTEXT,
            cadence_seconds=CADENCE_QUARTER, endpoint="venue_tickers ladder",
            detail="global venue flow; the denominator for regional divergence"),
         venue_ticker_miner(fetcher, "venue_tickers")),
        (MinerSpec(
            miner_id="global:regional_venues", enriches=Enriches.MARKET_CONTEXT,
            cadence_seconds=CADENCE_QUARTER, endpoint="regional_venues ladder",
            detail="KR/JP/IN/ID/TH/TR/LATAM/AFRICA flow in local currency"),
         venue_ticker_miner(fetcher, "regional_venues")),
        (MinerSpec(
            miner_id="global:market_regime", enriches=Enriches.MARKET_CONTEXT,
            cadence_seconds=CADENCE_HOURLY, endpoint="market_context ladder",
            detail="total cap, DEX volume, crowd risk appetite"),
         market_regime_miner(fetcher)),
        (MinerSpec(
            miner_id="chain:wallet_holdings", enriches=Enriches.WALLET_HISTORY,
            cadence_seconds=CADENCE_MINUTE, endpoint="getTokenAccountsByOwner",
            detail="what tracked wallets still hold, not merely what they did"),
         wallet_holdings_miner(rpc, tracked_wallets)),
        (MinerSpec(
            miner_id="chain:wallet_activity", enriches=Enriches.WALLET_HISTORY,
            cadence_seconds=CADENCE_MINUTE, endpoint="getSignaturesForAddress",
            detail="shallow and wide; finds the wallet that just woke up"),
         wallet_activity_miner(rpc, tracked_wallets)),
    )
    return {spec.miner_id: pool.register(spec, fetch)
            for spec, fetch in registrations}
