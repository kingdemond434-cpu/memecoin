"""The concrete miners: Solana chain state and the market around it.

Every endpoint here is a documented public interface. The ones needing a
credential say which, by name, and are simply not registered without it --
a miner that runs and fails on every pass is noise, and a miner that is absent
because a key is absent is a coverage gap with a fix.

What each one is FOR, since a miner nobody consumes is dead weight:

  holder structure   who actually holds this, and how concentrated. The single
                     most predictive non-price fact about a new launch, and the
                     one a price path cannot tell you.
  token metadata     what the coin claims to be. Feeds the authenticity
                     resolver and the copycat ranker; also the first place a
                     deployer's reuse of an image or a description shows up.
  venue liquidity    where else it trades and how deep. Decides whether an
                     exit that looks feasible on the curve is feasible at all.
  market context     what the wider market was doing while this launch ran.
                     Without it every regime looks the same in the ledger, and
                     a strategy that only works in one of them looks universal.

The RPC miners take the desk's own RPC manager rather than a URL: it already
holds the endpoints, the failover and the rate discipline, and a second HTTP
client pointed at the same provider is a second thing to rate limit.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from src.research.data_miners import (
    CADENCE_DAILY, CADENCE_FAST, CADENCE_HOURLY, CADENCE_MINUTE, CADENCE_QUARTER,
    DataMinerPool, Enriches, MinerSpec, RateLimited,
)

logger = logging.getLogger(__name__)

# Public, documented, no account required.
#
# token.jup.ag/all is retired -- DNS no longer resolves it at all ("No address
# associated with hostname"), not merely 404/410, which is why this miner ran
# silent (0 records, ERROR state) rather than reporting a route to fix. The v2
# tag endpoint is the live replacement; "verified" is the closer match to what
# this miner actually wants ("the routable token universe") than the full
# unfiltered list at cache.jup.ag/tokens, which runs ~14x larger (66MB vs
# 4.7MB) for an hourly fetch on a memory-constrained box.
JUPITER_TOKENS_URL = "https://api.jup.ag/tokens/v2/tag?query=verified"
DEXSCREENER_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"
DEXSCREENER_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_PAIRS_URL = "https://api.dexscreener.com/latest/dex/tokens/{addresses}"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
COINGECKO_SOL_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=solana&vs_currencies=usd&include_24hr_change=true")

# How many mints one DexScreener pair lookup may name. Their documented cap;
# exceeding it returns an error rather than a truncated answer, so the caller
# batches instead of discovering that at runtime.
DEXSCREENER_BATCH = 30


async def _get_json(client: Any, url: str) -> Any:
    """One JSON fetch through the shared HTTP client, with 429 distinguished.

    A rate limit is not a failure and must not be counted as one: the miner
    is working and the source is asking us to wait, and conflating them makes
    a healthy miner look broken and burns its backoff on the wrong thing.
    """
    status, body, _headers = await client.get(url)
    if status == 429:
        raise RateLimited(url)
    if status >= 400:
        raise RuntimeError(f"HTTP {status} from {url}")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"response was not JSON: {exc}") from exc


# --- chain state ---------------------------------------------------------

def holder_structure_miner(rpc: Any, tokens: Callable[[], Sequence[str]],
                           *, per_pass: int = 12) -> Callable[[], Awaitable[List[Dict]]]:
    """Holder concentration for the tokens the desk is currently watching.

    `getTokenLargestAccounts` returns the top twenty holders, which is where
    the answer is: a launch whose top holder is 40% of supply and one whose
    top holder is 2% are different propositions no price path distinguishes.

    Bounded per pass and driven by what the desk is actually watching, because
    mining the holder structure of every mint on Solana is both impossible and
    useless -- the ones that matter are the ones a position might be taken in.
    """
    async def fetch() -> List[Dict[str, Any]]:
        watched = list(tokens())[:per_pass]
        records: List[Dict[str, Any]] = []
        for mint in watched:
            try:
                largest = await rpc.request(
                    "getTokenLargestAccounts", [mint, {"commitment": "confirmed"}])
                supply = await rpc.request(
                    "getTokenSupply", [mint, {"commitment": "confirmed"}])
            except Exception as exc:
                logger.debug("holder mine failed for %s: %s", mint, exc)
                continue
            holders = ((largest or {}).get("value") or [])
            total = float(((supply or {}).get("value") or {}).get("amount", 0) or 0)
            if not holders or total <= 0:
                # No holders decoded is not "well distributed". Skipped, so the
                # lake carries no row rather than a flattering one.
                continue
            amounts = [float(item.get("amount", 0) or 0) for item in holders]
            amounts.sort(reverse=True)
            records.append({
                "mint": mint,
                "supply": total,
                "holders_sampled": len(amounts),
                "top1_share": amounts[0] / total,
                "top5_share": sum(amounts[:5]) / total,
                "top10_share": sum(amounts[:10]) / total,
                "top20_share": sum(amounts[:20]) / total,
                # Herfindahl over the sampled holders. One number that rises
                # both when a few hold a lot and when there are few of them.
                "concentration_hhi": sum((value / total) ** 2 for value in amounts),
                "data_status": "OK",
            })
        return records

    return fetch


def token_metadata_miner(rpc: Any, tokens: Callable[[], Sequence[str]],
                         *, per_pass: int = 12) -> Callable[[], Awaitable[List[Dict]]]:
    """Mint authority, freeze authority and decimals for watched tokens.

    Retained mint or freeze authority is the difference between a coin that
    CAN be rugged by its creator and one that cannot, and it is a fact about
    the account rather than an inference from behaviour -- so it is worth
    reading directly rather than predicting.
    """
    async def fetch() -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for mint in list(tokens())[:per_pass]:
            try:
                info = await rpc.request(
                    "getAccountInfo",
                    [mint, {"encoding": "jsonParsed", "commitment": "confirmed"}])
            except Exception as exc:
                logger.debug("metadata mine failed for %s: %s", mint, exc)
                continue
            parsed = ((((info or {}).get("value") or {}).get("data") or {})
                      .get("parsed") or {})
            if parsed.get("type") != "mint":
                continue
            fields = parsed.get("info") or {}
            records.append({
                "mint": mint,
                "decimals": fields.get("decimals"),
                "supply": fields.get("supply"),
                # None means renounced. Recorded as the distinct thing it is:
                # "no authority" and "we did not look" must never read alike.
                "mint_authority": fields.get("mintAuthority"),
                "freeze_authority": fields.get("freezeAuthority"),
                "mint_renounced": fields.get("mintAuthority") is None,
                "freeze_renounced": fields.get("freezeAuthority") is None,
                "data_status": "OK",
            })
        return records

    return fetch


# --- market and venue ----------------------------------------------------

def jupiter_token_list_miner(client: Any) -> Callable[[], Awaitable[List[Dict]]]:
    """The routable token universe, with names and symbols.

    Slow-moving, so mined hourly. Its value is not the list -- it is that a
    brand-new mint sharing a name or symbol with something already routable is
    the copycat pattern, and this is the corpus that detects it.
    """
    async def fetch() -> List[Dict[str, Any]]:
        payload = await _get_json(client, JUPITER_TOKENS_URL)
        if not isinstance(payload, list):
            raise RuntimeError("token list did not return a list")
        # v2 keys the mint as "id", not "address" -- the field this miner's
        # whole output is keyed on, so getting it wrong reads as an empty
        # list rather than a broken mapping.
        return [{"mint": item.get("id"), "symbol": item.get("symbol"),
                 "name": item.get("name"), "decimals": item.get("decimals"),
                 "tags": item.get("tags") or [], "data_status": "OK"}
                for item in payload if isinstance(item, dict) and item.get("id")]

    return fetch


def dexscreener_profiles_miner(client: Any) -> Callable[[], Awaitable[List[Dict]]]:
    """Tokens whose teams just paid to be seen.

    A profile or a boost is money spent on attention, which is a signal about
    intent rather than about quality -- it says somebody is marketing, not that
    the coin is good. Both readings are useful and they are not the same, so
    the record says which it is rather than collapsing them into a score.
    """
    async def fetch() -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for url, kind in ((DEXSCREENER_PROFILES_URL, "profile"),
                          (DEXSCREENER_BOOSTS_URL, "boost")):
            payload = await _get_json(client, url)
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("chainId", "")).lower() not in ("solana", ""):
                    continue
                records.append({
                    "mint": row.get("tokenAddress"),
                    "promotion_kind": kind,
                    "amount": row.get("amount"),
                    "total_amount": row.get("totalAmount"),
                    "links": row.get("links") or [],
                    "description": (row.get("description") or "")[:2_000],
                    "data_status": "OK",
                })
        return [row for row in records if row.get("mint")]

    return fetch


def dexscreener_pairs_miner(client: Any, tokens: Callable[[], Sequence[str]],
                            ) -> Callable[[], Awaitable[List[Dict]]]:
    """Where a watched token trades, and how deep each venue is.

    A curve quote says what the bonding curve would pay. It says nothing about
    a pool on another venue that would pay more, or about liquidity that has
    already left -- and an exit priced on one venue while the depth sits on
    another is an exit priced wrong.
    """
    async def fetch() -> List[Dict[str, Any]]:
        watched = list(tokens())[:DEXSCREENER_BATCH]
        if not watched:
            return []
        payload = await _get_json(
            client, DEXSCREENER_PAIRS_URL.format(addresses=",".join(watched)))
        pairs = (payload or {}).get("pairs") or []
        records: List[Dict[str, Any]] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            liquidity = pair.get("liquidity") or {}
            records.append({
                "mint": (pair.get("baseToken") or {}).get("address"),
                "pair_address": pair.get("pairAddress"),
                "dex": pair.get("dexId"),
                "price_usd": pair.get("priceUsd"),
                "liquidity_usd": liquidity.get("usd"),
                "volume_24h": (pair.get("volume") or {}).get("h24"),
                "txns_5m": (pair.get("txns") or {}).get("m5"),
                "price_change_5m": (pair.get("priceChange") or {}).get("m5"),
                "created_at": pair.get("pairCreatedAt"),
                "data_status": "OK",
            })
        return [row for row in records if row.get("mint")]

    return fetch


def market_context_miner(client: Any) -> Callable[[], Awaitable[List[Dict]]]:
    """What the wider market was doing while this launch ran.

    Without it every regime looks the same in the forward ledger, and a
    strategy that only works when SOL is rising reads as one that always
    works -- which is the single most expensive way to be wrong about a
    backtest.
    """
    async def fetch() -> List[Dict[str, Any]]:
        sol = await _get_json(client, COINGECKO_SOL_URL)
        overall = await _get_json(client, COINGECKO_GLOBAL_URL)
        data = (overall or {}).get("data") or {}
        solana = (sol or {}).get("solana") or {}
        return [{
            "sol_usd": solana.get("usd"),
            "sol_change_24h": solana.get("usd_24h_change"),
            "total_market_cap_usd": (data.get("total_market_cap") or {}).get("usd"),
            "total_volume_usd": (data.get("total_volume") or {}).get("usd"),
            "btc_dominance": (data.get("market_cap_percentage") or {}).get("btc"),
            "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd"),
            "data_status": "OK",
        }]

    return fetch


# --- registration --------------------------------------------------------

def register_solana_miners(pool: DataMinerPool, *, rpc: Any, http: Any,
                           watched_tokens: Callable[[], Sequence[str]],
                           ) -> Dict[str, bool]:
    """Declare the standard set. Returns which were registered runnable.

    Cadences are chosen from what is being measured, not from appetite. Holder
    structure moves every block and is mined fast; a routable token list moves
    over hours; the market's own state moves over minutes and mining it faster
    reads the same number repeatedly while burning a public rate limit that
    everything else here shares.
    """
    registrations = (
        (MinerSpec(
            miner_id="chain:holder_structure", enriches=Enriches.HOLDER_STRUCTURE,
            cadence_seconds=CADENCE_FAST, endpoint="rpc:getTokenLargestAccounts",
            detail="top-holder concentration for watched mints"),
         holder_structure_miner(rpc, watched_tokens)),
        (MinerSpec(
            miner_id="chain:mint_authority", enriches=Enriches.TOKEN_METADATA,
            cadence_seconds=CADENCE_MINUTE, endpoint="rpc:getAccountInfo",
            detail="mint and freeze authority, decimals, supply"),
         token_metadata_miner(rpc, watched_tokens)),
        (MinerSpec(
            miner_id="market:jupiter_tokens", enriches=Enriches.TOKEN_METADATA,
            cadence_seconds=CADENCE_HOURLY, endpoint=JUPITER_TOKENS_URL,
            max_records=50_000,
            detail="routable universe; the corpus copycat detection needs"),
         jupiter_token_list_miner(http)),
        (MinerSpec(
            miner_id="market:dexscreener_promotions", enriches=Enriches.NARRATIVE,
            cadence_seconds=CADENCE_MINUTE, endpoint=DEXSCREENER_PROFILES_URL,
            detail="paid profiles and boosts; intent, not quality"),
         dexscreener_profiles_miner(http)),
        (MinerSpec(
            miner_id="market:dexscreener_pairs", enriches=Enriches.VENUE_LIQUIDITY,
            cadence_seconds=CADENCE_FAST, endpoint="dexscreener token pairs",
            detail="every venue a watched mint trades on, and its depth"),
         dexscreener_pairs_miner(http, watched_tokens)),
        (MinerSpec(
            miner_id="market:context", enriches=Enriches.MARKET_CONTEXT,
            cadence_seconds=CADENCE_QUARTER, endpoint=COINGECKO_GLOBAL_URL,
            detail="SOL price, total cap, dominance; the regime a launch ran in"),
         market_context_miner(http)),
    )
    return {spec.miner_id: pool.register(spec, fetch)
            for spec, fetch in registrations}
