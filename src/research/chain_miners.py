"""Chain facts the program stream does not carry.

The Pump/PumpSwap stream tells the desk what happened to a token. It does not
tell it three things that decide whether acting on that is profitable:

  what the chain COSTS right now      `getRecentPrioritizationFees` is the only
                                      honest answer to "what does landing in
                                      the next slot actually cost". A bid
                                      derived from a constant is a bid that
                                      overpays in calm and misses in a rush,
                                      and both errors are paid every trade.

  whether the chain is HEALTHY        `getRecentPerformanceSamples` gives the
                                      real slot rate. During congestion the
                                      landing probability our sizing assumes
                                      is simply wrong, and the correct
                                      response is to trade smaller or not at
                                      all -- not to bid harder into a chain
                                      that is dropping blocks.

  whether the SUPPLY is fixed         mint authority, freeze authority and LP
                                      burn. A rug is a supply event before it
                                      is a price event, so a supply fact seen
                                      early is worth more than a price fact
                                      seen fast.

Plus the deployer's own history, which is the single most reusable prior on
Solana: the same wallet, or the same funder, launches again and again, and its
last twenty launches are public.

Everything here goes through the desk's own RPC manager. It already holds the
endpoints, the failover and the concurrency discipline; a second HTTP client
pointed at the same provider is a second thing to get rate limited.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from src.research.data_miners import (
    CADENCE_FAST, CADENCE_MINUTE, CADENCE_QUARTER, CADENCE_REALTIME,
    DataMinerPool, Enriches, MinerSpec,
)

logger = logging.getLogger(__name__)

LAMPORTS_PER_SOL = 1_000_000_000

#: Solana's nominal slot time. Used only to turn a sample's slot count into a
#: rate; the whole point of the sample is that the real rate differs from it.
NOMINAL_SLOT_S = 0.4


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Empty in, zero out is wrong here, so callers
    must check first -- an unmeasured fee is not a free one."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


# --- what the chain costs ------------------------------------------------

def priority_fee_miner(rpc: Any, accounts: Callable[[], Sequence[str]],
                       ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """The fee distribution actually paid in recent slots.

    `getRecentPrioritizationFees` reports, per slot, the minimum
    prioritization fee among transactions that touched the given accounts.
    Asked about the accounts we are about to write to, that is the real
    clearing price for the contention we are about to join -- which is a
    different and much more useful number than the chain-wide average.

    The percentiles are what the bidder wants: p50 is the going rate, p90 is
    what it costs to be confident, and the gap between them is how contested
    this particular account is right now.
    """
    async def fetch() -> List[Dict[str, Any]]:
        watched = [a for a in accounts() if a][:16]
        params: List[Any] = [watched] if watched else []
        samples = await rpc.request("getRecentPrioritizationFees", params)
        fees = [float(row.get("prioritizationFee", 0) or 0)
                for row in (samples or []) if isinstance(row, dict)]
        if not fees:
            # No sample is not "fees are zero". Nothing is emitted, so the
            # bidder keeps its previous reading and the gap is visible in
            # health rather than being papered over with a flattering number.
            return []
        paying = [fee for fee in fees if fee > 0]
        return [{
            "scope": "accounts" if watched else "chain",
            "accounts_sampled": len(watched),
            "slots_sampled": len(fees),
            # The share of sampled slots where anyone paid at all. When this
            # is low the chain is quiet and any bid lands; when it is high the
            # p50 below is a floor, not a target.
            "contested_share": len(paying) / len(fees),
            "fee_p50_lamports": _percentile(fees, 0.50),
            "fee_p75_lamports": _percentile(fees, 0.75),
            "fee_p90_lamports": _percentile(fees, 0.90),
            "fee_max_lamports": max(fees),
            "data_status": "OK",
        }]

    return fetch


def network_health_miner(rpc: Any) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """Real slot rate and transaction throughput.

    Landing probability is estimated from a model that assumes slots arrive on
    schedule. When they do not -- and during exactly the launches worth
    sniping they often do not -- that model is optimistic in the direction
    that costs money. This measures the assumption instead of trusting it.
    """
    async def fetch() -> List[Dict[str, Any]]:
        samples = await rpc.request("getRecentPerformanceSamples", [4])
        rows = [row for row in (samples or []) if isinstance(row, dict)]
        if not rows:
            return []
        records: List[Dict[str, Any]] = []
        for row in rows:
            period = float(row.get("samplePeriodSecs", 0) or 0)
            slots = float(row.get("numSlots", 0) or 0)
            txs = float(row.get("numTransactions", 0) or 0)
            if period <= 0 or slots <= 0:
                continue
            observed_slot_s = period / slots
            records.append({
                "slot": row.get("slot"),
                "sample_period_s": period,
                "slots": slots,
                "transactions": txs,
                "tps": txs / period,
                "observed_slot_seconds": observed_slot_s,
                # >1 means slots are arriving SLOWER than nominal, so every
                # latency budget expressed in slots is really longer in
                # wall-clock than the model believes.
                "slot_time_ratio": observed_slot_s / NOMINAL_SLOT_S,
                "data_status": "OK",
            })
        return records

    return fetch


# --- whether the supply is fixed -----------------------------------------

def lp_supply_miner(rpc: Any, lp_mints: Callable[[], Sequence[str]],
                    ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """LP token supply for migrated tokens: burned, locked, or still held.

    After migration the deployer's power over the pool is expressed entirely
    through LP tokens. A supply of zero means the LP is burned and the pool
    cannot be pulled; a live supply concentrated in one account means it can
    be pulled at any moment. That is the difference between a token worth
    holding through a drawdown and one worth exiting on the first wobble, and
    no candle shows it.
    """
    async def fetch() -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for lp_mint in list(lp_mints())[:12]:
            try:
                supply = await rpc.request(
                    "getTokenSupply", [lp_mint, {"commitment": "confirmed"}])
            except Exception as exc:
                logger.debug("lp supply mine failed for %s: %s", lp_mint, exc)
                continue
            value = (supply or {}).get("value") or {}
            amount = float(value.get("amount", 0) or 0)
            record: Dict[str, Any] = {
                "lp_mint": lp_mint,
                "lp_supply": amount,
                "lp_burned": amount == 0.0,
                "data_status": "OK",
            }
            if amount > 0:
                try:
                    largest = await rpc.request(
                        "getTokenLargestAccounts",
                        [lp_mint, {"commitment": "confirmed"}])
                    holders = ((largest or {}).get("value") or [])
                    amounts = sorted(
                        (float(h.get("amount", 0) or 0) for h in holders),
                        reverse=True)
                    if amounts:
                        record["lp_top1_share"] = amounts[0] / amount
                        record["lp_holders_sampled"] = len(amounts)
                except Exception as exc:
                    logger.debug("lp holder mine failed for %s: %s", lp_mint, exc)
            records.append(record)
        return records

    return fetch


# --- who deployed it, and what they did before ---------------------------

def deployer_history_miner(rpc: Any, deployers: Callable[[], Sequence[str]],
                           *, per_pass: int = 8, depth: int = 100,
                           ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """A deployer's recent on-chain activity, from public signatures.

    The strongest reusable prior on this chain is that deployers repeat. The
    same wallet launches, rugs, and launches again; the same funder stands
    behind a dozen of them. `getSignaturesForAddress` is public and gives the
    shape of that history: how long the account has existed, how busy it is,
    and how many of its recent transactions failed.

    This is behavioural inference from public chain data only. It reads what
    the account did in the open ledger and nothing else.
    """
    async def fetch() -> List[Dict[str, Any]]:
        now = time.time()
        records: List[Dict[str, Any]] = []
        for address in list(deployers())[:per_pass]:
            try:
                sigs = await rpc.request(
                    "getSignaturesForAddress",
                    [address, {"limit": depth, "commitment": "confirmed"}])
            except Exception as exc:
                logger.debug("deployer mine failed for %s: %s", address, exc)
                continue
            rows = [row for row in (sigs or []) if isinstance(row, dict)]
            if not rows:
                continue
            times = [float(row.get("blockTime") or 0) for row in rows]
            times = [t for t in times if t > 0]
            failed = len([row for row in rows if row.get("err") is not None])
            record: Dict[str, Any] = {
                "address": address,
                "signatures_sampled": len(rows),
                # A window that fills the requested depth means the account is
                # busier than the sample can see; the caller should treat the
                # rate as a floor.
                "sample_saturated": len(rows) >= depth,
                "failed_share": failed / len(rows),
                "data_status": "OK",
            }
            if times:
                oldest, newest = min(times), max(times)
                record["oldest_seen_age_s"] = max(0.0, now - oldest)
                record["newest_seen_age_s"] = max(0.0, now - newest)
                span = max(1.0, newest - oldest)
                record["tx_per_hour"] = len(rows) / (span / 3600.0)
            records.append(record)
        return records

    return fetch


def account_balance_miner(rpc: Any, accounts: Callable[[], Sequence[str]],
                          *, per_pass: int = 25,
                          ) -> Callable[[], Awaitable[List[Dict[str, Any]]]]:
    """SOL balances for watched wallets, batched.

    `getMultipleAccounts` answers for a hundred addresses in one round trip,
    which is what makes watching a real elite set affordable. A tracked
    wallet's balance falling sharply is it deploying capital somewhere; rising
    sharply is it having exited something.
    """
    async def fetch() -> List[Dict[str, Any]]:
        watched = [a for a in accounts() if a][:per_pass]
        if not watched:
            return []
        result = await rpc.request(
            "getMultipleAccounts",
            [watched, {"commitment": "confirmed", "encoding": "base64"}])
        values = (result or {}).get("value") or []
        records: List[Dict[str, Any]] = []
        for address, value in zip(watched, values):
            if value is None:
                # The account does not exist on chain. That is a real reading
                # -- a funder that has been emptied and closed -- and is not
                # the same as a balance of zero on a live account.
                records.append({"address": address, "exists": False,
                                "data_status": "OK"})
                continue
            records.append({
                "address": address,
                "exists": True,
                "lamports": float(value.get("lamports", 0) or 0),
                "sol": float(value.get("lamports", 0) or 0) / LAMPORTS_PER_SOL,
                "owner": value.get("owner"),
                "executable": bool(value.get("executable")),
                "data_status": "OK",
            })
        return records

    return fetch


def register_chain_miners(pool: DataMinerPool, *, rpc: Any,
                          hot_accounts: Callable[[], Sequence[str]],
                          lp_mints: Callable[[], Sequence[str]],
                          deployers: Callable[[], Sequence[str]],
                          watched_wallets: Callable[[], Sequence[str]],
                          ) -> Dict[str, bool]:
    """Declare the chain-side set.

    Cadence follows what the measurement is worth at the moment it is read.
    The fee distribution decides the next bid and is mined at the fastest
    class; network health changes over a minute; a deployer's history over
    fifteen. Mining any of them faster reads the same number again and spends
    the RPC budget the hot path needs.
    """
    registrations = (
        (MinerSpec(
            miner_id="chain:priority_fees", enriches=Enriches.EXECUTION_CONDITIONS,
            cadence_seconds=CADENCE_FAST,
            endpoint="rpc:getRecentPrioritizationFees",
            detail="observed clearing fee for the accounts we contend on"),
         priority_fee_miner(rpc, hot_accounts)),
        (MinerSpec(
            miner_id="chain:network_health", enriches=Enriches.EXECUTION_CONDITIONS,
            cadence_seconds=CADENCE_MINUTE,
            endpoint="rpc:getRecentPerformanceSamples",
            detail="real slot rate and TPS; the landing model's assumption"),
         network_health_miner(rpc)),
        (MinerSpec(
            miner_id="chain:lp_supply", enriches=Enriches.SUPPLY_CONTROL,
            cadence_seconds=CADENCE_MINUTE, endpoint="rpc:getTokenSupply",
            detail="LP burned, locked or still pullable, for migrated tokens"),
         lp_supply_miner(rpc, lp_mints)),
        (MinerSpec(
            miner_id="chain:deployer_history", enriches=Enriches.WALLET_HISTORY,
            cadence_seconds=CADENCE_QUARTER,
            endpoint="rpc:getSignaturesForAddress",
            detail="public activity history of the wallets that deploy"),
         deployer_history_miner(rpc, deployers)),
        (MinerSpec(
            miner_id="chain:wallet_balances", enriches=Enriches.WALLET_HISTORY,
            cadence_seconds=CADENCE_MINUTE, endpoint="rpc:getMultipleAccounts",
            detail="SOL balances of tracked wallets, batched"),
         account_balance_miner(rpc, watched_wallets)),
    )
    return {spec.miner_id: pool.register(spec, fetch)
            for spec, fetch in registrations}
