"""Reconstructing past launches, and never confusing them with observed ones.

The moat is millions of correctly timestamped launch states, and waiting for
them to accumulate forward costs months. Backfill buys most of that history
now: past launches can be rebuilt from chain data into the same point-in-time
episode shape the live collector writes.

What backfill cannot buy is equivalence. A reconstructed episode is
systematically different from an observed one in ways that flatter it, and
every one of those differences makes a model trained on it look better than it
will be:

  Survivorship. Reconstruction starts from launches that left a trace worth
  finding. The ones that died in eight seconds leaving nothing are exactly the
  ones a selective bot must learn to skip, and they are the hardest to
  enumerate.

  Latency. A live snapshot at T+250ms contains what had actually reached us by
  then. A reconstructed one contains what the chain recorded, which is more.
  Training on the second and serving on the first is training-serving skew
  with a clock in it.

  Depth. Live episodes carry executable size measured by quoting. A
  reconstruction infers it from reserves, which is an upper bound.

So every reconstructed episode is stamped, the stamp survives into every
downstream dataset, and the trainers can weight or exclude on it. The one
thing this module must never do is produce a row that cannot be told apart
from a live one -- at that point the moat is contaminated and there is no way
back short of discarding everything.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

BACKFILL_SCHEMA_VERSION = "v1"

# The key stamped onto every reconstructed episode. Deliberately not something
# a downstream consumer could read as ordinary metadata and drop.
PROVENANCE_KEY = "data_provenance"
LIVE_PROVENANCE = "observed_live"
BACKFILL_PROVENANCE = "reconstructed"


class Limitation(Enum):
    """Ways a reconstruction differs from an observation, stated per episode."""

    SURVIVORSHIP = "survivorship_selected"
    NO_OBSERVATION_LATENCY = "no_observation_latency"
    INFERRED_DEPTH = "inferred_depth_upper_bound"
    NO_SOCIAL_TIMESTAMPS = "no_social_timestamps"
    NO_ROUTE_FEASIBILITY = "no_route_feasibility"
    PARTIAL_BUYER_SET = "partial_buyer_set"


@dataclass
class RawLaunch:
    """The chain-side material a reconstruction is built from."""

    token: str
    created_at: float
    creator: str = ""
    bonding_curve: str = ""
    trades: List[Dict[str, Any]] = field(default_factory=list)
    funding_transfers: List[Dict[str, Any]] = field(default_factory=list)
    migrated_at: Optional[float] = None
    last_seen_at: Optional[float] = None


@dataclass
class ReconstructionResult:
    status: str
    episode: Optional[Dict[str, Any]] = None
    limitations: List[Limitation] = field(default_factory=list)
    detail: str = ""


def _first_buyers(trades: Sequence[Dict[str, Any]], depth: int) -> List[Dict[str, Any]]:
    """First ``depth`` distinct buyers in order.

    Order, not set: "bad wallet, then good, then good independent" and "ten
    linked wallets, then retail" share every aggregate and mean opposite
    things.
    """
    seen: set = set()
    ordered: List[Dict[str, Any]] = []
    for trade in sorted(trades, key=lambda item: float(item.get("timestamp", 0) or 0)):
        if trade.get("side") != "buy":
            continue
        wallet = trade.get("wallet")
        if not wallet or wallet in seen:
            continue
        seen.add(wallet)
        ordered.append({"wallet": wallet,
                        "timestamp": float(trade.get("timestamp", 0) or 0),
                        "notional_sol": trade.get("notional_sol")})
        if len(ordered) >= depth:
            break
    return ordered


def _price_path(trades: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Observations in the shape `lifecycle_from_episode` reads.

    Depth is carried only where the raw trade actually recorded reserves. An
    inferred number would be an upper bound presented as a measurement, and
    the replay harness treats a measurement as fillable.
    """
    path: List[Dict[str, Any]] = []
    entry_price: Optional[float] = None
    for trade in sorted(trades, key=lambda item: float(item.get("timestamp", 0) or 0)):
        price = trade.get("price_sol_per_token") or trade.get("curve_price_raw")
        if price is None:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        if entry_price is None:
            entry_price = price
        path.append({
            "type": "trade", "timestamp": float(trade.get("timestamp", 0) or 0),
            "side": trade.get("side"), "wallet": trade.get("wallet"),
            "notional_sol": trade.get("notional_sol"),
            "price_multiple": price / entry_price,
            "measurement": "reconstructed_from_chain",
            # Deliberately absent unless the raw record carried reserves:
            # `executable_sol` missing means DATA_BLOCKED downstream, which is
            # the correct reading of "we did not measure depth here".
            **({"executable_sol": trade["executable_sol"]}
               if trade.get("executable_sol") is not None else {}),
        })
    return path


def reconstruct(raw: RawLaunch, min_trades: int = 5,
                buyer_depth: int = 25) -> ReconstructionResult:
    """Rebuild one launch into an episode, stamped with what it is.

    Refuses rather than guesses on thin material: a launch with three recorded
    trades is not a reconstructed lifecycle, it is three trades, and treating
    it as the former puts noise into the moat wearing the shape of evidence.
    """
    if raw.created_at <= 0:
        return ReconstructionResult("DATA_BLOCKED", detail="no creation time")
    if len(raw.trades) < min_trades:
        return ReconstructionResult(
            "DATA_BLOCKED",
            detail=f"{len(raw.trades)} trades, below the {min_trades} needed to "
                   "reconstruct a lifecycle")

    observations = _price_path(raw.trades)
    if not observations:
        return ReconstructionResult(
            "DATA_BLOCKED", detail="no trade carried a usable price")

    limitations = [
        Limitation.SURVIVORSHIP,
        Limitation.NO_OBSERVATION_LATENCY,
        Limitation.NO_SOCIAL_TIMESTAMPS,
        Limitation.NO_ROUTE_FEASIBILITY,
    ]
    if not any("executable_sol" in item for item in observations):
        limitations.append(Limitation.INFERRED_DEPTH)
    buyers = _first_buyers(raw.trades, buyer_depth)
    if len(buyers) < buyer_depth:
        limitations.append(Limitation.PARTIAL_BUYER_SET)

    peak = max(item["price_multiple"] for item in observations)
    final = observations[-1]["price_multiple"]
    episode = {
        "token": raw.token,
        "chain": "solana",
        "created_at": raw.created_at,
        "deployer": raw.creator,
        "pair": raw.bonding_curve,
        "market_observations": observations,
        "first_buyers": buyers,
        "funding_transfers": list(raw.funding_transfers),
        "final_outcome": {
            "migrated": raw.migrated_at is not None,
            "peak_multiple": peak,
            "final_multiple": final,
            # A collapse to near zero is recorded as observed collapse, not as
            # a rug: reconstruction cannot see WHO caused it, and labelling
            # intent from price alone is how a rug model learns to predict
            # drawdowns instead of rugs.
            "collapsed": final <= 0.05 * peak,
            "rugged": None,
        },
        PROVENANCE_KEY: {
            "source": BACKFILL_PROVENANCE,
            "schema_version": BACKFILL_SCHEMA_VERSION,
            "reconstructed_at": time.time(),
            "limitations": [item.value for item in limitations],
        },
    }
    return ReconstructionResult("OK", episode=episode, limitations=limitations,
                                detail=f"{len(observations)} observations, "
                                       f"{len(buyers)} first buyers")


def is_reconstructed(episode: Dict[str, Any]) -> bool:
    """True when this episode was rebuilt rather than observed.

    Anything without an explicit live stamp is treated as reconstructed. The
    safe default is the pessimistic one: mistaking a reconstruction for an
    observation inflates every downstream result, while the reverse only
    wastes data.
    """
    provenance = episode.get(PROVENANCE_KEY)
    if not isinstance(provenance, dict):
        return True
    return provenance.get("source") != LIVE_PROVENANCE


def stamp_live(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Mark an episode as genuinely observed, with no limitations."""
    return {**episode, PROVENANCE_KEY: {"source": LIVE_PROVENANCE,
                                        "schema_version": BACKFILL_SCHEMA_VERSION,
                                        "limitations": []}}


def partition_by_provenance(
    episodes: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(observed, reconstructed). The split every trainer has to make."""
    observed, reconstructed = [], []
    for episode in episodes:
        (reconstructed if is_reconstructed(episode) else observed).append(episode)
    return observed, reconstructed


@dataclass
class BackfillReport:
    attempted: int = 0
    reconstructed: int = 0
    blocked: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    limitation_counts: Dict[str, int] = field(default_factory=dict)
    #: Edges handed to the materialised actor graph, if one was attached.
    actor_edges: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted, "reconstructed": self.reconstructed,
            "blocked": self.blocked,
            "yield": (self.reconstructed / self.attempted) if self.attempted else None,
            "blocked_reasons": dict(sorted(self.reasons.items(),
                                           key=lambda item: -item[1])),
            "limitations": dict(sorted(self.limitation_counts.items(),
                                       key=lambda item: -item[1])),
        }


def run_backfill(
    raws: Sequence[RawLaunch],
    output_dir: Path,
    min_trades: int = 5,
    writer: Optional[Callable[[Path, Dict[str, Any]], None]] = None,
    actor_store: Optional[Any] = None,
) -> BackfillReport:
    """Reconstruct a batch, writing one file per episode.

    Runs on whatever machine has the history, never on the trading node: a
    backfill competes for RAM, CPU, disk and file descriptors exactly when a
    launch arrives, and the trading node's job is to be boring and predictable.

    `actor_store`, when given, is fed the SAME raw launches as edges stamped
    with when each was observable. That is the only moment the graph can be
    built correctly: the reconstruction path already has creation times, buy
    order and funding transfers in hand, and a graph assembled later from the
    episodes has to re-derive all three and gets the ordering wrong.

    Every launch is offered, including the ones reconstruction rejects. A
    launch too thin to be an episode is still a real deployment by a real
    deployer, and dropping it would understate exactly the prior-launch counts
    the store exists to answer.
    """
    report = BackfillReport()
    if actor_store is not None:
        try:
            ingested = actor_store.ingest_raw_launches(raws)
            report.actor_edges = int(ingested.get("edges", 0))
        except Exception as exc:
            logger.warning("actor store ingestion DATA_BLOCKED: %s", exc)
            report.actor_edges = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    write = writer or (lambda path, payload: path.write_text(
        json.dumps(payload, default=str)))

    for raw in raws:
        report.attempted += 1
        result = reconstruct(raw, min_trades=min_trades)
        if result.status != "OK" or result.episode is None:
            report.blocked += 1
            key = result.detail.split(";")[0][:60] or "unknown"
            report.reasons[key] = report.reasons.get(key, 0) + 1
            continue
        for limitation in result.limitations:
            report.limitation_counts[limitation.value] = (
                report.limitation_counts.get(limitation.value, 0) + 1)
        write(output_dir / f"{raw.token}.json", result.episode)
        report.reconstructed += 1
    return report
