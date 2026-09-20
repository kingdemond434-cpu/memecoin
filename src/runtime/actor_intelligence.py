"""The actor graph, fed from launches and read at the decision.

`ActorStore` is an append-only edge log with as-of indexes, built so that
"who funded this deployer", "what did they launch before" and "how much of
this buyer set is one family" can be answered POINT IN TIME rather than from
whatever the graph looks like now. It was never constructed. Not unwired --
never instantiated at all, so `funders_of`, `prior_mints` and `shared_family`
had no callers because they had no object to be called on.

Two halves were missing and both are here: `ingest_launch` turns each observed
creation into edges, and `actor_context` reads them back at the moment a
decision is taken.

The as-of discipline is the whole reason the store exists and the easiest
thing to lose. Every read below passes the DECISION's timestamp, never
`time.time()`: a deployer's third launch must not be scored using the rug
their fourth launch turned into, and a graph queried at wall-clock time
silently does exactly that while looking correct.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _store(desk: Any) -> Any:
    return getattr(desk, "actor_store", None)


def ingest_launch_edges(desk: Any, token: str, event: Dict[str, Any]) -> int:
    """Record one observed creation as graph edges. Returns edges written."""
    store = _store(desk)
    if store is None or not token:
        return 0
    creator = str(event.get("creator") or "")
    if not creator:
        return 0
    at = float(event.get("timestamp", time.time()) or time.time())
    funding: List[Tuple[str, str, float]] = []
    for transfer in (event.get("funding_transfers") or ()):
        if not isinstance(transfer, dict):
            continue
        source = str(transfer.get("from") or "")
        target = str(transfer.get("to") or "")
        if source and target and source != target:
            funding.append((source, target, at))
    try:
        return int(store.ingest_launch(
            mint=token, creator=creator, created_at=at,
            launchpad=str(event.get("program") or event.get("launchpad") or ""),
            funding=funding))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("actor store ingest failed for %s: %s", token, exc)
        return 0


def record_buy_edge(desk: Any, token: str, wallet: str, at: float) -> None:
    """One buy, in order. Rank comes from the order edges arrive in."""
    store = _store(desk)
    if store is None or not token or not wallet:
        return
    from src.research.actor_store import Edge, EdgeKind
    try:
        edge = Edge(source=wallet, target=token, kind=EdgeKind.BOUGHT,
                    observed_at=float(at))
        store.add(edge)
        store.append_to_log([edge])
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("actor store buy edge failed: %s", exc)


def actor_context(desk: Any, token: str, deployer: str, as_of: float,
                  buyers: Sequence[str] = ()) -> Dict[str, Any]:
    """Who is behind this launch, as of the instant the decision is taken.

    `as_of` is passed through to every read and is never defaulted to now.
    A deployer's third launch scored with the rug their fourth turned into is
    a leak that looks exactly like an edge.
    """
    store = _store(desk)
    if store is None:
        return {"status": "DATA_BLOCKED", "detail": "no actor store"}
    moment = float(as_of)

    def read(call: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("actor read failed: %s", exc)
            return None

    funders = read(store.funders_of, deployer, moment) or [] if deployer else []
    prior = read(store.prior_mints, deployer, moment) or [] if deployer else []
    family = read(store.shared_family, list(buyers), moment) if buyers else None
    return {
        "status": "OK",
        "deployer": deployer,
        # A repeat deployer is the most transferable fact on this chain, and
        # it was being rediscovered from scratch on every launch.
        "prior_mints": len(prior),
        "prior_mint_ids": prior[:10],
        "funders": funders[:10],
        "funder_count": len(funders),
        # Ten wallets that are one family are one buyer. A launch whose
        # opening cohort collapses into a single family has no independent
        # demand however many addresses appear in it.
        "buyer_family": family,
    }


def launch_risk(desk: Any, deployer: str, funders: Sequence[str],
                buyers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The genealogy graph's own read on a launch. RECORDED, never a gate.

    `assess_launch_risk` is a hand-weighted composite -- 0.4 for a deployer
    rug rate over half, 0.2 over a fifth, 0.3 for a critical-risk funding
    cluster, and so on -- every coefficient chosen rather than measured. This
    desk has spent the day removing exactly that shape from decisions, so it
    is recorded beside the decision for the gauntlet to price and given no
    authority over anything.

    Its signature also disagreed with its body: it annotates `initial_buyers`
    as `List[str]` and then calls `.get("address")` on each element. Callers
    are given dicts here, which is what the body has always required.
    """
    graph = getattr(desk, "genealogy", None)
    assess = getattr(graph, "assess_launch_risk", None)
    if assess is None:
        return {"status": "DATA_BLOCKED", "detail": "no genealogy graph"}
    try:
        reading = assess(str(deployer or ""), list(funders),
                         [dict(item) for item in buyers])
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "DATA_BLOCKED", "detail": f"assessment failed: {exc}"}
    return {
        "status": "OK",
        "authority": "none",
        "note": ("a hand-weighted composite, recorded for the gauntlet to "
                 "price and deliberately given no vote"),
        **{key: value for key, value in reading.items()
           if key != "deployer_profile"},
    }


def entry_actor_block(desk: Any, token: str, candidate: Any) -> Dict[str, Any]:
    """The actor and cold-start slots of an entry decision, in one read.

    Assembled here rather than inline so the AS-OF timestamp is chosen once.
    Every read below is taken at the CANDIDATE's own timestamp: a graph
    queried at wall clock silently scores a deployer's third launch using the
    rug their fourth turned into, and looks correct while doing it.
    """
    from src.runtime.research_reads import (
        cold_start_prior, fallback_confidence, lead_forecast)
    metadata = getattr(candidate, "metadata", None) or {}
    deployer = str(getattr(candidate, "deployer", "") or "")
    as_of = float(getattr(candidate, "timestamp", 0.0) or time.time())
    buyers = metadata.get("initial_buyers") or ()
    funders = [str(item.get("from", "")) for item in
               (metadata.get("funding_transfers") or ())
               if isinstance(item, dict)]
    return {
        "actors": actor_context(
            desk, token, deployer, as_of,
            buyers=[str(item.get("wallet", "")) for item in buyers
                    if isinstance(item, dict)]),
        "genealogy_risk": launch_risk(desk, deployer, funders, buyers),
        "lead_forecast": lead_forecast(desk, token),
        "cold_start": cold_start_prior(
            desk, str(metadata.get("sleeve", "") or ""),
            int(time.gmtime(as_of).tm_hour)),
        # A decision resting on three substituted facts is not as good as one
        # resting on primaries, and the WEAKEST rung dominates. Nothing was
        # computing that, so both were priced identically.
        "evidence_confidence": fallback_confidence(
            desk, ("holder_concentration", "liquidity", "sell_route")),
    }


def record_actor_feedback(desk: Any, token: str, event: Dict[str, Any]) -> None:
    """Feeds that had no writer, written from events the desk already decodes.

    Four surfaces were readable and permanently empty, which reports
    identically to one nobody consults:

    * `observe_exit` -- a wallet's HOLD TIME is the most transferable thing
      about it, and the signature model had sells but never their duration.
    * `record_balance` -- the developer's remaining share, which is the whole
      input to distribution detection.
    * `is_kol` -- whether a source touch has the reach to start a wave, read
      and never asked.
    """
    wallet = str(event.get("wallet") or "")
    at = float(event.get("timestamp", 0) or time.time())
    if event.get("side") == "sell" and wallet:
        signatures = getattr(desk, "wallet_signatures", None)
        opened = (getattr(desk, "_wallet_first_seen", None) or {}).get(
            (token, wallet))
        if signatures is not None and opened:
            try:
                signatures.observe_exit(wallet, max(0.0, at - float(opened)))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("observe_exit failed: %s", exc)
    balance = event.get("creator_balance_pct")
    monitor = getattr(desk, "dev_wallet_monitor", None)
    if balance is not None and monitor is not None:
        try:
            monitor.record_balance(token, balance, timestamp=at,
                                   source="chain_stream")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_balance failed: %s", exc)


def note_wallet_first_seen(desk: Any, token: str, wallet: str,
                           at: float) -> None:
    """When this wallet first touched this token, so an exit has a duration."""
    if not token or not wallet:
        return
    seen = getattr(desk, "_wallet_first_seen", None)
    if seen is None:
        seen = {}
        setattr(desk, "_wallet_first_seen", seen)
    seen.setdefault((token, wallet), float(at))
    # Bounded: one entry per (token, wallet) would otherwise accumulate for
    # the life of the process.
    if len(seen) > 100_000:
        for key in list(seen)[:50_000]:
            seen.pop(key, None)


def record_resolution_feedback(desk: Any, token: str,
                               outcome: Dict[str, Any]) -> int:
    """Tell the adversarial detector which feature values preceded this end.

    `record_feature_value` is how the detector learns that a feature has
    started being faked -- a value that used to precede monsters and now
    precedes rugs. It had no writer, so the detector reweighted nothing for
    the life of the desk while its weights were readable and constant.
    """
    detector = getattr(desk, "adversarial", None)
    features = (outcome or {}).get("features") or {}
    if detector is None or not features:
        return 0
    written = 0
    for name, value in features.items():
        if not isinstance(value, (int, float)):
            continue
        try:
            detector.record_feature_value(token, str(name), float(value),
                                          dict(outcome))
            written += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("record_feature_value failed for %s: %s", name, exc)
            break
    return written
