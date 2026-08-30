"""Periodic upkeep that keeps the desk's own measurements honest.

Extracted from ``main`` rather than added to it, for the reason its own
line-budget test gives: a service belongs in a service. Everything here
runs on the maintenance cadence, off the decision path, and exists because
a measurement the desk stops taking is one it silently starts guessing at.

Four jobs, each closing a hole that made a real number wrong:

* **Death verdicts.** ``_resolve_census_death`` had no caller, so 2,714
  launches awaiting a rug verdict produced one. The sweep gives every
  resolved-but-unlabelled launch a chance, and eviction gives it a last
  one -- the sweep only reaches RESIDENT tokens, and eviction takes exactly
  the stalest, which are the likeliest to be dead.

* **Memory bounds.** Five dicts keyed by mint with no eviction OOM-killed
  this service twelve times in one hour. Pruning is bounded by staleness
  and never touches a token we hold: a blinded exit is worse than the RAM.

* **Wallet DNA seeding.** 56,636 wallets had been observed and none of that
  history informed which wallets the desk watched; it rediscovered
  everything from scratch on each restart.

* **Follow persistence.** An open follow is an unresolved measurement with
  a 300s horizon. Held only in memory, every restart voided them and no
  wallet ever reached the twelve-outcome ranking bar.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List

from src.research import rug_mechanism

logger = logging.getLogger(__name__)

#: How long a token must sit with no new observation before a death verdict
#: is attempted. Below this, "no new trade yet" and "dead" are
#: indistinguishable, and classifying early permanently brands a token that
#: is only mid-launch.
DEATH_CLASSIFICATION_QUIET_S = 300.0


class DeskMaintenance:
    """Upkeep the desk mixes in. Requires the desk's own attributes."""

    def _prune_hazard_tracking(self) -> int:
        """Bound the hazard model's per-token memory. Returns tokens evicted.

        The model registers every launch the stream reports and never let
        one go, which is what OOM-killed this service repeatedly under a
        continuous pump.fun feed -- the growth is in the NUMBER of tokens,
        so no per-token cap could have caught it.

        Only open positions are protected by name, and that is deliberate.
        A token being actively decided on is written to on every trade, so
        recency already keeps it -- eviction takes the STALEST first. The
        broader sets are the wrong tool here: hot_state.active_tokens is
        capped at 4,000, which is above this cap and would defeat it
        outright, and the curve/pool state dicts are themselves unbounded,
        so protecting by them would mean one leak holding another open.

        A token we hold must never be evicted at any staleness: the exit
        policy reads its hazard, and a silently blinded exit is a far worse
        failure than the memory it costs to keep.
        """
        # Classify on the way out. The sweep only reaches resident tokens,
        # so without this a token evicted while still unlabelled has its
        # question closed silently -- and eviction takes exactly the stale
        # tokens that are most likely to be dead.
        return self.rug_hazard.prune(
            set(self.elogw_engine.open_positions),
            on_evict=self._classify_before_eviction)

    def _classify_before_eviction(self, token: str) -> None:
        """Give a token one last chance at a death verdict.

        The quiet-period guard is deliberately skipped: eviction already
        means this token is among the stalest tracked, which is the same
        evidence the guard exists to establish.
        """
        observations = list(self.rug_hazard.observations.get(token, ()) or ())
        priced = sum(1 for row in observations
                     if row.get("price_multiple") is not None)
        if priced < 2:
            return
        pool = self._latest_pool_state.get(token)
        verdict = rug_mechanism.classify(observations, migrated=(pool is not None))
        if verdict.mechanism is rug_mechanism.RugMechanism.SURVIVED:
            return
        self.launch_census.resolve(
            token, rugged=True, rug_mechanism=verdict.mechanism.value)
        self._record_ops_event("rug_mechanisms", {
            "token": token, "at": "eviction", **verdict.to_dict()})

    def _sweep_rug_classification(self) -> None:
        """Give every resolved-but-unlabeled launch a chance at a verdict.

        _resolve_census_death above was written to do exactly this and had
        no caller anywhere in the desk -- the reason rugs sat at 0 across
        thousands of resolved launches was never that memecoins on this feed
        don't rug, it was that nothing ever asked. Runs on the same 60s
        cadence as the rest of the intelligence loop, off the decision path.
        """
        for token in self.launch_census.mints_pending_death_classification():
            self._resolve_census_death(token)

    def _wallet_dna_seeds(self) -> List[str]:
        """Wallets worth watching, from distilled history. Never invented.

        Ranked on the SHRUNK enrichment, so a wallet cannot earn a seat by
        touching one token that happened to moon -- the raw rate put
        one-resolved-token wallets at the top when this was first built,
        which is noise wearing the costume of a ranking.

        A missing or unreadable artifact yields nothing rather than an
        error: the desk must start without it, and a seed list is an
        accelerator, not a dependency.
        """
        path = Path(self.global_config.get("ops_state_dir", "data/state")) / "wallet_dna.json"
        minimum = int(self.global_config.get("wallet_dna_min_resolved", 10))
        limit = int(self.global_config.get("wallet_dna_seed_limit", 500))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        seeds = [
            str(row.get("wallet"))
            for row in (payload.get("records") or [])
            if row.get("wallet")
            and int(row.get("resolved_tokens") or 0) >= minimum
            and (row.get("monster_enrichment") or 0.0) > 1.0
        ]
        if seeds:
            logger.info("seeding wallet watch list with %d wallets distilled "
                        "from %s observed launches", len(seeds[:limit]),
                        payload.get("universe_resolved"))
        return seeds[:limit]

    def _follow_state_path(self) -> Path:
        return (Path(self.global_config.get("ops_state_dir", "data/state"))
                / "follow_candidates.json")

    def save_follow_candidates(self) -> bool:
        """Checkpoint open follows so a restart does not void them.

        A follow is an unresolved measurement with a 300s horizon, and the
        wallet model needs 12 resolved outcomes before it will rank a wallet
        at all. Holding them only in memory meant every restart threw away
        every follow still inside its horizon: measured 2026-08-29, 56 open
        follows against 9 resolved outcomes across 7 wallets, so no wallet
        could ever reach the threshold on a desk that restarts. The records
        are plain JSON-safe dicts, so this is a straight dump.
        """
        try:
            path = self._follow_state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema": "v1", "saved_at": time.time(),
                       "candidates": self._follow_candidates}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("follow checkpoint failed: %s: %s",
                           type(exc).__name__, exc)
            return False

    def load_follow_candidates(self) -> int:
        """Restore open follows. Returns how many were revived.

        Follows already past their horizon are dropped rather than resolved
        against a price hours later: the horizon defines what the
        measurement MEANS, and closing one late would record a different
        experiment under the same name.
        """
        path = self._follow_state_path()
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("follow checkpoint unreadable, discarded: %s", exc)
            return 0
        horizon = float(self.global_config.get("follow_horizon_seconds", 300.0))
        cutoff = time.time() - horizon
        revived = 0
        for token, pending in (payload.get("candidates") or {}).items():
            kept = [item for item in pending
                    if isinstance(item, dict)
                    and float(item.get("opened_at", 0) or 0) > cutoff]
            if kept:
                self._follow_candidates[token] = kept
                revived += len(kept)
        if revived:
            logger.info("revived %d open follows from checkpoint", revived)
        return revived
    def _memecoin_state_features(self, token: str,
                                 as_of: Optional[float] = None) -> Dict[str, Any]:
        """The same PIT memecoin state used by snapshots and live inference."""
        holder = getattr(self, "holder_trajectory", None)
        developer = getattr(self, "dev_wallet_monitor", None)
        rotation = getattr(self, "rotation_tracker", None)
        return {
            "holder_trajectory": (holder.state(token, as_of) if holder else {
                "status": "DATA_BLOCKED", "detail": "holder monitor unavailable"}),
            "dev_wallet": (developer.state(token, as_of) if developer else {
                "status": "DATA_BLOCKED", "detail": "developer monitor unavailable"}),
            "capital_rotation": (rotation.report(as_of) if rotation else {
                "status": "DATA_BLOCKED", "detail": "rotation tracker unavailable"}),
        }

    async def _refresh_open_safety(self, token: str, position: Dict[str, Any]) -> None:
        """Re-read mutable mint, route and holder facts off the event hot path."""
        now = time.time()
        interval = float(self.global_config.get("mutable_safety_refresh_seconds", 15.0))
        if now - self._safety_refreshed_at.get(token, 0.0) < interval:
            return
        self._safety_refreshed_at[token] = now
        candidate = position.get("candidate")
        if candidate is None:
            return
        previous = position.get("risk_object")
        current = await self.rug_detector.analyze(
            token, getattr(candidate, "pair", None), getattr(candidate, "base_token", None),
            getattr(candidate, "deployer", None))
        self._enrich_holder_actor_concentration(
            token, current, str(getattr(candidate, "deployer", "") or ""))
        position["risk_object"] = current
        position["risk_report"] = _jsonable(current)
        self.dataset_builder.record_risk_report(token, _jsonable(current))
        self.holder_trajectory.record_mapping(token, {
            "top_10_pct": getattr(current, "top_10_pct", None),
            "top_20_pct": getattr(current, "top_20_pct", None),
            "dev_pct": getattr(current, "deployer_balance_pct", None),
            "insider_pct": getattr(current, "insider_pct", None),
            "bundler_pct": getattr(current, "bundler_pct", None),
            "fresh_wallet_pct": getattr(current, "fresh_wallet_pct", None),
            "whale_pct": getattr(current, "whale_pct", None),
            "cluster_pct": getattr(current, "connected_cluster_pct", None),
            "holder_count": getattr(current, "holder_count", None),
        }, timestamp=getattr(current, "timestamp", now), source="native_spl_refresh")
        trajectory = self.holder_trajectory.state(token)
        position["holder_trajectory"] = trajectory
        top10_change = trajectory.get("changes", {}).get("top_10_pct")
        if top10_change is not None:
            self.rug_hazard.record_observation(token, {
                "type": "concentration", "top10_change_pct": float(top10_change) / 100.0,
                "timestamp": now, "source": "native_spl_refresh"})

        if previous is not None:
            gained_authority = (
                (not bool(getattr(previous, "can_mint", False))
                 and bool(getattr(current, "can_mint", False)))
                or (not bool(getattr(previous, "can_freeze", False))
                    and bool(getattr(current, "can_freeze", False))))
            if gained_authority:
                self.dev_wallet_monitor.record(token, DevEvent(
                    now, "authority_mutation",
                    wallet=str(getattr(candidate, "deployer", "") or ""),
                    severity="critical", evidence=_jsonable(current)))
                self.rug_hazard.record_observation(token, {
                    "type": "dev_wallet_activation", "strength": 1.0,
                    "confidence": 1.0, "timestamp": now,
                    "reason": "authority_mutation"})
            if (getattr(previous, "sell_route_feasible", None) is not False
                    and getattr(current, "sell_route_feasible", None) is False):
                self.dev_wallet_monitor.record(token, DevEvent(
                    now, "sell_route_failed", severity="critical",
                    evidence=(getattr(current, "checks", {}) or {}).get("sell_route", {})))
                self.rug_hazard.record_observation(token, {
                    "type": "route", "feasible": False, "timestamp": now,
                    "source": "native_safety_refresh"})

    def _resolve_census_death(self, token: str) -> None:
        """Classify how a token died and record it against the denominator.

        Runs over the observations the hazard tracker already collects, so no
        new stream is needed. A death with no mechanism evidence is recorded
        as unclassified rather than being folded into slow bleed, because a
        residual that absorbs every unexplained death is how a rug model
        learns nothing while reporting full coverage.
        """
        # Cheapest test first, and without materialising anything: this runs
        # for every resolved-but-unlabelled launch on every sweep and the
        # overwhelming majority are still trading. O(1) last-touched beats
        # scanning up to 750 observations across thousands of tokens.
        last_seen = self.rug_hazard.last_touched(token)
        if last_seen is None:
            return
        if time.time() - last_seen < DEATH_CLASSIFICATION_QUIET_S:
            # Still receiving. "No new trade yet" and "dead" are
            # indistinguishable here, and classifying early would brand a
            # token that is merely mid-launch.
            return
        observations = list(self.rug_hazard.observations.get(token, ()) or ())
        if not observations:
            return
        priced = sum(1 for row in observations
                     if row.get("price_multiple") is not None)
        if priced < 2:
            # classify() reports UNCLASSIFIED both for "died, cause unknown"
            # and "too little price data to know it died at all". Conflating
            # those would mislabel lightly-observed tokens as rugs.
            return
        pool = self._latest_pool_state.get(token)
        verdict = rug_mechanism.classify(
            observations, migrated=(pool is not None))
        if verdict.mechanism is rug_mechanism.RugMechanism.SURVIVED:
            return
        self.launch_census.resolve(
            token, rugged=True, rug_mechanism=verdict.mechanism.value)
        self._resolve_corpus(token, rugged=True,
                             rug_mechanism=verdict.mechanism.value)
        self._record_ops_event("rug_mechanisms", {
            "token": token, **verdict.to_dict()})

    def _enrich_holder_actor_concentration(self, token: str, risk: Any,
                                           deployer: str) -> Dict[str, Any]:
        """Attach evidence-backed actor lower bounds to native holder data.

        The RPC layer resolves token accounts to wallet owners.  This method
        joins those public owners to the already-observed genealogy and
        coordination graphs.  A graph that has not linked a wallet does not
        prove independence, so unobserved labels remain ``None``; positive
        matches are safe lower bounds and can still trigger a hard veto.
        """
        checks = getattr(risk, "checks", None)
        holders = checks.get("holders") if isinstance(checks, dict) else None
        accounts = holders.get("accounts", []) if isinstance(holders, dict) else []
        by_owner: Dict[str, float] = {}
        for account in accounts:
            owner = str(account.get("owner", "") or "")
            if not owner:
                continue
            try:
                share = float(account.get("supply_pct"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(share) and share >= 0:
                by_owner[owner] = by_owner.get(owner, 0.0) + share
        if not by_owner:
            result = {"status": "DATA_BLOCKED",
                      "reason": "largest token-account owners unresolved"}
            if isinstance(holders, dict):
                holders["entity_enrichment"] = result
            return result

        evidence: Dict[str, Any] = {
            "status": "PARTIAL", "semantics": "observed_lower_bounds",
            "resolved_owners": len(by_owner),
        }
        whale_threshold = float(self.global_config.get(
            "holder_whale_supply_pct", 1.0))
        whale_owners = {owner for owner, pct in by_owner.items()
                        if pct >= whale_threshold}
        risk.whale_pct = sum(by_owner[owner] for owner in whale_owners)
        evidence.update({
            "whale_supply_threshold_pct": whale_threshold,
            "whale_pct_lower_bound": risk.whale_pct,
            "whale_owners": len(whale_owners),
        })

        coordination = getattr(self, "public_coordination", None)
        coordinated_wallets = set()
        kinds = set()
        if coordination is not None:
            for item in coordination.token_evidence.get(token, []):
                if item.kind in {
                    "same_slot_buy_cluster", "shared_funder",
                    "near_identical_buy_sizes",
                }:
                    coordinated_wallets.update(item.wallets)
                    kinds.add(item.kind)
        if kinds:
            risk.bundler_pct = sum(
                pct for owner, pct in by_owner.items()
                if owner in coordinated_wallets)
            evidence.update({
                "bundler_pct_lower_bound": risk.bundler_pct,
                "bundler_evidence": sorted(kinds),
                "coordinated_holders": len(set(by_owner) & coordinated_wallets),
            })
        else:
            evidence["bundler_status"] = "DATA_BLOCKED: no qualifying public coordination evidence"

        genealogy = getattr(self, "genealogy", None)
        developer_cluster = (genealogy.find_cluster(deployer)
                             if genealogy is not None and deployer else None)
        if developer_cluster is not None:
            cluster_wallets = set(developer_cluster.wallets)
            risk.connected_cluster_pct = sum(
                pct for owner, pct in by_owner.items()
                if owner in cluster_wallets)
            evidence.update({
                "connected_cluster_pct_lower_bound": risk.connected_cluster_pct,
                "connected_cluster_id": developer_cluster.cluster_id,
                "connected_holders": len(set(by_owner) & cluster_wallets),
            })
        else:
            evidence["connected_cluster_status"] = "DATA_BLOCKED: developer cluster unproven"

        insider_wallets = set()
        if genealogy is not None:
            for owner in by_owner:
                profile = genealogy.get_wallet_profile(owner)
                if profile is not None and getattr(profile, "is_insider", False):
                    insider_wallets.add(owner)
        if insider_wallets:
            risk.insider_pct = sum(by_owner[owner] for owner in insider_wallets)
            evidence.update({
                "insider_pct_lower_bound": risk.insider_pct,
                "insider_holders": len(insider_wallets),
            })
        else:
            evidence["insider_status"] = "DATA_BLOCKED: no evidence-labeled insider holder"

        # Chain age is not derivable from a wallet's first local observation.
        # Keep fresh-wallet concentration unavailable until an actual
        # transaction-history timestamp is measured.
        evidence["fresh_wallet_status"] = "DATA_BLOCKED: chain-age enrichment unavailable"
        if isinstance(holders, dict):
            holders["entity_enrichment"] = evidence
        return evidence
