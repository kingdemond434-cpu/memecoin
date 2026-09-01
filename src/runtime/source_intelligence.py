"""Public sources in, actor and authenticity evidence out.

Split out of `main.py`. Source consumption, entity authenticity, source DNA,
ignition and disagreement all read the same public-post stream and none of
them is on the latency path, so they share a file and stop sharing a merge
surface with the trading code.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from src.collectors.transports import (
    build_transports,
    start_transports,
    stop_transports,
    transport_report,
)
from src.strategies.actor_graph import (
    Entry,
    IndependenceReport,
    SwarmPredictor,
    WalletIndependence,
    build_fingerprint,
)
from src.strategies.authenticity import EntityRegistry, ProofLevel, SourceSignal, load_entities
from src.strategies.source_genealogy import SourcePost, build_source_dna
from src.strategies.funder_ancestry import compress_independence
from src.strategies.temporal_funding import (
    find_clusters, independence_discounts, measure_source_rate)
from src.strategies.disagreement import DisagreementModel, views_from_intelligence
from src.strategies.ignition import IgnitionModel, touches_from_events
import logging

logger = logging.getLogger(__name__)


class SourceIntelligence:
    """Consuming public sources, and turning them into actor evidence.

    The third mixin, on the same terms as the others: every `self.` resolves
    exactly as it did, so a regression can only come from the move. Grouped
    because these methods share one job -- take what a public source said,
    decide who said it and whether that has ever been worth anything, and
    hand the answer to the models that price a launch.

    None of it is on the decision path. Sources arrive on their own loop and
    write into state a decision reads later, which is the property that lets
    this live in its own file at all: a change to how a Telegram post is
    indexed cannot reach the code that sizes a position."""

    def _authenticity(self, token: str, candidate: Any) -> Dict[str, Any]:
        """Is this token the entity it claims to be, and how do we know.

        Both proofs available without private access are used: the chain-side
        one (a wallet already known to be the entity created or funded it) and
        the publication-side one (a message from a canonical account of that
        entity naming this mint). They are combined rather than raced, because
        a single strong proof should win outright while several weak
        independent ones should only count if they are genuinely independent.
        """
        if not self._watched_entities:
            return {"status": "DATA_BLOCKED", "reason": "no watched entities declared",
                    "registry_size": 0}
        # Funders as well as the creator: an entity that funded the deployer
        # is chain-side proof too, and a launch is routinely made from a wallet
        # one hop from the one anybody has heard of.
        funders = list(dict.fromkeys(
            str(item.get("funder", "")) for item
            in list(self.public_coordination.funding.get(token, ()))[:64]
            if item.get("funder")))
        verdicts = [self.authenticity.resolve_creator(
            token, candidate.deployer or "", funders)]
        for event in list(self._source_events.get(token, ()))[-20:]:
            verdicts.append(self.authenticity.resolve_signal(SourceSignal(
                platform=event.source_id, account_id=str(event.author_id or ""),
                text=event.text or "", timestamp=event.observed_at,
                url=(list(event.urls) or [""])[0])))
        combined = self.authenticity.combine(verdicts)
        return {
            "status": "OK", "level": combined.level.value,
            "rank": combined.level.rank, "entity_id": combined.entity_id,
            "tradeable": bool(combined.tradeable),
            "sources": list(combined.supporting_sources)[:8],
            "detail": combined.detail,
            "registry_size": len(self._watched_entities),
        }

    def _source_dna(self, token: str) -> Dict[str, Any]:
        """Whether the sources that named this token have historically paid.

        A source can be a superb predictor of flow and a terrible thing to
        trade directly -- that is what a distributor looks like from the
        outside: reliably followed by a move, reliably followed by a dump.
        The two properties are reported separately rather than collapsed into
        one score, because collapsing them is how the desk ends up buying into
        the exit liquidity it correctly predicted.
        """
        events = self._source_events.get(token) or []
        if not events:
            return {"status": "DATA_BLOCKED", "reason": "no source named this token"}
        profiles = []
        for source_id in dict.fromkeys(event.source_id for event in events):
            outcomes = self._source_outcomes.get(source_id) or []
            dna = build_source_dna(source_id, outcomes)
            profiles.append({
                "source_id": source_id, "status": dna.status,
                "posts": dna.posts,
                "median_observation_lag": dna.median_observation_lag,
                "tradeable_directly": bool(dna.tradeable_directly),
                "useful_as_flow_signal": bool(dna.useful_as_flow_signal),
                "is_distributor": bool(dna.is_distributor),
                "upstream_of": [lead.follower for lead in
                                self.source_genealogy.upstream_of(source_id)][:5],
            })
        measured = [profile for profile in profiles if profile["status"] == "MEASURED"]
        return {
            "status": "OK" if measured else "MEASURING",
            "reason": "" if measured else "no source has enough resolved posts for a verdict",
            "sources": profiles[:8],
            "any_distributor": any(profile["is_distributor"] for profile in measured),
            "any_tradeable": any(profile["tradeable_directly"] for profile in measured),
        }

    def _publish_attribution(self) -> None:
        """Write the edge-decay and ledger state the weekly audit pack reads.

        Computed here rather than in the pack builder so the numbers come from
        the same process that made the trades, and so a pack built on a node
        whose research package is unavailable still gets them. Failures are
        swallowed: reporting must never be able to halt trading.
        """
        if time.time() - self._attribution_published_at < self._attribution_interval:
            return
        self._attribution_published_at = time.time()
        try:
            root = Path(self.global_config.get("ops_state_dir", "data/state"))
            root.mkdir(parents=True, exist_ok=True)
            for mechanism, growth in self._mechanism_growth.items():
                for value in growth:
                    self.edge_decay.record(mechanism, value)
                growth.clear()
            (root / "edge_decay.json").write_text(
                json.dumps(self.edge_decay.report(), default=str))
        except (OSError, ValueError) as exc:
            logger.debug("attribution publish failed: %s", exc)

    def _record_actor_entry(self, token: str, event: Dict[str, Any],
                            observation: Dict[str, Any]) -> None:
        """Feed one buy into the independence graph and bound the hot state.

        Only buys, and only the FIRST buy per wallet per token: the graph asks
        who chose to enter and in what order, and a wallet adding to its own
        position is not further evidence about anyone else's decision.

        Wallet skill is attached where the intelligence engine already has a
        score. Where it does not, the field is left None rather than zero --
        the distinction between "scored at zero" and "never seen" is what
        stops a wave of unknown wallets reading as a wave of bad ones.
        """
        wallet = event.get("wallet")
        if not wallet or event.get("side") != "buy":
            return
        seen = self._actor_seen.setdefault(token, set())
        if wallet in seen:
            return
        seen.add(wallet)
        self.hot_state.touch_token(token)
        # The per-token wallet sets are unbounded otherwise: a day of launches
        # would accumulate every buyer of every token that ever traded.
        for stale in [key for key in self._actor_seen
                      if key not in self.hot_state.active_tokens]:
            self._actor_seen.pop(stale, None)

        score = None
        if self.wallet_intel is not None:
            try:
                score = self.wallet_intel.get_wallet_score(wallet)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("wallet score unavailable for %s: %s", wallet, exc)
        notional = observation.get("notional_sol")
        entry = Entry(
            token=token, wallet=str(wallet),
            timestamp=float(observation.get("timestamp", time.time())),
            skill=(float(getattr(score, "overall_score", 0.0)) if score is not None else None),
            capital_usd=((float(notional) * self.sol_price_usd)
                         if notional and self.sol_price_usd > 0 else None),
        )
        self.wallet_independence.record_entries([entry])
        # Retained so First25 DNA, actor-adjusted flow and swarm probability
        # have something to read. Bounded to the fingerprint depth: only the
        # opening sequence is what those models consume, and keeping every
        # buyer of every token is how a day of launches becomes a leak.
        entries = self._actor_entries.setdefault(token, [])
        if len(entries) < self.buyer_dna.depth:
            entries.append(entry)
        # A roster wallet entering is the only moment its decision can be
        # captured point-in-time. Recorded here, on the live stream, because
        # the follow verdict needs the price WE could have got after OUR
        # delay -- an explorer can say what they paid, not what following
        # them would have cost us.
        observer = getattr(self, "observe_benchmark_entry", None)
        if observer is not None:
            try:
                observer(token, entry.wallet, entry.timestamp,
                         price=event.get("price"),
                         launch_age_s=event.get("launch_age_s"),
                         buyer_rank=len(entries))
            except Exception as exc:  # pragma: no cover - measurement only
                logger.debug("benchmark entry not recorded: %s", exc)
        for stale in [key for key in self._actor_entries
                      if key not in self.hot_state.active_tokens]:
            self._actor_entries.pop(stale, None)

    async def _source_consumer_loop(self):
        """Index every source event the instant it arrives.

        The mesh used to be polled in a batch, on a cadence, by nobody -- the
        beautiful source architecture existed and the runtime never called it.
        Now producers run per source and this consumer awaits the fan-in
        queue, so a chat channel that saw a launch first reaches the decision
        without waiting for the slowest feed in the forest to finish its
        request.
        """
        # Connections first. A relay or a Telegram client that has not
        # connected answers its first poll with a failure, and the mesh would
        # count that against the source rather than against the connection.
        # The collector's client, when it has one: it is already connected
        # and already authorised, and it holds the session file every other
        # Telegram client in this process would otherwise fail to open.
        failures = await start_transports(
            self.transports,
            telegram_client=getattr(self.social_intel, "_telegram_client", None))
        if failures:
            logger.warning("TRANSPORTS %d of %d failed to start: %s",
                           len(failures), len(self.transports),
                           ", ".join(sorted(failures)))
        self.transport_start_failures = failures
        started = await self.source_mesh.start()
        logger.info("SOURCE_MESH started %d producers (%d transports, %d connected)",
                    started, len(self.transports), len(self.transports) - len(failures))
        while self._running:
            try:
                event = await self.source_mesh.next_event()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("source consumer error: %s", exc)
                await asyncio.sleep(0.01)
                continue
            try:
                self._index_source_event(event)
            except Exception as exc:
                logger.warning("source event indexing failed: %s", exc)

    def _index_source_event(self, event: Any) -> None:
        """One event into the per-token index and the lead-lag graph."""
        for token in event.token_addresses:
            observations = self._source_events.setdefault(token, [])
            observations.append(event)
            # Bounded per token: a viral mint attracts thousands of posts and
            # only the earliest few carry lead information.
            if len(observations) > 50:
                observations.pop(0)
            self.source_genealogy.record(SourcePost(
                source_id=event.source_id, token=token,
                posted_at=event.source_at, observed_at=event.observed_at))
            # A source naming a token we hold is new evidence about it.
            self.request_redecision(token)

    async def _poll_sources(self) -> int:
        """Collect from every declared source and index events by token.

        Runs off the money path. Source events inform the decision; they do
        not gate it, and a mesh that is entirely dead must degrade the
        decision's evidence rather than stop it from being made.
        """
        if not self.source_mesh.sources:
            return 0
        try:
            events = await self.source_mesh.collect()
        except Exception as exc:  # pragma: no cover - the mesh guards its own
            logger.warning("source mesh collection failed: %s", exc)
            return 0
        for event in events:
            for token in event.token_addresses:
                observations = self._source_events.setdefault(token, [])
                observations.append(event)
                # Bounded per token: a viral mint attracts thousands of posts
                # and only the earliest few carry lead information.
                if len(observations) > 50:
                    observations.pop(0)
                # The genealogy learns which source leads which from the same
                # stream, so the lead-lag graph is built from what we actually
                # observed rather than from publication timestamps a source
                # controls and can backdate.
                self.source_genealogy.record(SourcePost(
                    source_id=event.source_id, token=token,
                    posted_at=event.source_at, observed_at=event.observed_at))
        for stale in [key for key in self._source_events
                      if key not in self.hot_state.active_tokens]:
            self._source_events.pop(stale, None)
        return len(events)

    def _refresh_independence(self) -> None:
        """Recompute the independence matrix on a cadence, not per trade.

        Pair statistics are quadratic in the wallets sharing a launch, so this
        deliberately does not run on the hot path. Independence changes over
        launches rather than over individual trades, so a periodic recompute
        loses nothing a per-trade one would have caught.
        """
        if time.time() - self._independence_computed_at < self._independence_interval:
            return
        self._independence_computed_at = time.time()
        self.independence_report = self.wallet_independence.compute()
        # Ancestry over every wallet the matrix knows, applied once here so
        # every consumer of the report sees the compressed number rather than
        # each having to remember to compress it.
        if self.independence_report.status == "OK" and self.independence_report.scores:
            ancestry = self.funder_ancestry.analyse(
                list(self.independence_report.scores))
            if ancestry.ok:
                before = sum(self.independence_report.scores.values())
                self.independence_report.scores = compress_independence(
                    self.independence_report.scores, ancestry)
                after = sum(self.independence_report.scores.values())
                self.ancestry_report = ancestry
                logger.info(
                    "FUNDER ANCESTRY %s; independence mass %.2f -> %.2f",
                    ancestry.detail, before, after)
        # The third compression, after behaviour and after ancestry: wallets
        # funded out of the same exchange hot wallet inside the same seconds,
        # in similar amounts. Ancestry cannot see that -- the hot wallet funds
        # hundreds of thousands of unrelated people, so the edge carries no
        # information, which is exactly what routing through an exchange buys.
        # Applied last because it is the weakest evidence of the three, and
        # capped so it can only ever discount.
        self._apply_temporal_clusters()
        logger.info("INDEPENDENCE status=%s pairs=%d wallets=%d",
                    self.independence_report.status,
                    self.independence_report.observed_pairs,
                    len(self.independence_report.scores))

    def _apply_temporal_clusters(self) -> None:
        """Discount wallets funded together out of one exchange hot wallet.

        Silent and harmless without measured emission rates: a cluster scored
        against a guessed base rate would make every busy exchange look like
        a conspiracy, so `find_clusters` refuses and this does nothing.
        """
        withdrawals = list(getattr(self, "exchange_withdrawals", ()) or ())
        if not withdrawals or self.independence_report.status != "OK":
            return
        span = (max(w.timestamp for w in withdrawals)
                - min(w.timestamp for w in withdrawals))
        rates = dict(getattr(self, "exchange_rates", {}) or {})
        for source in {w.source for w in withdrawals}:
            measured = measure_source_rate(withdrawals, source, span)
            if measured is not None:
                rates[source] = measured
        self.exchange_rates = rates
        clusters = find_clusters(
            withdrawals, target_buyers=list(self.independence_report.scores),
            source_rates=rates)
        self.temporal_clusters = clusters
        discounts = independence_discounts(clusters)
        self.temporal_discounts = discounts
        if not discounts:
            return
        for wallet, multiplier in discounts.items():
            if wallet in self.independence_report.scores:
                self.independence_report.scores[wallet] *= multiplier
        logger.info(
            "TEMPORAL FUNDING %d cluster(s) discounted %d wallet(s); "
            "strongest cut %.0f%%",
            len([c for c in clusters if c.status == "OK"]), len(discounts),
            (1.0 - min(discounts.values())) * 100.0)

    def _read_ignition(self, token: str):
        """Where this token's narrative is in its lifecycle.

        Buyer arrivals are the INDEPENDENT ones -- the actor graph's
        compression has already been applied. Passing raw wallet counts would
        let a Sybil manufacture the acceleration this exists to detect.
        """
        events = self._source_events.get(token) or []
        dnas = {dna.source_id: dna for dna in (self._source_dnas or {}).values()} \
            if isinstance(getattr(self, "_source_dnas", None), dict) else {}
        leads = {item.upstream: item.lead_rate
                 for item in (self.source_genealogy.lead_lag() or ())} \
            if hasattr(self.source_genealogy, "lead_lag") else {}
        touches = touches_from_events(events, dnas=dnas, lead_rates=leads)
        entries = self._actor_entries.get(token) or []
        scores = (self.independence_report.scores
                  if self.independence_report.status == "OK" else {})
        arrivals: List[float] = []
        for entry in entries:
            weight = float(scores.get(entry.wallet, 1.0))
            # A wallet counted at a third of an arrival is a wallet the graph
            # has said is a third of an actor. Rounding it up to one is how a
            # cluster becomes a crowd.
            if weight >= 0.5:
                arrivals.append(float(entry.timestamp))
        return self.ignition.read(touches, arrivals)

    def _read_disagreement(self, token: str, candidate: Any, prediction: Any,
                           liquidity: float):
        """Dispersion across the desk's own readings of this launch.

        Assembled from the reports the desk already produces rather than by
        re-running the models: a second set of calls would be a second set of
        answers, and two views of one model disagreeing with itself is not
        disagreement.
        """
        intelligence: Dict[str, Any] = {}
        try:
            intelligence["actor"] = self.actor_intelligence(token, liquidity)
        except Exception as exc:
            logger.debug("actor view unavailable for %s: %s", token, exc)
        hazard = self.rug_hazard.get_hazard(token)
        if hazard is not None:
            intelligence["hazard"] = {"status": hazard.data_status,
                                      "hazard_30s": hazard.hazard_30s}
        monster = getattr(prediction, "monster_probability", None)
        if monster is not None:
            intelligence["monster"] = {"status": "OK", "probability": monster}
        events = self._source_events.get(token) or []
        if events:
            credibility = [float(getattr(item, "credibility", 0.0) or 0.0)
                           for item in events
                           if getattr(item, "credibility", None) is not None]
            if credibility:
                intelligence["source"] = {"status": "OK",
                                          "credibility": max(credibility)}
        reading = self._read_ignition(token)
        if reading.ok and reading.probability is not None:
            # The crowd question, not the source question. Whether the next
            # wave of independent buyers is coming is what decides whether a
            # position compounds; how good the source was does not.
            intelligence["ignition"] = {"status": "OK",
                                        "probability": reading.probability}
        probe = {"size_tokens": 0}
        status, ratio = self._exit_capacity(token, probe)
        intelligence["capacity"] = {
            "status": "OK" if str(status).startswith("OK") else "DATA_BLOCKED",
            "ratio": ratio}
        reading = self.disagreement.read(views_from_intelligence(intelligence))
        if reading.ok and reading.sigma > 0:
            logger.debug("DISAGREEMENT %s sigma=%.3f shrink=%.2f (%s)",
                         token, reading.sigma, reading.shrink, reading.detail)
        return reading
