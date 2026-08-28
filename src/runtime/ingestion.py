"""Where a mined fact goes, and which facts are worth mining.

Split out of `main.py`. Every method here is on the ingestion path and none
is on the decision path, which is exactly the boundary worth drawing: a
change to how a Telegram message is parsed cannot now touch how a position
is sized, and the two stop sharing a merge surface.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class MinedRecordIngestion:
    """Routing mined and streamed records into the models that consume them.

    A mixin for the same reason the reporting surface is one: every method
    here reads and writes the desk's own subsystems, so a collaborator would
    need all of them injected and rewriting that plumbing is a
    behaviour-changing refactor dressed as a tidy-up. Moved verbatim, so a
    regression can only come from the move.

    The grouping is a real seam rather than a convenient one. Nothing here
    decides anything: these methods take what a miner or a stream produced,
    decide which model it belongs to, and hand it over. The half that chooses
    WHICH tokens and wallets are worth a miner pass lives here too, because
    it answers the same question from the other end -- what is worth
    collecting, and where does what we collected go."""

    def _mineable_tokens(self) -> List[str]:
        """Which mints are worth spending a miner pass on, most urgent first.

        Open positions before candidates before merely-observed launches.
        Mining the holder structure of every mint on Solana is impossible and
        useless; the ones that matter are the ones a position is in or might
        be taken in, and a miner pass spent elsewhere is a pass not spent
        here.
        """
        ordered: List[str] = []
        seen = set()
        for group in (self.elogw_engine.open_positions,
                      self._latest_curve_state,
                      self._latest_pool_state):
            for token in group:
                if token and token not in seen:
                    seen.add(token)
                    ordered.append(token)
        return ordered

    def _contended_accounts(self) -> List[str]:
        """The accounts our next transaction will write to.

        Prioritization fees are per-account: the fee that cleared on some
        unrelated NFT mint says nothing about what it costs to land on THIS
        curve. Asking about the pools and mints we are actually about to touch
        is the difference between a measured bid and a chain-wide average
        dressed up as one.
        """
        accounts: List[str] = []
        seen = set()
        for token in self._mineable_tokens()[:8]:
            pool = self._latest_pool_state.get(token)
            for address in (getattr(pool, "pool", ""), token):
                if address and address not in seen:
                    seen.add(address)
                    accounts.append(address)
        return accounts

    def _known_lp_mints(self) -> List[str]:
        """LP mints of pools we have actually decoded.

        Only decoded pools appear here. Deriving the PDA and mining it for a
        pool we have never read would produce a confident answer about an
        account we cannot prove belongs to this token.
        """
        mints: List[str] = []
        seen = set()
        for account in self._pool_accounts.values():
            lp_mint = getattr(account, "lp_mint", "")
            if lp_mint and lp_mint not in seen:
                seen.add(lp_mint)
                mints.append(lp_mint)
        return mints

    def _known_deployers(self) -> List[str]:
        """Deployers of the tokens currently worth spending a pass on.

        Public chain addresses, read from the stream's own events. Their
        history is mined from the public ledger and nowhere else.
        """
        addresses: List[str] = []
        seen = set()
        for token in self._mineable_tokens()[:12]:
            curve = self._latest_curve_state.get(token)
            pool = self._latest_pool_state.get(token)
            for address in (getattr(curve, "creator", ""),
                            getattr(pool, "coin_creator", "")):
                if address and address not in seen:
                    seen.add(address)
                    addresses.append(address)
        return addresses

    def _tracked_wallets(self) -> List[str]:
        """Wallets whose balance we want to watch move.

        The elite set first, because a tracked wallet's balance dropping is it
        deploying into something we have not seen yet -- which is the earliest
        signal available that is not on the curve at all.
        """
        wallets: List[str] = []
        seen = set()
        for source in (getattr(self.wallet_intelligence, "elite_wallets", None) or {},
                       getattr(self, "_recent_funders", None) or {}):
            for address in source:
                if address and address not in seen:
                    seen.add(address)
                    wallets.append(address)
        return wallets[:100]

    def _name_search_terms(self) -> List[str]:
        """Names and symbols worth searching the wider venue set for.

        A symbol is only meaningful against the corpus of everything else
        called that. This supplies the queries; the corpus miner supplies the
        comparison.
        """
        terms: List[str] = []
        seen = set()
        for token in self._mineable_tokens()[:8]:
            curve = self._latest_curve_state.get(token)
            for field in ("symbol", "name"):
                value = str(getattr(curve, field, "") or "").strip()
                if len(value) >= 2 and value.lower() not in seen:
                    seen.add(value.lower())
                    terms.append(value)
        return terms

    def _ingest_mined_records(self, miner_id: str,
                              records: List[Dict[str, Any]]) -> None:
        """Route mined records into the lake and the models that use them.

        Written as observations rather than as state: a mined fact is
        something we were told at a time, and the point-in-time builder is
        what keeps it usable for training later. Overwriting live state from a
        source polled every fifteen seconds would put a stale number in front
        of a decision the stream had already updated.
        """
        if getattr(self, "channel_book", None) is not None:
            self._harvest_channels(miner_id, records)
        for record in records:
            # Chain-wide execution conditions belong to no episode: they are
            # the state of the world every decision in this moment is made in.
            if record.get("slot_time_ratio") is not None:
                self._network_health = dict(record)
                continue
            if record.get("fee_p50_lamports") is not None:
                self._priority_fees = dict(record)
                continue
            # Supply control on a migrated pool, keyed by LP mint rather than
            # by the token's own mint.
            lp_mint = str(record.get("lp_mint", "") or "")
            if lp_mint:
                self._ingest_lp_supply(lp_mint, record)
                continue
            # A wallet reading: the deployer's public history, or a tracked
            # balance that moved.
            address = str(record.get("address", "") or "")
            if address:
                self._ingest_wallet_record(address, record)
                continue
            mint = str(record.get("mint", "") or "")
            if not mint:
                # A market-wide row belongs to every episode, not to one.
                self._market_context = dict(record)
                continue
            try:
                self.dataset_builder.record_market_observation(
                    mint, {"type": "mined", "measurement": miner_id,
                           "timestamp": float(record.get("_fetched_at", time.time())),
                           **record})
            except Exception as exc:
                logger.debug("mined record for %s not recorded: %s", mint, exc)
            # Holder concentration is a hazard input, not a curiosity: a top
            # holder at 40% of supply is a rug that has not happened yet.
            if record.get("top1_share") is not None:
                self.rug_hazard.record_observation(mint, {
                    "type": "holder_structure",
                    "timestamp": float(record.get("_fetched_at", time.time())),
                    "top1_share": record.get("top1_share"),
                    "top10_share": record.get("top10_share"),
                    "concentration_hhi": record.get("concentration_hhi"),
                    "data_status": "OK"})
            # Retained mint or freeze authority is the difference between a
            # coin that CAN be rugged by its creator and one that cannot.
            if record.get("mint_renounced") is not None:
                self.rug_hazard.record_observation(mint, {
                    "type": "authority",
                    "timestamp": float(record.get("_fetched_at", time.time())),
                    "mint_renounced": record.get("mint_renounced"),
                    "freeze_renounced": record.get("freeze_renounced"),
                    "data_status": "OK"})

    def _harvest_channels(self, miner_id: str,
                          records: List[Dict[str, Any]]) -> None:
        """Pull public Telegram handles out of whatever any miner just returned.

        This is the bootstrap for the whole Telegram side, and it is the only
        honest one available: rather than guessing channel names, the desk
        reads the links that tokens, repos and posts publish about themselves.
        A Pump token's own profile links its own channel; a channel's messages
        link the channels it forwards. Discovery therefore converges on what
        the market is actually pointing at rather than on what was configured
        some months ago.

        Every harvested handle is only a CANDIDATE. It is verified by fetching
        its own public preview before a single message is read from it.
        """
        book = getattr(self, "channel_book", None)
        if book is None or not records:
            return
        for record in records[:200]:
            for field in ("description", "text", "title", "url", "links"):
                value = record.get(field)
                if isinstance(value, (list, tuple)):
                    value = " ".join(str(item.get("url", item))
                                     if isinstance(item, dict) else str(item)
                                     for item in value)
                if not value:
                    continue
                try:
                    book.harvest(str(value), source=miner_id)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("channel harvest from %s: %s", miner_id, exc)

    def _ingest_telegram_messages(self, records: List[Dict[str, Any]]) -> None:
        """Public Telegram messages: corroboration first, then attention.

        The corroboration half runs before anything else and is the reason
        this hook exists. A mint carried by a channel the figure registry
        already knew belonged to a public figure is the only thing that can
        turn a celebrity CLAIM into a celebrity ANNOUNCEMENT, and the registry
        was written before this launch existed, which is what stops a launch
        supplying its own confirmation.
        """
        watch = getattr(self, "identity_watch", None)
        for record in records:
            channel = str(record.get("channel", "") or "")
            mints = [mint for mint in (record.get("mints") or []) if mint]
            if watch is not None and channel and mints:
                try:
                    watch.note_channel_message(
                        channel, mints, at=self._telegram_timestamp(record))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("telegram corroboration: %s", exc)
            # Views are the closest thing to a MEASURED crowd reading that
            # public Telegram exposes: a call into a channel nobody reads and
            # one into a channel with forty thousand readers are different
            # events, and a message count cannot tell them apart.
            views = record.get("views")
            for mint in mints:
                try:
                    self.dataset_builder.record_market_observation(
                        mint, {"type": "telegram_public",
                               "measurement": "telegram:public_preview",
                               "timestamp": self._telegram_timestamp(record),
                               "channel": channel, "views": views,
                               "text": str(record.get("text", ""))[:500],
                               "data_status": "OK"})
                except Exception as exc:
                    logger.debug("telegram observation for %s: %s", mint, exc)

    @staticmethod
    def _telegram_timestamp(record: Dict[str, Any]) -> float:
        """The message's own posting time, falling back to when we read it.

        Never silently: a message whose timestamp cannot be parsed is stamped
        with the read time, which is later, and lateness is the safe direction
        -- it can only make a corroboration look like a reaction, never make a
        reaction look like an announcement.
        """
        raw = str(record.get("posted_at", "") or "")
        if raw:
            try:
                return datetime.fromisoformat(raw).timestamp()
            except ValueError:
                pass
        return float(record.get("_fetched_at", time.time()))

    def _ingest_discovered_pools(self, records: List[Dict[str, Any]]) -> None:
        """Pools an outside operator saw that our own stream may not have.

        The launch census is the denominator the whole promotion ladder rests
        on, and it is only as complete as discovery is. A pool reported here
        that never appeared in our census is a decoder gap or a program we do
        not decode -- and from the inside, that failure is indistinguishable
        from a quiet market, which is exactly why it is counted separately
        rather than merged in.
        """
        census = getattr(self, "launch_census", None)
        for record in records:
            mint = str(record.get("mint", "") or "")
            if not mint:
                continue
            self._discovered_pools[mint] = float(
                record.get("_fetched_at", time.time()))
            if census is not None and not census.knows(mint):
                self._discovery_misses += 1
            try:
                self.dataset_builder.record_market_observation(
                    mint, {"type": "discovered_pool",
                           "measurement": str(record.get("_source", "")),
                           "timestamp": float(record.get("_fetched_at", time.time())),
                           "liquidity_usd": record.get("liquidity_usd"),
                           "fdv_usd": record.get("fdv_usd"),
                           "venue": record.get("venue", ""),
                           "data_status": "OK"})
            except Exception as exc:
                logger.debug("discovered pool %s not recorded: %s", mint, exc)

    def _assess_identity(self, mint: str, event: Dict[str, Any]) -> None:
        """Who does this launch claim to be, and did that name confirm it.

        Recorded as an observation rather than acted on. The verdict classes
        have no realised base rate yet -- `identity_watch.report()` says which
        ones are still DATA_BLOCKED -- and sizing on a class whose payoff has
        never been measured is exactly the guess this desk refuses elsewhere.
        So the assessment becomes a point-in-time FEATURE now and a sizing
        input only once the forward ledger has priced it.

        The one thing it does immediately is name the launch on /status, which
        is worth having on its own: a desk that cannot tell an operator "this
        one claims a head of state and nothing has confirmed it" is a desk
        whose operator finds out from the chart.
        """
        watch = getattr(self, "identity_watch", None)
        if watch is None or not watch.figures:
            return
        try:
            assessment = watch.assess(
                mint,
                symbol=str(event.get("symbol", "") or ""),
                name=str(event.get("name", "") or ""),
                description=str(event.get("description", "") or "")[:2000],
                links=[str(value) for value in
                       (event.get("telegram"), event.get("twitter"),
                        event.get("website"), event.get("uri")) if value],
                deployer=str(event.get("creator", "") or ""),
                created_at=float(event.get("timestamp", time.time())))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("identity assessment for %s: %s", mint, exc)
            return
        if not assessment.claimed:
            return
        self._identity_claims[mint] = assessment
        while len(self._identity_claims) > 2_000:
            self._identity_claims.pop(next(iter(self._identity_claims)))
        logger.info("IDENTITY %s claims %s -> %s", mint,
                    ", ".join(claim.display for claim in assessment.claims),
                    assessment.verdict.value)
        try:
            self.dataset_builder.record_market_observation(
                mint, {"type": "identity_claim", "measurement": "identity_watch",
                       "timestamp": float(event.get("timestamp", time.time())),
                       **assessment.to_dict(), "data_status": "OK"})
        except Exception as exc:
            logger.debug("identity observation for %s: %s", mint, exc)

    def _ingest_lp_supply(self, lp_mint: str, record: Dict[str, Any]) -> None:
        """Route an LP reading onto the token whose pool it belongs to.

        A rug after migration is an LP event: the pool is drained by whoever
        holds the LP tokens. Burned LP means that cannot happen and the hazard
        should fall; a live supply concentrated in one holder means it can
        happen in one transaction and the hazard should rise. Neither is
        visible on the curve until it has already happened.
        """
        token = ""
        for mint, account in self._pool_accounts.items():
            if getattr(account, "lp_mint", "") == lp_mint:
                token = mint
                break
        if not token:
            return
        self.rug_hazard.record_observation(token, {
            "type": "lp_supply",
            "timestamp": float(record.get("_fetched_at", time.time())),
            "lp_burned": record.get("lp_burned"),
            "lp_supply": record.get("lp_supply"),
            "lp_top1_share": record.get("lp_top1_share"),
            "data_status": "OK"})
        try:
            self.dataset_builder.record_market_observation(
                token, {"type": "mined", "measurement": "chain:lp_supply",
                        "timestamp": float(record.get("_fetched_at", time.time())),
                        **record})
        except Exception as exc:
            logger.debug("lp record for %s not recorded: %s", token, exc)

    def _ingest_wallet_record(self, address: str, record: Dict[str, Any]) -> None:
        """A wallet's public history, onto the tokens that wallet deployed.

        Everything here is read from the open ledger: signature counts, their
        timing, and a balance. It is behavioural inference over public chain
        data and reaches nothing that is not already public.

        The reading lands on the tokens this address deployed, because that is
        where it changes a decision. A deployer account a few hours old whose
        signature window is already saturated is the serial-launcher pattern,
        and that is a hazard fact about every token it launched -- not a
        curiosity filed against an address nobody looks up.
        """
        when = float(record.get("_fetched_at", time.time()))
        observation = {
            "type": "deployer_history",
            "timestamp": when,
            "deployer": address,
            **{key: value for key, value in record.items()
               if not key.startswith("_")},
        }
        touched = 0
        for token in self._tokens_deployed_by(address):
            self.rug_hazard.record_observation(token, dict(observation))
            try:
                self.dataset_builder.record_market_observation(
                    token, {"measurement": "chain:deployer", **observation})
            except Exception as exc:
                logger.debug("deployer record for %s not recorded: %s", token, exc)
            touched += 1
        if not touched:
            # A tracked wallet we hold no position against. Kept as a balance
            # reading so a funder emptying itself is still visible later, but
            # it changes no decision now and is not pretended to.
            self._wallet_readings[address] = dict(record)

    def _tokens_deployed_by(self, address: str) -> List[str]:
        """Which watched tokens this address deployed, from stream state."""
        tokens: List[str] = []
        for token, curve in self._latest_curve_state.items():
            if getattr(curve, "creator", "") == address:
                tokens.append(token)
        for token, pool in self._latest_pool_state.items():
            if getattr(pool, "coin_creator", "") == address and token not in tokens:
                tokens.append(token)
        return tokens
