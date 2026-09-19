"""Depth resolution: what can actually be traded, and what can be left.

Split out of main.py because two different questions kept being answered by
one number. "Liquidity" is what the curve is holding; "depth" is what a
position could really be sold into inside an impact we would accept. They are
close enough to confuse and far enough apart to lose money on: a flat
fraction of the first was standing in for the second everywhere, and a
constant-product round trip costs the fee at any size, so that fraction was
never pricing cost. It was a guess at exit, in a system that can measure it.
"""

import logging
import time
from typing import Any, Dict, Optional

from src.detection.token_detector import TokenCandidate
from src.execution.tradeability import curve_depth_usd

#: Wrapped SOL, the quote side of every Jupiter probe here.
WSOL_MINT = "So11111111111111111111111111111111111111112"

logger = logging.getLogger(__name__)


class DepthResolution:
    """Mixin: resolves liquidity and measured exit depth for one token."""

    def _local_liquidity(self, token: str) -> float:
        """Tradeable depth from the streamed curve, in USD. Zero when unknown.

        A bonding curve's quote-side depth IS its SOL reserve: that is what a
        seller can be paid out of, and no quote from anywhere makes it larger.
        Reading it locally removes a network round trip from directly in front
        of the T0 sizing decision.

        The VIRTUAL reserve is the reading, not the real one. This used to
        prefer real reserves and fall back to virtual, which sounds more
        conservative and is actually incoherent: a fresh Pump curve holds zero
        real SOL and so reported its full 30 SOL of virtual depth, then the
        first 0.1 SOL buy made it report $20. Depth fell thirtyfold because
        somebody bought. Everything downstream of that number -- the minimum
        liquidity gate at $5,000, the concentration ceiling, the position
        cap -- was reading a quantity that collapses exactly when a launch
        starts working, so the launches most worth sizing were the ones it
        sized smallest, when it did not reject them outright.

        Virtual reserves are monotone in curve progress and are what the
        constant product actually prices impact from, so that is what
        "liquidity" means here. The real reserve does bound what the curve can
        pay a seller, and that question is answered properly by
        ``_measured_depth_usd`` instead of being smuggled into this one.
        """
        state = self._latest_curve_state.get(token)
        if state is None or not state.tradeable or self.sol_price_usd <= 0:
            return 0.0
        lamports = int(state.virtual_sol_reserves or 0)
        if lamports <= 0:
            return 0.0
        return (lamports / 1e9) * float(self.sol_price_usd)

    async def _resolve_liquidity(self, candidate: TokenCandidate) -> float:
        explicit = candidate.initial_liquidity_usd or candidate.metadata.get("liquidity_usd")
        if explicit and float(explicit) > 0:
            return float(explicit)
        # The curve already tells us this. Asking Jupiter meant a T0 decision
        # paid a network round trip to learn something the streamed reserves
        # state outright -- and the sizing engine cannot start until the
        # answer arrives, so the round trip sat directly in front of the
        # decision it was feeding.
        local = self._local_liquidity(candidate.address)
        if local > 0:
            return local
        if not self.jupiter or not self.jupiter._session or self.sol_price_usd <= 0:
            return 0.0
        quote = await self.jupiter.get_quote(WSOL_MINT, candidate.address, 100_000_000, slippage_bps=300)
        if not quote or quote.output_amount <= 0:
            return 0.0
        impact = max(float(quote.price_impact_pct), 0.001)
        estimate = (0.1 * self.sol_price_usd) / impact
        observation = {"type": "liquidity", "liquidity_usd": estimate, "source": "quote_depth_estimate",
                       "price_impact_pct": quote.price_impact_pct, "timestamp": time.time()}
        self.dataset_builder.record_market_observation(candidate.address, observation)
        self.rug_hazard.record_observation(candidate.address, observation)
        return estimate

    def _measured_depth_usd(self, token: str) -> Optional[float]:
        """Notional this token can really be SOLD into, or None if unmeasured.

        The ceiling a position is sized against, when it exists. None is not a
        zero and not a licence: it means the exit frontier could not be taken
        (no streamed curve, or a curve holding no real SOL yet), and the
        caller falls back to the declared flat fraction knowing it is an
        assumption rather than a measurement.
        """
        return curve_depth_usd(
            self._latest_curve_state.get(token),
            float(self.global_config.get("acceptable_exit_impact", 0.10)),
            float(self.sol_price_usd))


class FollowableTrades:
    """Trades by wallets whose followability has been measured.

    Separated from the decode handler because the same three facts -- is this
    wallet worth listening to, is this a buy or an exit, is the deployer
    distributing -- feed three different consumers, and inlining them put a
    composite score and a hand-picked threshold in the middle of the hot path.
    """

    #: Below this the wallet's followability is a formula's opinion rather
    #: than a measurement, and it does not get to raise an elite-buy event.
    MEASURED_FLOOR = 0.50

    def _record_followable_trade(self, token: str, event: Dict[str, Any]) -> None:
        wallet = str(event.get("wallet") or "")
        if not wallet:
            return
        consensus = getattr(self, "wallet_consensus", None)
        if consensus is None:
            return
        confidences = consensus.followable_provider() or {}
        confidence = confidences.get(wallet)
        if confidence is None:
            return
        at = float(event.get("timestamp", time.time()) or time.time())
        is_buy = event.get("side") == "buy"
        if is_buy:
            consensus.observe_buy(token, wallet, at)
        elif wallet and wallet == str(
                getattr(self._latest_curve_state.get(token), "creator", "") or ""):
            # The deployer taking the other side of the agreement. Several of
            # the consensus rules exist precisely to refuse that case.
            consensus.observe_deployer_sell(token, at)
        # An elite-buy event used to require `overall_score >= 0.7`: a
        # hand-weighted composite compared against a number somebody picked.
        # A measured wallet is one whose followed outcomes have a positive
        # lower bound, and only those raise the event.
        if confidence < self.MEASURED_FLOOR:
            return
        from src.strategies.information_graph import LeadEventType
        self.info_graph.record_event(
            token,
            LeadEventType.ELITE_WALLET_BUY if is_buy else LeadEventType.SMART_WALLET_EXIT,
            wallet, "wallet", at, event)
