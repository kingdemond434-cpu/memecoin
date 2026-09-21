"""Raptor as a challenger route, never as an assumption.

Raptor is a Solana DEX aggregator that routes across ~20 liquidity sources with
sub-millisecond local quote computation, keeps pool state warm off a Yellowstone
stream rather than polling RPC, and can submit through Jet TPU. On paper that is
strictly better than paying Jupiter's 1 RPS keyed free tier for a quote the desk
needs in the first two seconds of a launch.

"On paper" is the problem. The desk has been wrong before about a route that
looked faster, and the way it was wrong was invisible: a better QUOTE that lands
less often is worse, and quote quality is the only half that is easy to measure.
So Raptor enters exactly the way the Rust ingress did -- in SHADOW, quoting
alongside the incumbent, promoted only by forward evidence on realised value.

**What is recorded per paired observation** (`RouteObservation`): quote latency,
quoted out, realised out when a fill is later attributed, the route taken, RPC
calls avoided, whether it landed, and total execution cost. Both arms, same
mint, same size, same moment.

**What promotion requires.** Not a better average. `RaptorShadow.verdict()`
demands a minimum number of paired observations in which BOTH arms produced a
realised fill, and then an exact paired sign test on realised value net of cost.
Quote-only pairs are counted and reported separately and are explicitly not
evidence for promotion -- a quote is a claim, and the entire point of this
module is that the desk stopped taking claims.

**On self-hosting.** The default base URL is a local instance, because Raptor's
whole latency argument evaporates over a hosted round trip. This module does not
download, install or launch anything: it speaks HTTP to a URL the operator
configured. A binary the desk did not put there is not a dependency this module
will acquire on its own.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import shape only
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

from src.execution.jupiter_jito import RouteType, SwapQuote

logger = logging.getLogger(__name__)

#: A local Raptor. Overridable; the hosted Solana Tracker endpoint works too and
#: costs the round trip this route exists to avoid.
DEFAULT_RAPTOR_URL = "http://127.0.0.1:8080"

#: Paired observations with a realised fill on BOTH arms before the shadow is
#: willing to say anything at all. Chosen so a sign test at this size can reach
#: significance -- with fewer, "Raptor won" is a coin that came up heads.
DEFAULT_MIN_PAIRED_FILLS = 200

#: One-sided p below which the challenger is judged genuinely better.
DEFAULT_ALPHA = 0.01

#: A challenger that loses this badly is demoted permanently rather than left
#: quoting forever. Same latch discipline as the ingress parity gate.
DEFAULT_DEMOTE_ALPHA = 0.01


class ShadowStatus(Enum):
    DATA_BLOCKED = "DATA_BLOCKED"
    SHADOW = "SHADOW"
    PROMOTED = "PROMOTED"
    DEMOTED = "DEMOTED"


def _binomial_tail(successes: int, trials: int) -> float:
    """P(X >= successes) for a fair coin. Exact, so small samples stay honest.

    A normal approximation at n=30 is wrong in exactly the direction that
    flatters a challenger, which is the direction this gate exists to resist.
    """
    if trials <= 0:
        return 1.0
    successes = max(0, min(int(successes), int(trials)))
    total = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    return total / float(2 ** trials)


@dataclass
class RouteObservation:
    """One arm's answer to one question, with everything it cost.

    `realised_out` is deliberately Optional and deliberately separate from
    `quoted_out`. A route that quotes 1.05x and fills at 0.97x lost, and a
    record that folds the two together cannot say so.
    """

    route: str
    mint: str
    input_amount: int
    quote_latency_ms: float = 0.0
    quoted_out: Optional[int] = None
    realised_out: Optional[int] = None
    route_path: List[str] = field(default_factory=list)
    rpc_calls: int = 0
    landed: bool = False
    landing_latency_ms: Optional[float] = None
    total_cost_lamports: int = 0
    slot: Optional[int] = None
    error: str = ""
    observed_at: float = field(default_factory=time.time)

    @property
    def quoted(self) -> bool:
        return self.quoted_out is not None and self.quoted_out > 0

    @property
    def filled(self) -> bool:
        return self.landed and self.realised_out is not None

    def net_value(self) -> Optional[float]:
        """Realised output net of everything paying for it.

        Returns None rather than zero when unrealised. Unmeasured is not zero;
        scoring an absent fill as a zero-value fill would let a route that
        never lands look merely mediocre instead of useless.
        """
        if not self.filled:
            return None
        assert self.realised_out is not None
        return float(self.realised_out) - float(self.total_cost_lamports)


@dataclass
class PairedObservation:
    """The incumbent and the challenger, asked the same question at once."""

    key: str
    incumbent: RouteObservation
    challenger: RouteObservation

    @property
    def both_filled(self) -> bool:
        return self.incumbent.filled and self.challenger.filled

    @property
    def both_quoted(self) -> bool:
        return self.incumbent.quoted and self.challenger.quoted

    def value_delta(self) -> Optional[float]:
        left = self.challenger.net_value()
        right = self.incumbent.net_value()
        if left is None or right is None:
            return None
        return left - right

    def quote_delta_bps(self) -> Optional[float]:
        if not self.both_quoted:
            return None
        assert self.incumbent.quoted_out and self.challenger.quoted_out
        base = float(self.incumbent.quoted_out)
        if base <= 0:
            return None
        return (float(self.challenger.quoted_out) - base) / base * 10_000.0

    def latency_delta_ms(self) -> float:
        return (self.challenger.quote_latency_ms
                - self.incumbent.quote_latency_ms)


class RaptorClient:
    """HTTP client for a Raptor instance. Quotes only, unless told otherwise.

    Every failure returns None and logs a DATA_BLOCKED rather than raising, so
    a challenger that is down degrades the shadow's sample size instead of the
    desk's execution.
    """

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, *,
                 timeout_s: float = 3.0, session: Any = None):
        self.base_url = (base_url or os.getenv("RAPTOR_API_URL")
                         or DEFAULT_RAPTOR_URL).rstrip("/")
        self.api_key = api_key or os.getenv("RAPTOR_API_KEY", "")
        self.timeout_s = float(timeout_s)
        self._session = session
        self._owns_session = session is None
        #: Counted, not assumed. Raptor's claim is that it avoids RPC by
        #: keeping pool state warm off a gRPC stream; the shadow reports how
        #: many RPC calls each arm actually made, and this counter is this
        #: arm's honest zero.
        self.rpc_calls = 0

    async def start(self) -> None:
        if self._session is not None:
            return
        if aiohttp is None:  # pragma: no cover - environment
            raise RuntimeError("aiohttp is required for the Raptor client")
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            headers=headers)
        self._owns_session = True

    async def stop(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None

    async def healthy(self) -> bool:
        if self._session is None:
            return False
        try:
            async with self._session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception as exc:
            logger.debug("Raptor health DATA_BLOCKED: %s", exc)
            return False

    async def get_quote(self, input_mint: str, output_mint: str, amount: int,
                        *, slippage_bps: int = 100
                        ) -> Tuple[Optional[SwapQuote], float]:
        """Returns the quote and the latency it took, always both.

        Latency is returned rather than logged because it is half the reason
        this route is a candidate at all, and a comparison that records the
        output but not the delay cannot evaluate the actual claim.
        """
        if self._session is None:
            raise RuntimeError("Raptor client is not started")
        if amount <= 0:
            return None, 0.0
        params = {"inputMint": input_mint, "outputMint": output_mint,
                  "amount": str(int(amount)),
                  "slippageBps": str(int(slippage_bps))}
        started = time.perf_counter()
        try:
            async with self._session.get(f"{self.base_url}/quote",
                                         params=params) as resp:
                elapsed = (time.perf_counter() - started) * 1000.0
                if resp.status != 200:
                    logger.warning("Raptor quote DATA_BLOCKED: HTTP %s",
                                   resp.status)
                    return None, elapsed
                data = await resp.json()
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.warning("Raptor quote DATA_BLOCKED: %s", exc)
            return None, elapsed
        return self._parse_quote(data, input_mint, output_mint, amount,
                                 slippage_bps), elapsed

    @staticmethod
    def _parse_quote(data: Any, input_mint: str, output_mint: str,
                     amount: int, slippage_bps: int) -> Optional[SwapQuote]:
        """Read Raptor's payload without inventing the fields it omits.

        A missing output amount yields None. It does not yield a quote of zero,
        which downstream would read as a real route offering nothing and would
        make the challenger look merely unattractive rather than unavailable.
        """
        if not isinstance(data, dict):
            return None
        out = data.get("amountOut", data.get("outAmount"))
        try:
            output_amount = int(out)
        except (TypeError, ValueError):
            return None
        if output_amount <= 0:
            return None
        raw_route = data.get("route") or data.get("routePlan") or []
        route: List[Dict[str, Any]] = [
            item if isinstance(item, dict) else {"leg": str(item)}
            for item in (raw_route if isinstance(raw_route, list) else [])]
        impact = data.get("priceImpactPct", data.get("priceImpact"))
        try:
            impact_pct = float(impact)
        except (TypeError, ValueError):
            impact_pct = 0.0
        minimum = data.get("minAmountOut", data.get("otherAmountThreshold"))
        try:
            min_output = int(minimum)
        except (TypeError, ValueError):
            min_output = int(output_amount * (1 - slippage_bps / 10_000.0))
        return SwapQuote(
            input_mint=input_mint, output_mint=output_mint,
            input_amount=int(amount), output_amount=output_amount,
            price_impact_pct=impact_pct, route=route,
            route_type=RouteType.RAPTOR, fees_bps=0,
            min_output_amount=min_output, raw_quote=dict(data))

    async def observe(self, input_mint: str, output_mint: str, amount: int,
                      *, slippage_bps: int = 100) -> RouteObservation:
        """A quote expressed as a shadow observation, error included."""
        quote, latency = await self.get_quote(
            input_mint, output_mint, amount, slippage_bps=slippage_bps)
        observation = RouteObservation(
            route="raptor", mint=output_mint, input_amount=int(amount),
            quote_latency_ms=latency, rpc_calls=self.rpc_calls)
        if quote is None:
            observation.error = "no_quote"
            return observation
        observation.quoted_out = quote.output_amount
        observation.route_path = [
            str(leg.get("label") or leg.get("ammKey") or leg.get("leg") or "?")
            for leg in quote.route]
        return observation


class RaptorShadow:
    """Paired forward comparison between the incumbent route and Raptor.

    Holds no authority. `should_route_through_challenger` answers False until
    `verdict()` says PROMOTED, and once DEMOTED it latches -- a route that has
    been shown to lose does not get to re-litigate on a lucky streak.
    """

    def __init__(self, *, incumbent: str = "jupiter_v1",
                 min_paired_fills: int = DEFAULT_MIN_PAIRED_FILLS,
                 alpha: float = DEFAULT_ALPHA,
                 demote_alpha: float = DEFAULT_DEMOTE_ALPHA):
        self.incumbent = incumbent
        self.min_paired_fills = int(min_paired_fills)
        self.alpha = float(alpha)
        self.demote_alpha = float(demote_alpha)
        self.pairs: List[PairedObservation] = []
        self._latched_demotion: Optional[str] = None
        self._promoted = False

    def record(self, pair: PairedObservation) -> None:
        self.pairs.append(pair)

    def record_pair(self, key: str, incumbent: RouteObservation,
                    challenger: RouteObservation) -> PairedObservation:
        pair = PairedObservation(key=key, incumbent=incumbent,
                                 challenger=challenger)
        self.record(pair)
        return pair

    # -- evidence --------------------------------------------------------

    def _filled_pairs(self) -> List[PairedObservation]:
        return [pair for pair in self.pairs if pair.both_filled]

    def quote_summary(self) -> Dict[str, Any]:
        """Quote-only evidence, reported and explicitly not promotable."""
        deltas = [pair.quote_delta_bps() for pair in self.pairs]
        deltas = [value for value in deltas if value is not None]
        latencies = [pair.latency_delta_ms() for pair in self.pairs
                     if pair.both_quoted]
        summary: Dict[str, Any] = {
            "paired_quotes": len(deltas),
            "challenger_quote_wins": sum(1 for value in deltas if value > 0),
            "promotable": False,
            "why_not": ("a quote is a claim; promotion requires realised "
                        "fills on both arms"),
        }
        if deltas:
            summary["median_quote_delta_bps"] = statistics.median(deltas)
        if latencies:
            summary["median_latency_delta_ms"] = statistics.median(latencies)
        return summary

    def verdict(self) -> Dict[str, Any]:
        """PROMOTED / DEMOTED / SHADOW / DATA_BLOCKED, with the arithmetic."""
        if self._latched_demotion:
            return {"status": ShadowStatus.DEMOTED.value,
                    "reason": self._latched_demotion,
                    "latched": True,
                    "paired_fills": len(self._filled_pairs())}

        filled = self._filled_pairs()
        deltas = [pair.value_delta() for pair in filled]
        deltas = [value for value in deltas if value is not None]
        decisive = [value for value in deltas if value != 0.0]
        wins = sum(1 for value in decisive if value > 0)
        trials = len(decisive)

        base: Dict[str, Any] = {
            "incumbent": self.incumbent,
            "challenger": "raptor",
            "paired_observations": len(self.pairs),
            "paired_fills": len(filled),
            "decisive_pairs": trials,
            "challenger_wins": wins,
            "quotes": self.quote_summary(),
        }
        if deltas:
            base["median_net_value_delta"] = statistics.median(deltas)

        if len(filled) < self.min_paired_fills:
            base.update({
                "status": ShadowStatus.DATA_BLOCKED.value,
                "reason": (f"paired realised fills {len(filled)} below the "
                           f"{self.min_paired_fills} required; quote-only "
                           "agreement is not evidence of execution quality")})
            return base

        win_p = _binomial_tail(wins, trials)
        lose_p = _binomial_tail(trials - wins, trials)
        base["p_challenger_better"] = win_p
        base["p_challenger_worse"] = lose_p

        if lose_p < self.demote_alpha:
            self._latched_demotion = (
                f"challenger lost {trials - wins}/{trials} decisive paired "
                f"fills (p={lose_p:.2e}); demotion is permanent")
            base.update({"status": ShadowStatus.DEMOTED.value,
                         "reason": self._latched_demotion, "latched": True})
            return base

        if win_p < self.alpha:
            self._promoted = True
            base.update({"status": ShadowStatus.PROMOTED.value,
                         "reason": (f"challenger won {wins}/{trials} decisive "
                                    f"paired fills (p={win_p:.2e})")})
            return base

        base.update({"status": ShadowStatus.SHADOW.value,
                     "reason": (f"challenger won {wins}/{trials} decisive "
                                f"paired fills (p={win_p:.3f}); not separated "
                                "from a coin")})
        return base

    def should_route_through_challenger(self) -> bool:
        """The only question the execution path is allowed to ask."""
        if self._latched_demotion:
            return False
        return self._promoted and (
            self.verdict().get("status") == ShadowStatus.PROMOTED.value)


async def observe_both(incumbent_call: Any, challenger_call: Any, *,
                       key: str) -> PairedObservation:
    """Quote both arms concurrently so the comparison is of one moment.

    Sequential quoting on a token minted forty seconds ago compares two
    different markets and attributes the difference to the router. Both
    coroutines are awaited together, and an exception on either arm becomes an
    errored observation rather than losing the pair.
    """
    async def _guard(call: Any, route: str) -> RouteObservation:
        try:
            return await call
        except Exception as exc:
            return RouteObservation(route=route, mint="", input_amount=0,
                                    error=f"{type(exc).__name__}: {exc}")

    left, right = await asyncio.gather(
        _guard(incumbent_call, "incumbent"),
        _guard(challenger_call, "raptor"))
    return PairedObservation(key=key, incumbent=left, challenger=right)
