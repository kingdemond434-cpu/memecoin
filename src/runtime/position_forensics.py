"""What a position proved about itself, kept out of the decision path.

Four readings that share one property: none of them decides anything at the
moment it is taken, and all of them change what the desk does next time.

  * exit readiness -- was the sell executable before it was needed
  * entry-state key -- the bucket a position's excursions belong to
  * excursions      -- what the position actually put the desk through
  * exit mode       -- which return distribution it turned out to be in

They live here rather than in main.py because main.py has a line budget that
exists for a reason, and because these four are one concern: measuring a
position rather than running one.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from src.execution.exit_readiness import MODE_UNDECIDED, choose_exit_mode

logger = logging.getLogger(__name__)


class PositionForensics:
    """Mixin. Every method here is measurement; none is on the money path."""

    def _record_exit_readiness(self, token: str, position: Dict[str, Any],
                               staged: Optional[Any]) -> None:
        """How long after the fill the exit became executable."""
        ledger = getattr(self, "exit_readiness", None)
        if ledger is None:
            return
        from src.execution.exit_readiness import SellTemplate

        now = time.time()
        template = SellTemplate(
            token=token, built_at=now,
            accounts=tuple(getattr(staged, "accounts", ()) or ()),
            program_id=str(getattr(staged, "program_id", "") or "staged"),
            blockhash=str(getattr(staged, "blockhash", "") or ""),
            blockhash_at=now, ready=staged is not None,
            detail=str(getattr(staged, "detail", "") or ""))
        try:
            ledger.on_fill(token, float(position.get("entry_time", now)), template)
        except Exception as exc:  # pragma: no cover - reporting only
            logger.debug("exit readiness not recorded for %s: %s", token, exc)

    def _entry_state_key(self, position: Dict[str, Any]) -> str:
        """A coarse bucket for the state this position was ENTERED in.

        Coarse deliberately. The excursion profile needs enough positions per
        bucket to mean anything, and a key fine enough to be interesting is a
        key with one observation in it.
        """
        candidate = position.get("candidate") or {}
        source = str((candidate.get("metadata") or {}).get("sleeve", "")
                     or position.get("sleeve", "") or "t0")
        liquidity = float(position.get("liquidity_usd", 0.0) or 0.0)
        band = ("thin" if liquidity < 5_000 else
                "mid" if liquidity < 50_000 else "deep")
        return f"{source}:{band}"

    def _record_excursion(self, position: Dict[str, Any]) -> None:
        """MFE and MAE for a closed position, on EXECUTABLE marks.

        Executable, for the same reason the label fix mattered: the highest
        price a token printed is not a price anyone could have sold into, and
        an excursion measured on chart highs flatters every entry equally.
        """
        ledger = getattr(self, "excursions", None)
        if ledger is None:
            return
        favourable = position.get("feasible_high_water_multiple")
        if favourable is None:
            # No measured capacity means no executable high. Recording the
            # chart high here would quietly reintroduce the bug that made
            # the model predict peaks nobody could fill.
            return
        adverse = float(position.get("low_water_multiple", 1.0) or 1.0)
        try:
            ledger.record(self._entry_state_key(position),
                          mfe=float(favourable) - 1.0, mae=adverse - 1.0)
        except Exception as exc:  # pragma: no cover - reporting only
            logger.debug("excursion not recorded: %s", exc)

    def _exit_mode(self, token: str, position: Dict[str, Any]) -> str:
        """Which return distribution this position is in, from evidence.

        Consulted by the exit path so the two distributions are not served by
        one blended policy: most tokens are +30% and gone, rare ones are on
        their way to +3000%, and a policy tuned for either destroys the
        other. UNDECIDED is a real answer and the default -- neither mode is
        chosen from nothing.
        """
        report = getattr(self, "cohort_reports", {}).get(token)
        prediction = position.get("prediction") or {}
        probability = prediction.get("monster_probability")
        try:
            choice = choose_exit_mode(
                report, float(probability) if probability is not None else None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("exit mode undecided for %s: %s", token, exc)
            return MODE_UNDECIDED
        previous = position.get("exit_mode")
        if choice.mode != previous:
            position["exit_mode"] = choice.mode
            position["exit_mode_reasons"] = list(choice.reasons)
            if choice.mode != MODE_UNDECIDED:
                logger.info("EXIT MODE %s -> %s (%s)", token, choice.mode,
                            "; ".join(choice.reasons) or choice.detail)
        return choice.mode


    # --- benchmark wallets ------------------------------------------------

    def observe_benchmark_entry(self, token: str, wallet: str, at: float,
                                price: Optional[float],
                                launch_age_s: Optional[float] = None,
                                buyer_rank: Optional[int] = None) -> bool:
        """Record a roster wallet entering a launch, at the moment it does.

        This is the miner the corpus was missing. A roster of famous
        addresses with nothing in it produces a permanently DATA_BLOCKED
        report -- which is honest, and useless.

        Reconstruction happens HERE, on the live stream, rather than by
        pulling their history from an explorer, for a reason that matters to
        the answer: the follow verdict needs the price WE could have got at
        +50/100/250/500/1000ms after their fill, and that is a property of
        this desk's own view of the chain. An explorer can tell you what they
        paid. Only the live stream can tell you what following them would
        have cost us.
        """
        corpus = getattr(self, "benchmark_corpus", None)
        if corpus is None or wallet not in getattr(corpus, "wallets", {}):
            return False
        from src.research.benchmark_wallets import WalletDecision

        corpus.record(WalletDecision(
            wallet=wallet, token=token, entered_at=float(at),
            launch_age_s=launch_age_s, buyer_rank=buyer_rank,
            entry_price=price))
        logger.info("BENCHMARK %s entered %s; following it is now measurable "
                    "once this launch resolves",
                    corpus.wallets[wallet].label or wallet[:8], token[:12])
        return True

    def mark_benchmark_delays(self, token: str, at: float, price: float) -> int:
        """Fill in what a follower would have paid, at each delay.

        Called on every mark. A price observed 120ms after a roster wallet's
        entry is the price a follower 120ms behind would have paid, so it
        lands in the nearest delay bucket that is not yet filled. Buckets are
        filled once and never overwritten -- the FIRST observation at or
        after a delay is the one a follower would have hit.
        """
        corpus = getattr(self, "benchmark_corpus", None)
        if corpus is None or price is None or price <= 0:
            return 0
        from src.research.benchmark_wallets import FOLLOW_DELAYS_S

        filled = 0
        for wallet in getattr(corpus, "wallets", {}):
            for row in corpus.decisions(wallet):
                if row.token != token or row.resolved:
                    continue
                elapsed = float(at) - float(row.entered_at)
                for delay in FOLLOW_DELAYS_S:
                    if elapsed >= delay and delay not in row.price_at_delay:
                        row.price_at_delay[delay] = float(price)
                        filled += 1
        return filled

    def resolve_benchmark_decisions(self, token: str, at: float,
                                    price: float) -> int:
        """Close every open roster decision on this token, once.

        Written once and never revised, like every other resolution in this
        desk: a payoff that can be edited after the fact is not a measurement
        of anything.
        """
        corpus = getattr(self, "benchmark_corpus", None)
        if corpus is None or price is None or price <= 0:
            return 0
        closed = 0
        for wallet in getattr(corpus, "wallets", {}):
            for row in corpus.decisions(wallet):
                if row.token != token or row.resolved:
                    continue
                row.exit_price = float(price)
                row.exited_at = float(at)
                closed += 1
        return closed

    def _promote_benchmark_candidates(self) -> None:
        """Find wallets in OUR stream that behave like the roster's.

        The unbiased half of the corpus, and the reason it exists. The roster
        is a list of addresses other people published BECAUSE they won -- a
        sample selected on the outcome, with no denominator. This runs over
        wallets the desk observed itself, where the ones that lost were
        equally visible, so a statistic computed over what it finds means
        something the roster's never can.

        Candidates are registered for reconstruction, not trusted: being
        found here earns a wallet the same follow verdict every roster
        address has to pass, which is a bar most of them will fail.
        """
        corpus = getattr(self, "benchmark_corpus", None)
        if corpus is None:
            return
        from src.research.benchmark_wallets import BenchmarkWallet, discover_candidates

        observations: Dict[str, list] = {}
        for wallet in list(getattr(corpus, "wallets", {})):
            rows = corpus.decisions(wallet)
            if rows:
                observations[wallet] = rows
        for wallet, rows in getattr(self, "_observed_wallet_decisions", {}).items():
            observations.setdefault(wallet, rows)
        if not observations:
            return
        try:
            found = discover_candidates(observations)
        except Exception as exc:  # pragma: no cover - measurement only
            logger.debug("benchmark discovery skipped: %s", exc)
            return
        for address, evidence in found:
            if address in corpus.wallets:
                continue
            corpus.register(BenchmarkWallet(
                address=address, label=f"discovered_{address[:6]}",
                rationale=(f"found in this desk's own stream: "
                           f"{evidence['decisions']} resolved decisions, "
                           f"median launch age {evidence['median_launch_age_s']}s. "
                           + evidence["basis"])))
            logger.info("BENCHMARK discovered %s from our own observations "
                        "(%d decisions); it now has to pass the same follow "
                        "verdict as every published address",
                        address[:12], evidence["decisions"])
