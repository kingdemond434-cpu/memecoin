"""Which market is this, and what the promotion ledger is told about it.

Extracted from `src/main.py`, which enforces a line budget precisely so that
a service gets pulled out rather than the ceiling raised. These two belong
together and belong away from the desk: `current_regime` is the label every
recorded outcome is filed under, and `_record_forward_evidence` is the only
writer of the ledger the promotion gate reads. Neither is on the hot path and
neither decides a trade.

The regime label is deliberately coarse, and "unknown" is deliberately not a
regime: the gate requires three of them before it will call a mechanism
general, and a bucket named for the absence of measurement would satisfy that
requirement with nothing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from src.research.forward_evidence import Outcome as ForwardOutcome

logger = logging.getLogger(__name__)


class RegimeAndEvidence:
    """Mixin: what market this is, and what the ledger is told about it."""

    @property
    def current_regime(self) -> str:
        """A coarse label for the market the desk is trading in right now.

        Deliberately coarse and deliberately observable: launch rate and the
        24h SOL move are things the desk already measures, and a finely
        conditioned regime over a handful of observations is a worse label
        that looks better.

        Returns "unknown" when the inputs are missing, and "unknown" does not
        count toward the promotion gate's diversity requirement -- a desk that
        never measured the market must not satisfy it with one bucket.
        """
        builder = getattr(self, "dataset_builder", None)
        stats = (builder.current_market_state()
                 if builder is not None and hasattr(builder, "current_market_state")
                 else {}) or {}
        # Compatibility for isolated callers which provide the measurements
        # directly through a research stub. The production miner is not used:
        # it discovers mechanisms and does not collect market state.
        if (stats.get("meme_launch_rate_1h") is None
                or stats.get("sol_change_24h") is None):
            research = getattr(self, "global_research", None)
            fallback = (research.get_stats() if research else {}) or {}
            if fallback.get("meme_launch_rate_1h") is not None:
                stats = fallback
        launch_rate = stats.get("meme_launch_rate_1h")
        sol_change = stats.get("sol_change_24h")
        if launch_rate is None or sol_change is None:
            return "unknown"
        hot = float(launch_rate) >= float(
            self.global_config.get("regime_hot_launch_rate", 300))
        rising = float(sol_change) >= 0
        if hot and rising:
            return "euphoria"
        if hot:
            return "churn"
        return "bull" if rising else "bear"

    def _record_forward_evidence(self, payload: Dict[str, Any]) -> None:
        """Feed one trade outcome into the promotion ledger.

        Declines are recorded too. A ledger fed only on entries measures the
        trades we took and says nothing about the ones we passed on, which is
        half of what a decision policy does and the half that hides its
        mistakes.
        """
        try:
            self.forward_evidence.record(ForwardOutcome(
                token=str(payload.get("token", "")),
                entered=bool(payload.get("entered")),
                regime=str((payload.get("regime") or self.current_regime or "unknown")),
                realized_pnl_usd=float(payload.get("realized_pnl_usd", 0.0) or 0.0),
                equity_at_decision_usd=float(self.wallet_equity_usd or 0.0),
                real_fill=bool(payload.get("entered") and not self.dry_run),
                rugged=bool(payload.get("rugged")),
                max_multiple=(float(payload["max_feasible_multiple"])
                              if payload.get("max_feasible_multiple") is not None else None),
                execution_attempted=bool(payload.get("attempted")),
                execution_succeeded=bool(payload.get("entered")),
                catastrophic=bool(payload.get("rugged")
                                  and float(payload.get("realized_pnl_usd", 0.0) or 0.0)
                                  <= -float(self.wallet_equity_usd or 0.0) * 0.5),
            ))
        except (TypeError, ValueError) as exc:
            logger.debug("forward evidence record failed: %s", exc)
        # Persisted on a cadence rather than every outcome: an fsync per trade
        # is latency the decision path does not need to pay, and losing at
        # most a minute of counts to a crash costs a minute of shadow running.
        if time.time() - self._evidence_saved_at > 60.0:
            self._evidence_saved_at = time.time()
            self.forward_evidence.save()
        # The census carries the denominator every ratio above is computed
        # against; losing it to a restart would silently reset those ratios.
        if time.time() - self._census_saved_at > 120.0:
            self._census_saved_at = time.time()
            self.launch_census.save()
            self.calibration.save()
