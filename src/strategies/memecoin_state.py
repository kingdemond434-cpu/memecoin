"""Point-in-time state for holder, developer, rotation and narrative evidence.

These are measurements, not trading models.  They deliberately expose
``DATA_BLOCKED`` and ``MEASURING`` states instead of converting a missing
wallet owner, price mark or historical score into zero.  Predictive authority
still belongs to the chronologically validated models which consume the
records written here.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence


PCT_FIELDS = (
    "top_10_pct", "top_20_pct", "dev_pct", "insider_pct",
    "bundler_pct", "fresh_wallet_pct", "whale_pct", "cluster_pct",
)


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentage(value: Any) -> Optional[float]:
    number = _finite(value)
    if number is None or number < 0 or number > 100:
        return None
    return number


@dataclass(frozen=True)
class HolderSnapshot:
    timestamp: float
    top_10_pct: Optional[float] = None
    top_20_pct: Optional[float] = None
    dev_pct: Optional[float] = None
    insider_pct: Optional[float] = None
    bundler_pct: Optional[float] = None
    fresh_wallet_pct: Optional[float] = None
    whale_pct: Optional[float] = None
    cluster_pct: Optional[float] = None
    unique_holders: Optional[int] = None
    source: str = ""

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any], *,
                     timestamp: Optional[float] = None,
                     source: str = "") -> "HolderSnapshot":
        values = {name: _percentage(payload.get(name)) for name in PCT_FIELDS}
        holders = payload.get("unique_holders", payload.get("holder_count"))
        try:
            unique = int(holders) if holders is not None and int(holders) >= 0 else None
        except (TypeError, ValueError):
            unique = None
        return cls(timestamp=float(timestamp if timestamp is not None else
                                   payload.get("timestamp", time.time())),
                   unique_holders=unique, source=source or str(payload.get("source", "")),
                   **values)


class HolderTrajectoryMonitor:
    """Distribution levels and changes without pretending accounts are actors."""

    def __init__(self, capacity_per_token: int = 512):
        self.capacity = max(2, int(capacity_per_token))
        self._history: Dict[str, Deque[HolderSnapshot]] = defaultdict(
            lambda: deque(maxlen=self.capacity))

    def record(self, token: str, snapshot: HolderSnapshot) -> bool:
        if not token or not math.isfinite(snapshot.timestamp):
            return False
        if all(getattr(snapshot, name) is None for name in PCT_FIELDS) \
                and snapshot.unique_holders is None:
            return False
        history = self._history[token]
        if history and snapshot.timestamp <= history[-1].timestamp:
            # Retain PIT order. A delayed observation can still be useful to
            # the offline lake, but inserting it here would rewrite live state.
            return False
        history.append(snapshot)
        return True

    def record_mapping(self, token: str, payload: Dict[str, Any], *,
                       timestamp: Optional[float] = None,
                       source: str = "") -> bool:
        return self.record(token, HolderSnapshot.from_mapping(
            payload, timestamp=timestamp, source=source))

    def state(self, token: str, as_of: Optional[float] = None) -> Dict[str, Any]:
        cutoff = time.time() if as_of is None else float(as_of)
        history = [item for item in self._history.get(token, ())
                   if item.timestamp <= cutoff]
        if not history:
            return {"status": "DATA_BLOCKED", "detail": "no holder snapshot"}
        latest = history[-1]
        current = {name: getattr(latest, name) for name in PCT_FIELDS
                   if getattr(latest, name) is not None}
        if latest.unique_holders is not None:
            current["unique_holders"] = latest.unique_holders
        result: Dict[str, Any] = {
            "status": "OK", "timestamp": latest.timestamp,
            "source": latest.source, "current": current,
            "observed_fields": sorted(current), "snapshots": len(history),
            "trajectory_status": "MEASURING",
            "changes": {}, "velocity_per_second": {},
        }
        if len(history) < 2:
            result["detail"] = "one snapshot; levels measured, trajectory pending"
            return result
        previous = history[-2]
        elapsed = latest.timestamp - previous.timestamp
        if elapsed <= 0:
            result["detail"] = "latest snapshots have no positive time separation"
            return result
        changes: Dict[str, float] = {}
        velocities: Dict[str, float] = {}
        for name in PCT_FIELDS:
            before, after = getattr(previous, name), getattr(latest, name)
            if before is not None and after is not None:
                changes[name] = after - before
                velocities[name] = (after - before) / elapsed
        if previous.unique_holders is not None and latest.unique_holders is not None:
            changes["unique_holders"] = float(latest.unique_holders - previous.unique_holders)
            velocities["unique_holders"] = changes["unique_holders"] / elapsed
        result.update({"changes": changes, "velocity_per_second": velocities})
        if changes:
            result["trajectory_status"] = "OK"
            result["detail"] = f"change over {elapsed:.3f}s"
        else:
            result["detail"] = "successive snapshots have no common measured fields"
        return result


@dataclass(frozen=True)
class DevEvent:
    timestamp: float
    event_type: str
    wallet: str = ""
    token_amount: Optional[float] = None
    supply_share_pct: Optional[float] = None
    destination: str = ""
    severity: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


class DevWalletMonitor:
    """Observed creator/linked-wallet behaviour and explicit emergency facts."""

    EMERGENCY_TYPES = frozenset({
        "authority_mutation", "lp_removed", "sell_route_failed",
        "malicious_mint_operation", "transfer_restriction_changed",
    })

    def __init__(self, capacity_per_token: int = 2_048):
        self.capacity = max(16, int(capacity_per_token))
        self._developers: Dict[str, str] = {}
        self._linked: Dict[str, set[str]] = defaultdict(set)
        self._events: Dict[str, Deque[DevEvent]] = defaultdict(
            lambda: deque(maxlen=self.capacity))
        self._balance_pct: Dict[str, tuple[float, float, str]] = {}

    def register(self, token: str, developer: str,
                 linked_wallets: Iterable[str] = ()) -> None:
        if token and developer:
            self._developers[token] = str(developer)
            self._linked[token].add(str(developer))
        self._linked[token].update(str(wallet) for wallet in linked_wallets if wallet)

    def controls(self, token: str, wallet: str) -> bool:
        return bool(wallet and wallet in self._linked.get(token, set()))

    def record_balance(self, token: str, balance_pct: Any, *,
                       timestamp: Optional[float] = None,
                       source: str = "") -> bool:
        value = _percentage(balance_pct)
        if not token or value is None:
            return False
        self._balance_pct[token] = (value, float(timestamp or time.time()), source)
        return True

    def record(self, token: str, event: DevEvent) -> bool:
        if not token or not event.event_type or not math.isfinite(event.timestamp):
            return False
        events = self._events[token]
        if events and event.timestamp < events[-1].timestamp:
            return False
        events.append(event)
        return True

    def record_trade(self, token: str, *, wallet: str, side: str,
                     timestamp: float, token_amount: Any = None,
                     supply_share_pct: Any = None,
                     evidence: Optional[Dict[str, Any]] = None) -> bool:
        if not self.controls(token, wallet):
            return False
        kind = "dev_sell" if side == "sell" else "dev_buy"
        return self.record(token, DevEvent(
            timestamp=float(timestamp), event_type=kind, wallet=wallet,
            token_amount=_finite(token_amount),
            supply_share_pct=_percentage(supply_share_pct),
            evidence=dict(evidence or {})))

    def state(self, token: str, as_of: Optional[float] = None,
              recent_seconds: float = 300.0) -> Dict[str, Any]:
        cutoff = time.time() if as_of is None else float(as_of)
        events = [event for event in self._events.get(token, ())
                  if event.timestamp <= cutoff]
        recent = [event for event in events
                  if 0 <= cutoff - event.timestamp <= recent_seconds]
        balance = self._balance_pct.get(token)
        if not self._developers.get(token) and not events and balance is None:
            return {"status": "DATA_BLOCKED", "detail": "developer not identified"}
        emergency = [event for event in recent
                     if event.event_type in self.EMERGENCY_TYPES
                     or event.severity.lower() == "critical"]
        sells = [event for event in recent if event.event_type == "dev_sell"]
        measured_sell_share = sum(event.supply_share_pct for event in sells
                                  if event.supply_share_pct is not None)
        result = {
            "status": "OK" if events or balance is not None else "MEASURING",
            "developer": self._developers.get(token, ""),
            "linked_wallets": len(self._linked.get(token, set())),
            "events": len(events), "recent_events": len(recent),
            "recent_dev_sells": len(sells),
            "recent_measured_sell_supply_pct": measured_sell_share,
            "hard_vetoes": sorted({event.event_type for event in emergency}),
            "balance_pct": balance[0] if balance else None,
            "balance_status": "OK" if balance else "DATA_BLOCKED",
            "latest_events": [asdict(event) for event in recent[-10:]],
        }
        if result["status"] == "MEASURING":
            result["detail"] = "developer identified; no balance or behaviour observed yet"
        return result


@dataclass(frozen=True)
class RotationTrade:
    token: str
    wallet: str
    side: str
    timestamp: float
    wallet_quality: Optional[float]
    independence_weight: Optional[float]
    notional_usd: Optional[float]
    narrative: str = ""


class SmartWalletRotationTracker:
    """Quality- and independence-weighted capital flow across tokens/narratives."""

    def __init__(self, capacity: int = 20_000):
        self._trades: Deque[RotationTrade] = deque(maxlen=max(100, int(capacity)))

    def record(self, trade: RotationTrade) -> bool:
        if not trade.token or not trade.wallet or trade.side not in {"buy", "sell"}:
            return False
        if not math.isfinite(trade.timestamp):
            return False
        self._trades.append(trade)
        return True

    def report(self, as_of: Optional[float] = None,
               window_seconds: float = 120.0) -> Dict[str, Any]:
        cutoff = time.time() if as_of is None else float(as_of)
        recent = [trade for trade in self._trades
                  if 0 <= cutoff - trade.timestamp <= window_seconds]
        measured = [trade for trade in recent
                    if trade.wallet_quality is not None
                    and trade.independence_weight is not None
                    and trade.notional_usd is not None]
        if not measured:
            return {"status": "DATA_BLOCKED",
                    "detail": "no trade has quality, independence and notional together",
                    "observations": len(recent)}
        by_token: Dict[str, float] = defaultdict(float)
        by_narrative: Dict[str, float] = defaultdict(float)
        actors: Dict[str, set[str]] = defaultdict(set)
        for trade in measured:
            sign = 1.0 if trade.side == "buy" else -1.0
            value = (sign * float(trade.wallet_quality)
                     * float(trade.independence_weight)
                     * float(trade.notional_usd))
            by_token[trade.token] += value
            if trade.narrative:
                by_narrative[trade.narrative] += value
                actors[trade.narrative].add(trade.wallet)
        ranked_tokens = sorted(by_token.items(), key=lambda item: item[1], reverse=True)
        ranked_narratives = sorted(by_narrative.items(), key=lambda item: item[1], reverse=True)
        return {
            "status": "OK", "measurement": "quality_x_independence_x_notional",
            "window_seconds": window_seconds, "observations": len(measured),
            "unmeasured_observations": len(recent) - len(measured),
            "token_flow": dict(ranked_tokens),
            "narrative_flow": dict(ranked_narratives),
            "independent_wallets_by_narrative": {
                name: len(wallets) for name, wallets in actors.items()},
            # This is a ranking signal, explicitly not a continuation probability.
            "leader": ranked_tokens[0][0] if ranked_tokens and ranked_tokens[0][1] > 0 else None,
        }


def social_price_disagreement(
    social: Sequence[Dict[str, Any]],
    market: Sequence[Dict[str, Any]],
    *, as_of: Optional[float] = None,
) -> Dict[str, Any]:
    """Observed attention change relative to observed repricing.

    The result is an uncalibrated research feature.  It never returns a
    probability and therefore cannot acquire trading authority by itself.
    """
    cutoff = time.time() if as_of is None else float(as_of)
    social_points = sorted(
        ((float(item.get("timestamp", 0) or 0), _finite(item.get("velocity")))
         for item in social if float(item.get("timestamp", 0) or 0) <= cutoff),
        key=lambda item: item[0])
    social_points = [(timestamp, value) for timestamp, value in social_points
                     if value is not None and value >= 0]
    market_points = sorted(
        ((float(item.get("timestamp", 0) or 0),
          _finite(item.get("price_multiple", item.get("curve_progress"))))
         for item in market if float(item.get("timestamp", 0) or 0) <= cutoff),
        key=lambda item: item[0])
    market_points = [(timestamp, value) for timestamp, value in market_points
                     if value is not None and value > 0]
    if len(social_points) < 2 or len(market_points) < 2:
        return {"status": "DATA_BLOCKED", "evidence_score": None,
                "detail": "two social and two market observations are required"}
    social_before, social_after = social_points[0][1], social_points[-1][1]
    price_before, price_after = market_points[0][1], market_points[-1][1]
    social_log_change = math.log1p(social_after) - math.log1p(social_before)
    price_log_change = math.log(price_after / price_before)
    return {
        "status": "OK", "authority": "research_feature_only",
        "social_log_change": social_log_change,
        "price_log_change": price_log_change,
        "evidence_score": social_log_change - max(0.0, price_log_change),
        "social_points": len(social_points), "market_points": len(market_points),
    }
