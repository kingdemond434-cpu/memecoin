"""Continuous, evidence-backed Solana exit hazard tracking."""

import asyncio
import json
import os
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
from pathlib import Path

import joblib
import numpy as np

from src.chains.rpc_manager import ChainConfig, RPCManager
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.information_graph import AdversarialAdaptationDetector
from src.strategies.wallet_intelligence import WalletIntelligenceEngine

logger = logging.getLogger(__name__)

HAZARD_FEATURE_NAMES = (
    "sell_share_60s", "buy_deceleration", "volume_collapse", "drawdown_to_date",
    "route_degradation", "liquidity_withdrawal", "concentration_increase", "explicit_risk_event",
)


class HazardTrigger(Enum):
    CREATOR_TRANSFER = "creator_transfer"
    INSIDER_SELL = "insider_sell"
    SMART_WALLET_EXIT = "smart_wallet_exit"
    LIQUIDITY_WITHDRAWAL = "liquidity_withdrawal"
    CONCENTRATION_CHANGE = "concentration_change"
    BUY_DECELERATION = "buy_deceleration"
    SELL_ACCELERATION = "sell_acceleration"
    HOLDER_DISTRIBUTION = "holder_distribution"
    VOLUME_COLLAPSE = "volume_collapse"
    ROUTE_DEGRADATION = "route_degradation"
    SOCIAL_VELOCITY_COLLAPSE = "social_velocity_collapse"
    FAILED_MIGRATION = "failed_migration"
    DEV_WALLET_ACTIVATION = "dev_wallet_activation"
    BUNDLE_DETECTION = "bundle_detection"


@dataclass
class HazardSignal:
    trigger: HazardTrigger
    strength: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardState:
    token: str
    chain: str
    current_hazard: float = 0.0
    hazard_30s: float = 0.0
    hazard_5m: float = 0.0
    hazard_30m: float = 0.0
    signals: List[HazardSignal] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    exit_recommended: bool = False
    exit_urgency: str = "NONE"
    data_status: str = "DATA_BLOCKED"
    blocked_reason: str = "no_market_observations"


#: Tokens RESIDENT at once, across hazard_states, observations and
#: token_metadata. Every launch on the feed registers in all three and
#: nothing ever left them: they are keyed by mint with no eviction, so a
#: desk watching a continuous launch stream grows until the kernel stops
#: it. That is not hypothetical -- on 2026-08-29 this service was
#: OOM-killed repeatedly at roughly eight-minute intervals with ~4,300
#: tokens accumulated in a few hours, while its own cgroup ceiling sat
#: untouched, because the growth is in the NUMBER of tokens rather than
#: in any one of them.
#:
#: Calibrated, not guessed. One observation measures ~1.9 KB, so a token
#: carrying 50 of them costs ~25 KB and the arithmetic is unforgiving:
#: 100k tokens is 2.5 GB and 1M tokens is 25 GB, on a 4 GB box shared with
#: ~150 other processes under a 900 MB cgroup ceiling. 8,000 resident
#: tokens is ~200 MB, which is what this desk can actually hold while
#: staying alive.
#:
#: This bounds what is RESIDENT, not what is OBSERVED. Evictions spill to
#: the research lake first, so the desk still accumulates history over
#: millions of tokens -- it just stops trying to hold them all in RAM at
#: once. An uncapped desk does not observe more; it observes nothing,
#: because it is killed every eight minutes and loses all of it.
DEFAULT_MAX_TRACKED_TOKENS = 8_000

#: Observations kept per token. Was 5,000, which is ~9.7 MB for a single
#: hot token -- one mint could take 1% of the cgroup on its own. Nothing
#: reads that far back: the hazard windows are 60s and 300s, and the
#: rug-mechanism classifier needs a first-30s cohort and >=20 priced
#: points. 750 covers every one of those with room to spare and bounds
#: the worst case to ~1.5 MB.
DEFAULT_MAX_OBSERVATIONS_PER_TOKEN = 750

#: Size at which the spill file rotates, keeping one previous generation.
DEFAULT_MAX_SPILL_BYTES = 512 * 1024 * 1024


class ContinuousRugHazardModel:
    """Combines observed flow, route, holder and liquidity deterioration."""

    def __init__(self, chain_config: ChainConfig, rpc: RPCManager, genealogy: GenealogyGraph,
                 wallet_intel: WalletIntelligenceEngine, adversarial: AdversarialAdaptationDetector,
                 max_tracked_tokens: int = DEFAULT_MAX_TRACKED_TOKENS,
                 max_observations_per_token: int = DEFAULT_MAX_OBSERVATIONS_PER_TOKEN,
                 spill_path: Optional[Path] = None,
                 max_spill_bytes: int = DEFAULT_MAX_SPILL_BYTES):
        self.chain_config = chain_config
        self.rpc = rpc
        self.genealogy = genealogy
        self.wallet_intel = wallet_intel
        self.adversarial = adversarial
        self.hazard_states: Dict[str, HazardState] = {}
        self.max_observations_per_token = max(50, int(max_observations_per_token))
        self.observations: Dict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_observations_per_token))
        self.token_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_tracked_tokens = max(100, int(max_tracked_tokens))
        #: Where evicted token history goes. Without it the memory bound
        #: would be a data-loss bound too; with it the desk observes across
        #: millions of tokens cumulatively while holding thousands.
        self.spill_path = Path(spill_path) if spill_path else None
        self.tokens_spilled = 0
        self.spill_failures = 0
        #: Rotate at this size, keeping one previous generation. 512 MB of
        #: live file plus 512 MB rotated is ~1 GB worst case against the
        #: 9.8 GB free measured on this box -- bounded, and small next to
        #: the disk-full alert the watchdog already raises at 90%.
        self.max_spill_bytes = max(1_000_000, int(max_spill_bytes))
        self.spill_rotations = 0
        #: When each token was last written to, so eviction can pick the
        #: stalest rather than an arbitrary one. Kept separately because a
        #: token can be registered without ever being observed, and the
        #: observation deque cannot date a token it never received.
        self._last_touched: Dict[str, float] = {}
        self.tokens_evicted = 0
        self.is_trained = False
        self.data_status = "DATA_BLOCKED"
        self.data_status_detail = "no versioned chronological hazard training artifact"
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._historical_models: Dict[str, Any] = {}
        self._historical_calibrators: Dict[str, Any] = {}
        self.trigger_weights = {
            HazardTrigger.CREATOR_TRANSFER: 0.38, HazardTrigger.INSIDER_SELL: 0.38,
            HazardTrigger.SMART_WALLET_EXIT: 0.24, HazardTrigger.LIQUIDITY_WITHDRAWAL: 0.45,
            HazardTrigger.CONCENTRATION_CHANGE: 0.20, HazardTrigger.BUY_DECELERATION: 0.15,
            HazardTrigger.SELL_ACCELERATION: 0.22, HazardTrigger.VOLUME_COLLAPSE: 0.20,
            HazardTrigger.ROUTE_DEGRADATION: 0.45, HazardTrigger.SOCIAL_VELOCITY_COLLAPSE: 0.08,
            HazardTrigger.FAILED_MIGRATION: 0.30, HazardTrigger.DEV_WALLET_ACTIVATION: 0.25,
            HazardTrigger.BUNDLE_DETECTION: 0.25,
        }

    async def start(self):
        self._running = True
        await self._load_historical_model()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                return

    async def _load_historical_model(self):
        model_dir = Path(os.getenv("MODEL_DIR", "models"))
        candidates = sorted(model_dir.glob("rug-hazard-*.joblib"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            self.data_status = "DATA_BLOCKED"
            self.data_status_detail = "no versioned chronological hazard training artifact"
            return
        try:
            artifact = joblib.load(candidates[0])
            if artifact.get("schema_version") != 1:
                raise ValueError("unsupported hazard artifact schema")
            if tuple(artifact.get("feature_names", ())) != HAZARD_FEATURE_NAMES:
                raise ValueError("hazard feature schema mismatch")
            if artifact.get("validation", {}).get("status") != "PASSED":
                raise ValueError("hazard artifact lacks passed chronological validation")
            self._historical_models = dict(artifact.get("models") or {})
            self._historical_calibrators = dict(artifact.get("calibrators") or {})
            if not {"rug_30s", "rug_5m"}.issubset(self._historical_models):
                raise ValueError("hazard artifact is missing required heads")
            self.is_trained = True
            self.data_status = "OK"
            self.data_status_detail = candidates[0].name
        except Exception as exc:
            self.is_trained = False
            self.data_status = "DATA_BLOCKED"
            self.data_status_detail = f"invalid chronological hazard artifact: {exc}"

    def register_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> HazardState:
        state = self.hazard_states.setdefault(token, HazardState(token=token, chain=self.chain_config.name))
        if metadata:
            self.token_metadata[token] = {**self.token_metadata.get(token, {}), **metadata}
        # Stamped only on FIRST registration. Deliberately not refreshed here:
        # _monitor_loop calls _compute_hazard -> register_token for every
        # tracked token every 2 seconds, so refreshing on registration meant
        # "last touched" tracked when the desk last LOOKED rather than when
        # data last ARRIVED. Nothing could ever be seen as quiet, which
        # silently disabled death classification entirely -- 2,714 launches
        # awaiting a verdict produced exactly one, measured 2026-08-29.
        self._last_touched.setdefault(token, time.time())
        return state

    def _rotate_spill_if_large(self) -> bool:
        """Keep the lake to one live file plus one rotation. Returns True if
        a rotation happened.

        An append-only spill solves a memory bound by creating a disk one:
        this box was at 73% disk with 9.8 GB free when the spill was added,
        so an unbounded JSONL would eventually trade OOM kills for a full
        disk -- which is worse, because a full disk stops the evidence
        ledger and the episode writer too, not just this model.

        Two generations, not N: the older file is what a trainer reads for
        history, and keeping more would be storing data nothing consumes.
        """
        if self.spill_path is None:
            return False
        try:
            if self.spill_path.stat().st_size < self.max_spill_bytes:
                return False
        except OSError:
            return False
        previous = self.spill_path.with_suffix(self.spill_path.suffix + ".1")
        try:
            self.spill_path.replace(previous)
            self.spill_rotations += 1
            logger.info("hazard spill rotated at %.0f MB -> %s",
                        self.max_spill_bytes / 1e6, previous.name)
            return True
        except OSError as exc:
            logger.warning("hazard spill rotation failed: %s", exc)
            return False

    def _spill(self, token: str) -> bool:
        """Append one token's observed history to the lake before it leaves.

        Eviction is an unload, not a delete: the desk is meant to accumulate
        evidence across every token it has ever seen, and only the RESIDENT
        set is what the box can bound. A token with nothing observed is not
        written -- an empty row is not evidence.
        """
        if self.spill_path is None:
            return False
        rows = self.observations.get(token)
        if not rows:
            return False
        try:
            self.spill_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_spill_if_large()
            record = {
                "token": token,
                "metadata": self.token_metadata.get(token, {}),
                "last_touched": self._last_touched.get(token),
                "observations": list(rows),
            }
            with self.spill_path.open("a") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
            self.tokens_spilled += 1
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.spill_failures += 1
            logger.warning("hazard spill failed for %s: %s: %s",
                           token, type(exc).__name__, exc)
            return False

    def last_touched(self, token: str) -> Optional[float]:
        """When this token was last written to, in O(1).

        Callers asking "has this gone quiet" would otherwise scan the whole
        observation deque for a max timestamp -- up to 750 entries per
        token, across thousands of tokens, on a box already stalled ~21% of
        the time on CPU. The model tracks this for eviction anyway.
        """
        return self._last_touched.get(token)

    def prune(self, protected: Optional[Set[str]] = None,
              on_evict: Optional[Callable[[str], None]] = None) -> int:
        """Drop the stalest tokens once past the cap. Returns how many went.

        Protection is the caller's to define, because this module cannot
        know which tokens carry an open position or are mid-decision.
        Anything protected stays however stale it is: evicting a token the
        desk still holds would silently blind the exit policy, and a
        blinded exit is a far worse failure than using more memory.
        """
        if len(self._last_touched) <= self.max_tracked_tokens:
            return 0
        keep = set(protected or ())
        candidates = sorted(
            ((stamp, token) for token, stamp in self._last_touched.items()
             if token not in keep),
            key=lambda row: row[0])
        # Never evict below the cap counting protected tokens, but never
        # refuse to evict either: if protection alone exceeds the cap the
        # desk is legitimately watching that much and the bound yields.
        excess = len(self._last_touched) - self.max_tracked_tokens
        evicted = 0
        for _stamp, token in candidates[:max(0, excess)]:
            # Last call on this token's observations. A token evicted while
            # still awaiting a death verdict never gets one: the sweep can
            # only reach RESIDENT tokens, so eviction silently ended the
            # question. Measured 2026-08-29 across 71,748 spilled tokens,
            # 107 of them satisfied the death criteria and none had been
            # classified. Called before the spill so a callback that raises
            # cannot cost us the archive copy too.
            if on_evict is not None:
                try:
                    on_evict(token)
                except Exception as exc:
                    logger.warning("evict hook failed for %s: %s: %s",
                                   token, type(exc).__name__, exc)
            # Spill BEFORE dropping. If the write fails the history is still
            # lost, but the failure is counted rather than silent -- an
            # invisible hole in the research lake is what makes a corpus
            # quietly wrong.
            self._spill(token)
            self.hazard_states.pop(token, None)
            self.observations.pop(token, None)
            self.token_metadata.pop(token, None)
            self._last_touched.pop(token, None)
            evicted += 1
        self.tokens_evicted += evicted
        if evicted:
            logger.info("hazard model evicted %d stale tokens (tracking %d, cap %d)",
                        evicted, len(self._last_touched), self.max_tracked_tokens)
        return evicted

    def record_observation(self, token: str, observation: Dict[str, Any]):
        if not token or not isinstance(observation, dict):
            return
        item = dict(observation)
        item.setdefault("timestamp", time.time())
        item.setdefault("type", "unknown")
        self.register_token(token)
        self.observations[token].append(item)
        # THIS is what freshness means: a real observation arrived. Stamped
        # from the wall clock rather than item["timestamp"], because a
        # backfilled or mis-stamped payload carrying an old time would mark a
        # token we are actively receiving as stale.
        self._last_touched[token] = time.time()

    async def _monitor_loop(self):
        while self._running:
            try:
                for token in list(self.hazard_states):
                    await self._compute_hazard(token)
            except Exception as exc:
                logger.error("Hazard monitor error: %s", exc)
            await asyncio.sleep(2)

    async def _compute_hazard(self, token: str) -> HazardState:
        state = self.register_token(token)
        observations = list(self.observations.get(token, ()))
        if not observations:
            state.data_status = "DATA_BLOCKED"
            state.blocked_reason = "no_market_observations"
            state.last_update = time.time()
            return state
        signals = await self._collect_hazard_signals(token)
        state.signals = signals[-50:]
        survival = 1.0
        for signal in signals:
            weight = self.trigger_weights.get(signal.trigger, 0.10)
            adaptive = self.adversarial.get_adaptive_weight(signal.trigger.value, 1.0)
            component = float(np.clip(signal.strength * signal.confidence * weight * adaptive, 0, 0.95))
            survival *= 1.0 - component
        state.current_hazard = float(np.clip(1.0 - survival, 0, 1))
        state.hazard_30s = self._project_hazard(state.current_hazard, 30)
        state.hazard_5m = self._project_hazard(state.current_hazard, 300)
        if self.is_trained:
            features = self.feature_vector_from_observations(observations, time.time()).reshape(1, -1)
            for key, attr in (("rug_30s", "hazard_30s"), ("rug_5m", "hazard_5m")):
                raw = self._historical_models[key].predict_proba(features)[:, 1]
                calibrator = self._historical_calibrators.get(key)
                probability = float(calibrator.predict(raw)[0] if calibrator else raw[0])
                setattr(state, attr, max(getattr(state, attr), float(np.clip(probability, 0, 1))))
        state.hazard_30m = self._project_hazard(state.current_hazard, 1_800)
        state.exit_urgency = self._get_urgency(state.hazard_30s, state.hazard_5m)
        state.exit_recommended = state.exit_urgency in {"HIGH", "CRITICAL"}
        state.data_status = "OK"
        state.blocked_reason = ""
        state.last_update = time.time()
        return state

    @staticmethod
    def _project_hazard(current: float, seconds: int) -> float:
        persistence = 1.0 - np.exp(-seconds / 600.0)
        return float(np.clip(current + (1.0 - current) * current * persistence, 0, 1))

    @staticmethod
    def _get_urgency(hazard_30s: float, hazard_5m: float) -> str:
        if hazard_30s >= 0.80 or hazard_5m >= 0.90:
            return "CRITICAL"
        if hazard_30s >= 0.60 or hazard_5m >= 0.75:
            return "HIGH"
        if hazard_30s >= 0.40 or hazard_5m >= 0.50:
            return "MEDIUM"
        if hazard_30s >= 0.20 or hazard_5m >= 0.30:
            return "LOW"
        return "NONE"

    async def _collect_hazard_signals(self, token: str) -> List[HazardSignal]:
        observations = list(self.observations.get(token, ()))
        if not observations:
            return []
        now = time.time()
        recent = [item for item in observations if now - float(item.get("timestamp", now)) <= 60]
        prior = [item for item in observations if 60 < now - float(item.get("timestamp", now)) <= 300]
        signals: List[HazardSignal] = []
        mapping = {"creator_transfer": HazardTrigger.CREATOR_TRANSFER,
                   "dev_wallet_activation": HazardTrigger.DEV_WALLET_ACTIVATION,
                   "failed_migration": HazardTrigger.FAILED_MIGRATION, "bundle": HazardTrigger.BUNDLE_DETECTION}
        for item in recent:
            if item.get("type") in mapping:
                signals.append(self._signal(mapping[item["type"]], item.get("strength", 1), item.get("confidence", 0.8), item))

        trades_recent = [item for item in recent if item.get("type") == "trade"]
        trades_prior = [item for item in prior if item.get("type") == "trade"]
        buy_recent = sum(self._notional(item) for item in trades_recent if item.get("side") == "buy")
        sell_recent = sum(self._notional(item) for item in trades_recent if item.get("side") == "sell")
        buy_prior = sum(self._notional(item) for item in trades_prior if item.get("side") == "buy") / 4.0
        sell_prior = sum(self._notional(item) for item in trades_prior if item.get("side") == "sell") / 4.0
        total_recent, total_prior = buy_recent + sell_recent, buy_prior + sell_prior
        if total_recent > 0 and sell_recent / total_recent >= 0.65:
            signals.append(self._signal(HazardTrigger.SELL_ACCELERATION, sell_recent / total_recent, 0.85, {"sell_share": sell_recent / total_recent}))
        elif len(trades_recent) >= 4:
            # Pump instruction arguments expose a limit, not the actual quote
            # paid. Until balance-delta enrichment arrives, trade counts are a
            # lower-confidence fallback and are never presented as notional.
            sell_count = sum(item.get("side") == "sell" for item in trades_recent)
            sell_share = sell_count / len(trades_recent)
            if sell_share >= 0.75:
                signals.append(self._signal(
                    HazardTrigger.SELL_ACCELERATION, sell_share, 0.55,
                    {"sell_share_by_count": sell_share, "measurement": "count_fallback"},
                ))
        if buy_prior > 0 and buy_recent < buy_prior * 0.35:
            signals.append(self._signal(HazardTrigger.BUY_DECELERATION, 1 - buy_recent / buy_prior, 0.75, {}))
        if total_prior > 0 and total_recent < total_prior * 0.25:
            signals.append(self._signal(HazardTrigger.VOLUME_COLLAPSE, 1 - total_recent / total_prior, 0.70, {}))

        for item in self._latest_by_type(observations).values():
            event_type = item.get("type")
            if event_type == "liquidity" and float(item.get("change_pct", 0)) <= -0.15:
                signals.append(self._signal(HazardTrigger.LIQUIDITY_WITHDRAWAL, min(abs(float(item["change_pct"])), 1), 0.95, item))
            elif event_type == "concentration" and float(item.get("top10_change_pct", 0)) >= 0.10:
                signals.append(self._signal(HazardTrigger.CONCENTRATION_CHANGE, min(float(item["top10_change_pct"]) * 3, 1), 0.80, item))
            elif event_type == "route":
                feasible, impact = item.get("feasible"), float(item.get("price_impact_pct", 0) or 0)
                if feasible is False or impact >= 0.15:
                    signals.append(self._signal(HazardTrigger.ROUTE_DEGRADATION, 1 if feasible is False else min(impact * 3, 1), 0.98, item))
            elif event_type == "social" and float(item.get("velocity_change_pct", 0)) <= -0.70:
                signals.append(self._signal(HazardTrigger.SOCIAL_VELOCITY_COLLAPSE, abs(float(item["velocity_change_pct"])), 0.50, item))

        smart_wallets = {score.wallet: score for score in self.wallet_intel.get_top_wallets(limit=50)}
        for item in trades_recent:
            if item.get("side") != "sell":
                continue
            score = smart_wallets.get(item.get("wallet"))
            if score:
                strength = min(self._notional(item) / 1_000, 1) if self._notional(item) else 0.25
                signals.append(self._signal(HazardTrigger.SMART_WALLET_EXIT, strength, score.overall_score, item))
            if item.get("is_insider"):
                strength = min(self._notional(item) / 1_000, 1) if self._notional(item) else 0.25
                signals.append(self._signal(HazardTrigger.INSIDER_SELL, strength, 0.90, item))
        return signals

    @staticmethod
    def _notional(item: Dict[str, Any]) -> float:
        if item.get("notional_usd") is not None:
            return max(0.0, float(item["notional_usd"]))
        return max(0.0, float(item.get("amount", 0) or 0) * float(item.get("price", 0) or 0))

    @staticmethod
    def feature_vector_from_observations(observations: List[Dict[str, Any]], as_of: float) -> np.ndarray:
        """Build a PIT-only vector shared by offline training and live inference."""
        eligible = [item for item in observations if float(item.get("timestamp", 0) or 0) <= as_of]
        recent = [item for item in eligible if 0 <= as_of - float(item.get("timestamp", 0) or 0) <= 60]
        prior = [item for item in eligible if 60 < as_of - float(item.get("timestamp", 0) or 0) <= 300]
        recent_trades = [item for item in recent if item.get("type") == "trade"]
        prior_trades = [item for item in prior if item.get("type") == "trade"]

        def volumes(items: List[Dict[str, Any]]) -> Tuple[float, float]:
            buys = sum(ContinuousRugHazardModel._notional(item) for item in items if item.get("side") == "buy")
            sells = sum(ContinuousRugHazardModel._notional(item) for item in items if item.get("side") == "sell")
            if buys + sells <= 0 and items:
                buys = float(sum(item.get("side") == "buy" for item in items))
                sells = float(sum(item.get("side") == "sell" for item in items))
            return buys, sells

        buy_recent, sell_recent = volumes(recent_trades)
        buy_prior, sell_prior = volumes(prior_trades)
        prior_scale = 4.0
        buy_prior /= prior_scale
        sell_prior /= prior_scale
        recent_total, prior_total = buy_recent + sell_recent, buy_prior + sell_prior
        sell_share = sell_recent / recent_total if recent_total else 0.0
        buy_deceleration = max(0.0, 1 - buy_recent / buy_prior) if buy_prior else 0.0
        volume_collapse = max(0.0, 1 - recent_total / prior_total) if prior_total else 0.0

        price_items = sorted(
            (item for item in eligible if float(item.get("price_multiple", item.get("price_usd", 0)) or 0) > 0),
            key=lambda item: float(item.get("timestamp", 0) or 0),
        )
        drawdown = 0.0
        if price_items:
            if all(float(item.get("price_multiple", 0) or 0) > 0 for item in price_items):
                values = [float(item["price_multiple"]) for item in price_items]
            else:
                entry = float(price_items[0].get("price_usd", 0) or 0)
                values = [float(item.get("price_usd", 0) or 0) / max(entry, 1e-12) for item in price_items]
            drawdown = 1 - values[-1] / max(max(values), 1e-12)

        latest = ContinuousRugHazardModel._latest_by_type(eligible)
        route = latest.get("route", {})
        impact = float(route.get("price_impact_pct", 0) or 0)
        route_degradation = 1.0 if route.get("feasible") is False else min(1.0, impact / 0.15)
        liquidity = latest.get("liquidity", {})
        liquidity_withdrawal = min(1.0, max(0.0, -float(liquidity.get("change_pct", 0) or 0)))
        concentration = latest.get("concentration", {})
        concentration_increase = min(1.0, max(0.0, float(concentration.get("top10_change_pct", 0) or 0) * 3))
        explicit_types = {"creator_transfer", "dev_wallet_activation", "failed_migration", "bundle"}
        explicit = min(1.0, sum(str(item.get("type")) in explicit_types for item in recent) / 2)
        return np.asarray([
            sell_share, buy_deceleration, volume_collapse, max(0.0, min(1.0, drawdown)),
            route_degradation, liquidity_withdrawal, concentration_increase, explicit,
        ], dtype=float)

    @staticmethod
    def _latest_by_type(observations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for item in observations:
            key = str(item.get("type", "unknown"))
            if key not in latest or float(item.get("timestamp", 0)) >= float(latest[key].get("timestamp", 0)):
                latest[key] = item
        return latest

    @staticmethod
    def _signal(trigger: HazardTrigger, strength: Any, confidence: Any, metadata: Dict[str, Any]) -> HazardSignal:
        return HazardSignal(trigger, float(np.clip(float(strength or 0), 0, 1)),
                            float(np.clip(float(confidence or 0), 0, 1)),
                            float(metadata.get("timestamp", time.time())), dict(metadata))

    def get_hazard(self, token: str) -> Optional[HazardState]:
        return self.hazard_states.get(token)

    def should_exit(self, token: str, position: Dict[str, Any]) -> Tuple[bool, str, float]:
        state = self.hazard_states.get(token)
        if not state or state.data_status != "OK":
            return False, "DATA_BLOCKED", 0.0
        if state.exit_urgency == "CRITICAL":
            return True, "CRITICAL", 1.0
        if state.exit_urgency == "HIGH":
            return True, "HIGH", 0.50
        return False, "hold", 0.0

    def get_stats(self) -> Dict[str, Any]:
        states = list(self.hazard_states.values())
        return {"tracked_tokens": len(states), "critical": sum(s.exit_urgency == "CRITICAL" for s in states),
                "high": sum(s.exit_urgency == "HIGH" for s in states),
                "data_blocked_tokens": sum(s.data_status == "DATA_BLOCKED" for s in states),
                "observations": sum(len(items) for items in self.observations.values()),
                "model_trained": self.is_trained, "model_status": self.data_status,
                "model_status_detail": self.data_status_detail}
