"""Memecoin research and execution desk.

The shipped runtime defaults to dry-run. Live transaction submission requires
both an explicit ``--live`` launch and the execution engine's independent
``ALLOW_LIVE_TRADING=yes-i-understand`` acknowledgement.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from aiohttp import web
from solders.keypair import Keypair

from src.chains.provider_credentials import normalize_provider_environment
from src.chains.rpc_manager import ChainRegistry, RPCManager
from src.chains.yellowstone_grpc import (
    NATIVE_FASTPATH_STATUS, PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream, YellowstoneClient,
    create_combined_subscription,
)
from src.detection.rug_detector import RugDetector
from src.detection.token_detector import DetectionSource, TokenCandidate, TokenDetectionEngine
from src.execution.jupiter_jito import (
    ExecutionEngine,
    JupiterClient,
    JitoClient,
    PriorityFeeOptimizer,
    SolanaTransactionBuilder,
    USDC_MINT,
)
from src.research.dataset_builder import PointInTimeDatasetBuilder
from src.research.feature_engine import build_features
from src.research.global_research_miner import GlobalResearchMiner
from src.research.attribution import EdgeDecayMonitor
from src.runtime.hot_state import HotState, HotStateBudget
from src.strategies.action_value import (
    Action as ActionValue, ActionValuePolicy, Decision as ActionDecision,
    PositionState as ActionState,
)
from src.strategies.actor_graph import (
    Entry, IndependenceReport, WalletIndependence,
)
from src.strategies.champion_challenger import ChampionChallengerFramework, HypothesisSpec, TrialResult
from src.strategies.exit_policy import ExitPolicy, evaluate_exit, load_latest_exit_policy
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.information_graph import (
    AdversarialAdaptationDetector,
    CounterfactualExecutionLab,
    InformationLeadGraph,
    LeadEventType,
)
from src.strategies.multihead_predictor import ElogwEngine, MultiHeadPredictor, PredictionFeatures
from src.chains.pump_curve import BondingCurveState, quote_buy, quote_sell
from src.execution.tradeability import curve_tradeability, exit_capacity_ratio
from src.strategies.distribution import DistributionDetector
from src.strategies.mega_event import MegaEventReserve
from src.strategies.escape import (
    EscapeEstimate, HazardMechanism, escape_probability,
    hazard_curve_from_probabilities,
)
from src.strategies.monster import (
    MonsterEvidence, MonsterState, MonsterStateMachine, hold_versus_exit,
)
from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator
from src.strategies.prelaunch_intent import PrelaunchIntentModel
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.social_intelligence import SocialIntelligenceEngine
from src.strategies.wallet_intelligence import WalletIntelligenceEngine

logger = logging.getLogger(__name__)

WSOL_MINT = "So11111111111111111111111111111111111111112"
MODEL_HYPOTHESIS_ID = "production_multihead_v1"


# Rejections that mean "capital is committed elsewhere", not "this token is
# bad". Only these may be revisited by a cross-sectional contest -- a safety
# rejection, a rug-risk rejection or the daily-loss kill switch never can be,
# or the allocator would become a way to argue past the risk limits.
CAPACITY_REJECTIONS = frozenset({
    "max_concurrent_positions", "total_exposure_limit", "portfolio_risk_limit",
})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


class MemecoinQuantDesk:
    def __init__(self, config_path: str = "config/chains.yaml", *, dry_run_override: Optional[bool] = None,
                 offline: bool = False):
        self.config_path = config_path
        self.dry_run_override = dry_run_override
        self.offline = offline
        self.config: Dict[str, Any] = {}
        self.global_config: Dict[str, Any] = {}
        self.dry_run = True
        self.keypair: Optional[Keypair] = None
        self.chain_registry: Optional[ChainRegistry] = None
        self.solana_config = None
        self.solana_rpc: Optional[RPCManager] = None
        self.yellowstone: Optional[YellowstoneClient] = None
        self.pump_monitor: Optional[PumpFunMonitor] = None
        self.pump_swap_monitor: Optional[PumpSwapMonitor] = None
        self.raydium_monitor: Optional[RaydiumMonitor] = None
        self.rpc_program_stream: Optional[SolanaRpcProgramStream] = None
        self.detection_engine: Optional[TokenDetectionEngine] = None
        self.rug_detector: Optional[RugDetector] = None
        self.genealogy: Optional[GenealogyGraph] = None
        self.wallet_intel: Optional[WalletIntelligenceEngine] = None
        self.social_intel: Optional[SocialIntelligenceEngine] = None
        self.prelaunch: Optional[PrelaunchIntentModel] = None
        self.info_graph: Optional[InformationLeadGraph] = None
        self.counterfactual_lab: Optional[CounterfactualExecutionLab] = None
        self.adversarial: Optional[AdversarialAdaptationDetector] = None
        self.rug_hazard: Optional[ContinuousRugHazardModel] = None
        self.public_coordination: Optional[PublicCoordinationMiner] = None
        self.predictor: Optional[MultiHeadPredictor] = None
        self.exit_policy: ExitPolicy = ExitPolicy.default()
        self.exit_policy_status = "DATA_BLOCKED"
        self.exit_policy_detail = "not initialized"
        self.elogw_engine: Optional[ElogwEngine] = None
        self.champion_challenger: Optional[ChampionChallengerFramework] = None
        self.jupiter: Optional[JupiterClient] = None
        self.jito: Optional[JitoClient] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.fee_optimizer: Optional[PriorityFeeOptimizer] = None
        self.dataset_builder: Optional[PointInTimeDatasetBuilder] = None
        self.global_research: Optional[GlobalResearchMiner] = None
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._market_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._candidate_pipelines: Dict[str, asyncio.Task] = {}
        self._candidate_semaphore: Optional[asyncio.Semaphore] = None
        self._web_runner: Optional[web.AppRunner] = None
        self.start_time = time.time()
        self.last_intelligence_update = 0.0
        self.sol_price_usd = 0.0
        self.wallet_equity_usd = 0.0
        self.equity_status = "DATA_BLOCKED"
        self.trade_count = 0
        self.successful_exits = 0
        self.total_pnl = 0.0
        self._market_observed_at: Dict[str, float] = {}
        self._market_observation_cohort: set[str] = set()
        self._market_entry_price: Dict[str, float] = {}
        self._curve_entry_price: Dict[str, float] = {}
        self._latest_stream_mark: Dict[str, Dict[str, float]] = {}
        # Latest bonding-curve reserves decoded straight off the trade
        # stream, so exit capacity is answerable locally in the window a
        # decision actually has, with no RPC round trip.
        self._latest_curve_state: Dict[str, Any] = {}
        # Partial-exit PnL accumulated per token, banked into the final
        # outcome row when the position actually closes.
        self._closed_pnl: Dict[str, float] = {}
        self._market_cursor = 0
        self._model_artifact_mtime = 0.0

    async def initialize(self):
        normalize_provider_environment(os.environ)
        with open(self.config_path, encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.global_config = self.config.get("global", {})
        self._candidate_semaphore = asyncio.Semaphore(
            int(self.global_config.get("max_candidate_concurrency", 8))
        )
        configured_dry_run = bool(self.global_config.get("dry_run", True))
        self.dry_run = configured_dry_run if self.dry_run_override is None else bool(self.dry_run_override)
        await self._setup_keys()
        await self._setup_chains()
        await self._setup_yellowstone()
        await self._setup_intelligence()
        await self._setup_prediction()
        await self._setup_execution()
        await self._setup_detection_and_risk()
        await self._setup_research()
        await self._refresh_portfolio_state()
        logger.info("Desk initialized: mode=%s live_submission_locked=%s", "DRY_RUN" if self.dry_run else "LIVE",
                    os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand")

    async def _setup_keys(self):
        encoded = os.getenv("SOLANA_PRIVATE_KEY", "").strip()
        if not encoded:
            if not self.dry_run:
                raise RuntimeError("SOLANA_PRIVATE_KEY is required for live mode")
            self.keypair = Keypair()
            logger.warning("Using an ephemeral paper wallet; no private key was loaded")
            return
        try:
            if encoded.startswith("["):
                raw = bytes(json.loads(encoded))
                self.keypair = Keypair.from_bytes(raw)
            else:
                try:
                    self.keypair = Keypair.from_base58_string(encoded)
                except ValueError:
                    raw = base64.b64decode(encoded, validate=True)
                    self.keypair = Keypair.from_bytes(raw) if len(raw) == 64 else Keypair.from_seed(raw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("SOLANA_PRIVATE_KEY has an unsupported or invalid encoding") from exc
        logger.info("Wallet public key: %s", self.keypair.pubkey())

    async def _setup_chains(self):
        self.chain_registry = ChainRegistry(self.config_path)
        enabled = self.global_config.get("enabled_chains", ["solana"])
        if "solana" not in enabled:
            raise RuntimeError("Solana must be enabled for this build")
        if self.offline:
            chain = self.chain_registry.get_chain("solana")
            self.chain_registry.rpc_managers["solana"] = RPCManager(chain)
        else:
            await self.chain_registry.start_all(enabled)
        self.solana_config = self.chain_registry.get_chain("solana")
        self.solana_rpc = self.chain_registry.get_rpc("solana")
        if not self.solana_config or not self.solana_rpc:
            raise RuntimeError("Solana chain configuration is unavailable")

    async def _setup_yellowstone(self):
        endpoint = os.getenv("YELLOWSTONE_GRPC_URL", "").strip()
        self.yellowstone = YellowstoneClient(
            endpoint or "https://yellowstone.invalid",
            os.getenv("YELLOWSTONE_GRPC_TOKEN", os.getenv("HELIUS_API_KEY", "")),
        )
        connected = False if self.offline or not endpoint else await self.yellowstone.connect()
        if self.offline:
            self.yellowstone.status = "DATA_BLOCKED"
            self.yellowstone.status_detail = "offline smoke-test mode"
        elif not endpoint:
            self.yellowstone.status = "DATA_BLOCKED"
            self.yellowstone.status_detail = "YELLOWSTONE_GRPC_URL is missing; RPC fallback enabled"
        if connected:
            await self.yellowstone.subscribe(create_combined_subscription())
            self.pump_monitor = PumpFunMonitor(self.yellowstone, self._on_pump_event)
            self.pump_swap_monitor = PumpSwapMonitor(self.yellowstone, self._on_pump_event)
            self.raydium_monitor = RaydiumMonitor(self.yellowstone, self._on_raydium_event)
        elif not self.offline:
            self.rpc_program_stream = SolanaRpcProgramStream(
                self.solana_rpc, [PumpFunMonitor.PUMP_FUN_PROGRAM, PumpSwapMonitor.PUMP_AMM_PROGRAM,
                                  RaydiumMonitor.RAYDIUM_AMM_V4, RaydiumMonitor.RAYDIUM_CPMM,
                                  RaydiumMonitor.RAYDIUM_CLMM, RaydiumMonitor.METEORA_DLMM,
                                  RaydiumMonitor.METEORA_DYNAMIC_AMM, RaydiumMonitor.ORCA_WHIRLPOOL]
            )
            self.pump_monitor = PumpFunMonitor(self.rpc_program_stream, self._on_pump_event)
            self.pump_swap_monitor = PumpSwapMonitor(self.rpc_program_stream, self._on_pump_event)
            self.raydium_monitor = RaydiumMonitor(self.rpc_program_stream, self._on_raydium_event)

    async def _setup_intelligence(self):
        helius = os.getenv("HELIUS_API_KEY", "")
        self.genealogy = GenealogyGraph(self.solana_config, self.solana_rpc, helius)
        self.wallet_intel = WalletIntelligenceEngine(self.solana_config, self.solana_rpc, self.genealogy, helius)
        self.social_intel = SocialIntelligenceEngine(
            self.solana_config, self.solana_rpc, self.genealogy, self.wallet_intel,
            {"helius": helius, "x_bearer": os.getenv("X_BEARER_TOKEN", ""),
             "telegram": os.getenv("TELEGRAM_BOT_TOKEN", ""),
             "telegram_api_id": os.getenv("TELEGRAM_API_ID", ""),
             "telegram_api_hash": os.getenv("TELEGRAM_API_HASH", ""),
             "telegram_channels": os.getenv("TELEGRAM_CHANNELS", ""),
             "youtube": os.getenv("YOUTUBE_API_KEY", ""),
             "reddit": os.getenv("REDDIT_CLIENT_ID", ""),
             "reddit_secret": os.getenv("REDDIT_CLIENT_SECRET", "")},
        )
        self.prelaunch = PrelaunchIntentModel(self.solana_config, self.solana_rpc, self.genealogy, self.wallet_intel, helius)
        self.counterfactual_lab = CounterfactualExecutionLab()
        self.adversarial = AdversarialAdaptationDetector()
        for feature, fakeability in {
            "buyer_count": 0.9, "fresh_wallet_volume": 0.8, "social_engagement": 0.85,
            "wallet_history_6m": 0.2, "independent_funding": 0.3, "creator_genealogy": 0.15,
            "narrative_acceleration": 0.4, "real_liquidity": 0.1,
        }.items():
            self.adversarial.set_fakeability(feature, fakeability)
        self.info_graph = InformationLeadGraph(self.solana_config, self.solana_rpc, self.genealogy,
                                               self.wallet_intel, self.social_intel, self.prelaunch)
        self.rug_hazard = ContinuousRugHazardModel(self.solana_config, self.solana_rpc, self.genealogy,
                                                   self.wallet_intel, self.adversarial)
        self.public_coordination = PublicCoordinationMiner(self.genealogy, self.wallet_intel)
        self.social_intel.on_mention(self._on_social_mention)
        if not self.offline:
            for component in (self.genealogy, self.wallet_intel, self.social_intel, self.prelaunch,
                              self.info_graph, self.rug_hazard):
                await component.start()

    async def _setup_prediction(self):
        self.predictor = MultiHeadPredictor()
        self.predictor.initialize_models()
        model_loaded = self.predictor.load_latest()
        trained_policy, policy_report = load_latest_exit_policy(os.getenv("MODEL_DIR", "models"))
        # Without a validated policy the operator's configured hold time still
        # governs; a trained policy has earned the right to set its own.
        self.exit_policy = trained_policy or dataclasses_replace(
            ExitPolicy.default(),
            max_hold_seconds=float(self.global_config.get("max_hold_time_minutes", 60)) * 60,
        )
        self.exit_policy_status = "OK" if trained_policy else "DATA_BLOCKED"
        self.exit_policy_detail = (
            policy_report.get("model_path", "") if trained_policy
            else "no chronologically validated exit policy; using default thresholds"
        )
        # Shadow mode runs a deliberately broader book: many small independent
        # trials capture rare tails without touching the separately gated
        # live-capital limits, which keep their unprefixed values.
        prefix = "shadow_" if self.dry_run else ""
        setting = lambda name, default: self.global_config.get(f"{prefix}{name}", self.global_config.get(name, default))
        self.elogw_engine = ElogwEngine(
            self.predictor,
            max_position_pct=float(setting("max_position_pct", 0.05)),
            max_position_usd=float(setting("max_position_size_usd", 500)),
            max_portfolio_risk=float(setting("max_portfolio_risk", 0.10)),
            max_total_exposure_pct=float(setting("max_total_exposure_pct", 0.30)),
            max_concurrent_positions=int(setting("max_concurrent_positions", 10)),
            max_daily_loss_usd=float(self.global_config.get("max_daily_loss_usd", 1_000)),
            max_daily_loss_pct=(float(self.global_config["max_daily_loss_pct"])
                                if self.global_config.get("max_daily_loss_pct") is not None else None),
            daily_giveback_pct=(float(self.global_config["daily_giveback_pct"])
                                if self.global_config.get("daily_giveback_pct") is not None else None),
            daily_giveback_arm_pct=float(self.global_config.get("daily_giveback_arm_pct", 0.5)),
            max_liquidity_fraction=float(self.global_config.get("max_liquidity_fraction", 0.01)),
            harvest_trigger_ratio=float(setting("harvest_trigger_ratio", 1.5)),
            harvest_slope=float(setting("harvest_slope", 0.5)),
            small_account_mode=bool(setting("small_account_mode", False)),
            small_account_negligible_share=float(setting("small_account_negligible_share", 0.002)),
        )
        self.monster_machine = MonsterStateMachine(
            monster_probability_threshold=float(
                self.global_config.get("monster_probability_threshold", 0.15)),
            candidate_probability_threshold=float(
                self.global_config.get("monster_candidate_threshold", 0.06)),
            degrade_confirmations=int(self.global_config.get("monster_degrade_confirmations", 3)),
            min_degrade_dimensions=int(self.global_config.get("monster_degrade_dimensions", 2)),
            bank_fractions={
                state: float(self.global_config.get(f"monster_bank_{state.value}", default))
                for state, default in MonsterStateMachine.DEFAULT_BANK_FRACTIONS.items()
            },
        )
        self.min_escape_probability = float(
            self.global_config.get("min_escape_probability", 0.05))
        self.mega_event_reserve = MegaEventReserve(
            baseline_fraction=float(self.global_config.get("mega_event_baseline_reserve", 0.0)),
            max_fraction=float(self.global_config.get("mega_event_max_reserve", 0.35)),
            arm_probability=float(self.global_config.get("mega_event_arm_probability", 0.05)),
        )
        # None until an event detector supplies a measured probability. The
        # reserve then holds only its baseline, which defaults to nothing --
        # an unmeasured event must not silently withhold capital every week.
        self.mega_event_probability: Optional[float] = None
        self.mega_event_authenticated: bool = False
        self.distribution_detector = DistributionDetector(
            min_coverage=float(self.global_config.get("distribution_min_coverage", 0.3)))
        self.opportunity_allocator = OpportunityAllocator(
            replacement_cost_pct=float(self.global_config.get("replacement_cost_pct", 0.02)),
            min_displacement_gain_ratio=float(
                self.global_config.get("min_displacement_gain_ratio", 1.5)),
            max_displacements_per_cycle=int(
                self.global_config.get("max_displacements_per_cycle", 2)),
        )
        self.last_slate_report: Dict[str, Any] = {}
        self.mega_event_reserve_state: Dict[str, Any] = {}
        self.action_policy = ActionValuePolicy(
            min_edge=float(self.global_config.get("action_min_edge", 1e-4)),
            max_add_fraction=float(setting("max_position_pct", 0.05)),
        )
        self.wallet_independence = WalletIndependence()
        self.independence_report = IndependenceReport(status="DATA_BLOCKED")
        self._actor_seen: Dict[str, set] = {}
        self._independence_computed_at = 0.0
        self._independence_interval = 300.0
        self.edge_decay = EdgeDecayMonitor(
            min_trades=int(self.global_config.get("edge_decay_min_trades", 30)))
        self._mechanism_growth: Dict[str, List[float]] = {}
        self._attribution_published_at = 0.0
        self._attribution_interval = 900.0
        self.hot_state = HotState(
            HotStateBudget(
                max_active_tokens=int(self.global_config.get("max_active_tokens", 4_000)),
                max_hot_wallets=int(self.global_config.get("max_hot_wallets", 20_000)),
            ),
            archive_root=Path(self.global_config.get("ops_state_dir", "data/state")) / "spool",
        )
        self.champion_challenger = ChampionChallengerFramework(
            state_path=os.getenv("CHAMPION_STATE_PATH", "data/research/champion_state.json")
        )
        if not self.offline:
            await self.champion_challenger.start()
        feature_hash = hashlib.sha256(json.dumps({
            "artifact_version": self.predictor.ARTIFACT_VERSION,
            "features": list(self.predictor.feature_names),
        }, sort_keys=True).encode()).hexdigest()
        hypothesis = HypothesisSpec(
            hypothesis_id=MODEL_HYPOTHESIS_ID, mechanism="calibrated multi-head tail and rug probabilities",
            target="net_elogw", features=list(self.predictor.feature_names), feature_hash=feature_hash,
            model_type="gradient_boosting_calibrated", model_params={}, training_window="expanding_point_in_time",
            threshold=0.0, sizing_rule={"objective": "max_net_elogw", "hard_limits": True},
            exit_rule={"profit_ratchet": True, "rug_hazard": True}, execution_policy={"jupiter": True, "jito": True},
            fakeability={}, cost_model={"slippage": "quote_observed"}, falsifier="chronological OOS net ElogW <= 0",
            kill_thesis="forward decay or calibration failure", source_provenance="local validated model bundle",
            trial_family="production_multihead", created_at=time.time(),
        )
        self.champion_challenger.submit_hypothesis(hypothesis)
        self.champion_challenger.freeze_hypothesis(MODEL_HYPOTHESIS_ID)
        if not model_loaded:
            self.champion_challenger.mark_data_blocked(MODEL_HYPOTHESIS_ID, "no validated persisted multi-head model bundle")
        else:
            self._register_model_validation(self.predictor.validation_report)
            self._model_artifact_mtime = self._latest_model_mtime()

    def _register_model_validation(self, report: Dict[str, Any]):
        if not report or report.get("status") != "PASSED":
            return
        net_elogw = float(report.get("net_elogw_proxy", 0) or 0)
        self.champion_challenger.record_trial_result(TrialResult(
            hypothesis_id=MODEL_HYPOTHESIS_ID, stage="CHRONOLOGICAL_OOS",
            samples=int(report.get("oos_samples", 0) or 0), metrics={"brier_skill": float(report.get("mean_brier_skill", 0) or 0)},
            oos_metrics={"elogw": net_elogw}, portfolio_impact=net_elogw,
            passed=net_elogw > 0, timestamp=float(report.get("created_at", time.time()) or time.time()),
            notes="route-feasible OOS proxy; forward shadow remains mandatory",
        ))

    @staticmethod
    def _latest_model_mtime() -> float:
        if not os.path.isdir("models"):
            return 0.0
        return max((os.path.getmtime(os.path.join("models", name)) for name in os.listdir("models")
                    if name.endswith((".joblib", ".pkl"))), default=0.0)

    async def _setup_execution(self):
        self.jupiter = JupiterClient()
        self.jito = JitoClient()
        builder = SolanaTransactionBuilder(self.solana_rpc, self.keypair)
        self.execution_engine = ExecutionEngine(self.solana_config, self.solana_rpc, self.jupiter, self.jito,
                                                builder, self.counterfactual_lab, dry_run=self.dry_run)
        self.fee_optimizer = PriorityFeeOptimizer()
        if not self.offline:
            await self.execution_engine.start()

    async def _setup_detection_and_risk(self):
        self.rug_detector = RugDetector(self.solana_config, self.solana_rpc, self.jupiter)
        self.detection_engine = TokenDetectionEngine(self.chain_registry)

    async def _setup_research(self):
        self.dataset_builder = PointInTimeDatasetBuilder(
            self.solana_config, self.solana_rpc, self.genealogy, self.wallet_intel, self.social_intel,
            self.prelaunch, self.info_graph, self.rug_hazard, self.champion_challenger,
        )
        self.info_graph.set_outcome_provider(self.dataset_builder.get_outcome)
        if hasattr(self.genealogy, "set_outcome_provider"):
            self.genealogy.set_outcome_provider(self.dataset_builder.get_outcome)
        self.global_research = GlobalResearchMiner(self.champion_challenger)
        if not self.offline:
            await self.dataset_builder.start()
            await self.global_research.start()
            # Event callbacks write to the PIT builder, hazard tracker, and
            # research graphs. Start the stream only after all consumers exist.
            if self.rpc_program_stream:
                await self.rpc_program_stream.start()

    async def start(self):
        if self.offline:
            return
        self._running = True
        await self._setup_health_server()
        self._main_task = asyncio.create_task(self._main_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        self._market_task = asyncio.create_task(self._market_observer_loop())

    async def stop(self):
        self._running = False
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        for task in (self._main_task, self._health_task, self._market_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self._close_health_server()
        for component in (self.global_research, self.dataset_builder, self.execution_engine, self.rug_hazard,
                          self.info_graph, self.prelaunch, self.social_intel, self.wallet_intel, self.genealogy,
                          self.detection_engine, self.rpc_program_stream, self.champion_challenger, self.chain_registry):
            if component and hasattr(component, "stop"):
                try:
                    await component.stop()
                except Exception as exc:
                    logger.error("Error stopping %s: %s", component.__class__.__name__, exc)
        if self.yellowstone:
            await self.yellowstone.close()

    def _spawn_background(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _main_loop(self):
        while self._running:
            try:
                await self._process_new_tokens()
                await self._manage_positions()
                await self._update_intelligence()
            except Exception as exc:
                logger.exception("Main loop error: %s", exc)
            await asyncio.sleep(0.5)

    async def _market_observer_loop(self):
        """Keep research quotes off the latency-sensitive decision loop."""
        while self._running:
            try:
                await self._observe_active_markets()
            except Exception as exc:
                logger.exception("Market observer error: %s", exc)
            await asyncio.sleep(float(self.global_config.get("market_observer_sleep_seconds", 0.25)))

    async def _process_new_tokens(self):
        try:
            candidate = await asyncio.wait_for(self.detection_engine.get_candidate(), timeout=0.05)
        except asyncio.TimeoutError:
            return
        token = candidate.address
        if not token or token in self._candidate_pipelines:
            return
        if len(self._candidate_pipelines) >= int(self.global_config.get("max_candidate_pipelines", 100)):
            logger.warning("Candidate pipeline saturated; preserving DATA_BLOCKED decision for %s", token)
            self._record_blocked_decision(token, "DATA_BLOCKED_candidate_pipeline_saturated", {})
            return
        task = asyncio.create_task(self._candidate_pipeline(candidate))
        self._candidate_pipelines[token] = task
        self._background_tasks.add(task)

        def completed(done: asyncio.Task):
            self._candidate_pipelines.pop(token, None)
            self._background_tasks.discard(done)
            if not done.cancelled():
                exc = done.exception()
                if exc:
                    logger.error(
                        "Candidate pipeline failed for %s", token,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )

        task.add_done_callback(completed)

    async def _candidate_pipeline(self, candidate: TokenCandidate):
        delays = self.global_config.get("candidate_recheck_delays_seconds", [0, 1, 3, 5, 10])
        checkpoints = sorted({max(0.0, float(delay)) for delay in delays})
        if self.predictor is not None and not self.predictor._is_trained:
            # A blocked model has no trading authority. One pass registers the
            # episode/risk state; Yellowstone continues collecting outcomes.
            # Holding thousands of five-pass tasks here adds latency but no evidence.
            checkpoints = [0.0]
        started = time.monotonic()
        for delay in checkpoints:
            await asyncio.sleep(max(0.0, started + delay - time.monotonic()))
            if candidate.address in self.elogw_engine.open_positions:
                return
            async with self._candidate_semaphore:
                await self._evaluate_candidate(candidate)

    async def _evaluate_candidate(self, candidate: TokenCandidate):
        if candidate.chain != "solana" or not candidate.address:
            return
        token = candidate.address
        if token in self.elogw_engine.open_positions:
            return
        self.dataset_builder.start_episode(
            token, candidate.deployer or "", candidate.factory or "", candidate.pair or "",
            candidate.base_token or WSOL_MINT, detected_at=candidate.timestamp,
            prelaunch_context=self._prelaunch_context(candidate.deployer or "", candidate.timestamp),
        )
        risk = await self.rug_detector.analyze(token, candidate.pair, candidate.base_token)
        risk_data = _jsonable(risk)
        self.dataset_builder.record_risk_report(token, risk_data)
        self.rug_hazard.register_token(token, {"deployer": candidate.deployer or "", "pair": candidate.pair or ""})
        self.public_coordination.record_creator(token, candidate.deployer or "")
        # Wallets funded within the same atomic launch transaction as the pool
        # creation -- the only funding evidence available without an extra
        # RPC call. record_funding keys on the *funded* wallet so that N
        # different wallets sharing one funder inside this transaction (a
        # common bundled-sniper-bot pattern) is what actually trips the
        # shared_funder coordination signal.
        for item in candidate.metadata.get("funding_transfers", []):
            self.public_coordination.record_funding(
                token, item.get("to", ""), item.get("from", ""),
                float(item.get("lamports", 0) or 0) / 1e9, candidate.timestamp,
            )
        if risk.risk_level.value in {"high", "critical", "honeypot", "rugged"}:
            self._record_blocked_decision(token, "safety_rejection", risk_data)
            return
        if risk.data_status == "DATA_BLOCKED" and self.global_config.get("reject_data_blocked_safety_checks", True):
            self._record_blocked_decision(token, "DATA_BLOCKED_safety_checks", risk_data)
            return
        if not self.predictor._is_trained:
            self._record_blocked_decision(token, "DATA_BLOCKED_prediction_model", {})
            return
        liquidity = await self._resolve_liquidity(candidate)
        if liquidity <= 0:
            self._record_blocked_decision(token, "DATA_BLOCKED_liquidity", {})
            return
        features = await self._build_prediction_features(candidate, risk, liquidity)
        prediction = self.predictor.predict(features)
        if prediction is None:
            self._record_blocked_decision(token, "DATA_BLOCKED_prediction_model", {})
            return
        if not self.dry_run and not self.champion_challenger.is_live(MODEL_HYPOTHESIS_ID):
            self._record_blocked_decision(token, "champion_not_promoted_for_live_authority", _jsonable(prediction))
            return
        await self._refresh_portfolio_state()
        should_trade, trade_info = self.elogw_engine.should_trade(
            prediction, self.sol_price_usd, liquidity, self.wallet_equity_usd,
        )
        decision = {"should_trade": should_trade, "trade_info": trade_info,
                    "authority": "shadow" if self.dry_run else "champion"}
        if not should_trade and trade_info.get("reason") in CAPACITY_REJECTIONS:
            # The book being full is not evidence this candidate is bad. It is
            # only evidence that capital is currently committed elsewhere, and
            # whether that is the right place for it is a cross-sectional
            # question the per-token hurdle cannot ask.
            should_trade, trade_info = await self._contest_for_capital(
                token, candidate, prediction, liquidity, trade_info)
            decision.update({"should_trade": should_trade, "trade_info": trade_info,
                             "contested_for_capital": True})
        decision_id = self.counterfactual_lab.record_decision(token, _jsonable(prediction), decision)
        if not should_trade:
            return
        result = await self.execution_engine.execute_swap(
            candidate.base_token or WSOL_MINT, token, int(trade_info["position_size_sol"] * 1e9),
            slippage_bps=100,
            priority_fee=self.fee_optimizer.get_optimal_fee(trade_info["position_value_usd"], 0.5),
            jito_tip=self.fee_optimizer.get_jito_tip(trade_info["position_value_usd"], "MEDIUM"),
            use_jito=True, decision_id=decision_id,
        )
        self.dataset_builder.record_execution_attempt(token, _jsonable(result))
        self._record_ops_event("execution_attempts", {
            "token": token, "side": "buy", "success": bool(result.success),
            "status": getattr(result.status, "value", str(result.status)),
            "error": result.error, "simulated": bool(result.simulated),
            "decision_id": decision_id,
        })
        if not result.success:
            return
        intended_cost = float(trade_info["position_value_usd"])
        if result.simulated:
            acquisition_cost = intended_cost
        else:
            native_spent = max(0, -int(result.native_balance_delta_lamports)) / 1e9 * self.sol_price_usd
            token_input = int(result.actual_input_amount) / 1e9 * self.sol_price_usd
            acquisition_cost = native_spent or token_input
            if acquisition_cost <= 0:
                self.dataset_builder.record_execution_attempt(token, {
                    "status": "DATA_BLOCKED_ACCOUNTING",
                    "reason": "landed buy lacks verifiable wallet input delta",
                    "execution": _jsonable(result),
                })
                return
        position = {
            "token": token, "size_tokens": int(result.output_amount), "initial_size_tokens": int(result.output_amount),
            "remaining_cost_usd": acquisition_cost,
            "initial_cost_usd": acquisition_cost,
            "risk_contribution": float(trade_info["risk_contribution"]),
            "initial_risk_contribution": float(trade_info["risk_contribution"]),
            "entry_time": time.time(), "entry_sol": float(trade_info["position_size_sol"]),
            "prediction": _jsonable(prediction), "prediction_object": prediction,
            "prediction_at": time.time(), "prediction_status": "OK",
            "risk_report": risk_data, "trade_info": trade_info,
            "decision_id": decision_id, "paper": self.dry_run, "high_water_multiple": 1.0,
            # Retained so the position can be re-predicted on fresh evidence
            # for marginal-E[log W] scale-in rather than sized once at entry.
            "candidate": candidate, "risk_object": risk, "liquidity_usd": liquidity,
            "ratchet_stages": [],
        }
        self.elogw_engine.update_position(token, position)
        self.trade_count += 1
        logger.info("%s BUY %s %.4f SOL status=%s", "PAPER" if self.dry_run else "LIVE", token,
                    trade_info["position_size_sol"], result.status.value)

    def _record_blocked_decision(self, token: str, reason: str, evidence: Dict[str, Any]):
        # Recorded so the weekly audit can ask what the rejected launches went
        # on to do. A missed monster is invisible unless the rejection was
        # written down next to the outcome.
        self._record_ops_event("trade_outcomes", {
            "token": token, "entered": False, "attempted": False,
            "rejection_reason": reason,
        })
        self.counterfactual_lab.record_decision(token, evidence, {"should_trade": False, "reason": reason})

    def _prelaunch_context(self, deployer: str, detected_at: float) -> Optional[Dict[str, Any]]:
        profile = self.prelaunch.get_entity_profile(deployer) if deployer else None
        if not profile or profile.last_active > detected_at:
            return None
        return {
            "as_of": profile.last_active,
            "deployer_features": {"prior_launches": len(profile.prior_launches),
                                  "prior_success_rate": profile.prior_success_rate,
                                  "prior_rug_rate": profile.prior_rug_rate},
            "wallet_features": {"cluster_id": profile.cluster_id or ""},
            "social_features": {"social_creations": len(profile.social_creations)},
            "entity_graph_features": {"intent_score": profile.intent_score},
        }

    async def _resolve_liquidity(self, candidate: TokenCandidate) -> float:
        explicit = candidate.initial_liquidity_usd or candidate.metadata.get("liquidity_usd")
        if explicit and float(explicit) > 0:
            return float(explicit)
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

    async def _build_prediction_features(self, candidate: TokenCandidate, risk: Any, liquidity: float) -> PredictionFeatures:
        as_of = time.time()
        episode = self.dataset_builder.active_episodes.get(candidate.address)
        episode_meta = {
            "token": candidate.address,
            "chain": candidate.chain,
            "created_at": float(getattr(episode, "created_at", candidate.timestamp or as_of)),
        }

        if episode is not None:
            deployer_features = await self.dataset_builder._capture_deployer_features(episode, as_of)
            wallet_features = await self.dataset_builder._capture_wallet_features(episode, as_of)
            flow_features = await self.dataset_builder._capture_flow_features(episode, as_of)
            graph_features = await self.dataset_builder._capture_entity_graph_features(episode, as_of)
            social_features = await self.dataset_builder._capture_social_features(episode, as_of)
            token_features = await self.dataset_builder._capture_token_features(episode, as_of)
            market_features = await self.dataset_builder._capture_market_features(episode, as_of)
        else:
            # No episode yet: report every group DATA_BLOCKED rather than
            # substituting zeros that would read as real observations.
            blocked = {"status": "DATA_BLOCKED", "reason": "episode_not_started"}
            deployer_features = {"has_profile": False}
            wallet_features = {}
            flow_features = dict(blocked)
            graph_features = dict(blocked)
            token_features = dict(blocked)
            market_features = dict(blocked)
            social_features = self.social_intel.get_token_social_signal(candidate.address, as_of=as_of)

        # The safety report is fresher than the episode snapshot for the
        # fields it owns, so it takes precedence where both are present.
        token_features = {
            **token_features,
            "status": risk.data_status,
            "ownership_renounced": bool(risk.ownership_renounced),
            "can_mint": bool(risk.can_mint),
            "can_freeze": bool(risk.can_freeze),
            "top_10_pct": float(risk.top_10_pct),
            "extension_risk": float(getattr(risk, "extension_risk", 0) or 0),
            "sell_route_feasible": risk.sell_route_feasible,
        }
        liquidity_features = (
            {"status": "OK", "liquidity_usd": liquidity, "liquidity_locked": bool(risk.liquidity_locked)}
            if liquidity > 0 else {"status": "DATA_BLOCKED", "reason": "liquidity_not_observed"}
        )

        snapshot = {
            "timestamp": as_of,
            "deployer_features": deployer_features,
            "wallet_features": wallet_features,
            "flow_features": flow_features,
            "liquidity_features": liquidity_features,
            "social_features": social_features,
            "token_features": token_features,
            "market_features": market_features,
            "entity_graph_features": graph_features,
        }
        return build_features(episode_meta, snapshot)

    async def _manage_positions(self):
        for token, position in list(self.elogw_engine.open_positions.items()):
            should_hazard_exit, urgency, pct = self.rug_hazard.should_exit(token, position)
            if should_hazard_exit:
                await self._execute_exit(token, position, pct, f"rug_hazard_{urgency}")
                continue
            marked = await self._mark_position(token, position)
            if marked is None:
                continue
            multiple, current_value = marked
            position["high_water_multiple"] = max(float(position.get("high_water_multiple", 1)), multiple)
            stages = position.setdefault("ratchet_stages", [])
            # Refresh before the exit decision, not after it. The continuation
            # probability decides whether a runner is held through a drawdown;
            # answering it from the entry-time prediction means holding on
            # evidence that has since been contradicted by every trade that
            # arrived after entry -- which is precisely the evidence that
            # distinguishes a 20x from a distribution phase.
            await self._refresh_position_prediction(token, position)
            prediction = position.get("prediction") or {}
            continuation = max(float(prediction.get("p_5x", 0)),
                               float(prediction.get("p_10x", 0)))
            distribution = self._read_distribution(token)
            position["distribution"] = {
                "status": distribution.status, "evidence": distribution.evidence_score,
                "coverage": distribution.coverage,
                "drivers": dict(DistributionDetector.top_contributors(distribution)),
                "probabilities": {str(k): v for k, v in distribution.probabilities.items()},
            }
            monster = self._update_monster_state(token, position, distribution, multiple)
            if monster.action == "emergency_exit":
                await self._execute_exit(token, position, 1.0, f"monster_{monster.reason}")
                continue
            if monster.action == "bank" and monster.bank_fraction > 0:
                await self._execute_exit(token, position, monster.bank_fraction,
                                         f"monster_{monster.reason}")
                continue

            # The action-value policy is asked first, because it is the only
            # component that prices every move against one forward
            # distribution. The threshold policy below remains the fallback for
            # states it cannot price -- an unmeasured capacity, a missing
            # distribution -- rather than being deleted while nothing validated
            # has replaced it.
            action_decision = self._score_actions(token, position, multiple, distribution)
            position["action_value"] = {
                "status": action_decision.status, "action": action_decision.action.value,
                "q": action_decision.q, "detail": action_decision.detail,
            }
            if action_decision.status == "OK" and action_decision.action is not ActionValue.HOLD:
                handled = await self._apply_action(token, position, action_decision, multiple)
                if handled:
                    continue

            decision = evaluate_exit(
                self.exit_policy, multiple, float(position["high_water_multiple"]), continuation,
                set(stages), time.time() - float(position["entry_time"]),
            )
            if decision and self.monster_machine.overrides_ordinary_exit(token):
                # A ratchet banks harder the higher a position goes, which is
                # right for ordinary winners and exactly wrong for the rare one
                # that would carry the account: it sells that one first and
                # hardest. Inside a monster state -- reachable only from a
                # CALIBRATED monster probability -- the ratchet and trail stand
                # down. The hazard exit above is not stood down, and never is.
                logger.info("MONSTER_HOLD %s suppressing %s at %.2fx", token, decision[0], multiple)
                position.setdefault("suppressed_exits", []).append(
                    {"reason": decision[0], "multiple": multiple, "at": time.time()})
                decision = None
            # A calibrated distribution reading reaches the exit through the
            # monster machine above, which is the single path that owns
            # conviction state. Two independent branches deciding the same
            # thing on the same input is how they end up disagreeing.
            if not decision:
                await self._consider_scale_in(token, position, multiple)
                continue
            reason, exit_pct = decision
            stage_name = {"profit_ratchet_cost_recovery": "cost_recovery",
                          "profit_ratchet_5x": "bank_5x",
                          "profit_ratchet_10x": "bank_10x"}.get(reason)
            if stage_name:
                stages.append(stage_name)
            await self._execute_exit(token, position, exit_pct, reason)

    async def _contest_for_capital(
        self, token: str, candidate: TokenCandidate, prediction: Any,
        liquidity: float, trade_info: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Let a rejected-on-capacity candidate contest the weakest position.

        A hurdle applied one token at a time makes capital a first-come
        resource: ten mediocre positions that each cleared it will lock out an
        eleventh that is far better, for as long as they are held. This asks
        the only question that actually matters -- is this the best place for
        the next dollar right now -- and, if it clearly is, proposes closing
        the worst incumbent to fund it.

        It proposes; it does not widen any limit. The freed capital still has
        to clear ``should_trade`` afterwards, so the exposure ceiling, the
        portfolio risk budget and the daily-loss kill switch all still apply.
        """
        elogw, fraction, size_sol = self.elogw_engine.calculate_expected_log_growth(
            prediction, self.sol_price_usd, liquidity)
        if not math.isfinite(elogw) or size_sol <= 0:
            return False, {**trade_info, "contest": "DATA_BLOCKED_CHALLENGER_ELOGW"}

        equity = max(self.wallet_equity_usd, 1e-9)
        challenger = Opportunity(
            token=token, elogw=elogw, capital_usd=size_sol * self.sol_price_usd,
            expected_hold_seconds=(float(prediction.expected_hold_time)
                                   if prediction.expected_hold_time > 0 else None),
            liquidity_usd=liquidity, sleeve=candidate.metadata.get("sleeve", "t0_sniper"),
            # Freed capital rarely matches the optimal size exactly, so the
            # allocator is given the means to re-price this candidate at
            # whatever a displacement would actually fund.
            elogw_at=lambda usd: self.elogw_engine.log_growth_at_fraction(
                prediction, min(usd / equity, self.elogw_engine.exposure_cap(liquidity))),
        )
        incumbents = await self._incumbent_opportunities()
        slate = self.opportunity_allocator.rank(incumbents + [challenger])
        self.last_slate_report = slate.report()
        self._feed_opportunity_quality(slate)

        move = next((item for item in slate.displacements
                     if item.challenger.token == token), None)
        if move is None:
            return False, {**trade_info, "contest": "no_incumbent_worth_displacing",
                           "slate": self.last_slate_report}

        incumbent_position = self.elogw_engine.open_positions.get(move.incumbent.token)
        if incumbent_position is None:
            return False, {**trade_info, "contest": "incumbent_closed_before_displacement"}
        logger.info("DISPLACE %s (score=%.3e) for %s (score=%.3e, cost=$%.2f)",
                    move.incumbent.token, move.incumbent_score, token,
                    move.challenger_score_after_cost, move.round_trip_cost_usd)
        await self._execute_exit(move.incumbent.token, incumbent_position, 1.0,
                                 f"displaced_by_{token}")
        if move.incumbent.token in self.elogw_engine.open_positions:
            # The exit did not actually clear the position (no fill, or an
            # unverified balance change). Capital was never freed, so the
            # challenger must not be funded as though it had been.
            return False, {**trade_info, "contest": "displacement_exit_did_not_fill"}

        await self._refresh_portfolio_state()
        return self.elogw_engine.should_trade(
            prediction, self.sol_price_usd, liquidity, self.wallet_equity_usd)

    async def _incumbent_opportunities(self) -> List[Opportunity]:
        """Score every open position on its FORWARD growth, not its history.

        What a position has already made is sunk. The only thing that competes
        with a new candidate is what it is expected to add from here, per
        dollar still tied up, per second it stays tied up.
        """
        opportunities: List[Opportunity] = []
        for held_token, position in self.elogw_engine.open_positions.items():
            prediction = position.get("prediction_object")
            held_cost = float(position.get("remaining_cost_usd", 0) or 0)
            liquidity = position.get("liquidity_usd")
            if prediction is None or held_cost <= 0:
                continue
            multiple = float(position.get("high_water_multiple", 1.0) or 1.0)
            held_fraction = (held_cost / self.elogw_engine.portfolio_value
                             if self.elogw_engine.portfolio_value > 0 else 0.0)
            forward = self.elogw_engine.marginal_log_growth(prediction, held_fraction, multiple, 0.0)
            hold_time = float(getattr(prediction, "expected_hold_time", 0) or 0)
            elapsed = max(0.0, time.time() - float(position.get("entry_time", time.time())))
            opportunities.append(Opportunity(
                token=held_token,
                elogw=forward,
                capital_usd=held_cost,
                # Remaining expected hold, not total: capital already committed
                # for 200s of a 300s expected hold is 100s from being free.
                expected_hold_seconds=(max(1.0, hold_time - elapsed) if hold_time > 0 else None),
                liquidity_usd=(float(liquidity) if liquidity else None),
                sleeve=str(position.get("sleeve", "t0_sniper")),
                is_open_position=True,
                held_multiple=multiple,
            ))
        return opportunities

    def _feed_opportunity_quality(self, slate: Any) -> None:
        """Tell the risk engine how good today's remaining opportunities look.

        The giveback allowance and the harvest hurdle both need this, and the
        allocator has already measured it. Deriving it a second time would be
        a second opinion that can disagree with the one actually allocating
        capital.

        Quality is the best available edge over the engine's own hurdle;
        uncertainty is the share of the slate it could not rank at all, since
        a universe we mostly cannot measure is not a universe we should be
        tolerating extra drawdown for.
        """
        best = getattr(slate, "best", None)
        ranked = len(getattr(slate, "ranked", ()) or ())
        blocked = len(getattr(slate, "blocked", ()) or ())
        total = ranked + blocked
        if best is None or total == 0:
            self.elogw_engine.observe_opportunity_set(None)
            return
        hurdle = max(self.elogw_engine.min_edge_bps / 10_000.0, 1e-9)
        self.elogw_engine.observe_opportunity_set(
            quality=float(best.elogw) / hurdle,
            uncertainty=float(blocked) / float(total),
        )

    def _exit_capacity(self, token: str, position: Dict[str, Any]) -> Tuple[str, float]:
        """What share of this position is actually liquidatable right now.

        Read from the streamed bonding-curve state when one is available, which
        needs no RPC round trip and is therefore answerable inside the window
        the decision has to be made in. There is no permissive fallback: an
        unmeasurable capacity is DATA_BLOCKED, never 1.0.
        """
        state = self._latest_curve_state.get(token)
        if state is None:
            return "DATA_BLOCKED_NO_CURVE_STATE", 0.0
        report = curve_tradeability(state, quote_buy, quote_sell)
        position["tradeability"] = report.report()
        return exit_capacity_ratio(
            int(position.get("size_tokens", 0) or 0), report.exit,
            acceptable_impact=float(self.global_config.get("acceptable_exit_impact", 0.10)),
        )

    def _score_actions(self, token: str, position: Dict[str, Any], multiple: float,
                       distribution: Any):
        """Price every move this position can make against one distribution.

        Assembled from the numbers already computed this cycle -- the refreshed
        prediction, the measured exit capacity, the escape estimate, the
        allocator's best alternative -- so the action-value policy cannot
        disagree with the components that produced them.
        """
        prediction = position.get("prediction_object")
        if prediction is None:
            return ActionDecision(status="DATA_BLOCKED", detail="no current prediction")
        capacity_status = position.get("exit_capacity_status", "")
        if not str(capacity_status).startswith("OK"):
            return ActionDecision(status="DATA_BLOCKED",
                                  detail=f"exit capacity {capacity_status or 'unmeasured'}")

        equity = max(self.wallet_equity_usd, 1e-9)
        held_cost = float(position.get("remaining_cost_usd", 0.0) or 0.0)
        held_fraction = min(1.0, held_cost / equity)
        liquidity = float(position.get("liquidity_usd", 0.0) or 0.0)

        # The forward distribution from HERE. `probability_bins` returns
        # (name, probability, gross) already normalised.
        bins = [(probability, gross)
                for _, probability, gross in self.elogw_engine.probability_bins(prediction)]
        if not bins:
            return ActionDecision(status="DATA_BLOCKED", detail="no probability bins")

        add_fraction = None
        if liquidity > 0:
            planned, gain = self.elogw_engine.plan_scale_in(
                prediction, held_cost, multiple, liquidity, portfolio_value=equity)
            add_fraction = planned if planned > 0 and gain > 0 else None

        escape = position.get("escape") or {}
        state = ActionState(
            held_fraction=held_fraction,
            current_multiple=max(0.0, multiple),
            forward_bins=tuple(bins),
            exit_cost=float(self.global_config.get("assumed_exit_cost", 0.02)),
            entry_cost=float(self.global_config.get("assumed_entry_cost", 0.02)),
            exit_capacity_ratio=float(position.get("exit_capacity_ratio", 0.0) or 0.0),
            escape_probability=(float(escape["probability"])
                                if escape.get("status") == "OK" else None),
            expected_remaining_seconds=max(
                1.0, float(getattr(prediction, "expected_hold_time", 0.0) or 0.0)
                - (time.time() - float(position.get("entry_time", time.time())))),
            alternative_growth_per_second=(self.last_slate_report or {}).get("best_score"),
            add_fraction=add_fraction,
            add_capacity_fraction=(self.elogw_engine.exposure_cap(liquidity) - held_fraction
                                   if liquidity > 0 else None),
        )
        return self.action_policy.score(state)

    async def _apply_action(self, token: str, position: Dict[str, Any],
                            decision: Any, multiple: float) -> bool:
        """Execute a chosen action. Returns True when the position is done for this cycle."""
        action = decision.action
        if action is ActionValue.ADD:
            await self._consider_scale_in(token, position, multiple)
            return True
        fraction = action.bank_fraction
        if fraction <= 0:
            return False
        logger.info("ACTION %s %s q=%.6f at %.2fx", action.value, token, decision.q, multiple)
        await self._execute_exit(token, position, fraction, f"action_{action.value}")
        return True

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
        self.wallet_independence.record_entries([Entry(
            token=token, wallet=str(wallet),
            timestamp=float(observation.get("timestamp", time.time())),
            skill=(float(getattr(score, "overall_score", 0.0)) if score is not None else None),
            capital_usd=((float(notional) * self.sol_price_usd)
                         if notional and self.sol_price_usd > 0 else None),
        )])

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
        logger.info("INDEPENDENCE status=%s pairs=%d wallets=%d",
                    self.independence_report.status,
                    self.independence_report.observed_pairs,
                    len(self.independence_report.scores))

    def _estimate_escape(self, token: str, position: Dict[str, Any], hazard: Any):
        """P(this position gets out before the event), at its current size.

        A predicted 5x is worth nothing if the probability of the sell landing
        before the collapse is near zero, so the two are estimated separately
        and multiplied rather than one standing in for the other.

        Mechanism matters as much as rate: speed can outrun a seller and
        cannot outrun a frozen mint, which is why the hazard is decomposed
        before the race is evaluated rather than after.
        """
        if hazard is None:
            return EscapeEstimate(status="DATA_BLOCKED", detail="no hazard state for this token")
        mechanisms: Dict[HazardMechanism, Tuple[float, float]] = {}
        for mechanism, horizon, value in (
            (HazardMechanism.CREATOR_SELLING, 30.0, getattr(hazard, "hazard_30s", None)),
            (HazardMechanism.INSIDER_CLUSTER_EXIT, 300.0, getattr(hazard, "hazard_5m", None)),
        ):
            if value is not None and 0 <= float(value) < 1.0:
                mechanisms[mechanism] = (float(value), horizon)
        route = (self.rug_hazard.observations.get(token) or [])
        if any(item.get("type") == "route" and item.get("feasible") is False
               for item in list(route)[-20:]):
            # A route that has stopped quoting is a sellability signal, and
            # speed is no answer to it.
            mechanisms[HazardMechanism.SELLABILITY_LOSS] = (0.5, 30.0)
        curve = hazard_curve_from_probabilities(mechanisms)
        position["hazard_curve"] = curve.report()
        if curve.status != "OK":
            return EscapeEstimate(status="DATA_BLOCKED", detail="hazard could not be decomposed")

        state = self._latest_curve_state.get(token)
        sellable = None
        if state is not None:
            report = curve_tradeability(state, quote_buy, quote_sell)
            sellable = report.exit.size_at(
                float(self.global_config.get("acceptable_exit_impact", 0.10)))
        return escape_probability(
            position_size=int(position.get("size_tokens", 0) or 0),
            sellable_size=sellable,
            expected_latency_s=float(self.global_config.get("expected_exit_latency_s", 0.4)),
            hazard=curve,
        )

    def _update_monster_state(self, token: str, position: Dict[str, Any],
                              distribution: Any, multiple: float):
        """Advance this position's conviction state on currently observed evidence.

        Monster probability is supplied only when a calibrated source exists.
        Without one the machine stays in NORMAL, `overrides_ordinary_exit` is
        False, and every ordinary exit rule applies unchanged -- an override
        that lets a position ignore its stop, granted on an unvalidated number,
        is the most expensive fabrication available.
        """
        hazard = self.rug_hazard.get_hazard(token)
        prediction = position.get("prediction_object")
        catastrophic = bool(hazard and getattr(hazard, "urgency", "") == "critical")
        escape = self._estimate_escape(token, position, hazard)
        position["escape"] = {"status": escape.status, "probability": escape.probability,
                              "fillable_share": escape.fillable_share,
                              "detail": escape.detail}
        if (escape.status == "OK" and escape.probability <= self.min_escape_probability
                and not self.monster_machine.overrides_ordinary_exit(token)):
            # A position we are measurably unlikely to get out of is not a
            # position, whatever its predicted upside. Being early to a 5x we
            # cannot sell is worse than missing it, because the capital is
            # still committed when the window closes.
            catastrophic = True
        evidence = MonsterEvidence(
            monster_probability=None,
            monster_probability_calibrated=False,
            distribution_probability=distribution.probability(3.0),
            distribution_calibrated=distribution.calibrated,
            rug_probability=(float(getattr(hazard, "hazard_5m", 0.0)) if hazard else None),
            catastrophic_hazard=catastrophic,
        )
        features = distribution.features or {}
        if distribution.coverage > 0:
            # Sign-flipped distribution features are the same observations read
            # as continuation evidence; they are not a second opinion.
            evidence.smart_wallet_net_accumulation = 0.5 - features.get("smart_wallet_exit_rate", 0.5)
            evidence.buyer_quality_trend = -features.get("buyer_quality_decline", 0.0)
            evidence.independent_buyer_acceleration = -features.get("buy_acceleration_rollover", 0.0)
            evidence.sell_absorption = 1.0 - features.get("sell_absorption_failure", 0.0)
            evidence.audience_penetration = features.get("new_buyer_saturation", None)

        value = None
        capacity_status, capacity = self._exit_capacity(token, position)
        position["exit_capacity_status"] = capacity_status
        position["exit_capacity_ratio"] = capacity
        if (prediction is not None and prediction.expected_feasible_multiple > 0
                and capacity_status.startswith("OK")):
            # No value comparison without a measured exit capacity. Assuming a
            # position is fully sellable is how a theoretical return becomes a
            # real loss, and it is the assumption that would be doing the most
            # work here if it were allowed.
            value = hold_versus_exit(
                remaining_upside_multiple=max(1e-9, float(prediction.expected_feasible_multiple)),
                distribution_probability=float(distribution.probability(10.0) or 0.0),
                rug_probability=float(getattr(hazard, "hazard_5m", 0.0) or 0.0) if hazard else 0.0,
                exit_capacity_ratio=capacity,
                alternative_growth_per_second=float(
                    (self.last_slate_report or {}).get("best_score") or 0.0)
                * float(position.get("remaining_cost_usd", 0.0) or 0.0),
                expected_remaining_seconds=max(
                    1.0, float(getattr(prediction, "expected_hold_time", 0.0) or 0.0)),
            )
        decision = self.monster_machine.update(token, evidence, value)
        position["monster_state"] = decision.state.value
        position["monster_action"] = decision.action
        return decision

    def _read_distribution(self, token: str):
        """Evaluate the distribution detector on this token's observed flow."""
        observations = list(getattr(self.rug_hazard, "observations", {}).get(token, ()))
        return self.distribution_detector.evaluate(observations, time.time())

    async def _refresh_position_prediction(self, token: str, position: Dict[str, Any]) -> None:
        """Re-price the open position on current evidence.

        Both the exit decision and the scale-in decision consume this, so they
        cannot disagree about what the model currently believes. Liquidity is
        re-resolved rather than reused: entry-time depth is the one number that
        is guaranteed to be wrong later, and it is the number that caps how
        much can actually be added or sold.
        """
        if self.predictor is None or not self.predictor._is_trained:
            return
        candidate = position.get("candidate")
        risk = position.get("risk_object")
        if candidate is None or risk is None:
            return
        liquidity = await self._resolve_liquidity(candidate)
        if liquidity <= 0:
            # Depth we cannot observe is not depth we may assume. Leave the
            # previous prediction in place and mark it stale rather than
            # re-predicting against a fabricated zero.
            position["prediction_status"] = "DATA_BLOCKED_LIQUIDITY"
            return
        position["liquidity_usd"] = liquidity
        features = await self._build_prediction_features(candidate, risk, liquidity)
        prediction = self.predictor.predict(features)
        if prediction is None:
            position["prediction_status"] = "DATA_BLOCKED_PREDICTION"
            return
        position["prediction"] = _jsonable(prediction)
        position["prediction_object"] = prediction
        position["prediction_at"] = time.time()
        position["prediction_status"] = "OK"

    async def _consider_scale_in(self, token: str, position: Dict[str, Any], multiple: float):
        """Add to a winner only while the NEXT unit still raises E[log W].

        Committing the whole position at T0 forces sizing before the flow that
        actually separates a launch has arrived. Re-predicting on current
        evidence and adding on the marginal quantity deploys capital as the
        evidence appears, and stops the moment it stops paying.
        """
        if self.predictor is None or not self.predictor._is_trained:
            return
        if not self.dry_run and not self.champion_challenger.is_live(MODEL_HYPOTHESIS_ID):
            return
        candidate = position.get("candidate")
        prediction = position.get("prediction_object")
        liquidity = float(position.get("liquidity_usd", 0) or 0)
        if candidate is None or prediction is None or liquidity <= 0:
            return
        if prediction.p_rug_30s > 0.40 or prediction.p_rug_5m > 0.50:
            return

        await self._refresh_portfolio_state()
        fraction, gain = self.elogw_engine.plan_scale_in(
            prediction, float(position["remaining_cost_usd"]), multiple, liquidity,
            portfolio_value=self.wallet_equity_usd,
        )
        if fraction <= 0 or gain <= 0 or self.sol_price_usd <= 0:
            return

        add_usd = self.wallet_equity_usd * fraction
        result = await self.execution_engine.execute_swap(
            candidate.base_token or WSOL_MINT, token, int(add_usd / self.sol_price_usd * 1e9),
            slippage_bps=100,
            priority_fee=self.fee_optimizer.get_optimal_fee(add_usd, 0.5),
            jito_tip=self.fee_optimizer.get_jito_tip(add_usd, "MEDIUM"),
            use_jito=True, decision_id=position.get("decision_id"),
        )
        attempt = {**_jsonable(result), "scale_in": True, "marginal_elogw": gain,
                   "added_fraction": fraction, "at_multiple": multiple}
        self.dataset_builder.record_execution_attempt(token, attempt)
        if not result.success:
            return

        if result.simulated:
            added_cost = add_usd
        else:
            native_spent = max(0, -int(result.native_balance_delta_lamports)) / 1e9 * self.sol_price_usd
            added_cost = native_spent or add_usd
            if added_cost <= 0:
                return
        position["size_tokens"] = int(position["size_tokens"]) + int(result.output_amount)
        position["initial_size_tokens"] = int(position.get("initial_size_tokens", 0)) + int(result.output_amount)
        position["remaining_cost_usd"] = float(position["remaining_cost_usd"]) + added_cost
        position["initial_cost_usd"] = float(position.get("initial_cost_usd", 0.0)) + added_cost
        position.setdefault("scale_ins", []).append(
            {"fraction": fraction, "cost_usd": added_cost, "multiple": multiple, "elogw_gain": gain}
        )
        logger.info("%s SCALE-IN %s +$%.2f at %.2fx marginal_elogw=%.6f",
                    "PAPER" if self.dry_run else "LIVE", token, added_cost, multiple, gain)

    async def _mark_position(self, token: str, position: Dict[str, Any]):
        stream_mark = self._latest_stream_mark.get(token)
        if self.dry_run and stream_mark and time.time() - stream_mark["timestamp"] <= 3.0:
            multiple = float(stream_mark["multiple"])
            current_value = max(0.0, float(position["remaining_cost_usd"]) * multiple)
            observation = {"type": "stream_mark", "feasible": True, "value_usd": current_value,
                           "price_multiple": multiple, "timestamp": stream_mark["timestamp"],
                           "measurement": "decoded_onchain_reserve_event", "data_status": "OK"}
            self.rug_hazard.record_observation(token, observation)
            self.dataset_builder.record_market_observation(token, observation)
            self.counterfactual_lab.record_market_observation(token, multiple, observation["timestamp"])
            return multiple, current_value
        quote = await self.jupiter.get_quote(token, USDC_MINT, int(position["size_tokens"]), slippage_bps=500)
        if not quote:
            self.rug_hazard.record_observation(token, {"type": "route", "feasible": False, "timestamp": time.time()})
            return None
        current_value = quote.output_amount / 1_000_000
        remaining_cost = max(float(position["remaining_cost_usd"]), 1e-9)
        multiple = current_value / remaining_cost
        observation = {"type": "route", "feasible": True, "price_impact_pct": quote.price_impact_pct,
                       "value_usd": current_value, "price_multiple": multiple, "timestamp": time.time()}
        self.rug_hazard.record_observation(token, observation)
        self.dataset_builder.record_market_observation(token, observation)
        self.counterfactual_lab.record_market_observation(token, multiple, observation["timestamp"])
        return multiple, current_value

    async def _observe_active_markets(self):
        """Collect outcome paths independently of prediction/trade authority.

        A blocked model must not prevent the research lake from learning. The
        loop is intentionally budgeted and age-adaptive so thousands of active
        episodes cannot create an unbounded quote storm.
        """
        if not self.jupiter or not self.jupiter._session or not self.dataset_builder.active_episodes:
            return
        self._refresh_market_observation_cohort()
        tokens = list(self._market_observation_cohort)
        if not tokens:
            return
        budget = min(int(self.global_config.get("market_observation_budget", 5)), len(tokens))
        now = time.time()
        inspected = 0
        due = []
        while inspected < len(tokens) and budget > 0:
            token = tokens[self._market_cursor % len(tokens)]
            self._market_cursor = (self._market_cursor + 1) % max(len(tokens), 1)
            inspected += 1
            episode = self.dataset_builder.active_episodes.get(token)
            if not episode:
                continue
            age = max(0.0, now - episode.created_at)
            interval = 2.0 if age < 60 else 10.0 if age < 300 else 60.0
            if now - self._market_observed_at.get(token, 0) < interval:
                continue
            self._market_observed_at[token] = now
            budget -= 1
            due.append(token)
        all_active = set(self.dataset_builder.active_episodes)
        for stale in set(self._market_observed_at) - all_active:
            self._market_observed_at.pop(stale, None)
            self._market_entry_price.pop(stale, None)
            self._curve_entry_price.pop(stale, None)
        if due:
            results = await asyncio.gather(
                *(self._observe_token_market(token, now) for token in due),
                return_exceptions=True,
            )
            for token, result in zip(due, results):
                if isinstance(result, Exception):
                    logger.warning("Market observation failed for %s: %s", token, result)

    def _refresh_market_observation_cohort(self):
        """Keep a bounded cohort long enough to collect repeated executable marks."""
        active = self.dataset_builder.active_episodes
        limit = max(1, int(self.global_config.get("market_observation_cohort_size", 100)))
        self._market_observation_cohort.intersection_update(active)
        if len(self._market_observation_cohort) >= limit:
            return
        newest = sorted(
            (episode for token, episode in active.items() if token not in self._market_observation_cohort),
            key=lambda episode: (float(episode.created_at), episode.token), reverse=True,
        )
        for episode in newest[:limit - len(self._market_observation_cohort)]:
            self._market_observation_cohort.add(episode.token)

    async def _observe_token_market(self, token: str, observed_at: float):
        probe_lamports = int(self.global_config.get("market_probe_lamports", 10_000_000))
        buy_quote = await self.jupiter.get_quote(WSOL_MINT, token, probe_lamports, slippage_bps=300)
        if not buy_quote or buy_quote.output_amount <= 0:
            observation = {"type": "route", "feasible": False, "timestamp": observed_at,
                           "data_status": "DATA_BLOCKED", "reason": "buy_quote_unavailable"}
            self.dataset_builder.record_market_observation(token, observation)
            self.rug_hazard.record_observation(token, observation)
            return
        sell_quote = await self.jupiter.get_quote(token, USDC_MINT, buy_quote.output_amount, slippage_bps=500)
        if not sell_quote or sell_quote.output_amount <= 0:
            observation = {"type": "route", "feasible": False, "timestamp": observed_at,
                           "data_status": "DATA_BLOCKED", "reason": "sell_quote_unavailable"}
            self.dataset_builder.record_market_observation(token, observation)
            self.rug_hazard.record_observation(token, observation)
            return
        value_usd = sell_quote.output_amount / 1_000_000
        unit_price = value_usd / buy_quote.output_amount
        entry_price = self._market_entry_price.setdefault(token, unit_price)
        multiple = unit_price / max(entry_price, 1e-18)
        impact = max(float(buy_quote.price_impact_pct), float(sell_quote.price_impact_pct))
        liquidity_estimate = (probe_lamports / 1e9 * self.sol_price_usd) / max(impact, 0.001)
        observation = {
            "type": "market_mark", "timestamp": observed_at, "data_status": "OK",
            "price_usd": unit_price, "price_multiple": multiple, "value_usd": value_usd,
            "route_feasible": True, "feasible": True, "price_impact_pct": impact,
            "liquidity_usd": liquidity_estimate, "sol_price_usd": self.sol_price_usd,
            "measurement": "jupiter_round_trip_probe",
        }
        self.dataset_builder.record_market_observation(token, observation)
        self.rug_hazard.record_observation(token, {**observation, "type": "route"})
        self.counterfactual_lab.record_market_observation(token, multiple, observed_at)

    async def _execute_exit(self, token: str, position: Dict[str, Any], exit_pct: float, reason: str):
        current_tokens = int(position["size_tokens"])
        sold_tokens = min(current_tokens, max(1, int(current_tokens * min(max(exit_pct, 0), 1))))
        result = await self.execution_engine.execute_sell(token, sold_tokens, slippage_bps=500, use_jito=True,
                                                          decision_id=position.get("decision_id"))
        attempt = {**_jsonable(result), "exit_reason": reason, "exit_pct": sold_tokens / max(current_tokens, 1)}
        self._record_ops_event("execution_attempts", {
            "token": token, "side": "sell", "success": bool(result.success),
            "status": getattr(result.status, "value", str(result.status)),
            "error": result.error, "simulated": bool(result.simulated),
            "exit_reason": reason,
        })
        if not result.success:
            self.dataset_builder.record_execution_attempt(token, attempt)
            return
        actual_sold = int(result.actual_input_amount)
        if actual_sold <= 0 or actual_sold > current_tokens:
            attempt.update({
                "status": "DATA_BLOCKED_ACCOUNTING",
                "reason": "filled exit lacks a valid verified input-token balance decrease",
            })
            self.dataset_builder.record_execution_attempt(token, attempt)
            return
        proceeds = result.output_amount / 1_000_000
        native_delta_usd = int(result.native_balance_delta_lamports) / 1e9 * self.sol_price_usd
        allocated_cost = float(position["remaining_cost_usd"]) * actual_sold / max(current_tokens, 1)
        pnl = proceeds + native_delta_usd - allocated_cost
        self.total_pnl += pnl
        self.successful_exits += int(pnl > 0)
        self.elogw_engine.update_pnl(pnl)
        self.elogw_engine.reduce_position(token, actual_sold, allocated_cost)
        attempt.update({"exit_pct": actual_sold / max(current_tokens, 1),
                        "requested_tokens": sold_tokens, "actual_sold_tokens": actual_sold,
                        "proceeds_usd": proceeds, "native_balance_delta_usd": native_delta_usd,
                        "allocated_cost_usd": allocated_cost,
                        "realized_pnl_usd": pnl})
        self.dataset_builder.record_execution_attempt(token, attempt)
        remaining = int(position["size_tokens"]) - actual_sold
        if remaining <= 0:
            # The position is closed, so its outcome is now final and can be
            # attributed. Partial exits are deliberately not recorded here:
            # attributing a trade that is still open would count the same
            # capital twice in the ledger.
            mechanism = str(position.get("sleeve", "t0_sniper"))
            self._mechanism_growth.setdefault(mechanism, []).append(
                math.log(max(1e-9, 1.0 + pnl / max(self.wallet_equity_usd, 1e-9))))
            self._record_ops_event("trade_outcomes", {
                "token": token, "entered": True, "attempted": True,
                "realized_pnl_usd": self._closed_pnl.pop(token, 0.0) + pnl,
                "realized_multiple": (float(position.get("high_water_multiple", 1.0))
                                      if pnl > 0 else max(0.0, 1.0 + pnl / max(
                                          float(position.get("initial_cost_usd", 1.0)), 1e-9))),
                "max_feasible_multiple": float(position.get("high_water_multiple", 1.0)),
                "position_fraction": (float(position.get("initial_cost_usd", 0.0))
                                      / max(self.wallet_equity_usd, 1e-9)),
                "capacity_usd": position.get("liquidity_usd"),
                "exit_reason": reason,
                "mechanism": position.get("sleeve", "t0_sniper"),
                "monster_state": position.get("monster_state"),
                "exit_capacity_status": position.get("exit_capacity_status"),
                "wealth_multiple": 1.0 + pnl / max(self.wallet_equity_usd, 1e-9),
                "rugged": reason.startswith("rug_hazard") or reason.startswith("monster_catastrophic"),
            })
        else:
            self._closed_pnl[token] = self._closed_pnl.get(token, 0.0) + pnl
        resolved = self.counterfactual_lab.resolve_decision(position.get("decision_id", ""), pnl)
        for counterfactual in resolved:
            self.dataset_builder.record_counterfactual(token, counterfactual)
        logger.info("%s EXIT %s %.1f%% reason=%s proceeds=$%.2f allocated_cost=$%.2f pnl=$%.2f",
                    "PAPER" if self.dry_run else "LIVE", token, actual_sold / max(current_tokens, 1) * 100,
                    reason, proceeds, allocated_cost, pnl)

    async def _refresh_portfolio_state(self):
        if self.offline or not self.jupiter or not self.jupiter._session:
            self.equity_status = "DATA_BLOCKED"
            return
        quote = await self.jupiter.get_quote(WSOL_MINT, USDC_MINT, 1_000_000_000, slippage_bps=50)
        if not quote or quote.output_amount <= 0:
            self.equity_status = "DATA_BLOCKED"
            return
        self.sol_price_usd = quote.output_amount / 1_000_000
        if self.dry_run:
            self.wallet_equity_usd = float(self.global_config.get("paper_equity_usd", 10_000)) + self.total_pnl
        else:
            balance = await self.solana_rpc.request("getBalance", [str(self.keypair.pubkey()), {"commitment": "confirmed"}])
            sol = float((balance or {}).get("value", 0)) / 1e9
            marked_positions = 0.0
            for token, position in self.elogw_engine.open_positions.items():
                token_quote = await self.jupiter.get_quote(token, USDC_MINT, int(position["size_tokens"]), slippage_bps=500)
                if token_quote:
                    marked_positions += token_quote.output_amount / 1_000_000
            self.wallet_equity_usd = sol * self.sol_price_usd + marked_positions
        # Capital withheld for a rare event is capital the sizing path must not
        # see. Reducing the portfolio value the engine reasons about is the
        # only place to apply it: every ceiling downstream is a fraction of
        # that number, so the reserve reaches all of them at once and cannot
        # be forgotten by one.
        reserve = self.mega_event_reserve.decide(
            self.mega_event_probability, self.mega_event_authenticated)
        self.mega_event_reserve_state = {
            "status": reserve.status, "fraction": reserve.reserve_fraction,
            "reason": reserve.reason, "event_probability": reserve.event_probability,
        }
        self.elogw_engine.portfolio_value = self.mega_event_reserve.deployable_equity(
            self.wallet_equity_usd, reserve)
        self.equity_status = "OK"

    async def _update_intelligence(self):
        if time.time() - self.last_intelligence_update < 60:
            return
        self.last_intelligence_update = time.time()
        await self._refresh_portfolio_state()
        await self.genealogy.build_clusters()
        self._refresh_independence()
        self._publish_attribution()
        if self.dry_run:
            latest_mtime = self._latest_model_mtime()
            if latest_mtime > self._model_artifact_mtime:
                candidate = MultiHeadPredictor()
                candidate.initialize_models()
                if candidate.load_latest():
                    self.predictor = candidate
                    self.elogw_engine.predictor = candidate
                    self._model_artifact_mtime = latest_mtime
                    self._register_model_validation(candidate.validation_report)
                    logger.info("Activated chronologically validated shadow model %s", candidate.model_version)

    async def _on_pump_event(self, event: Dict[str, Any]):
        token = event.get("token", "")
        if not token:
            return
        if event.get("type") == "token_created":
            self.wallet_intel.record_token_lifecycle(token, launch_at=event.get("timestamp", time.time()))
            self.dataset_builder.start_episode(
                token, event.get("creator", ""), event.get("program", PumpFunMonitor.PUMP_FUN_PROGRAM),
                event.get("bonding_curve", ""), WSOL_MINT, detected_at=event.get("timestamp", time.time()),
                prelaunch_context=self._prelaunch_context(event.get("creator", ""), event.get("timestamp", time.time())),
            )
            self.info_graph.record_event(token, LeadEventType.DEPLOYER_ACTIVITY, event.get("creator", ""),
                                         "deployer", event.get("timestamp", time.time()), event)
            if hasattr(self.genealogy, "record_token_creation"):
                self.genealogy.record_token_creation(token, event.get("creator", ""), event)
            self._spawn_background(self.wallet_intel.analyze_token_early_buyers(token))
            self._spawn_background(self.social_intel.scan_token(token))
            await self.detection_engine._on_candidate(TokenCandidate(
                address=token, chain="solana", source=DetectionSource.FACTORY, block_number=int(event.get("slot", 0)),
                tx_hash=event.get("signature"), deployer=event.get("creator"), factory=PumpFunMonitor.PUMP_FUN_PROGRAM,
                pair=event.get("bonding_curve"), base_token=WSOL_MINT, timestamp=event.get("timestamp", time.time()),
                metadata={"name": event.get("name"), "symbol": event.get("symbol"), "uri": event.get("uri"),
                         "funding_wallets": event.get("funding_wallets", []),
                         "funding_transfers": event.get("funding_transfers", [])},
            ))
        elif event.get("type") == "token_trade":
            curve_price = float(event.get("curve_price_raw", 0) or 0)
            virtual_sol = int(event.get("virtual_sol_reserves") or 0)
            virtual_token = int(event.get("virtual_token_reserves") or 0)
            if virtual_sol > 0 and virtual_token > 0:
                self._latest_curve_state[token] = BondingCurveState(
                    virtual_token_reserves=virtual_token, virtual_sol_reserves=virtual_sol,
                    # The TradeEvent carries no real reserves. Leaving these at
                    # zero is what marks every frontier derived from this state
                    # as an upper bound rather than a measurement.
                    real_token_reserves=0, real_sol_reserves=0, token_total_supply=0,
                    complete=False, creator="",
                )
            curve_multiple = None
            if curve_price > 0:
                curve_entry = self._curve_entry_price.setdefault(token, curve_price)
                curve_multiple = curve_price / max(curve_entry, 1e-30)
                self._latest_stream_mark[token] = {
                    "multiple": curve_multiple, "timestamp": float(event.get("timestamp", time.time()))
                }
            observation = {"type": "trade", "side": event.get("side"), "wallet": event.get("wallet"),
                           "amount": event.get("actual_token_amount_ui"),
                           "amount_raw": event.get("actual_token_delta_raw"),
                           "notional_sol": event.get("notional_sol"),
                           "price": event.get("price_sol_per_token"),
                           "curve_price_raw": curve_price or None,
                           "price_multiple": curve_multiple,
                           "fill_data_status": event.get("fill_data_status", "DATA_BLOCKED"),
                           "instruction_token_amount": event.get("token_amount", 0),
                           "quote_limit_amount": event.get("quote_limit_amount", 0),
                           "timestamp": event.get("timestamp", time.time()), "slot": event.get("slot"),
                           "signature": event.get("signature"), "program": event.get("program")}
            self._record_actor_entry(token, event, observation)
            self.rug_hazard.record_observation(token, observation)
            self.public_coordination.record_trade(token, observation)
            self.dataset_builder.record_market_observation(token, observation)
            if hasattr(self.wallet_intel, "record_live_trade"):
                self.wallet_intel.record_live_trade(token, observation)
            score = self.wallet_intel.get_wallet_score(event.get("wallet", ""))
            if score and score.overall_score >= 0.7:
                event_type = LeadEventType.ELITE_WALLET_BUY if event.get("side") == "buy" else LeadEventType.SMART_WALLET_EXIT
                self.info_graph.record_event(token, event_type, event.get("wallet", ""), "wallet",
                                             event.get("timestamp", time.time()), event)
        elif event.get("type") == "token_migrated":
            self.wallet_intel.record_token_lifecycle(token, migration_at=event.get("timestamp", time.time()))
            self.info_graph.record_event(token, LeadEventType.MIGRATION, "pump_fun", "program",
                                         event.get("timestamp", time.time()), event)
            self.dataset_builder.record_market_observation(token, {"type": "migration", **event})
        elif event.get("type") == "pool_created":
            self.wallet_intel.record_token_lifecycle(token, migration_at=event.get("timestamp", time.time()))
            self.dataset_builder.start_episode(
                token, event.get("creator", ""), event.get("program", PumpSwapMonitor.PUMP_AMM_PROGRAM),
                event.get("pool", ""), event.get("quote_mint") or WSOL_MINT,
                detected_at=event.get("timestamp", time.time()),
                prelaunch_context=self._prelaunch_context(event.get("creator", ""), event.get("timestamp", time.time())),
            )
            self.info_graph.record_event(token, LeadEventType.MIGRATION, event.get("pool", ""), "pumpswap_pool",
                                         event.get("timestamp", time.time()), event)
            self.dataset_builder.record_market_observation(token, {"type": "migration", **event})
            await self.detection_engine._on_candidate(TokenCandidate(
                address=token, chain="solana", source=DetectionSource.FACTORY,
                block_number=int(event.get("slot", 0)), tx_hash=event.get("signature"),
                deployer=event.get("creator"), factory=event.get("program"), pair=event.get("pool"),
                base_token=event.get("quote_mint") or WSOL_MINT,
                timestamp=event.get("timestamp", time.time()),
                metadata={"initial_base_amount": event.get("initial_base_amount"),
                          "initial_quote_amount": event.get("initial_quote_amount")},
            ))

    async def _on_raydium_event(self, event: Dict[str, Any]):
        if event.get("type") != "pool_created":
            return
        mints = (event.get("mint_a"), event.get("mint_b"))
        for mint in mints:
            if not mint or mint in {WSOL_MINT, USDC_MINT}:
                continue
            self.wallet_intel.record_token_lifecycle(mint, migration_at=event.get("timestamp", time.time()))
            quote_mint = next((other for other in mints if other in {WSOL_MINT, USDC_MINT}), None)
            self.dataset_builder.start_episode(
                mint, event.get("creator", ""), event.get("program", ""), event.get("pool", ""),
                quote_mint or "", detected_at=event.get("timestamp", time.time()),
                prelaunch_context=self._prelaunch_context(event.get("creator", ""), event.get("timestamp", time.time())),
            )
            self.info_graph.record_event(mint, LeadEventType.MIGRATION, event.get("pool", ""), "raydium_pool",
                                         event.get("timestamp", time.time()), event)
            self.dataset_builder.record_market_observation(mint, {"type": "migration", **event})
            if hasattr(self.genealogy, "record_token_creation"):
                self.genealogy.record_token_creation(mint, event.get("creator", ""), event)
            await self.detection_engine._on_candidate(TokenCandidate(
                address=mint, chain="solana", source=DetectionSource.FACTORY, block_number=int(event.get("slot", 0)),
                tx_hash=event.get("signature"), deployer=event.get("creator"), factory=event.get("program"),
                pair=event.get("pool"), base_token=quote_mint, timestamp=event.get("timestamp", time.time()),
                metadata={"initial_base_amount": event.get("initial_base_amount"),
                          "initial_quote_amount": event.get("initial_quote_amount"),
                          "venue": event.get("venue"),
                          "data_status": event.get("data_status", "DATA_BLOCKED"),
                          "funding_wallets": event.get("funding_wallets", []),
                          "funding_transfers": event.get("funding_transfers", [])},
            ))

    async def _on_social_mention(self, signal: Dict[str, Any]):
        token = signal.get("token", "")
        if not token:
            return
        if signal.get("type") == "new_mention":
            self.info_graph.record_event(token, LeadEventType.OBSCURE_X_MENTION, signal.get("account", ""),
                                         "social", signal.get("timestamp", time.time()), signal)
        self.rug_hazard.record_observation(token, {"type": "social", **signal})
        self.dataset_builder.record_market_observation(token, {"type": "social", **signal})
        if signal.get("type") == "new_mention" and signal.get("first_mention"):
            self._spawn_background(self._triage_social_candidate(signal))

    async def _triage_social_candidate(self, signal: Dict[str, Any]):
        """Evaluate social addresses only after verifying that the account is an SPL mint."""
        token = str(signal.get("token", ""))
        if not token or not self.solana_rpc or not self.detection_engine:
            return
        try:
            result = await self.solana_rpc.request(
                "getAccountInfo", [token, {"encoding": "jsonParsed", "commitment": "processed"}],
            )
            value = (result or {}).get("value") or {}
            owner = str(value.get("owner", ""))
            parsed = (((value.get("data") or {}).get("parsed") or {}))
            if owner not in {
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
            } or parsed.get("type") != "mint":
                return
            await self.detection_engine._on_candidate(TokenCandidate(
                address=token, chain="solana", source=DetectionSource.SOCIAL,
                block_number=0, timestamp=float(signal.get("timestamp", time.time())),
                metadata={
                    "social_platform": signal.get("platform"),
                    "social_account": signal.get("account"),
                    "social_credibility": signal.get("credibility"),
                    "mint_verified": True,
                },
            ))
        except Exception as exc:
            logger.debug("Social candidate mint verification blocked for %s: %s", token, exc)

    def readiness(self) -> Dict[str, Any]:
        return {
            "mode": "DRY_RUN" if self.dry_run else "LIVE",
            "live_submission_locked": os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand",
            "offline": self.offline, "rpc": self.chain_registry.get_all_stats() if self.chain_registry else {},
            "yellowstone": self.yellowstone.get_status() if self.yellowstone else {"status": "NOT_STARTED"},
            "rpc_program_stream": self.rpc_program_stream.get_status() if self.rpc_program_stream else None,
            "prediction": "OK" if self.predictor and self.predictor._is_trained else "DATA_BLOCKED",
            "exit_policy": {"status": self.exit_policy_status, "detail": self.exit_policy_detail,
                            "policy": asdict(self.exit_policy)},
            "equity": {"status": self.equity_status, "wallet_equity_usd": self.wallet_equity_usd,
                       "sol_price_usd": self.sol_price_usd},
            "execution": {"dry_run": self.execution_engine.dry_run if self.execution_engine else True},
            "native_fastpath": NATIVE_FASTPATH_STATUS,
            "action_policy": {"trained": self.action_policy.is_trained,
                              "min_edge": self.action_policy.min_edge},
            "actor_graph": {"independence_status": self.independence_report.status,
                            "measured_pairs": self.independence_report.observed_pairs,
                            "scored_wallets": len(self.independence_report.scores)},
            "hot_state": self.hot_state.report(),
            "mega_event_reserve": self.mega_event_reserve_state,
            "portfolio": self.elogw_engine.get_portfolio_state() if self.elogw_engine else {},
            "rug_hazard": self.rug_hazard.get_stats() if self.rug_hazard else {},
            "dataset": self.dataset_builder.get_stats() if self.dataset_builder else {},
            "research": self.global_research.get_stats() if self.global_research else {},
            "social": self.social_intel.get_stats() if self.social_intel else {},
            "public_coordination": self.public_coordination.get_stats() if self.public_coordination else {},
            "champions": self.champion_challenger.get_stats() if self.champion_challenger else {},
        }

    async def _health_loop(self):
        while self._running:
            snapshot = _jsonable(self.readiness())
            logger.info("HEALTH %s", json.dumps(snapshot, separators=(",", ":")))
            self._persist_readiness(snapshot)
            await asyncio.sleep(60)

    def _record_ops_event(self, stream: str, payload: Dict[str, Any]) -> None:
        """Append one operational telemetry row for the monitor and audit pack.

        Deliberately separate from the research lake. The lake is optimised for
        point-in-time correctness and completeness; this is optimised for a
        monitor being able to answer "what is the recent failure rate" in one
        cheap pass without loading episodes. Failures are swallowed for the
        same reason readiness persistence is: telemetry must never be able to
        halt the desk it describes.
        """
        try:
            root = Path(self.global_config.get("ops_state_dir", "data/state"))
            root.mkdir(parents=True, exist_ok=True)
            row = {"timestamp": time.time(), **payload}
            with (root / f"{stream}.jsonl").open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError as exc:
            logger.debug("ops telemetry write failed for %s: %s", stream, exc)

    def _persist_readiness(self, snapshot: Dict[str, Any]) -> None:
        """Write the snapshot the out-of-process monitor reads.

        Logging health is not the same as exposing it. A monitor that has to
        parse the log stream cannot tell "the desk is fine and quiet" from "the
        desk stopped writing", whereas the mtime of this file answers that
        directly and is the first thing the monitor checks.

        Written to a temporary file and renamed, so a reader never catches a
        half-written snapshot and concludes the node is broken. Failures here
        are logged and swallowed: a monitor that cannot be updated must never
        be able to take down the desk it monitors.
        """
        path = Path(self.global_config.get("readiness_path", "data/state/readiness.json"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(snapshot, default=str))
            tmp.replace(path)
        except OSError as exc:
            logger.warning("could not persist readiness snapshot to %s: %s", path, exc)

    async def _setup_health_server(self):
        app = web.Application()
        app.router.add_get("/health", self._health_endpoint)
        app.router.add_get("/metrics", self._metrics_endpoint)
        app.router.add_get("/status", self._status_endpoint)
        self._web_runner = web.AppRunner(app)
        await self._web_runner.setup()
        await web.TCPSite(self._web_runner, "0.0.0.0", int(os.getenv("HEALTH_PORT", "8080"))).start()

    async def _health_endpoint(self, request):
        return web.json_response({"status": "healthy" if self._running else "stopping", "dry_run": self.dry_run,
                                  "uptime_seconds": time.time() - self.start_time,
                                  "live_submission_locked": os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand"})

    async def _metrics_endpoint(self, request):
        return web.json_response(_jsonable({"portfolio": self.elogw_engine.get_portfolio_state(),
                                            "total_pnl": self.total_pnl, "trade_count": self.trade_count,
                                            "successful_exits": self.successful_exits}))

    async def _status_endpoint(self, request):
        return web.json_response(_jsonable(self.readiness()))

    async def _close_health_server(self):
        if self._web_runner:
            await self._web_runner.cleanup()
            self._web_runner = None


async def _run(args: argparse.Namespace):
    dry_override = False if args.live else True if args.dry_run else None
    desk = MemecoinQuantDesk(args.config, dry_run_override=dry_override, offline=args.smoke_test)
    try:
        await desk.initialize()
        if args.smoke_test:
            print(json.dumps(_jsonable(desk.readiness()), indent=2, sort_keys=True))
            return
        await desk.start()
        if args.run_seconds:
            await asyncio.sleep(args.run_seconds)
        else:
            while True:
                await asyncio.sleep(3_600)
    finally:
        await desk.stop()


def main():
    parser = argparse.ArgumentParser(description="Solana memecoin research desk")
    parser.add_argument("--config", default="config/chains.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="quote and record trades without transaction submission")
    mode.add_argument("--live", action="store_true", help="request live mode; still requires the independent environment acknowledgement")
    parser.add_argument("--smoke-test", action="store_true", help="validate local wiring without network connections")
    parser.add_argument("--run-seconds", type=float, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
