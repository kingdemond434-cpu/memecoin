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
import os
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Optional

import yaml
from aiohttp import web
from solders.keypair import Keypair

from src.chains.provider_credentials import normalize_provider_environment
from src.chains.rpc_manager import ChainRegistry, RPCManager
from src.chains.yellowstone_grpc import (
    PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream, YellowstoneClient,
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
from src.research.global_research_miner import GlobalResearchMiner
from src.strategies.champion_challenger import ChampionChallengerFramework, HypothesisSpec, TrialResult
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.information_graph import (
    AdversarialAdaptationDetector,
    CounterfactualExecutionLab,
    InformationLeadGraph,
    LeadEventType,
)
from src.strategies.multihead_predictor import ElogwEngine, MultiHeadPredictor, PredictionFeatures
from src.strategies.prelaunch_intent import PrelaunchIntentModel
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.social_intelligence import SocialIntelligenceEngine
from src.strategies.wallet_intelligence import WalletIntelligenceEngine

logger = logging.getLogger(__name__)

WSOL_MINT = "So11111111111111111111111111111111111111112"
MODEL_HYPOTHESIS_ID = "production_multihead_v1"


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
        self.elogw_engine = ElogwEngine(
            self.predictor,
            max_position_pct=float(self.global_config.get("max_position_pct", 0.05)),
            max_position_usd=float(self.global_config.get("max_position_size_usd", 500)),
            max_portfolio_risk=float(self.global_config.get("max_portfolio_risk", 0.10)),
            max_total_exposure_pct=float(self.global_config.get("max_total_exposure_pct", 0.30)),
            max_concurrent_positions=int(self.global_config.get("max_concurrent_positions", 10)),
            max_daily_loss_usd=float(self.global_config.get("max_daily_loss_usd", 1_000)),
            max_liquidity_fraction=float(self.global_config.get("max_liquidity_fraction", 0.01)),
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
        for item in candidate.metadata.get("funding", []):
            self.public_coordination.record_funding(
                token, item.get("wallet", ""), item.get("funder", ""),
                float(item.get("amount_sol", 0) or 0), item.get("timestamp"),
            )
        if risk.risk_level.value in {"high", "critical", "honeypot", "rugged"}:
            self._record_blocked_decision(token, "safety_rejection", risk_data)
            return
        if risk.data_status == "DATA_BLOCKED" and self.global_config.get("reject_data_blocked_safety_checks", True):
            self._record_blocked_decision(token, "DATA_BLOCKED_safety_checks", risk_data)
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
            "prediction": _jsonable(prediction), "risk_report": risk_data, "trade_info": trade_info,
            "decision_id": decision_id, "paper": self.dry_run, "high_water_multiple": 1.0,
            "ratchet_stages": [],
        }
        self.elogw_engine.update_position(token, position)
        self.trade_count += 1
        logger.info("%s BUY %s %.4f SOL status=%s", "PAPER" if self.dry_run else "LIVE", token,
                    trade_info["position_size_sol"], result.status.value)

    def _record_blocked_decision(self, token: str, reason: str, evidence: Dict[str, Any]):
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
        features = PredictionFeatures(token=candidate.address, chain=candidate.chain, timestamp=time.time())
        deployer = self.genealogy.get_deployer_profile(candidate.deployer or "")
        if deployer:
            features.deployer_rug_rate = deployer.rug_rate
            features.deployer_success_rate = deployer.success_rate
            features.deployer_avg_multiple = deployer.avg_max_multiple
        graph_risk = self.genealogy.assess_launch_risk(candidate.deployer or "",
                                                       candidate.metadata.get("funding_wallets", []),
                                                       candidate.metadata.get("initial_buyers", []))
        features.deployer_cluster_risk = graph_risk.get("risk_score", 0)
        buys = [item for item in self.wallet_intel._recent_buys if item.get("token") == candidate.address]
        features.initial_buyers = len({item.get("wallet") for item in buys if item.get("wallet")})
        buyer_scores = [self.wallet_intel.get_wallet_score(item.get("wallet", "")) for item in buys]
        features.smart_buyers = sum(score is not None and score.overall_score >= 0.7 for score in buyer_scores)
        features.wallet_history_available = any(score is not None for score in buyer_scores)
        features.sol_volume = sum(float(item.get("amount", 0) or 0) * float(item.get("price", 0) or 0) for item in buys)
        episode = self.dataset_builder.active_episodes.get(candidate.address)
        if episode:
            flow = await self.dataset_builder._capture_flow_features(episode, features.timestamp)
            if flow.get("status") == "OK":
                features.flow_available = True
                features.buy_velocity = float(flow.get("buy_velocity", 0) or 0)
                features.buyer_acceleration = float(flow.get("buy_acceleration", 0) or 0)
                features.sol_volume = float(flow.get("buy_notional_sol_10s", features.sol_volume) or 0)
        coordination = self.public_coordination.get_features(candidate.address)
        if coordination.get("status") == "OK":
            features.bundle_concentration = float(coordination["coordinated_buyer_fraction"])
            features.organic_ratio = float(coordination["organic_ratio"])
            features.insider_buyers = len(coordination["coordinated_wallets"])
            features.coordination_available = True
        features.liquidity_usd = liquidity
        features.liquidity_locked = bool(risk.liquidity_locked)
        features.can_mint = bool(risk.can_mint)
        features.can_freeze = bool(risk.can_freeze)
        features.holder_concentration = float(risk.top_10_pct) / 100
        features.top_10_pct = float(risk.top_10_pct)
        social = self.social_intel.get_token_social_signal(candidate.address)
        features.social_velocity = float(social.get("avg_velocity", 0) or 0)
        features.social_acceleration = float(social.get("acceleration", 0) or 0)
        features.social_credibility = float(social.get("avg_credibility", 0) or 0)
        features.chain_before_social = float(social.get("chain_before_pct", 0) or 0)
        features.cross_platform = bool(social.get("cross_platform", False))
        features.social_available = bool(social.get("mention_count", 0))
        coverage_checks = [
            deployer is not None, risk.data_status == "OK", liquidity > 0,
            features.wallet_history_available, features.coordination_available, features.social_available,
            features.flow_available,
        ]
        features.data_coverage = sum(coverage_checks) / len(coverage_checks)
        return features

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
            if multiple <= 0.70:
                await self._execute_exit(token, position, 1.0, "hard_stop_loss")
                continue
            if multiple >= 2 and "cost_recovery" not in stages:
                stages.append("cost_recovery")
                await self._execute_exit(token, position, min(0.50, 1.0 / multiple), "profit_ratchet_cost_recovery")
                continue
            if multiple >= 5 and "bank_5x" not in stages:
                stages.append("bank_5x")
                await self._execute_exit(token, position, 0.25, "profit_ratchet_5x")
                continue
            if multiple >= 10 and "bank_10x" not in stages:
                stages.append("bank_10x")
                await self._execute_exit(token, position, 0.20, "profit_ratchet_10x")
                continue
            high = position["high_water_multiple"]
            continuation = max(float(position["prediction"].get("p_5x", 0)), float(position["prediction"].get("p_10x", 0)))
            trail_ratio = 0.78 if continuation < 0.15 else 0.68 if high >= 5 else 0.58
            floor = max(1.10 if high >= 2 else 0.70, high * trail_ratio)
            if multiple <= floor and high >= 1.5:
                await self._execute_exit(token, position, 1.0, "adaptive_profit_trailing_stop")
                continue
            max_hold = float(self.global_config.get("max_hold_time_minutes", 60)) * 60
            if time.time() - float(position["entry_time"]) >= max_hold:
                await self._execute_exit(token, position, 1.0, "time_stop")

    async def _mark_position(self, token: str, position: Dict[str, Any]):
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
        self.elogw_engine.portfolio_value = self.wallet_equity_usd
        self.equity_status = "OK"

    async def _update_intelligence(self):
        if time.time() - self.last_intelligence_update < 60:
            return
        self.last_intelligence_update = time.time()
        await self._refresh_portfolio_state()
        await self.genealogy.build_clusters()
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
                metadata={"name": event.get("name"), "symbol": event.get("symbol"), "uri": event.get("uri")},
            ))
        elif event.get("type") == "token_trade":
            curve_price = float(event.get("curve_price_raw", 0) or 0)
            curve_multiple = None
            if curve_price > 0:
                curve_entry = self._curve_entry_price.setdefault(token, curve_price)
                curve_multiple = curve_price / max(curve_entry, 1e-30)
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
            self.info_graph.record_event(token, LeadEventType.MIGRATION, "pump_fun", "program",
                                         event.get("timestamp", time.time()), event)
            self.dataset_builder.record_market_observation(token, {"type": "migration", **event})
        elif event.get("type") == "pool_created":
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
            quote_mint = next((other for other in mints if other in {WSOL_MINT, USDC_MINT}), None)
            self.dataset_builder.start_episode(
                mint, event.get("creator", ""), event.get("program", ""), event.get("pool", ""),
                quote_mint or "", detected_at=event.get("timestamp", time.time()),
                prelaunch_context=self._prelaunch_context(event.get("creator", ""), event.get("timestamp", time.time())),
            )
            self.info_graph.record_event(mint, LeadEventType.MIGRATION, event.get("pool", ""), "raydium_pool",
                                         event.get("timestamp", time.time()), event)
            self.dataset_builder.record_market_observation(mint, {"type": "migration", **event})
            await self.detection_engine._on_candidate(TokenCandidate(
                address=mint, chain="solana", source=DetectionSource.FACTORY, block_number=int(event.get("slot", 0)),
                tx_hash=event.get("signature"), deployer=event.get("creator"), factory=event.get("program"),
                pair=event.get("pool"), base_token=quote_mint, timestamp=event.get("timestamp", time.time()),
                metadata={"initial_base_amount": event.get("initial_base_amount"),
                          "initial_quote_amount": event.get("initial_quote_amount"),
                          "venue": event.get("venue"),
                          "data_status": event.get("data_status", "DATA_BLOCKED")},
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
            "equity": {"status": self.equity_status, "wallet_equity_usd": self.wallet_equity_usd,
                       "sol_price_usd": self.sol_price_usd},
            "execution": {"dry_run": self.execution_engine.dry_run if self.execution_engine else True},
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
            logger.info("HEALTH %s", json.dumps(_jsonable(self.readiness()), separators=(",", ":")))
            await asyncio.sleep(60)

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
