"""Memecoin research and execution desk.

The shipped runtime defaults to dry-run. Live transaction submission requires
both an explicit ``--live`` launch and the execution engine's independent
``ALLOW_LIVE_TRADING=yes-i-understand`` acknowledgement.
"""

import argparse
import asyncio
import base64
import collections
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from src.research.forward_evidence import ForwardEvidence, Outcome as ForwardOutcome
from src.research.contribution import (
    ContributionLedger, GateFlip, action_value_contributions,
)
from src.research.attribution import EdgeDecayMonitor
from src.runtime.hot_state import HotState, HotStateBudget
from src.strategies.action_value import (
    Action as ActionValue, ActionValuePolicy, Decision as ActionDecision,
    PositionState as ActionState,
)
from src.collectors.event_source import Event, SourceMesh
from src.collectors.registry import build_sources, load_declarations
from src.collectors.transports import (
    HttpClient, build_transports, start_transports, stop_transports, transport_report,
)
from src.strategies.decision_snapshot import (
    DecisionSnapshot, StateSequencer, guard as decision_guard, state_hash,
)
from src.strategies.actor_graph import (
    BuyerDNA, Entry, IndependenceReport, SwarmPredictor, WalletIndependence,
    aggregate_smart_flow, build_fingerprint,
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
from src.strategies.age_banded import AgeBandedPredictor
from src.strategies.multihead_predictor import ElogwEngine, MultiHeadPredictor, PredictionFeatures
from src.chains.pump_curve import (
    BondingCurveState, parse_bonding_curve, quote_buy, quote_sell,
)
from src.chains.idl import report as idl_report
from src.chains.pump_route import (
    TOKEN_2022_PROGRAM, TOKEN_PROGRAM, NativePumpRoute, PumpRouteConfig,
)
from src.chains.pumpswap_curve import PumpSwapPoolState
from src.chains.pumpswap_curve import quote_buy as pool_quote_buy
from src.chains.pumpswap_curve import quote_sell as pool_quote_sell
from src.chains.pumpswap_route import PoolState, PumpSwapRoute, PumpSwapRouteConfig, parse_pool
from src.execution.pump_fees import (
    DEFAULT_SCHEDULE as PUMP_FEE_SCHEDULE, VENUE_BONDING_CURVE,
)
from src.execution.tradeability import curve_tradeability, exit_capacity_ratio, pool_tradeability
from src.strategies.distribution import DistributionDetector
from src.strategies.mega_event import MegaEventReserve
from src.strategies.escape import (
    EscapeEstimate, HazardMechanism, LandingLatency, escape_probability,
    hazard_curve_from_probabilities, mechanisms_from_signals,
)
from src.strategies.monster import (
    MonsterEvidence, MonsterState, MonsterStateMachine, hold_versus_exit,
)
from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator
from src.runtime.intelligence_manifest import CoverageTracker, audit as audit_intelligence
from src.strategies.prelaunch_intent import PrelaunchIntentModel
from src.strategies.authenticity import (
    AuthenticityResolver, EntityRegistry, ProofLevel, SourceSignal, load_entities,
)
from src.strategies.reentry import ReentryBook, ReentryPolicy, ReentryVerdict
from src.strategies.source_genealogy import (
    SourceGenealogy, SourcePost, build_source_dna,
)
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
        # How long our sells actually take to land. Constructed here rather
        # than in prediction setup because `_execute_exit` feeds it, and an
        # exit path that depends on a setup step is an exit path that can be
        # unavailable. Fed only by landed, real sells, so a paper run cannot
        # teach the escape race a latency that was never paid.
        self.landing_latency = LandingLatency()
        # Event-driven redecision. Bounded on purpose: dropping a request is
        # survivable because the safety sweep still runs, while an unbounded
        # queue under a trade burst is not. Both drop counters are reported,
        # because a queue that is silently shedding work looks exactly like a
        # quiet market.
        self._redecide: asyncio.Queue = asyncio.Queue(
            maxsize=int(self.global_config.get("redecision_queue_size", 2_048)))
        self._redecide_pending: set = set()
        self._redecision_drops = 0
        self._candidate_drops = 0
        # How often the objective actually owned the decision. A fallback that
        # quietly becomes the main path is the failure this counts: if
        # unpriced cycles dominate, the desk is being run by the threshold
        # policy while the readiness surface says otherwise.
        self._priced_holds = 0
        self._unpriced_cycles = 0
        self._suppressed_monster_banks = 0
        # Facts about a curve that never change and never arrive twice.
        # Bounded against the hot-state token set, because a day of launches
        # would otherwise accumulate every creator we have ever seen.
        self._curve_static: Dict[str, Dict[str, Any]] = {}
        # The forward-shadow ledger. Every audit until now reported the proof
        # "insufficient", which was true and unhelpful: insufficient and not
        # started are the same sentence, and only one of them improves by
        # waiting. This is the thing that makes the number go up, and it
        # persists so a restart does not put it back to zero -- a requirement
        # of five thousand decisions is unreachable by a counter that resets.
        self.forward_evidence = ForwardEvidence(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "forward_evidence.json")
        self._evidence_saved_at = 0.0
        self._redecision_tasks: List[asyncio.Task] = []
        self._safety_task: Optional[asyncio.Task] = None
        self._intelligence_task: Optional[asyncio.Task] = None
        self._source_task: Optional[asyncio.Task] = None
        # Coverage says a module was consulted; this says whether it mattered.
        # A component that is disconnected and one that is connected but inert
        # both look like trades that would have happened anyway, and only one
        # of them is fixed by wiring.
        self.contribution_ledger = ContributionLedger()
        self._market_observed_at: Dict[str, float] = {}
        self._market_observation_cohort: set[str] = set()
        self._market_entry_price: Dict[str, float] = {}
        self._curve_entry_price: Dict[str, float] = {}
        self._latest_stream_mark: Dict[str, Dict[str, float]] = {}
        # Latest bonding-curve reserves decoded straight off the trade
        # stream, so exit capacity is answerable locally in the window a
        # decision actually has, with no RPC round trip.
        self._latest_curve_state: Dict[str, Any] = {}
        # The post-graduation counterpart. Reserves and the fee schedule the
        # pool actually charges, maintained off the PumpSwap event stream;
        # and the decoded Pool account, which is the only source for the
        # vaults, the coin_creator and the mayhem flag.
        self._latest_pool_state: Dict[str, PumpSwapPoolState] = {}
        self._pool_accounts: Dict[str, PoolState] = {}
        self._pool_account_pending: Set[str] = set()
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
        # One brain per launch age. A pooled model trained across every horizon
        # learns the average launch and there is no such thing: at 100ms the
        # holder distribution does not exist yet, and by five minutes the
        # features that mattered at 100ms have been overwritten by their own
        # consequences. The pooled model stays as a LABELLED bridge while the
        # per-band artifacts accumulate; `band_status` on every prediction says
        # which answered, so promotion can require the band's own.
        self.predictor = AgeBandedPredictor(
            os.getenv("MODEL_DIR", "models"),
            allow_pooled_fallback=bool(
                self.global_config.get("allow_pooled_model_fallback", True)),
        )
        loaded = self.predictor.load_latest()
        model_loaded = any(loaded.values())
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
        self.state_sequencer = StateSequencer()
        # Re-entry is a post-exit candidate, not an action an open
        # position can take: `Action.REENTER` is only scorable at zero held
        # fraction, so the book lives outside the position loop and gates
        # the ordinary entry path instead of running a parallel one.
        # Whether a token is the entity it claims to be. The registry ships
        # empty on purpose -- an entity entry asserts that an account or
        # wallet canonically IS a named public figure, and a wrong one makes
        # an impersonator look verified. Until entries are filled in from
        # each entity's own published account, every verdict is DATA_BLOCKED,
        # which is the honest reading of an empty registry: not "nothing is a
        # copycat" but "we cannot tell".
        self._watched_entities = load_entities(
            # Schema and policy in the repository; verified entities on the
            # node that could actually reach their pages.
            self.global_config.get(
                "entities_path", "config/entities.yaml,config/entities.verified.yaml"))
        self.entity_registry = EntityRegistry(self._watched_entities)
        self.authenticity = AuthenticityResolver(self.entity_registry)
        # Which source spoke first, and whether its posts have historically
        # been tradeable or merely a place distributors advertise.
        self.source_genealogy = SourceGenealogy()
        self._source_outcomes: Dict[str, List[Any]] = {}
        # The fee actually charged, versioned by date and market cap, rather
        # than a constant that stops being true on the first of September.
        self.fee_schedule = PUMP_FEE_SCHEDULE
        self.entry_coverage = CoverageTracker("entry")
        self.position_coverage = CoverageTracker("position")
        self.reentry_book = ReentryBook(ReentryPolicy(
            cooldown_seconds=float(self.global_config.get("reentry_cooldown_seconds", 90.0)),
            window_seconds=float(self.global_config.get("reentry_window_seconds", 1800.0)),
            max_reentries=int(self.global_config.get("max_reentries_per_token", 2)),
            min_hazard_improvement=float(
                self.global_config.get("reentry_min_hazard_improvement", 0.25)),
        ), action_policy=self.action_policy)
        # The mesh is constructed from the declared universe. Sources without
        # a production fetcher are reported NO_FETCHER by the registry rather
        # than silently absent -- "we have adapters" is not "those signals
        # reach T0 decisions", and the registry report is what says which.
        declarations = load_declarations(
            # Seed first, then the operator's verified overlay -- which
            # tools/verify_sources.py writes and which wins on any id it
            # names. Endpoints that answered on the trading node beat
            # endpoints that looked plausible in a repository.
            self.global_config.get(
                "source_registry",
                "config/sources.yaml,config/sources.verified.yaml"))
        # Real transports, built from the declarations themselves. This map was
        # empty, so build_sources ran from nothing and reported every source
        # NO_FETCHER -- an adapter library rather than a source of signal.
        # Anything an operator injected before construction is kept and wins,
        # so a bespoke connection is still theirs to supply.
        self.http_client = HttpClient(
            timeout_s=float(self.global_config.get("source_http_timeout", 10.0)))
        built, self.transport_report, self.http_client = build_transports(
            declarations, self.http_client)
        injected = dict(getattr(self, "source_fetchers", {}) or {})
        self.transports: Dict[str, Any] = {**built, **injected}
        self.source_fetchers: Dict[str, Any] = dict(self.transports)
        sources, self.source_registry_report = build_sources(
            declarations, self.source_fetchers)
        self.source_mesh = SourceMesh(
            sources,
            dedupe_window=float(self.global_config.get("source_dedupe_window", 300.0)),
            poll_timeout=float(self.global_config.get("source_poll_timeout", 5.0)))
        # token -> observations naming it, newest last.
        self._source_events: Dict[str, List[Any]] = {}
        # Set properly once the predictor is constructed; until then a
        # decision records that it was priced by no validated model.
        self.model_feature_hash = "untrained"
        self.wallet_independence = WalletIndependence()
        self.buyer_dna = BuyerDNA(
            depth=int(self.global_config.get("buyer_dna_depth", 25)),
            min_corpus=int(self.global_config.get("buyer_dna_min_corpus", 50)))
        self.swarm_predictor = SwarmPredictor(
            skill_threshold=float(self.global_config.get("swarm_skill_threshold", 0.6)),
            independence_threshold=float(
                self.global_config.get("swarm_independence_threshold", 0.5)))
        # Entries per token, kept only for tokens the hot state still holds.
        self._actor_entries: Dict[str, List[Entry]] = {}
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
        # Stamped onto every frozen decision, so a fill can be traced to the
        # exact model that priced it rather than to whatever is loaded now.
        self.model_feature_hash = feature_hash[:16]
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
        # Nothing here is an operator secret any more: the program addresses
        # come from the vendored IDLs and the fee recipients from Pump's own
        # published list. What remains configurable is the pair of token
        # programs, because whether a mint is Token-2022 depends on how it was
        # created and getting that wrong changes every associated token
        # address in the instruction.
        self.pump_route = NativePumpRoute(PumpRouteConfig(
            base_token_program=str(self.global_config.get(
                "pump_base_token_program", TOKEN_2022_PROGRAM)),
            quote_token_program=str(self.global_config.get(
                "pump_quote_token_program", TOKEN_PROGRAM)),
        ))
        self.pumpswap_route = PumpSwapRoute(PumpSwapRouteConfig(
            base_token_program=str(self.global_config.get(
                "pump_base_token_program", TOKEN_2022_PROGRAM)),
            quote_token_program=str(self.global_config.get(
                "pump_quote_token_program", TOKEN_PROGRAM)),
        ))
        self.execution_engine = ExecutionEngine(self.solana_config, self.solana_rpc, self.jupiter, self.jito,
                                                builder, self.counterfactual_lab, dry_run=self.dry_run,
                                                pump_route=self.pump_route,
                                                pumpswap_route=self.pumpswap_route)
        # The desk owns the streamed curve state; the engine reads it through
        # this rather than keeping its own, because two views of the price we
        # are about to trade at is one view too many.
        self.execution_engine.curve_state_provider = self._latest_curve_state.get
        # Graduation should change the venue and nothing else. The same desk
        # state, the same policy and the same signing path continue across it;
        # only the pool these two providers describe is new.
        self.execution_engine.pool_state_provider = self._latest_pool_state.get
        self.execution_engine.pool_account_provider = self._pool_accounts.get
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
        # Four independent loops rather than one clocked sweep. The two that
        # decide -- candidates and redecisions -- are driven by events and
        # never sleep; the two that maintain are driven by the clock and never
        # sit in front of a decision.
        self._main_task = asyncio.create_task(self._candidate_dispatch_loop())
        self._redecision_tasks = [
            asyncio.create_task(self._redecision_loop())
            for _ in range(int(self.global_config.get("redecision_workers", 4)))]
        self._safety_task = asyncio.create_task(self._safety_sweep_loop())
        self._intelligence_task = asyncio.create_task(self._intelligence_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        self._market_task = asyncio.create_task(self._market_observer_loop())
        self._source_task = asyncio.create_task(self._source_consumer_loop())

    async def stop(self):
        self._running = False
        # Producers first: they hold sockets, and cancelling the consumer
        # while producers keep publishing fills a queue nobody drains.
        try:
            await self.source_mesh.stop()
        except Exception as exc:  # pragma: no cover - shutdown only
            logger.warning("source mesh shutdown: %s", exc)
        try:
            await stop_transports(self.transports, self.http_client)
        except Exception as exc:  # pragma: no cover - shutdown only
            logger.warning("transport shutdown: %s", exc)
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        for task in (self._main_task, self._health_task, self._market_task,
                     self._safety_task, self._intelligence_task, self._source_task,
                     *self._redecision_tasks):
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

    async def _candidate_dispatch_loop(self):
        """Dispatch every candidate the instant it is detected.

        This used to be one pass of a 500ms control loop that dequeued a
        single candidate and then slept. On a chain where the first four
        buyers decide the trade, a candidate detected just after a pass began
        waited out the whole cycle before anything looked at it, and a burst
        of ten launches took five seconds to even start evaluating.

        The detection queue already blocks until something arrives, so the
        sleep bought nothing at all: it was latency with no corresponding
        saving. Now the loop awaits the queue directly and dispatches
        immediately, with concurrency bounded by the pipeline cap rather than
        by the clock.
        """
        while self._running:
            try:
                candidate = await self.detection_engine.get_candidate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Candidate dispatch error: %s", exc)
                await asyncio.sleep(0.01)
                continue
            try:
                self._dispatch_candidate(candidate)
            except Exception as exc:
                logger.exception("Candidate dispatch failed: %s", exc)

    async def _redecision_loop(self):
        """Re-decide a position the moment its market changes.

        Open positions were swept on the same 500ms cadence, so a rug
        beginning at t+10ms was acted on at t+510ms at best. Trades that touch
        a position now queue an immediate redecision, coalesced per token so a
        burst of forty trades in one slot produces one decision rather than
        forty -- and one decision is also the correct number, because they all
        describe the same state.
        """
        while self._running:
            try:
                token = await self._redecide.get()
            except asyncio.CancelledError:
                raise
            self._redecide_pending.discard(token)
            position = self.elogw_engine.open_positions.get(token)
            if position is None:
                continue
            try:
                await self._manage_one_position(token, position)
            except Exception as exc:
                logger.exception("Redecision failed for %s: %s", token, exc)
            finally:
                try:
                    position["intelligence"] = self._position_intelligence(token, position)
                    position["coverage"] = self.position_coverage.record(
                        position["intelligence"]).to_dict()
                except Exception as exc:  # pragma: no cover - reporting only
                    logger.warning("position coverage failed for %s: %s", token, exc)

    def request_redecision(self, token: str) -> bool:
        """Ask for one. Coalesced, non-blocking, safe from any callback.

        Called from the stream decode path, which must never await on a
        decision: a handler that blocks is a handler that drops the next
        event, and the next event is the one that matters.
        """
        if not token or token in self._redecide_pending:
            return False
        if token not in self.elogw_engine.open_positions:
            return False
        self._redecide_pending.add(token)
        try:
            self._redecide.put_nowait(token)
        except asyncio.QueueFull:
            # Bounded on purpose. Dropping a redecision request is survivable
            # -- the safety sweep still runs -- while an unbounded queue under
            # a trade burst is not.
            self._redecide_pending.discard(token)
            self._redecision_drops += 1
            return False
        return True

    async def _safety_sweep_loop(self):
        """The backstop, not the decision path.

        A position whose token has stopped trading produces no events, so it
        would otherwise never be re-examined -- and "no trades at all" is
        itself one of the shapes a rug takes. This sweep exists for that case
        and runs at a cadence chosen for cost, not for latency, because
        anything latency-sensitive arrives as an event.
        """
        interval = float(self.global_config.get("safety_sweep_seconds", 1.0))
        while self._running:
            await asyncio.sleep(interval)
            try:
                await self._manage_positions()
            except Exception as exc:
                logger.exception("Safety sweep error: %s", exc)

    async def _intelligence_loop(self):
        """Cluster rebuilds and attribution, off the decision path entirely."""
        while self._running:
            await asyncio.sleep(float(self.global_config.get(
                "intelligence_sweep_seconds", 5.0)))
            try:
                await self._update_intelligence()
            except Exception as exc:
                logger.exception("Intelligence loop error: %s", exc)

    async def _market_observer_loop(self):
        """Keep research quotes off the latency-sensitive decision loop."""
        while self._running:
            try:
                await self._observe_active_markets()
            except Exception as exc:
                logger.exception("Market observer error: %s", exc)
            await asyncio.sleep(float(self.global_config.get("market_observer_sleep_seconds", 0.25)))

    async def _process_new_tokens(self):
        """One candidate, pulled with a short timeout. Retained for tests only.

        The live path is `_candidate_dispatch_loop`, which awaits the queue
        with no timeout. This wrapper stays because it is the shape the tests
        drive, and because a timeout is the right behaviour when a caller
        wants to make exactly one attempt.
        """
        try:
            candidate = await asyncio.wait_for(self.detection_engine.get_candidate(), timeout=0.05)
        except asyncio.TimeoutError:
            return
        self._dispatch_candidate(candidate)

    def _dispatch_candidate(self, candidate: TokenCandidate) -> bool:
        """Start a candidate's pipeline. Synchronous, so dispatch cannot block.

        Returns whether a pipeline was started, which is what makes
        saturation observable rather than merely logged.
        """
        token = candidate.address
        if not token or token in self._candidate_pipelines:
            return False
        if len(self._candidate_pipelines) >= int(self.global_config.get("max_candidate_pipelines", 100)):
            logger.warning("Candidate pipeline saturated; preserving DATA_BLOCKED decision for %s", token)
            self._record_blocked_decision(token, "DATA_BLOCKED_candidate_pipeline_saturated", {})
            self._candidate_drops += 1
            return False
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
        return True

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
        # A token we have already exited is not an ordinary candidate. The
        # cheap gates -- cooldown, permanent bars, whether the hazard we fled
        # has actually receded -- run before the expensive path, because there
        # is no point buying a risk report and a prediction for a token that a
        # 90-second cooldown already excludes.
        hazard_state = self.rug_hazard.get_hazard(token)
        gate = self.reentry_book.admits(
            token, hazard_now=(float(hazard_state.hazard_30s)
                               if hazard_state is not None else None))
        if not gate.admitted:
            self._record_blocked_decision(token, f"reentry_{gate.status.lower()}",
                                          gate.as_dict())
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
        # Hard admissibility, then a size, then Q. NOT `should_trade`, which
        # composes all three plus four economic thresholds -- and those
        # thresholds are opinions about quantities the objective already
        # prices, so running them first meant the threshold decided and Q
        # rubber-stamped. A hurdle on P(2x) and an expected-log-growth
        # objective are two views of one question; only one of them can be
        # the policy.
        admitted, admission = self.elogw_engine.admissible(
            prediction, self.sol_price_usd, liquidity, self.wallet_equity_usd)
        if not admitted:
            trade_info = admission
            should_trade = False
        else:
            trade_info = self.elogw_engine.size_candidate(
                prediction, self.sol_price_usd, liquidity)
            has_room, room = self.elogw_engine.portfolio_room(trade_info)
            if not has_room:
                trade_info = {**trade_info, **room}
                should_trade = False
            else:
                # Q owns the economic question from here.
                should_trade = True
        intelligence = self._entry_intelligence(
            token, candidate, risk, prediction, trade_info, liquidity)
        coverage = self.entry_coverage.record(intelligence)
        decision = {"should_trade": should_trade, "trade_info": trade_info,
                    "authority": "shadow" if self.dry_run else "champion",
                    "intelligence": intelligence, "coverage": coverage.to_dict(),
                    "actor_intelligence": intelligence["actors"],
                    "source_intelligence": intelligence["sources"]}
        # A re-entry that has cleared the ordinary hurdle has still not paid
        # for itself: holding through would have cost nothing, and the round
        # trip costs an exit and an entry. The premium is charged here, on the
        # fresh prediction, so a re-entry can never be admitted on the
        # conviction that existed before the exit.
        # Entry priced on the same objective as every other action. This is
        # now THE economic decision rather than a second opinion on one
        # already taken, so a state it cannot price stops the trade: an
        # unpriceable entry is one whose capacity or escape we failed to
        # measure, and buying into a book we cannot measure our way out of is
        # not a trade, it is a hope.
        if should_trade:
            entry = self._score_entry(token, prediction, liquidity, trade_info)
            decision["entry_action"] = intelligence["entry_action"] = (
                {"status": entry.status, "action": entry.action.value,
                 "q": entry.q, "detail": entry.detail}
                if entry is not None else
                {"status": "DATA_BLOCKED", "reason": "entry not priceable"})
            declined = (entry is None or entry.status != "OK"
                        or entry.action is ActionValue.IGNORE)
            if declined:
                reason = ("action_value_ignore" if entry is not None
                          and entry.status == "OK" else "entry_q_data_blocked")
                self.contribution_ledger.record_gate(GateFlip(
                    gate="entry_action_value", token=token, before=True,
                    after=False, reason=decision["entry_action"].get("detail", "")))
                should_trade = False
                trade_info = {**trade_info, "reason": reason,
                              "entry_detail": decision["entry_action"]}
                decision.update({"should_trade": False, "trade_info": trade_info})
        if should_trade and self.reentry_book.get(token) is not None:
            verdict = self._price_reentry(token, prediction, liquidity, trade_info)
            decision["reentry"] = intelligence["reentry"] = verdict.as_dict()
            self.contribution_ledger.record_gate(GateFlip(
                gate="reentry_premium", token=token, before=True,
                after=verdict.admitted, reason=verdict.detail))
            if not verdict.admitted:
                should_trade = False
                trade_info = {**trade_info, "reason": f"reentry_{verdict.status.lower()}",
                              "reentry_detail": verdict.detail}
                decision.update({"should_trade": False, "trade_info": trade_info})
            else:
                decision["reentry_opportunity"] = (
                    verdict.opportunity.metadata if verdict.opportunity else {})
        if not should_trade and trade_info.get("reason") in CAPACITY_REJECTIONS:
            # The book being full is not evidence this candidate is bad. It is
            # only evidence that capital is currently committed elsewhere, and
            # whether that is the right place for it is a cross-sectional
            # question the per-token hurdle cannot ask.
            contested, trade_info = await self._contest_for_capital(
                token, candidate, prediction, liquidity, trade_info)
            self.contribution_ledger.record_gate(GateFlip(
                gate="capital_contest", token=token, before=should_trade,
                after=contested, reason=str(trade_info.get("reason", ""))))
            should_trade = contested
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
            # The EDGE landing buys, not the position notional. A $500
            # position is not $500 of expected value: E[log W] times the book
            # is what the fill is actually worth, and bidding against the
            # notional overpays for a marginal trade and underpays for a good
            # one -- the two errors that matter, in the two directions that
            # matter.
            expected_edge_usd=max(0.0, float(trade_info.get("elogw", 0.0) or 0.0)
                                  * max(self.wallet_equity_usd, 0.0)),
            sol_price_usd=float(self.sol_price_usd or 0.0),
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
        if self.reentry_book.get(token) is not None:
            # Counted only on a fill, not on admission: the next re-entry into
            # this token has to clear a strictly higher bar, and a bar that
            # rose on rejected attempts would punish the token for our own
            # indecision rather than for having cycled us.
            self.reentry_book.note_reentry(token)
            position["reentry"] = True
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

    def _local_liquidity(self, token: str) -> float:
        """Tradeable depth from the streamed curve, in USD. Zero when unknown.

        A bonding curve's quote-side depth IS its SOL reserve: that is what a
        seller can be paid out of, and no quote from anywhere makes it larger.
        Reading it locally removes a network round trip from directly in front
        of the T0 sizing decision.

        Real reserves are preferred where an account update has supplied them.
        Where only a trade event has been seen, the virtual reserve is used
        and is an upper bound -- which is why the frontier built on it is
        already labelled as one rather than treated as a measurement.
        """
        state = self._latest_curve_state.get(token)
        if state is None or not state.tradeable or self.sol_price_usd <= 0:
            return 0.0
        lamports = int(state.real_sol_reserves or 0) or int(state.virtual_sol_reserves or 0)
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

        # Actor intelligence is computed from live entries rather than from
        # the episode snapshot, so it reaches the decision at the age it was
        # measured. Its status travels with it: a launch with no scored buyers
        # must not read as one whose buyers scored zero.
        actors = self.actor_intelligence(candidate.address, as_of)
        graph_features = {
            **graph_features,
            "actor_status": actors.get("status", "DATA_BLOCKED"),
            "observed_buyers": actors.get("observed_buyers", 0),
        }
        flow = actors.get("smart_flow") or {}
        if flow.get("status") == "OK":
            graph_features["actor_adjusted_flow"] = flow.get("evidence")
            graph_features["sybil_discount"] = flow.get("discount")
        swarm = actors.get("swarm") or {}
        if swarm.get("status") == "OK":
            graph_features["swarm_probability"] = swarm.get("probability")
        elif swarm:
            graph_features["swarm_evidence_uncalibrated"] = swarm.get("evidence")
        dna = actors.get("buyer_dna") or {}
        if dna.get("status") == "OK":
            graph_features["first25_label"] = dna.get("label")
            graph_features["first25_confidence"] = dna.get("confidence")

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
            # Coverage is recorded in `finally` so it reflects what this cycle
            # ACTUALLY consulted, including the paths that exit early. Recording
            # it mid-body would have measured the manifest against a half-filled
            # position and reported modules as orphaned that simply had not been
            # reached yet -- which is the same false alarm as reporting a live
            # module missing, and would have trained an operator to ignore it.
            try:
                await self._manage_one_position(token, position)
            finally:
                try:
                    position["intelligence"] = self._position_intelligence(token, position)
                    position["coverage"] = self.position_coverage.record(
                        position["intelligence"]).to_dict()
                except Exception as exc:  # pragma: no cover - reporting only
                    # Coverage is a report about trading, not a part of it.
                    # Letting it raise from a `finally` would abort the
                    # position loop for every remaining position, which is a
                    # far worse failure than a missing coverage row.
                    logger.warning("position coverage failed for %s: %s", token, exc)

    async def _manage_one_position(self, token: str, position: Dict[str, Any]) -> None:
        """One position, one cycle. Returns early where the cycle is resolved."""
        should_hazard_exit, urgency, pct = self.rug_hazard.should_exit(token, position)
        if should_hazard_exit:
            await self._execute_exit(token, position, pct, f"rug_hazard_{urgency}")
            return
        marked = await self._mark_position(token, position)
        if marked is None:
            return
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
        # ONE bypass, and it is a bypass of the objective rather than a
        # second opinion about it: a catastrophic reading says the position is
        # about to stop being sellable, and a policy that prices forward
        # returns cannot represent "there will be no forward". Everything
        # else the monster machine wants -- including banking a runner -- is
        # an opinion about expected growth, and opinions about expected growth
        # go through Q or they do not happen. Banking used to execute here,
        # BEFORE the action-value engine was consulted, which meant the
        # component that owns the objective was routinely told what had
        # already been done.
        if monster.action == "emergency_exit":
            await self._execute_exit(token, position, 1.0, f"monster_{monster.reason}")
            return
        if monster.action == "bank" and monster.bank_fraction > 0:
            # Recorded rather than executed, and recorded rather than dropped.
            # The reading is real evidence and Q prices the same forward
            # distribution it was derived from, so acting on it here would be
            # counting that evidence twice. If these accumulate while Q keeps
            # choosing HOLD, that is a disagreement worth investigating -- and
            # it is only investigable because it is written down.
            position.setdefault("suppressed_monster_banks", []).append(
                {"reason": monster.reason, "fraction": monster.bank_fraction,
                 "multiple": multiple, "at": time.time()})
            self._suppressed_monster_banks += 1

        # Everything else is priced against one forward distribution. The
        # threshold policy below remains ONLY for states this cannot price --
        # an unmeasured capacity, a missing distribution -- because deleting a
        # working fallback while nothing validated has replaced it is how a
        # desk ends up with no exit rule at all.
        action_decision = self._score_actions(token, position, multiple, distribution)
        position["action_value"] = {
            "status": action_decision.status, "action": action_decision.action.value,
            "q": action_decision.q, "detail": action_decision.detail,
        }
        if action_decision.status == "OK":
            if action_decision.action is not ActionValue.HOLD:
                handled = await self._apply_action(token, position, action_decision, multiple)
                if handled:
                    return
            else:
                # A priced HOLD is a decision, not a gap to be filled. The
                # ratchet used to sell after this and scale-in used to add
                # after it, so a position the objective had just decided to
                # leave alone got traded anyway by two components reasoning
                # from different quantities. If Q says hold, the cycle is over.
                position["ratchet"] = {
                    "status": "NOT_CONSULTED",
                    "reason": "action-value priced this state and chose HOLD"}
                self._priced_holds += 1
                return

        # Only unpriceable states reach here.
        self._unpriced_cycles += 1
        decision = evaluate_exit(
            self.exit_policy, multiple, float(position["high_water_multiple"]), continuation,
            set(stages), time.time() - float(position["entry_time"]),
        )
        position["ratchet"] = ({"status": "OK", "reason": decision[0],
                                "exit_pct": decision[1]} if decision
                               else {"status": "OK", "reason": "no threshold reached"})
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
            return
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

    def _entry_intelligence(self, token: str, candidate: Any, risk: Any,
                            prediction: Any, trade_info: Dict[str, Any],
                            liquidity: float) -> Dict[str, Any]:
        """One slot per module that must be visible in an entry decision.

        Assembled in ONE place on purpose. The four modules that were reported
        wired and were not went missing because the evidence they produce was
        read at their own call sites, so removing a call removed the evidence
        and nothing downstream could tell. A single flat map, checked against a
        declared manifest, makes a disconnected module fail loudly: the slot
        goes MISSING, which is distinguishable from DATA_BLOCKED -- consulted
        and unable to answer -- and only one of the two is a bug.
        """
        hazard = self.rug_hazard.get_hazard(token)
        reentry = self.reentry_book.get(token)
        return {
            "safety": {"status": getattr(risk, "data_status", "DATA_BLOCKED"),
                       "ownership_renounced": bool(getattr(risk, "ownership_renounced", False)),
                       "risk_score": getattr(risk, "risk_score", None)},
            "hazard": ({"status": hazard.data_status, "hazard_30s": hazard.hazard_30s,
                        "hazard_5m": hazard.hazard_5m, "urgency": hazard.exit_urgency,
                        "reason": hazard.blocked_reason}
                       if hazard is not None
                       else {"status": "DATA_BLOCKED", "reason": "token not registered"}),
            "prediction": ({"status": "OK", "p_5x": getattr(prediction, "p_5x", None),
                            "p_10x": getattr(prediction, "p_10x", None),
                            "expected_hold_time": getattr(prediction, "expected_hold_time", None)}
                           if prediction is not None
                           else {"status": "DATA_BLOCKED", "reason": "no prediction"}),
            "actors": self.actor_intelligence(token),
            "sources": self.source_intelligence(token),
            "source_dna": self._source_dna(token),
            "authenticity": self._authenticity(token, candidate),
            "cost_model": self._cost_model(token),
            "prelaunch": (self._prelaunch_context(candidate.deployer or "", candidate.timestamp)
                          or {"status": "DATA_BLOCKED", "reason": "no deployer profile"}),
            "coordination": self.public_coordination.get_features(token),
            "social": self.social_intel.get_token_social_signal(token),
            "information": {"status": "OK", "lead_sequence": len(
                self.info_graph.get_lead_sequence(token))} if self.info_graph else {
                "status": "DATA_BLOCKED", "reason": "information graph not constructed"},
            "opportunity": (self.last_slate_report
                            or {"status": "DATA_BLOCKED", "reason": "no slate ranked yet"}),
            "reentry": (self.reentry_book.admits(token).as_dict() if reentry is not None
                        else {"status": "NOT_A_REENTRY", "detail": "never held"}),
            "mega_event": (self.mega_event_reserve_state
                           or {"status": "DATA_BLOCKED", "reason": "equity not refreshed yet"}),
            "authority": {"status": "OK" if self.champion_challenger.is_live(MODEL_HYPOTHESIS_ID)
                          else "SHADOW", **self.champion_challenger.get_stats()},
            # Seeded here so the slot exists on every decision; overwritten
            # with the real score once the sizing engine has produced a size
            # to price. A candidate the hurdle already refused was genuinely
            # never priced on this axis, and saying so is accurate.
            "entry_action": {"status": "DATA_BLOCKED",
                             "reason": "no size to price; sizing refused first"},
            "sizing": {"status": "OK" if trade_info.get("position_size_sol") else "DATA_BLOCKED",
                       **{key: value for key, value in trade_info.items()
                          if key in ("position_size_sol", "position_value_usd",
                                     "risk_contribution", "reason")}},
        }

    def _position_intelligence(self, token: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """One slot per module that must be visible in an open-position decision."""
        hazard = self.rug_hazard.get_hazard(token)
        return {
            "distribution": position.get("distribution")
            or {"status": "DATA_BLOCKED", "reason": "not read this cycle"},
            "monster": position.get("monster")
            or {"status": "DATA_BLOCKED", "reason": "state machine not consulted"},
            "escape": position.get("escape")
            or {"status": "DATA_BLOCKED", "reason": "escape not estimated"},
            "hazard_mechanisms": position.get("hazard_mechanisms")
            or {"status": "DATA_BLOCKED", "reason": "hazard not decomposed"},
            "exit_latency": position.get("exit_latency")
            or {"status": "DATA_BLOCKED", "reason": "latency not estimated"},
            "exit_capacity": {"status": position.get("exit_capacity_status", "DATA_BLOCKED"),
                              "ratio": position.get("exit_capacity_ratio")},
            "action_value": position.get("action_value")
            or {"status": "DATA_BLOCKED", "reason": "action policy not consulted"},
            "contribution": position.get("contribution")
            or {"status": "DATA_BLOCKED", "reason": "decision not attributed"},
            "ratchet": position.get("ratchet")
            or {"status": "DATA_BLOCKED", "reason": "threshold policy not consulted"},
            "hazard": ({"status": hazard.data_status, "hazard_30s": hazard.hazard_30s,
                        "urgency": hazard.exit_urgency}
                       if hazard is not None
                       else {"status": "DATA_BLOCKED", "reason": "token not registered"}),
        }

    def ingest_curve_account(self, token: str, data: bytes) -> bool:
        """Adopt a full bonding-curve account when the raw bytes are available.

        A trade event carries virtual reserves only, so a state built from one
        is complete enough to price and to build a transaction, but its real
        reserves are unknown -- which is what makes every frontier derived
        from it an upper bound rather than a measurement. An account update
        carries all of it, so when one arrives it replaces the reconstruction
        rather than being merged into it: mixing a measured field into a
        reconstructed record produces a row that is neither, and nothing
        downstream can tell which parts to trust.

        Returns whether the account was adopted, so a caller can distinguish
        "no account available" from "the bytes were not a bonding curve".
        """
        state = parse_bonding_curve(data)
        if state is None or not state.tradeable:
            return False
        self._latest_curve_state[token] = state
        if state.creator:
            self._curve_static[token] = {
                "creator": state.creator,
                "token_total_supply": int(state.token_total_supply or 0),
            }
        self.state_sequencer.bump(token)
        # An account update is a market change like any other, and the
        # position holding it should think again on the same terms.
        self.request_redecision(token)
        return True

    def _seed_pool_state(self, token: str, event: Dict[str, Any]) -> None:
        """Open pool state at migration, from the CreatePoolEvent.

        Reserves and identity only. The fee schedule is deliberately left
        unset: it is read off the first trade that pays it, and until then a
        quote refuses rather than assuming a rate. Assuming one would price
        every graduated coin as though it were cheaper to trade than it is.
        """
        pool = str(event.get("pool", "") or "")
        if not token or not pool:
            return
        existing = self._latest_pool_state.get(token)
        state = PumpSwapPoolState(
            pool=pool, base_mint=str(event.get("base_mint", token) or token),
            quote_mint=str(event.get("quote_mint", "") or ""),
            base_reserves=int(event.get("initial_base_amount", 0) or 0),
            quote_reserves=int(event.get("initial_quote_amount", 0) or 0),
            base_decimals=int(event.get("base_mint_decimals", 0) or 0),
            updated_at=float(event.get("timestamp", time.time())),
            slot=int(event.get("slot", 0) or 0), source="pool_created")
        if existing is not None and existing.pool == pool:
            # A second creation event for a pool we already track carries no
            # newer reserves than the trades we have seen since.
            state.total_fee_bps = existing.total_fee_bps
            state.lp_fee_bps = existing.lp_fee_bps
            state.protocol_fee_bps = existing.protocol_fee_bps
            state.coin_creator_fee_bps = existing.coin_creator_fee_bps
            state.coin_creator = existing.coin_creator
            state.base_supply = existing.base_supply
        self._latest_pool_state[token] = state
        self.state_sequencer.bump(token)
        self._spawn_background(self._fetch_pool_account(token, pool))

    def _update_pool_state(self, token: str, event: Dict[str, Any]) -> None:
        """Adopt post-trade reserves and the fee bps that trade actually paid.

        The PumpSwap trade events carry both, so the pool is priced against
        measurements rather than against a fee table we would otherwise have
        to keep in step with the protocol by hand.
        """
        reserves_base = int(event.get("pool_base_reserves", 0) or 0)
        reserves_quote = int(event.get("pool_quote_reserves", 0) or 0)
        if not token or reserves_base <= 0 or reserves_quote <= 0:
            return
        previous = self._latest_pool_state.get(token)
        pool = str(event.get("pool", "") or (previous.pool if previous else ""))
        if not pool:
            return
        state = PumpSwapPoolState(
            pool=pool,
            base_mint=str(event.get("token", token) or token),
            quote_mint=str(event.get("quote_mint", "")
                           or (previous.quote_mint if previous else "")),
            base_reserves=reserves_base, quote_reserves=reserves_quote,
            base_decimals=(previous.base_decimals if previous else 0),
            updated_at=float(event.get("timestamp", time.time())),
            slot=int(event.get("slot", 0) or 0), source="token_trade")
        if event.get("tail_data_status") == "OK":
            state.lp_fee_bps = int(event.get("lp_fee_bps", 0) or 0)
            state.protocol_fee_bps = int(event.get("protocol_fee_bps", 0) or 0)
            state.coin_creator_fee_bps = int(event.get("coin_creator_fee_bps", 0) or 0)
            state.total_fee_bps = int(event.get("total_fee_bps", 0) or 0)
            state.virtual_quote_reserves = int(event.get("virtual_quote_reserves", 0) or 0)
            state.base_supply = int(event.get("base_supply", 0) or 0)
            state.coin_creator = str(event.get("coin_creator", "") or "")
        elif previous is not None:
            # An event whose tail we could not read still tells us the
            # reserves. Carrying the last measured schedule forward is honest
            # -- it is a measurement, just not of this trade -- where
            # defaulting it to zero would not be.
            state.total_fee_bps = previous.total_fee_bps
            state.lp_fee_bps = previous.lp_fee_bps
            state.protocol_fee_bps = previous.protocol_fee_bps
            state.coin_creator_fee_bps = previous.coin_creator_fee_bps
            state.virtual_quote_reserves = previous.virtual_quote_reserves
            state.base_supply = previous.base_supply
            state.coin_creator = previous.coin_creator
        self._latest_pool_state[token] = state
        # Reserves moved, so a decision priced against the old ones is stale.
        self.state_sequencer.bump(token)
        self.request_redecision(token)

    async def _fetch_pool_account(self, token: str, pool: str) -> bool:
        """Decode the Pool account once, off the hot path.

        The vaults, the coin_creator and the mayhem flag are not in any event
        and cannot be inferred: the mayhem flag alone selects which published
        fee-recipient set the instruction must name, and choosing from the
        wrong one builds a transaction that is well formed and fails. So this
        is fetched at migration, when nothing is waiting on it, and the route
        stays DATA_BLOCKED until it lands.
        """
        if not pool or not self.solana_rpc or pool in self._pool_account_pending:
            return False
        self._pool_account_pending.add(pool)
        try:
            result = await self.solana_rpc.request(
                "getAccountInfo", [pool, {"encoding": "base64", "commitment": "confirmed"}])
            value = (result or {}).get("value") or {}
            encoded = (value.get("data") or [None])[0]
            if not encoded:
                return False
            decoded = parse_pool(base64.b64decode(encoded), pool)
            if not decoded.ok:
                logger.warning("pool %s did not decode: %s", pool, decoded.detail)
                return False
            self._pool_accounts[token] = decoded
            state = self._latest_pool_state.get(token)
            if state is not None and not state.coin_creator:
                state.coin_creator = decoded.coin_creator
            return True
        except (ConnectionError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            logger.warning("pool account fetch failed for %s: %s", pool, exc)
            return False
        finally:
            self._pool_account_pending.discard(pool)

    def pool_route_report(self) -> Dict[str, Any]:
        """Whether graduation actually keeps native execution.

        A route that exists, is wired, and answers DATA_BLOCKED on every
        migrated coin is indistinguishable from one that was never wired --
        except that it looks finished. This is what tells the two apart.
        """
        tracked = len(self._latest_pool_state)
        decoded = len(self._pool_accounts)
        priced = sum(1 for state in self._latest_pool_state.values()
                     if state.blocked_reason() is None)
        executable = sum(1 for token, state in self._latest_pool_state.items()
                         if state.blocked_reason() is None and token in self._pool_accounts)
        return {
            "status": "OK" if executable else "DATA_BLOCKED",
            "pools_tracked": tracked, "accounts_decoded": decoded,
            "pools_priceable": priced, "pools_executable": executable,
            "executable_share": (executable / tracked) if tracked else None,
            "reasons": dict(collections.Counter(
                state.blocked_reason() or "ok"
                for state in self._latest_pool_state.values())),
        }

    def _prune_curve_static(self) -> int:
        """Drop static facts for tokens the hot state no longer tracks."""
        stale = [key for key in self._curve_static
                 if key not in self.hot_state.active_tokens]
        for key in stale:
            self._curve_static.pop(key, None)
        return len(stale)

    def _cost_model(self, token: str, at_utc: Optional[float] = None) -> Dict[str, Any]:
        """The round-trip protocol cost of this token, priced rather than assumed.

        The desk used two config constants for entry and exit cost. They were
        right until 2026-09-01, when Pump replaced the flat 100 bps with a
        market-cap tier schedule -- and a constant that silently stops being
        true is worse than one that was never trusted, because every E[log W]
        downstream keeps quoting it with full confidence.

        Where the schedule can answer, it does. Where it cannot -- the tier
        table is published as an image, so it is only loadable from an
        operator transcription -- the config constant is used AND LABELLED as
        an assumption, so a decision made on an unpriced fee is distinguishable
        after the fact from one made on a measured one.
        """
        at_utc = time.time() if at_utc is None else float(at_utc)
        market_cap = None
        state = self._latest_curve_state.get(token)
        if state is not None and state.virtual_sol_reserves > 0:
            # Market cap in lamports at the marginal curve price: total supply
            # priced off the virtual reserves, which is the quantity the tier
            # table is indexed on.
            market_cap = int(state.token_total_supply * state.virtual_sol_reserves
                             // max(1, state.virtual_token_reserves))
        status, round_trip_bps, detail = self.fee_schedule.round_trip_bps(
            venue=VENUE_BONDING_CURVE,
            entry_market_cap_lamports=market_cap, exit_market_cap_lamports=market_cap,
            entry_utc=at_utc, exit_utc=at_utc,
        )
        if status == "OK":
            leg = round_trip_bps / 2 / 10_000
            return {"status": "OK", "assumed": False, "entry_cost": leg, "exit_cost": leg,
                    "round_trip_bps": round_trip_bps,
                    "schedule_version": detail["entry"].schedule_version,
                    "market_cap_lamports": market_cap}
        return {
            "status": "DATA_BLOCKED_FEE_SCHEDULE", "assumed": True,
            "entry_cost": float(self.global_config.get("assumed_entry_cost", 0.02)),
            "exit_cost": float(self.global_config.get("assumed_exit_cost", 0.02)),
            "reason": detail["entry"].reason or detail["exit"].reason,
            "schedule_version": self.fee_schedule.version,
            "market_cap_lamports": market_cap,
        }

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

    def _score_entry(self, token: str, prediction: Any, liquidity: float,
                     trade_info: Dict[str, Any]):
        """Price IGNORE against PROBE on the same objective as every other action.

        Entry used to be decided by `should_trade` alone -- a per-token hurdle
        reasoning from its own quantities -- while every subsequent action was
        priced by Q. A desk whose entry test and exit policy are different
        objects will, sooner or later, buy something its own exit policy would
        immediately sell, and both components will be individually defensible
        while it happens.

        This does not replace the sizing engine: `should_trade` still chooses
        the size and still owns the hard limits. It adds the question the
        hurdle cannot ask, which is whether committing this size to this
        distribution beats doing nothing on the one axis everything else is
        measured on.
        """
        equity = max(self.wallet_equity_usd, 1e-9)
        size_usd = float(trade_info.get("position_value_usd", 0.0) or 0.0)
        bins = [(probability, gross) for _, probability, gross
                in self.elogw_engine.probability_bins(prediction)]
        if not bins or size_usd <= 0:
            return None
        # The size we would actually hold, quoted off the live curve. This
        # read `trade_info["expected_tokens"]`, which the sizing engine has
        # never populated -- so capacity was measured at zero tokens, came
        # back DATA_BLOCKED, and every entry Q was unpriceable. The old
        # threshold decided every entry while the readiness surface showed an
        # objective that was in fact never consulted.
        state = self._latest_curve_state.get(token)
        if state is None:
            return None
        size_sol = float(trade_info.get("position_size_sol", 0.0) or 0.0)
        if size_sol <= 0:
            return None
        quote = quote_buy(state, int(size_sol * 1e9))
        expected_tokens = int(quote.output_amount or 0)
        if expected_tokens <= 0 or quote.data_status != "OK":
            return None
        probe: Dict[str, Any] = {"size_tokens": expected_tokens}
        capacity_status, capacity = self._exit_capacity(token, probe)
        # Escape estimated at the size this entry would hold, not assumed.
        # `entry_escape_prior` defaulted to 1.0, which told the objective
        # every prospective position was certain to get out -- the single most
        # flattering assumption available, and most wrong on exactly the
        # tokens where escape is hardest.
        escape = self._estimate_escape(token, probe, self.rug_hazard.get_hazard(token))
        costs = self._cost_model(token)
        entry_state = ActionState(
            held_fraction=0.0,
            current_multiple=1.0,
            forward_bins=tuple(bins),
            exit_cost=float(costs.get("exit_cost", 0.02)),
            entry_cost=float(costs.get("entry_cost", 0.02)),
            # An entry whose exit capacity cannot be measured is an entry into
            # something we do not know how to leave. Blocked rather than
            # assumed liquid, which is the flattering direction.
            exit_capacity_ratio=(capacity if str(capacity_status).startswith("OK") else None),
            escape_probability=(float(escape.probability)
                                if escape.status == "OK" else None),
            probe_fraction=min(1.0, size_usd / equity),
        )
        return self.action_policy.score(entry_state)

    def _price_reentry(self, token: str, prediction: Any, liquidity: float,
                       trade_info: Dict[str, Any]):
        """Charge the re-entry premium against a distribution built after the exit.

        Capacity and escape are measured at the size this trade would actually
        hold, not at the size the previous position held. Re-using the old
        measurement would price the new trade on the old book -- and the book
        is exactly what our own exit changed.
        """
        size_sol = float(trade_info.get("position_size_sol", 0.0) or 0.0)
        size_usd = float(trade_info.get("position_value_usd", 0.0) or 0.0)
        equity = max(self.wallet_equity_usd, 1e-9)
        bins = [(probability, gross) for _, probability, gross
                in self.elogw_engine.probability_bins(prediction)]
        if not bins or size_sol <= 0:
            return ReentryVerdict(token=token, status="DATA_BLOCKED",
                                  detail="no sized forward distribution")

        # The size we would hold, quoted off the live curve rather than
        # assumed. Without it there is no honest capacity measurement, and an
        # assumed one is the permissive default this codebase refuses.
        state = self._latest_curve_state.get(token)
        if state is None:
            return ReentryVerdict(token=token, status="DATA_BLOCKED",
                                  detail="no curve state; re-entry capacity unmeasurable")
        quote = quote_buy(state, int(size_sol * 1e9))
        expected_tokens = int(quote.output_amount or 0)
        if expected_tokens <= 0 or quote.data_status != "OK":
            return ReentryVerdict(token=token, status="DATA_BLOCKED",
                                  detail=f"re-entry buy unquotable: "
                                         f"{quote.reason or quote.data_status}")
        probe: Dict[str, Any] = {"size_tokens": expected_tokens}
        capacity_status, capacity = self._exit_capacity(token, probe)
        escape = self._estimate_escape(token, probe, self.rug_hazard.get_hazard(token))
        return self.reentry_book.price(
            token,
            bins=bins,
            size_fraction=min(1.0, size_usd / equity),
            capital_usd=size_usd,
            expected_hold_seconds=float(getattr(prediction, "expected_hold_time", 0.0) or 0.0) or None,
            liquidity_usd=liquidity or None,
            exit_capacity_ratio=(capacity if str(capacity_status).startswith("OK") else None),
            escape_probability=(float(escape.probability)
                                if escape.status == "OK" else None),
            prediction_at=time.time(),
            entry_cost=float(self.global_config.get("assumed_entry_cost", 0.02)),
            exit_cost=float(self.global_config.get("assumed_exit_cost", 0.02)),
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
            # Graduated. The same question, asked of the pool the coin moved
            # into -- because a position that migrates would otherwise be
            # unmeasurable for the rest of its life, and the exit policy
            # cannot tell an absent answer from a cautious one.
            pool = self._latest_pool_state.get(token)
            if pool is None or pool.blocked_reason() is not None:
                return "DATA_BLOCKED_NO_CURVE_STATE", 0.0
            report = pool_tradeability(pool, pool_quote_buy, pool_quote_sell)
            position["tradeability"] = report.report()
            return exit_capacity_ratio(
                int(position.get("size_tokens", 0) or 0), report.exit,
                acceptable_impact=float(self.global_config.get("acceptable_exit_impact", 0.10)),
            )
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
        decision = self.action_policy.score(state)
        # Cheap: a handful of pure evaluations over bins already built. Run on
        # every decision rather than on a sample, because the decisions worth
        # attributing are the rare ones and a sample misses exactly those.
        contribution = action_value_contributions(self.action_policy, state, token=token)
        self.contribution_ledger.record(contribution)
        if contribution is not None:
            position["contribution"] = contribution.to_dict()
        if decision.status == "OK" and decision.action is not ActionValue.HOLD:
            decision.snapshot = self._freeze_decision(
                token, position, decision, state, add_fraction)
        return decision

    def _freeze_decision(self, token: str, position: Dict[str, Any], decision: Any,
                         state: Any, add_fraction: Optional[float]):
        """Turn a chosen action into an immutable, executable object.

        The exact size and the protective limit are carried, never the inputs
        to recompute them: anything recomputable at execution time will be
        recomputed, and then the executed trade is not the decided trade.
        """
        if decision.action is ActionValue.ADD:
            slippage_bps = int(self.global_config.get("scale_in_slippage_bps", 100))
            add_usd = max(self.wallet_equity_usd, 0.0) * float(add_fraction or 0.0)
            size = int(add_usd / max(self.sol_price_usd, 1e-9) * 1e9)
            # The bound the decision was made under, not one derived later
            # from a price that has since moved.
            limit = int(size * (1 + slippage_bps / 10_000))
        else:
            slippage_bps = int(self.global_config.get("exit_slippage_bps", 500))
            held = int(position.get("size_tokens", 0) or 0)
            size = int(held * decision.action.bank_fraction)
            # Minimum acceptable proceeds for the slice being sold, frozen at
            # the marked price rather than re-derived at submit time.
            marked = size * max(0.0, state.current_multiple)
            limit = int(marked * (1 - slippage_bps / 10_000))
        return DecisionSnapshot(
            token=token, action=decision.action.value,
            state_seq=self.state_sequencer.current(token),
            size_base_units=max(0, size), protective_limit=limit,
            feature_hash=state_hash({"forward_bins": list(state.forward_bins),
                                     "held": state.held_fraction,
                                     "multiple": state.current_multiple,
                                     "capacity": state.exit_capacity_ratio,
                                     "escape": state.escape_probability}),
            model_hash=self.model_feature_hash,
            expiry_seconds=float(self.global_config.get("decision_expiry_seconds", 1.5)),
            q_value=float(decision.q),
            evidence={"fraction": add_fraction, "slippage_bps": slippage_bps},
        )

    async def _apply_action(self, token: str, position: Dict[str, Any],
                            decision: Any, multiple: float) -> bool:
        """Execute a chosen action. Returns True when the position is done for this cycle."""
        action = decision.action
        if action is ActionValue.ADD:
            snapshot = getattr(decision, "snapshot", None)
            await self._consider_scale_in(token, position, multiple, snapshot)
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
        failures = await start_transports(self.transports)
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

    def source_intelligence(self, token: str) -> Dict[str, Any]:
        """What public sources have said about this token, and how early.

        Reports the first observation and its lag, because who was first and
        how stale their information already was when it reached us is the
        whole signal. A token nobody mentioned is DATA_BLOCKED, not silent
        agreement that it is uninteresting.
        """
        observations = self._source_events.get(token) or []
        if not observations:
            return {"status": "DATA_BLOCKED",
                    "detail": "no public source has named this token",
                    "mesh_sources": len(self.source_mesh.sources)}
        first = observations[0]
        return {
            "status": "OK",
            "observations": len(observations),
            "first_source": first.source_id,
            "first_source_class": first.source_class.value,
            "first_observation_lag_s": first.observation_lag,
            "repeaters": self.source_mesh.repeaters_of(first.content_hash),
            "languages": sorted({event.language for event in observations
                                 if event.language}),
        }

    def actor_intelligence(self, token: str, as_of: Optional[float] = None) -> Dict[str, Any]:
        """First25 DNA, actor-adjusted flow and forward swarm probability.

        Built from the same entries the independence graph consumes, so the
        three cannot disagree about who bought and in what order. Every field
        is DATA_BLOCKED rather than defaulted: a launch with no scored buyers
        must not read as a launch whose buyers scored zero.
        """
        as_of = time.time() if as_of is None else as_of
        entries = self._actor_entries.get(token) or []
        report = self.independence_report
        intelligence: Dict[str, Any] = {
            "observed_buyers": len(entries),
            "independence_status": report.status,
        }

        if not entries:
            intelligence["status"] = "DATA_BLOCKED"
            intelligence["detail"] = "no scored buyer observed for this token yet"
            return intelligence

        fingerprint = build_fingerprint(token, entries, report,
                                        depth=self.buyer_dna.depth)
        match = self.buyer_dna.match(fingerprint)
        intelligence["buyer_dna"] = {
            "status": match.status, "label": match.label,
            "confidence": match.confidence, "detail": match.detail,
            "depth": fingerprint.depth,
        }

        flow = aggregate_smart_flow(entries, report)
        intelligence["smart_flow"] = {
            "status": flow.status, "evidence": flow.evidence,
            "naive_evidence": flow.naive_evidence, "discount": flow.discount,
            "measured_wallets": flow.measured_wallets,
            "unmeasured_wallets": flow.unmeasured_wallets,
        }

        swarm = self.swarm_predictor.evaluate(entries, report, as_of)
        intelligence["swarm"] = {
            "status": swarm.status, "evidence": swarm.evidence,
            "probability": swarm.probability,
            "independent_skilled": swarm.independent_skilled_so_far,
        }
        intelligence["status"] = "OK"
        return intelligence

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
        route = (self.rug_hazard.observations.get(token) or [])
        sellability_lost = any(item.get("type") == "route" and item.get("feasible") is False
                               for item in list(route)[-20:])
        # The hazard model detects seven mechanisms; the race used to see two,
        # both built from the same aggregate number under different names. The
        # four it dropped included three that speed cannot answer at all, so
        # the mechanisms being ignored were exactly the ones that decide
        # whether running is even the right move.
        risk = position.get("risk_object")
        authority_live = None
        if risk is not None and getattr(risk, "data_status", "") == "OK":
            authority_live = bool(getattr(risk, "can_mint", False)
                                  or getattr(risk, "can_freeze", False))
        decomposition = mechanisms_from_signals(
            getattr(hazard, "signals", ()) or (),
            authority_live=authority_live, sellability_lost=sellability_lost)
        position["hazard_mechanisms"] = decomposition.report()
        mechanisms = dict(decomposition.mechanisms)
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
        else:
            # Escape probability after graduation is measured against the pool
            # for the same reason: how much can be sold before the price moves
            # is the whole question, and it does not stop having an answer
            # because the venue changed.
            pool = self._latest_pool_state.get(token)
            if pool is not None and pool.blocked_reason() is None:
                report = pool_tradeability(pool, pool_quote_buy, pool_quote_sell)
                sellable = report.exit.size_at(
                    float(self.global_config.get("acceptable_exit_impact", 0.10)))
        # Latency measured from our own landed sells, not assumed. A constant
        # is fine right up until the moment it matters -- congestion, a
        # degraded relay, a priority fee that stopped clearing -- and those
        # are precisely the moments a position needs to be out. Below the
        # sample floor the configured value is used AND LABELLED, so an
        # escape priced on an assumption is distinguishable from one priced
        # on evidence.
        latency = self.landing_latency.estimate()
        position["exit_latency"] = latency.report()
        expected_latency_s = (latency.seconds if latency.status == "OK"
                              else float(self.global_config.get("expected_exit_latency_s", 0.4)))
        return escape_probability(
            position_size=int(position.get("size_tokens", 0) or 0),
            sellable_size=sellable,
            expected_latency_s=expected_latency_s,
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
        # A fresh prediction is new state by definition.
        self.state_sequencer.bump(token)
        position["prediction"] = _jsonable(prediction)
        position["prediction_object"] = prediction
        position["prediction_at"] = time.time()
        position["prediction_status"] = "OK"

    async def _consider_scale_in(self, token: str, position: Dict[str, Any],
                                 multiple: float, decision: Any = None):
        """Add to a winner only while the NEXT unit still raises E[log W].

        Committing the whole position at T0 forces sizing before the flow that
        actually separates a launch has arrived. Re-predicting on current
        evidence and adding on the marginal quantity deploys capital as the
        evidence appears, and stops the moment it stops paying.

        When the action policy supplies a `DecisionSnapshot`, the size in that
        snapshot is what executes. Previously the policy scored ADD from one
        `plan_scale_in` result and this method refreshed state and called
        `plan_scale_in` again before submitting -- both calls correct, neither
        of them the decision, and the trade that executed was sized from a
        market state the policy never evaluated. Nothing in the logs showed
        it: the decision and the fill were both recorded, referring to
        different instants.
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

        if decision is not None:
            outcome = decision_guard(decision, self.state_sequencer)
            if not outcome.executed:
                # A refusal is kept rather than swallowed: how often decisions
                # go stale measures how fast state moves relative to how fast
                # we decide, which is a latency number worth having.
                logger.info("SCALE-IN refused for %s: %s", token, outcome.detail)
                self._record_ops_event("stale_decisions", {
                    "token": token, "action": "add", "status": outcome.status.value,
                    "detail": outcome.detail,
                    "decision_id": decision.decision_id})
                return
            decision.consume()
            lamports = int(decision.size_base_units)
            gain = float(decision.q_value)
            fraction = float(decision.evidence.get("fraction", 0.0) or 0.0)
            add_usd = lamports / 1e9 * self.sol_price_usd
            slippage_bps = int(decision.evidence.get("slippage_bps", 100))
        else:
            await self._refresh_portfolio_state()
            fraction, gain = self.elogw_engine.plan_scale_in(
                prediction, float(position["remaining_cost_usd"]), multiple, liquidity,
                portfolio_value=self.wallet_equity_usd,
            )
            if fraction <= 0 or gain <= 0 or self.sol_price_usd <= 0:
                return
            add_usd = self.wallet_equity_usd * fraction
            lamports = int(add_usd / self.sol_price_usd * 1e9)
            slippage_bps = 100

        if lamports <= 0 or self.sol_price_usd <= 0:
            return
        result = await self.execution_engine.execute_swap(
            candidate.base_token or WSOL_MINT, token, lamports,
            slippage_bps=slippage_bps,
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
        # Only landed, non-simulated sells. A paper fill is not evidence about
        # the network, and a submission that never landed has no latency at
        # all -- counting it as its timeout would make a failing relay look
        # merely slow, which is the direction that gets a position trapped.
        self.landing_latency.record(getattr(result, "latency_ms", 0),
                                    landed=bool(getattr(result, "landed", False)),
                                    simulated=bool(result.simulated))
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
        # `reduce_position` has already applied this sale to the position, so
        # what it now holds IS the remainder. Subtracting the sale again here
        # double-counted it: every bank of half the position or more computed
        # a non-positive remainder and was recorded as a FINAL trade outcome
        # while the position was still open -- attributing log growth to
        # capital that had not been returned, and leaving `_closed_pnl` empty
        # so the eventual real close counted the same PnL a second time.
        remaining = int(position["size_tokens"])
        if remaining <= 0:
            # The position is closed, so its outcome is now final and can be
            # attributed. Partial exits are deliberately not recorded here:
            # attributing a trade that is still open would count the same
            # capital twice in the ledger.
            # The position is flat, so re-entry becomes a question that can
            # actually be asked. What is recorded is the REASON, not the price:
            # a token banked at 8x and a token fled at 0.3x are different
            # propositions, and the hazard reading is captured now because a
            # hazard-driven exit can only be cleared later by showing that the
            # same measurement has fallen -- which is impossible if it was
            # never taken.
            hazard_state = self.rug_hazard.get_hazard(token)
            self.reentry_book.record_exit(
                token, reason,
                exit_multiple=float(position.get("high_water_multiple", 1.0)),
                realized_pnl_usd=self._closed_pnl.get(token, 0.0) + pnl,
                hazard_at_exit=(float(hazard_state.hazard_30s)
                                if hazard_state is not None else None),
            )
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
        self._prune_curve_static()
        self._refresh_independence()
        self._publish_attribution()
        if self.dry_run:
            latest_mtime = self._latest_model_mtime()
            if latest_mtime > self._model_artifact_mtime:
                candidate = AgeBandedPredictor(
                    os.getenv("MODEL_DIR", "models"),
                    allow_pooled_fallback=bool(
                        self.global_config.get("allow_pooled_model_fallback", True)))
                if any(candidate.load_latest().values()):
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
            # Recorded before the candidate is dispatched, so the pipeline can
            # already build a transaction on the first trade that arrives.
            if event.get("creator"):
                self._curve_static[token] = {
                    "creator": str(event["creator"]),
                    "token_total_supply": int(event.get("token_total_supply", 0) or 0),
                }
            # Derive the twenty-seven accounts now, while nothing is waiting.
            # They are derivations of constants for this (mint, creator,
            # wallet) and never change, so paying for them at execution time
            # is ~2ms of avoidable work inside the window the whole system
            # exists to win.
            if event.get("creator") and self.execution_engine is not None:
                try:
                    self.pump_route.warm(token, str(event["creator"]),
                                         self.execution_engine.tx_builder.public_key)
                except Exception as exc:  # pragma: no cover - warming is optional
                    logger.debug("account prewarm failed for %s: %s", token, exc)
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
            if event.get("program") == PumpSwapMonitor.PUMP_AMM_PROGRAM:
                self._update_pool_state(token, event)
            curve_price = float(event.get("curve_price_raw", 0) or 0)
            virtual_sol = int(event.get("virtual_sol_reserves") or 0)
            virtual_token = int(event.get("virtual_token_reserves") or 0)
            if virtual_sol > 0 and virtual_token > 0:
                # Reserves moved, so any decision priced against the old ones
                # is stale. Bumping here rather than on a timer is what makes
                # staleness mean "the market changed" instead of "time passed".
                self.state_sequencer.bump(token)
                # Static facts about the curve -- who created it, how large
                # the supply is -- do not arrive on trade events and do not
                # change. Carrying them forward from the creation event is
                # what lets a trade event produce a state complete enough to
                # build a transaction from. Without it the native route
                # refused every trade for want of a creator it had already
                # been told once.
                static = self._curve_static.get(token, {})
                self._latest_curve_state[token] = BondingCurveState(
                    virtual_token_reserves=virtual_token, virtual_sol_reserves=virtual_sol,
                    # The TradeEvent carries no real reserves. Leaving these at
                    # zero is what marks every frontier derived from this state
                    # as an upper bound rather than a measurement.
                    real_token_reserves=0, real_sol_reserves=0,
                    token_total_supply=int(static.get("token_total_supply", 0) or 0),
                    complete=False, creator=str(static.get("creator", "") or ""),
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
            # Our own transactions come back on this same stream. Telling the
            # execution engine the moment one appears is what turns fill
            # reconciliation from a poll into a notification.
            if self.execution_engine is not None and event.get("signature"):
                self.execution_engine.observe_signature(
                    str(event["signature"]), event.get("slot"))
            self._record_actor_entry(token, event, observation)
            self.rug_hazard.record_observation(token, observation)
            # The event that changes a position's world is the one that should
            # make it think again. Coalesced and non-blocking, because a decode
            # handler that awaits a decision is a handler that drops the next
            # event -- and the next event is the one that matters.
            self.request_redecision(token)
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
            self._seed_pool_state(token, event)
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
            "age_bands": self.predictor.report() if self.predictor else {"status": "DATA_BLOCKED"},
            "exit_policy": {"status": self.exit_policy_status, "detail": self.exit_policy_detail,
                            "policy": asdict(self.exit_policy)},
            "equity": {"status": self.equity_status, "wallet_equity_usd": self.wallet_equity_usd,
                       "sol_price_usd": self.sol_price_usd},
            "execution": {"dry_run": self.execution_engine.dry_run if self.execution_engine else True},
            "native_fastpath": NATIVE_FASTPATH_STATUS,
            "native_route": (self.execution_engine.native_route_report()
                             if self.execution_engine else {"status": "DATA_BLOCKED"}),
            "pumpswap_route": self.pumpswap_route.report(),
            "pumpswap_execution": self.pool_route_report(),
            "idl": idl_report(),
            "action_policy": {"trained": self.action_policy.is_trained,
                              "min_edge": self.action_policy.min_edge},
            # An empty registry is not "nothing is a copycat". It is "we
            # cannot tell", and a status page that stays silent about it lets
            # an operator read silence as safety.
            "entity_registry": self.entity_registry.report(),
            "source_mesh": {**self.source_mesh.health(),
                            "registry": self.source_registry_report.to_dict(),
                            # What is wired, what answered, and what could not
                            # be built and why. A declaration with no transport
                            # is a coverage hole; one with a transport that has
                            # never returned a record is a different hole, and
                            # the two need telling apart.
                            "transports": {**self.transport_report.to_dict(),
                                           **transport_report(self.transports)},
                            "transport_start_failures": dict(
                                getattr(self, "transport_start_failures", {}))},
            "actor_graph": {"independence_status": self.independence_report.status,
                            "measured_pairs": self.independence_report.observed_pairs,
                            "scored_wallets": len(self.independence_report.scores)},
            "reentry": self.reentry_book.report(),
            # Distance to the next promotion stage, as ratios. A gate that
            # says FAIL cannot distinguish a week away from a year away, and
            # that difference decides whether to keep running or change
            # something.
            "forward_evidence": self.forward_evidence.report(),
            "regime": self.current_regime,
            # A queue silently shedding work looks exactly like a quiet market,
            # so both drop counters are surfaced rather than only logged.
            "event_loop": {
                "redecision_queued": self._redecide.qsize(),
                "redecision_drops": self._redecision_drops,
                "candidate_drops": self._candidate_drops,
                "candidate_pipelines": len(self._candidate_pipelines),
                "redecision_workers": len(self._redecision_tasks),
            },
            # Whether the objective actually owns the decisions. A fallback
            # that quietly becomes the main path is the failure this catches.
            "action_authority": {
                "priced_holds": self._priced_holds,
                "unpriced_cycles": self._unpriced_cycles,
                "suppressed_monster_banks": self._suppressed_monster_banks,
            },
            "exit_latency": self.landing_latency.estimate().report(),
            "decision_contribution": self.contribution_ledger.report(),
            "wallet_coverage": (self.wallet_intel.coverage_report()
                                if self.wallet_intel else {"status": "DATA_BLOCKED"}),
            # Which declared modules actually reached a decision. A rate that
            # falls to zero between two audit packs means a component was
            # disconnected, and no test will say so.
            "intelligence_coverage": {"entry": self.entry_coverage.report(),
                                      "position": self.position_coverage.report()},
            "authenticity": {"watched_entities": len(self._watched_entities)},
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
        stats = (self.global_research.get_stats() if self.global_research else {}) or {}
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

    def _record_ops_event(self, stream: str, payload: Dict[str, Any]) -> None:
        """Append one operational telemetry row for the monitor and audit pack.

        Deliberately separate from the research lake. The lake is optimised for
        point-in-time correctness and completeness; this is optimised for a
        monitor being able to answer "what is the recent failure rate" in one
        cheap pass without loading episodes. Failures are swallowed for the
        same reason readiness persistence is: telemetry must never be able to
        halt the desk it describes.
        """
        # Trade outcomes are the promotion ledger's only input, and this is
        # the one place every outcome passes through -- entered or declined.
        if stream == "trade_outcomes":
            self._record_forward_evidence(payload)
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
