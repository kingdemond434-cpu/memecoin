"""Memecoin research and execution desk.

The shipped runtime defaults to dry-run. Live transaction submission requires
both an explicit ``--live`` launch and the execution engine's independent
``ALLOW_LIVE_TRADING=yes-i-understand`` acknowledgement.
"""

import argparse
import asyncio
import base64
from collections import deque
import json
import logging
import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from aiohttp import web
from solders.keypair import Keypair

from src.chains.provider_credentials import normalize_provider_environment
from src.chains.rpc_manager import ChainRegistry, RPCManager
from src.chains.yellowstone_grpc import (
    PumpFunMonitor,
    PumpSwapMonitor,
    RaydiumMonitor,
    SolanaRpcProgramStream,
    YellowstoneClient,
)
from src.detection.rug_detector import RugDetector
from src.detection.token_detector import DetectionSource, TokenCandidate, TokenDetectionEngine
from src.execution.jupiter_jito import (
    ExecutionEngine,
    JupiterClient,
    JitoClient,
    PriorityFeeOptimizer,
    USDC_MINT,
)
from src.strategies.risk_veto import RiskVeto
from src.strategies.memecoin_state import (
    DevEvent, DevWalletMonitor, HolderTrajectoryMonitor, ObservedHolderLedger,
    SmartWalletRotationTracker, social_price_disagreement)
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.dataset_builder import (
    PointInTimeDatasetBuilder, PUMP_INITIAL_VIRTUAL_SOL,
    PUMP_INITIAL_VIRTUAL_TOKEN)
from src.research.feature_engine import build_features
from src.research.global_research_miner import GlobalResearchMiner
from src.research.calibration import CalibrationBook
from src.research.counterfactual_corpus import (
    ActionOption, CounterfactualCorpus, RouteOption,
)
from src.research.calibration import Provenance
from src.research.fallback import FallbackResolver, Source
from src.runtime.latency import LatencyLedger
from src.runtime.serialisation import jsonable as _jsonable
from src.runtime.reporting import ReportingSurface
from src.runtime.ingestion import MinedRecordIngestion
from src.runtime.evidence import EvidenceRecording
from src.runtime.supervision import TaskSupervision
from src.runtime.maintenance import DeskMaintenance
from src.runtime.wiring import SubsystemWiring
from src.runtime.source_intelligence import SourceIntelligence
from src.runtime.offload import OffloadedPool, install_fast_event_loop
from src.runtime.sd_notify import SystemdNotifier, watchdog_interval_s
from src.runtime.memory_governor import (
    Band, MemoryGovernor, Relief, DEFAULT_SOFT_FRACTION,
    DEFAULT_HARD_FRACTION)
from src.research.launch_census import LaunchCensus
from src.research import rug_mechanism
from src.research.launch_census import (
    MONSTER_MULTIPLE, rug_mechanism_monster_threshold)
from src.strategies.screen_policy import (
    ScreenPolicy, ScreenReading, Verdict as ScreenVerdict, graded, veto,
)
from src.research.data_miners import DataMinerPool
from src.research.source_catalogue import default_registry
from src.research.telegram_miners import (
    ChannelBook, extract_handles, register_telegram_miners,
)
from src.research.identity_watch import IdentityWatch
from src.research.forward_evidence import ForwardEvidence, Outcome as ForwardOutcome
from src.research.contribution import (
    ContributionLedger, GateFlip, action_value_contributions,
)
from src.strategies.action_value import (
    Action as ActionValue, ActionValuePolicy, Decision as ActionDecision,
    PositionState as ActionState,
)
from src.collectors.event_source import Event
from src.collectors.transports import (
    build_transports,
    start_transports,
    stop_transports,
    transport_report,
)
from src.strategies.decision_snapshot import (
    DecisionSnapshot, StateSequencer, guard as decision_guard, state_hash,
)
from src.strategies.actor_graph import (
    Entry,
    IndependenceReport,
    SwarmPredictor,
    WalletIndependence,
    build_fingerprint,
)
from src.strategies.champion_challenger import ChampionChallengerFramework
from src.strategies.exit_policy import ExitPolicy, evaluate_exit
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.information_graph import (
    AdversarialAdaptationDetector,
    CounterfactualExecutionLab,
    InformationLeadGraph,
    LeadEventType,
)
from src.strategies.age_banded import AgeBandedPredictor
from src.strategies.multihead_predictor import SURVIVAL_LEVELS, ElogwEngine, MultiHeadPredictor, PredictionFeatures
from src.chains.pump_curve import (
    LAMPORTS_PER_SOL, BondingCurveState, parse_bonding_curve, quote_buy, quote_sell,
)
from src.chains.pump_curve import quote_sell as curve_quote_sell
from src.chains.pump_route import TOKEN_PROGRAM, NativePumpRoute, PumpRouteConfig
from src.chains.pumpswap_curve import PumpSwapPoolState
from src.chains.pumpswap_curve import quote_buy as pool_quote_buy
from src.chains.pumpswap_curve import quote_sell as pool_quote_sell
from src.chains.pumpswap_route import PoolState, parse_pool
from src.execution.pump_fees import (
    DEFAULT_SCHEDULE as PUMP_FEE_SCHEDULE, VENUE_BONDING_CURVE,
)
from src.execution.tradeability import curve_tradeability, exit_capacity_ratio, pool_tradeability
from src.strategies.distribution import DistributionDetector
from src.strategies.escape import (
    EscapeEstimate, HazardMechanism, LandingLatency, escape_probability,
    hazard_curve_from_probabilities, mechanisms_from_signals,
)
from src.strategies.monster import (
    MonsterEvidence, MonsterState, MonsterStateMachine, hold_versus_exit,
)
from src.strategies.opportunity_allocator import Opportunity
from src.strategies.prelaunch_intent import PrelaunchIntentModel
from src.strategies.authenticity import EntityRegistry, ProofLevel, SourceSignal, load_entities
from src.strategies.reentry import ReentryVerdict
from src.strategies.source_genealogy import SourcePost, build_source_dna
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.social_intelligence import SocialIntelligenceEngine
from src.strategies.wallet_intelligence import WalletIntelligenceEngine
from src.strategies.wallet_value import FollowOutcome
from src.strategies.t0_kernel import SurvivalInputs
from src.strategies.funder_ancestry import compress_independence
from src.execution.staged_exits import StagedExits
from src.execution.slot_value import SlotValueModel
from src.strategies.disagreement import DisagreementModel, views_from_intelligence
from src.strategies.ignition import IgnitionModel, touches_from_events

logger = logging.getLogger(__name__)

#: How long a token must sit with no new observation before a death
#: verdict is attempted. Below this, "no new trade yet" and "dead" are
#: indistinguishable, and classifying early permanently brands a token that
#: is only mid-launch.
DEATH_CLASSIFICATION_QUIET_S = 300.0

WSOL_MINT = "So11111111111111111111111111111111111111112"
MODEL_HYPOTHESIS_ID = "production_multihead_v1"


# Rejections that mean "capital is committed elsewhere", not "this token is
# bad". Only these may be revisited by a cross-sectional contest -- a safety
# rejection, a rug-risk rejection or the daily-loss kill switch never can be,
# or the allocator would become a way to argue past the risk limits.
CAPACITY_REJECTIONS = frozenset({
    "max_concurrent_positions", "total_exposure_limit", "portfolio_risk_limit",
})




class MemecoinQuantDesk(ReportingSurface, MinedRecordIngestion,
                       DeskMaintenance, TaskSupervision, EvidenceRecording,
                       SubsystemWiring, SourceIntelligence):
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
        # Telegram can replay its configured channels as soon as the client
        # connects, before the PIT dataset finishes constructing. Preserve
        # those signals and drain them once every downstream consumer exists.
        self.trade_evidence: Optional[TradeEvidenceLedger] = None
        self._fatal_task_event = asyncio.Event()
        self._fatal_task_detail = ""
        self._task_health: Dict[str, Dict[str, Any]] = {}
        self._background_failures = 0
        self._pending_social_signals: deque = deque(maxlen=10_000)
        self._pending_social_dropped = 0
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
        # Every launch, not only the ones we had an opinion about. Ratios
        # computed downstream of our own filters cannot see what those filters
        # discarded, and a screen that throws away monsters looks like a clean
        # record from inside.
        # Screens that shrink a position rather than discard the launch. The
        # census measured what hard screens cost in monsters, and the answer
        # was that the losses they prevent are bounded at one position while
        # the gains they forgo are not.
        self.screen_policy = ScreenPolicy(
            size_floor=float(self.global_config.get("screen_size_floor", 0.05)))
        # Facts resolve down a graded ladder instead of blocking. Always an
        # answer, never a forgotten guess.
        self.facts = FallbackResolver()
        self._last_screen: Dict[str, Any] = {}
        # Shed context before the kernel sheds the process. This desk was
        # OOM-killed once already, twelve hours in, taking the accumulated
        # evidence with it -- the worst failure available to a system whose
        # only real bottleneck is evidence, because it deletes rather than
        # degrades.
        self.memory = MemoryGovernor(
            soft_fraction=float(self.global_config.get("memory_soft_fraction", 0.70)),
            hard_fraction=float(self.global_config.get("memory_hard_fraction", 0.85)))
        self.launch_census = LaunchCensus(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "launch_census.json")
        self.launch_census.load()
        # Holder structure without an RPC call. getTokenLargestAccounts is
        # unserved by the free pool (publicnode 403, mainnet-beta 429) and
        # was blocking 41 of 63 launches in a measured sample; every trade
        # already decoded carries a wallet and a signed token delta.
        self.observed_holders = ObservedHolderLedger()
        # Measurements, not parallel trading authorities: each reports into
        # the decision record so the audit can tell a module that ran from
        # one that never did. A contributor with no slot is an orphan.
        self.holder_trajectory = HolderTrajectoryMonitor()
        self.dev_wallet_monitor = DevWalletMonitor()
        self.rotation_tracker = SmartWalletRotationTracker()
        self.risk_veto: RiskVeto = RiskVeto()
        self._census_saved_at = 0.0
        # Are the stated probabilities true? Kelly is exquisitely sensitive to
        # them and nothing has ever checked.
        self.calibration = CalibrationBook(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "calibration.json")
        self.calibration.load()
        self.forward_evidence = ForwardEvidence(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "forward_evidence.json")
        self._evidence_saved_at = 0.0
        self._redecision_tasks: List[asyncio.Task] = []
        self._safety_task: Optional[asyncio.Task] = None
        self._intelligence_task: Optional[asyncio.Task] = None
        self._parity_task: Optional[asyncio.Task] = None
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
        # Followed-wallet trades awaiting their forward result. token ->
        # candidates. This is what turns "we watch smart wallets" into a
        # number: without it the wallet-value model has no evidence and ranks
        # nothing, which is the state the composite score was invented to
        # paper over.
        self._follow_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self._follow_resolved = 0
        self._follow_unresolved = 0
        # Marking provenance. A desk that believes it marks locally and in
        # fact pays a round trip on every decision looks identical from the
        # outside; these are what tell the two apart.
        # Pure and cheap, and both are reached from the exit path -- so they
        # are constructed here rather than in async setup. A desk that can
        # execute an exit but has not run setup is a shape the tests build and
        # the runtime can reach during teardown.
        self.state_sequencer = StateSequencer()
        # Disagreement across the desk's independent views, as a sizing
        # multiplier rather than a veto.
        # Where a narrative is in its life, which is a different question
        # from how good the source was: an excellent source can post into a
        # narrative that never spreads.
        self.ignition = IgnitionModel(
            kol_reach=int(self.global_config.get("kol_reach_threshold", 10_000)),
            horizon_seconds=float(self.global_config.get("ignition_horizon_s", 300.0)))
        self.disagreement = DisagreementModel(
            min_views=int(self.global_config.get("disagreement_min_views", 3)),
            floor=float(self.global_config.get("disagreement_floor", 0.25)),
            sensitivity=float(self.global_config.get("disagreement_sensitivity", 3.0)))
        # Exit instructions built before they are needed. Everything expensive
        # about a sell is independent of the amount, so the escape path should
        # never pay for account derivation at the moment it fires.
        self.staged_exits = StagedExits(
            bound_max_age_s=float(self.global_config.get("staged_bound_max_age_s", 2.0)))
        self._marks_local = 0
        self._marks_router = 0
        # token -> recent (timestamp, multiple). Bounded and short: a token
        # that moved two minutes ago and has been flat since is not decaying
        # now, and a long history would say it was.
        self._mark_history: Dict[str, Any] = {}
        # Market-wide state from the context miner. One row, not per token:
        # what the market was doing belongs to every episode running at the
        # time, and copying it into each one would be the same fact stored a
        # thousand times and updated in none of them.
        self._market_context: Dict[str, Any] = {}
        # Every event type the stream has actually delivered. The cheapest
        # possible answer to "is the denominator empty because nothing is
        # launching, or because creations are not reaching us".
        self._stream_events: Dict[str, int] = {}
        # Chain-wide execution conditions, mined rather than assumed. Empty
        # means unmeasured, which the bidder reads as None and the landing
        # model buckets as unknown -- never as calm.
        self._network_health: Dict[str, Any] = {}
        self._priority_fees: Dict[str, Any] = {}
        # Balance readings for wallets we hold no position against. Kept for
        # the ledger, consumed by nothing right now, and reported as such.
        self._wallet_readings: Dict[str, Any] = {}
        self.data_miners = DataMinerPool()
        self.miner_registration: Dict[str, bool] = {}
        # Global breadth. Built in _setup_research; declared here so an
        # offline desk and a half-constructed one both report them honestly
        # rather than raising on a status call.
        self.substitution = default_registry()
        self.channel_book: Optional[ChannelBook] = None
        self.identity_watch = IdentityWatch()
        # Pools an outside operator saw. `_discovery_misses` counts the ones
        # our own stream never reported, which is the only measurement of how
        # complete the census denominator actually is.
        self._discovered_pools: Dict[str, float] = {}
        self._discovery_misses = 0
        # Launches that claim a public figure, most recent last. Bounded:
        # this is a display and training surface, not a store of record.
        self._identity_claims: Dict[str, Any] = {}
        # Where the milliseconds actually go. Built here rather than in
        # _setup_research because the stream can deliver before research is
        # up, and an event that arrives with no ledger to open a trace on is
        # the one measurement that cannot be recovered later.
        self.latency = LatencyLedger()
        # Every decision, including the launches walked away from. A corpus
        # of trades taken can answer "did we make money" and cannot answer
        # "should we have been there", and the second question is where the
        # returns are.
        self.counterfactual_corpus = CounterfactualCorpus(
            path=str(Path(self.global_config.get("ops_state_dir", "data/state"))
                     / "decision_corpus.jsonl")
            if not self.offline else None)
        self.miner_offload: Optional[OffloadedPool] = None
        self.slot_value = SlotValueModel()
        self._mark_checked_at: Dict[str, float] = {}
        self._mark_checks = 0
        self._mark_checks_blocked = 0
        self._mark_checks_diverged = 0
        self._mark_drift_total = 0.0
        self._mark_divergences: List[Dict[str, Any]] = []
        # Partial-exit PnL accumulated per token, banked into the final
        # outcome row when the position actually closes.
        self._closed_pnl: Dict[str, float] = {}
        self._market_cursor = 0
        self._model_artifact_mtime = 0.0
        # The unit is Type=notify with WatchdogSec. Both are promises to the
        # service manager that nothing in this process was keeping: see
        # src/runtime/sd_notify.py. Constructed here so every phase of
        # startup can report progress into `systemctl status`.
        self.systemd = SystemdNotifier()
        self._watchdog_interval_s = watchdog_interval_s()
        self._watchdog_pings = 0
        self._watchdog_task: Optional[asyncio.Task] = None
        # Non-empty while a startup phase is in flight. The health endpoints
        # read it so a probe that arrives mid-boot gets an honest "starting,
        # on phase X" instead of an exception from a subsystem that does not
        # exist yet.
        self._starting_phase = "constructing"

    async def initialize(self):
        normalize_provider_environment(os.environ)
        with open(self.config_path, encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.global_config = self.config.get("global", {})
        # The governor is constructed before the config is read, so
        # memory_soft_fraction/memory_hard_fraction silently took defaults
        # and had never once applied. Rebound rather than reconstructed:
        # components built in between register reliefs on this object.
        self.memory.soft_fraction = float(
            self.global_config.get("memory_soft_fraction", DEFAULT_SOFT_FRACTION))
        self.memory.hard_fraction = float(
            self.global_config.get("memory_hard_fraction", DEFAULT_HARD_FRACTION))
        self._candidate_semaphore = asyncio.Semaphore(
            int(self.global_config.get("max_candidate_concurrency", 8))
        )
        configured_dry_run = bool(self.global_config.get("dry_run", True))
        self.dry_run = configured_dry_run if self.dry_run_override is None else bool(self.dry_run_override)
        # The health server binds BEFORE the subsystems, not after. Startup
        # restores thousands of wallets, loads model artifacts and resolves
        # dozens of channels, and while it did that the desk answered nothing
        # on its port -- so the one moment an operator most needs to see what
        # is happening was the one moment the desk was invisible. /health and
        # /status now answer from the first second, and say `starting`.
        if not self.offline:
            await self._setup_health_server()
        for phase, step in (
                ("keys", self._setup_keys),
                ("chains", self._setup_chains),
                ("intelligence", self._setup_intelligence),
                ("prediction", self._setup_prediction),
                ("execution", self._setup_execution),
                ("detection and risk", self._setup_detection_and_risk),
                ("research", self._setup_research)):
            self._starting_phase = phase
            self.systemd.status(f"starting: {phase}")
            # Each phase buys its own deadline rather than the unit carrying
            # one timeout big enough for the worst of them. A phase that
            # stops progressing still fails within 90s.
            self.systemd.extend_timeout(90.0)
            began = time.time()
            await step()
            logger.info("STARTUP phase %s ready in %.1fs", phase, time.time() - began)
        self._starting_phase = "social signals"
        await self._flush_pending_social_signals()
        # The stream connects LAST, after every consumer exists. A producer
        # that starts first delivers events into a desk that has nothing
        # wired to handle them, and those launches are lost silently.
        await self._setup_yellowstone()
        await self._refresh_portfolio_state()
        logger.info("Desk initialized: mode=%s live_submission_locked=%s", "DRY_RUN" if self.dry_run else "LIVE",
                    os.getenv("ALLOW_LIVE_TRADING", "").lower() != "yes-i-understand")











    async def start(self):
        if self.offline:
            return
        self._running = True
        # An open follow is an unresolved measurement with a 300s horizon.
        # Discarding it on restart is why no wallet ever reached the
        # 12-outcome ranking bar: 56 open against 9 resolved.
        self.load_follow_candidates()
        await self._setup_health_server()
        # Four independent loops rather than one clocked sweep. The two that
        # decide -- candidates and redecisions -- are driven by events and
        # never sleep; the two that maintain are driven by the clock and never
        # sit in front of a decision.
        self._main_task = self._start_runtime_task(
            "dispatch", self._candidate_dispatch_loop())
        self._redecision_tasks = [
            self._start_runtime_task(f"redecision_{index}", self._redecision_loop())
            for index in range(int(self.global_config.get("redecision_workers", 4)))]
        self._safety_task = self._start_runtime_task(
            "safety", self._safety_sweep_loop())
        self._intelligence_task = self._start_runtime_task(
            "intelligence", self._intelligence_loop())
        self._parity_task = self._start_runtime_task("parity", self._parity_loop())
        self._register_memory_reliefs()
        self._health_task = self._start_runtime_task(
            "health", self._health_loop())
        self._market_task = self._start_runtime_task(
            "market", self._market_observer_loop())
        self._source_task = self._start_runtime_task(
            "sources", self._source_consumer_loop())
        # Every loop is scheduled and the port is bound: this is what
        # `Type=notify` has been waiting to hear since the directive was
        # added. Without it `systemctl start` blocks until TimeoutStartSec,
        # the unit reports `activating` for its whole life, and anything
        # ordered After= this unit waits on a readiness that never arrives.
        self._starting_phase = ""
        self.systemd.ready(
            f"desk running: {'DRY_RUN' if self.dry_run else 'LIVE'}, "
            f"{len(self.transports)} transports")
        if self._watchdog_interval_s:
            self._watchdog_task = self._start_runtime_task(
                "watchdog", self._systemd_watchdog_loop())
            logger.info("systemd watchdog ping every %.0fs",
                        self._watchdog_interval_s)

    async def _systemd_watchdog_loop(self):
        """Ping WatchdogSec, on its own task, at a third of the interval.

        Its own task rather than a line in _health_loop on purpose. That loop
        also computes readiness and writes it to disk, and a readiness pass
        that blocks for longer than the watchdog interval would have systemd
        kill a process whose only fault was that reporting took too long --
        the failure mode dressed as the thing it was meant to detect.

        This does not make a hang undetectable: the ping runs on the same
        event loop as the decision path, so a blocked loop still misses it
        and systemd still acts. It only stops slow REPORTING from counting
        as a hang.
        """
        interval = float(self._watchdog_interval_s or 0.0)
        if interval <= 0:
            return
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                break
            if self.systemd.watchdog():
                self._watchdog_pings += 1

    async def stop(self):
        self._running = False
        # Announced before the teardown, not after: shutdown flushes ledgers
        # and can take seconds, and systemd counts that against
        # TimeoutStopSec unless it has been told the exit is deliberate.
        self.systemd.stopping("shutting down")
        # Producers first: they hold sockets, and cancelling the consumer
        # while producers keep publishing fills a queue nobody drains.
        try:
            await self.source_mesh.stop()
        except Exception as exc:  # pragma: no cover - shutdown only
            logger.warning("source mesh shutdown: %s", exc)
        try:
            if self.miner_offload is not None:
                await self.miner_offload.stop()
            else:
                await self.data_miners.stop()
        except Exception as exc:  # pragma: no cover - shutdown only
            logger.warning("data miner shutdown: %s", exc)
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
                     self._parity_task, self._watchdog_task,
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
        self._flush_ledgers()

    def _flush_ledgers(self) -> None:
        """Persist the three ledgers a restart must not lose.

        They are written on a cadence during the run -- an fsync per outcome
        is latency the decision path does not need to pay -- which means a
        shutdown that does not flush discards up to the last interval. On a
        planned restart that is a minute of evidence thrown away for no
        reason, and evidence is the one thing here that cannot be regenerated.

        Failures are logged and swallowed: a desk that cannot shut down
        because a disk is full is worse than one that loses a minute.
        """
        # The corpus is append-only and buffered, so a shutdown that does not
        # flush it loses decisions that cannot be reconstructed from anything.
        try:
            self.counterfactual_corpus.flush()
        except Exception as exc:
            logger.warning("could not flush the decision corpus: %s", exc)
        for name, ledger in (("forward evidence", self.forward_evidence),
                             ("launch census", self.launch_census),
                             ("calibration", self.calibration)):
            try:
                ledger.save()
            except Exception as exc:
                logger.warning("could not flush %s on shutdown: %s", name, exc)
        engine = getattr(self, "execution_engine", None)
        if engine is not None:
            try:
                engine.landing_model.close()
            except Exception as exc:
                logger.warning("could not close the landing log: %s", exc)





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
            # The market moved, so the ladder's protective bounds were priced
            # against a state that no longer holds. Refreshed BEFORE the
            # decision, so that if this redecision chooses to leave, the rung
            # it reaches for is already current.
            try:
                self._reprice_staged_exits(token)
            except Exception as exc:
                logger.debug("staged exit reprice failed for %s: %s", token, exc)
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

    async def _parity_loop(self):
        """Verify promoted Rust decisions against Python, off the money path.

        Its own loop rather than a line in the intelligence sweep, and on a
        much shorter clock: this is the only thing standing between a Rust
        kernel that has quietly started disagreeing and a session of trades
        nobody checked. The queue is bounded, so falling behind loses checks
        rather than memory -- and `parity_dropped` in /status is what makes
        that loss visible instead of silent.
        """
        while self._running:
            await asyncio.sleep(float(self.global_config.get(
                "parity_sweep_seconds", 0.5)))
            try:
                self.t0_kernel.drain_parity(
                    budget=int(self.global_config.get("parity_budget", 64)))
            except Exception as exc:
                logger.exception("Parity loop error: %s", exc)

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
        self.latency.mark(token, "decode_to_dispatch")
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
        self.latency.mark(token, "dispatch_to_decide")
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
        # Safety is graded, not binary. HONEYPOT, RUGGED and CRITICAL are
        # untradeable at any size and stay vetoes; HIGH is a worse token, not
        # an impossible one, and a hard reject on it discards the launches
        # whose upside pays for the ones it prevents.
        screen = self._screen_entry(token, risk)
        if screen.rejected:
            self._record_blocked_decision(token, screen.census_reason, risk_data)
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
            # How much the desk's independent views of this launch scatter.
            # A contested launch is a smaller position, not a rejected one --
            # and a launch every view calls good gets full size, which is the
            # ordering a majority vote gets exactly backwards.
            disagreement = self._read_disagreement(token, candidate, prediction,
                                                   liquidity)
            trade_info = self.elogw_engine.size_candidate(
                prediction, self.sol_price_usd, liquidity,
                disagreement=disagreement)
            # The screens' verdict, applied as size rather than as a gate.
            # Composed multiplicatively with the disagreement shrink already
            # in trade_info: two independent reasons to be smaller are two
            # reasons, not the larger of the two.
            trade_info = self._apply_screen_size(trade_info, screen)
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
        # Frozen BEFORE the action, with every alternative the policy priced.
        # A row amended afterwards leaks the future into its own features, and
        # a model trained on that looks extraordinary in backtest and fails
        # forward.
        self._record_corpus_decision(token, decision_id, decision, trade_info,
                                     should_trade)
        self._record_trade_evidence_packet(
            token, candidate, risk, liquidity, trade_info, intelligence,
            decision="ENTER" if should_trade else str(trade_info.get("reason", "IGNORE")),
            veto=hard_veto.to_dict())
        if not should_trade:
            # A launch we screened is a closed trace, not an abandoned one.
            # Screening is most of the funnel, and a ledger that only ever saw
            # the trades would report the latency of the easy cases.
            self.latency.close(token, "screened")
            return
        self.latency.mark(token, "decide_to_build")
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
            slot_value=self._entry_slot_value(
                token, max(0.0, float(trade_info.get("elogw", 0.0) or 0.0)
                           * max(self.wallet_equity_usd, 0.0))),
            sol_price_usd=float(self.sol_price_usd or 0.0),
        )
        # Build, sign and submission happen inside the engine, which reports
        # its own split; from here the whole call is one stage. Marked before
        # the result is inspected so a rejected submission is timed the same
        # as an accepted one -- the failure path is the one that has to be
        # fast too, because a failed entry is a slot spent.
        self.latency.mark(token, "sign_to_submit")
        self.latency.close(token, "entered" if result.success else "submit_failed")
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
        staged = self._stage_exits(token, position)
        if staged is not None:
            position["staged_exits"] = staged.detail
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

    #: Risk levels that are untradeable at any size, as opposed to merely
    #: worse. Enumerated rather than implied, so the line between a veto and a
    #: discount is visible in one place.
    VETO_RISK_LEVELS = frozenset({"honeypot", "rugged", "critical"})

    #: How much of full size each surviving risk level justifies. HIGH is a
    #: quarter position, not a rejection: the launches a hard reject on HIGH
    #: discards include the ones whose upside pays for the rest.
    RISK_LEVEL_SIZE = {"safe": 1.0, "low": 1.0, "medium": 0.6, "high": 0.25}

    def _screen_entry(self, token: str, risk: Any):
        """Grade this launch's screens into a size, or decline with a reason.

        Everything the desk knows against a launch composes into one
        multiplier. Only genuine impossibilities veto.
        """
        readings: List[ScreenReading] = []
        level = str(getattr(getattr(risk, "risk_level", None), "value", "") or "")
        if level in self.VETO_RISK_LEVELS:
            readings.append(veto("safety", f"risk level {level}"))
        elif level:
            readings.append(ScreenReading(
                name="safety", multiplier=self.RISK_LEVEL_SIZE.get(level, 0.5),
                reason=f"risk level {level}"))
        if getattr(risk, "data_status", "") == "DATA_BLOCKED":
            # Unmeasured safety is not safe. Previously this rejected
            # outright, which threw away every launch too young to have been
            # checked -- which is most of them, at the moment that matters.
            readings.append(ScreenReading(
                name="safety_unmeasured", multiplier=0.35,
                reason="safety checks could not complete in time"))
        hazard = self.rug_hazard.get_hazard(token)
        p_rug = getattr(hazard, "p_rug_5m", None) if hazard else None
        readings.append(graded(
            "rug_hazard", p_rug, benign=0.05, severe=0.60,
            confidence=1.0 if p_rug is not None else 0.0))
        outcome = self.screen_policy.evaluate(readings)
        self._last_screen[token] = outcome
        return outcome

    def _apply_screen_size(self, trade_info: Dict[str, Any],
                           screen: Any) -> Dict[str, Any]:
        """Scale a sized position by the screens' composed multiplier."""
        multiplier = float(getattr(screen, "size_multiplier", 1.0) or 0.0)
        if multiplier >= 0.999:
            return trade_info
        scaled = dict(trade_info)
        for field in ("position_size_sol", "position_value_usd",
                      "risk_contribution"):
            if scaled.get(field) is not None:
                try:
                    scaled[field] = float(scaled[field]) * multiplier
                except (TypeError, ValueError):
                    continue
        scaled["screen_multiplier"] = multiplier
        scaled["screen_detail"] = getattr(screen, "reason", "")
        return scaled

    def _mark_feasible_high_water(self, position: Dict[str, Any],
                                  multiple: float) -> None:
        """Raise the best mark this position could actually have sold at.

        This is the distinction the whole training corpus rests on. A token's
        chart high is what it PRINTED; the feasible high is what our size
        could have exited into, after the venue's capacity and after the cost
        of getting out. On a thin curve those differ by an order of magnitude,
        and a model fitted to the first learns to predict prices that were
        never available to anyone.

        Only marks taken while capacity was MEASURED count. An unmeasured
        moment raises nothing -- not the chart value, not a guess -- because
        reading unknown capacity as full capacity is the flattering direction
        and it flatters hardest on exactly the illiquid tokens where the gap
        is widest.
        """
        status = str(position.get("exit_capacity_status", "") or "")
        ratio = position.get("exit_capacity_ratio")
        if not status.startswith("OK") or ratio is None:
            return
        try:
            capacity = min(1.0, max(0.0, float(ratio)))
            cost = min(1.0, max(0.0, float(position.get("exit_cost", 0.0) or 0.0)))
        except (TypeError, ValueError):
            return
        feasible = float(multiple) * capacity * (1.0 - cost)
        position["feasible_marks"] = int(position.get("feasible_marks", 0)) + 1
        current = position.get("feasible_high_water_multiple")
        position["feasible_high_water_multiple"] = (
            feasible if current is None else max(float(current), feasible))

    def _record_blocked_decision(self, token: str, reason: str, evidence: Dict[str, Any]):
        # Recorded so the weekly audit can ask what the rejected launches went
        # on to do. A missed monster is invisible unless the rejection was
        # written down next to the outcome.
        self.launch_census.screen(token, reason)
        self._record_ops_event("trade_outcomes", {
            "token": token, "entered": False, "attempted": False,
            "rejection_reason": reason,
        })
        decision_id = self.counterfactual_lab.record_decision(
            token, evidence, {"should_trade": False, "reason": reason})
        # The row that makes the corpus a corpus. A rejection recorded here
        # and resolved from the census later is the only way the desk can
        # ever learn that a screen was wrong -- a missed hundred-x costs
        # nothing that any other ledger records.
        try:
            self.counterfactual_corpus.record(
                decision_id or f"blocked:{token}:{time.time():.6f}", token,
                state=_jsonable(evidence), chosen_action="ignore",
                screen_reason=reason, regime=str(self.current_regime or "unknown"),
                options=[ActionOption(action="ignore", status="OK")])
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("corpus row for %s not recorded: %s", token, exc)

    def _record_corpus_decision(self, token: str, decision_id: str,
                                decision: Dict[str, Any],
                                trade_info: Optional[Dict[str, Any]],
                                should_trade: bool) -> None:
        """One frozen row: the state, every feasible action, and what was taken.

        The action set is stored with each option's STATUS rather than with a
        sentinel Q for the ones that were unavailable. An action the state
        could not support -- ADD with no capital, REENTER on an open position
        -- is infeasible, not terrible, and a large negative number would
        teach the model the difference backwards.
        """
        corpus = getattr(self, "counterfactual_corpus", None)
        if corpus is None:
            return
        info = trade_info or {}
        scores = (decision.get("action_scores") or decision.get("scores") or [])
        options = []
        for score in scores:
            if isinstance(score, dict):
                options.append(ActionOption(
                    action=str(score.get("action", "")),
                    q=score.get("q"),
                    status=str(score.get("status", "OK"))))
        routes = []
        router = getattr(self.execution_engine, "landing_router", None)
        if router is not None:
            try:
                for route in router.enabled_routes():
                    routes.append(RouteOption(name=route.name, kind=route.kind))
            except Exception:  # pragma: no cover - defensive
                routes = []
        try:
            corpus.record(
                decision_id or f"decision:{token}:{time.time():.6f}", token,
                state=_jsonable(decision.get("prediction") or {}),
                options=options,
                chosen_action=str(decision.get("action")
                                  or ("enter" if should_trade else "ignore")),
                chosen_q=info.get("elogw"),
                size_fraction=info.get("position_fraction"),
                entry_price=info.get("entry_price"),
                routes=routes,
                screen_reason=("" if should_trade
                               else str(decision.get("reason", ""))),
                regime=str(self.current_regime or "unknown"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("corpus decision for %s not recorded: %s", token, exc)



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
        # Consult the safety modules before anything else, and RECORD what
        # they said. A module that runs but never writes its finding into the
        # position is an orphan: the decision cannot cite it and the audit
        # cannot tell it apart from one that never ran.
        dev_monitor = getattr(self, "dev_wallet_monitor", None)
        holder_monitor = getattr(self, "holder_trajectory", None)
        dev_state = (dev_monitor.state(token) if dev_monitor else {
            "status": "DATA_BLOCKED", "detail": "developer monitor unavailable"})
        position["dev_wallet"] = dev_state
        position["holder_trajectory"] = (holder_monitor.state(token) if holder_monitor else {
            "status": "DATA_BLOCKED", "detail": "holder monitor unavailable"})
        risk = position.get("risk_object")
        veto_engine = getattr(self, "risk_veto", None)
        if risk is not None and veto_engine is not None:
            veto = veto_engine.evaluate(
                risk, dev_state=dev_state,
                connected_holder_pct=getattr(risk, "connected_cluster_pct", None))
            position["risk_veto"] = veto.to_dict()
            # A non-negotiable safety fact discovered while HOLDING is still
            # non-negotiable. Exiting on it is the whole point of re-running
            # the veto against an open position rather than only at entry.
            if veto.status == "VETO":
                await self._execute_exit(
                    token, position, 1.0,
                    "hard_safety_veto:" + ",".join(veto.reasons))
                return
        should_hazard_exit, urgency, pct = self.rug_hazard.should_exit(token, position)
        if should_hazard_exit:
            await self._execute_exit(token, position, pct, f"rug_hazard_{urgency}")
            return
        marked = await self._mark_position(token, position)
        if marked is None:
            return
        multiple, current_value = marked
        position["high_water_multiple"] = max(float(position.get("high_water_multiple", 1)), multiple)
        # The chart high above, and separately the best mark this position
        # could actually have SOLD at. They are not the same number, and
        # training on the first is how a model learns to predict peaks nobody
        # could fill.
        marker = getattr(self, "_mark_feasible_high_water", None)
        if marker is not None:
            marker(position, multiple)
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
        observations = list(self.rug_hazard.observations.get(token, ()))
        social_disagreement = social_price_disagreement(
            [item for item in observations if item.get("type") == "social"],
            [item for item in observations if item.get("type") in {"trade", "market"}],
        )
        dev_state = self.dev_wallet_monitor.state(token)
        veto = self.risk_veto.evaluate(
            risk, dev_state=dev_state,
            position_value_usd=(float(trade_info["position_value_usd"])
                                if trade_info.get("position_value_usd") is not None else None),
            liquidity_usd=liquidity if liquidity > 0 else None,
            connected_holder_pct=getattr(risk, "connected_cluster_pct", None))
        return {
            "safety": {"status": getattr(risk, "data_status", "DATA_BLOCKED"),
                       "ownership_renounced": bool(getattr(risk, "ownership_renounced", False)),
                       "risk_score": getattr(risk, "score", None)},
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
            "risk_veto": veto.to_dict(),
            "holder_trajectory": self.holder_trajectory.state(token),
            "dev_wallet": dev_state,
            "capital_rotation": self.rotation_tracker.report(),
            "social_price_disagreement": social_disagreement,
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
        holder = getattr(self, "holder_trajectory", None)
        developer = getattr(self, "dev_wallet_monitor", None)
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
            "holder_trajectory": (holder.state(token) if holder else {
                "status": "DATA_BLOCKED", "reason": "holder monitor unavailable"}),
            "dev_wallet": (developer.state(token) if developer else {
                "status": "DATA_BLOCKED", "reason": "developer monitor unavailable"}),
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


    def _follow_quote(self, token: str, lamports: int):
        """What a buy of ``lamports`` would get us right now, on either venue.

        The same local quoting the execution path uses, so what is measured is
        what could have been executed -- fees and price impact included, at
        the size the desk would actually have taken.
        """
        curve = self._latest_curve_state.get(token)
        if curve is not None:
            quote = quote_buy(curve, lamports)
            return quote if quote.data_status == "OK" else None
        pool = self._latest_pool_state.get(token)
        if pool is not None and pool.blocked_reason() is None:
            quote = pool_quote_buy(pool, lamports)
            return quote if quote.data_status == "OK" else None
        return None

    def _follow_exit_quote(self, token: str, size_tokens: int):
        """Proceeds of selling the whole followed position now.

        Selling the position through the curve prices its own impact, so
        there is no separate capacity fudge: this IS what we would receive.
        """
        curve = self._latest_curve_state.get(token)
        if curve is not None:
            quote = quote_sell(curve, size_tokens)
            return quote if quote.data_status == "OK" else None
        pool = self._latest_pool_state.get(token)
        if pool is not None and pool.blocked_reason() is None:
            quote = pool_quote_sell(pool, size_tokens)
            return quote if quote.data_status == "OK" else None
        return None

    def _open_follow_candidate(self, token: str, event: Dict[str, Any]) -> bool:
        """Record what following this wallet's buy would have bought us.

        Priced AFTER their trade, which is the point: a wallet whose edge is
        gone by the time its transaction reaches us has no edge we can take,
        and measuring from their fill instead of ours is what makes such a
        wallet look profitable to follow.
        """
        wallet = str(event.get("wallet", "") or "")
        if not wallet or event.get("side") != "buy":
            return False
        if not self.wallet_intel.is_watched(wallet):
            return False
        pending = self._follow_candidates.setdefault(token, [])
        if any(item["wallet"] == wallet for item in pending):
            # One open follow per wallet per token. A wallet adding to a
            # position is one decision to follow, not three.
            return False
        reference = int(float(self.global_config.get("follow_reference_sol", 0.5))
                        * LAMPORTS_PER_SOL)
        quote = self._follow_quote(token, reference)
        if quote is None or quote.output_amount <= 0:
            return False
        pending.append({
            "wallet": wallet, "token": token,
            "observed_at": float(event.get("timestamp", time.time())),
            "opened_at": time.time(),
            "cost_lamports": reference, "size_tokens": int(quote.output_amount),
            "regime": str(event.get("regime", "") or ""),
        })
        # Bounded: a token nobody resolves must not accumulate follows for
        # ever, and the oldest are the ones the horizon will retire first.
        if len(pending) > 64:
            del pending[:-64]
        return True

    def _resolve_follow_candidates(self, now: Optional[float] = None) -> int:
        """Close out follows that have reached their horizon.

        The result is the executable multiple: proceeds of selling the whole
        followed position, over what it cost, both quoted locally. A token
        whose state has gone away entirely resolves as a total loss -- because
        for a position we could not have quoted an exit for, that is what it
        was.
        """
        now = time.time() if now is None else now
        horizon = float(self.global_config.get("follow_horizon_seconds", 300.0))
        resolved = 0
        for token, pending in list(self._follow_candidates.items()):
            keep: List[Dict[str, Any]] = []
            for candidate in pending:
                if now - candidate["opened_at"] < horizon:
                    keep.append(candidate)
                    continue
                exit_quote = self._follow_exit_quote(token, candidate["size_tokens"])
                hazard = self.rug_hazard.get_hazard(token)
                rugged = exit_quote is None
                proceeds = float(exit_quote.output_amount) if exit_quote else 0.0
                multiple = proceeds / max(1.0, float(candidate["cost_lamports"]))
                accepted = self.wallet_intel.record_follow_outcome(FollowOutcome(
                    wallet=candidate["wallet"], token=token,
                    observed_at=candidate["observed_at"],
                    executable_multiple=multiple, regime=candidate["regime"],
                    rugged=bool(rugged),
                    follow_latency_s=max(0.0, candidate["opened_at"]
                                         - candidate["observed_at"]),
                    data_status="OK"))
                resolved += int(accepted)
                self._follow_resolved += int(accepted)
                self._follow_unresolved += int(not accepted)
                if hazard is not None:
                    # Recorded on the observation, not folded into the score:
                    # the multiple already contains what the rug did.
                    candidate["hazard_at_exit"] = hazard.data_status
            if keep:
                self._follow_candidates[token] = keep
            else:
                self._follow_candidates.pop(token, None)
        return resolved


    def _prune_curve_static(self) -> int:
        """Drop static facts for tokens the hot state no longer tracks.

        Also prunes the latest curve and pool state on the same rule. Those
        two were unbounded dicts keyed by mint -- they grew with every launch
        the stream reported and nothing ever removed an entry, which is the
        shape of leak that OOM-killed this service twelve times in one hour.
        hot_state.active_tokens is the right boundary because it is itself
        hard-capped and age-expiring, so "still active there" already means
        "recent enough to be worth pricing".

        An open position is kept whatever the hot state says: a position we
        cannot quote an exit for is the one state we must never discard.
        """
        held = set(self.elogw_engine.open_positions)
        dropped = 0
        for store in (self._curve_static, self._latest_curve_state,
                      self._latest_pool_state):
            stale = [key for key in store
                     if key not in self.hot_state.active_tokens and key not in held]
            for key in stale:
                store.pop(key, None)
            dropped += len(stale)
        return dropped

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
        return self.t0_kernel.score(
            entry_state, survival=self._survival_inputs(prediction),
            age_seconds=float(trade_info.get("time_since_launch", 0.0) or 0.0),
            virtual_sol=int(getattr(state, "virtual_sol_reserves", 0) or 0),
            virtual_token=int(getattr(state, "virtual_token_reserves", 0) or 0))

    @staticmethod
    def _survival_inputs(prediction: Any) -> Optional[SurvivalInputs]:
        """The raw distribution the Rust kernel derives its own bins from.

        Raw levels, not the Python bins: handing the kernel the bins Python
        already built would compare Python's arithmetic against itself and
        report a parity it had not established.
        """
        try:
            levels = [float(getattr(prediction, target.value))
                      for target, _multiple in SURVIVAL_LEVELS]
        except (AttributeError, TypeError, ValueError):
            return None
        return SurvivalInputs(
            levels=levels,
            p_rug_30s=float(getattr(prediction, "p_rug_30s", 0.0) or 0.0),
            p_rug_5m=float(getattr(prediction, "p_rug_5m", 0.0) or 0.0),
            expected_feasible_multiple=float(
                getattr(prediction, "expected_feasible_multiple", 0.0) or 0.0))

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
        curve = self._latest_curve_state.get(token)
        decision = self.t0_kernel.score(
            state, survival=self._survival_inputs(prediction),
            age_seconds=max(0.0, time.time() - float(position.get("entry_time", time.time()))),
            virtual_sol=int(getattr(curve, "virtual_sol_reserves", 0) or 0),
            virtual_token=int(getattr(curve, "virtual_token_reserves", 0) or 0))
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
            # The same economics the entry bid uses, on the same axis. `gain`
            # is the MARGINAL E[log W] this add buys, so the dollar value of
            # winning this particular race is that gain against the book --
            # not the notional being added, which says nothing about whether
            # the add was worth making. Without this an ADD fell back to the
            # fixed ladder while the entry beside it bid on economics.
            expected_edge_usd=max(0.0, gain * max(self.wallet_equity_usd, 0.0)),
            sol_price_usd=self.sol_price_usd,
            slot_value=self._entry_slot_value(
                token, max(0.0, gain * max(self.wallet_equity_usd, 0.0))),
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

    def _maybe_cross_check_mark(self, token: str, position: Dict[str, Any],
                                local_multiple: float) -> bool:
        """Ask the router what it thinks, off the decision path.

        Sampled rather than run on every mark: the point is to catch our own
        pricing drifting away from the market, and that is a property of the
        model rather than of any single position. Running it on every
        redecision would reintroduce the quote storm this change removed,
        just one step later.
        """
        interval = float(self.global_config.get("mark_cross_check_seconds", 30.0))
        if interval <= 0 or not self.jupiter or self.offline:
            return False
        now = time.time()
        if now - self._mark_checked_at.get(token, 0.0) < interval:
            return False
        self._mark_checked_at[token] = now
        self._spawn_background(
            self._cross_check_mark(token, int(position.get("size_tokens", 0) or 0),
                                   float(local_multiple),
                                   float(position.get("remaining_cost_usd", 0.0) or 0.0)))
        return True

    async def _cross_check_mark(self, token: str, size_tokens: int,
                               local_multiple: float, remaining_cost_usd: float) -> None:
        """Compare our local mark against the router's. Records, never decides.

        A divergence does not move capital and must not: the router is a
        second opinion about price, not an authority over it. What it is good
        for is telling us that our own curve state has gone stale or that the
        token has moved to a venue we are not reading -- both of which show up
        here long before they show up in a fill.
        """
        if size_tokens <= 0 or remaining_cost_usd <= 0:
            return
        try:
            quote = await self.jupiter.get_quote(
                token, USDC_MINT, size_tokens, slippage_bps=500)
        except Exception as exc:
            logger.debug("mark cross-check failed for %s: %s", token, exc)
            return
        if not quote or quote.output_amount <= 0:
            self._mark_checks_blocked += 1
            return
        router_multiple = (quote.output_amount / 1_000_000) / max(remaining_cost_usd, 1e-9)
        self._mark_checks += 1
        denominator = max(abs(local_multiple), abs(router_multiple), 1e-9)
        drift = abs(local_multiple - router_multiple) / denominator
        self._mark_drift_total += drift
        tolerance = float(self.global_config.get("mark_cross_check_tolerance", 0.10))
        if drift > tolerance:
            self._mark_checks_diverged += 1
            logger.warning(
                "local mark for %s is %.1f%% from the router (local %.4f, router %.4f); "
                "curve state may be stale or the token may have moved venue",
                token, drift * 100, local_multiple, router_multiple)
            if len(self._mark_divergences) < 20:
                self._mark_divergences.append({
                    "token": token, "local": local_multiple,
                    "router": router_multiple, "drift": drift,
                    "timestamp": time.time()})


    def _entry_slot_value(self, token: str, expected_edge_usd: float):
        """What one slot of delay costs a BUY of this token.

        Measured from the token's own recent path rather than inferred from a
        label. A "monster" label is a claim about the outcome; the slope of
        the curve over the last few seconds is a measurement of the present,
        and the present is what a slot of delay is spent in.
        """
        return self.slot_value.from_marks(
            list(self._mark_history.get(token) or ()),
            expected_edge_usd=expected_edge_usd, buying=True)

    def _exit_slot_value(self, token: str, expected_edge_usd: float):
        """What one slot of delay costs a SELL of this token.

        An exit races the hazard, not the drift: leaving a position that is
        about to become unsellable is worth the position, whatever the price
        is doing this second.
        """
        hazard = (self.rug_hazard.get_hazard(token)
                  if getattr(self, "rug_hazard", None) is not None else None)
        rate = None
        if hazard is not None and getattr(hazard, "data_status", "") == "OK":
            horizon = float(getattr(hazard, "hazard_30s", 0.0) or 0.0)
            if 0.0 < horizon < 1.0:
                # A 30-second survival probability, converted to the constant
                # rate that would produce it.
                rate = -math.log(1.0 - horizon) / 30.0
            elif horizon >= 1.0:
                rate = float("inf")
        if rate == float("inf"):
            # Certain within the horizon. The slot is worth the whole slice.
            return self.slot_value.from_hazard(1e6, expected_edge_usd=expected_edge_usd)
        return self.slot_value.from_hazard(rate, expected_edge_usd=expected_edge_usd)

    #: Every environment variable the desk reads, and what it unblocks.
    #: Presence is reported, never a value -- a status page that echoes a
    #: credential is a status page that leaks one to anything that can read it.
    CREDENTIALS: Tuple[Tuple[str, str], ...] = (
        ("YELLOWSTONE_GRPC_URL", "lowest-latency program stream"),
        ("YELLOWSTONE_GRPC_TOKEN", "authenticates the Yellowstone stream"),
        ("HELIUS_API_KEY", "wallet transaction history, and the Solana RPC endpoint"),
        # The RPC endpoints in config/chains.yaml interpolate this. Without it
        # the desk falls back to public RPC, which is rate limited to the point
        # of being unusable for a sniper -- so its absence is not a missing
        # nice-to-have, it is the chain read path degraded.
        ("ALCHEMY_KEY", "Solana and EVM RPC endpoints in config/chains.yaml"),
        ("TELEGRAM_API_ID", "public Telegram channels"),
        ("TELEGRAM_API_HASH", "public Telegram channels"),
        ("TELEGRAM_CHANNELS", "which channels the mesh watches"),
        ("JUPITER_API_KEY", "router quotes above the anonymous rate limit"),
        ("YOUTUBE_API_KEY", "YouTube Data API research"),
        ("NEYNAR_API_KEY", "Farcaster casts"),
        ("TWITCH_CLIENT_ID", "Twitch EventSub"),
        ("TWITCH_CLIENT_SECRET", "Twitch EventSub"),
        ("GITHUB_TOKEN", "higher public-repository quota"),
        ("X_BEARER_TOKEN", "official recent-search"),
        ("REDDIT_CLIENT_ID", "approved Reddit application-only OAuth"),
        ("REDDIT_CLIENT_SECRET", "approved Reddit application-only OAuth"),
    )










    def _resolve_corpus(self, token: str, **outcome: Any) -> None:
        """Attach an outcome to every decision made about one launch.

        Every decision, not only the entry: a launch is decided on repeatedly
        and resolving only the first would throw away every hold the desk got
        right or wrong. Defensive, because a corpus write must never be able
        to take down the stream handler that feeds it.
        """
        corpus = getattr(self, "counterfactual_corpus", None)
        if corpus is None:
            return
        try:
            corpus.resolve_by_mint(token, **outcome)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("corpus resolution for %s failed: %s", token, exc)









    def _measured_congestion(self) -> Optional[float]:
        """Chain congestion in [0, 1], or None when unmeasured.

        Derived from the observed slot time against nominal: slots arriving
        at twice their nominal spacing is a chain that is not keeping up, and
        every latency budget expressed in slots is optimistic by that factor.
        Returns None rather than a default, because a default of "calm" bids
        low into exactly the conditions where a low bid misses.
        """
        health = getattr(self, "_network_health", None)
        if not health:
            return None
        ratio = health.get("slot_time_ratio")
        if ratio is None:
            return None
        # 1.0x nominal is calm, 2.0x or worse is fully congested. Linear
        # between, clamped: the model buckets this anyway, so precision beyond
        # the bucket boundaries would be false.
        return max(0.0, min(1.0, (float(ratio) - 1.0)))
















    def _stage_exits(self, token: str, position: Dict[str, Any]) -> Optional[Any]:
        """Build the exit ladder for a position the moment it opens.

        Everything expensive about an exit -- deriving accounts, resolving the
        venue, encoding the discriminator -- does not depend on how much is
        being sold. Doing it now means that when something changes and the
        position has to be out, the work left is two integers.
        """
        size_tokens = int(position.get("size_tokens", 0) or 0)
        if size_tokens <= 0 or self.execution_engine is None:
            return None
        builders = self._exit_builders(token, position)
        if builders is None:
            return None
        build_sell, quote_sell, venue = builders
        return self.staged_exits.stage(
            token, size_tokens,
            state_version=self.state_sequencer.current(token),
            build_sell=build_sell, quote_sell=quote_sell,
            slippage_bps=int(self.global_config.get("exit_slippage_bps", 500)),
            venue=venue)

    def _exit_builders(self, token: str, position: Dict[str, Any]):
        """(build_sell, quote_sell, venue) for whichever venue owns this token.

        Injected into the ladder rather than known by it: the ladder has the
        same shape on a curve and on a pool, and a second copy of the routing
        rules is a second thing to keep in step with the router.
        """
        engine = self.execution_engine
        if engine is None:
            return None
        curve = self._latest_curve_state.get(token)
        if curve is not None:
            creator = str(getattr(curve, "creator", "") or "")
            route = getattr(engine, "pump_route", None)
            if route is None or not creator:
                return None
            user = engine.tx_builder.public_key

            def build_sell(size: int, minimum: int):
                return route.build_sell(token, creator, user, size, minimum)

            def quote_sell(size: int) -> Optional[int]:
                quote = curve_quote_sell(curve, size)
                return int(quote.output_amount) if quote.data_status == "OK" else None

            return build_sell, quote_sell, "pump_curve"

        reserves = self._latest_pool_state.get(token)
        account = self._pool_accounts.get(token)
        route = getattr(engine, "pumpswap_route", None)
        if (reserves is None or account is None or route is None
                or reserves.blocked_reason() is not None):
            return None
        user = engine.tx_builder.public_key

        def build_pool_sell(size: int, minimum: int):
            return route.build_sell(account, user, size, minimum)

        def quote_pool_sell(size: int) -> Optional[int]:
            quote = pool_quote_sell(reserves, size)
            return int(quote.output_amount) if quote.data_status == "OK" else None

        return build_pool_sell, quote_pool_sell, "pumpswap"

    def _reprice_staged_exits(self, token: str) -> int:
        """Refresh the protective bounds after the market moved.

        A minimum computed against a curve from thirty seconds ago is not
        protection, it is an invitation -- and repricing costs one local quote
        rather than a rebuild.
        """
        position = self.elogw_engine.open_positions.get(token)
        if position is None:
            return 0
        builders = self._exit_builders(token, position)
        if builders is None:
            return 0
        build_sell, quote_sell, venue = builders
        return self.staged_exits.reprice(
            token, state_version=self.state_sequencer.current(token),
            build_sell=build_sell, quote_sell=quote_sell,
            slippage_bps=int(self.global_config.get("exit_slippage_bps", 500)),
            venue=venue)

    def _local_mark(self, token: str, position: Dict[str, Any]):
        """What this position is worth right now, from streamed state alone.

        Quoted by selling the WHOLE position through the curve or the pool, so
        the mark carries its own price impact -- which is the difference
        between what the position is worth and what it would fetch.

        No await, no round trip. This ran only under `if self.dry_run`, so
        every live HOLD/ADD/BANK/EXIT redecision waited on a Jupiter quote
        after an event that had already arrived: the desk paid for its
        latency advantage and then handed it back at the moment of deciding.
        """
        size_tokens = int(position.get("size_tokens", 0) or 0)
        if size_tokens <= 0:
            return None
        quote = self._follow_exit_quote(token, size_tokens)
        if quote is not None and self.sol_price_usd > 0:
            value_usd = (float(quote.output_amount) / LAMPORTS_PER_SOL) * self.sol_price_usd
            remaining = max(float(position.get("remaining_cost_usd", 0.0) or 0.0), 1e-9)
            # Labelled by the venue that actually priced it. A sell into a
            # bonding curve and a sell into a migrated pool are different
            # measurements with different impact behaviour, and pooling them
            # under one name makes the two indistinguishable downstream.
            venue = ("local_pumpswap_executable_sell"
                     if self._latest_pool_state.get(token) is not None
                     else "local_pump_executable_sell")
            return (value_usd / remaining, value_usd, venue,
                    quote.price_impact_pct)
        # No local quote. A recent streamed price ratio is still a measurement
        # of this market, just not of this size -- so it is used, and labelled
        # as the weaker thing it is.
        stream_mark = self._latest_stream_mark.get(token)
        if stream_mark and time.time() - float(stream_mark["timestamp"]) <= 3.0:
            multiple = float(stream_mark["multiple"])
            return (multiple,
                    max(0.0, float(position.get("remaining_cost_usd", 0.0) or 0.0) * multiple),
                    "decoded_onchain_reserve_event", None)
        return None

    async def _mark_position(self, token: str, position: Dict[str, Any]):
        local = self._local_mark(token, position)
        if local is not None:
            multiple, current_value, measurement, impact = local
            self._marks_local += 1
            observation = {"type": "stream_mark", "feasible": True, "value_usd": current_value,
                           "price_multiple": multiple, "timestamp": time.time(),
                           "measurement": measurement, "data_status": "OK"}
            if impact is not None:
                observation["price_impact_pct"] = impact
            self.rug_hazard.record_observation(token, observation)
            self.dataset_builder.record_market_observation(token, observation)
            self.counterfactual_lab.record_market_observation(token, multiple, observation["timestamp"])
            # The router still gets asked -- afterwards, off the decision path,
            # as an independent check on our own pricing. A local mark nothing
            # ever contradicts is a local mark nobody has verified.
            self._maybe_cross_check_mark(token, position, multiple)
            return multiple, current_value
        # Nothing local could answer. Paying the round trip is right here:
        # a decision priced on no mark at all is worse than a slow one.
        self._marks_router += 1
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

    def _exit_edge_usd(self, position: Dict[str, Any], exit_pct: float, reason: str,
                       supplied: Optional[float] = None) -> float:
        """The dollar value of winning the race to get out of this slice.

        An escape is not an economic optimisation. When the reason for leaving
        is that the thing may be about to stop being sellable, the value of
        landing is the whole slice, and a bid sized on the marginal E[log W]
        of a routine bank would under-bid exactly the trade that must not
        miss.
        """
        if supplied is not None:
            return max(0.0, float(supplied))
        slice_value = (max(0.0, float(position.get("remaining_cost_usd", 0.0) or 0.0))
                       * max(0.0, float(position.get("current_multiple", 1.0) or 1.0))
                       * min(max(float(exit_pct), 0.0), 1.0))
        urgent = any(token in str(reason) for token in
                     ("rug_hazard", "escape", "emergency", "sellability", "authority"))
        if urgent:
            return slice_value
        action_value = position.get("action_value") or {}
        q = float(action_value.get("q", 0.0) or 0.0)
        return max(0.0, q * max(self.wallet_equity_usd, 0.0))

    async def _execute_exit(self, token: str, position: Dict[str, Any], exit_pct: float,
                            reason: str, expected_edge_usd: Optional[float] = None):
        current_tokens = int(position["size_tokens"])
        sold_tokens = min(current_tokens, max(1, int(current_tokens * min(max(exit_pct, 0), 1))))
        # Whether the ladder had this exit prepared. Recorded on the attempt
        # rather than acted on here: the engine owns submission, and a staged
        # rung is an instruction, never a reason to sell.
        rung, staged_status = self.staged_exits.take(
            token, min(max(float(exit_pct), 0.0), 1.0),
            state_version=self.state_sequencer.current(token))
        if rung is not None and staged_status.startswith("STALE"):
            # One local quote, not a rebuild. Still far cheaper than
            # constructing the instruction from nothing at the worst moment.
            self._reprice_staged_exits(token)
            rung, staged_status = self.staged_exits.take(
                token, min(max(float(exit_pct), 0.0), 1.0),
                state_version=self.state_sequencer.current(token))
        result = await self.execution_engine.execute_sell(
            token, sold_tokens, slippage_bps=500, use_jito=True,
            decision_id=position.get("decision_id"),
            # What losing this race costs. For an ordinary bank that is the Q
            # the decision priced; for an escape it is the slice itself,
            # because a rug exit that does not land loses the position rather
            # than an increment of expected growth. Bidding the same fixed
            # ladder for both is the error in both directions at once.
            expected_edge_usd=self._exit_edge_usd(position, exit_pct, reason,
                                                  expected_edge_usd),
            sol_price_usd=self.sol_price_usd,
            slot_value=self._exit_slot_value(
                token, self._exit_edge_usd(position, exit_pct, reason,
                                           expected_edge_usd)))
        # Only landed, non-simulated sells. A paper fill is not evidence about
        # the network, and a submission that never landed has no latency at
        # all -- counting it as its timeout would make a failing relay look
        # merely slow, which is the direction that gets a position trapped.
        self.landing_latency.record(getattr(result, "latency_ms", 0),
                                    landed=bool(getattr(result, "landed", False)),
                                    simulated=bool(result.simulated))
        attempt = {**_jsonable(result), "exit_reason": reason,
                   "exit_pct": sold_tokens / max(current_tokens, 1),
                   "staged_exit": staged_status,
                   "staged_rung": (rung.fraction if rung is not None else None)}
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
        # The position is a different size now, so every rung is priced for a
        # size that no longer exists. Restaged on a partial, released on a
        # close -- a ladder held for a position that is gone is a ladder that
        # will one day be handed to the wrong token.
        if remaining > 0:
            try:
                self._stage_exits(token, position)
            except Exception as exc:
                logger.debug("restaging exits for %s failed: %s", token, exc)
        else:
            self.staged_exits.release(token)
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
                # What this trade ACTUALLY returned, from its own proceeds.
                # This reported the position's PEAK whenever the trade was
                # green, which inflated every winner in the evidence ledger by
                # the whole distance between the high and the exit -- the
                # exact quantity the exit policy exists to minimise, scored as
                # though it had been captured.
                "realized_multiple": max(0.0, 1.0 + pnl / max(
                    float(position.get("initial_cost_usd", 1.0)), 1e-9)),
                # The best mark this position could have been SOLD at, not the
                # highest it printed. None when capacity was never measured:
                # an unmeasured ceiling is not an infinite one, and a chart
                # high nobody could fill is a label that teaches a model to
                # predict the unattainable.
                "max_feasible_multiple": position.get("feasible_high_water_multiple"),
                "chart_high_multiple": float(position.get("high_water_multiple", 1.0)),
                "feasible_marks_measured": int(position.get("feasible_marks", 0)),
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
        self._sweep_rug_classification()
        self._prune_hazard_tracking()
        # Checkpointed on the cadence follows are resolved on, so a restart
        # costs one cycle of measurement rather than every open follow.
        self.save_follow_candidates()
        self._resolve_follow_candidates()
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
        # Counted before anything can reject it, including the empty-token
        # guard below. "The stream is connected" and "the stream is delivering
        # the event type we need" are different claims, and only the second
        # one matters -- a desk seeing a million trades and no creations has a
        # healthy connection and an empty denominator.
        kind = str(event.get("type", "") or "unknown")
        self._stream_events[kind] = self._stream_events.get(kind, 0) + 1
        token = event.get("token", "")
        if kind == "token_created" and token:
            # Opened on the first line the handler owns, so `receive_to_decode`
            # measures the monitor's decode and hand-off and nothing of ours.
            # `block_time` is the cluster's, so chain_to_receive carries this
            # box's NTP skew -- the ledger reports that caveat rather than
            # quoting the number as though it were exact.
            self.latency.open(token, block_time=event.get("timestamp"),
                              slot=event.get("slot"))
            self.latency.mark(token, "receive_to_decode")
        if not token:
            self._stream_events[f"{kind}:no_token"] = (
                self._stream_events.get(f"{kind}:no_token", 0) + 1)
            return
        if event.get("type") == "token_created":
            # Counted before anything can filter it. This is the denominator.
            self.launch_census.see(
                token, creator=str(event.get("creator", "") or ""),
                at=float(event.get("timestamp", time.time())),
                regime=str(self.current_regime or "unknown"))
            self.wallet_intel.record_token_lifecycle(token, launch_at=event.get("timestamp", time.time()))
            self._assess_identity(token, event)
            # Seed the curve at its protocol-defined starting depth. The
            # CreateEvent carries no reserves, so without this the desk is
            # blind at exactly T0 -- _local_liquidity returns 0 and liquidity
            # is DATA_BLOCKED at the one instant a launch has to be priced.
            # Measured across 400 live episodes, liquidity_features was 0%
            # populated at T0.
            #
            # Program constants, not a claim about this token, and verified
            # continuously against the constant product by
            # pump_curve_invariant_holds. Replaced by real reserves on the
            # first trade, so it is only ever the pre-first-trade answer --
            # and an UPPER BOUND, since real reserves are unknowable here.
            #
            # Ordered before start_episode deliberately: that call captures
            # the t0 snapshot, so seeding after it leaves t0 blind.
            if token not in self._latest_curve_state:
                self._latest_curve_state[token] = BondingCurveState(
                    virtual_token_reserves=PUMP_INITIAL_VIRTUAL_TOKEN,
                    virtual_sol_reserves=PUMP_INITIAL_VIRTUAL_SOL,
                    real_token_reserves=0, real_sol_reserves=0,
                    token_total_supply=int(event.get("token_total_supply", 0) or 0),
                    complete=False, creator=str(event.get("creator", "") or ""),
                )
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
                # Migration is itself an outcome, and one that separates a
                # curve that stalled from one that graduated and then died.
                self.launch_census.resolve(token, migrated=True)
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
                stamp = float(event.get("timestamp", time.time()))
                self._latest_stream_mark[token] = {
                    "multiple": curve_multiple, "timestamp": stamp
                }
                # A short path, not just the latest point. One mark says where
                # the price is; a few say how fast it is moving, and how fast
                # it is moving is what a slot of delay costs.
                self._mark_history.setdefault(token, deque(maxlen=32)).append(
                    (stamp, curve_multiple))
                # Resolved from the stream, independent of whether we ever
                # traded it. That independence is the whole value: a peak
                # measured only on tokens we entered cannot price what a
                # screen threw away.
                self.launch_census.resolve(
                    token, peak_multiple=curve_multiple, at=stamp)
                # The same peak, attached to every decision made about this
                # launch. This is the counterfactual: what was available,
                # measured on a token the desk may never have touched.
                self._resolve_corpus(token, peak_multiple=curve_multiple)
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
                           "signature": event.get("signature"), "program": event.get("program"),
                           # Curve depth, free on the event already decoded.
                           # liquidity_features measured 0% populated at t0
                           # because it waited for a DexScreener quote a
                           # pre-migration launch cannot have. Storing the
                           # reserves keeps the derivation point-in-time.
                           "virtual_sol_reserves": virtual_sol or None,
                           "virtual_token_reserves": virtual_token or None}
            # Reconstruct the holder set from fills we watched: free, exact
            # for a launch this young, and available at stream latency
            # rather than behind an RPC method nothing will serve.
            _delta = event.get("actual_token_delta_raw")
            if event.get("wallet") and _delta:
                self.observed_holders.record_trade(
                    token, str(event["wallet"]), float(_delta))
                _observed = self.observed_holders.snapshot(token)
                if _observed:
                    self.holder_trajectory.record_mapping(
                        token, _observed, timestamp=observation["timestamp"],
                        source="observed_trades")
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
            self._open_follow_candidate(token, event)
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
        if self.dataset_builder is None:
            if len(self._pending_social_signals) == self._pending_social_signals.maxlen:
                self._pending_social_dropped += 1
            self._pending_social_signals.append(dict(signal))
            return
        if signal.get("type") == "new_mention":
            self.info_graph.record_event(token, LeadEventType.OBSCURE_X_MENTION, signal.get("account", ""),
                                         "social", signal.get("timestamp", time.time()), signal)
        self.rug_hazard.record_observation(token, {"type": "social", **signal})
        self.dataset_builder.record_market_observation(token, {"type": "social", **signal})
        if signal.get("type") == "new_mention" and signal.get("first_mention"):
            self._spawn_background(self._triage_social_candidate(signal))

    async def _flush_pending_social_signals(self) -> None:
        while self._pending_social_signals:
            await self._on_social_mention(self._pending_social_signals.popleft())
        if self._pending_social_dropped:
            logger.error(
                "social startup buffer overflowed; %d earliest signals were dropped",
                self._pending_social_dropped,
            )

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



    def _register_memory_reliefs(self) -> None:
        """What the desk gives up under pressure, in order of cheapness.

        Every one of these costs context, and none touches the decision path.
        A desk under memory pressure must get quieter, never dumber -- shedding
        the models to keep running trades the thing the desk is for the
        ability to keep going, and it is better to die loudly and restart than
        to trade blind.
        """
        census = self.launch_census
        original_cap = census.max_records

        def trim_census():
            census.max_records = max(100, original_cap // 2)
            census._evict_if_needed()

        def shed_census():
            census.max_records = max(100, original_cap // 8)
            census._evict_if_needed()
            census.save()

        self.memory.register(Relief(
            name="launch_census", trim=trim_census, shed=shed_census,
            restore=lambda: setattr(census, "max_records", original_cap),
            detail="spill per-launch detail sooner; totals are already counted"))

        miners = self.data_miners
        original_concurrency = miners.concurrency

        def trim_miners():
            miners.concurrency = max(1, original_concurrency // 2)
            miners._semaphore = asyncio.Semaphore(miners.concurrency)

        self.memory.register(Relief(
            name="data_miners", trim=trim_miners,
            restore=lambda: (setattr(miners, "concurrency", original_concurrency),
                             setattr(miners, "_semaphore",
                                     asyncio.Semaphore(original_concurrency))),
            detail="fewer simultaneous fetches; context is not on the hot path"))

        def trim_marks():
            # Shorter price paths for tokens we hold no position in. Slot
            # value degrades slightly for exactly the tokens whose slot value
            # we are not currently spending.
            held = set(self.elogw_engine.open_positions)
            for token in list(self._mark_history):
                if token not in held:
                    self._mark_history.pop(token, None)

        self.memory.register(Relief(
            name="mark_history", trim=trim_marks,
            detail="drop price paths for tokens we hold nothing in"))

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
            try:
                await asyncio.wait_for(
                    desk._fatal_task_event.wait(), timeout=args.run_seconds)
            except asyncio.TimeoutError:
                pass
            else:
                raise RuntimeError(
                    f"critical runtime task stopped: {desk._fatal_task_detail}")
        else:
            await desk._fatal_task_event.wait()
            raise RuntimeError(
                f"critical runtime task stopped: {desk._fatal_task_detail}")
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
    # Before the loop exists, not after: a policy set once the loop is running
    # changes nothing, which is the quiet way this optimisation gets applied
    # and has no effect.
    loop_status = install_fast_event_loop()
    logging.getLogger(__name__).info("EVENT LOOP %s", loop_status)
    os.environ.setdefault("MEMECOIN_EVENT_LOOP", loop_status)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
