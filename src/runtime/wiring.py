"""Building the desk: keys, chains, models, execution, research.

Split out of `main.py`. Startup wiring shares nothing with the trading path
except the object it populates, so keeping them in one file meant every
change to how a subsystem is constructed sat in the same merge surface as
every change to how a position is sized.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from solders.keypair import Keypair
from src.chains.rpc_manager import ChainRegistry, RPCManager
from src.chains.yellowstone_grpc import (
    NATIVE_FASTPATH_STATUS, PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream, YellowstoneClient,
    create_combined_subscription,
)
from src.detection.rug_detector import RiskLevel, RugDetector
from src.detection.t0_risk import LaunchInvariantLedger, T0RiskView
from src.detection.token_detector import DetectionSource, TokenCandidate, TokenDetectionEngine
from src.execution.jupiter_jito import (
    ExecutionEngine,
    JupiterClient,
    JitoClient,
    PriorityFeeOptimizer,
    SolanaTransactionBuilder,
    USDC_MINT,
)
from src.research.trade_evidence import TradeEvidenceLedger
from src.research.dataset_builder import PointInTimeDatasetBuilder
from src.research.global_research_miner import GlobalResearchMiner
from src.research.chain_miners import register_chain_miners
from src.execution.signer import signer_from_env
from src.runtime.offload import OffloadedPool, install_fast_event_loop
from src.research.data_miners import DataMinerPool
from src.research.solana_miners import register_solana_miners
from src.research.source_catalogue import default_registry
from src.research.regional_miners import register_regional_miners
from src.research.telegram_miners import (
    ChannelBook, extract_handles, register_telegram_miners,
)
from src.research.identity_watch import IdentityWatch, Verdict as IdentityVerdict
from src.research.web_miners import register_web_miners
from src.research.attribution import EdgeDecayMonitor
from src.runtime.hot_state import HotState, HotStateBudget
from src.strategies.action_value import (
    Action as ActionValue, ActionValuePolicy, Decision as ActionDecision,
    PositionState as ActionState,
)
from src.collectors.event_source import Event, SourceMesh
from src.collectors.registry import build_sources, expand_env_channels, load_declarations
from src.collectors.transports import (
    HttpClient, build_transports, start_transports, stop_transports, transport_report,
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
from src.strategies.multihead_predictor import SURVIVAL_LEVELS, ElogwEngine, MultiHeadPredictor, PredictionFeatures
from src.chains.pump_route import (
    TOKEN_2022_PROGRAM, TOKEN_PROGRAM, NativePumpRoute, PumpRouteConfig,
)
from src.chains.pumpswap_route import PoolState, PumpSwapRoute, PumpSwapRouteConfig, parse_pool
from src.execution.pump_fees import (
    DEFAULT_SCHEDULE as PUMP_FEE_SCHEDULE, VENUE_BONDING_CURVE,
)
from src.strategies.distribution import DistributionDetector
from src.strategies.mega_event import MegaEventReserve
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
from src.strategies.t0_kernel import SurvivalInputs, T0Kernel
from src.chains.launchpads import LaunchpadRegistry
from src.execution.exit_readiness import (
    ExcursionLedger, ExitReadinessLedger, choose_exit_mode)
from src.runtime.feed_race import FeedRace
from src.research.cold_distillation import ColdDistillate
from src.research.latency_value import LatencyValueLedger
from src.runtime.load_shedding import EconomicLoadShedder
from src.chains.native_ingress import NativeIngress
from src.runtime.process_offload import ProcessOffloadedPool
from src.research.benchmark_wallets import BenchmarkCorpus, load_roster
from src.strategies.cohort_lifecycle import evaluate_cohorts
from src.strategies.funder_ancestry import FunderAncestry, compress_independence
from src.strategies.temporal_funding import (
    Withdrawal, find_clusters, independence_discounts, measure_source_rate)
import logging
MODEL_HYPOTHESIS_ID = "production_multihead_v1"

logger = logging.getLogger(__name__)


def _tagged_feed(callback, feed: str):
    """Stamp which feed delivered an event, without touching the decoders.

    The race ledger keys on `event["feed"]`, and every decoder is shared
    between the two paths -- so the tag has to be applied where the paths
    differ, which is here, at the callback each one was given.
    """
    async def tagged(event):
        if isinstance(event, dict):
            event.setdefault("feed", feed)
        return await callback(event)

    return tagged


class SubsystemWiring:
    """Constructing the desk's subsystems, in the order they depend on each other.

    The setup half of a 5,500-line class, moved verbatim as a mixin. These
    methods run once at startup, touch every subsystem the desk owns, and are
    on no latency path at all -- which makes them both the safest slice to
    move and one of the most valuable, because they carried the largest
    import surface in the file. Roughly forty of `main.py`'s imports existed
    only to construct something, and they now live beside the construction.

    The ORDER these are called in is load-bearing and is not encoded here: it
    lives in `initialize`, which stays on the desk. Prediction must exist
    before execution can be given a predictor, and every event consumer must
    exist before a stream is started or the first launch arrives at a
    half-built desk. A mixin that also owned the sequencing would hide that
    dependency inside a call graph rather than stating it in one place."""

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
            # HANDLERS FIRST, stream second. Each monitor registers its
            # callbacks in its constructor, and `subscribe` starts consuming
            # immediately -- so constructing them afterwards left a window in
            # which decoded events found `self._handlers` empty and were
            # dropped with no record anywhere. It is a short window, but it
            # sits at exactly the moment the desk comes up after a restart,
            # and every launch inside it is unbackfillable.
            self.pump_monitor = PumpFunMonitor(self.yellowstone, self._on_pump_event)
            self.pump_swap_monitor = PumpSwapMonitor(self.yellowstone, self._on_pump_event)
            self.raydium_monitor = RaydiumMonitor(self.yellowstone, self._on_raydium_event)
            # The native receiver comes up ALONGSIDE, not instead, and BEFORE
            # the reference stream: a gRPC subscription takes a moment to be
            # served, and if the reference starts first every event in that
            # gap reads as a native miss, which under a permanent-demotion
            # latch means the shadow can never be promoted however correct it
            # is. Starting it first turns that skew into `native_only`, which
            # is counted and not punished.
            ingress = getattr(self, "native_ingress", None)
            if ingress is not None and not ingress.start():
                logger.info("NATIVE INGRESS not running: %s",
                            ingress.unavailable_reason)
            await self.yellowstone.subscribe(create_combined_subscription())
            # A SECOND feed, running at the same time rather than instead.
            #
            # The RPC program stream existed only as a fallback for when
            # Yellowstone would not connect, which means the desk's
            # redundancy was conditional on total failure -- the shape of
            # outage that never happens. What does happen is one feed being
            # a hundred milliseconds slower on a particular launch, or
            # dropping a single update, and against that a standby is worth
            # nothing because it is not running.
            #
            # Racing is free here because the duplicate guard makes it safe:
            # every feed delivering the same launch calls `FeedRace.observe`
            # and only the first gets a decision. The rest are the
            # measurement, and the measurement is what says whether the
            # second feed is ever first.
            if bool(self.global_config.get("race_secondary_feed", True)):
                await self._start_secondary_feed()
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

    async def _start_secondary_feed(self) -> bool:
        """Bring up the RPC program stream BESIDE Yellowstone, tagged.

        Its events carry `feed: "rpc_ws"`, so the race ledger can say which
        path was first per launch instead of attributing everything to the
        one name the primary happens to use.
        """
        try:
            stream = SolanaRpcProgramStream(
                self.solana_rpc,
                [PumpFunMonitor.PUMP_FUN_PROGRAM, PumpSwapMonitor.PUMP_AMM_PROGRAM,
                 RaydiumMonitor.RAYDIUM_AMM_V4, RaydiumMonitor.RAYDIUM_CPMM,
                 RaydiumMonitor.RAYDIUM_CLMM, RaydiumMonitor.METEORA_DLMM,
                 RaydiumMonitor.METEORA_DYNAMIC_AMM, RaydiumMonitor.ORCA_WHIRLPOOL])
            self.secondary_pump_monitor = PumpFunMonitor(
                stream, _tagged_feed(self._on_pump_event, "rpc_ws"))
            self.secondary_pump_swap_monitor = PumpSwapMonitor(
                stream, _tagged_feed(self._on_pump_event, "rpc_ws"))
            self.secondary_raydium_monitor = RaydiumMonitor(
                stream, _tagged_feed(self._on_raydium_event, "rpc_ws"))
            await stream.start()
        except Exception as exc:
            # A second feed is redundancy, not a dependency. Failing to bring
            # it up must never stop the desk that already has a primary.
            logger.warning("secondary feed not started: %s", exc)
            self.secondary_stream = None
            return False
        self.secondary_stream = stream
        if getattr(self, "feed_race", None) is not None:
            self.feed_race.register_feed("yellowstone")
            self.feed_race.register_feed("rpc_ws")
        logger.info("SECONDARY FEED racing Yellowstone: RPC program stream")
        return True

    def _optional(self, name, fallback, build):
        """Construct an OBSERVATIONAL subsystem, or run without it.

        The distinction this enforces: a subsystem that informs a decision
        may fail loudly, because trading on a half-built brain is worse than
        not trading. A subsystem that only WATCHES -- scores other people's
        wallets, counts which feed was first, records how quickly the sell
        template was ready -- must never be able to stop the desk.

        The asymmetry is about what is recoverable. A research convenience
        that is missing for an hour can be computed later from the same
        public data. Forward evidence not gathered during an hour the desk
        was down is gone permanently, and with StartLimitBurst a startup
        exception does not cost an hour, it leaves the unit stopped until
        somebody notices and runs reset-failed.

        The fallback is the empty instance, so every caller and every
        /status reader gets the shape it expects and simply sees nothing
        measured -- which is exactly what happened.
        """
        try:
            return build()
        except Exception as exc:
            logger.error(
                "OPTIONAL SUBSYSTEM %s failed to build (%s: %s); the desk "
                "continues without it rather than refusing to start",
                name, type(exc).__name__, exc)
            try:
                return fallback()
            except Exception:  # pragma: no cover - fallback must be trivial
                return None

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
            # Seed the watch list from distilled DNA before the engine
            # starts, so the desk does not rediscover every wallet from
            # scratch on each restart while 56,636 observed wallets sit
            # unread in its own spill.
            seeds = self._wallet_dna_seeds()
            for component in (self.genealogy, self.wallet_intel, self.social_intel, self.prelaunch,
                              self.info_graph, self.rug_hazard):
                if component is self.wallet_intel and seeds:
                    await component.start(initial_wallets=seeds)
                else:
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
        # The entry thesis, frozen before the action it justifies.
        self.trade_evidence = TradeEvidenceLedger(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "trade_evidence.jsonl")
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
        # The Rust T0 core, wired into the canonical path rather than sitting
        # beside it. It shadows the Python policy on every ordinary decision
        # and takes over only after a run of measured agreement -- and a
        # single disagreement while it is deciding demotes it for the rest of
        # the session. Promotion by evidence, demotion by default, exactly as
        # a model is promoted here.
        self.t0_kernel = T0Kernel(
            self.action_policy,
            mode=str(self.global_config.get("t0_kernel_mode", "auto")),
            promote_after=int(self.global_config.get("t0_kernel_promote_after", 500)))
        logger.info("T0 kernel: mode=%s native=%s",
                    self.t0_kernel.mode.value, self.t0_kernel.native_status)

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
        # The channels the operator already listed for the social collector.
        # Asking for them twice -- once in an env var, once in YAML -- is
        # asking for two lists that disagree.
        declarations = expand_env_channels(declarations)
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
        # The half of independence that behaviour cannot see. Wallets funded
        # and deployed for one launch have no co-occurrence history, so every
        # one reads as independent -- and a Sybil built for this launch is
        # invisible at the moment it is used.
        self.funder_ancestry = FunderAncestry(
            self.genealogy,
            max_hops=int(self.global_config.get("funder_max_hops", 4)),
            hub_threshold=int(self.global_config.get("funder_hub_threshold", 50)))
        self.buyer_dna = BuyerDNA(
            depth=int(self.global_config.get("buyer_dna_depth", 25)),
            min_corpus=int(self.global_config.get("buyer_dna_min_corpus", 50)))
        self.swarm_predictor = SwarmPredictor(
            skill_threshold=float(self.global_config.get("swarm_skill_threshold", 0.6)),
            independence_threshold=float(
                self.global_config.get("swarm_independence_threshold", 0.5)))
        # What the opening cohort did AFTER its fill. BuyerDNA stops at the
        # entry; these carry the position through absorption and distribution.
        self.cohort_reports: Dict[str, Any] = {}
        # Coordination that a centralised-exchange withdrawal was used to
        # hide. Needs each hot wallet's own emission rate as a denominator,
        # so it stays empty -- and says so -- until rates are measured.
        self.temporal_clusters: List[Any] = []
        self.temporal_discounts: Dict[str, float] = {}
        self.exchange_withdrawals: List[Withdrawal] = []
        self.exchange_rates: Dict[str, float] = {}
        # Every launchpad normalises to one launch event. Programs start as
        # HYPOTHESES and are promoted only by decoding cleanly on this node,
        # so coverage reports what was seen rather than what was hoped for.
        self.launchpads = self._optional(
            "launchpad registry", LaunchpadRegistry, LaunchpadRegistry)
        # Which feed reaches this box first, measured. Coverage outranks
        # speed: a feed that misses events is blind on them, not slow.
        self.feed_race = self._optional("feed race", FeedRace, FeedRace)
        # Under a burst the desk cannot look at every launch, and which ones
        # it drops is a decision rather than a queue discipline. See
        # src/runtime/load_shedding.py.
        # Years of the chain, compressed to a lookup a T0 decision can
        # afford. Absent on a fresh box, which is the honest state: a
        # deployer nobody has history for is unknown, not safe.
        self.cold_distillate = ColdDistillate.load(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "cold_distillate.json")
        if self.cold_distillate is not None:
            logger.info(
                "COLD DISTILLATE loaded: %d deployers (%d with a rate), "
                "%d funders, covering to %s",
                len(self.cold_distillate.deployers),
                sum(1 for dna in self.cold_distillate.deployers.values()
                    if dna.measurable),
                len(self.cold_distillate.funders),
                self.cold_distillate.covers_until)
        # Everything done for speed has been done on the belief that earlier
        # is better. That belief has never been priced, and the answer
        # decides whether the next month goes to latency or to alpha.
        self.latency_value = LatencyValueLedger()
        self.load_shedder = EconomicLoadShedder(
            int(self.global_config.get("max_candidate_pipelines", 100)))
        # The sell must exist before it is needed, and be proven to.
        self.exit_readiness = self._optional(
            "exit readiness", ExitReadinessLedger, ExitReadinessLedger)
        # MFE:MAE per entry state -- what win rate cannot see.
        self.excursions = self._optional(
            "excursion ledger", ExcursionLedger, ExcursionLedger)
        # Public wallets worth reconstructing, and what FOLLOWING them costs.
        # Headline PnL is recorded as a claim here and never scored.
        #
        # Wrapped, like every other purely OBSERVATIONAL subsystem above,
        # because none of them is worth the desk's life. This one reads a
        # YAML roster and a JSON corpus off disk at startup; a malformed
        # file, a permission the unit does not have, or a library that
        # behaves differently on another Python would otherwise raise here,
        # take the process down, and -- with StartLimitBurst -- leave the
        # desk permanently stopped. A desk that cannot score other people's
        # wallets is a desk missing a research convenience. A desk that is
        # not running has lost the forward evidence that is the whole point
        # of it, and that evidence cannot be backfilled.
        self.benchmark_corpus = self._optional(
            "benchmark corpus", BenchmarkCorpus, lambda: load_roster(
            str(self.global_config.get("benchmark_roster",
                                       "config/benchmark_wallets.yaml")),
            BenchmarkCorpus(
                path=(None if self.offline else str(
                    Path(self.global_config.get("ops_state_dir", "data/state"))
                    / "benchmark_wallets.json")),
                cost_per_round_trip=float(
                    self.global_config.get("follow_cost_round_trip", 0.02)))))
        if not self.offline:
            try:
                self.benchmark_corpus.load()
            except Exception as exc:  # pragma: no cover - disk only
                logger.warning("benchmark corpus did not load (%s); the desk "
                               "runs without it", exc)
        # Entries per token, kept only for tokens the hot state still holds.
        self._actor_entries: Dict[str, List[Entry]] = {}
        self.independence_report = IndependenceReport(status="DATA_BLOCKED")
        self.ancestry_report: Optional[Any] = None
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
        # Isolated when MEMECOIN_SIGNER_SOCKET names a socket, local otherwise.
        # Chosen once, here, and reported on /status -- so "the key is in this
        # process" is a stated configuration rather than an unexamined default.
        builder = SolanaTransactionBuilder(
            self.solana_rpc, signer_from_env(self.keypair))
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
        # The landing corpus is the only dataset real fills produce, and it
        # was held in memory alone -- so every restart destroyed it and a desk
        # restarted a dozen times in a day had none. A landing attempt cannot
        # be reconstructed after the fact by any means.
        self.execution_engine.landing_model.path = (
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "landing_attempts.jsonl")
        restored = self.execution_engine.landing_model.load()
        if restored:
            logger.info("landing model restored %d attempts from disk", restored)
        # The desk owns the streamed curve state; the engine reads it through
        # this rather than keeping its own, because two views of the price we
        # are about to trade at is one view too many.
        self.execution_engine.curve_state_provider = self._latest_curve_state.get
        # Graduation should change the venue and nothing else. The same desk
        # state, the same policy and the same signing path continue across it;
        # only the pool these two providers describe is new.
        self.execution_engine.pool_state_provider = self._latest_pool_state.get
        self.execution_engine.pool_account_provider = self._pool_accounts.get
        # Measured congestion for the bid and for the attempt record. Until
        # this was supplied the landing model bucketed every attempt as
        # "unknown", so it could never learn that a bid clearing in calm
        # conditions misses in a rush.
        self.execution_engine.congestion_provider = self._measured_congestion
        self.fee_optimizer = PriorityFeeOptimizer()
        if not self.offline:
            await self.execution_engine.start()

    async def _setup_detection_and_risk(self):
        self.rug_detector = RugDetector(
            self.solana_config, self.solana_rpc, self.jupiter,
            # The streamed curve, so a mint the router has never indexed is
            # still known to be sellable back to its own bonding curve. The
            # router was asked about mints seconds old and its ignorance was
            # recorded as a confident "no route", which hard-vetoed 100% of
            # decided launches.
            curve_state_provider=self._latest_curve_state.get)
        # What a decision can know for free, and the ledger that decides how
        # much that is. The full audit is three to five sequential RPC round
        # trips; it now runs BESIDE the decision rather than in front of it,
        # and every completed report teaches this ledger what the launch
        # program guarantees so the next launch needs fewer of them.
        self.invariant_ledger = LaunchInvariantLedger(
            Path(self.global_config.get("ops_state_dir", "data/state"))
            / "launch_invariants.json")
        self.t0_risk = T0RiskView(
            self.invariant_ledger,
            curve_state_provider=self._latest_curve_state.get,
            risk_level_enum=RiskLevel)
        self.detection_engine = TokenDetectionEngine(self.chain_registry)

    async def _setup_research(self):
        self.dataset_builder = PointInTimeDatasetBuilder(
            self.solana_config, self.solana_rpc, self.genealogy, self.wallet_intel, self.social_intel,
            self.prelaunch, self.info_graph, self.rug_hazard, self.champion_challenger,
        )
        # Attached rather than constructed inside, so the builder's own
        # constructor stays a data-collection concern and the ledger can be
        # driven directly by a test.
        self.dataset_builder.latency_value = self.latency_value
        self.info_graph.set_outcome_provider(self.dataset_builder.get_outcome)
        if hasattr(self.genealogy, "set_outcome_provider"):
            self.genealogy.set_outcome_provider(self.dataset_builder.get_outcome)
        self.global_research = GlobalResearchMiner(self.champion_challenger)
        # Market and chain context, each source on its own clock. The program
        # stream is the fastest and most trustworthy data the desk has; what
        # it does not carry is why a price path looks the way it does, and
        # that is what a forward ledger needs to explain an outcome rather
        # than only record it.
        self.data_miners = DataMinerPool(
            concurrency=int(self.global_config.get("data_miner_concurrency", 6)),
            on_records=self._ingest_mined_records)
        self.miner_registration = dict(register_solana_miners(
            self.data_miners, rpc=self.solana_rpc, http=self.http_client,
            watched_tokens=self._mineable_tokens))
        # Chain facts the program stream does not carry: what landing costs
        # right now, whether the chain is keeping up, whether the supply is
        # still under someone's control, and what the deployer did before.
        self.miner_registration.update(register_chain_miners(
            self.data_miners, rpc=self.solana_rpc,
            hot_accounts=self._contended_accounts,
            lp_mints=self._known_lp_mints,
            deployers=self._known_deployers,
            watched_wallets=self._tracked_wallets))
        # Public web: measured attention rather than mentions, and the corpus
        # a name has to be compared against before it means anything.
        self.miner_registration.update(register_web_miners(
            self.data_miners, http=self.http_client,
            search_terms=self._name_search_terms,
            youtube_key=lambda: os.getenv("YOUTUBE_API_KEY", ""),
            github_token=lambda: os.getenv("GITHUB_TOKEN", "")))
        # Global breadth. Every endpoint below sits behind a substitution
        # ladder rather than being named in a miner, so a public source that
        # refuses this address or moves a path costs the desk one pass rather
        # than a whole domain. Regional venues are in there deliberately: a
        # desk reading only two US aggregators is blind for the hours the
        # Asian session is the one that leads.
        self.substitution = default_registry()
        self.miner_registration.update(register_regional_miners(
            self.data_miners, http=self.http_client, rpc=self.solana_rpc,
            registry=self.substitution,
            watched_tokens=self._mineable_tokens,
            tracked_wallets=self._tracked_wallets,
            on_discovery=self._ingest_discovered_pools))
        # Public Telegram. The channel book starts empty and fills itself from
        # t.me links the desk already mines, verifying each handle by fetching
        # its own public preview before anything is read from it. Nothing here
        # holds a credential, so nothing here can open a private channel.
        self.channel_book = ChannelBook(
            path=str(self.global_config.get("telegram_channel_book",
                                            "data/telegram/channels.json")))
        self.miner_registration.update(register_telegram_miners(
            self.data_miners, http=self.http_client, book=self.channel_book,
            on_message=self._ingest_telegram_messages))
        # Who a launch claims to be, and whether that name ever confirmed it.
        # The registry ships with names and no channels, which means every
        # claim reads UNCORROBORATED until an operator fills in handles from
        # the figures' own verified profiles -- reported as DEGRADED rather
        # than passed off as a clean read.
        self.identity_watch = IdentityWatch()
        figures = self.identity_watch.load_yaml(
            str(self.global_config.get("figure_registry", "config/figures.yaml")))
        seeded = self.channel_book.seed(
            (handle for figure in self.identity_watch.figures.values()
             for handle in figure.channels), source="figures")
        logger.info("IDENTITY WATCH %d figure(s), %d owned channel(s) seeded",
                    figures, seeded)
        # Miners run on their OWN loop in their own thread. The expensive
        # part of a miner is not the socket -- that yields properly -- it is
        # the synchronous JSON and HTML parsing afterwards, and under the GIL
        # with no pre-emption that parse does not slow the decision path, it
        # stops it. Records come back through a bounded queue that a drainer
        # empties on this loop, so every mutation of desk state still happens
        # here and only the parsing moved.
        self.miner_offload = OffloadedPool(
            self.data_miners, sink=self._ingest_mined_records,
            queue_depth=int(self.global_config.get("miner_queue_depth", 4096)),
            name="miners")
        # The chain receive path in Rust: socket, HTTP/2, prost decode,
        # program filter and signature dedupe, with only what survives
        # crossing into Python. Runs in SHADOW beside the grpc.aio client,
        # which stays the reference -- the same bargain the Rust transaction
        # builder made, because a receiver that is faster on what it catches
        # and blind to one launch in a thousand is not an improvement.
        self.native_ingress = self._optional(
            "native ingress", lambda: None,
            lambda: NativeIngress(
                endpoint=os.getenv("YELLOWSTONE_GRPC_URL", ""),
                token=os.getenv("YELLOWSTONE_GRPC_TOKEN", ""),
                programs=tuple(self.global_config.get(
                    "native_ingress_programs",
                    ("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
                     "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"))),
                mode=str(self.global_config.get(
                    "native_ingress_mode", "SHADOW")).upper(),
                promote_after=int(self.global_config.get(
                    "native_ingress_promote_after", 5000))))
        # The heavy half, in its own INTERPRETER rather than its own thread.
        # A thread converts a multi-megabyte parse into interruptible slices;
        # it still holds the GIL against the decision path. A process does
        # not. The miners that move are the ones with no desk dependency,
        # which -- from the desk's own report -- are also the expensive ones:
        # venue_tickers 401 records a pass, regional_venues 331,
        # jupiter_tokens 3,174.
        #
        # Off by default. It is a real behaviour change on a box with two
        # vCPUs, and it should be turned on against a measured p99 rather
        # than on the assumption that isolation is free.
        self.context_offload: Optional[ProcessOffloadedPool] = None
        if bool(self.global_config.get("offload_context_miners", False)):
            self.context_offload = self._optional(
                "context miner process", lambda: None,
                lambda: ProcessOffloadedPool(
                    "src.research.context_pool.build_context_pool",
                    {"concurrency": int(self.global_config.get(
                        "context_miner_concurrency", 4)),
                     "search_terms": list(self.global_config.get(
                         "context_search_terms", ("pump.fun", "solana"))),
                     "youtube_key": os.getenv("YOUTUBE_API_KEY", ""),
                     "github_token": os.getenv("GITHUB_TOKEN", "")},
                    sink=self._ingest_mined_records,
                    queue_depth=int(self.global_config.get(
                        "miner_queue_depth", 4096)),
                    affinity=tuple(self.global_config.get(
                        "context_miner_cpus", ()) or ()),
                    name="context"))
        if not self.offline:
            await self.dataset_builder.start()
            await self.global_research.start()
            if self.context_offload is not None:
                await self.context_offload.start()
            if bool(self.global_config.get("offload_miners", True)):
                await self.miner_offload.start()
                started = len(self.miner_registration)
            else:
                started = await self.data_miners.start()
            logger.info("DATA MINERS %d runnable of %d declared (%s)",
                        started, len(self.miner_registration),
                        "offloaded thread" if self.miner_offload.report()["status"] != "OFF"
                        else "main loop")
            # Event callbacks write to the PIT builder, hazard tracker, and
            # research graphs. Start the stream only after all consumers exist.
            if self.rpc_program_stream:
                await self.rpc_program_stream.start()
