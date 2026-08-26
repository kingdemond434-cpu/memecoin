import ast
import dataclasses
import asyncio
import base64
import gzip
import hashlib
import json
import functools
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from solders.hash import Hash
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.message import MessageV0
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from src.chains.provider_credentials import extract_provider_key
from src.chains.rpc_manager import ChainConfig, ChainRegistry, ChainType, RPCEndpointConfig, RPCHealth, RPCManager
from src.chains.yellowstone_grpc import (
    SYSTEM_PROGRAM, PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream,
    YellowstoneClient, b58encode, create_combined_subscription, enrich_trade_balances,
    extract_system_transfers,
)
from src.detection.rug_detector import TOKEN_2022_PROGRAM, TOKEN_PROGRAM, RugDetector
from src.detection.token_detector import DetectionSource, TokenCandidate
from src.execution.landing_model import (
    BID_BUCKETS, Attempt, LandingModel, bid_bucket, congestion_bucket,
)
from src.execution.jupiter_jito import (
    ExecutionEngine, ExecutionResult, JitoClient, JupiterClient, RouteType, SolanaTransactionBuilder, SwapQuote,
    SwapTransaction, TransactionStatus,
)
from src.main import CAPACITY_REJECTIONS, WSOL_MINT, MemecoinQuantDesk, _jsonable
from src.strategies.information_graph import CounterfactualExecutionLab
from src.strategies.age_banded import (
    BAND_NAMES, AgeBandedPredictor, band_model_dir,
)
from src.strategies.multihead_predictor import (
    AGE_BANDS, SURVIVAL_LEVELS, ElogwEngine, MultiHeadPrediction,
    MultiHeadPredictor, PredictionFeatures, band_for,
)
from src.collectors.registry import (
    ADAPTER_KINDS, SourceDeclaration, SourceDiscovery, build_sources,
    load_declarations,
)
from src.collectors.adapters import (
    bluesky_source, code_repository_source, coverage_report, discord_gateway_source,
    farcaster_source, mastodon_source, metadata_artifact_source, nostr_source,
    official_site_source, rss_source, telegram_source, twitch_eventsub_source,
    youtube_websub_source,
)
from src.collectors.event_source import (
    Event, EventSource, SourceClass, SourceMesh, SourceState,
)
from src.collectors.transports import (
    BlueskyJetstreamTransport, GithubRepoTransport, JsonPollTransport,
    MastodonTimelineTransport, NostrRelayTransport, OfficialSiteTransport,
    QueueTransport, RssTransport, TelegramChannelTransport, TransportError,
    build_transport, build_transports, parse_timestamp, transport_report,
)
from src.strategies.decision_snapshot import (
    DecisionSnapshot, DecisionStatus, StateSequencer, guard as decision_guard,
    state_hash,
)
from src.strategies.action_value import (
    Action, Action as ActionValue, ActionValuePolicy, Decision,
    Decision as ActionDecision, PositionState, PositionState as ActionState,
)
from src.strategies.actor_graph import (
    BuyerDNA, BuyerFingerprint, Entry, IndependenceReport, SwarmPredictor,
    WalletIndependence, aggregate_smart_flow, build_fingerprint,
)
from src.strategies.authenticity import (
    ENTITY_STALE_AFTER_DAYS, AuthenticityResolver, AuthenticityVerdict,
    EntityRegistry, ProofLevel, SourceSignal, WatchedEntity, extract_mints,
    host_matches, load_entities, looks_like_mint, rank_copycats,
)
from tools import verify_entities
from src.strategies.escape import (
    HAZARD_HORIZONS, TRIGGER_MECHANISMS, UNESCAPABLE_MECHANISMS, HazardCurve,
    HazardMechanism, LandingLatency, escape_probability,
    hazard_curve_from_probabilities, liquidation_ladder,
    mechanisms_from_signals, ride_or_reject,
)
from src.strategies.distribution import (
    DISTRIBUTION_FEATURE_NAMES, DISTRIBUTION_HORIZONS, DistributionDetector,
    distribution_features,
)
from src.strategies.source_genealogy import (
    PostOutcome, SourceGenealogy, SourcePost, build_source_dna, rank_sources,
    source_value,
)
from src.strategies.mega_event import (
    ESCALATION_LADDER, AudienceTier, MegaEventReserve, plan_capacity_escalation,
    remaining_audience,
)
from src.strategies.monster import (
    MONSTER_STATES, MonsterEvidence, MonsterState, MonsterStateMachine,
    hold_versus_exit, premature_exit_rates, tail_capture_ratio,
)
from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator
from src.runtime.intelligence_manifest import (
    ENTRY_CONTRIBUTORS, POSITION_CONTRIBUTORS, CoverageTracker, audit as audit_intelligence,
)
from src.strategies.reentry import (
    BARRED_DISPOSITIONS, ExitDisposition, ReentryBook, ReentryPolicy, classify_exit,
)
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.genealogy_graph import WalletProfile
from src.strategies.wallet_intelligence import (
    WalletIntelligenceEngine, WalletRegime, WalletScore,
)
from src.research.dataset_builder import (
    SNAPSHOT_OFFSETS_S, TAIL_THRESHOLDS, LaunchEpisode, LaunchSnapshot,
    PointInTimeDatasetBuilder, SnapshotTimepoint,
)
from src.research.shadow_trainer import (
    SNAPSHOT_ORDER, chronological_episode_split, train_age_bands, train_shadow,
)
from src.chains.idl import (
    PUMP_AMM_IDL, PUMP_FEES_IDL, PUMP_IDL, IdlError, account_names, build_accounts,
    discriminator, encode_u64_args, instruction, load_idl, program_id, unresolvable,
)
from src.chains.pump_fee_config import (
    FEE_CONFIG_DISCRIMINATOR, bonding_curve_market_cap, calculate_fee_tier,
    fee_config_address, parse_fee_config, pool_market_cap,
)
from src.research.band_split import evaluate_cuts, split_warrant
from src.strategies.t0_kernel import KernelMode, SurvivalInputs, T0Kernel

try:  # The native kernel is optional; the Python path is the reference.
    import solana_fastpath as _solana_fastpath  # noqa: F401

    _NATIVE_KERNEL = True
except ImportError:  # pragma: no cover - depends on the build
    _NATIVE_KERNEL = False
from src.strategies import multihead_predictor as multihead_predictor_module
from src.strategies.wallet_value import (
    FollowOutcome, WalletValue, WalletValueModel, executable_multiple,
)
from src.chains.blockhash import BlockhashCache, BlockhashState
from src.chains.pumpswap_curve import (
    PumpSwapPoolState, sell_capacity_base,
)
from src.chains.pumpswap_curve import quote_buy as pool_quote_buy
from src.chains.pumpswap_curve import quote_sell as pool_quote_sell
from src.chains.pumpswap_route import (
    POOL_DISCRIMINATOR, PoolState, PumpSwapRoute, derive_pool, parse_pool,
)
from src.chains.pump_route import (
    PUBLISHED_GLOBAL, PUMP_PROGRAM, ROUTE_STATUS, TOKEN_2022_PROGRAM, TOKEN_PROGRAM,
    NativePumpRoute, PumpRouteConfig, associated_token_address, derived_global,
    fee_recipients, select_buyback_recipient, select_fee_recipient,
)
from src.chains.pump_curve import (
    BONDING_CURVE_DISCRIMINATOR, observation_from_state, parse_bonding_curve,
    quote_buy, quote_sell, sell_capacity_lamports,
)
from src.execution.tradeability import (
    Frontier, TradeabilityReport, build_frontier, curve_tradeability,
    exit_capacity_ratio,
)
from src.execution.pump_fees import (
    DYNAMIC_FEE_ACTIVATION_UTC, LEGACY_TOTAL_FEE_BPS, PumpFeeSchedule,
    VENUE_BONDING_CURVE, VENUE_PUMPSWAP_CANONICAL,
)
from src.chains.pump_curve import (
    DEFAULT_FEE_BPS, BondingCurveState, quote_buy, quote_sell, resolve_fee_bps,
)
from ops.audit_pack import build_audit_pack
from ops.health import (
    Check, HealthReport, State, check_intelligence_coverage, run_health_checks,
)
from ops.monitor import main as monitor_main
from src.runtime.hot_state import (
    AsyncArchiveWriter, CompactWalletDNA, EconomicCache, HotState, HotStateBudget,
)
from src.research.promotion_gate import (
    DEFAULT_CRITERIA, Evidence, PromotionCriteria, PromotionLedger, Stage,
    can_advance, evaluate, next_stage,
)
from src.research.backfill import (
    BACKFILL_PROVENANCE, PROVENANCE_KEY, Limitation, RawLaunch,
    is_reconstructed, partition_by_provenance, reconstruct, run_backfill,
    stamp_live,
)
from src.research.action_value_trainer import (
    PolicyMetrics, chronological_split, evaluate_candidate, measure_policy,
    save_report, select_policy, tail_preservation_gate,
)
from src.research.lifecycle_replay import (
    DEFAULT_DELAYS_S, DEFAULT_EXIT_RULES, Cell, Lifecycle, Mark,
    delay_decay, hold_to_end, lifecycle_from_episode, replay_cell,
    replay_lifecycle, sniper_scoreboard,
)
from src.research.forward_evidence import ForwardEvidence, Outcome
from src.research.contribution import (
    Contribution, ContributionLedger, DecisionContribution, GateFlip,
    action_value_contributions,
)
from src.research.attribution import (
    EdgeDecayMonitor, Leak, alpha_ledger, execution_miss, find_leaks,
    missed_monster, premature_exit, rank_research, rug_loss, sizing_leak,
    tail_contribution,
)
from src.research.feature_engine import build_features
from src.research import hazard_trainer
from src.research.exit_policy_trainer import simulate, train_exit_policy
from src.strategies.exit_policy import ExitPolicy, evaluate_exit, load_latest_exit_policy
from src.research.global_research_miner import GlobalResearchMiner, ResearchLead
from src.strategies.champion_challenger import ChampionChallengerFramework, HypothesisSpec, ModelStatus, TrialResult
from src.strategies.rug_hazard import (
    ContinuousRugHazardModel, HAZARD_FEATURE_NAMES,
)
from src.strategies.genealogy_graph import GenealogyGraph, RelationshipType
from src.strategies.social_intelligence import SocialAccount, SocialIntelligenceEngine, SocialPlatform


def solana_chain():
    return ChainConfig(
        name="Solana Mainnet", chain_id="solana", chain_type=ChainType.SOLANA,
        rpc_endpoints=[RPCEndpointConfig("https://example.invalid")], explorer_api="", explorer_key="",
        native_token="SOL", decimals=9, block_time=0.4, factories={}, routers={}, base_tokens=[],
        min_liquidity_usd=2_000, max_tax=0, honeypot_check=False, programs={},
    )


class TestProviderCredentials(unittest.TestCase):
    def test_extracts_helius_key_from_rpc_url(self):
        self.assertEqual(
            extract_provider_key("https://mainnet.helius-rpc.com/?api-key=helius-secret", "helius"),
            "helius-secret",
        )

    def test_extracts_alchemy_key_from_rpc_url(self):
        self.assertEqual(
            extract_provider_key("https://solana-mainnet.g.alchemy.com/v2/alchemy-secret", "alchemy"),
            "alchemy-secret",
        )

    def test_preserves_bare_keys(self):
        self.assertEqual(extract_provider_key("bare-secret", "helius"), "bare-secret")


class DummyYellowstone:
    def __init__(self):
        self.handlers = {}

    def on(self, event_type, handler):
        self.handlers[event_type] = handler


class TestSolanaParsing(unittest.IsolatedAsyncioTestCase):
    def test_pump_trade_program_event_decodes_without_http_transaction_fetch(self):
        monitor = PumpFunMonitor(DummyYellowstone(), lambda _: None)
        mint_raw = bytes(range(1, 33))
        user_raw = bytes(range(33, 65))
        data = (
            PumpFunMonitor.TRADE_EVENT + mint_raw + struct.pack("<QQ?", 2_000_000_000, 5_000_000, True)
            + user_raw + struct.pack("<qQQ", 1_700_000_000, 30_000_000_000, 10_000_000_000)
        )
        event = monitor._decode_program_event(data, "event-sig", 98)
        from src.chains.yellowstone_grpc import b58encode
        self.assertEqual(event["token"], b58encode(mint_raw))
        self.assertEqual(event["wallet"], b58encode(user_raw))
        self.assertEqual(event["side"], "buy")
        self.assertEqual(event["notional_sol"], 2.0)
        self.assertEqual(event["curve_price_raw"], 3.0)
        self.assertEqual(event["fill_data_status"], "OBSERVED_PROGRAM_EVENT")

    def test_pumpswap_program_events_populate_pool_then_trade(self):
        monitor = PumpSwapMonitor(DummyYellowstone(), lambda _: None)
        creator, mint, quote, pool, user = (bytes([value]) * 32 for value in range(1, 6))
        create = bytearray(205)
        create[:8] = PumpSwapMonitor.CREATE_POOL_EVENT
        struct.pack_into("<qH", create, 8, 1_700_000_000, 7)
        create[18:50], create[50:82], create[82:114] = creator, mint, quote
        create[114], create[115] = 6, 9
        struct.pack_into("<QQ", create, 116, 5_000_000, 2_000_000_000)
        create[173:205] = pool
        created = monitor._decode_program_event(bytes(create), "create-sig", 99)
        self.assertEqual(created["initial_base_amount"], 5_000_000)

        buy = bytearray(184)
        buy[:8] = PumpSwapMonitor.BUY_EVENT
        struct.pack_into("<qQ", buy, 8, 1_700_000_001, 1_000_000)
        struct.pack_into("<QQ", buy, 48, 10_000_000, 20_000_000_000)
        struct.pack_into("<Q", buy, 64, 500_000_000)
        buy[120:152], buy[152:184] = pool, user
        traded = monitor._decode_program_event(bytes(buy), "buy-sig", 100)
        self.assertEqual(traded["token"], created["token"])
        self.assertEqual(traded["side"], "buy")
        self.assertEqual(traded["actual_token_amount_ui"], 1.0)
        self.assertEqual(traded["curve_price_raw"], 2_000.0)

    def test_websocket_launch_filter_rejects_position_noise(self):
        self.assertTrue(SolanaRpcProgramStream._looks_like_pool_creation([
            "Program log: Instruction: InitializePoolV2",
        ]))
        self.assertFalse(SolanaRpcProgramStream._looks_like_pool_creation([
            "Program log: Instruction: CreatePosition",
        ]))

    async def test_real_historical_pump_sell_inner_instruction(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "pump_sell_441417557.json").read_text())
        fixture["blockTime"] = 1_700_000_000
        events = []
        monitor = PumpFunMonitor(DummyYellowstone(), events.append)
        await monitor._on_transaction(fixture)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["side"], "sell")
        self.assertEqual(events[0]["token"], "FySyjuXTts9mTz2wjyuSXAz4bEBv6v5qxCTcLAMd4mVX")
        self.assertEqual(events[0]["wallet"], "V4SyF9Vv3EgzPyUEQQckN8b5eWSRndu9j2584WpnkCg")
        self.assertEqual(events[0]["slot"], 441417557)
        self.assertEqual(events[0]["signature"], fixture["signature"])
        self.assertEqual(events[0]["timestamp"], 1_700_000_000)
        self.assertEqual(events[0]["block_time"], 1_700_000_000)
        self.assertLessEqual(events[0]["received_ns"], events[0]["decoded_ns"])

    def test_extract_system_transfers_decodes_native_transfer_instruction(self):
        keys = ["funder", "recipient", "unrelated-program", SYSTEM_PROGRAM]
        transfer_data = struct.pack("<IQ", 2, 5_000_000_000)
        instructions = [
            {"programIdIndex": 3, "accounts": [0, 1], "data": b58encode(transfer_data)},
            {"programIdIndex": 2, "accounts": [0, 1], "data": b58encode(b"not a transfer")},
        ]
        transfers = extract_system_transfers(keys, instructions)
        self.assertEqual(transfers, [{"from": "funder", "to": "recipient", "lamports": 5_000_000_000}])

    async def test_pump_create_attaches_same_transaction_funding_wallets(self):
        events = []
        monitor = PumpFunMonitor(DummyYellowstone(), events.append)
        creator_raw = bytes(range(1, 33))
        creator_address = b58encode(creator_raw)
        keys = ["mint-key", "funder-wallet", "curve-key", creator_address,
               PumpFunMonitor.PUMP_FUN_PROGRAM, SYSTEM_PROGRAM]

        def length_prefixed(value: str) -> bytes:
            encoded = value.encode("utf-8")
            return struct.pack("<I", len(encoded)) + encoded

        payload = length_prefixed("Test Token") + length_prefixed("TEST") + length_prefixed("uri://x") + creator_raw
        discriminator = {name: disc for disc, name in PumpFunMonitor.DISCRIMINATORS.items()}["create"]
        tx_data = {
            "signature": "create-sig", "slot": 12345,
            "transaction": {
                "signatures": ["create-sig"],
                "message": {
                    "accountKeys": keys,
                    "instructions": [
                        {"programIdIndex": 5, "accounts": [1, 3],
                         "data": b58encode(struct.pack("<IQ", 2, 5_000_000_000))},
                        {"programIdIndex": 4, "accounts": [0, 1, 2], "data": b58encode(discriminator + payload)},
                    ],
                },
            },
            "meta": {"innerInstructions": []},
        }
        await monitor._on_transaction(tx_data)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "token_created")
        self.assertEqual(event["token"], "mint-key")
        self.assertEqual(event["bonding_curve"], "curve-key")
        self.assertEqual(event["creator"], creator_address)
        self.assertEqual(event["funding_wallets"], ["funder-wallet"])
        self.assertEqual(event["funding_transfers"],
                         [{"from": "funder-wallet", "to": creator_address, "lamports": 5_000_000_000}])

    async def test_raydium_pool_created_attaches_same_transaction_funding_wallets(self):
        events = []
        monitor = RaydiumMonitor(DummyYellowstone(), events.append)
        keys = [f"key{i}" for i in range(9)]
        keys[0] = "creator-wallet"   # CPMM account index 0
        keys[3] = "pool-key"         # CPMM account index 3
        keys[4] = "mint-a"           # CPMM account index 4
        keys[5] = "mint-b"           # CPMM account index 5
        keys[6] = "funder-wallet"
        keys[7] = RaydiumMonitor.RAYDIUM_CPMM
        keys[8] = SYSTEM_PROGRAM
        data = RaydiumMonitor.CPMM_INITIALIZE + struct.pack("<QQQ", 75_000, 50_000, 1234)
        tx_data = {
            "signature": "cpmm-sig", "slot": 555,
            "transaction": {
                "signatures": ["cpmm-sig"],
                "message": {
                    "accountKeys": keys,
                    "instructions": [
                        {"programIdIndex": 8, "accounts": [6, 0],
                         "data": b58encode(struct.pack("<IQ", 2, 2_000_000_000))},
                        {"programIdIndex": 7, "accounts": list(range(9)), "data": b58encode(data)},
                    ],
                },
            },
            "meta": {"innerInstructions": []},
        }
        await monitor._on_transaction(tx_data)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "pool_created")
        self.assertEqual(event["creator"], "creator-wallet")
        self.assertEqual(event["pool"], "pool-key")
        self.assertEqual(event["mint_a"], "mint-a")
        self.assertEqual(event["mint_b"], "mint-b")
        self.assertEqual(event["funding_wallets"], ["funder-wallet"])
        self.assertEqual(event["funding_transfers"],
                         [{"from": "funder-wallet", "to": "creator-wallet", "lamports": 2_000_000_000}])

    async def test_raydium_v4_initialize2_layout(self):
        keys = [f"key{i}" for i in range(18)]
        accounts = list(range(18))
        payload = struct.pack("<BQQQ", 7, 1234, 50_000, 75_000)
        event = RaydiumMonitor._decode_v4(1, keys, accounts, payload, "sig", 99)
        self.assertEqual(event["pool"], "key4")
        self.assertEqual(event["mint_a"], "key8")
        self.assertEqual(event["mint_b"], "key9")
        self.assertEqual(event["creator"], "key17")

    def test_raydium_cpmm_initialize_uses_official_idl_layout(self):
        keys = [f"key{i}" for i in range(20)]
        data = RaydiumMonitor.CPMM_INITIALIZE + struct.pack("<QQQ", 75_000, 50_000, 1234)
        event = RaydiumMonitor._decode_cpmm(keys, list(range(20)), data, "cpmm-sig", 101)
        self.assertEqual(event["program"], RaydiumMonitor.RAYDIUM_CPMM)
        self.assertEqual(event["pool"], "key3")
        self.assertEqual(event["mint_a"], "key4")
        self.assertEqual(event["mint_b"], "key5")
        self.assertEqual(event["creator"], "key0")
        self.assertEqual(event["initial_base_amount"], 75_000)
        self.assertEqual(event["initial_quote_amount"], 50_000)
        self.assertEqual(event["data_status"], "OK")

    def test_raydium_clmm_create_pool_uses_official_idl_layout(self):
        keys = [f"key{i}" for i in range(13)]
        sqrt_price_x64 = 1 << 64
        data = RaydiumMonitor.CLMM_CREATE_POOL + sqrt_price_x64.to_bytes(16, "little") + struct.pack("<Q", 1234)
        event = RaydiumMonitor._decode_clmm(keys, list(range(13)), data, "clmm-sig", 102)
        self.assertEqual(event["program"], RaydiumMonitor.RAYDIUM_CLMM)
        self.assertEqual(event["pool"], "key2")
        self.assertEqual(event["mint_a"], "key3")
        self.assertEqual(event["mint_b"], "key4")
        self.assertEqual(event["creator"], "key0")
        self.assertEqual(event["sqrt_price_x64"], sqrt_price_x64)
        self.assertEqual(event["data_status"], "OK")

    def test_raydium_official_program_addresses_are_subscribed(self):
        programs = create_combined_subscription().transactions["memecoin_programs"].accounts_include
        self.assertIn(RaydiumMonitor.RAYDIUM_CPMM, programs)
        self.assertIn(RaydiumMonitor.RAYDIUM_CLMM, programs)

    def test_meteora_dlmm_initialize_uses_official_idl_layout(self):
        keys = [f"key{i}" for i in range(14)]
        data = RaydiumMonitor.METEORA_DLMM_INITIALIZE + struct.pack("<iH", -25, 10)
        event = RaydiumMonitor._decode_meteora_dlmm(keys, list(range(14)), data, "dlmm-sig", 103)
        self.assertEqual(event["pool"], "key0")
        self.assertEqual(event["mint_a"], "key2")
        self.assertEqual(event["mint_b"], "key3")
        self.assertEqual(event["creator"], "key8")
        self.assertEqual(event["active_id"], -25)
        self.assertEqual(event["bin_step"], 10)

    def test_meteora_dynamic_amm_uses_official_idl_layout(self):
        keys = [f"key{i}" for i in range(27)]
        discriminator = bytes((118, 173, 41, 157, 173, 72, 97, 103))
        event = RaydiumMonitor._decode_meteora_amm(keys, list(range(27)), discriminator, "amm-sig", 104)
        self.assertEqual(event["pool"], "key0")
        self.assertEqual(event["mint_a"], "key2")
        self.assertEqual(event["mint_b"], "key3")
        self.assertEqual(event["creator"], "key17")

    def test_orca_v2_initialize_uses_official_program_layout(self):
        keys = [f"key{i}" for i in range(14)]
        sqrt_price_x64 = 1 << 64
        data = RaydiumMonitor.ORCA_INITIALIZE_POOL_V2 + struct.pack("<H", 64) + sqrt_price_x64.to_bytes(16, "little")
        event = RaydiumMonitor._decode_orca(keys, list(range(14)), data, "orca-sig", 105)
        self.assertEqual(event["pool"], "key6")
        self.assertEqual(event["mint_a"], "key1")
        self.assertEqual(event["mint_b"], "key2")
        self.assertEqual(event["creator"], "key5")
        self.assertEqual(event["tick_spacing"], 64)
        self.assertEqual(event["sqrt_price_x64"], sqrt_price_x64)

    def test_all_native_amm_programs_are_subscribed(self):
        programs = create_combined_subscription().transactions["memecoin_programs"].accounts_include
        self.assertIn(RaydiumMonitor.METEORA_DLMM, programs)
        self.assertIn(RaydiumMonitor.METEORA_DYNAMIC_AMM, programs)
        self.assertIn(RaydiumMonitor.ORCA_WHIRLPOOL, programs)

    async def test_real_historical_native_pool_initializers(self):
        fixtures = {
            "raydium_cpmm_441464653.json": RaydiumMonitor.RAYDIUM_CPMM,
            "raydium_clmm_441462011.json": RaydiumMonitor.RAYDIUM_CLMM,
            "meteora_amm_441464551.json": RaydiumMonitor.METEORA_DYNAMIC_AMM,
            "orca_441468893.json": RaydiumMonitor.ORCA_WHIRLPOOL,
        }
        for filename, program in fixtures.items():
            with self.subTest(filename=filename):
                fixture = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())
                events = []
                monitor = RaydiumMonitor(DummyYellowstone(), events.append)
                await monitor._on_transaction(fixture)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["program"], program)
                self.assertEqual(events[0]["type"], "pool_created")
                self.assertEqual(events[0]["data_status"], "OK")
                self.assertEqual(events[0]["signature"], fixture["signature"])
                self.assertEqual(events[0]["slot"], fixture["slot"])

    async def test_pumpswap_uses_native_amm_account_layout(self):
        keys = [f"key{i}" for i in range(24)] + [PumpSwapMonitor.PUMP_AMM_PROGRAM]
        program_index = len(keys) - 1
        accounts = list(range(23))
        instruction = {
            "programIdIndex": program_index,
            "accounts": accounts,
            "data": "",  # populated below with the on-chain base58 encoding
        }
        from src.chains.yellowstone_grpc import b58encode
        instruction["data"] = b58encode(bytes((102, 6, 61, 18, 1, 218, 235, 234)) + struct.pack("<QQB", 50, 75, 1))
        tx = {"slot": 10, "transaction": {"signatures": ["sig"],
              "message": {"accountKeys": keys, "instructions": [instruction]}}, "meta": {}}
        events = []
        monitor = PumpSwapMonitor(DummyYellowstone(), events.append)
        await monitor._on_transaction(tx)
        self.assertEqual(events[0]["type"], "token_trade")
        self.assertEqual(events[0]["pool"], "key0")
        self.assertEqual(events[0]["wallet"], "key1")
        self.assertEqual(events[0]["token"], "key3")
        self.assertEqual(events[0]["quote_mint"], "key4")

    def test_pump_migrate_uses_official_idl_indices(self):
        keys = [f"key{i}" for i in range(25)]
        event = PumpFunMonitor(DummyYellowstone(), lambda _: None)._decode_instruction(
            "migrate", keys, list(range(25)), b"", "sig", 11,
        )
        self.assertEqual(event["token"], "key2")
        self.assertEqual(event["pool"], "key9")
        self.assertEqual(event["wallet"], "key5")

    def test_trade_event_uses_observed_balance_deltas_not_instruction_limit(self):
        event = {"type": "token_trade", "side": "buy", "token": "mint", "wallet": "wallet",
                 "quote_limit_amount": 999_999_999}
        tx = {"meta": {
            "fee": 5_000,
            "preBalances": [2_000_000_000], "postBalances": [1_499_995_000],
            "preTokenBalances": [{"mint": "mint", "owner": "wallet",
                                   "uiTokenAmount": {"amount": "0", "decimals": 2}}],
            "postTokenBalances": [{"mint": "mint", "owner": "wallet",
                                    "uiTokenAmount": {"amount": "2500", "decimals": 2}}],
        }}
        enrich_trade_balances(event, tx, ["wallet"])
        self.assertEqual(event["fill_data_status"], "OBSERVED_WALLET_BALANCE_DELTA")
        self.assertEqual(event["actual_token_amount_ui"], 25)
        self.assertEqual(event["notional_sol"], 0.5)


class TestOfficialSocialCollectors(unittest.IsolatedAsyncioTestCase):
    def make_engine(self, api_keys=None):
        wallet_intel = SimpleNamespace(get_top_wallets=lambda limit=100: [], register_social_wallet=lambda wallet: None)
        return SocialIntelligenceEngine(
            solana_chain(), SimpleNamespace(request=self._rpc_request), SimpleNamespace(), wallet_intel,
            api_keys or {},
        )

    async def _rpc_request(self, method, params):
        if method == "getTokenLargestAccounts":
            return {"value": []}
        raise AssertionError(f"unexpected RPC method: {method}")

    async def test_youtube_video_extracts_real_contract_and_engagement(self):
        engine = self.make_engine()
        account = SocialAccount(SocialPlatform.YOUTUBE, "channel", "channel", "Channel")
        mint = "FySyjuXTts9mTz2wjyuSXAz4bEBv6v5qxCTcLAMd4mVX"
        await engine._process_youtube_video(account, {
            "id": {"videoId": "video1"},
            "snippet": {
                "title": f"New Solana launch {mint}", "description": "observed, not endorsed",
                "publishedAt": "2026-01-01T00:00:00Z",
            },
        }, {"viewCount": "12", "likeCount": "3", "commentCount": "2"})
        self.assertEqual(len(engine.mentions), 1)
        self.assertEqual(engine.mentions[0].token, mint)
        self.assertEqual(engine.mentions[0].engagement["views"], 12)
        self.assertEqual(engine.mentions[0].url, "https://www.youtube.com/watch?v=video1")

    def test_youtube_discovery_rotates_foreign_language_queries(self):
        queries = " ".join(SocialIntelligenceEngine.YOUTUBE_QUERIES)
        self.assertIn("模因币", queries)
        self.assertIn("ミームコイン", queries)
        self.assertIn("밈코인", queries)

    async def test_reddit_stays_blocked_without_approved_credentials(self):
        engine = self.make_engine()
        self.assertIsNone(await engine._reddit_headers())
        self.assertIn("REDDIT_CLIENT_ID/SECRET", engine.data_status["reddit"])

    async def test_telegram_channels_are_normalized(self):
        engine = self.make_engine()
        engine.api_keys["telegram_channels"] = "@alpha, https://t.me/beta"
        values = [
            value.strip().lstrip("@").replace("https://t.me/", "")
            for value in engine.api_keys["telegram_channels"].split(",") if value.strip()
        ]
        self.assertEqual(values, ["alpha", "beta"])

    async def test_telegram_setup_blocks_without_credentials(self):
        engine = self.make_engine({"telegram_channels": "@alpha"})
        with patch("src.strategies.social_intelligence.TelegramClient") as client_cls:
            await engine._setup_telegram()
        client_cls.assert_not_called()
        self.assertIn("TELEGRAM_API_ID/API_HASH missing", engine.data_status["telegram"])
        self.assertEqual(engine.accounts, {})

    async def test_telegram_setup_blocks_and_disconnects_when_session_unauthorized(self):
        engine = self.make_engine({
            "telegram_api_id": "123", "telegram_api_hash": "hash", "telegram_channels": "@alpha",
        })
        fake_client = FakeTelegramClient("path", 123, "hash")
        fake_client.authorized = False
        with patch("src.strategies.social_intelligence.TelegramClient", return_value=fake_client):
            await engine._setup_telegram()
        self.assertIn("interactive Telegram authorization required", engine.data_status["telegram"])
        self.assertTrue(fake_client.disconnected)
        self.assertIsNone(engine._telegram_client)
        self.assertEqual(engine.accounts, {})

    async def test_telegram_setup_normalizes_and_registers_configured_channels(self):
        engine = self.make_engine({
            "telegram_api_id": "123", "telegram_api_hash": "hash",
            "telegram_channels": "@alpha, https://t.me/beta ,gamma",
        })
        fake_client = FakeTelegramClient("path", 123, "hash")
        with patch("src.strategies.social_intelligence.TelegramClient", return_value=fake_client):
            await engine._setup_telegram()
        self.assertIn(engine.data_status["telegram"], {"OK_PUSH", "OK_POLLING"})
        self.assertTrue(fake_client.connected)
        self.assertEqual(set(engine.accounts), {"telegram:alpha", "telegram:beta", "telegram:gamma"})
        for handle in ("alpha", "beta", "gamma"):
            self.assertEqual(engine.accounts[f"telegram:{handle}"].handle, handle)

    async def test_fetch_telegram_posts_extracts_contract_and_dedupes_read_only(self):
        engine = self.make_engine()
        engine._telegram_client = FakeTelegramClient("path", 123, "hash")
        mint = "FySyjuXTts9mTz2wjyuSXAz4bEBv6v5qxCTcLAMd4mVX"
        message = FakeTelegramMessage(1, f"gm, new call {mint}", 1_700_000_000, views=42, forwards=3)
        engine._telegram_client.messages_by_entity["alpha"] = [message]
        account = SocialAccount(SocialPlatform.TELEGRAM, "alpha", "alpha", "alpha")

        await engine._fetch_telegram_posts(account)
        self.assertEqual(engine.data_status["telegram"], "OK")
        self.assertEqual(len(engine.mentions), 1)
        mention = engine.mentions[0]
        self.assertEqual(mention.token, mint)
        self.assertEqual(mention.engagement["views"], 42)
        self.assertEqual(mention.url, "https://t.me/alpha/1")

        # A second pass over the same message must not double-count it, and
        # the collector never mutates the channel (no send/delete/react call
        # exists anywhere on FakeTelegramClient for it to reach).
        await engine._fetch_telegram_posts(account)
        self.assertEqual(len(engine.mentions), 1)
    async def test_telegram_push_and_polling_share_dedupe_path(self):
        engine = self.make_engine()
        account = SocialAccount(SocialPlatform.TELEGRAM, "alerts_bot", "1", "Alerts")
        mint = "FySyjuXTts9mTz2wjyuSXAz4bEBv6v5qxCTcLAMd4mVX"
        message = SimpleNamespace(
            id=42, message=f"new token {mint}",
            date=SimpleNamespace(timestamp=lambda: 1_700_000_000.0),
            views=3, forwards=1, replies=None,
        )
        await engine._process_telegram_message(account, message)
        await engine._process_telegram_message(account, message)
        self.assertEqual(len(engine.mentions), 1)
        self.assertEqual(engine.mentions[0].token, mint)
        self.assertEqual(engine.mentions[0].engagement["views"], 3)

    async def test_one_invalid_telegram_channel_does_not_mask_healthy_ingestion(self):
        engine = self.make_engine()

        class TelegramStub:
            async def iter_messages(self, handle, limit):
                if handle == "missing":
                    raise ValueError("No user has missing as username")
                if False:
                    yield None

        engine._telegram_client = TelegramStub()
        healthy = SocialAccount(SocialPlatform.TELEGRAM, "healthy", "1", "Healthy")
        missing = SocialAccount(SocialPlatform.TELEGRAM, "missing", "2", "Missing")
        await engine._fetch_telegram_posts(healthy)
        await engine._fetch_telegram_posts(missing)
        self.assertEqual(engine.data_status["telegram"], "OK_PARTIAL: 1 configured channels unavailable")

    async def test_numeric_telegram_entity_is_polled_as_integer(self):
        engine = self.make_engine()

        class TelegramStub:
            target = None

            async def iter_messages(self, target, limit):
                self.target = target
                if False:
                    yield None

        engine._telegram_client = TelegramStub()
        account = SocialAccount(SocialPlatform.TELEGRAM, "-10012345", "-10012345", "Private bot")
        await engine._fetch_telegram_posts(account)
        self.assertEqual(engine._telegram_client.target, -10012345)


class TestNativeMintChecks(unittest.TestCase):
    @staticmethod
    def mint_bytes(mint_tag=0, freeze_tag=0, supply=1_000_000, decimals=6):
        raw = bytearray(82)
        struct.pack_into("<I", raw, 0, mint_tag)
        struct.pack_into("<Q", raw, 36, supply)
        raw[44] = decimals
        raw[45] = 1
        struct.pack_into("<I", raw, 46, freeze_tag)
        return raw

    def test_legacy_spl_authorities(self):
        state = RugDetector.parse_spl_mint(bytes(self.mint_bytes(1, 0)), TOKEN_PROGRAM)
        self.assertTrue(state["mint_authority_present"])
        self.assertFalse(state["freeze_authority_present"])
        self.assertEqual(state["extensions"], [])

    def test_token_2022_permanent_delegate_extension(self):
        raw = self.mint_bytes()
        raw.extend(b"\x01")
        raw.extend(struct.pack("<HH", 12, 0))
        state = RugDetector.parse_spl_mint(bytes(raw), TOKEN_2022_PROGRAM)
        self.assertIn("permanent_delegate", state["extensions"])

    def test_invalid_authority_coption_is_rejected(self):
        with self.assertRaises(ValueError):
            RugDetector.parse_spl_mint(bytes(self.mint_bytes(2, 0)), TOKEN_PROGRAM)


class TestProbabilityAndAccounting(unittest.TestCase):
    def test_nested_probabilities_become_disjoint_distribution_with_p50(self):
        prediction = MultiHeadPrediction("mint", "solana", 0, p_2x=0.4, p_5x=0.7,
                                         p_10x=0.2, p_20x=0.1, p_50x=0.05,
                                         p_rug_5m=0.1, expected_feasible_multiple=3.0)
        bins = ElogwEngine.probability_bins(prediction)
        self.assertAlmostEqual(sum(probability for _, probability, _ in bins), 1.0)
        self.assertLessEqual(prediction.p_5x, prediction.p_2x)
        self.assertLessEqual(prediction.p_10x, prediction.p_5x)
        self.assertLessEqual(prediction.p_50x, prediction.p_20x)
        by_name = {name: probability for name, probability, _ in bins}
        # p_100x is untrained, so everything at or above 50x sits in one
        # bucket rather than being spread across a tail nobody measured.
        self.assertAlmostEqual(by_name["50x_to_100x"], 0.045)
        self.assertAlmostEqual(by_name["100x_to_250x"], 0.0)
        self.assertLessEqual(max(outcome for _, _, outcome in bins), 2.0)

    def test_the_tail_no_longer_stops_at_fifty(self):
        """A 500x priced as a 50x is the single most expensive rounding here."""
        prediction = MultiHeadPrediction(
            "mint", "solana", 0, p_2x=0.5, p_5x=0.3, p_10x=0.2, p_20x=0.12,
            p_50x=0.06, p_100x=0.03, p_250x=0.01, p_500x=0.004)
        bins = ElogwEngine.probability_bins(prediction)
        by_name = {name: (probability, outcome) for name, probability, outcome in bins}
        self.assertIn("500x_plus", by_name)
        self.assertAlmostEqual(by_name["500x_plus"][0], 0.004)
        # Each bucket pays its LOWER bound, never a midpoint.
        self.assertAlmostEqual(by_name["500x_plus"][1], 499.0)
        self.assertAlmostEqual(by_name["100x_to_250x"][1], 99.0)
        self.assertAlmostEqual(sum(probability for _, probability, _ in bins), 1.0)

    def test_untrained_tail_heads_collapse_rather_than_invent_conviction(self):
        """Extending the curve must not manufacture a tail nobody measured."""
        trained = MultiHeadPrediction("mint", "solana", 0, p_2x=0.5, p_5x=0.3,
                                      p_10x=0.2, p_20x=0.1)
        bins = ElogwEngine.probability_bins(trained)
        by_name = {name: probability for name, probability, _ in bins}
        self.assertAlmostEqual(by_name["10x_to_20x"], 0.1)
        self.assertAlmostEqual(by_name["20x_to_50x"], 0.1)
        for empty in ("50x_to_100x", "100x_to_250x", "250x_to_500x", "500x_plus"):
            self.assertAlmostEqual(by_name[empty], 0.0, msg=empty)

    def test_risk_constrained_elogw_and_partial_cost_basis(self):
        predictor = MultiHeadPredictor()
        predictor._is_trained = True
        engine = ElogwEngine(predictor, min_edge_bps=-1, drawdown_aversion_lambda=3)
        engine.portfolio_value = 10_000
        prediction = MultiHeadPrediction("mint", "solana", 0, p_2x=0.9, p_5x=0.65,
                                         p_10x=0.3, p_50x=0.05, p_rug_30s=0.01,
                                         p_rug_5m=0.02, expected_slippage=0.01)
        growth, fraction, size_sol = engine.calculate_expected_log_growth(prediction, 200, 100_000)
        self.assertGreaterEqual(fraction, 0)
        self.assertLessEqual(fraction, 0.05)
        self.assertGreaterEqual(size_sol, 0)
        engine.update_position("mint", {"size_tokens": 1_000, "remaining_cost_usd": 100,
                                         "risk_contribution": 0.02})
        engine.reduce_position("mint", 250, 25)
        engine.reduce_position("mint", 250, 25)
        position = engine.open_positions["mint"]
        self.assertEqual(position["size_tokens"], 500)
        self.assertEqual(position["remaining_cost_usd"], 50)
        self.assertAlmostEqual(position["risk_contribution"], 0.01)

    def test_feature_schema_includes_missingness_indicators(self):
        from src.strategies.multihead_predictor import PredictionFeatures
        predictor = MultiHeadPredictor()
        features = PredictionFeatures("token", "solana", 1)
        self.assertEqual(len(features.to_array()), len(predictor.feature_names))

    def test_model_artifact_requires_passed_chronological_validation(self):
        predictor = MultiHeadPredictor()
        predictor._is_trained = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                predictor.save(str(Path(directory) / "unsafe.joblib"), {"status": "REJECTED"})

    def test_model_loader_ignores_other_validated_artifact_families(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "rug-hazard-1.joblib").write_bytes(b"not a multihead artifact")
            predictor = MultiHeadPredictor(directory)
            self.assertFalse(predictor.load_latest())


class FakeExecutionEngineForExit:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.buys = []

    async def execute_sell(self, token, sold_tokens, **kwargs):
        self.calls.append((token, sold_tokens, kwargs))
        return self.result

    async def execute_swap(self, input_mint, output_mint, amount, **kwargs):
        self.buys.append((input_mint, output_mint, amount, kwargs))
        return self.result


class FakeDatasetBuilderForExit:
    def __init__(self):
        self.attempts = []
        self.counterfactuals = []

    def record_execution_attempt(self, token, attempt):
        self.attempts.append((token, attempt))

    def record_counterfactual(self, token, counterfactual):
        self.counterfactuals.append((token, counterfactual))


class FakeCounterfactualLabForExit:
    def resolve_decision(self, decision_id, pnl):
        return []



def _attach_manifest(desk):
    """Give a partial fixture desk what the coverage tracker needs.

    `_manage_positions` records manifest coverage in a `finally`, so every
    fixture that drives it needs the tracker and the per-position helper bound.
    Attaching them here rather than in each fixture means a new declared
    contributor breaks one place, not eight.
    """
    desk.position_coverage = CoverageTracker("position")
    desk._position_intelligence = (
        lambda token, position:
        MemecoinQuantDesk._position_intelligence(desk, token, position))
    desk._manage_one_position = (
        lambda token, position:
        MemecoinQuantDesk._manage_one_position(desk, token, position))
    return desk


class TestPartialExitAccounting(unittest.IsolatedAsyncioTestCase):
    def _desk(self, result):
        desk = SimpleNamespace(
            execution_engine=FakeExecutionEngineForExit(result),
            dataset_builder=FakeDatasetBuilderForExit(),
            counterfactual_lab=FakeCounterfactualLabForExit(),
            elogw_engine=ElogwEngine(None),
            sol_price_usd=150.0, total_pnl=0.0, successful_exits=0, dry_run=True,
            wallet_equity_usd=10_000.0,
            rug_hazard=SimpleNamespace(get_hazard=lambda token: None),
            reentry_book=ReentryBook(),
            landing_latency=LandingLatency(),
        )
        desk.ops_events = []
        desk._record_ops_event = (
            lambda stream, payload: desk.ops_events.append((stream, payload)))
        desk._closed_pnl = {}
        desk._mechanism_growth = {}
        desk._exit_edge_usd = (
            lambda position, exit_pct, reason, supplied=None:
            MemecoinQuantDesk._exit_edge_usd(desk, position, exit_pct, reason, supplied))
        return desk

    @staticmethod
    def _position():
        return {
            "size_tokens": 1_000, "initial_size_tokens": 1_000,
            "remaining_cost_usd": 100.0, "initial_cost_usd": 100.0,
            "risk_contribution": 0.02, "initial_risk_contribution": 0.02,
            "decision_id": "d1",
        }

    async def test_partial_exit_banks_pnl_on_only_the_sold_slice_and_leaves_remainder_open(self):
        result = ExecutionResult(
            success=True, status=TransactionStatus.SIMULATED, simulated=True,
            quoted_output_amount=150_000_000, native_balance_delta_lamports=0,
            actual_input_amount=500,
        )
        desk = self._desk(result)
        position = self._position()
        desk.elogw_engine.update_position("mint", position)

        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 0.5, "profit_ratchet_cost_recovery")

        self.assertEqual(desk.execution_engine.calls[0][1], 500)
        self.assertAlmostEqual(desk.total_pnl, 100.0)
        self.assertEqual(desk.successful_exits, 1)
        self.assertIn("mint", desk.elogw_engine.open_positions)
        remaining = desk.elogw_engine.open_positions["mint"]
        self.assertEqual(remaining["size_tokens"], 500)
        self.assertAlmostEqual(remaining["remaining_cost_usd"], 50.0)
        self.assertAlmostEqual(remaining["risk_contribution"], 0.01)
        attempt = desk.dataset_builder.attempts[-1][1]
        self.assertAlmostEqual(attempt["proceeds_usd"], 150.0)
        self.assertAlmostEqual(attempt["allocated_cost_usd"], 50.0)
        self.assertAlmostEqual(attempt["realized_pnl_usd"], 100.0)

    async def test_full_exit_closes_the_position(self):
        result = ExecutionResult(
            success=True, status=TransactionStatus.SIMULATED, simulated=True,
            quoted_output_amount=40_000_000, native_balance_delta_lamports=0,
            actual_input_amount=1_000,
        )
        desk = self._desk(result)
        position = self._position()
        desk.elogw_engine.update_position("mint", position)

        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 1.0, "hard_stop_loss")

        self.assertAlmostEqual(desk.total_pnl, -60.0)
        self.assertEqual(desk.successful_exits, 0)
        self.assertNotIn("mint", desk.elogw_engine.open_positions)

    async def test_live_exit_pnl_nets_out_the_native_sol_fee_delta(self):
        result = ExecutionResult(
            success=True, status=TransactionStatus.FILLED, filled=True, landed=True, submitted=True,
            filled_output_amount=150_000_000, native_balance_delta_lamports=-5_000_000,
            actual_input_amount=500,
        )
        desk = self._desk(result)
        desk.dry_run = False
        position = self._position()
        desk.elogw_engine.update_position("mint", position)

        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 0.5, "profit_ratchet_cost_recovery")

        fee_usd = 5_000_000 / 1e9 * 150.0
        self.assertAlmostEqual(desk.total_pnl, 100.0 - fee_usd)

    async def test_failed_exit_never_mutates_pnl_or_the_position(self):
        result = ExecutionResult(success=False, status=TransactionStatus.TIMEOUT, error="no fill")
        desk = self._desk(result)
        position = self._position()
        desk.elogw_engine.update_position("mint", position)

        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 0.5, "profit_ratchet_cost_recovery")

        self.assertEqual(desk.total_pnl, 0.0)
        self.assertEqual(desk.successful_exits, 0)
        still_open = desk.elogw_engine.open_positions["mint"]
        self.assertEqual(still_open["size_tokens"], 1_000)
        self.assertEqual(still_open["remaining_cost_usd"], 100.0)
        self.assertEqual(len(desk.dataset_builder.attempts), 1)
        self.assertNotIn("realized_pnl_usd", desk.dataset_builder.attempts[0][1])


class RecordingJitoSession:
    """Captures which URL each JSON-RPC method was actually POSTed to."""

    def __init__(self, results=None, tip_payload=None):
        self.posts = []
        self.results = results or {}
        self.tip_payload = tip_payload

    def post(self, url, json=None, **kwargs):
        self.posts.append((url, (json or {}).get("method")))
        result = self.results.get((json or {}).get("method"))
        return _FakeAsyncResponse({"jsonrpc": "2.0", "id": 1, "result": result})

    def get(self, url, **kwargs):
        return _FakeAsyncResponse(self.tip_payload)


class _FakeAsyncResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, **kwargs):
        return self._payload


class TestPumpCurve(unittest.TestCase):
    """Local curve pricing removes Jupiter from the first seconds of a launch.

    A newborn Pump token is tradeable on its own curve well before any
    aggregator indexes it; asking Jupiter first turned "not indexed yet" into
    "no executable evidence, do not trade".
    """

    @staticmethod
    def curve_bytes(virtual_token=1_073_000_000_000_000, virtual_sol=30_000_000_000,
                    real_token=793_100_000_000_000, real_sol=0,
                    total_supply=1_000_000_000_000_000, complete=0, creator=None):
        raw = bytearray(BONDING_CURVE_DISCRIMINATOR)
        raw += struct.pack("<QQQQQ", virtual_token, virtual_sol, real_token, real_sol, total_supply)
        raw.append(complete)
        if creator is not None:
            raw += creator
        return bytes(raw)

    def test_discriminator_matches_the_anchor_account_hash(self):
        self.assertEqual(
            BONDING_CURVE_DISCRIMINATOR,
            hashlib.sha256(b"account:BondingCurve").digest()[:8],
        )

    def test_parses_reserves_and_creator(self):
        state = parse_bonding_curve(self.curve_bytes(creator=bytes(range(1, 33))))
        self.assertEqual(state.virtual_sol_reserves, 30_000_000_000)
        self.assertEqual(state.virtual_token_reserves, 1_073_000_000_000_000)
        self.assertFalse(state.complete)
        self.assertTrue(state.tradeable)
        self.assertEqual(state.creator, b58encode(bytes(range(1, 33))))

    def test_rejects_a_foreign_account_rather_than_mispricing_it(self):
        raw = bytearray(self.curve_bytes())
        raw[0] ^= 0xFF
        with self.assertRaises(ValueError):
            parse_bonding_curve(bytes(raw))

    def test_completed_curve_is_not_tradeable_on_the_curve(self):
        state = parse_bonding_curve(self.curve_bytes(complete=1))
        self.assertFalse(state.tradeable)
        self.assertIsNone(state.price_sol_per_token)
        self.assertEqual(quote_buy(state, 10_000_000).data_status, "DATA_BLOCKED")

    def test_buy_conserves_the_constant_product_and_charges_the_fee(self):
        state = parse_bonding_curve(self.curve_bytes())
        quote = quote_buy(state, 1_000_000_000)
        self.assertEqual(quote.data_status, "OK")
        self.assertEqual(quote.fee_amount, 10_000_000)
        net = 1_000_000_000 - quote.fee_amount
        expected = (net * state.virtual_token_reserves) // (state.virtual_sol_reserves + net)
        self.assertEqual(quote.output_amount, expected)
        self.assertGreater(quote.output_amount, 0)

    def test_larger_buys_get_strictly_worse_average_prices(self):
        state = parse_bonding_curve(self.curve_bytes())
        small = quote_buy(state, 100_000_000)
        large = quote_buy(state, 10_000_000_000)
        self.assertLess(small.price_impact_pct, large.price_impact_pct)
        self.assertLess(small.average_price_sol_per_token, large.average_price_sol_per_token)
        self.assertLess(large.output_amount, small.output_amount * 100)

    def test_round_trip_loses_only_fees_and_impact_never_gains(self):
        state = parse_bonding_curve(self.curve_bytes(real_sol=50_000_000_000))
        spend = 500_000_000
        bought = quote_buy(state, spend)
        back = quote_sell(state, bought.output_amount)
        self.assertEqual(back.data_status, "OK")
        self.assertLess(back.output_amount, spend)

    def test_sell_cannot_pay_out_more_than_real_sol_reserves(self):
        state = parse_bonding_curve(self.curve_bytes(real_sol=1_000_000))
        quote = quote_sell(state, 10_000_000_000_000)
        self.assertEqual(quote.data_status, "OK")
        self.assertLessEqual(quote.output_amount, 1_000_000)

    def test_capacity_is_the_largest_size_within_the_impact_bound(self):
        state = parse_bonding_curve(self.curve_bytes(real_sol=50_000_000_000))
        capacity = sell_capacity_lamports(state, max_impact_pct=0.15)
        self.assertGreater(capacity, 0)
        self.assertLessEqual(quote_sell(state, capacity).price_impact_pct, 0.15)
        beyond = quote_sell(state, int(capacity * 1.5))
        self.assertGreater(beyond.price_impact_pct, 0.15)

    def test_quotes_are_never_marked_execution_verified(self):
        state = parse_bonding_curve(self.curve_bytes())
        self.assertFalse(quote_buy(state, 1_000_000_000).verified)
        self.assertFalse(quote_sell(state, 1_000_000).verified)
        self.assertFalse(observation_from_state(state)["execution_verified"])

    def test_observation_marks_price_without_any_aggregator_call(self):
        state = parse_bonding_curve(self.curve_bytes(real_sol=50_000_000_000))
        observation = observation_from_state(state)
        self.assertTrue(observation["feasible"])
        self.assertEqual(observation["measurement"], "pump_curve_local")
        self.assertGreater(observation["price_sol_per_token"], 0)
        self.assertLess(observation["round_trip_retention"], 1.0)
        self.assertIsNotNone(observation["curve_progress"])


class TestFeatureParity(unittest.IsolatedAsyncioTestCase):
    """Training and serving must build features from one implementation.

    A model can pass chronological OOS on feature definition A and then be
    served definition B; the measured edge simply does not transfer, and
    nothing fails loudly. These lock the two paths together.
    """

    @staticmethod
    def _snapshot():
        return {
            "timestamp": 1_000.0,
            "deployer_features": {"has_profile": True, "rug_rate": 0.25, "success_rate": 0.4,
                                  "avg_max_multiple": 6.0},
            "wallet_features": {"initial_buyer_count": 12, "smart_buyer_count": 3,
                                "insider_buyer_count": 1, "total_sol_volume": 42.5},
            "flow_features": {"status": "OK", "buy_velocity": 1.2, "buy_acceleration": 0.4,
                              "organic_ratio": 0.8, "bundle_concentration": 0.2,
                              "observed_trade_count": 9},
            "liquidity_features": {"status": "OK", "liquidity_usd": 25_000, "liquidity_locked": True},
            "social_features": {"mention_count": 4, "avg_velocity": 2.0, "acceleration": 0.5,
                                "avg_credibility": 0.6, "chain_before_pct": 0.3, "cross_platform": True},
            "token_features": {"status": "OK", "ownership_renounced": True, "can_mint": False,
                               "can_freeze": False, "top_10_pct": 35.0},
            "entity_graph_features": {"status": "OK", "deployer_cluster_risk": 0.5,
                                      "funding_wallet_risk": 0.7},
        }

    def test_trainer_delegates_to_the_shared_engine(self):
        from src.research.shadow_trainer import snapshot_to_features
        episode = {"token": "mint", "chain": "solana", "created_at": 900.0}
        snapshot = self._snapshot()
        self.assertEqual(
            snapshot_to_features(episode, snapshot).to_array().tolist(),
            build_features(episode, snapshot).to_array().tolist(),
        )

    def test_entity_graph_supplies_both_risk_features(self):
        """funding_wallet_risk was trained on but never populated live, and
        deployer_cluster_risk came from a different function entirely."""
        episode = {"token": "mint", "chain": "solana", "created_at": 900.0}
        features = build_features(episode, self._snapshot())
        self.assertEqual(features.deployer_cluster_risk, 0.5)
        self.assertEqual(features.funding_wallet_risk, 0.7)

    def test_flow_group_supplies_organic_and_bundle_not_coordination(self):
        episode = {"token": "mint", "chain": "solana", "created_at": 900.0}
        features = build_features(episode, self._snapshot())
        self.assertEqual(features.organic_ratio, 0.8)
        self.assertEqual(features.bundle_concentration, 0.2)
        self.assertEqual(features.sol_volume, 42.5)  # wallet group, not flow

    def test_missing_groups_are_blocked_not_silently_zeroed(self):
        episode = {"token": "mint", "chain": "solana", "created_at": 900.0}
        features = build_features(episode, {"timestamp": 1_000.0})
        self.assertFalse(features.flow_available)
        self.assertFalse(features.social_available)
        self.assertFalse(features.coordination_available)
        self.assertFalse(features.wallet_history_available)
        self.assertEqual(features.data_coverage, 0.0)

    async def test_live_builder_matches_the_trainer_on_the_same_episode(self):
        """The end-to-end guarantee: one episode, captured live and replayed
        through the trainer, must yield the same feature vector."""
        builder = PointInTimeDatasetBuilder(
            solana_chain(), FakeRpc(), FakeGenealogy(),
            SimpleNamespace(get_top_wallets=lambda limit=50: [],
                            get_wallet_score=lambda wallet: None,
                            _recent_buys=[]),
            SimpleNamespace(get_token_social_signal=lambda token, as_of=None: {"mention_count": 0}),
            SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
        )
        builder.start_episode("mint", "dev", "pump", "curve", "wsol", detected_at=900.0)
        episode = builder.active_episodes["mint"]
        episode.market_observations.extend([
            {"type": "trade", "side": "buy", "wallet": "a", "slot": 1, "timestamp": 995.0,
             "notional_sol": 2.0},
            {"type": "trade", "side": "buy", "wallet": "b", "slot": 2, "timestamp": 997.0,
             "notional_sol": 3.0},
            {"type": "trade", "side": "buy", "wallet": "c", "slot": 3, "timestamp": 999.0,
             "notional_sol": 1.0},
        ])
        as_of = 1_000.0
        snapshot = {
            "timestamp": as_of,
            "deployer_features": await builder._capture_deployer_features(episode, as_of),
            "wallet_features": await builder._capture_wallet_features(episode, as_of),
            "flow_features": await builder._capture_flow_features(episode, as_of),
            "liquidity_features": {"status": "DATA_BLOCKED"},
            "social_features": await builder._capture_social_features(episode, as_of),
            "token_features": {"status": "DATA_BLOCKED"},
            "entity_graph_features": await builder._capture_entity_graph_features(episode, as_of),
        }
        episode_meta = {"token": "mint", "chain": "solana", "created_at": 900.0}

        from src.research.shadow_trainer import snapshot_to_features
        self.assertEqual(
            build_features(episode_meta, snapshot).to_array().tolist(),
            snapshot_to_features(episode_meta, snapshot).to_array().tolist(),
        )
        # And the flow group really did drive the model input.
        live = build_features(episode_meta, snapshot)
        self.assertTrue(live.flow_available)
        self.assertTrue(live.coordination_available)
        self.assertAlmostEqual(live.buy_velocity, 0.3)


class TestJitoEndpointRouting(unittest.IsolatedAsyncioTestCase):
    def _client(self, session):
        client = JitoClient(regions=["https://mainnet.block-engine.jito.wtf"])
        client._session = session
        return client

    async def test_each_method_uses_its_own_documented_path(self):
        session = RecordingJitoSession(results={
            "sendBundle": "bundle-1",
            "getBundleStatuses": {"value": [{"confirmationStatus": "confirmed"}]},
            "getTipAccounts": ["tip-account"],
            "getInflightBundleStatuses": {"value": [{"status": "Pending"}]},
        })
        client = self._client(session)
        await client.send_bundle(["tx"])
        await client.get_bundle_status("bundle-1")
        await client.get_inflight_bundle_status("bundle-1")
        await client._rpc("getTipAccounts", [])

        routed = dict((method, url) for url, method in session.posts)
        base = "https://mainnet.block-engine.jito.wtf"
        # Posting every method at /api/v1/bundles is the bug: status and tip
        # lookups silently fail, so a landed bundle looks permanently pending.
        self.assertEqual(routed["sendBundle"], f"{base}/api/v1/bundles")
        self.assertEqual(routed["getBundleStatuses"], f"{base}/api/v1/getBundleStatuses")
        self.assertEqual(routed["getInflightBundleStatuses"],
                         f"{base}/api/v1/getInflightBundleStatuses")
        self.assertEqual(routed["getTipAccounts"], f"{base}/api/v1/getTipAccounts")

    async def test_single_transaction_lane_uses_the_transactions_path(self):
        session = RecordingJitoSession(results={"sendTransaction": "sig-1"})
        client = self._client(session)
        self.assertEqual(await client.send_transaction("tx"), "sig-1")
        self.assertEqual(session.posts[0][0],
                         "https://mainnet.block-engine.jito.wtf/api/v1/transactions")

    async def test_bundle_is_raced_across_regions_with_one_identical_payload(self):
        session = RecordingJitoSession(results={"sendBundle": "bundle-1"})
        client = JitoClient(regions=[
            "https://amsterdam.mainnet.block-engine.jito.wtf",
            "https://frankfurt.mainnet.block-engine.jito.wtf",
        ])
        client._session = session
        self.assertEqual(await client.send_bundle(["signed-tx"]), "bundle-1")
        hosts = {url for url, _ in session.posts}
        self.assertEqual(hosts, {
            "https://amsterdam.mainnet.block-engine.jito.wtf/api/v1/bundles",
            "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/bundles",
        })

    async def test_duplicate_regions_are_collapsed(self):
        client = JitoClient(regions=[
            "https://mainnet.block-engine.jito.wtf",
            "https://mainnet.block-engine.jito.wtf/",
        ])
        self.assertEqual(client.regions, ["https://mainnet.block-engine.jito.wtf"])

    async def test_a_full_bundles_url_is_normalized_back_to_its_base(self):
        client = JitoClient(jito_url="https://ny.mainnet.block-engine.jito.wtf/api/v1/bundles")
        self.assertEqual(client.regions, ["https://ny.mainnet.block-engine.jito.wtf"])
        self.assertEqual(client.endpoint("getTipAccounts"),
                         "https://ny.mainnet.block-engine.jito.wtf/api/v1/getTipAccounts")

    async def test_tip_floor_is_converted_from_sol_to_lamports(self):
        session = RecordingJitoSession(tip_payload=[{"landed_tips_75th_percentile": 0.000123}])
        client = self._client(session)
        self.assertEqual(await client.get_tip_floor_lamports(75), 123_000)

    async def test_tip_floor_is_data_blocked_rather_than_guessed(self):
        session = RecordingJitoSession(tip_payload=[{}])
        client = self._client(session)
        self.assertIsNone(await client.get_tip_floor_lamports(75))


class TestIncrementalScaleIn(unittest.TestCase):
    """Capital should follow evidence rather than commit once at T0.

    A single all-at-once entry has to size before the flow that separates a
    launch has arrived. Scaling in asks the marginal question instead: does
    the NEXT unit still raise expected log growth?
    """

    def _engine(self, **kwargs):
        predictor = MultiHeadPredictor()
        predictor._is_trained = True
        engine = ElogwEngine(predictor, min_edge_bps=-1, drawdown_aversion_lambda=0, **kwargs)
        engine.portfolio_value = 10_000.0
        return engine

    @staticmethod
    def _strong():
        return MultiHeadPrediction("mint", "solana", 0, p_2x=0.85, p_5x=0.6, p_10x=0.3,
                                   p_50x=0.05, p_rug_30s=0.01, p_rug_5m=0.02,
                                   expected_slippage=0.01)

    @staticmethod
    def _weak():
        return MultiHeadPrediction("mint", "solana", 0, p_2x=0.10, p_5x=0.02, p_10x=0.0,
                                   p_50x=0.0, p_rug_30s=0.35, p_rug_5m=0.55,
                                   expected_slippage=0.05)

    def test_improving_evidence_justifies_adding_to_an_open_position(self):
        engine = self._engine()
        fraction, gain = engine.plan_scale_in(self._strong(), held_cost_usd=100.0,
                                              current_multiple=1.2, liquidity_usd=500_000)
        self.assertGreater(fraction, 0)
        self.assertGreater(gain, 0)

    def test_deteriorating_evidence_stops_scaling(self):
        engine = self._engine()
        fraction, gain = engine.plan_scale_in(self._weak(), held_cost_usd=100.0,
                                              current_multiple=0.9, liquidity_usd=500_000)
        self.assertEqual(fraction, 0.0)
        self.assertEqual(gain, 0.0)

    def test_scaling_respects_the_remaining_position_headroom(self):
        engine = self._engine(max_position_pct=0.02)
        fraction, _ = engine.plan_scale_in(self._strong(), held_cost_usd=0.0,
                                           current_multiple=1.0, liquidity_usd=500_000)
        self.assertLessEqual(fraction, 0.02 + 1e-9)

    def test_a_position_already_at_the_cap_cannot_add(self):
        engine = self._engine(max_position_pct=0.02)
        fraction, gain = engine.plan_scale_in(self._strong(), held_cost_usd=200.0,
                                              current_multiple=1.5, liquidity_usd=500_000)
        self.assertEqual(fraction, 0.0)
        self.assertEqual(gain, 0.0)

    def test_thin_liquidity_caps_the_add_below_the_position_cap(self):
        engine = self._engine(max_position_pct=0.05, max_liquidity_fraction=0.01)
        thin, _ = engine.plan_scale_in(self._strong(), held_cost_usd=0.0,
                                       current_multiple=1.0, liquidity_usd=10_000)
        deep, _ = engine.plan_scale_in(self._strong(), held_cost_usd=0.0,
                                       current_multiple=1.0, liquidity_usd=1_000_000)
        self.assertLess(thin, deep)

    def test_marginal_growth_accounts_for_the_existing_slice_basis(self):
        """A slice already up 3x contributes its appreciated value, so the
        marginal calculation is not the same as sizing from scratch."""
        engine = self._engine()
        at_entry = engine.marginal_log_growth(self._strong(), 0.01, 1.0, 0.01)
        appreciated = engine.marginal_log_growth(self._strong(), 0.01, 3.0, 0.01)
        self.assertNotAlmostEqual(at_entry, appreciated)
        self.assertGreater(appreciated, at_entry)

    def test_overcommitted_cash_is_rejected_rather_than_levered(self):
        engine = self._engine()
        self.assertEqual(
            engine.marginal_log_growth(self._strong(), 0.8, 1.0, 0.5), -float("inf"))


class TestDailyLossBudget(unittest.TestCase):
    def _engine(self, **kwargs):
        predictor = MultiHeadPredictor()
        predictor._is_trained = True
        engine = ElogwEngine(predictor, **kwargs)
        engine.portfolio_value = 10_000.0
        engine._roll_day_if_needed()
        return engine

    def test_percentage_limit_scales_with_day_start_equity(self):
        engine = self._engine(max_daily_loss_pct=0.10)
        self.assertAlmostEqual(engine.daily_loss_limit(), 1_000.0)
        # A larger book gets a proportionally larger budget, unlike a fixed
        # dollar limit that would be 1% of a $100k book.
        big = self._engine(max_daily_loss_pct=0.10)
        big.portfolio_value = 100_000.0
        big._day_start_equity = 100_000.0
        self.assertAlmostEqual(big.daily_loss_limit(), 10_000.0)

    def test_percentage_limit_does_not_tighten_as_equity_falls(self):
        engine = self._engine(max_daily_loss_pct=0.10)
        engine.portfolio_value = 6_000.0  # mid-day drawdown
        # Anchored to day-start equity, so the budget stays $1,000 rather than
        # shrinking to $600 and halting the book early.
        self.assertAlmostEqual(engine.daily_loss_limit(), 1_000.0)

    def test_absolute_limit_still_applies_when_no_percentage_configured(self):
        engine = self._engine(max_daily_loss_usd=250.0)
        self.assertAlmostEqual(engine.daily_loss_limit(), 250.0)

    def test_kill_switch_trips_on_the_percentage_budget(self):
        engine = self._engine(max_daily_loss_pct=0.10)
        engine.update_pnl(-999.0)
        self.assertFalse(engine.kill_switch_active)
        engine.update_pnl(-2.0)
        self.assertTrue(engine.kill_switch_active)

    def test_budget_and_kill_switch_reset_at_the_utc_day_boundary(self):
        engine = self._engine(max_daily_loss_pct=0.10)
        engine.update_pnl(-1_500.0)
        self.assertTrue(engine.kill_switch_active)
        # Roll the clock forward one UTC day.
        engine._pnl_day -= 1
        engine._roll_day_if_needed()
        self.assertFalse(engine.kill_switch_active)
        self.assertEqual(engine.daily_pnl, 0.0)

    def test_profit_offsets_loss_within_the_same_day_but_not_across_days(self):
        engine = self._engine(max_daily_loss_pct=0.10)
        engine.update_pnl(+5_000.0)
        engine.update_pnl(-5_500.0)
        # Net -500 against a 1,000 budget: still trading.
        self.assertFalse(engine.kill_switch_active)
        # Yesterday's profit must not finance today's loss.
        engine._pnl_day -= 1
        engine._roll_day_if_needed()
        self.assertEqual(engine.daily_pnl, 0.0)
        engine.update_pnl(-1_100.0)
        self.assertTrue(engine.kill_switch_active)


class TestDailyGivebackGuard(unittest.TestCase):
    def _engine(self, **kwargs):
        predictor = MultiHeadPredictor()
        predictor._is_trained = True
        engine = ElogwEngine(predictor, max_daily_loss_pct=0.10, **kwargs)
        engine.portfolio_value = 10_000.0
        engine._roll_day_if_needed()
        return engine

    def test_gains_are_uncapped_on_the_way_up(self):
        engine = self._engine(daily_giveback_pct=0.35)
        for _ in range(20):
            engine.update_pnl(+500.0)
        self.assertFalse(engine.kill_switch_active)
        self.assertAlmostEqual(engine.daily_pnl, 10_000.0)

    def test_guard_halts_after_handing_back_the_configured_share_of_the_peak(self):
        engine = self._engine(daily_giveback_pct=0.35)
        engine.update_pnl(+2_000.0)
        self.assertAlmostEqual(engine.giveback_floor(), 1_300.0)
        engine.update_pnl(-600.0)   # down to 1,400: still above the floor
        self.assertFalse(engine.kill_switch_active)
        engine.update_pnl(-150.0)   # down to 1,250: through the floor
        self.assertTrue(engine.kill_switch_active)

    def test_floor_ratchets_up_with_a_new_peak(self):
        engine = self._engine(daily_giveback_pct=0.35)
        engine.update_pnl(+2_000.0)
        self.assertAlmostEqual(engine.giveback_floor(), 1_300.0)
        engine.update_pnl(+2_000.0)
        # New 4,000 peak lifts the floor rather than leaving it at the old one.
        self.assertAlmostEqual(engine.giveback_floor(), 2_600.0)

    def test_guard_stays_unarmed_on_trivial_gains(self):
        engine = self._engine(daily_giveback_pct=0.35)
        engine.update_pnl(+100.0)  # far below 50% of the 1,000 budget
        self.assertIsNone(engine.giveback_floor())
        engine.update_pnl(-90.0)
        self.assertFalse(engine.kill_switch_active)

    def test_guard_is_off_when_not_configured(self):
        engine = self._engine()
        engine.update_pnl(+5_000.0)
        engine.update_pnl(-4_900.0)
        self.assertIsNone(engine.giveback_floor())
        self.assertFalse(engine.kill_switch_active)


class TestExitPolicy(unittest.TestCase):
    def test_default_policy_reproduces_the_previous_hardcoded_thresholds(self):
        policy = ExitPolicy.default()
        self.assertEqual(evaluate_exit(policy, 0.70, 1.0, 0.0, set(), 0),
                         ("hard_stop_loss", 1.0))
        reason, fraction = evaluate_exit(policy, 2.0, 2.0, 0.0, set(), 0)
        self.assertEqual(reason, "profit_ratchet_cost_recovery")
        self.assertAlmostEqual(fraction, 0.5)
        self.assertEqual(evaluate_exit(policy, 5.0, 5.0, 0.0, {"cost_recovery"}, 0),
                         ("profit_ratchet_5x", 0.25))
        self.assertEqual(evaluate_exit(policy, 10.0, 10.0, 0.0, {"cost_recovery", "bank_5x"}, 0),
                         ("profit_ratchet_10x", 0.20))
        self.assertEqual(evaluate_exit(policy, 1.5, 1.5, 0.0, set(), 4000),
                         ("time_stop", 1.0))

    def test_a_ratchet_stage_only_fires_once(self):
        policy = ExitPolicy.default()
        self.assertIsNotNone(evaluate_exit(policy, 2.5, 2.5, 0.9, set(), 0))
        # Same price, stage already banked, and the trailing stop has not been
        # breached -> the policy must hold rather than re-selling.
        self.assertIsNone(evaluate_exit(policy, 2.5, 2.5, 0.9, {"cost_recovery"}, 0))

    def test_trailing_stop_widens_when_continuation_probability_is_low(self):
        policy = ExitPolicy.default()
        # High continuation -> tighter trail -> a higher floor, so a pullback
        # to 3.5 from a 6x high water exits...
        self.assertEqual(
            evaluate_exit(policy, 3.5, 6.0, 0.9, {"cost_recovery", "bank_5x"}, 0),
            ("adaptive_profit_trailing_stop", 1.0),
        )
        # ...while low continuation uses the widest trail, whose floor sits
        # above that same price, so it also exits. The meaningful difference
        # is the floor itself.
        self.assertGreater(policy.trail_floor(6.0, 0.05), policy.trail_floor(6.0, 0.9))

    def test_trailing_stop_does_not_fire_before_the_activation_high_water(self):
        policy = ExitPolicy.default()
        self.assertIsNone(evaluate_exit(policy, 1.05, 1.2, 0.0, set(), 0))


class TestExitPolicyTrainer(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_episode(storage: Path, token: str, created_at: float, marks):
        directory = storage / "day"
        directory.mkdir(parents=True, exist_ok=True)
        observations = [
            {"timestamp": created_at + elapsed, "price_multiple": multiple,
             "route_feasible": True, "price_impact_pct": 0.02}
            for elapsed, multiple in marks
        ]
        with gzip.open(directory / f"{token}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump({
                "token": token, "created_at": created_at,
                "market_observations": observations,
                "final_outcome": {"status": "OK"},
            }, handle)

    def test_insufficient_history_remains_explicitly_data_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = train_exit_policy(root / "episodes", root / "models", min_episodes=60)
            self.assertEqual(report["status"], "DATA_BLOCKED")
            persisted = json.loads((root / "models" / "last_exit_policy_report.json").read_text())
            self.assertEqual(persisted["status"], "DATA_BLOCKED")

    def test_high_water_ignores_marks_the_router_could_not_fill(self):
        policy = ExitPolicy.default()
        # A 10x spike that was never quotable, then a fallback to 3x that was.
        # Live, _mark_position returns None on an unquotable position and the
        # high-water mark never moves, so the trailing stop must not treat the
        # unsellable spike as a realized peak and dump the position at 3x.
        unsellable_spike = [(0.0, 1.0, True), (10.0, 10.0, False), (20.0, 3.0, True), (30.0, 3.1, True)]
        self.assertIsNone(
            evaluate_exit(policy, 3.0, 1.0, 0.0, {"cost_recovery"}, 20.0),
            "sanity: 3x against a 1x high water should not trail-stop",
        )
        # The simulator must reach the end still holding, i.e. score close to
        # the final mark rather than an exit forced by a phantom peak.
        self.assertGreater(simulate(policy, unsellable_spike), math.log(2.0))

    def test_simulation_only_credits_route_feasible_exits(self):
        policy = ExitPolicy.default()
        # Price collapses through the hard stop, but the route is never
        # feasible at or after the breach, so the position cannot actually be
        # sold there and must be marked down rather than booked at the stop.
        marks = [(0.0, 1.0, True), (10.0, 0.5, False), (20.0, 0.4, False)]
        blocked = simulate(policy, marks)
        feasible = simulate(policy, [(0.0, 1.0, True), (10.0, 0.5, True), (20.0, 0.4, True)])
        self.assertLess(blocked, feasible)

    def test_a_policy_only_ships_when_it_beats_default_and_hold_out_of_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "episodes"
            # Every episode spikes then round-trips back down: cutting losses
            # and banking beats holding to the end, so some non-default
            # candidate should win and clear both baselines.
            for index in range(80):
                self._write_episode(storage, f"pump-dump-{index}", float(index * 1000),
                                    [(0.0, 1.0), (30.0, 3.0), (60.0, 6.0), (120.0, 0.6), (600.0, 0.2)])
            report = train_exit_policy(storage, root / "models", min_episodes=60)
            self.assertIn(report["status"], {"PASSED", "REJECTED"})
            if report["status"] == "PASSED":
                self.assertNotEqual(report["selected_policy"], "default")
                self.assertGreater(report["oos_elogw"], report["oos_elogw_default_policy"])
                self.assertGreater(report["oos_elogw"], report["oos_elogw_hold_baseline"])
                policy, loaded = load_latest_exit_policy(str(root / "models"))
                self.assertIsNotNone(policy)
                self.assertEqual(loaded["status"], "PASSED")

    def test_a_policy_that_cannot_beat_holding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "episodes"
            # Monotonic winners: any early exit strictly loses to holding, so
            # nothing should ship.
            for index in range(80):
                self._write_episode(storage, f"moon-{index}", float(index * 1000),
                                    [(0.0, 1.0), (30.0, 2.0), (60.0, 5.0), (120.0, 12.0), (600.0, 30.0)])
            report = train_exit_policy(storage, root / "models", min_episodes=60)
            self.assertEqual(report["status"], "REJECTED", report)
            self.assertIsNone(load_latest_exit_policy(str(root / "models"))[0])


class TestScaleInWiring(unittest.IsolatedAsyncioTestCase):
    """The marginal-E[log W] add must actually run from the position manager."""

    def _desk(self, result, prediction, plan=(0.01, 0.5)):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), min_edge_bps=-1)
        engine.portfolio_value = 10_000.0
        engine.plan_scale_in = lambda *a, **k: plan
        desk = SimpleNamespace(
            predictor=SimpleNamespace(_is_trained=True, predict=lambda features: prediction),
            dry_run=True,
            champion_challenger=SimpleNamespace(is_live=lambda _id: True),
            elogw_engine=engine,
            execution_engine=FakeExecutionEngineForExit(result),
            dataset_builder=FakeDatasetBuilderForExit(),
            fee_optimizer=SimpleNamespace(get_optimal_fee=lambda *a: 5_000,
                                          get_jito_tip=lambda *a: 100_000),
            wallet_equity_usd=10_000.0,
            sol_price_usd=150.0,
            _build_prediction_features=self._features,
            _refresh_portfolio_state=self._noop,
        )
        return desk

    @staticmethod
    async def _features(candidate, risk, liquidity):
        return PredictionFeatures("mint", "solana", 0)

    @staticmethod
    async def _noop():
        return None

    @staticmethod
    def _position():
        return {
            "size_tokens": 1_000, "initial_size_tokens": 1_000,
            "remaining_cost_usd": 100.0, "initial_cost_usd": 100.0,
            "risk_contribution": 0.02, "decision_id": "d1",
            "candidate": SimpleNamespace(base_token=None),
            "risk_object": SimpleNamespace(),
            "liquidity_usd": 500_000.0,
            "prediction_object": TestScaleInWiring._prediction(),
        }

    @staticmethod
    def _prediction(rug_30s=0.01, rug_5m=0.02):
        return MultiHeadPrediction("mint", "solana", 0, p_2x=0.8, p_5x=0.5,
                                   p_rug_30s=rug_30s, p_rug_5m=rug_5m)

    async def test_add_increases_size_and_cost_basis(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED, simulated=True,
                                 quoted_output_amount=500, actual_input_amount=1)
        desk = self._desk(result, self._prediction())
        position = self._position()
        await MemecoinQuantDesk._consider_scale_in(desk, "mint", position, 1.5)
        self.assertEqual(position["size_tokens"], 1_500)
        self.assertGreater(position["remaining_cost_usd"], 100.0)
        self.assertEqual(len(position["scale_ins"]), 1)
        self.assertEqual(position["scale_ins"][0]["elogw_gain"], 0.5)

    async def test_no_add_when_marginal_growth_is_not_positive(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED, simulated=True,
                                 quoted_output_amount=500)
        desk = self._desk(result, self._prediction(), plan=(0.0, 0.0))
        position = self._position()
        await MemecoinQuantDesk._consider_scale_in(desk, "mint", position, 1.5)
        self.assertEqual(position["size_tokens"], 1_000)
        self.assertEqual(position["remaining_cost_usd"], 100.0)
        self.assertEqual(desk.execution_engine.buys, [])

    async def test_rug_risk_blocks_adding_even_on_a_winner(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED, simulated=True,
                                 quoted_output_amount=500)
        desk = self._desk(result, self._prediction(rug_30s=0.9, rug_5m=0.9))
        position = self._position()
        position["prediction_object"] = self._prediction(rug_30s=0.9, rug_5m=0.9)
        await MemecoinQuantDesk._consider_scale_in(desk, "mint", position, 3.0)
        self.assertEqual(position["size_tokens"], 1_000)
        self.assertEqual(desk.execution_engine.buys, [])

    async def test_untrained_model_never_scales_in(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED, simulated=True)
        desk = self._desk(result, self._prediction())
        desk.predictor = SimpleNamespace(_is_trained=False, predict=lambda f: None)
        position = self._position()
        await MemecoinQuantDesk._consider_scale_in(desk, "mint", position, 1.5)
        self.assertEqual(position["size_tokens"], 1_000)
        self.assertEqual(desk.execution_engine.buys, [])

    async def test_failed_add_leaves_the_position_untouched(self):
        result = ExecutionResult(success=False, status=TransactionStatus.TIMEOUT, error="no fill")
        desk = self._desk(result, self._prediction())
        position = self._position()
        await MemecoinQuantDesk._consider_scale_in(desk, "mint", position, 1.5)
        self.assertEqual(position["size_tokens"], 1_000)
        self.assertEqual(position["remaining_cost_usd"], 100.0)
        self.assertNotIn("scale_ins", position)


class TestShadowTrainer(unittest.TestCase):
    def test_chronological_split_keeps_launch_episodes_disjoint(self):
        samples = []
        for token_index in range(5):
            for offset in (0, 60):
                samples.append((
                    PredictionFeatures(f"token-{token_index}", "solana", token_index * 100 + offset),
                    {}, {},
                ))
        train, oos = chronological_episode_split(samples)
        train_tokens = {item[0].token for item in train}
        oos_tokens = {item[0].token for item in oos}
        self.assertFalse(train_tokens & oos_tokens)
        self.assertEqual(train_tokens | oos_tokens, {f"token-{index}" for index in range(5)})

    def test_snapshot_labels_derive_rug_targets_from_drawdown_based_rug_time(self):
        """Rug horizons are measured from the snapshot, not from launch.

        ``outcome['rug_time']`` is seconds-since-launch, but a model asked at
        T+220s is predicting "does this rug in the next 30 seconds?" from where
        it stands. Labelling it against the launch-relative time would teach the
        model that a rug 20 seconds away is 240 seconds away, which is exactly
        backwards at the moment the answer matters most.
        """
        from src.research.shadow_trainer import snapshot_labels
        from src.strategies.multihead_predictor import PredictionTarget
        episode = {"token": "mint", "created_at": 1_000.0}

        at_launch = {"timestamp": 1_000.0, "labels": {}, "liquidity_features": {}}
        fast = snapshot_labels(at_launch, episode, {"rugged": True, "rug_time": 12.0})
        self.assertEqual(fast[PredictionTarget.P_RUG_30S], 1.0)
        self.assertEqual(fast[PredictionTarget.P_RUG_5M], 1.0)

        slow = snapshot_labels(at_launch, episode, {"rugged": True, "rug_time": 240.0})
        self.assertEqual(slow[PredictionTarget.P_RUG_30S], 0.0)
        self.assertEqual(slow[PredictionTarget.P_RUG_5M], 1.0)

        # Same 240s rug, but observed at T+220s: now it is 20 seconds away.
        late = {"timestamp": 1_220.0, "labels": {}, "liquidity_features": {}}
        imminent = snapshot_labels(late, episode, {"rugged": True, "rug_time": 240.0})
        self.assertEqual(imminent[PredictionTarget.P_RUG_30S], 1.0)
        self.assertEqual(imminent[PredictionTarget.P_RUG_5M], 1.0)

        # A snapshot taken after the rug carries a negative horizon and must not
        # be labelled as a future rug.
        after = {"timestamp": 1_400.0, "labels": {}, "liquidity_features": {}}
        stale = snapshot_labels(after, episode, {"rugged": True, "rug_time": 240.0})
        self.assertEqual(stale[PredictionTarget.P_RUG_30S], 0.0)
        self.assertEqual(stale[PredictionTarget.P_RUG_5M], 0.0)

        # No rug_time is missing information, never a confident "did not rug".
        unknown = snapshot_labels(at_launch, episode, {"rugged": True, "rug_time": None})
        self.assertEqual(unknown[PredictionTarget.P_RUG_30S], 0.0)
        self.assertEqual(unknown[PredictionTarget.P_RUG_5M], 0.0)

    def test_insufficient_history_remains_explicitly_data_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = train_shadow(root / "episodes", root / "models", min_samples=10)
            self.assertEqual(report["status"], "DATA_BLOCKED")
            persisted = json.loads((root / "models" / "last_training_report.json").read_text())
            self.assertEqual(persisted["status"], "DATA_BLOCKED")


def _write_hazard_episode(storage: Path, token: str, created_at: float, observations, final_outcome):
    directory = storage / "day"
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"{token}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({
            "token": token, "chain": "solana", "created_at": created_at,
            "market_observations": observations, "final_outcome": final_outcome,
            "snapshots": {name: {"timestamp": created_at + offset}
                          for name, offset in (("t10s", 10), ("t30s", 30), ("t1m", 60))},
        }, handle)


def _build_hazard_fixture(storage: Path, count: int = 100):
    """40 episodes interleaved rug/healthy (3-in-8 rug) so both the train
    and the chronologically-last OOS fold contain rug episodes."""
    for index in range(count):
        created_at = float(index * 1000)
        if index % 8 < 3:
            observations = [
                {"type": "liquidity", "timestamp": created_at + 15, "change_pct": -0.5},
                {"type": "route", "timestamp": created_at + 16, "feasible": False},
                {"type": "trade", "side": "buy", "timestamp": created_at + 35, "notional_usd": 1},
            ]
            _write_hazard_episode(storage, f"rug-{index}", created_at, observations,
                                 {"status": "OK", "rugged": True, "rug_time": 45})
        else:
            observations = [
                {"type": "trade", "side": "buy", "timestamp": created_at + 1, "notional_usd": 50},
                {"type": "trade", "side": "buy", "timestamp": created_at + 650, "notional_usd": 50},
            ]
            _write_hazard_episode(storage, f"healthy-{index}", created_at, observations,
                                 {"status": "OK", "rugged": False, "rug_time": None})


class TestHazardTrainer(unittest.IsolatedAsyncioTestCase):
    def test_insufficient_history_remains_explicitly_data_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = hazard_trainer.train(root / "episodes", root / "models", min_rows=200)
            self.assertEqual(report["status"], "DATA_BLOCKED")
            persisted = json.loads((root / "models" / "last_hazard_training_report.json").read_text())
            self.assertEqual(persisted["status"], "DATA_BLOCKED")

    def test_feature_vector_separates_rug_like_observations_from_clean_ones(self):
        rug_like = ContinuousRugHazardModel.feature_vector_from_observations(
            [{"type": "liquidity", "timestamp": 15, "change_pct": -0.5},
             {"type": "route", "timestamp": 16, "feasible": False}], 20,
        )
        healthy = ContinuousRugHazardModel.feature_vector_from_observations(
            [{"type": "trade", "side": "buy", "timestamp": 1, "notional_usd": 50}], 20,
        )
        self.assertEqual(len(rug_like), len(HAZARD_FEATURE_NAMES))
        route = HAZARD_FEATURE_NAMES.index("route_degradation")
        liquidity = HAZARD_FEATURE_NAMES.index("liquidity_withdrawal")
        self.assertGreater(rug_like[route], healthy[route])
        self.assertGreater(rug_like[liquidity], healthy[liquidity])

    def test_trains_on_chronological_snapshot_rows_and_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "episodes"
            _build_hazard_fixture(storage)
            report = hazard_trainer.train(storage, root / "models", min_rows=200)
            self.assertEqual(report["status"], "PASSED", report)
            self.assertTrue(Path(report["model_path"]).exists())
            for key in ("rug_30s", "rug_5m"):
                self.assertEqual(report["metrics"][key]["status"], "PASSED", report["metrics"])
                self.assertGreater(report["metrics"][key]["train_positive"], 0)
                self.assertGreater(report["metrics"][key]["oos_positive"], 0)

    async def test_rug_hazard_model_loads_and_applies_a_passed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "episodes"
            _build_hazard_fixture(storage)
            model_dir = root / "models"
            report = hazard_trainer.train(storage, model_dir, min_rows=200)
            self.assertEqual(report["status"], "PASSED", report)

            with patch.dict("os.environ", {"MODEL_DIR": str(model_dir)}):
                class WalletIntel:
                    def get_top_wallets(self, limit=50):
                        return []
                class Adversarial:
                    def get_adaptive_weight(self, feature, base):
                        return base
                model = ContinuousRugHazardModel(solana_chain(), FakeRpc(), FakeGenealogy(),
                                                 WalletIntel(), Adversarial())
                await model._load_historical_model()
            self.assertTrue(model.is_trained)
            self.assertEqual(model.data_status, "OK")
            self.assertTrue({"rug_30s", "rug_5m"}.issubset(model._historical_models))

            model.register_token("token")
            model.record_observation("token", {"type": "liquidity", "change_pct": -0.5})
            model.record_observation("token", {"type": "route", "feasible": False})
            state = await model._compute_hazard("token")
            self.assertEqual(state.data_status, "OK")
            self.assertGreaterEqual(state.hazard_30s, 0)
            self.assertGreaterEqual(state.hazard_5m, 0)


class TestApplicationStartup(unittest.IsolatedAsyncioTestCase):
    def test_market_observation_cohort_is_bounded_and_stable(self):
        desk = MemecoinQuantDesk()
        desk.global_config = {"market_observation_cohort_size": 2}
        desk.dataset_builder = SimpleNamespace(active_episodes={
            "old": SimpleNamespace(token="old", created_at=1),
            "middle": SimpleNamespace(token="middle", created_at=2),
            "new": SimpleNamespace(token="new", created_at=3),
        })
        desk._refresh_market_observation_cohort()
        self.assertEqual(desk._market_observation_cohort, {"middle", "new"})
        del desk.dataset_builder.active_episodes["middle"]
        desk._refresh_market_observation_cohort()
        self.assertEqual(desk._market_observation_cohort, {"old", "new"})

    async def test_offline_dry_run_initializes_without_provider_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"CHAMPION_STATE_PATH": str(Path(directory) / "champion.json")}
        ):
            desk = MemecoinQuantDesk(dry_run_override=True, offline=True)
            try:
                await desk.initialize()
                self.assertTrue(desk.dry_run)
                self.assertEqual(desk.readiness()["mode"], "DRY_RUN")
            finally:
                await desk.stop()


class TestGenealogyClustering(unittest.IsolatedAsyncioTestCase):
    async def test_profiles_without_graph_nodes_do_not_break_cluster_rebuild(self):
        graph = GenealogyGraph(solana_chain(), FakeRpc(), "")
        await graph._process_wallet_activity({"wallet": "isolated", "timestamp": 1})
        await graph._process_wallet_activity({"wallet": "a", "timestamp": 1})
        await graph._process_wallet_activity({"wallet": "b", "timestamp": 1})
        for _ in range(3):
            graph._add_relationship("a", "b", RelationshipType.CO_BOUGHT, 1)
        graph._add_relationship("a", "unhydrated-node", RelationshipType.TRANSFERRED, 1)
        await graph.build_clusters(min_connections=2)
        self.assertIsNone(graph.wallets["isolated"].cluster_id)
        self.assertEqual(graph.wallets["a"].cluster_id, graph.wallets["b"].cluster_id)
        self.assertEqual(len(graph.clusters), 1)


class TestResearchLedger(unittest.IsolatedAsyncioTestCase):
    async def test_promotion_evidence_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = str(Path(directory) / "champion_state.json")
            hypothesis = HypothesisSpec(
                hypothesis_id="model-v1", mechanism="test", target="net_elogw",
                features=["flow"], feature_hash="schema-1", model_type="test",
                model_params={}, training_window="point-in-time", threshold=0.0,
                sizing_rule={}, exit_rule={}, execution_policy={}, fakeability={},
                cost_model={}, falsifier="negative oos", kill_thesis="decay",
                source_provenance="fixture", trial_family="test", created_at=1.0,
            )
            first = ChampionChallengerFramework(
                min_oos_samples=1, min_portfolio_impact=0.0, state_path=state_path,
            )
            first.submit_hypothesis(hypothesis)
            first.record_trial_result(TrialResult(
                hypothesis_id="model-v1", stage="CHRONOLOGICAL_OOS", samples=10,
                metrics={}, oos_metrics={"elogw": 0.01}, portfolio_impact=0.01,
                passed=True, timestamp=2.0,
            ))
            await first._evaluate_challengers()
            await first.stop()

            second = ChampionChallengerFramework(state_path=state_path)
            self.assertIn("model-v1", second.shadow_models)
            self.assertEqual(len(second.trial_results), 1)
            self.assertEqual(second.hypotheses["model-v1"].feature_hash, "schema-1")

    @staticmethod
    def _hypothesis(hypothesis_id: str) -> HypothesisSpec:
        return HypothesisSpec(
            hypothesis_id=hypothesis_id, mechanism="test", target="net_elogw",
            features=["flow"], feature_hash="schema-1", model_type="test",
            model_params={}, training_window="point-in-time", threshold=0.0,
            sizing_rule={}, exit_rule={}, execution_policy={}, fakeability={},
            cost_model={}, falsifier="negative oos", kill_thesis="decay",
            source_provenance="fixture", trial_family="test", created_at=1.0,
        )

    async def test_full_promotion_pipeline_discovered_to_live_champion(self):
        framework = ChampionChallengerFramework(
            min_oos_samples=1, min_portfolio_impact=0.0,
            shadow_duration_hours=0, canary_duration_hours=0,
        )
        framework.submit_hypothesis(self._hypothesis("promo-1"))
        self.assertEqual(framework.hypotheses["promo-1"].status, ModelStatus.DISCOVERED.value)

        framework.record_trial_result(TrialResult(
            hypothesis_id="promo-1", stage="CHRONOLOGICAL_OOS", samples=10,
            metrics={}, oos_metrics={"elogw": 0.02}, portfolio_impact=0.02,
            passed=True, timestamp=2.0,
        ))
        await framework._evaluate_challengers()
        self.assertIn("promo-1", framework.shadow_models)
        self.assertNotIn("promo-1", framework.challengers)
        self.assertEqual(framework.hypotheses["promo-1"].status, ModelStatus.FORWARD_SHADOW.value)

        for _ in range(5):
            framework.record_forward_result("promo-1", {"elogw": 0.01, "pnl": 5.0})
        await framework._evaluate_shadow_models()
        self.assertIn("promo-1", framework.canary_models)
        self.assertNotIn("promo-1", framework.shadow_models)
        self.assertEqual(framework.hypotheses["promo-1"].status, ModelStatus.CANARY.value)

        for _ in range(20):
            framework.record_forward_result("promo-1", {"elogw": 0.01, "pnl": 5.0})
        await framework._evaluate_canary_models()
        self.assertIn("promo-1", framework.champions)
        self.assertNotIn("promo-1", framework.canary_models)
        self.assertTrue(framework.is_live("promo-1"))
        self.assertEqual(framework.hypotheses["promo-1"].status, ModelStatus.LIVE.value)
        self.assertEqual(framework.get_stats()["live_champions"], 1)

        for _ in range(framework.decay_window):
            framework.record_forward_result("promo-1", {"elogw": -0.01, "pnl": -5.0})
        await framework._monitor_champion_decay()
        self.assertEqual(framework.champions["promo-1"].status, "HIBERNATED")
        self.assertEqual(framework.get_stats()["hibernated_champions"], 1)

    async def test_shadow_and_canary_failures_retire_the_hypothesis(self):
        shadow_framework = ChampionChallengerFramework(
            min_oos_samples=1, min_portfolio_impact=0.0,
            shadow_duration_hours=0, canary_duration_hours=0,
        )
        shadow_framework.submit_hypothesis(self._hypothesis("shadow-fail"))
        shadow_framework.record_trial_result(TrialResult(
            hypothesis_id="shadow-fail", stage="CHRONOLOGICAL_OOS", samples=10,
            metrics={}, oos_metrics={"elogw": 0.02}, portfolio_impact=0.02,
            passed=True, timestamp=2.0,
        ))
        await shadow_framework._evaluate_challengers()
        for _ in range(5):
            shadow_framework.record_forward_result("shadow-fail", {"elogw": -0.01, "pnl": -5.0})
        await shadow_framework._evaluate_shadow_models()
        self.assertNotIn("shadow-fail", shadow_framework.shadow_models)
        self.assertNotIn("shadow-fail", shadow_framework.canary_models)
        self.assertEqual(shadow_framework.hypotheses["shadow-fail"].status, ModelStatus.RETIRED.value)
        self.assertEqual(shadow_framework.get_stats()["retired"], 1)

        canary_framework = ChampionChallengerFramework(
            min_oos_samples=1, min_portfolio_impact=0.0,
            shadow_duration_hours=0, canary_duration_hours=0,
        )
        canary_framework.submit_hypothesis(self._hypothesis("canary-fail"))
        canary_framework.record_trial_result(TrialResult(
            hypothesis_id="canary-fail", stage="CHRONOLOGICAL_OOS", samples=10,
            metrics={}, oos_metrics={"elogw": 0.02}, portfolio_impact=0.02,
            passed=True, timestamp=2.0,
        ))
        await canary_framework._evaluate_challengers()
        for _ in range(5):
            canary_framework.record_forward_result("canary-fail", {"elogw": 0.01, "pnl": 5.0})
        await canary_framework._evaluate_shadow_models()
        self.assertIn("canary-fail", canary_framework.canary_models)
        # Fewer than 20 canary trades -> retired for insufficient evidence, not promoted.
        for _ in range(5):
            canary_framework.record_forward_result("canary-fail", {"elogw": 0.01, "pnl": 5.0})
        await canary_framework._evaluate_canary_models()
        self.assertNotIn("canary-fail", canary_framework.canary_models)
        self.assertNotIn("canary-fail", canary_framework.champions)
        self.assertEqual(canary_framework.hypotheses["canary-fail"].status, ModelStatus.RETIRED.value)

    async def test_hourly_leads_survive_restart_and_restore_challengers(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "leads.jsonl"
            first_framework = ChampionChallengerFramework()
            first = GlobalResearchMiner(first_framework, ledger_path=str(ledger))
            await first._register_lead(ResearchLead(
                source_type="github", title="wallet copy policy", url="https://example.test/repo",
                summary="smart wallet copy research", language="en", license_spdx="Apache-2.0",
            ))
            self.assertTrue(ledger.exists())
            second_framework = ChampionChallengerFramework()
            second = GlobalResearchMiner(second_framework, ledger_path=str(ledger))
            await second._load_ledger()
            self.assertEqual(len(second.leads), 1)
            self.assertEqual(len(second_framework.hypotheses), 1)

    async def test_chinese_rss_lead_maps_to_research_mechanism(self):
        framework = ChampionChallengerFramework()
        miner = GlobalResearchMiner(framework)
        await miner._register_lead(ResearchLead(
            source_type="publisher_rss", title="聪明钱钱包跟单研究",
            url="https://example.test/zh-lead", summary="链上钱包行为", language="zh-cn",
            license_spdx="RSS_SUMMARY_ONLY",
        ), persist=False)
        self.assertEqual(miner.leads[0].mechanism, "wallet_copy_policy")
        self.assertEqual(len(framework.hypotheses), 1)


class FakeJupiter:
    def __init__(self):
        self.swap_build_called = False

    async def get_quote(self, input_mint, output_mint, amount, slippage_bps):
        return SwapQuote(input_mint, output_mint, amount, 12345, 0.01, [], RouteType.JUPITER_V1,
                         30, 12000, raw_quote={"outAmount": "12345"})

    async def get_swap_transaction(self, *args, **kwargs):
        self.swap_build_called = True
        raise AssertionError("dry-run must not build a transaction")


class FakeJito:
    async def send_bundle(self, transactions):
        raise AssertionError("dry-run must not submit a bundle")


class FakeRpc:
    async def request(self, method, params):
        raise AssertionError("dry-run must not submit or confirm a transaction")


class LiveFakeJupiter:
    """Returns a real, signable unsigned VersionedTransaction for live-path tests."""

    def __init__(self, keypair):
        self.keypair = keypair

    async def get_quote(self, input_mint, output_mint, amount, slippage_bps):
        return SwapQuote(input_mint, output_mint, amount, 12345, 0.01, [], RouteType.JUPITER_V1,
                         30, 12000, raw_quote={"outAmount": "12345"})

    async def get_swap_transaction(self, quote, user_public_key, *, priority_fee_lamports=0, jito_tip_lamports=0):
        message = MessageV0.try_compile(self.keypair.pubkey(), [], [], Hash.default())
        unsigned = VersionedTransaction.populate(message, [Signature.default()])
        return SwapTransaction(
            transaction=base64.b64encode(bytes(unsigned)).decode("ascii"),
            last_valid_block_height=1000, fee_payer=user_public_key, quote=quote,
            priority_fee=priority_fee_lamports, jito_tip=jito_tip_lamports,
            route_type=RouteType.JITO_BUNDLE if jito_tip_lamports else quote.route_type,
        )


class LiveFakeJito:
    def __init__(self, bundle_id=None, status=None):
        self.bundle_id = bundle_id
        self.status = status or {"value": []}
        self.sent = []

    async def send_bundle(self, transactions):
        self.sent.append(transactions)
        return self.bundle_id

    async def get_bundle_status(self, bundle_id):
        return self.status

    async def get_tip_floor_lamports(self, percentile=75):
        return None


class LiveFakeRpc:
    def __init__(self, send_transaction_response=None, get_transaction_response=None):
        self.send_transaction_response = send_transaction_response
        self.get_transaction_response = get_transaction_response
        self.sent = []

    async def request(self, method, params):
        if method == "sendTransaction":
            self.sent.append(params)
            return self.send_transaction_response
        if method == "getTransaction":
            return self.get_transaction_response
        raise AssertionError(f"unexpected RPC method: {method}")


class FakeTelegramMessage:
    def __init__(self, message_id, text, ts, views=0, forwards=0):
        self.id = message_id
        self.message = text
        self.date = SimpleNamespace(timestamp=lambda: ts)
        self.views = views
        self.forwards = forwards
        self.replies = None


class FakeTelegramClient:
    """Stands in for telethon.TelegramClient: connect/read only, never send/delete/react."""

    def __init__(self, session_path, api_id, api_hash, receive_updates=False):
        self.session_path = session_path
        self.api_id = api_id
        self.api_hash = api_hash
        self.receive_updates = receive_updates
        self.connected = False
        self.disconnected = False
        self.authorized = True
        self.messages_by_entity = {}
        self.event_handlers = []

    def add_event_handler(self, callback, event):
        # Read-only push subscription: telethon's real signature, recorded so a
        # test can assert the collector subscribed rather than fell back to
        # polling. No send/delete/react counterpart exists to be reached.
        self.event_handlers.append((callback, event))

    async def connect(self):
        self.connected = True

    async def is_user_authorized(self):
        return self.authorized

    async def disconnect(self):
        self.disconnected = True

    async def iter_messages(self, entity, limit=100):
        for message in self.messages_by_entity.get(entity, [])[:limit]:
            yield message


class TestExecution(unittest.IsolatedAsyncioTestCase):
    def test_jito_defaults_to_parallel_nearby_regions(self):
        with patch.dict("os.environ", {"JITO_BLOCK_ENGINE_URLS": ""}, clear=False):
            client = JitoClient()
        self.assertGreaterEqual(len(client.jito_urls), 4)
        self.assertTrue(any("dublin" in url for url in client.jito_urls))
        self.assertTrue(any("frankfurt" in url for url in client.jito_urls))

    async def test_identical_bundle_is_raced_across_relays(self):
        client = JitoClient()
        client.jito_urls = ["relay-a", "relay-b", "relay-c"]
        called = []

        async def rpc_at(url, method, params):
            called.append((url, method))
            await asyncio.sleep(0)
            return "same-bundle-id"

        client._rpc_at = rpc_at
        bundle_id = await client.send_bundle(["signed-transaction"])
        self.assertEqual(bundle_id, "same-bundle-id")
        self.assertEqual({url for url, _ in called}, set(client.jito_urls))
        self.assertEqual(len(client._bundle_routes[bundle_id]), 3)

    def test_jupiter_uses_current_gateway_and_free_tier_rate_gates(self):
        keyless = JupiterClient()
        keyed = JupiterClient(api_key="test-key")
        self.assertEqual(keyless.base_url, "https://api.jup.ag/swap/v1")
        self.assertGreaterEqual(keyless._minimum_interval, 2.0)
        self.assertGreaterEqual(keyed._minimum_interval, 1.0)

    async def test_dry_run_never_builds_signs_or_submits(self):
        keypair = Keypair()
        jupiter = FakeJupiter()
        builder = SolanaTransactionBuilder(FakeRpc(), keypair)
        engine = ExecutionEngine(solana_chain(), FakeRpc(), jupiter, FakeJito(), builder,
                                 CounterfactualExecutionLab(), dry_run=True)
        result = await engine.execute_swap("in", "out", 1000)
        self.assertTrue(result.success)
        self.assertTrue(result.simulated)
        self.assertFalse(result.submitted)
        self.assertFalse(result.landed)
        self.assertFalse(result.filled)
        self.assertEqual(result.status, TransactionStatus.SIMULATED)
        self.assertEqual(result.output_amount, 12345)
        self.assertEqual(result.actual_input_amount, 1000)
        self.assertEqual(result.native_balance_delta_lamports, 0)
        self.assertFalse(jupiter.swap_build_called)

    async def test_live_path_needs_second_acknowledgement(self):
        keypair = Keypair()
        engine = ExecutionEngine(solana_chain(), FakeRpc(), FakeJupiter(), FakeJito(),
                                 SolanaTransactionBuilder(FakeRpc(), keypair), None, dry_run=False)
        result = await engine.execute_swap("in", "out", 1000)
        self.assertEqual(result.status, TransactionStatus.REJECTED)
        self.assertIn("locked", result.error)

    async def test_versioned_transaction_signature_replacement(self):
        keypair = Keypair()
        message = MessageV0.try_compile(keypair.pubkey(), [], [], Hash.default())
        unsigned = VersionedTransaction.populate(message, [Signature.default()])
        encoded = SolanaTransactionBuilder(FakeRpc(), keypair).sign_versioned_transaction(bytes(unsigned))
        signed = VersionedTransaction.from_bytes(base64.b64decode(encoded))
        self.assertTrue(signed.verify_with_results()[0])

    def test_landed_wallet_deltas_drive_accounting(self):
        owner = "wallet"
        tx = {
            "transaction": {"message": {"accountKeys": [{"pubkey": owner}, {"pubkey": "other"}]}},
            "meta": {"preBalances": [1_000, 2_000], "postBalances": [725, 2_000]},
        }
        self.assertEqual(ExecutionEngine._native_balance_delta(tx, owner), -275)
        meta = {
            "preTokenBalances": [{"mint": "token", "owner": owner, "uiTokenAmount": {"amount": "500"}}],
            "postTokenBalances": [{"mint": "token", "owner": owner, "uiTokenAmount": {"amount": "125"}}],
        }
        self.assertEqual(ExecutionEngine._token_balance_decrease(meta, "token", owner), 375)

    def _live_engine(self, jupiter=None, jito=None, rpc=None, confirmation_timeout=45.0):
        keypair = Keypair()
        builder = SolanaTransactionBuilder(rpc or LiveFakeRpc(), keypair)
        engine = ExecutionEngine(
            solana_chain(), rpc or LiveFakeRpc(), jupiter or LiveFakeJupiter(keypair),
            jito or LiveFakeJito(), builder, CounterfactualExecutionLab(),
            dry_run=False, confirmation_timeout=confirmation_timeout,
        )
        return engine

    async def test_live_jito_bundle_fills_with_verified_output_balance_delta(self):
        with patch.dict("os.environ", {"ALLOW_LIVE_TRADING": "yes-i-understand"}):
            rpc = LiveFakeRpc(get_transaction_response={
                "slot": 555, "meta": {
                    "err": None, "fee": 5000,
                    "preTokenBalances": [], "postTokenBalances": [
                        {"mint": "out", "owner": None, "uiTokenAmount": {"amount": "12345"}},
                    ],
                    "preBalances": [2_000_000_000], "postBalances": [1_998_995_000],
                },
                "transaction": {"message": {"accountKeys": [{"pubkey": None}]}},
            })
            jito = LiveFakeJito(bundle_id="bundle-1",
                                status={"value": [{"confirmationStatus": "confirmed", "transactions": ["sig-1"]}]})
            engine = self._live_engine(rpc=rpc, jito=jito)
            rpc.owner = engine.tx_builder.public_key
            rpc.get_transaction_response["meta"]["postTokenBalances"][0]["owner"] = engine.tx_builder.public_key
            rpc.get_transaction_response["transaction"]["message"]["accountKeys"][0]["pubkey"] = engine.tx_builder.public_key
            result = await engine.execute_swap("in", "out", 1000, use_jito=True)
        self.assertEqual(result.status, TransactionStatus.FILLED)
        self.assertTrue(result.success)
        self.assertTrue(result.submitted)
        self.assertTrue(result.landed)
        self.assertTrue(result.filled)
        self.assertEqual(result.signature, "sig-1")
        self.assertEqual(result.bundle_id, "bundle-1")
        self.assertEqual(result.filled_output_amount, 12345)
        self.assertEqual(result.slot, 555)

    async def test_live_jito_bundle_rejected_never_reaches_submitted(self):
        with patch.dict("os.environ", {"ALLOW_LIVE_TRADING": "yes-i-understand"}):
            jito = LiveFakeJito(bundle_id=None)
            engine = self._live_engine(jito=jito)
            result = await engine.execute_swap("in", "out", 1000, use_jito=True)
        self.assertEqual(result.status, TransactionStatus.FAILED)
        self.assertFalse(result.success)
        self.assertFalse(result.submitted)
        self.assertIsNone(result.signature)

    async def test_live_jito_bundle_submitted_but_never_confirmed_is_a_distinct_timeout_state(self):
        with patch.dict("os.environ", {"ALLOW_LIVE_TRADING": "yes-i-understand"}):
            jito = LiveFakeJito(bundle_id="bundle-2", status={"value": []})
            engine = self._live_engine(jito=jito, confirmation_timeout=0.05)
            result = await engine.execute_swap("in", "out", 1000, use_jito=True)
        self.assertEqual(result.status, TransactionStatus.TIMEOUT)
        self.assertFalse(result.success)
        self.assertTrue(result.submitted)
        self.assertEqual(result.bundle_id, "bundle-2")
        self.assertIsNone(result.signature)

    async def test_live_raw_submission_lands_but_does_not_fill_on_chain_revert(self):
        with patch.dict("os.environ", {"ALLOW_LIVE_TRADING": "yes-i-understand"}):
            rpc = LiveFakeRpc(
                send_transaction_response="sig-revert",
                get_transaction_response={"slot": 9, "meta": {"err": {"InstructionError": [0, "Custom"]}, "fee": 5000}},
            )
            engine = self._live_engine(rpc=rpc)
            result = await engine.execute_swap("in", "out", 1000, use_jito=False)
        self.assertEqual(result.status, TransactionStatus.LANDED)
        self.assertFalse(result.success)
        self.assertTrue(result.submitted)
        self.assertTrue(result.landed)
        self.assertFalse(result.filled)
        self.assertEqual(result.signature, "sig-revert")

    async def test_live_raw_submission_that_never_lands_times_out(self):
        with patch.dict("os.environ", {"ALLOW_LIVE_TRADING": "yes-i-understand"}):
            rpc = LiveFakeRpc(send_transaction_response="sig-lost", get_transaction_response=None)
            engine = self._live_engine(rpc=rpc, confirmation_timeout=0.05)
            result = await engine.execute_swap("in", "out", 1000, use_jito=False)
        self.assertEqual(result.status, TransactionStatus.TIMEOUT)
        self.assertFalse(result.success)
        self.assertTrue(result.submitted)
        self.assertFalse(result.landed)
        self.assertFalse(result.filled)
    async def test_partial_exit_uses_verified_token_debit_for_cost_basis(self):
        class PartialExecution:
            async def execute_sell(self, *args, **kwargs):
                return ExecutionResult(
                    True, TransactionStatus.FILLED, actual_input_amount=200,
                    filled_output_amount=40_000_000, filled=True, landed=True, submitted=True,
                )
        class Recorder:
            def __init__(self): self.attempts=[]
            def record_execution_attempt(self, token, attempt): self.attempts.append(attempt)
        desk = MemecoinQuantDesk()
        desk.execution_engine = PartialExecution()
        desk.dataset_builder = Recorder()
        desk.elogw_engine = ElogwEngine(MultiHeadPredictor())
        desk.counterfactual_lab = CounterfactualExecutionLab()
        desk.sol_price_usd = 100
        position = {
            "token": "token", "size_tokens": 1_000, "remaining_cost_usd": 100,
            "risk_contribution": 0.02, "decision_id": "missing",
        }
        desk.elogw_engine.update_position("token", position)
        await desk._execute_exit("token", position, 0.5, "test_partial")
        remaining = desk.elogw_engine.open_positions["token"]
        self.assertEqual(remaining["size_tokens"], 800)
        self.assertEqual(remaining["remaining_cost_usd"], 80)
        self.assertEqual(desk.dataset_builder.attempts[-1]["requested_tokens"], 500)
        self.assertEqual(desk.dataset_builder.attempts[-1]["actual_sold_tokens"], 200)


class FakeGenealogy:
    wallets = {}
    token_launch_times = {}

    def get_deployer_profile(self, address):
        return None


class TestWalletAndCoordination(unittest.IsolatedAsyncioTestCase):
    def test_standard_solana_transaction_becomes_wallet_swap(self):
        wallet = "wallet"
        tx = {
            "blockTime": 123,
            "transaction": {"message": {"accountKeys": [{"pubkey": wallet}]}},
            "meta": {
                "fee": 5_000, "preBalances": [2_000_005_000], "postBalances": [1_000_000_000],
                "preTokenBalances": [{"owner": wallet, "mint": "token",
                                       "uiTokenAmount": {"uiAmountString": "0"}}],
                "postTokenBalances": [{"owner": wallet, "mint": "token",
                                        "uiTokenAmount": {"uiAmountString": "100"}}],
            },
        }
        enhanced = WalletIntelligenceEngine._standard_tx_to_enhanced(
            wallet, {"signature": "sig", "blockTime": 123}, tx,
        )
        normalized = WalletIntelligenceEngine._normalize_swap(wallet, enhanced)
        self.assertEqual(normalized["side"], "buy")
        self.assertEqual(normalized["amount"], 100)
        self.assertEqual(normalized["base_value"], 1.0)

    def test_wallet_history_uses_fifo_closed_round_trips(self):
        wallet = "wallet"
        engine = WalletIntelligenceEngine(solana_chain(), FakeRpc(), FakeGenealogy(), "")
        txs = [
            {"signature": "buy", "timestamp": 1, "nativeTransfers": [{"fromUserAccount": wallet, "amount": 1_000_000_000}],
             "tokenTransfers": [{"toUserAccount": wallet, "fromUserAccount": "curve", "mint": "token", "tokenAmount": 100}]},
            {"signature": "sell", "timestamp": 11, "nativeTransfers": [{"toUserAccount": wallet, "amount": 2_500_000_000}],
             "tokenTransfers": [{"fromUserAccount": wallet, "toUserAccount": "curve", "mint": "token", "tokenAmount": 50}]},
        ]
        result = engine._reconstruct_wallet_trades(wallet, txs)
        self.assertEqual(result["swap_count"], 2)
        self.assertEqual(len(result["closed_trades"]), 1)
        self.assertAlmostEqual(result["closed_trades"][0]["multiple"], 5.0)
        self.assertAlmostEqual(result["closed_trades"][0]["realized_pnl"], 2.0)

    def test_wallet_regime_is_not_fabricated_from_missing_history(self):
        engine = WalletIntelligenceEngine(solana_chain(), FakeRpc(), FakeGenealogy(), "")
        self.assertEqual(engine._classify_regime({"timestamp": 11, "multiple": 5.0, "data_status": "OK"}),
                         WalletRegime.GENERAL_HISTORY)
        self.assertIsNone(engine._classify_regime({"timestamp": 11, "multiple": 5.0,
                                                   "data_status": "DATA_BLOCKED"}))
        self.assertEqual(engine._classify_regime({"regime": "post_migration"}), WalletRegime.POST_MIGRATION)

    async def test_wallet_history_classifies_ultra_early_from_a_real_detected_launch_time(self):
        wallet = "wallet"
        graph = GenealogyGraph(solana_chain(), FakeRpc(), "")
        await graph._process_token_creation({"token": "token", "deployer": "dev", "timestamp": 0})
        engine = WalletIntelligenceEngine(solana_chain(), FakeRpc(), graph, "")
        txs = [
            {"signature": "buy", "timestamp": 1, "nativeTransfers": [{"fromUserAccount": wallet, "amount": 1_000_000_000}],
             "tokenTransfers": [{"toUserAccount": wallet, "fromUserAccount": "curve", "mint": "token", "tokenAmount": 100}]},
            {"signature": "sell", "timestamp": 11, "nativeTransfers": [{"toUserAccount": wallet, "amount": 2_500_000_000}],
             "tokenTransfers": [{"fromUserAccount": wallet, "toUserAccount": "curve", "mint": "token", "tokenAmount": 50}]},
        ]
        await engine._build_wallet_history(wallet, txs)
        self.assertEqual(engine.data_status[wallet], "OK")
        perf = engine.regime_performances[wallet][WalletRegime.ULTRA_EARLY]
        self.assertEqual(perf.trades, 1)
        self.assertEqual(perf.win_rate_2x, 1.0)

    async def test_wallet_history_classifies_early_curve_and_leaves_late_entries_unclassified(self):
        wallet = "wallet"
        graph = GenealogyGraph(solana_chain(), FakeRpc(), "")
        await graph._process_token_creation({"token": "early", "deployer": "dev", "timestamp": 0})
        await graph._process_token_creation({"token": "late", "deployer": "dev", "timestamp": 0})
        engine = WalletIntelligenceEngine(solana_chain(), FakeRpc(), graph, "")
        early_txs = [
            {"signature": "buy-early", "timestamp": 60,
             "nativeTransfers": [{"fromUserAccount": wallet, "amount": 1_000_000_000}],
             "tokenTransfers": [{"toUserAccount": wallet, "fromUserAccount": "curve", "mint": "early", "tokenAmount": 100}]},
            {"signature": "sell-early", "timestamp": 70,
             "nativeTransfers": [{"toUserAccount": wallet, "amount": 2_000_000_000}],
             "tokenTransfers": [{"fromUserAccount": wallet, "toUserAccount": "curve", "mint": "early", "tokenAmount": 50}]},
        ]
        late_txs = [
            {"signature": "buy-late", "timestamp": 900,
             "nativeTransfers": [{"fromUserAccount": wallet, "amount": 1_000_000_000}],
             "tokenTransfers": [{"toUserAccount": wallet, "fromUserAccount": "curve", "mint": "late", "tokenAmount": 100}]},
            {"signature": "sell-late", "timestamp": 910,
             "nativeTransfers": [{"toUserAccount": wallet, "amount": 2_000_000_000}],
             "tokenTransfers": [{"fromUserAccount": wallet, "toUserAccount": "curve", "mint": "late", "tokenAmount": 50}]},
        ]
        await engine._build_wallet_history(wallet, early_txs)
        self.assertIn(WalletRegime.EARLY_CURVE, engine.regime_performances[wallet])
        self.assertNotIn(WalletRegime.ULTRA_EARLY, engine.regime_performances[wallet])

        await engine._build_wallet_history(wallet, late_txs)
        # A launch time is known for "late" too, but the entry is well past
        # both timing buckets. It may be recorded as a verified round trip,
        # yet must never be promoted into a timing bucket the evidence does
        # not support, nor into a migration-phase regime with no migration
        # timestamp on record.
        self.assertNotIn(WalletRegime.ULTRA_EARLY, engine.regime_performances[wallet])
        self.assertNotIn(WalletRegime.PRE_MIGRATION, engine.regime_performances[wallet])
        self.assertNotIn(WalletRegime.POST_MIGRATION, engine.regime_performances[wallet])

    async def test_wallet_history_stays_unclassified_without_a_known_launch_time(self):
        wallet = "wallet"
        engine = WalletIntelligenceEngine(solana_chain(), FakeRpc(), FakeGenealogy(), "")
        txs = [
            {"signature": "buy", "timestamp": 1, "nativeTransfers": [{"fromUserAccount": wallet, "amount": 1_000_000_000}],
             "tokenTransfers": [{"toUserAccount": wallet, "fromUserAccount": "curve", "mint": "token", "tokenAmount": 100}]},
            {"signature": "sell", "timestamp": 11, "nativeTransfers": [{"toUserAccount": wallet, "amount": 2_500_000_000}],
             "tokenTransfers": [{"fromUserAccount": wallet, "toUserAccount": "curve", "mint": "token", "tokenAmount": 50}]},
        ]
        await engine._build_wallet_history(wallet, txs)
        # A verified closed round trip is real evidence and may be recorded as
        # GENERAL_HISTORY, but with no launch or migration timestamp on record
        # no launch-relative regime may be fabricated from it.
        for timed in (WalletRegime.ULTRA_EARLY, WalletRegime.EARLY_CURVE,
                      WalletRegime.PRE_MIGRATION, WalletRegime.POST_MIGRATION):
            self.assertNotIn(timed, engine.regime_performances[wallet])

    def test_public_coordination_requires_evidence_and_detects_same_slot(self):
        class WalletIntel:
            def get_top_wallets(self, limit=50):
                return []
        miner = PublicCoordinationMiner(FakeGenealogy(), WalletIntel())
        self.assertEqual(miner.get_features("token")["status"], "DATA_BLOCKED")
        for index in range(3):
            miner.record_trade("token", {"side": "buy", "wallet": f"w{index}", "slot": 9,
                                          "amount": index + 1})
        features = miner.get_features("token")
        self.assertEqual(features["status"], "OK")
        self.assertGreater(features["coordination_score"], 0)
        self.assertEqual(features["coordinated_buyer_fraction"], 1)

    def test_shared_funder_fires_from_same_transaction_funding_transfers(self):
        """End to end: a bundled launch tx funds 3 different buyer wallets
        from one operator wallet -- exactly what extract_system_transfers
        surfaces from a real transaction with zero extra RPC calls."""
        class WalletIntel:
            def get_top_wallets(self, limit=50):
                return []
        miner = PublicCoordinationMiner(FakeGenealogy(), WalletIntel())
        funding_transfers = [
            {"from": "operator", "to": f"buyer{i}", "lamports": 1_000_000_000} for i in range(3)
        ]
        for transfer in funding_transfers:
            miner.record_funding("token", transfer["to"], transfer["from"], transfer["lamports"] / 1e9, 100)
        for i in range(3):
            miner.record_trade("token", {"side": "buy", "wallet": f"buyer{i}", "amount": 1})
        features = miner.get_features("token")
        self.assertEqual(features["status"], "OK")
        kinds = {item["kind"] for item in features["evidence"]}
        self.assertIn("shared_funder", kinds)
        self.assertEqual(set(features["coordinated_wallets"]), {"buyer0", "buyer1", "buyer2"})


class TestRpcProtocol(unittest.TestCase):
    def test_healthy_rpc_is_preferred_over_zero_latency_degraded_provider(self):
        chain = solana_chain()
        chain.rpc_endpoints = [
            RPCEndpointConfig("https://healthy.invalid", "wss://healthy.invalid"),
            RPCEndpointConfig("https://rejected.invalid", "wss://rejected.invalid"),
        ]
        manager = RPCManager(chain)
        manager.endpoints[0].health = RPCHealth.HEALTHY
        manager.endpoints[0].latency_ms = 100
        manager.endpoints[1].health = RPCHealth.DEGRADED
        manager.endpoints[1].latency_ms = 0
        self.assertEqual(manager._select_endpoint(prefer_ws=True), manager.endpoints[0])
        self.assertEqual(manager.get_ws_urls(), ["wss://healthy.invalid", "wss://rejected.invalid"])

    def test_official_helius_endpoint_and_websocket_are_paired(self):
        with patch.dict("os.environ", {"HELIUS_API_KEY": "helius-test", "ALCHEMY_KEY": "alchemy-test"}):
            solana = ChainRegistry("config/chains.yaml").get_chain("solana")
        helius = next(ep for ep in solana.rpc_endpoints if "helius" in ep.url)
        self.assertEqual(helius.url, "https://mainnet.helius-rpc.com/?api-key=helius-test")
        self.assertEqual(helius.ws_url, "wss://mainnet.helius-rpc.com/?api-key=helius-test")

    def test_health_stats_never_expose_rpc_credentials(self):
        chain = solana_chain()
        chain.rpc_endpoints = [RPCEndpointConfig(
            "https://mainnet.helius-rpc.com/?api-key=super-secret",
            "wss://solana-mainnet.g.alchemy.com/v2/another-secret",
        )]
        stats = RPCManager(chain).get_stats()
        serialized = json.dumps(stats)
        self.assertNotIn("super-secret", serialized)
        self.assertNotIn("another-secret", serialized)
        self.assertEqual(stats["endpoints"][0]["url"], "https://mainnet.helius-rpc.com")

    def test_chain_aware_health_probe(self):
        solana = RPCManager(solana_chain())
        self.assertEqual(solana._health_probe(), ("getHealth", []))
        evm_chain = solana_chain()
        evm_chain.chain_type = ChainType.EVM
        self.assertEqual(RPCManager(evm_chain)._health_probe(), ("eth_blockNumber", []))

    def test_vendored_yellowstone_bindings_build_bidirectional_request(self):
        client = YellowstoneClient("https://example.com")
        validation = client.validate_setup()
        self.assertEqual(validation["status"], "OK")
        client._proto = validation["proto"]
        request = client._build_grpc_request(create_combined_subscription())
        self.assertEqual(request.commitment, 0)
        self.assertIn("memecoin_programs", request.transactions)


class TestRpcFallback(unittest.IsolatedAsyncioTestCase):
    async def test_initial_backlog_is_primed_not_replayed(self):
        class Rpc:
            def __init__(self):
                self.poll = 0

            async def request(self, method, params):
                if method == "getSignaturesForAddress":
                    self.poll += 1
                    return ([{"signature": "new", "err": None}, {"signature": "old", "err": None}]
                            if self.poll > 1 else [{"signature": "old", "err": None}])
                if method == "getTransaction":
                    return {"signature": params[0]}
                raise AssertionError(method)

        events = []
        stream = SolanaRpcProgramStream(Rpc(), ["program"])
        stream.on("transaction", events.append)
        successes, failures = await stream.poll_once()
        self.assertEqual((successes, failures), (1, {}))
        self.assertEqual(events, [])
        await stream.poll_once()
        self.assertEqual(events, [{"signature": "new"}])

    async def test_one_failing_program_does_not_poison_rpc_fallback(self):
        class Rpc:
            async def request(self, method, params):
                if params[0] == "bad-program":
                    raise RuntimeError("provider throttled")
                return []

        stream = SolanaRpcProgramStream(Rpc(), ["good-program", "bad-program"])
        successes, failures = await stream.poll_once()
        self.assertEqual(successes, 1)
        self.assertEqual(failures, {"bad-program": "provider throttled"})


class TestShadowMarketObservation(unittest.IsolatedAsyncioTestCase):
    async def test_social_candidate_requires_verified_spl_mint(self):
        class Rpc:
            async def request(self, method, params):
                self.method = method
                return {"value": {
                    "owner": TOKEN_PROGRAM,
                    "data": {"parsed": {"type": "mint", "info": {}}},
                }}

        class Detector:
            def __init__(self):
                self.candidates = []

            async def _on_candidate(self, candidate):
                self.candidates.append(candidate)

        desk = MemecoinQuantDesk()
        desk.solana_rpc = Rpc()
        desk.detection_engine = Detector()
        await desk._triage_social_candidate({
            "token": "FySyjuXTts9mTz2wjyuSXAz4bEBv6v5qxCTcLAMd4mVX",
            "platform": "telegram", "account": "alerts_bot", "credibility": 0.5,
            "timestamp": 1_700_000_000,
        })
        self.assertEqual(len(desk.detection_engine.candidates), 1)
        candidate = desk.detection_engine.candidates[0]
        self.assertEqual(candidate.source, DetectionSource.SOCIAL)
        self.assertTrue(candidate.metadata["mint_verified"])

    async def test_candidate_pipeline_rechecks_first_seconds_without_duplicates(self):
        desk = MemecoinQuantDesk()
        desk.global_config = {"candidate_recheck_delays_seconds": [0, 0.001, 0.003]}
        desk._candidate_semaphore = asyncio.Semaphore(1)
        desk.elogw_engine = SimpleNamespace(open_positions={})
        observed = []

        async def evaluate(candidate):
            observed.append(candidate.address)

        desk._evaluate_candidate = evaluate
        await desk._candidate_pipeline(SimpleNamespace(address="token"))
        self.assertEqual(observed, ["token", "token", "token"])

    async def test_blocked_prediction_still_collects_price_path(self):
        class Jupiter:
            def __init__(self):
                self.quotes = [SimpleNamespace(output_amount=1_000_000, price_impact_pct=0.01),
                               SimpleNamespace(output_amount=2_000_000, price_impact_pct=0.02)]

            async def get_quote(self, *args, **kwargs):
                return self.quotes.pop(0)

        class Recorder:
            def __init__(self):
                self.items = []

            def record_market_observation(self, token, observation):
                self.items.append((token, observation))

        class Hazard:
            def __init__(self):
                self.items = []

            def record_observation(self, token, observation):
                self.items.append((token, observation))

        desk = MemecoinQuantDesk()
        desk.jupiter = Jupiter()
        desk.dataset_builder = Recorder()
        desk.rug_hazard = Hazard()
        desk.counterfactual_lab = CounterfactualExecutionLab()
        desk.sol_price_usd = 100
        await desk._observe_token_market("token", 123)
        observation = desk.dataset_builder.items[0][1]
        self.assertEqual(observation["data_status"], "OK")
        self.assertEqual(observation["price_multiple"], 1)
        self.assertTrue(observation["route_feasible"])
        self.assertEqual(desk.rug_hazard.items[0][1]["type"], "route")


class TestPointInTimeResearch(unittest.IsolatedAsyncioTestCase):
    def builder(self):
        no_op = SimpleNamespace()
        return PointInTimeDatasetBuilder(
            solana_chain(), FakeRpc(), no_op, no_op, no_op, no_op, no_op, no_op, no_op,
        )

    async def test_pit_outcome_uses_observed_path_and_route_feasibility(self):
        builder = self.builder()
        episode = LaunchEpisode("token", "solana", 100, "dev", "pump", "curve", "wsol")
        episode.market_observations.extend([
            {"timestamp": 100, "price_usd": 1.0, "route_feasible": True, "price_impact_pct": 0.02},
            {"timestamp": 110, "price_usd": 6.0, "route_feasible": True, "price_impact_pct": 0.05,
             "migrated": True},
            {"timestamp": 120, "price_usd": 0.3, "route_feasible": False, "price_impact_pct": 0.9},
        ])
        episode.execution_attempts.append({"realized_pnl_usd": 12.5})
        outcome = await builder._determine_final_outcome(episode)
        self.assertEqual(outcome["status"], "OK")
        self.assertEqual(outcome["max_multiple"], 6)
        self.assertEqual(outcome["feasible_exit_multiple"], 6)
        self.assertTrue(outcome["migrated"])
        self.assertAlmostEqual(outcome["max_drawdown"], 0.95)
        self.assertEqual(outcome["rug_time"], 20)
        self.assertEqual(outcome["realized_pnl"], 12.5)
        # A drawdown past 90% is the only rug signal any producer in this
        # codebase currently emits (nothing sets "rugged": True on an
        # observation). It must supply rug_time itself, or every rugged
        # episode's rug_time stays null and the P_RUG_30S/P_RUG_5M labels
        # would be unconditionally 0.0 -- see snapshot_labels_uses_drawdown
        # below for the label-level consequence.
        self.assertTrue(outcome["rugged"])
        self.assertEqual(outcome["rug_time"], 20)

    async def test_slow_bleed_without_explicit_rug_event_still_gets_a_rug_time(self):
        builder = self.builder()
        episode = LaunchEpisode("token", "solana", 0, "dev", "pump", "curve", "wsol")
        episode.market_observations.extend([
            {"timestamp": 0, "price_usd": 1.0},
            {"timestamp": 5, "price_usd": 2.0},
            {"timestamp": 12, "price_usd": 0.1},
            {"timestamp": 40, "price_usd": 0.02},
        ])
        outcome = await builder._determine_final_outcome(episode)
        self.assertTrue(outcome["rugged"])
        self.assertEqual(outcome["rug_time"], 12)

    async def test_pit_outcome_recognizes_native_migration_event(self):
        episode = LaunchEpisode("token", "solana", 100, "dev", "pump", "curve", "wsol")
        episode.market_observations.extend([
            {"timestamp": 100, "price_usd": 1.0},
            {"timestamp": 110, "price_usd": 1.2, "type": "migration"},
            {"timestamp": 120, "price_usd": 1.1},
        ])
        outcome = await self.builder()._determine_final_outcome(episode)
        self.assertTrue(outcome["migrated"])

    async def test_missing_prices_are_explicitly_data_blocked(self):
        episode = LaunchEpisode("token", "solana", 100, "dev", "pump", "curve", "wsol")
        outcome = await self.builder()._determine_final_outcome(episode)
        self.assertEqual(outcome["status"], "DATA_BLOCKED")

    def test_prelaunch_snapshot_rejects_post_launch_context(self):
        builder = self.builder()
        builder.start_episode("token", "dev", "pump", "curve", "wsol", detected_at=100,
                              prelaunch_context={"as_of": 101, "social_features": {"calls": 5}})
        snapshot = builder.active_episodes["token"].snapshots[SnapshotTimepoint.PRELAUNCH]
        self.assertEqual(builder.active_episodes["token"].prelaunch_status, "DATA_BLOCKED")
        self.assertEqual(snapshot.social_features["status"], "DATA_BLOCKED")

    async def test_flow_snapshot_excludes_observations_after_cutoff(self):
        builder = self.builder()
        episode = LaunchEpisode("token", "solana", 100, "dev", "pump", "curve", "wsol")
        episode.market_observations.extend([
            {"type": "trade", "side": "buy", "wallet": "a", "slot": 1, "timestamp": 101},
            {"type": "trade", "side": "buy", "wallet": "b", "slot": 2, "timestamp": 109},
            {"type": "trade", "side": "buy", "wallet": "future", "slot": 2, "timestamp": 111},
        ])
        features = await builder._capture_flow_features(episode, as_of=110)
        self.assertEqual(features["status"], "OK")
        self.assertEqual(features["observed_trade_count"], 2)
        self.assertEqual(features["buy_velocity"], 0.2)

    async def test_active_episode_checkpoint_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.builder()
            first.storage_path = directory
            first.start_episode("token", "dev", "pump", "curve", "wsol", detected_at=100)
            first.record_market_observation("token", {"type": "trade", "timestamp": 101, "side": "buy"})
            await first._flush_active()

            second = self.builder()
            second.storage_path = directory
            second._load_active_checkpoints()
            self.assertIn("token", second.active_episodes)
            restored = second.active_episodes["token"]
            self.assertEqual(restored.created_at, 100)
            self.assertEqual(restored.market_observations[0]["timestamp"], 101)


class TestCounterfactuals(unittest.TestCase):
    def test_now_wait_and_no_jito_policies_are_resolved_from_observations(self):
        lab = CounterfactualExecutionLab()
        decision_id = lab.record_decision("token", {"signal": 1}, {"buy": True})
        created = lab.decisions[decision_id]["created_at"]
        lab.record_execution(decision_id, {"landed": True, "pnl": 10})
        lab.record_market_observation("token", 1.0, created)
        lab.record_market_observation("token", 1.2, created + 1)
        lab.record_market_observation("token", 1.5, created + 3)
        resolved = lab.resolve_decision(decision_id, 10)
        self.assertEqual({item["policy"] for item in resolved}, {"now", "wait_1s", "wait_3s", "no_jito"})
        self.assertEqual(lab.decisions[decision_id]["status"], "RESOLVED")


class TestHazardTracking(unittest.IsolatedAsyncioTestCase):
    def test_hazard_feature_vector_is_point_in_time_and_detects_deterioration(self):
        observations = [
            {"timestamp": 100, "type": "trade", "side": "buy", "notional_usd": 1000,
             "price_multiple": 2.0},
            {"timestamp": 260, "type": "trade", "side": "sell", "notional_usd": 800,
             "price_multiple": 0.1},
            {"timestamp": 270, "type": "route", "feasible": False, "price_impact_pct": 0.5},
            {"timestamp": 280, "type": "liquidity", "change_pct": -0.5},
            {"timestamp": 400, "type": "trade", "side": "buy", "notional_usd": 10},
        ]
        vector = ContinuousRugHazardModel.feature_vector_from_observations(observations, 300)
        self.assertEqual(len(vector), 8)
        self.assertEqual(vector[0], 1.0)
        self.assertGreaterEqual(vector[3], 0.9)
        self.assertEqual(vector[4], 1.0)
        self.assertEqual(vector[5], 0.5)

    async def test_route_loss_and_sell_pressure_create_actionable_hazard(self):
        class WalletIntel:
            def get_top_wallets(self, limit=50):
                return []
        class Adversarial:
            def get_adaptive_weight(self, feature, base):
                return base
        model = ContinuousRugHazardModel(solana_chain(), FakeRpc(), FakeGenealogy(), WalletIntel(), Adversarial())
        model.register_token("token")
        for index in range(4):
            model.record_observation("token", {"type": "trade", "side": "sell", "notional_usd": 250,
                                               "wallet": f"w{index}"})
        model.record_observation("token", {"type": "route", "feasible": False})
        state = await model._compute_hazard("token")
        self.assertEqual(state.data_status, "OK")
        self.assertGreater(state.current_hazard, 0.4)
        self.assertIn(state.exit_urgency, {"MEDIUM", "HIGH", "CRITICAL"})


if __name__ == "__main__":
    unittest.main()


class TestPositionPredictionRefresh(unittest.IsolatedAsyncioTestCase):
    """The exit decision must read a current prediction, not the entry-time one.

    A position opened at T0 and still open at T+90s has been re-priced by every
    trade in between. Deciding whether to hold a drawdown from the entry-time
    continuation probability is deciding on evidence the market has already
    contradicted.
    """

    def _desk(self, prediction, liquidity=250_000.0):
        seen = []

        async def resolve_liquidity(candidate):
            return liquidity

        async def build_features(candidate, risk, liq):
            seen.append(liq)
            return PredictionFeatures("mint", "solana", 0)

        desk = SimpleNamespace(
            predictor=SimpleNamespace(_is_trained=True, predict=lambda features: prediction),
            _resolve_liquidity=resolve_liquidity,
            _build_prediction_features=build_features,
            liquidity_seen=seen,
            state_sequencer=StateSequencer(),
        )
        return desk

    @staticmethod
    def _position():
        stale = MultiHeadPrediction("mint", "solana", 0, p_2x=0.9, p_5x=0.8, p_10x=0.7)
        return {
            "candidate": SimpleNamespace(base_token=None), "risk_object": SimpleNamespace(),
            "liquidity_usd": 500_000.0, "prediction": _jsonable(stale),
            "prediction_object": stale, "prediction_status": "OK",
        }

    async def test_refresh_replaces_the_entry_time_prediction_and_liquidity(self):
        fresh = MultiHeadPrediction("mint", "solana", 0, p_2x=0.2, p_5x=0.05, p_10x=0.01)
        desk = self._desk(fresh, liquidity=120_000.0)
        position = self._position()

        await MemecoinQuantDesk._refresh_position_prediction(desk, "mint", position)

        self.assertEqual(position["prediction_status"], "OK")
        self.assertIs(position["prediction_object"], fresh)
        self.assertAlmostEqual(position["prediction"]["p_5x"], 0.05)
        # Depth is re-resolved, not carried from entry, and the fresh number is
        # what the feature vector is actually built against.
        self.assertEqual(position["liquidity_usd"], 120_000.0)
        self.assertEqual(desk.liquidity_seen, [120_000.0])

    async def test_unobservable_liquidity_blocks_rather_than_predicting_on_zero(self):
        desk = self._desk(MultiHeadPrediction("mint", "solana", 0, p_5x=0.05), liquidity=0.0)
        position = self._position()

        await MemecoinQuantDesk._refresh_position_prediction(desk, "mint", position)

        self.assertEqual(position["prediction_status"], "DATA_BLOCKED_LIQUIDITY")
        # The stale prediction survives untouched: a known-old number beats a
        # fresh one computed against depth of zero that nobody observed.
        self.assertAlmostEqual(position["prediction"]["p_5x"], 0.8)
        self.assertEqual(position["liquidity_usd"], 500_000.0)
        self.assertEqual(desk.liquidity_seen, [])

    async def test_untrained_predictor_leaves_the_position_alone(self):
        desk = self._desk(MultiHeadPrediction("mint", "solana", 0))
        desk.predictor = SimpleNamespace(_is_trained=False, predict=lambda f: None)
        position = self._position()

        await MemecoinQuantDesk._refresh_position_prediction(desk, "mint", position)

        self.assertEqual(position["prediction_status"], "OK")
        self.assertAlmostEqual(position["prediction"]["p_5x"], 0.8)

    async def test_manage_positions_exits_on_the_refreshed_continuation(self):
        """Entry said 'monster, hold the drawdown'; now it says 'get out'."""
        collapsed = MultiHeadPrediction("mint", "solana", 0, p_2x=0.05, p_5x=0.01, p_10x=0.0)
        desk = self._desk(collapsed)
        exits = []
        position = self._position()
        position.update({"entry_time": time.time() - 120, "high_water_multiple": 4.0,
                         "size_tokens": 1_000, "remaining_cost_usd": 100.0,
                         "ratchet_stages": ["cost_recovery"]})

        # The entry-time prediction would have held this position: continuation
        # 0.8 buys the mid trail (floor 2.32) and 2.6x sits above it. That is
        # the bug -- the same state exits only once the prediction is refreshed.
        self.assertIsNone(evaluate_exit(
            ExitPolicy.default(), 2.6, 4.0, 0.8, {"cost_recovery"}, 120.0))

        async def mark(token, pos):
            return 2.6, 260.0

        desk._refresh_position_prediction = (
            lambda token, pos: MemecoinQuantDesk._refresh_position_prediction(desk, token, pos))
        desk.rug_hazard = SimpleNamespace(should_exit=lambda t, p: (False, "", 0.0),
                                          observations={}, get_hazard=lambda t: None)
        desk.distribution_detector = DistributionDetector()
        desk.monster_machine = MonsterStateMachine()
        desk.last_slate_report = {}
        _attach_manifest(desk)
        desk._priced_holds = 0
        desk._unpriced_cycles = 0
        desk._suppressed_monster_banks = 0
        desk._read_distribution = lambda token: MemecoinQuantDesk._read_distribution(desk, token)
        desk._latest_curve_state = getattr(desk, "_latest_curve_state", {})
        desk._latest_pool_state = getattr(desk, "_latest_pool_state", {})
        desk.ops_events = []
        desk._record_ops_event = (
            lambda stream, payload: desk.ops_events.append((stream, payload)))
        desk._closed_pnl = {}
        desk._mechanism_growth = {}
        desk._exit_capacity = (
            lambda token, pos: MemecoinQuantDesk._exit_capacity(desk, token, pos))
        desk.min_escape_probability = 0.05
        desk.state_sequencer = StateSequencer()
        desk.model_feature_hash = "test"
        desk.action_policy = ActionValuePolicy()
        desk._score_actions = (
            lambda token, pos, mult, dist:
            MemecoinQuantDesk._score_actions(desk, token, pos, mult, dist))
        desk._apply_action = (
            lambda token, pos, dec, mult:
            MemecoinQuantDesk._apply_action(desk, token, pos, dec, mult))
        desk._estimate_escape = (
            lambda token, pos, hazard: MemecoinQuantDesk._estimate_escape(
                desk, token, pos, hazard))
        desk._update_monster_state = (
            lambda token, pos, dist, mult:
            MemecoinQuantDesk._update_monster_state(desk, token, pos, dist, mult))
        desk.elogw_engine = SimpleNamespace(open_positions={"mint": position})
        desk._mark_position = mark
        desk.exit_policy = ExitPolicy.default()
        desk._execute_exit = lambda token, pos, pct, reason: exits.append((reason, pct)) or _async_none()
        desk._consider_scale_in = lambda token, pos, mult: _async_none()

        await MemecoinQuantDesk._manage_positions(desk)

        self.assertIs(position["prediction_object"], collapsed)
        # A 35% drawdown off a 4x high water with continuation collapsed to 1%
        # is a trailing stop, not a dip to be held.
        self.assertTrue(exits, "collapsed continuation must be able to fire the trailing stop")
        self.assertIn("trailing_stop", exits[0][0])


async def _async_none():
    return None


class TestOpportunityAllocator(unittest.TestCase):
    """Capital must contest, not queue.

    The single-token hurdle test has a silent failure: ten mediocre positions
    that each cleared it will lock out an eleventh that is five times better,
    for as long as they are held.
    """

    @staticmethod
    def _opportunity(token, elogw, capital, seconds, liquidity=100_000.0, **kwargs):
        return Opportunity(token=token, elogw=elogw, capital_usd=capital,
                           expected_hold_seconds=seconds, liquidity_usd=liquidity, **kwargs)

    def test_ranking_is_growth_per_dollar_per_second(self):
        # Same edge, same capital, one recycles 30x faster.
        slow = self._opportunity("slow", 0.030, 100.0, 2_700.0)
        fast = self._opportunity("fast", 0.010, 100.0, 90.0)
        slate = OpportunityAllocator().rank([slow, fast])

        self.assertEqual([item.token for item in slate.ranked], ["fast", "slow"])
        # +3% held 45 minutes really is worse than +1% recycled every 90s.
        self.assertGreater(fast.growth_velocity, slow.growth_velocity)

    def test_a_bigger_edge_on_more_capital_does_not_automatically_win(self):
        # 0.05 on $1000 is a worse use of the marginal dollar than 0.01 on $50
        # at equal holding time, which is exactly what per-dollar scoring says.
        large = self._opportunity("large", 0.05, 1_000.0, 60.0)
        small = self._opportunity("small", 0.01, 50.0, 60.0)
        slate = OpportunityAllocator().rank([large, small])
        self.assertEqual(slate.best.token, "small")

    def test_missing_hold_time_blocks_rather_than_defaulting(self):
        unknown = self._opportunity("unknown", 0.05, 100.0, None)
        known = self._opportunity("known", 0.001, 100.0, 600.0)
        slate = OpportunityAllocator().rank([unknown, known])

        self.assertEqual([item.token for item in slate.ranked], ["known"])
        self.assertEqual(slate.blocked[0][1], "DATA_BLOCKED_HOLD_TIME")
        # A denominator nobody predicted must not be invented: assuming any
        # holding time would have ranked the unknown token first on its edge.
        self.assertEqual(slate.report()["blocked_reasons"], ["DATA_BLOCKED_HOLD_TIME"])

    def test_unobserved_depth_blocks_rather_than_ranking(self):
        slate = OpportunityAllocator().rank([
            self._opportunity("no_depth", 0.05, 100.0, 60.0, liquidity=None),
            self._opportunity("zero_depth", 0.05, 100.0, 60.0, liquidity=0.0),
        ])
        self.assertEqual(slate.ranked, [])
        self.assertEqual({reason for _, reason in slate.blocked}, {"DATA_BLOCKED_LIQUIDITY"})

    def test_infinite_elogw_is_never_ranked(self):
        slate = OpportunityAllocator().rank([
            Opportunity("blocked", float("-inf"), 100.0, 60.0, 100_000.0),
            self._opportunity("real", 0.01, 100.0, 60.0),
        ])
        self.assertEqual([item.token for item in slate.ranked], ["real"])

    def test_a_clearly_better_challenger_displaces_the_weakest_position(self):
        incumbent = self._opportunity("weak", 0.0005, 100.0, 3_600.0, is_open_position=True,
                                      held_multiple=1.1)
        challenger = self._opportunity("strong", 0.30, 100.0, 120.0)
        allocator = OpportunityAllocator(replacement_cost_pct=0.02)
        slate = allocator.rank([incumbent, challenger])

        self.assertEqual(len(slate.displacements), 1)
        move = slate.displacements[0]
        self.assertEqual(move.incumbent.token, "weak")
        self.assertEqual(move.challenger.token, "strong")
        self.assertGreater(move.score_gain, 0)
        # Capital released is the mark, not the cost basis.
        self.assertAlmostEqual(move.freed_capital_usd, 110.0)

    def test_marginally_better_challengers_never_churn_the_book(self):
        incumbent = self._opportunity("held", 0.010, 100.0, 300.0, is_open_position=True)
        barely = self._opportunity("barely", 0.0104, 100.0, 300.0)
        slate = OpportunityAllocator(min_displacement_gain_ratio=1.5).rank([incumbent, barely])
        self.assertEqual(slate.displacements, [])

    def test_round_trip_cost_is_charged_before_comparing(self):
        """The same challenger wins on a cheap venue and loses on a dear one."""
        incumbent = self._opportunity("held", 0.040, 100.0, 300.0, is_open_position=True)
        challenger = self._opportunity("new", 0.075, 100.0, 300.0)

        cheap = OpportunityAllocator(replacement_cost_pct=0.0, min_displacement_gain_ratio=1.5)
        dear = OpportunityAllocator(replacement_cost_pct=0.10, min_displacement_gain_ratio=1.5)

        self.assertEqual(len(cheap.rank([incumbent, challenger]).displacements), 1)
        self.assertEqual(dear.rank([incumbent, challenger]).displacements, [])

    def test_a_challenger_the_venue_cannot_absorb_is_not_proposed(self):
        incumbent = self._opportunity("held", 0.0002, 5_000.0, 3_600.0, is_open_position=True)
        # $5,000 of edge in a $200 pool is not $5,000 of edge. A theoretical
        # return you cannot fill is not a return, so no displacement is priced.
        thin = self._opportunity("thin", 0.50, 5_000.0, 60.0, liquidity=200.0)
        slate = OpportunityAllocator(replacement_cost_pct=0.0).rank([incumbent, thin])
        self.assertEqual(slate.displacements, [])

    def test_freed_capital_must_cover_the_size_the_edge_was_measured_at(self):
        """E[log W] is not linear in capital, so it is not rescaled."""
        incumbent = self._opportunity("small_position", 0.0001, 50.0, 3_600.0,
                                      is_open_position=True, held_multiple=1.0)
        big = self._opportunity("needs_1000", 0.50, 1_000.0, 60.0)
        slate = OpportunityAllocator(replacement_cost_pct=0.0).rank([incumbent, big])
        # Closing a $50 position does not fund a $1,000 trade, and quietly
        # claiming the $1,000 edge on $50 of capital would be the invention.
        self.assertEqual(slate.displacements, [])

        funded = self._opportunity("needs_100", 0.50, 100.0, 60.0)
        rich = self._opportunity("rich_position", 0.0001, 100.0, 3_600.0,
                                 is_open_position=True, held_multiple=2.0)
        self.assertEqual(
            len(OpportunityAllocator(replacement_cost_pct=0.0).rank([rich, funded]).displacements), 1)

    def test_displacements_are_capped_per_cycle(self):
        opportunities = [
            self._opportunity(f"held-{i}", 0.0001, 100.0, 3_600.0, is_open_position=True)
            for i in range(5)
        ] + [self._opportunity(f"new-{i}", 0.50, 100.0, 60.0) for i in range(5)]
        slate = OpportunityAllocator(max_displacements_per_cycle=2).rank(opportunities)
        self.assertEqual(len(slate.displacements), 2)
        # Each challenger is spent once: two incumbents cannot both be told to
        # make way for the same token.
        self.assertEqual(len({m.challenger.token for m in slate.displacements}), 2)

    def test_a_profitable_incumbent_is_not_displaced_by_a_worse_score(self):
        incumbent = self._opportunity("winner", 0.08, 100.0, 120.0, is_open_position=True,
                                      held_multiple=3.0)
        challenger = self._opportunity("hopeful", 0.02, 100.0, 120.0)
        slate = OpportunityAllocator().rank([incumbent, challenger])
        self.assertEqual(slate.displacements, [])
        self.assertEqual(slate.best.token, "winner")

    def test_sleeve_grouping_keeps_attribution_separable(self):
        slate = OpportunityAllocator().rank([
            self._opportunity("a", 0.01, 100.0, 60.0, sleeve="t0_sniper"),
            self._opportunity("b", 0.02, 100.0, 60.0, sleeve="migration"),
            self._opportunity("c", 0.03, 100.0, 60.0, sleeve="migration"),
        ])
        self.assertEqual(slate.report()["sleeves"], {"migration": 2, "t0_sniper": 1})


class TestCapitalContestWiring(unittest.IsolatedAsyncioTestCase):
    """A full book must not be able to lock out a far better launch.

    `should_trade` answers one token at a time, so `max_concurrent_positions`
    reads as "this token is rejected" when what actually happened is "capital
    is committed elsewhere". Those are different statements and only one of
    them is about the token.
    """

    @staticmethod
    def _prediction(token="new", **kwargs):
        base = dict(p_2x=0.85, p_5x=0.6, p_10x=0.4, p_50x=0.1, expected_hold_time=90.0,
                    expected_slippage=0.01)
        base.update(kwargs)
        return MultiHeadPrediction(token, "solana", 0.0, **base)

    def _desk(self, exit_clears=True):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), min_edge_bps=-10_000)
        engine.portfolio_value = 10_000.0
        weak = self._prediction("weak", p_2x=0.5, p_5x=0.2, p_10x=0.05, p_50x=0.0,
                                expected_hold_time=3_600.0)
        engine.open_positions["weak"] = {
            "size_tokens": 1_000, "remaining_cost_usd": 100.0, "risk_contribution": 0.01,
            "prediction_object": weak, "liquidity_usd": 80_000.0,
            "high_water_multiple": 1.0, "entry_time": time.time() - 1_800,
        }
        exits = []

        async def execute_exit(token, position, pct, reason):
            exits.append((token, pct, reason))
            if exit_clears:
                engine.open_positions.pop(token, None)

        async def refresh():
            return None

        desk = SimpleNamespace(
            elogw_engine=engine, sol_price_usd=150.0, wallet_equity_usd=10_000.0,
            opportunity_allocator=OpportunityAllocator(replacement_cost_pct=0.005,
                                                       min_displacement_gain_ratio=1.5),
            last_slate_report={}, _execute_exit=execute_exit,
            _refresh_portfolio_state=refresh, exits=exits,
        )
        desk._incumbent_opportunities = (
            lambda: MemecoinQuantDesk._incumbent_opportunities(desk))
        desk._feed_opportunity_quality = (
            lambda slate: MemecoinQuantDesk._feed_opportunity_quality(desk, slate))
        return desk

    async def _contest(self, desk, prediction, liquidity=200_000.0, reason="max_concurrent_positions"):
        candidate = SimpleNamespace(address="new", metadata={})
        return await MemecoinQuantDesk._contest_for_capital(
            desk, "new", candidate, prediction, liquidity, {"reason": reason})

    async def test_a_challenger_is_repriced_at_the_capital_a_displacement_frees(self):
        """A $500 optimum funded by a $100 exit is not still worth $500 of edge."""
        desk = self._desk()
        desk.elogw_engine.open_positions["weak"]["remaining_cost_usd"] = 40.0
        sizes = []
        engine = desk.elogw_engine
        original = engine.log_growth_at_fraction

        def spy(prediction, fraction):
            sizes.append(fraction)
            return original(prediction, fraction)

        engine.log_growth_at_fraction = spy
        await self._contest(desk, self._prediction())
        # The allocator asked what the edge is worth at the freed size rather
        # than reusing the optimum computed for a size it cannot fund.
        self.assertTrue(sizes, "challenger was never repriced")
        self.assertLess(min(sizes), engine.exposure_cap(200_000.0))

    async def test_a_far_better_candidate_closes_the_weakest_position_and_enters(self):
        desk = self._desk()
        should_trade, info = await self._contest(desk, self._prediction())

        self.assertEqual(len(desk.exits), 1)
        token, pct, reason = desk.exits[0]
        self.assertEqual((token, pct), ("weak", 1.0))
        self.assertEqual(reason, "displaced_by_new")
        self.assertTrue(should_trade, info)
        # The freed capital still has to clear the ordinary sizing path.
        self.assertIn("position_size_sol", info)

    async def test_a_similar_candidate_does_not_disturb_the_book(self):
        desk = self._desk()
        # Same shape as the incumbent: nothing here justifies a round trip.
        twin = self._prediction(p_2x=0.5, p_5x=0.2, p_10x=0.05, p_50x=0.0,
                                expected_hold_time=3_600.0)
        should_trade, info = await self._contest(desk, twin)

        self.assertEqual(desk.exits, [])
        self.assertFalse(should_trade)
        self.assertEqual(info["contest"], "no_incumbent_worth_displacing")
        self.assertIn("weak", desk.elogw_engine.open_positions)

    async def test_an_exit_that_did_not_fill_never_funds_the_challenger(self):
        """Capital that was not actually freed must not be spent."""
        desk = self._desk(exit_clears=False)
        should_trade, info = await self._contest(desk, self._prediction())

        self.assertEqual(len(desk.exits), 1)
        self.assertFalse(should_trade)
        self.assertEqual(info["contest"], "displacement_exit_did_not_fill")

    async def test_a_prediction_with_no_hold_time_cannot_contest(self):
        desk = self._desk()
        should_trade, info = await self._contest(
            desk, self._prediction(expected_hold_time=0.0))

        self.assertEqual(desk.exits, [])
        self.assertFalse(should_trade)
        # Unknown holding time means the per-second denominator is unknown, so
        # the candidate simply does not rank rather than ranking on a guess.
        self.assertEqual(info["contest"], "no_incumbent_worth_displacing")
        self.assertEqual(info["slate"]["blocked_reasons"], ["DATA_BLOCKED_HOLD_TIME"])

    async def test_safety_rejections_are_outside_the_contest(self):
        """The contest is reachable only from capacity rejections."""
        for reason in ("safety_rejection", "rug_risk_too_high", "daily_loss_kill_switch",
                       "insufficient_upside", "slippage_too_high", "liquidity_too_low",
                       "edge_below_threshold", "DATA_BLOCKED"):
            self.assertNotIn(reason, CAPACITY_REJECTIONS)
        self.assertEqual(
            CAPACITY_REJECTIONS,
            {"max_concurrent_positions", "total_exposure_limit", "portfolio_risk_limit"})


class TestHeldPositionBaselineGrowth(unittest.TestCase):
    """A position already open cannot be rejected by the entry constraint."""

    def _engine(self):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), drawdown_aversion_lambda=3.0)
        engine.portfolio_value = 10_000.0
        return engine

    @staticmethod
    def _weak():
        # Tail bad enough that opening this position would violate the
        # drawdown bound outright.
        return MultiHeadPrediction("t", "solana", 0.0, p_2x=0.12, p_5x=0.05,
                                   p_10x=0.0, p_50x=0.0, expected_slippage=0.01)

    def test_baseline_is_a_number_not_negative_infinity(self):
        engine = self._engine()
        baseline = engine.marginal_log_growth(self._weak(), 0.01, 1.0, 0.0)
        self.assertTrue(math.isfinite(baseline))
        # It is still a bad position -- just a finite bad, which is what any
        # comparison against it requires.
        self.assertLess(baseline, 0.0)

    def test_the_drawdown_bound_still_rejects_adding_to_it(self):
        engine = self._engine()
        fraction, gain = engine.plan_scale_in(self._weak(), 100.0, 1.0, 500_000.0,
                                              portfolio_value=10_000.0)
        self.assertEqual((fraction, gain), (0.0, 0.0))

    def test_the_entry_path_still_refuses_to_open_it(self):
        engine = self._engine()
        elogw, fraction, size_sol = engine.calculate_expected_log_growth(
            self._weak(), 150.0, 500_000.0)
        self.assertEqual(fraction, 0.0)
        self.assertEqual(size_sol, 0.0)

    def test_log_growth_at_fraction_agrees_with_the_optimiser(self):
        engine = self._engine()
        good = MultiHeadPrediction("t", "solana", 0.0, p_2x=0.8, p_5x=0.5, p_10x=0.3,
                                   p_50x=0.05, expected_slippage=0.01)
        best, fraction, _ = engine.calculate_expected_log_growth(good, 150.0, 500_000.0)
        self.assertAlmostEqual(engine.log_growth_at_fraction(good, fraction), best, places=12)
        # And it is genuinely an optimum, not a coincidence of the grid.
        self.assertLessEqual(engine.log_growth_at_fraction(good, fraction * 0.5), best)
        self.assertEqual(engine.log_growth_at_fraction(good, 0.0), 0.0)


class TestPumpFeeSchedule(unittest.TestCase):
    """Fees must be one versioned function, and must refuse to guess.

    Pump's docs put a market-cap-dependent dynamic schedule into force on
    2026-09-01 20:00 UTC. The tier table itself is published as an image, so
    it cannot honestly be hardcoded here. A system that quietly kept charging
    the legacy flat fee after that instant would keep producing labels and
    counterfactuals against economics that no longer exist -- worse than
    producing none, because nothing would signal it.
    """

    JUST_BEFORE = DYNAMIC_FEE_ACTIVATION_UTC - 1
    AT_ACTIVATION = DYNAMIC_FEE_ACTIVATION_UTC

    def test_the_legacy_flat_fee_applies_right_up_to_the_activation_instant(self):
        quote = PumpFeeSchedule().quote(at_utc=self.JUST_BEFORE)
        self.assertTrue(quote.ok)
        self.assertEqual(quote.total_bps, 100)
        self.assertEqual(quote.fee_lamports(1_000_000_000), 10_000_000)

    def test_activation_is_inclusive_of_its_own_instant(self):
        schedule = PumpFeeSchedule()
        self.assertFalse(schedule.is_dynamic(self.JUST_BEFORE))
        self.assertTrue(schedule.is_dynamic(self.AT_ACTIVATION))

    def test_an_unloaded_dynamic_schedule_blocks_instead_of_falling_back(self):
        quote = PumpFeeSchedule().quote(at_utc=self.AT_ACTIVATION,
                                        market_cap_lamports=50_000_000_000)
        self.assertFalse(quote.ok)
        self.assertEqual(quote.status, "DATA_BLOCKED")
        self.assertIn("FeeConfig", quote.reason)
        # The failure must be loud at the point of use, not a silent zero.
        with self.assertRaises(ValueError):
            quote.fee_lamports(1_000_000_000)

    def test_a_missing_timestamp_blocks_rather_than_assuming_now(self):
        # A historical replay that silently used today's schedule would
        # relabel every past episode with fees nobody was charged.
        self.assertEqual(PumpFeeSchedule().quote().status, "DATA_BLOCKED")

    def _loaded(self, rows=None):
        rows = rows or [
            {"max_market_cap_lamports": 10_000_000_000, "protocol_fee_bps": 90,
             "creator_fee_bps": 10},
            {"max_market_cap_lamports": 100_000_000_000, "protocol_fee_bps": 40,
             "creator_fee_bps": 10},
            {"max_market_cap_lamports": None, "protocol_fee_bps": 15, "creator_fee_bps": 5},
        ]
        directory = tempfile.mkdtemp()
        path = Path(directory) / "tiers.json"
        path.write_text(json.dumps({VENUE_BONDING_CURVE: rows}))
        return PumpFeeSchedule.load(str(path))

    def test_a_loaded_tier_table_resolves_by_market_cap(self):
        schedule = self._loaded()
        for market_cap, expected_bps, expected_tier in [
            (1_000_000_000, 100, 0), (50_000_000_000, 50, 1), (500_000_000_000, 20, 2),
        ]:
            quote = schedule.quote(at_utc=self.AT_ACTIVATION, market_cap_lamports=market_cap)
            self.assertTrue(quote.ok, quote)
            self.assertEqual((quote.total_bps, quote.tier_index), (expected_bps, expected_tier))

    def test_tier_boundaries_are_exclusive_upper_bounds(self):
        schedule = self._loaded()
        below = schedule.quote(at_utc=self.AT_ACTIVATION, market_cap_lamports=9_999_999_999)
        at = schedule.quote(at_utc=self.AT_ACTIVATION, market_cap_lamports=10_000_000_000)
        self.assertEqual(below.tier_index, 0)
        self.assertEqual(at.tier_index, 1)

    def test_an_unsorted_table_is_sorted_on_load(self):
        """An out-of-order table would silently resolve the wrong bracket."""
        schedule = self._loaded(rows=[
            {"max_market_cap_lamports": None, "protocol_fee_bps": 15, "creator_fee_bps": 5},
            {"max_market_cap_lamports": 100_000_000_000, "protocol_fee_bps": 40,
             "creator_fee_bps": 10},
            {"max_market_cap_lamports": 10_000_000_000, "protocol_fee_bps": 90,
             "creator_fee_bps": 10},
        ])
        quote = schedule.quote(at_utc=self.AT_ACTIVATION, market_cap_lamports=1_000_000_000)
        self.assertEqual((quote.total_bps, quote.tier_index), (100, 0))

    def test_an_unobserved_market_cap_blocks_under_the_dynamic_schedule(self):
        quote = self._loaded().quote(at_utc=self.AT_ACTIVATION, market_cap_lamports=None)
        self.assertEqual(quote.status, "DATA_BLOCKED")

    def test_a_venue_with_no_table_blocks_even_when_another_venue_has_one(self):
        quote = self._loaded().quote(venue=VENUE_PUMPSWAP_CANONICAL,
                                     at_utc=self.AT_ACTIVATION,
                                     market_cap_lamports=1_000_000_000)
        self.assertEqual(quote.status, "DATA_BLOCKED")

    def test_round_trip_prices_each_leg_at_its_own_market_cap(self):
        schedule = self._loaded()
        status, total, detail = schedule.round_trip_bps(
            entry_market_cap_lamports=1_000_000_000, exit_market_cap_lamports=500_000_000_000,
            entry_utc=self.AT_ACTIVATION, exit_utc=self.AT_ACTIVATION + 300,
        )
        # Entry in the top-fee tier, exit in the cheapest: charging the exit at
        # the entry tier would overstate the cost of exactly the trades that
        # worked, and understate it for the ones that did not.
        self.assertEqual((status, total), ("OK", 120))
        self.assertEqual(detail["entry"].total_bps, 100)
        self.assertEqual(detail["exit"].total_bps, 20)

    def test_a_blocked_leg_blocks_the_whole_round_trip(self):
        status, total, _ = PumpFeeSchedule().round_trip_bps(
            entry_utc=self.JUST_BEFORE, exit_utc=self.AT_ACTIVATION,
            exit_market_cap_lamports=1_000_000_000)
        self.assertEqual((status, total), ("DATA_BLOCKED", 0))

    def test_curve_quotes_route_through_the_same_schedule(self):
        self.assertEqual(resolve_fee_bps(at_utc=self.JUST_BEFORE)[:2], ("OK", 100))
        self.assertEqual(
            resolve_fee_bps(at_utc=self.AT_ACTIVATION, market_cap_lamports=1)[0],
            "DATA_BLOCKED")
        # The curve module's default constant is the legacy fee, not a second
        # independent guess at what a trade costs.
        self.assertEqual(DEFAULT_FEE_BPS, LEGACY_TOTAL_FEE_BPS)

    def test_an_unreadable_tier_file_does_not_break_pre_activation_operation(self):
        schedule = PumpFeeSchedule.load("/nonexistent/tiers.json")
        self.assertTrue(schedule.quote(at_utc=self.JUST_BEFORE).ok)
        self.assertEqual(schedule.quote(at_utc=self.AT_ACTIVATION,
                                        market_cap_lamports=1).status, "DATA_BLOCKED")


class TestDistributionDetector(unittest.TestCase):
    """The top forming is a change in who is buying, not a change in price.

    A trailing stop fires on price decline, so it can only bank after the
    decline. On a token whose top forms in four seconds that is most of the
    giveback. These tests pin the distinction.
    """

    NOW = 1_700_000_000.0

    def _trade(self, offset, side, notional, **kwargs):
        return {"type": "trade", "timestamp": self.NOW - offset, "side": side,
                "notional_usd": notional, **kwargs}

    def _healthy(self):
        """Rising demand: bigger buys, better buyers, sells absorbed."""
        stream = []
        for index in range(12):
            stream.append(self._trade(70 - index * 4, "buy", 40.0,
                                      wallet_skill=0.4, first_time_buyer=True))
        for index in range(12):
            stream.append(self._trade(14 - index, "buy", 160.0,
                                      wallet_skill=0.8, first_time_buyer=False))
        stream.append(self._trade(30, "sell", 50.0, wallet_skill=0.3, creator_linked=False))
        stream.append(self._trade(6, "sell", 45.0, wallet_skill=0.2, creator_linked=False))
        stream.append({"type": "absorption", "timestamp": self.NOW - 5, "recovered": True})
        return stream

    def _distributing(self):
        """Same price story, opposite composition: skilled money leaving into
        a crowd of small first-time buyers, sells no longer absorbed."""
        stream = []
        for index in range(20):
            stream.append(self._trade(70 - index * 3, "buy", 200.0,
                                      wallet_skill=0.85, first_time_buyer=False))
        for index in range(30):
            stream.append(self._trade(14 - index * 0.4, "buy", 12.0,
                                      wallet_skill=0.1, first_time_buyer=True))
        for index in range(4):
            stream.append(self._trade(60 - index * 10, "sell", 30.0,
                                      wallet_skill=0.2, creator_linked=False))
        for index in range(6):
            stream.append(self._trade(12 - index * 2, "sell", 400.0,
                                      wallet_skill=0.9, creator_linked=True))
        stream.append({"type": "absorption", "timestamp": self.NOW - 9, "recovered": False})
        stream.append({"type": "absorption", "timestamp": self.NOW - 4, "recovered": False})
        return stream

    def test_healthy_flow_produces_almost_no_evidence(self):
        detector = DistributionDetector()
        reading = detector.evaluate(self._healthy(), self.NOW)
        self.assertLess(reading.evidence_score, 0.10, reading.contributions)

    def test_composition_change_produces_strong_evidence(self):
        detector = DistributionDetector()
        healthy = detector.evaluate(self._healthy(), self.NOW)
        turning = detector.evaluate(self._distributing(), self.NOW)
        self.assertGreater(turning.evidence_score, healthy.evidence_score * 3)
        drivers = dict(DistributionDetector.top_contributors(turning, limit=4))
        # The reading is driven by who is trading, not by how much.
        self.assertIn("smart_wallet_exit_rate", drivers)
        self.assertIn("creator_linked_sell_share", drivers)

    def test_price_collapse_alone_is_not_distribution(self):
        """The detector must not become a trailing stop with extra steps.

        Drawdown is the single most predictive feature of "price is about to
        be lower", so a model allowed to see it learns to wait for the decline
        -- which is exactly the lagging behaviour being replaced.
        """
        self.assertNotIn("drawdown", " ".join(DISTRIBUTION_FEATURE_NAMES))
        collapsing = self._healthy() + [
            {"type": "route", "timestamp": self.NOW - index, "price_multiple": 4.0 - index * 0.3}
            for index in range(10)
        ]
        reading = DistributionDetector().evaluate(collapsing, self.NOW)
        healthy = DistributionDetector().evaluate(self._healthy(), self.NOW)
        self.assertAlmostEqual(reading.evidence_score, healthy.evidence_score, places=12)

    def test_more_buyers_buying_less_is_the_exhaustion_pattern(self):
        """Neither half of the conjunction fires on its own."""
        base = [self._trade(70 - index * 4, "buy", 100.0) for index in range(10)]
        more_and_smaller = base + [self._trade(14 - index * 0.4, "buy", 10.0)
                                   for index in range(30)]
        # Genuinely fewer: one buy in the 15s window against ten in the prior
        # 60s is a lower arrival rate, not a higher one.
        fewer_and_smaller = base + [self._trade(7, "buy", 10.0)]
        more_and_bigger = base + [self._trade(14 - index * 0.4, "buy", 300.0)
                                  for index in range(30)]

        def signal(stream):
            features, _ = distribution_features(stream, self.NOW)
            return features["buyer_count_growth_with_shrinking_size"]

        self.assertGreater(signal(more_and_smaller), 0.2)
        self.assertEqual(signal(more_and_bigger), 0.0)
        self.assertEqual(signal(fewer_and_smaller), 0.0)

    def test_rollover_requires_prior_acceleration(self):
        """Slowing from flat is not a rollover; slowing from a climb is."""
        steady = [self._trade(70 - index * 1.0, "buy", 50.0) for index in range(70)]
        # Prior 60s: 12 buys (0.20/s). Older half of the window: 20 buys in
        # 7.5s (2.67/s) -- a real climb. Newer half: 3 buys (0.40/s) -- a real
        # roll.
        climbing = ([self._trade(70 - index * 4.5, "buy", 50.0) for index in range(12)]
                    + [self._trade(15 - index * 0.37, "buy", 50.0) for index in range(20)]
                    + [self._trade(7 - index * 2.0, "buy", 50.0) for index in range(3)])

        def signal(stream):
            features, _ = distribution_features(stream, self.NOW)
            return features["buy_acceleration_rollover"]

        self.assertEqual(signal(steady), 0.0)
        self.assertGreater(signal(climbing), 0.0)

    def test_an_untrained_detector_reports_no_probability(self):
        reading = DistributionDetector().evaluate(self._distributing(), self.NOW)
        self.assertEqual(reading.status, "DATA_BLOCKED")
        self.assertFalse(reading.calibrated)
        # Strong evidence must still not be readable as a probability: that is
        # how an unvalidated number acquires a position size.
        self.assertGreater(reading.evidence_score, 0.2)
        self.assertIsNone(reading.probability(3.0))

    def test_sparse_observations_block_rather_than_reading_as_calm(self):
        reading = DistributionDetector().evaluate(
            [self._trade(5, "buy", 10.0)], self.NOW)
        self.assertEqual(reading.status, "DATA_BLOCKED")
        self.assertIn("coverage", reading.detail)
        # "Nothing was recorded" and "nothing is happening" produce the same
        # evidence score, so coverage is what tells them apart.
        self.assertLess(reading.coverage, 0.3)

    def test_a_trained_model_yields_monotone_horizons(self):
        class FakeModel:
            def predict_proba(self, matrix):
                return np.asarray([[0.7, 0.3]])

        detector = DistributionDetector()
        self.assertTrue(detector.load_model(FakeModel(), DISTRIBUTION_FEATURE_NAMES, "v1"))
        reading = detector.evaluate(self._distributing(), self.NOW)
        self.assertEqual(reading.status, "OK")
        self.assertTrue(reading.calibrated)
        one, three, ten = (reading.probability(h) for h in DISTRIBUTION_HORIZONS)
        self.assertAlmostEqual(one, 0.3)
        self.assertLess(one, three)
        self.assertLess(three, ten)
        self.assertLessEqual(ten, 1.0)

    def test_a_model_trained_on_other_features_is_refused(self):
        class FakeModel:
            def predict_proba(self, matrix):
                return np.asarray([[0.5, 0.5]])

        detector = DistributionDetector()
        self.assertFalse(detector.load_model(FakeModel(), ("a", "b"), "v1"))
        self.assertFalse(detector.is_trained)
        # A model without the inference method it will be called through is
        # refused at load, not at the moment an exit depends on it.
        self.assertFalse(detector.load_model(object(), DISTRIBUTION_FEATURE_NAMES, "v1"))

    def test_features_are_point_in_time_only(self):
        stream = self._distributing() + [
            self._trade(-5, "sell", 5_000.0, wallet_skill=1.0, creator_linked=True),
        ]
        with_future, _ = distribution_features(stream, self.NOW)
        without_future, _ = distribution_features(self._distributing(), self.NOW)
        self.assertEqual(with_future, without_future)


class TestDistributionExitWiring(unittest.IsolatedAsyncioTestCase):
    """A calibrated distribution reading may exit early; an uncalibrated one may not."""

    def _desk(self, detector, observations):
        position = {
            "size_tokens": 1_000, "remaining_cost_usd": 100.0, "entry_time": time.time() - 60,
            "high_water_multiple": 3.0, "ratchet_stages": ["cost_recovery"],
            "prediction": {"p_5x": 0.9, "p_10x": 0.9}, "candidate": None, "risk_object": None,
        }
        exits = []
        desk = SimpleNamespace(
            elogw_engine=SimpleNamespace(open_positions={"mint": position}),
            rug_hazard=SimpleNamespace(should_exit=lambda t, p: (False, "", 0.0),
                                       observations={"mint": observations},
                                       get_hazard=lambda t: None),
            distribution_detector=detector,
            monster_machine=MonsterStateMachine(),
            last_slate_report={},
            exit_policy=ExitPolicy.default(),
            predictor=SimpleNamespace(_is_trained=False),
            global_config={},
            position=position, exits=exits,
        )
        _attach_manifest(desk)
        desk._priced_holds = 0
        desk._unpriced_cycles = 0
        desk._suppressed_monster_banks = 0
        desk._read_distribution = lambda token: MemecoinQuantDesk._read_distribution(desk, token)
        desk._latest_curve_state = getattr(desk, "_latest_curve_state", {})
        desk._latest_pool_state = getattr(desk, "_latest_pool_state", {})
        desk.ops_events = []
        desk._record_ops_event = (
            lambda stream, payload: desk.ops_events.append((stream, payload)))
        desk._closed_pnl = {}
        desk._mechanism_growth = {}
        desk._exit_capacity = (
            lambda token, pos: MemecoinQuantDesk._exit_capacity(desk, token, pos))
        desk.min_escape_probability = 0.05
        desk.state_sequencer = StateSequencer()
        desk.model_feature_hash = "test"
        desk.action_policy = ActionValuePolicy()
        desk._score_actions = (
            lambda token, pos, mult, dist:
            MemecoinQuantDesk._score_actions(desk, token, pos, mult, dist))
        desk._apply_action = (
            lambda token, pos, dec, mult:
            MemecoinQuantDesk._apply_action(desk, token, pos, dec, mult))
        desk._estimate_escape = (
            lambda token, pos, hazard: MemecoinQuantDesk._estimate_escape(
                desk, token, pos, hazard))
        desk._update_monster_state = (
            lambda token, pos, dist, mult:
            MemecoinQuantDesk._update_monster_state(desk, token, pos, dist, mult))
        desk._refresh_position_prediction = (
            lambda token, pos: MemecoinQuantDesk._refresh_position_prediction(desk, token, pos))
        desk._mark_position = lambda token, pos: _async_value((2.9, 290.0))
        desk._execute_exit = lambda token, pos, pct, reason: (
            exits.append((reason, pct)) or _async_none())
        desk._consider_scale_in = lambda token, pos, mult: _async_none()
        return desk

    @staticmethod
    def _turning_flow():
        now = time.time()
        stream = []
        for index in range(20):
            stream.append({"type": "trade", "timestamp": now - (70 - index * 3), "side": "buy",
                           "notional_usd": 200.0, "wallet_skill": 0.85, "first_time_buyer": False})
        for index in range(30):
            stream.append({"type": "trade", "timestamp": now - (14 - index * 0.4), "side": "buy",
                           "notional_usd": 12.0, "wallet_skill": 0.1, "first_time_buyer": True})
        for index in range(4):
            stream.append({"type": "trade", "timestamp": now - (60 - index * 10), "side": "sell",
                           "notional_usd": 30.0, "wallet_skill": 0.2, "creator_linked": False})
        for index in range(6):
            stream.append({"type": "trade", "timestamp": now - (12 - index * 2), "side": "sell",
                           "notional_usd": 400.0, "wallet_skill": 0.9, "creator_linked": True})
        stream.append({"type": "absorption", "timestamp": now - 4, "recovered": False})
        return stream

    class _Model:
        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, matrix):
            return np.asarray([[1 - self.probability, self.probability]])

    async def test_a_calibrated_high_reading_is_recorded_not_executed(self):
        """Banking is an opinion about expected growth, and Q owns those.

        This used to sell here, BEFORE the action-value engine was consulted,
        which meant the component that owns the objective was routinely told
        what had already been done. The reading is real evidence and Q prices
        the same forward distribution it was derived from, so acting on it
        separately counts that evidence twice.

        It is recorded rather than dropped: if these accumulate while Q keeps
        choosing HOLD, that is a disagreement worth investigating, and it is
        only investigable because it is written down.
        """
        detector = DistributionDetector()
        detector.load_model(self._Model(0.75), DISTRIBUTION_FEATURE_NAMES, "v1")
        desk = self._desk(detector, self._turning_flow())

        # At 2.9x off a 3.0x high water with continuation 0.9, the trailing
        # stop is nowhere near firing -- price has barely moved.
        self.assertIsNone(evaluate_exit(ExitPolicy.default(), 2.9, 3.0, 0.9,
                                        {"cost_recovery"}, 60.0))

        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual([exit_call[0] for exit_call in desk.exits], [])
        suppressed = desk.position.get("suppressed_monster_banks") or []
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["reason"], "distribution")
        self.assertAlmostEqual(
            suppressed[0]["fraction"],
            MonsterStateMachine.DEFAULT_BANK_FRACTIONS[MonsterState.DISTRIBUTION])
        self.assertEqual(desk._suppressed_monster_banks, 1)

    async def test_a_catastrophic_reading_still_bypasses_the_objective(self):
        """The one bypass, and it is a bypass of the objective, not a second opinion.

        A catastrophic reading says the position is about to stop being
        sellable, and a policy that prices forward returns cannot represent
        "there will be no forward".
        """
        detector = DistributionDetector()
        detector.load_model(self._Model(0.99), DISTRIBUTION_FEATURE_NAMES, "v1")
        desk = self._desk(detector, self._turning_flow())
        desk.monster_machine = MonsterStateMachine()
        # The evidence is rebuilt from live inputs each cycle, so the
        # catastrophic flag has to come from the hazard the desk actually
        # reads rather than from a pre-seeded machine state.
        desk.rug_hazard.get_hazard = lambda token: SimpleNamespace(
            urgency="critical", exit_urgency="critical", hazard_30s=0.95,
            hazard_5m=0.99, data_status="OK", blocked_reason="", signals=[])

        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual(len(desk.exits), 1)
        self.assertTrue(desk.exits[0][0].startswith("monster_"))
        self.assertAlmostEqual(desk.exits[0][1], 1.0)
        # And it did NOT go through the suppression path.
        self.assertEqual(desk._suppressed_monster_banks, 0)

    async def test_an_uncalibrated_reading_never_moves_capital(self):
        """Identical flow, identical evidence, no trained model: no exit."""
        desk = self._desk(DistributionDetector(), self._turning_flow())
        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual(desk.exits, [])
        recorded = desk.position["distribution"]
        self.assertEqual(recorded["status"], "DATA_BLOCKED")
        # The evidence is still recorded on the position, because that is what
        # the training set for this model is going to be built from.
        self.assertGreater(recorded["evidence"], 0.2)
        self.assertIn("smart_wallet_exit_rate", recorded["drivers"])

    async def test_a_calibrated_low_reading_holds_the_runner(self):
        detector = DistributionDetector()
        detector.load_model(self._Model(0.02), DISTRIBUTION_FEATURE_NAMES, "v1")
        desk = self._desk(detector, self._turning_flow())
        await MemecoinQuantDesk._manage_positions(desk)
        self.assertEqual(desk.exits, [])
        self.assertEqual(desk.position["distribution"]["status"], "OK")


async def _async_value(value):
    return value


class TestMonsterHold(unittest.TestCase):
    """Large unrealised profit is not evidence a move is over.

    Every threshold-based exit in this repository banks harder the higher a
    position goes, which is optimal for ordinary winners and catastrophic for
    the rare launch that would carry the account: it is guaranteed to sell
    that one first and hardest.
    """

    @staticmethod
    def _monster(**kwargs):
        base = dict(monster_probability=0.30, monster_probability_calibrated=True,
                    independent_buyer_acceleration=0.5, smart_wallet_net_accumulation=0.4,
                    buyer_quality_trend=0.2, sell_absorption=0.9, liquidity_expansion=0.3,
                    new_source_discovery_rate=0.2, audience_penetration=0.2)
        base.update(kwargs)
        return MonsterEvidence(**base)

    def test_an_uncalibrated_monster_probability_never_grants_an_override(self):
        machine = MonsterStateMachine()
        decision = machine.update("mint", self._monster(monster_probability=0.99,
                                                        monster_probability_calibrated=False))
        self.assertEqual(decision.state, MonsterState.NORMAL)
        self.assertFalse(machine.overrides_ordinary_exit("mint"))

    def test_a_calibrated_monster_reaches_hold_and_stands_the_ratchet_down(self):
        machine = MonsterStateMachine()
        decision = machine.update("mint", self._monster())
        self.assertEqual(decision.state, MonsterState.MONSTER_HOLD)
        self.assertEqual(decision.action, "add")
        self.assertTrue(machine.overrides_ordinary_exit("mint"))

    def test_state_tracks_evidence_not_price(self):
        """A 1.4x can be MONSTER_HOLD and an 8x can be merely PUMP_DETECTED."""
        machine = MonsterStateMachine()
        machine.update("early", self._monster())
        machine.update("stale", MonsterEvidence(monster_probability=0.01,
                                                monster_probability_calibrated=True,
                                                independent_buyer_acceleration=0.1))
        self.assertEqual(machine.state_of("early"), MonsterState.MONSTER_HOLD)
        self.assertEqual(machine.state_of("stale"), MonsterState.PUMP_DETECTED)

    def test_one_whale_selling_does_not_eject_a_monster(self):
        """The exact failure this machine exists to prevent."""
        machine = MonsterStateMachine(degrade_confirmations=3, min_degrade_dimensions=2)
        machine.update("mint", self._monster())

        transient = self._monster(monster_probability=0.02,
                                  smart_wallet_net_accumulation=-0.6,
                                  independent_buyer_acceleration=-0.2)
        decision = machine.update("mint", transient)
        self.assertEqual(decision.state, MonsterState.MONSTER_HOLD)
        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.degrade_streak, 1)

        # Evidence recovers on the next tick: the streak resets, so an earlier
        # wobble cannot be banked toward a later ejection.
        machine.update("mint", self._monster())
        self.assertEqual(machine.state_of("mint"), MonsterState.MONSTER_HOLD)
        self.assertEqual(machine.update("mint", transient).degrade_streak, 1)

    def test_persistent_broad_evidence_does_eject(self):
        machine = MonsterStateMachine(degrade_confirmations=3, min_degrade_dimensions=2)
        machine.update("mint", self._monster())
        turning = self._monster(monster_probability=0.01,
                                smart_wallet_net_accumulation=-0.6,
                                buyer_quality_trend=-0.4,
                                independent_buyer_acceleration=-0.3,
                                sell_absorption=0.1)
        states = [machine.update("mint", turning).state for _ in range(3)]
        self.assertEqual(states[:2], [MonsterState.MONSTER_HOLD, MonsterState.MONSTER_HOLD])
        self.assertNotIn(states[2], MONSTER_STATES)
        self.assertFalse(machine.overrides_ordinary_exit("mint"))

    def test_a_single_dimension_never_accumulates_a_streak(self):
        machine = MonsterStateMachine(degrade_confirmations=3, min_degrade_dimensions=2)
        machine.update("mint", self._monster())
        one_signal = self._monster(monster_probability=0.01,
                                   smart_wallet_net_accumulation=-0.6)
        for _ in range(10):
            decision = machine.update("mint", one_signal)
        self.assertEqual(decision.state, MonsterState.MONSTER_HOLD)
        self.assertEqual(decision.degrade_streak, 0)

    def test_catastrophic_hazard_bypasses_hysteresis_entirely(self):
        """Patience about a rug is not patience."""
        machine = MonsterStateMachine(degrade_confirmations=99)
        machine.update("mint", self._monster())
        decision = machine.update("mint", self._monster(catastrophic_hazard=True))
        self.assertEqual(decision.state, MonsterState.DISTRIBUTION)
        self.assertEqual(decision.action, "emergency_exit")
        self.assertEqual(decision.bank_fraction, 1.0)

    def test_calibrated_distribution_forces_the_distribution_state(self):
        machine = MonsterStateMachine()
        machine.update("mint", self._monster())
        decision = machine.update("mint", self._monster(distribution_probability=0.8,
                                                        distribution_calibrated=True))
        self.assertEqual(decision.state, MonsterState.DISTRIBUTION)
        self.assertEqual(decision.action, "bank")
        self.assertGreater(decision.bank_fraction, 0.5)

    def test_uncalibrated_distribution_does_not_force_it(self):
        machine = MonsterStateMachine()
        machine.update("mint", self._monster())
        decision = machine.update("mint", self._monster(distribution_probability=0.99,
                                                        distribution_calibrated=False))
        self.assertEqual(decision.state, MonsterState.MONSTER_HOLD)

    def test_staged_banking_fires_once_per_state(self):
        machine = MonsterStateMachine()
        machine.update("mint", self._monster())
        fomo = self._monster(new_source_discovery_rate=0.9, audience_penetration=0.6)
        first = machine.update("mint", fomo)
        second = machine.update("mint", fomo)
        self.assertEqual((first.action, first.state), ("bank", MonsterState.MASS_FOMO))
        self.assertAlmostEqual(first.bank_fraction, 0.15)
        # A staged bank is a one-off, not a per-tick tax on the runner.
        self.assertEqual(second.action, "hold")

    def test_saturation_banks_harder_than_mass_fomo(self):
        machine = MonsterStateMachine()
        self.assertLess(machine.bank_fractions[MonsterState.MASS_FOMO],
                        machine.bank_fractions[MonsterState.SATURATION])
        self.assertLess(machine.bank_fractions[MonsterState.SATURATION],
                        machine.bank_fractions[MonsterState.DISTRIBUTION])


class TestHoldVersusExitValue(unittest.TestCase):
    """The exit comparison must not contain the price level."""

    def _value(self, **kwargs):
        base = dict(remaining_upside_multiple=3.0, distribution_probability=0.1,
                    rug_probability=0.02, exit_capacity_ratio=1.0,
                    alternative_growth_per_second=1e-5, expected_remaining_seconds=120.0)
        base.update(kwargs)
        return hold_versus_exit(**base)

    def test_large_remaining_upside_holds(self):
        value = self._value()
        self.assertEqual(value.status, "OK")
        self.assertFalse(value.should_exit)
        self.assertGreater(value.v_hold, value.v_exit)

    def test_exhausted_upside_exits_even_at_a_huge_multiple(self):
        # The position is up 50x; that number appears nowhere. What decides it
        # is that there is no upside left and the capital has somewhere better.
        value = self._value(remaining_upside_multiple=1.01,
                            alternative_growth_per_second=1e-3)
        self.assertTrue(value.should_exit)

    def test_a_better_alternative_can_take_the_capital(self):
        patient = self._value(alternative_growth_per_second=1e-6)
        contested = self._value(alternative_growth_per_second=1e-2)
        self.assertFalse(patient.should_exit)
        self.assertTrue(contested.should_exit)
        # Only the redeploy term moved.
        self.assertAlmostEqual(patient.v_hold, contested.v_hold)

    def test_upside_that_cannot_be_sold_is_not_upside(self):
        liquid = self._value(exit_capacity_ratio=1.0)
        illiquid = self._value(exit_capacity_ratio=0.05)
        self.assertGreater(liquid.v_hold, illiquid.v_hold)

    def test_rug_risk_dominates_the_hold_branch(self):
        safe = self._value(rug_probability=0.01)
        dangerous = self._value(rug_probability=0.5)
        self.assertGreater(safe.v_hold, dangerous.v_hold)
        self.assertTrue(dangerous.should_exit)

    def test_missing_inputs_block(self):
        self.assertEqual(self._value(remaining_upside_multiple=0).status, "DATA_BLOCKED")
        self.assertEqual(self._value(exit_capacity_ratio=0).status, "DATA_BLOCKED")
        self.assertEqual(self._value(expected_remaining_seconds=0).status, "DATA_BLOCKED")
        self.assertEqual(self._value(rug_probability=1.5).status, "DATA_BLOCKED")


class TestTailCaptureMetrics(unittest.TestCase):
    """The number that catches an exit policy quietly killing monsters."""

    def test_tail_capture_ratio_credits_only_feasible_return(self):
        self.assertAlmostEqual(tail_capture_ratio(5.0, 20.0), 0.25)
        self.assertAlmostEqual(tail_capture_ratio(20.0, 20.0), 1.0)
        self.assertIsNone(tail_capture_ratio(5.0, 0.0))

    def test_premature_exit_rates_expose_a_win_rate_improvement_that_loses_wealth(self):
        # Every trade a winner; every 20x sold at 2.5x.
        greedy_ratchet = [{"realized_multiple": 2.5, "max_feasible_multiple": 25.0}
                          for _ in range(10)]
        report = premature_exit_rates(greedy_ratchet)
        self.assertEqual(report["exited_20x_below_10x"], 1.0)
        self.assertAlmostEqual(report["tail_capture_ratio"], 0.1)

        patient = [{"realized_multiple": 18.0, "max_feasible_multiple": 25.0}
                   for _ in range(10)]
        self.assertEqual(premature_exit_rates(patient)["exited_20x_below_10x"], 0.0)

    def test_a_threshold_with_no_eligible_trades_reports_none_not_zero(self):
        report = premature_exit_rates([{"realized_multiple": 1.5,
                                        "max_feasible_multiple": 2.0}])
        # Zero would read as "we never sold a 50x too early", which is a claim
        # about a population that contains no 50x opportunities.
        self.assertIsNone(report["exited_50x_below_20x"])


class TestMonsterHoldWiring(unittest.IsolatedAsyncioTestCase):
    """The ratchet must stand down inside a monster state, and only there."""

    def _desk(self, machine, hazard=None):
        position = {
            "size_tokens": 1_000, "remaining_cost_usd": 100.0, "entry_time": time.time() - 60,
            "high_water_multiple": 6.0, "ratchet_stages": [],
            "prediction": {"p_5x": 0.9, "p_10x": 0.8}, "candidate": None, "risk_object": None,
        }
        exits = []
        desk = SimpleNamespace(
            elogw_engine=SimpleNamespace(open_positions={"mint": position}),
            rug_hazard=SimpleNamespace(should_exit=lambda t, p: (False, "", 0.0),
                                       observations={}, get_hazard=lambda t: hazard),
            distribution_detector=DistributionDetector(),
            monster_machine=machine, last_slate_report={}, exit_policy=ExitPolicy.default(),
            predictor=SimpleNamespace(_is_trained=False),
            global_config={}, position=position, exits=exits,
        )
        _attach_manifest(desk)
        desk._priced_holds = 0
        desk._unpriced_cycles = 0
        desk._suppressed_monster_banks = 0
        desk._read_distribution = lambda token: MemecoinQuantDesk._read_distribution(desk, token)
        desk._latest_curve_state = getattr(desk, "_latest_curve_state", {})
        desk._latest_pool_state = getattr(desk, "_latest_pool_state", {})
        desk.ops_events = []
        desk._record_ops_event = (
            lambda stream, payload: desk.ops_events.append((stream, payload)))
        desk._closed_pnl = {}
        desk._mechanism_growth = {}
        desk._exit_capacity = (
            lambda token, pos: MemecoinQuantDesk._exit_capacity(desk, token, pos))
        desk.min_escape_probability = 0.05
        desk.state_sequencer = StateSequencer()
        desk.model_feature_hash = "test"
        desk.action_policy = ActionValuePolicy()
        desk._score_actions = (
            lambda token, pos, mult, dist:
            MemecoinQuantDesk._score_actions(desk, token, pos, mult, dist))
        desk._apply_action = (
            lambda token, pos, dec, mult:
            MemecoinQuantDesk._apply_action(desk, token, pos, dec, mult))
        desk._estimate_escape = (
            lambda token, pos, hazard: MemecoinQuantDesk._estimate_escape(
                desk, token, pos, hazard))
        desk._update_monster_state = (
            lambda token, pos, dist, mult:
            MemecoinQuantDesk._update_monster_state(desk, token, pos, dist, mult))
        desk._refresh_position_prediction = (
            lambda token, pos: MemecoinQuantDesk._refresh_position_prediction(desk, token, pos))
        # 6x high water, currently 5.5x: the cost-recovery ratchet is due.
        desk._mark_position = lambda token, pos: _async_value((5.5, 550.0))
        desk._execute_exit = lambda token, pos, pct, reason: (
            exits.append((reason, pct)) or _async_none())
        desk._consider_scale_in = lambda token, pos, mult: _async_none()
        return desk

    async def test_without_a_calibrated_model_the_ratchet_still_fires(self):
        """Nothing is suppressed by default; the override has to be earned."""
        desk = self._desk(MonsterStateMachine())
        await MemecoinQuantDesk._manage_positions(desk)
        self.assertEqual(len(desk.exits), 1)
        self.assertIn("profit_ratchet", desk.exits[0][0])
        self.assertEqual(desk.position["monster_state"], "normal")

    async def test_inside_a_monster_state_the_ratchet_is_suppressed(self):
        machine = MonsterStateMachine()
        # Stand the machine in MONSTER_HOLD the only way it can be reached:
        # from a calibrated monster probability.
        machine.update("mint", MonsterEvidence(
            monster_probability=0.4, monster_probability_calibrated=True,
            independent_buyer_acceleration=0.4, smart_wallet_net_accumulation=0.3))
        self.assertTrue(machine.overrides_ordinary_exit("mint"))

        desk = self._desk(machine)
        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual(desk.exits, [])
        suppressed = desk.position["suppressed_exits"]
        self.assertEqual(len(suppressed), 1)
        self.assertIn("profit_ratchet", suppressed[0]["reason"])
        self.assertAlmostEqual(suppressed[0]["multiple"], 5.5)

    async def test_a_critical_hazard_exits_a_monster_immediately(self):
        """Patience about a rug is not patience."""
        machine = MonsterStateMachine(degrade_confirmations=99)
        machine.update("mint", MonsterEvidence(
            monster_probability=0.4, monster_probability_calibrated=True,
            independent_buyer_acceleration=0.4, smart_wallet_net_accumulation=0.3))
        hazard = SimpleNamespace(urgency="critical", hazard_5m=0.9, hazard_30s=0.8,
                                 exit_urgency="critical", data_status="OK",
                                 blocked_reason="")

        desk = self._desk(machine, hazard=hazard)
        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual(len(desk.exits), 1)
        reason, fraction = desk.exits[0]
        self.assertEqual(reason, "monster_catastrophic_hazard")
        self.assertEqual(fraction, 1.0)

    async def test_the_separate_hazard_exit_is_never_suppressed(self):
        machine = MonsterStateMachine()
        machine.update("mint", MonsterEvidence(
            monster_probability=0.4, monster_probability_calibrated=True,
            independent_buyer_acceleration=0.4, smart_wallet_net_accumulation=0.3))
        desk = self._desk(machine)
        desk.rug_hazard.should_exit = lambda t, p: (True, "high", 0.75)

        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual(len(desk.exits), 1)
        self.assertEqual(desk.exits[0], ("rug_hazard_high", 0.75))


class TestAuthenticityResolver(unittest.TestCase):
    """A ticker-matching bot is the reliable buyer of every copycat.

    When a globally-followed account says "coin", dozens of impostor mints
    exist within seconds, most named exactly what a name-matching bot looks
    for. These tests pin the parsing rules where the obvious implementation is
    exploitable.
    """

    REAL = "So11111111111111111111111111111111111111112"
    FAKE = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    def _registry(self):
        return EntityRegistry([
            WatchedEntity(
                entity_id="figure", display_name="Some Figure",
                # Platform-agnostic on purpose: these are stable account ids,
                # not display names, and the resolver never assumes which
                # network a signal arrived on.
                accounts={"telegram": {"1001"}, "bluesky": {"did:plc:abc"},
                          "youtube": {"UC_official"}},
                official_domains={"figure.com"},
                known_wallets={"WaLLeT1111111111111111111111111111111111111"},
                aliases={"The Figure"},
                # Provenance, as production requires: the loader refuses an
                # entity without it, so a fixture without it is testing a
                # shape the desk can never actually hold.
                verified_from="https://figure.com/press", verified_at=time.time(),
            ),
            WatchedEntity(entity_id="newsdesk", display_name="News Desk",
                          accounts={"rss": {"newsdesk.example/feed"}},
                          official_domains={"newsdesk.example"},
                          verified_from="https://newsdesk.example/about",
                          verified_at=time.time()),
        ])

    def _resolver(self, **kwargs):
        return AuthenticityResolver(self._registry(), **kwargs)

    # --- mint validation -------------------------------------------------

    def test_a_mint_must_decode_to_thirty_two_bytes(self):
        self.assertTrue(looks_like_mint(self.REAL))
        # A long base58-looking word is not an address. Loose matching turns
        # any 40-character token in a post into a purchase candidate.
        self.assertFalse(looks_like_mint("A" * 40))
        self.assertFalse(looks_like_mint("z" * 44))
        self.assertFalse(looks_like_mint("short"))
        self.assertFalse(looks_like_mint(self.REAL + "1"))
        # 0, O, I and l are not in the base58 alphabet.
        self.assertFalse(looks_like_mint("0" + self.REAL[1:]))

    def test_mints_are_extracted_from_text_and_from_urls(self):
        text = f"launch {self.REAL} see https://pump.fun/coin/{self.FAKE}"
        self.assertEqual(extract_mints(text), [self.REAL, self.FAKE])
        self.assertEqual(extract_mints("no addresses here at all"), [])

    # --- domain matching -------------------------------------------------

    def test_domain_matching_is_never_a_substring(self):
        self.assertTrue(host_matches("figure.com", "figure.com"))
        self.assertTrue(host_matches("www.figure.com", "figure.com"))
        self.assertTrue(host_matches("https://token.figure.com/x?y=1", "figure.com"))
        # Both of these are registrable by an attacker for the price of a
        # domain, so substring matching here is equivalent to no matching.
        self.assertFalse(host_matches("figure.com.attacker.io", "figure.com"))
        self.assertFalse(host_matches("notfigure.com", "figure.com"))
        self.assertFalse(host_matches("figure.com.co", "figure.com"))
        self.assertFalse(host_matches("", "figure.com"))

    def test_userinfo_and_ports_do_not_smuggle_a_host(self):
        self.assertFalse(host_matches("https://figure.com@attacker.io/x", "figure.com"))
        self.assertTrue(host_matches("https://figure.com:8443/x", "figure.com"))

    # --- level A: direct mint --------------------------------------------

    def test_a_canonical_account_publishing_the_mint_is_the_strongest_proof(self):
        verdict = self._resolver().resolve_signal(SourceSignal(
            "telegram", "1001", f"our official token: {self.REAL}", 1.0))
        self.assertEqual(verdict.level, ProofLevel.DIRECT_MINT)
        self.assertEqual(verdict.mint, self.REAL)
        self.assertTrue(verdict.tradeable)

    def test_an_impostor_account_saying_the_name_proves_nothing(self):
        verdict = self._resolver().resolve_signal(SourceSignal(
            "telegram", "9999", f"Some Figure official coin {self.REAL}", 1.0))
        self.assertEqual(verdict.level, ProofLevel.NAME_ONLY)
        self.assertIsNone(verdict.mint)
        self.assertFalse(verdict.tradeable)

    def test_two_mints_in_one_official_post_is_refused_not_guessed(self):
        """Ambiguity is the state an impostor wants to create."""
        verdict = self._resolver().resolve_signal(SourceSignal(
            "telegram", "1001", f"{self.REAL} or maybe {self.FAKE}", 1.0))
        self.assertIsNone(verdict.mint)
        self.assertEqual(verdict.level, ProofLevel.NAME_ONLY)
        self.assertEqual(verdict.rejected[0][0], "multiple_mints_in_one_post")

    # --- level B: official domain ----------------------------------------

    def test_an_official_domain_link_resolves_the_mint(self):
        verdict = self._resolver().resolve_signal(
            SourceSignal("bluesky", "did:plc:abc", "details at https://www.figure.com/token", 1.0),
            domain_published_mints={"figure.com": self.REAL})
        self.assertEqual(verdict.level, ProofLevel.OFFICIAL_DOMAIN)
        self.assertEqual(verdict.mint, self.REAL)

    def test_a_lookalike_domain_is_not_an_official_domain(self):
        verdict = self._resolver().resolve_signal(
            SourceSignal("bluesky", "did:plc:abc", "https://figure.com.attacker.io/token", 1.0),
            domain_published_mints={"figure.com.attacker.io": self.FAKE})
        self.assertIsNone(verdict.mint)
        self.assertEqual(verdict.level, ProofLevel.NAME_ONLY)

    # --- level C: creator wallet -----------------------------------------

    def test_a_known_wallet_creating_the_token_is_chain_side_proof(self):
        verdict = self._resolver().resolve_creator(
            self.REAL, "WaLLeT1111111111111111111111111111111111111")
        self.assertEqual(verdict.level, ProofLevel.CREATOR_WALLET)
        self.assertEqual(verdict.entity_id, "figure")

    def test_a_known_wallet_funding_the_creator_also_counts(self):
        verdict = self._resolver().resolve_creator(
            self.REAL, "UnknownCreator11111111111111111111111111111",
            funders=["WaLLeT1111111111111111111111111111111111111"])
        self.assertEqual(verdict.level, ProofLevel.CREATOR_WALLET)

    def test_an_unknown_creator_proves_nothing(self):
        verdict = self._resolver().resolve_creator(
            self.REAL, "UnknownCreator11111111111111111111111111111")
        self.assertEqual(verdict.level, ProofLevel.NONE)
        self.assertFalse(verdict.tradeable)

    # --- level D: cross-source -------------------------------------------

    def test_two_independent_entities_agreeing_promotes_to_cross_source(self):
        resolver = self._resolver()
        combined = resolver.combine([
            AuthenticityVerdict(self.REAL, ProofLevel.NAME_ONLY, "figure",
                                supporting_sources=["telegram:1001"]),
            AuthenticityVerdict(self.REAL, ProofLevel.NAME_ONLY, "newsdesk",
                                supporting_sources=["rss:newsdesk.example/feed"]),
        ])
        self.assertEqual(combined.level, ProofLevel.CROSS_SOURCE)
        self.assertEqual(combined.mint, self.REAL)
        self.assertTrue(combined.tradeable)

    def test_one_account_posting_three_times_is_one_source(self):
        """Otherwise a single compromised account manufactures its own quorum."""
        resolver = self._resolver()
        combined = resolver.combine([
            AuthenticityVerdict(self.FAKE, ProofLevel.NAME_ONLY, "figure",
                                supporting_sources=["telegram:1001"])
            for _ in range(3)
        ])
        self.assertIsNone(combined.mint)
        self.assertFalse(combined.tradeable)

    def test_disagreement_lowers_confidence_rather_than_taking_a_majority(self):
        resolver = self._resolver()
        combined = resolver.combine([
            AuthenticityVerdict(self.REAL, ProofLevel.NAME_ONLY, "a"),
            AuthenticityVerdict(self.REAL, ProofLevel.NAME_ONLY, "b"),
            AuthenticityVerdict(self.FAKE, ProofLevel.NAME_ONLY, "c"),
            AuthenticityVerdict(self.FAKE, ProofLevel.NAME_ONLY, "d"),
        ])
        # Sources disagreeing means an impostor is in the set. Resolving that
        # by majority buys whichever mint the impostors flooded hardest.
        self.assertIsNone(combined.mint)
        self.assertEqual(combined.rejected[0][0], "conflicting_mints")

    def test_a_strong_single_proof_outranks_a_weak_quorum(self):
        resolver = self._resolver()
        combined = resolver.combine([
            AuthenticityVerdict(self.REAL, ProofLevel.DIRECT_MINT, "figure"),
            AuthenticityVerdict(self.FAKE, ProofLevel.NAME_ONLY, "b"),
            AuthenticityVerdict(self.FAKE, ProofLevel.NAME_ONLY, "c"),
        ])
        self.assertEqual((combined.mint, combined.level), (self.REAL, ProofLevel.DIRECT_MINT))

    # --- level E and the trading gate ------------------------------------

    def test_name_only_is_never_tradeable(self):
        for level in (ProofLevel.NONE, ProofLevel.NAME_ONLY):
            self.assertFalse(
                AuthenticityVerdict(self.REAL, level, "figure").tradeable,
                f"{level} must not authorise a trade")
        for level in (ProofLevel.CROSS_SOURCE, ProofLevel.CREATOR_WALLET,
                      ProofLevel.OFFICIAL_DOMAIN, ProofLevel.DIRECT_MINT):
            self.assertTrue(AuthenticityVerdict(self.REAL, level, "figure").tradeable)

    def test_a_verdict_with_no_mint_is_never_tradeable(self):
        self.assertFalse(
            AuthenticityVerdict(None, ProofLevel.DIRECT_MINT, "figure").tradeable)

    def test_alias_matching_is_whole_word_only(self):
        registry = self._registry()
        self.assertTrue(registry.match_name("what did Some Figure say"))
        # Substring matching would resolve every word containing the name.
        self.assertFalse(registry.match_name("configurehandsome"))

    # --- copycat swarm ---------------------------------------------------

    def test_copycats_rank_by_independent_capital_not_by_ticker(self):
        ranked = rank_copycats([
            {"mint": "first", "independent_buyers": 2, "independent_capital_usd": 50.0},
            {"mint": "later", "independent_buyers": 400, "independent_capital_usd": 9_000.0},
            {"mint": "unmeasured"},
        ])
        # The token worth anything is whichever is accumulating capital from
        # wallets that are not the creator's -- frequently neither the first
        # nor the best-named.
        self.assertEqual([item["mint"] for item in ranked], ["later", "first"])
        self.assertNotIn("unmeasured", [item["mint"] for item in ranked])


class TestTradeabilityFrontier(unittest.TestCase):
    """"Liquidity is $X" is not a tradeable fact.

    It does not say how much can be bought without moving the price, it does
    not say how much can be sold, and on a bonding curve those are not close
    to each other. A position marked at 20x that can only exit 4% of itself
    inside an acceptable impact is a 20x on 4% of the size.
    """

    @staticmethod
    def _linear_quote(slope):
        """Impact rises linearly with size -- monotone, as the search requires."""
        return lambda size: (True, size * slope)

    def test_the_frontier_finds_the_largest_size_inside_each_bound(self):
        frontier = build_frontier(self._linear_quote(1e-6), 1_000_000, "exit")
        self.assertTrue(frontier.ok)
        self.assertEqual(frontier.size_at(0.01), 10_000)
        self.assertEqual(frontier.size_at(0.05), 50_000)
        self.assertEqual(frontier.size_at(0.10), 100_000)

    def test_capacity_is_monotone_in_the_bound(self):
        frontier = build_frontier(self._linear_quote(3e-7), 10_000_000, "exit")
        sizes = [frontier.size_at(bound) for bound in sorted(frontier.sizes)]
        self.assertEqual(sizes, sorted(sizes))

    def test_a_venue_that_will_not_quote_blocks_rather_than_reading_as_deep(self):
        frontier = build_frontier(lambda size: (False, 0.0), 1_000_000, "exit")
        self.assertFalse(frontier.ok)
        self.assertEqual(frontier.status, "DATA_BLOCKED")
        # A refused quote is not zero impact, which is what an unguarded
        # implementation would conclude from (False, 0.0).
        self.assertEqual(frontier.sizes, {})

    def test_zero_capacity_everywhere_is_a_measurement_not_a_block(self):
        frontier = build_frontier(lambda size: (True, 5.0), 1_000_000, "exit")
        self.assertTrue(frontier.ok)
        self.assertEqual(frontier.size_at(0.10), 0)

    def test_exit_capacity_never_defaults_to_fully_sellable(self):
        blocked = Frontier(status="DATA_BLOCKED", side="exit")
        self.assertEqual(exit_capacity_ratio(1_000, blocked), ("DATA_BLOCKED", 0.0))
        # Assuming a position is fully sellable until proven otherwise is how
        # a theoretical return becomes a real loss.
        self.assertNotEqual(exit_capacity_ratio(1_000, blocked)[1], 1.0)

    def test_exit_capacity_is_the_sellable_share_of_the_position(self):
        frontier = build_frontier(self._linear_quote(1e-6), 10_000_000, "exit")
        status, ratio = exit_capacity_ratio(1_000_000, frontier, acceptable_impact=0.10)
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(ratio, 0.1)
        # A position smaller than capacity is fully sellable, never more.
        self.assertEqual(exit_capacity_ratio(10_000, frontier, 0.10)[1], 1.0)

    def test_an_unmeasured_bound_blocks_rather_than_using_a_neighbour(self):
        frontier = build_frontier(self._linear_quote(1e-6), 1_000_000, "exit",
                                  bounds=(0.01, 0.05))
        self.assertEqual(exit_capacity_ratio(1_000, frontier, 0.10)[0], "DATA_BLOCKED")

    def test_virtual_only_capacity_is_labelled_an_upper_bound(self):
        """Real reserves cap what the curve can physically pay out."""
        virtual_only = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000, virtual_sol_reserves=30_000_000_000,
            real_token_reserves=0, real_sol_reserves=0, token_total_supply=0,
            complete=False, creator="")
        report = curve_tradeability(virtual_only, quote_buy, quote_sell)
        self.assertTrue(report.exit.upper_bound_only)
        status, ratio = exit_capacity_ratio(1_000_000, report.exit)
        self.assertEqual(status, "OK_UPPER_BOUND")
        self.assertGreater(ratio, 0)

        measured = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000, virtual_sol_reserves=30_000_000_000,
            real_token_reserves=800_000_000_000, real_sol_reserves=5_000_000_000,
            token_total_supply=1_000_000_000_000, complete=False, creator="")
        self.assertFalse(curve_tradeability(measured, quote_buy, quote_sell).exit.upper_bound_only)

    def test_a_completed_curve_has_no_frontier(self):
        done = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000, virtual_sol_reserves=30_000_000_000,
            real_token_reserves=0, real_sol_reserves=0, token_total_supply=0,
            complete=True, creator="")
        report = curve_tradeability(done, quote_buy, quote_sell)
        self.assertFalse(report.ok)

    def test_round_trip_size_is_bound_by_the_exit_side(self):
        report = TradeabilityReport(
            entry=build_frontier(self._linear_quote(1e-7), 10_000_000, "entry"),
            exit=build_frontier(self._linear_quote(1e-5), 10_000_000, "exit"),
        )
        # Sizing to the entry frontier is how a position gets opened that
        # cannot be closed.
        self.assertEqual(report.round_trip_size(0.05), report.exit.size_at(0.05))
        self.assertLess(report.asymmetry(0.05), 1.0)

    def test_impact_for_reports_the_tightest_bound_a_size_fits(self):
        frontier = build_frontier(self._linear_quote(1e-6), 10_000_000, "exit")
        self.assertEqual(frontier.impact_for(5_000), 0.01)
        self.assertEqual(frontier.impact_for(40_000), 0.05)
        # None means "worse than every bound measured", not "no impact".
        self.assertIsNone(frontier.impact_for(9_000_000))


class TestCurveStateFromTradeStream(unittest.TestCase):
    """The reserves are already in the TradeEvent; they were being discarded."""

    @staticmethod
    def _decode(payload):
        monitor = PumpFunMonitor(SimpleNamespace(on=lambda *a, **k: None),
                                 callback=lambda event: None)
        return monitor._decode_program_event(payload, "sig", 1)

    def test_the_pump_trade_event_now_carries_its_reserves(self):
        payload = (
            PumpFunMonitor.TRADE_EVENT
            + bytes(32)                                    # mint
            + struct.pack("<QQ", 1_000_000_000, 500_000)   # sol, token amounts
            + bytes([1])                                   # is_buy
            + bytes(32)                                    # user
            + struct.pack("<q", 1_700_000_000)             # timestamp
            + struct.pack("<QQ", 31_000_000_000, 1_070_000_000_000)
        )
        decoded = self._decode(payload)
        self.assertEqual(decoded["type"], "token_trade")
        self.assertEqual(decoded["virtual_sol_reserves"], 31_000_000_000)
        self.assertEqual(decoded["virtual_token_reserves"], 1_070_000_000_000)
        # The price it was already deriving stays consistent with them.
        self.assertAlmostEqual(decoded["curve_price_raw"],
                               31_000_000_000 / 1_070_000_000_000)

    def test_a_truncated_event_reports_no_reserves_rather_than_zero(self):
        payload = (
            PumpFunMonitor.TRADE_EVENT + bytes(32)
            + struct.pack("<QQ", 1_000_000_000, 500_000) + bytes([1]) + bytes(32)
            + struct.pack("<q", 1_700_000_000)
        )
        decoded = self._decode(payload)
        self.assertIsNone(decoded["virtual_sol_reserves"])
        self.assertIsNone(decoded["virtual_token_reserves"])


class TestAdaptiveGivebackAndHarvest(unittest.TestCase):
    """A fixed giveback percentage is wrong in both directions.

    On a day where the remaining opportunity set is exceptional, refusing
    normal volatility halts the book in front of the trades worth taking. On a
    day where edge has collapsed, the same number is far more than should be
    risked to find out that it has.
    """

    def _engine(self, **kwargs):
        base = dict(max_daily_loss_usd=1_000.0, daily_giveback_pct=0.35,
                    daily_giveback_arm_pct=0.5)
        base.update(kwargs)
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), **base)
        engine.portfolio_value = 10_000.0
        engine._day_start_equity = 10_000.0
        return engine

    def test_an_unmeasured_opportunity_set_falls_back_to_the_configured_base(self):
        engine = self._engine()
        self.assertAlmostEqual(engine.giveback_allowance(), 0.35)
        engine.observe_opportunity_set(None)
        self.assertAlmostEqual(engine.giveback_allowance(), 0.35)

    def test_an_exceptional_slate_tolerates_more_giveback(self):
        engine = self._engine()
        engine.observe_opportunity_set(quality=2.0, uncertainty=0.0)
        self.assertGreater(engine.giveback_allowance(), 0.35)

    def test_a_collapsed_slate_locks_harder(self):
        engine = self._engine()
        engine.observe_opportunity_set(quality=0.0, uncertainty=0.0)
        self.assertLess(engine.giveback_allowance(), 0.35)

    def test_uncertainty_tightens_an_otherwise_attractive_slate(self):
        """A wide estimate is not the same as a genuinely good one."""
        engine = self._engine()
        engine.observe_opportunity_set(quality=2.0, uncertainty=0.0)
        confident = engine.giveback_allowance()
        engine.observe_opportunity_set(quality=2.0, uncertainty=0.9)
        self.assertLess(engine.giveback_allowance(), confident)

    def test_the_allowance_stays_inside_a_band(self):
        engine = self._engine()
        for quality, uncertainty in [(0.0, 1.0), (99.0, 0.0), (-5.0, 0.0), (2.0, -1.0)]:
            engine.observe_opportunity_set(quality=quality, uncertainty=uncertainty)
            allowance = engine.giveback_allowance()
            # A rule that can reach zero is a hair trigger; one that can reach
            # 1.0 is not a guard at all.
            self.assertGreaterEqual(allowance, 0.35 * ElogwEngine.GIVEBACK_FLOOR_SCALE)
            self.assertLessEqual(allowance, ElogwEngine.MAX_GIVEBACK_PCT)

    def test_the_floor_still_rises_with_the_peak(self):
        engine = self._engine()
        engine.update_pnl(2_000.0)
        floor = engine.giveback_floor()
        self.assertIsNotNone(floor)
        self.assertAlmostEqual(floor, 2_000.0 * (1 - 0.35))
        self.assertFalse(engine.kill_switch_active)

    def test_the_guard_still_fires_when_the_floor_is_breached(self):
        engine = self._engine()
        engine.update_pnl(2_000.0)
        engine.update_pnl(-800.0)
        self.assertTrue(engine.kill_switch_active)

    def test_a_better_slate_can_keep_the_book_open_through_the_same_dip(self):
        """The whole point: the same drawdown, two different verdicts."""
        strict = self._engine()
        strict.update_pnl(2_000.0)
        strict.update_pnl(-750.0)

        lenient = self._engine()
        lenient.observe_opportunity_set(quality=2.0, uncertainty=0.0)
        lenient.update_pnl(2_000.0)
        lenient.update_pnl(-750.0)

        self.assertTrue(strict.kill_switch_active)
        self.assertFalse(lenient.kill_switch_active)

    def test_the_loss_kill_switch_is_never_softened_by_a_good_slate(self):
        engine = self._engine()
        engine.observe_opportunity_set(quality=2.0, uncertainty=0.0)
        engine.update_pnl(-1_100.0)
        # The daily-loss limit is not negotiable against opportunity quality.
        self.assertTrue(engine.kill_switch_active)
        self.assertAlmostEqual(engine.daily_loss_limit(), 1_000.0)


class TestHarvestHurdle(unittest.TestCase):
    """After an exceptional run, ordinary risk gets more expensive -- not banned."""

    def _engine(self):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), min_edge_bps=50,
                             max_daily_loss_usd=1_000.0, harvest_trigger_ratio=1.5,
                             harvest_slope=0.5)
        engine.portfolio_value = 10_000.0
        engine._day_start_equity = 10_000.0
        return engine

    def test_an_ordinary_day_uses_the_ordinary_hurdle(self):
        engine = self._engine()
        self.assertEqual(engine.harvest_hurdle_bps(), 50)
        engine.update_pnl(500.0)
        self.assertEqual(engine.harvest_hurdle_bps(), 50)

    def test_an_exceptional_day_raises_the_hurdle(self):
        engine = self._engine()
        engine.update_pnl(3_000.0)
        self.assertGreater(engine.harvest_hurdle_bps(), 50)

    def test_the_hurdle_rises_but_never_becomes_a_cap(self):
        engine = self._engine()
        engine.update_pnl(1_000_000.0)
        hurdle = engine.harvest_hurdle_bps()
        # The book never stops trading; an extraordinary opportunity still
        # clears an extraordinary hurdle.
        self.assertTrue(math.isfinite(hurdle))
        self.assertLessEqual(hurdle, 50 * ElogwEngine.MAX_HARVEST_MULTIPLE)
        self.assertFalse(engine.kill_switch_active)

    def test_a_losing_day_never_raises_the_hurdle(self):
        engine = self._engine()
        engine.update_pnl(-400.0)
        self.assertEqual(engine.harvest_hurdle_bps(), 50)


class TestSmallAccountMode(unittest.TestCase):
    """A small account's structural advantage is that its size is negligible."""

    def _engine(self, equity, **kwargs):
        base = dict(max_position_pct=0.05, small_account_mode=True,
                    small_account_negligible_share=0.002)
        base.update(kwargs)
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), **base)
        engine.portfolio_value = equity
        return engine

    def test_a_tiny_account_may_concentrate_harder(self):
        engine = self._engine(500.0)
        widened = engine.small_account_concentration(liquidity_usd=5_000_000.0)
        self.assertGreater(widened, 0.05)
        self.assertLessEqual(widened, ElogwEngine.MAX_SMALL_ACCOUNT_POSITION_PCT)

    def test_the_ceiling_tightens_automatically_as_equity_grows(self):
        thin_pool = 100_000.0
        small = self._engine(500.0).small_account_concentration(thin_pool)
        large = self._engine(5_000_000.0).small_account_concentration(thin_pool)
        # The same trade stops being negligible, so the widening withdraws
        # itself rather than needing to be switched off.
        self.assertGreater(small, large)
        self.assertEqual(large, 0.05)

    def test_the_mode_is_off_by_default(self):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), max_position_pct=0.05)
        engine.portfolio_value = 100.0
        self.assertEqual(engine.small_account_concentration(10_000_000.0), 0.05)

    def test_unobservable_liquidity_never_widens_anything(self):
        engine = self._engine(500.0)
        self.assertEqual(engine.small_account_concentration(0.0), 0.05)
        self.assertEqual(engine.small_account_concentration(-1.0), 0.05)

    def test_only_the_position_ceiling_moves(self):
        engine = self._engine(500.0)
        engine.small_account_concentration(5_000_000.0)
        # Exposure, portfolio risk and the daily-loss switch bound ruin, and
        # this widens none of them.
        self.assertEqual(engine.max_total_exposure_pct, 0.30)
        self.assertEqual(engine.max_portfolio_risk, 0.10)
        self.assertEqual(engine.max_position_pct, 0.05)


class TestWalletIndependence(unittest.TestCase):
    """Twelve wallets buying is not twelve pieces of evidence."""

    @staticmethod
    def _entry(token, wallet, timestamp, skill=0.8, capital=100.0):
        return Entry(token=token, wallet=wallet, timestamp=timestamp,
                     skill=skill, capital_usd=capital)

    def _copycat_history(self, launches=10):
        """LEADER decides; FOLLOWER copies a second later, every time."""
        graph = WalletIndependence(min_observations=4)
        for index in range(launches):
            base = index * 1_000.0
            graph.record_entries([
                self._entry(f"t{index}", "LEADER", base),
                self._entry(f"t{index}", "FOLLOWER", base + 1.0),
                self._entry(f"t{index}", "SOLO", base + 400.0),
            ])
        return graph

    def test_a_consistent_follower_is_scored_as_dependent(self):
        report = self._copycat_history().compute()
        self.assertEqual(report.status, "OK")
        self.assertAlmostEqual(report.scores["FOLLOWER"], 0.0)
        self.assertAlmostEqual(report.scores["LEADER"], 1.0)
        self.assertAlmostEqual(report.scores["SOLO"], 1.0)
        self.assertEqual(report.followers["FOLLOWER"][0][0], "LEADER")

    def test_a_thin_history_blocks_rather_than_guessing(self):
        graph = WalletIndependence(min_observations=4)
        graph.record_entries([self._entry("t0", "A", 0.0), self._entry("t0", "B", 1.0)])
        report = graph.compute()
        # A follow ratio from two observations is noise with a decimal point.
        self.assertEqual(report.status, "DATA_BLOCKED")
        self.assertIsNone(graph.score_of("B", report))

    def test_entering_the_same_token_far_apart_is_not_following(self):
        graph = WalletIndependence(min_observations=4, follow_window=3.0)
        for index in range(8):
            base = index * 1_000.0
            graph.record_entries([self._entry(f"t{index}", "A", base),
                                  self._entry(f"t{index}", "B", base + 60.0)])
        report = graph.compute()
        # Two people reacting to the same public event minutes apart are two
        # actors, not one.
        self.assertAlmostEqual(report.scores["B"], 1.0)

    def test_a_wallet_buying_twice_is_not_evidence_about_anyone(self):
        graph = WalletIndependence(min_observations=2)
        for index in range(4):
            base = index * 1_000.0
            graph.record_entries([
                self._entry(f"t{index}", "A", base),
                self._entry(f"t{index}", "A", base + 0.5),
                self._entry(f"t{index}", "A", base + 1.0),
                self._entry(f"t{index}", "B", base + 500.0),
            ])
        report = graph.compute()
        self.assertAlmostEqual(report.scores["B"], 1.0)


class TestSmartFlowAggregation(unittest.TestCase):
    """Collapsing wallets to actors can only ever discount the signal."""

    @staticmethod
    def _entry(wallet, skill=0.8, capital=100.0, timestamp=0.0):
        return Entry(token="t", wallet=wallet, timestamp=timestamp,
                     skill=skill, capital_usd=capital)

    def test_a_sybil_cluster_scores_far_below_its_wallet_count(self):
        report = IndependenceReport(status="OK", scores={f"S{i}": 0.0 for i in range(10)})
        flow = aggregate_smart_flow([self._entry(f"S{i}") for i in range(10)], report)
        self.assertEqual(flow.status, "OK")
        self.assertAlmostEqual(flow.evidence, 0.0)
        self.assertAlmostEqual(flow.naive_evidence, 10 * 0.8 * 100.0)
        self.assertAlmostEqual(flow.discount, 0.0)

    def test_three_independent_wallets_outweigh_ten_linked_ones(self):
        sybil_report = IndependenceReport(status="OK", scores={f"S{i}": 0.05 for i in range(10)})
        clean_report = IndependenceReport(status="OK", scores={f"I{i}": 1.0 for i in range(3)})
        sybil = aggregate_smart_flow([self._entry(f"S{i}") for i in range(10)], sybil_report)
        clean = aggregate_smart_flow([self._entry(f"I{i}") for i in range(3)], clean_report)
        # A counter of buyers ranks these the other way round.
        self.assertGreater(clean.evidence, sybil.evidence)

    def test_independence_can_never_inflate_the_signal(self):
        report = IndependenceReport(status="OK", scores={"A": 1.0, "B": 1.0})
        flow = aggregate_smart_flow([self._entry("A"), self._entry("B")], report)
        self.assertLessEqual(flow.evidence, flow.naive_evidence + 1e-9)
        self.assertLessEqual(flow.discount, 1.0)

    def test_an_unmeasured_wallet_is_not_treated_as_independent(self):
        blocked = IndependenceReport(status="DATA_BLOCKED")
        flow = aggregate_smart_flow([self._entry("NEW")], blocked)
        # A brand-new wallet appearing beside a known cluster is what a Sybil
        # looks like, so unknown must not be the most persuasive state.
        self.assertLess(flow.discount, 1.0)
        self.assertGreater(flow.discount, 0.0)
        self.assertEqual(flow.unmeasured_wallets, 1)

    def test_buyers_without_a_skill_or_size_are_not_counted(self):
        report = IndependenceReport(status="OK", scores={"A": 1.0})
        flow = aggregate_smart_flow(
            [Entry("t", "A", 0.0, skill=None, capital_usd=100.0)], report)
        self.assertEqual(flow.status, "DATA_BLOCKED")


class TestBuyerDNA(unittest.TestCase):
    """Order matters: the same aggregate statistics mean opposite things."""

    @staticmethod
    def _fingerprint(token, skills, independence=None, linked=None):
        count = len(skills)
        return BuyerFingerprint(
            token=token, skills=list(skills),
            independence=list(independence or [1.0] * count),
            creator_linked=list(linked or [False] * count),
            sizes=[100.0] * count,
        )

    def _corpus(self, dna):
        for index in range(30):
            dna.add(self._fingerprint(f"monster-{index}", [0.2, 0.85, 0.9, 0.95]), "monster")
        for index in range(30):
            dna.add(self._fingerprint(f"dump-{index}", [0.9, 0.85, 0.2, 0.1],
                                      independence=[0.0] * 4,
                                      linked=[True] * 4), "dump_vehicle")
        return dna

    def test_a_thin_corpus_refuses_to_be_a_prior(self):
        dna = BuyerDNA(min_corpus=50)
        dna.add(self._fingerprint("only", [0.9, 0.9]), "monster")
        match = dna.match(self._fingerprint("query", [0.9, 0.9]))
        # 1-NN against three launches is a coincidence with a confidence
        # interval, not a prior.
        self.assertEqual(match.status, "DATA_BLOCKED")
        self.assertIn("below", match.detail)

    def test_improving_buyer_quality_matches_monsters(self):
        dna = self._corpus(BuyerDNA(min_corpus=50))
        match = dna.match(self._fingerprint("query", [0.25, 0.8, 0.88, 0.92]))
        self.assertEqual(match.status, "OK")
        self.assertEqual(match.label, "monster")

    def test_the_reversed_sequence_matches_dump_vehicles(self):
        """Same four skill values, opposite order, opposite conclusion."""
        dna = self._corpus(BuyerDNA(min_corpus=50))
        match = dna.match(self._fingerprint("query", [0.9, 0.85, 0.2, 0.1],
                                            independence=[0.0] * 4, linked=[True] * 4))
        self.assertEqual(match.label, "dump_vehicle")

    def test_padding_is_distinguishable_from_a_real_zero(self):
        short = self._fingerprint("short", [0.5, 0.5])
        vector = short.vector(4)
        # "only two buyers" and "the third buyer scored zero" must not collapse
        # into the same state.
        self.assertEqual(list(vector[:4]), [0.5, 0.5, -1.0, -1.0])

    def test_no_buyers_yet_blocks(self):
        dna = self._corpus(BuyerDNA(min_corpus=50))
        self.assertEqual(dna.match(self._fingerprint("query", [])).status, "DATA_BLOCKED")

    def test_the_fingerprint_keeps_the_first_unique_buyers_in_order(self):
        report = IndependenceReport(status="OK", scores={"A": 1.0, "B": 0.2})
        entries = [
            Entry("t", "B", 2.0, skill=0.3, capital_usd=10.0),
            Entry("t", "A", 1.0, skill=0.9, capital_usd=50.0),
            Entry("t", "A", 3.0, skill=0.9, capital_usd=70.0),
        ]
        fingerprint = build_fingerprint("t", entries, report, depth=25)
        self.assertEqual(fingerprint.skills, [0.9, 0.3])
        self.assertEqual(fingerprint.independence, [1.0, 0.2])


class TestSwarmPredictor(unittest.TestCase):
    """The forward question, not the backward one."""

    NOW = 1_000.0

    def _entries(self, count, skill=0.9, spread=1.0, start=9.0):
        return [Entry("t", f"W{i}", self.NOW - start + i * spread,
                      skill=skill, capital_usd=100.0) for i in range(count)]

    def _report(self, count, independence=1.0):
        return IndependenceReport(status="OK",
                                  scores={f"W{i}": independence for i in range(count)})

    def test_independent_skilled_arrivals_produce_evidence(self):
        reading = SwarmPredictor().evaluate(self._entries(6), self._report(6), self.NOW)
        self.assertGreater(reading.evidence, 0)
        self.assertEqual(reading.independent_skilled_so_far, 6)

    def test_a_linked_cluster_produces_none(self):
        reading = SwarmPredictor().evaluate(
            self._entries(6), self._report(6, independence=0.0), self.NOW)
        self.assertEqual(reading.independent_skilled_so_far, 0)
        self.assertEqual(reading.evidence, 0.0)

    def test_unskilled_arrivals_produce_none(self):
        reading = SwarmPredictor().evaluate(
            self._entries(6, skill=0.1), self._report(6), self.NOW)
        self.assertEqual(reading.independent_skilled_so_far, 0)

    def test_an_untrained_predictor_gives_no_probability(self):
        reading = SwarmPredictor().evaluate(self._entries(6), self._report(6), self.NOW)
        self.assertEqual(reading.status, "DATA_BLOCKED")
        self.assertIsNone(reading.probability)

    def test_a_trained_predictor_reports_one(self):
        class FakeModel:
            def predict_proba(self, matrix):
                return np.asarray([[0.3, 0.7]])

        predictor = SwarmPredictor()
        self.assertTrue(predictor.load_model(FakeModel()))
        reading = predictor.evaluate(self._entries(6), self._report(6), self.NOW)
        self.assertEqual(reading.status, "OK")
        self.assertAlmostEqual(reading.probability, 0.7)

    def test_a_model_without_predict_proba_is_refused_at_load(self):
        predictor = SwarmPredictor()
        self.assertFalse(predictor.load_model(object()))
        self.assertFalse(predictor.is_trained)

    def test_an_empty_window_blocks(self):
        reading = SwarmPredictor().evaluate(self._entries(3), self._report(3),
                                            self.NOW + 10_000)
        self.assertEqual(reading.status, "DATA_BLOCKED")


class TestLeakAttribution(unittest.TestCase):
    """Accuracy is the wrong objective when returns are tail-dominated.

    A thousand ordinary rejections weigh the same as the one 30x that was
    passed on, and the 30x cost more geometric growth than the thousand
    produced.
    """

    EQUITY = 10_000.0

    def test_a_missed_monster_is_measured_in_log_wealth(self):
        finding = missed_monster(
            {"token": "moon", "entered": False, "max_feasible_multiple": 30.0,
             "capacity_usd": 500.0, "rejection_reason": "insufficient_upside"},
            self.EQUITY, 0.05)
        self.assertEqual(finding.leak, Leak.MISSED_MONSTER)
        # 5% of the book at 30x: log(0.95 + 0.05*30).
        self.assertAlmostEqual(finding.forgone_log_growth, math.log(0.95 + 0.05 * 30.0))
        self.assertEqual(finding.evidence["rejection_reason"], "insufficient_upside")

    def test_a_leak_is_capped_by_what_was_actually_executable(self):
        """Blaming the book for a size the venue could not fill is fantasy."""
        thin = missed_monster(
            {"token": "thin", "entered": False, "max_feasible_multiple": 30.0,
             "capacity_usd": 100.0}, self.EQUITY, 0.05)
        deep = missed_monster(
            {"token": "deep", "entered": False, "max_feasible_multiple": 30.0,
             "capacity_usd": 500.0}, self.EQUITY, 0.05)
        # Same multiple, five times the fillable size: the deep one is the
        # larger leak, because the thin one was never a $500 opportunity.
        self.assertLess(thin.forgone_log_growth, deep.forgone_log_growth)

        # And capacity beyond the position ceiling earns no extra credit: the
        # book was never going to put more than 5% into one token.
        unlimited = missed_monster(
            {"token": "deep2", "entered": False, "max_feasible_multiple": 30.0,
             "capacity_usd": 5_000_000.0}, self.EQUITY, 0.05)
        self.assertAlmostEqual(unlimited.forgone_log_growth, deep.forgone_log_growth)

    def test_a_thin_monster_can_still_outrank_a_deep_mediocrity(self):
        """Log wealth, not multiples: the ranking has to survive both cases."""
        thin_monster = missed_monster(
            {"token": "thin", "entered": False, "max_feasible_multiple": 50.0,
             "capacity_usd": 200.0}, self.EQUITY, 0.05)
        deep_mediocrity = missed_monster(
            {"token": "deep", "entered": False, "max_feasible_multiple": 1.2,
             "capacity_usd": 50_000.0}, self.EQUITY, 0.05)
        self.assertGreater(thin_monster.forgone_log_growth,
                           deep_mediocrity.forgone_log_growth)

    def test_unobserved_capacity_produces_no_finding_rather_than_a_guess(self):
        self.assertIsNone(missed_monster(
            {"token": "x", "entered": False, "max_feasible_multiple": 30.0},
            self.EQUITY, 0.05))

    def test_a_premature_exit_is_the_gap_between_realized_and_feasible(self):
        finding = premature_exit(
            {"token": "runner", "entered": True, "realized_multiple": 3.0,
             "max_feasible_multiple": 30.0, "capacity_usd": 5_000.0,
             "position_fraction": 0.05, "exit_reason": "profit_ratchet_5x"},
            self.EQUITY, 0.05)
        self.assertEqual(finding.leak, Leak.PREMATURE_EXIT)
        self.assertAlmostEqual(finding.evidence["tail_capture"], 0.1)
        self.assertGreater(finding.forgone_log_growth, 0)

    def test_capturing_the_whole_move_is_not_a_leak(self):
        self.assertIsNone(premature_exit(
            {"token": "clean", "entered": True, "realized_multiple": 30.0,
             "max_feasible_multiple": 30.0, "capacity_usd": 5_000.0,
             "position_fraction": 0.05}, self.EQUITY, 0.05))

    def test_a_rug_loss_records_how_early_it_was_knowable(self):
        finding = rug_loss({"token": "rug", "entered": True, "rugged": True,
                            "realized_multiple": 0.05, "position_fraction": 0.02,
                            "earliest_warning_seconds": 4.0})
        self.assertEqual(finding.leak, Leak.RUG_LOSS)
        self.assertGreater(finding.forgone_log_growth, 0)
        self.assertEqual(finding.evidence["earliest_warning_seconds"], 4.0)

    def test_under_sizing_a_correct_call_is_its_own_leak(self):
        finding = sizing_leak(
            {"token": "small", "entered": True, "realized_multiple": 12.0,
             "position_fraction": 0.002, "capacity_usd": 5_000.0}, self.EQUITY, 0.05)
        self.assertEqual(finding.leak, Leak.UNDER_SIZED)
        self.assertGreater(finding.forgone_log_growth, 0)

    def test_over_sizing_beyond_capacity_is_also_a_leak(self):
        finding = sizing_leak(
            {"token": "big", "entered": True, "realized_multiple": 2.0,
             "position_fraction": 0.05, "capacity_usd": 100.0}, self.EQUITY, 0.05)
        self.assertEqual(finding.leak, Leak.OVER_SIZED)

    def test_an_execution_miss_is_distinct_from_a_prediction_failure(self):
        finding = execution_miss(
            {"token": "missed", "entered": False, "attempted": True,
             "max_feasible_multiple": 8.0, "capacity_usd": 5_000.0,
             "failure_reason": "bundle_not_landed"}, self.EQUITY, 0.05)
        self.assertEqual(finding.leak, Leak.EXECUTION_MISS)
        # One is fixed with a model, the other with tips, regions and code.
        self.assertEqual(finding.evidence["failure_reason"], "bundle_not_landed")

    def test_a_token_never_attempted_is_not_an_execution_miss(self):
        self.assertIsNone(execution_miss(
            {"token": "skipped", "entered": False, "attempted": False,
             "max_feasible_multiple": 8.0, "capacity_usd": 5_000.0}, self.EQUITY, 0.05))

    def test_research_ranking_points_at_the_cause_not_the_category(self):
        trades = [
            {"token": f"m{i}", "entered": False, "max_feasible_multiple": 25.0,
             "capacity_usd": 5_000.0, "rejection_reason": "insufficient_upside"}
            for i in range(5)
        ] + [
            {"token": "r", "entered": True, "rugged": True, "realized_multiple": 0.05,
             "position_fraction": 0.01},
        ]
        report = rank_research(find_leaks(trades, self.EQUITY))
        self.assertGreater(report["total_forgone_log_growth"], 0)
        top = report["top_causes"][0]
        # "We lose most to missed monsters" is not actionable; "...rejected for
        # insufficient_upside" is.
        self.assertEqual((top["leak"], top["reason"]),
                         ("missed_monster", "insufficient_upside"))
        self.assertEqual(len(report["worst_tokens"]), 5)


class TestAlphaLedger(unittest.TestCase):
    """An attribution that does not reconcile is a story."""

    def test_the_ledger_sums_to_the_books_pnl(self):
        trades = [
            {"entered": True, "mechanism": "smart_wallet_swarm", "realized_pnl_usd": 180.0},
            {"entered": True, "mechanism": "smart_wallet_swarm", "realized_pnl_usd": -30.0},
            {"entered": True, "mechanism": "creator_dna", "realized_pnl_usd": 42.0},
            {"entered": True, "mechanism": "exit_policy", "realized_pnl_usd": -64.0},
        ]
        ledger = alpha_ledger(trades)
        self.assertTrue(ledger["reconciles"])
        self.assertAlmostEqual(ledger["total_usd"], 128.0)
        self.assertEqual(ledger["entries"][0]["mechanism"], "smart_wallet_swarm")

    def test_unattributed_trades_are_bucketed_not_dropped(self):
        ledger = alpha_ledger([
            {"entered": True, "realized_pnl_usd": 50.0},
            {"entered": True, "mechanism": "creator_dna", "realized_pnl_usd": 50.0},
        ])
        # Silently dropping them is how the sum stops reconciling.
        self.assertTrue(ledger["reconciles"])
        self.assertIn("unattributed", [e["mechanism"] for e in ledger["entries"]])

    def test_concentration_shows_a_book_leaning_on_one_mechanism(self):
        ledger = alpha_ledger([
            {"entered": True, "mechanism": "one_trick", "realized_pnl_usd": 900.0},
            {"entered": True, "mechanism": "other", "realized_pnl_usd": 100.0},
        ])
        self.assertAlmostEqual(ledger["concentration"], 0.9)


class TestTailContribution(unittest.TestCase):
    def test_the_top_decile_share_is_reported(self):
        trades = ([{"entered": True, "wealth_multiple": 1.01} for _ in range(90)]
                  + [{"entered": True, "wealth_multiple": 3.0} for _ in range(10)])
        report = tail_contribution(trades)
        self.assertEqual(report["status"], "OK")
        self.assertGreater(report["top_10pct_share"], 0.5)

    def test_a_population_too_small_for_a_top_tenth_of_a_percent_reports_none(self):
        report = tail_contribution([{"entered": True, "wealth_multiple": 2.0}
                                    for _ in range(50)])
        # 0.0 would read as "the top 0.1% contributed nothing" about a
        # population that has no top 0.1%.
        self.assertIsNone(report["top_01pct_share"])
        self.assertIsNotNone(report["top_10pct_share"])

    def test_no_positive_trades_blocks(self):
        self.assertEqual(
            tail_contribution([{"entered": True, "wealth_multiple": 0.5}])["status"],
            "DATA_BLOCKED")


class TestEdgeDecay(unittest.TestCase):
    """A quiet edge is not a dead edge."""

    def test_a_small_sample_is_measuring_never_degraded(self):
        monitor = EdgeDecayMonitor(min_trades=30)
        monitor.set_baseline("swarm", 0.02)
        for _ in range(8):
            monitor.record("swarm", -0.01)
        health = monitor.health("swarm")
        # Retiring an edge on eight trades retires exactly the regime-specific
        # edges that go quiet and come back.
        self.assertEqual(health["status"], EdgeDecayMonitor.MEASURING)

    def test_sustained_underperformance_against_the_baseline_degrades(self):
        monitor = EdgeDecayMonitor(min_trades=30)
        monitor.set_baseline("swarm", 0.02)
        for _ in range(30):
            monitor.record("swarm", -0.005)
        self.assertEqual(monitor.health("swarm")["status"], EdgeDecayMonitor.DEGRADED)

    def test_partial_underperformance_is_weakening_not_degraded(self):
        monitor = EdgeDecayMonitor(min_trades=30, weakening_ratio=0.5)
        monitor.set_baseline("swarm", 0.02)
        for _ in range(30):
            monitor.record("swarm", 0.006)
        self.assertEqual(monitor.health("swarm")["status"], EdgeDecayMonitor.WEAKENING)

    def test_performance_at_the_baseline_stays_healthy(self):
        monitor = EdgeDecayMonitor(min_trades=30)
        monitor.set_baseline("swarm", 0.02)
        for _ in range(30):
            monitor.record("swarm", 0.021)
        self.assertEqual(monitor.health("swarm")["status"], EdgeDecayMonitor.HEALTHY)

    def test_a_mechanism_with_no_baseline_is_judged_on_sign(self):
        monitor = EdgeDecayMonitor(min_trades=5)
        for _ in range(5):
            monitor.record("new", 0.01)
        self.assertEqual(monitor.health("new")["status"], EdgeDecayMonitor.HEALTHY)


class TestHazardCurve(unittest.TestCase):
    """One p_rug conflates two questions that call for opposite actions."""

    def test_probabilities_at_different_horizons_become_combinable_rates(self):
        curve = hazard_curve_from_probabilities({
            HazardMechanism.CREATOR_SELLING: (0.5, 300.0),
            HazardMechanism.LIQUIDITY_REMOVAL: (0.1, 60.0),
        })
        self.assertEqual(curve.status, "OK")
        # Adding a 5-minute probability to a 1-minute one directly is a
        # category error; converting to rates first is what makes them add.
        self.assertAlmostEqual(curve.rates[HazardMechanism.CREATOR_SELLING],
                               -math.log(0.5) / 300.0)
        self.assertAlmostEqual(curve.rates[HazardMechanism.LIQUIDITY_REMOVAL],
                               -math.log(0.9) / 60.0)

    def test_eventual_death_and_imminent_collapse_are_different_states(self):
        doomed_but_slow = hazard_curve_from_probabilities({
            HazardMechanism.CREATOR_SELLING: (0.80, 1_200.0)})
        imminent = hazard_curve_from_probabilities({
            HazardMechanism.CREATOR_SELLING: (0.35, 1.0)})
        # 80% chance of eventually dying is not a reason to be out right now;
        # 35% chance of collapse in the next second is.
        self.assertLess(doomed_but_slow.probability_within(1.0), 0.01)
        self.assertGreater(imminent.probability_within(1.0), 0.3)

    def test_the_curve_is_monotone_across_horizons(self):
        curve = hazard_curve_from_probabilities({
            HazardMechanism.INSIDER_CLUSTER_EXIT: (0.2, 10.0)})
        values = [curve.probability_within(h) for h in HAZARD_HORIZONS]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(values[-1], 1.0)

    def test_the_dominant_mechanism_is_reported(self):
        curve = hazard_curve_from_probabilities({
            HazardMechanism.CREATOR_SELLING: (0.05, 10.0),
            HazardMechanism.SELLABILITY_LOSS: (0.60, 10.0),
        })
        self.assertEqual(curve.dominant(), HazardMechanism.SELLABILITY_LOSS)

    def test_no_usable_mechanism_blocks(self):
        self.assertEqual(hazard_curve_from_probabilities({}).status, "DATA_BLOCKED")
        self.assertEqual(
            hazard_curve_from_probabilities(
                {HazardMechanism.CREATOR_SELLING: (0.5, 0.0)}).status,
            "DATA_BLOCKED")


class TestEscapeProbability(unittest.TestCase):
    """A predicted 5x we cannot exit is not a 5x."""

    def _curve(self, mechanism=HazardMechanism.CREATOR_SELLING, probability=0.3,
               horizon=10.0):
        return hazard_curve_from_probabilities({mechanism: (probability, horizon)})

    def test_escape_needs_both_speed_and_depth(self):
        curve = self._curve()
        deep_fast = escape_probability(1_000, 1_000, 0.2, curve)
        deep_slow = escape_probability(1_000, 1_000, 30.0, curve)
        thin_fast = escape_probability(1_000, 100, 0.2, curve)
        self.assertGreater(deep_fast.probability, deep_slow.probability)
        self.assertGreater(deep_fast.probability, thin_fast.probability)
        self.assertAlmostEqual(thin_fast.fillable_share, 0.1)

    def test_either_factor_at_zero_takes_the_whole_thing_to_zero(self):
        curve = self._curve()
        # Nothing fillable: a perfectly fast transaction has not escaped.
        self.assertEqual(escape_probability(1_000, 0, 0.01, curve).probability, 0.0)
        # Nothing lands: a perfectly sized order has not escaped either.
        self.assertEqual(
            escape_probability(1_000, 1_000, 0.01, curve, landing_probability=0.0).probability,
            0.0)

    def test_speed_does_not_help_against_an_unsellable_token(self):
        """Once liquidity is gone there is nothing to sell into."""
        outrunnable = self._curve(HazardMechanism.CREATOR_SELLING, 0.9, 1.0)
        unescapable = self._curve(HazardMechanism.SELLABILITY_LOSS, 0.9, 1.0)
        fast_vs_creator = escape_probability(1_000, 1_000, 0.05, outrunnable)
        slow_vs_creator = escape_probability(1_000, 1_000, 2.0, outrunnable)
        fast_vs_frozen = escape_probability(1_000, 1_000, 0.05, unescapable)
        slow_vs_frozen = escape_probability(1_000, 1_000, 2.0, unescapable)

        # Being fast buys a lot against a seller and comparatively little
        # against a mint that has been frozen.
        creator_gain = fast_vs_creator.probability - slow_vs_creator.probability
        frozen_gain = fast_vs_frozen.probability - slow_vs_frozen.probability
        self.assertGreater(creator_gain, 0)
        self.assertGreater(creator_gain, frozen_gain)

    def test_unmeasured_capacity_blocks_rather_than_assuming_liquidity(self):
        estimate = escape_probability(1_000, None, 0.2, self._curve())
        self.assertEqual(estimate.status, "DATA_BLOCKED")
        # Unknown depth is not full depth.
        self.assertEqual(estimate.probability, 0.0)

    def test_a_blocked_hazard_curve_blocks_escape(self):
        self.assertEqual(
            escape_probability(1_000, 1_000, 0.2, HazardCurve(status="DATA_BLOCKED")).status,
            "DATA_BLOCKED")


class TestRideOrReject(unittest.TestCase):
    """'Likely to eventually rug' and 'about to rug' call for opposite actions."""

    def _verdict(self, hazard_probability, hazard_horizon, sellable=1_000,
                 latency=0.2, upside=3.0, p_up=0.6, mechanism=HazardMechanism.CREATOR_SELLING):
        curve = hazard_curve_from_probabilities({mechanism: (hazard_probability, hazard_horizon)})
        escape = escape_probability(1_000, sellable, latency, curve)
        return ride_or_reject(upside, p_up, curve, escape, position_fraction=0.02,
                              horizon_s=10.0)

    def test_a_doomed_token_with_a_wide_window_is_still_rideable(self):
        # 80% chance of eventual death over 20 minutes, 60% chance of another
        # 3x first, and we can get out in 200ms.
        verdict = self._verdict(0.80, 1_200.0)
        self.assertEqual(verdict.status, "OK")
        self.assertEqual(verdict.action, "ride")
        self.assertGreater(verdict.e_log_ride, 0)

    def test_an_imminent_collapse_is_rejected_despite_the_same_upside(self):
        verdict = self._verdict(0.35, 1.0)
        self.assertEqual(verdict.action, "reject")

    def test_upside_we_cannot_exit_is_not_upside(self):
        """The whole reason escape is computed separately."""
        liquid = self._verdict(0.5, 600.0, sellable=1_000, upside=5.0)
        trapped = self._verdict(0.5, 600.0, sellable=1, upside=5.0)
        self.assertEqual(liquid.action, "ride")
        self.assertLess(trapped.e_log_ride, liquid.e_log_ride)

    def test_a_bigger_predicted_move_does_not_rescue_a_frozen_token(self):
        frozen = self._verdict(0.9, 1.0, upside=50.0, p_up=0.9,
                               mechanism=HazardMechanism.SELLABILITY_LOSS)
        self.assertEqual(frozen.action, "reject")

    def test_missing_inputs_are_not_a_tradeable_state(self):
        curve = hazard_curve_from_probabilities(
            {HazardMechanism.CREATOR_SELLING: (0.3, 10.0)})
        blocked_escape = escape_probability(1_000, None, 0.2, curve)
        verdict = ride_or_reject(3.0, 0.6, curve, blocked_escape, 0.02, 10.0)
        self.assertEqual(verdict.status, "DATA_BLOCKED")
        self.assertEqual(verdict.action, "reject")


class TestLiquidationLadder(unittest.TestCase):
    """A chart can look healthy while executable exit liquidity rots."""

    def test_each_slice_reports_its_own_escape(self):
        frontier = build_frontier(lambda size: (True, size * 1e-6), 10_000_000, "exit")
        curve = hazard_curve_from_probabilities(
            {HazardMechanism.CREATOR_SELLING: (0.3, 30.0)})
        ladder = liquidation_ladder(1_000_000, frontier, curve, expected_latency_s=0.25)

        self.assertEqual(ladder["status"], "OK")
        shares = [rung["escape_probability"] for rung in ladder["rungs"]]
        # Smaller slices get out more reliably; asking the question at one size
        # only is how the deterioration goes unseen.
        self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertEqual(ladder["rungs"][0]["share"], 0.10)

    def test_an_unmeasurable_frontier_blocks_the_whole_ladder(self):
        curve = hazard_curve_from_probabilities(
            {HazardMechanism.CREATOR_SELLING: (0.3, 30.0)})
        ladder = liquidation_ladder(1_000, Frontier(status="DATA_BLOCKED", side="exit"),
                                    curve, 0.25)
        self.assertEqual(ladder["status"], "DATA_BLOCKED")


class TestHotStateBudget(unittest.TestCase):
    """The machine that reacts in milliseconds is not the one holding the moat."""

    @staticmethod
    def _wallet(name, launches=50, monster_rate=0.0, rug_exposure=0.0, pnl=0.0,
                cluster=""):
        return CompactWalletDNA(wallet=name, launches_seen=launches,
                                monster_rate=monster_rate, rug_exposure=rug_exposure,
                                pnl_quality=pnl, cluster_id=cluster)

    def test_a_distilled_record_is_small_and_fixed_shape(self):
        record = self._wallet("W", monster_rate=0.1)
        # Hundreds of bytes of conclusions, not megabytes of transactions.
        self.assertLess(len(repr(asdict(record)).encode()), 700)
        self.assertGreater(record.information_value, 0)

    def test_an_undistillable_wallet_is_unknown_not_average(self):
        blocked = CompactWalletDNA(wallet="W", status="DATA_BLOCKED", monster_rate=0.9)
        self.assertEqual(blocked.information_value, 0.0)
        self.assertEqual(self._wallet("W", launches=0, monster_rate=0.9).information_value,
                         0.0)

    def test_rug_history_is_worth_as_much_resident_as_monster_history(self):
        runner = self._wallet("R", monster_rate=0.3)
        rugger = self._wallet("G", rug_exposure=0.3)
        # Knowing which wallets show up before collapses is as valuable as
        # knowing which show up before runs.
        self.assertAlmostEqual(runner.information_value, rugger.information_value)

    def test_confidence_saturates_with_observations(self):
        few = self._wallet("A", launches=5, monster_rate=0.3).information_value
        some = self._wallet("B", launches=20, monster_rate=0.3).information_value
        many = self._wallet("C", launches=200, monster_rate=0.3).information_value
        self.assertLess(few, some)
        self.assertLess(some, many)
        # The hundredth launch says much less about a wallet than the tenth.
        self.assertLess(many - some, some - few)


class TestEconomicCache(unittest.TestCase):
    """Plain LRU retires exactly the actors worth keeping."""

    NOW = 1_000_000.0

    @staticmethod
    def _wallet(name, monster_rate=0.0, launches=50, cluster=""):
        return CompactWalletDNA(wallet=name, launches_seen=launches,
                                monster_rate=monster_rate, cluster_id=cluster)

    def test_a_valuable_quiet_actor_outranks_a_worthless_recent_one(self):
        cache = EconomicCache(capacity=2, half_life_seconds=1_800.0)
        # A known rugger cluster, quiet for an hour.
        cache.put("rugger", self._wallet("rugger", monster_rate=0.8),
                  now=self.NOW - 3_600)
        # A wallet that has never distinguished itself, traded a second ago.
        cache.put("noise", self._wallet("noise", monster_rate=0.0), now=self.NOW - 1)
        cache.put("new", self._wallet("new", monster_rate=0.4), now=self.NOW)

        self.assertIn("rugger", cache)
        self.assertNotIn("noise", cache)
        self.assertEqual(cache.evictions, 1)

    def test_recency_still_matters_between_comparable_records(self):
        cache = EconomicCache(capacity=2, half_life_seconds=600.0)
        cache.put("old", self._wallet("old", monster_rate=0.5), now=self.NOW - 7_200)
        cache.put("fresh", self._wallet("fresh", monster_rate=0.5), now=self.NOW)
        cache.put("newest", self._wallet("newest", monster_rate=0.5), now=self.NOW)
        self.assertNotIn("old", cache)

    def test_a_burst_of_irrelevant_wallets_cannot_expand_memory(self):
        cache = EconomicCache(capacity=100)
        cache.put("keeper", self._wallet("keeper", monster_rate=0.9), now=self.NOW)
        for index in range(5_000):
            cache.put(f"junk-{index}", self._wallet(f"junk-{index}"), now=self.NOW)
        # 200,000 irrelevant wallets becoming active must not grow the node.
        self.assertLessEqual(len(cache), 100)
        self.assertIn("keeper", cache)

    def test_pinned_entities_survive_pressure(self):
        cache = EconomicCache(capacity=2, pin_predicate=lambda r: r.cluster_id == "live")
        cache.put("pinned", self._wallet("pinned", cluster="live"), now=self.NOW)
        for index in range(20):
            cache.put(f"other-{index}", self._wallet(f"other-{index}", monster_rate=0.9),
                      now=self.NOW)
        self.assertIn("pinned", cache)

    def test_the_cap_holds_even_when_everything_is_pinned(self):
        cache = EconomicCache(capacity=3, pin_predicate=lambda r: True)
        for index in range(50):
            cache.put(f"w-{index}", self._wallet(f"w-{index}"), now=self.NOW + index)
        # Growing past the cap is not an option on a fixed-memory node.
        self.assertLessEqual(len(cache), 3)

    def test_peeking_does_not_skew_eviction(self):
        cache = EconomicCache(capacity=2)
        cache.put("a", self._wallet("a"), now=self.NOW - 10_000)
        cache.put("b", self._wallet("b"), now=self.NOW)
        cache.peek("a")
        cache.put("c", self._wallet("c"), now=self.NOW)
        self.assertNotIn("a", cache)


class TestAsyncArchiveWriter(unittest.TestCase):
    """A queue that blocks on slow storage turns a disk problem into a missed launch."""

    def _writer(self, directory, max_queue=5, quota_gb=5.0):
        return AsyncArchiveWriter(
            Path(directory),
            HotStateBudget(max_archive_queue=max_queue, max_local_archive_gb=quota_gb))

    def test_submit_never_blocks_and_overflow_drops_the_oldest(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory, max_queue=3)
            for index in range(10):
                writer.submit({"n": index})
            report = writer.report()
            self.assertEqual(report["pending"], 3)
            self.assertEqual(report["dropped"], 7)
            # The newest observation is the one a decision might still need.
            written = writer.drain()
            self.assertEqual(written, 3)

    def test_drops_are_counted_rather_than_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory, max_queue=2)
            for index in range(10):
                writer.submit({"n": index})
            # Silent loss is what makes a research lake quietly wrong.
            self.assertGreater(writer.report()["drop_rate"], 0.5)

    def test_records_reach_disk_on_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory)
            writer.submit({"token": "abc"})
            self.assertEqual(writer.drain(), 1)
            files = list(Path(directory).glob("events-*.log"))
            self.assertEqual(len(files), 1)
            self.assertIn("abc", files[0].read_text())

    def test_the_local_quota_stops_the_spool_from_filling_the_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory, quota_gb=0.0)
            writer.submit({"token": "abc"})
            self.assertEqual(writer.drain(), 0)
            # A stalled upload must not take the disk the trading process needs.
            self.assertEqual(writer.report()["quota_stops"], 1)
            self.assertEqual(writer.report()["dropped"], 1)

    def test_draining_an_empty_queue_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self._writer(directory).drain(), 0)


class TestHotState(unittest.TestCase):
    NOW = 2_000_000.0

    def test_active_tokens_are_capped_and_aged_out(self):
        with tempfile.TemporaryDirectory() as directory:
            state = HotState(HotStateBudget(max_active_tokens=10, max_event_age_seconds=60),
                             archive_root=Path(directory))
            for index in range(500):
                state.touch_token(f"t{index}", now=self.NOW + index * 0.001)
            self.assertLessEqual(len(state.active_tokens), 10)

            state.touch_token("recent", now=self.NOW + 10_000)
            self.assertEqual(len(state.active_tokens), 1)

    def test_the_budget_is_declared_in_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            state = HotState(archive_root=Path(directory))
            report = state.report()
            # Exceeding a cap should be a visible condition, not something
            # discovered when the kernel kills the process.
            self.assertIn("max_hot_wallets", report["budget"])
            self.assertEqual(report["archive"]["dropped"], 0)


class TestHealthChecks(unittest.TestCase):
    """A monitor that reports an unrunnable check as OK is worse than no monitor."""

    NOW = 1_800_000_000.0

    def _readiness(self, **overrides):
        base = {
            "mode": "DRY_RUN", "live_submission_locked": True,
            "execution": {"dry_run": True},
            "portfolio": {"kill_switch_active": False, "daily_pnl": 12.0},
            "yellowstone": {"status": "STREAMING"},
            "rpc_program_stream": {"status": "STREAMING"},
            "prediction": "OK",
            "rug_hazard": {"model_trained": True, "model_status": "OK"},
            "exit_policy": {"status": "OK", "detail": "trained"},
            "dataset": {"market_observed_at": self.NOW - 10, "active_episodes": 12},
            "social": {"data_status": {"telegram": "OK_PUSH"}},
            "research": {"leads": 3},
            "champions": {"live_champions": 1, "shadow_models": 2, "decaying_champions": 0},
        }
        base.update(overrides)
        return base

    def _run(self, readiness=None, directory=None, execution_rows=None, now=None):
        root = Path(directory)
        state = root / "state"
        state.mkdir(parents=True, exist_ok=True)
        readiness_path = state / "readiness.json"
        if readiness is not None:
            readiness_path.write_text(json.dumps(readiness))
            os.utime(readiness_path, (now or self.NOW, now or self.NOW))
        execution_log = state / "execution_attempts.jsonl"
        if execution_rows is not None:
            execution_log.write_text("".join(json.dumps(row) + "\n" for row in execution_rows))
        return run_health_checks(readiness_path, root / "models", execution_log, root,
                                 now=now or self.NOW)

    def _state_of(self, report, name):
        return next(check.state for check in report.checks if check.name == name)

    def test_a_healthy_node_reports_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(), directory)
            self.assertEqual(self._state_of(report, "safety_live_lock"), State.OK)
            self.assertEqual(self._state_of(report, "feed_yellowstone"), State.OK)
            self.assertEqual(self._state_of(report, "data_market_observations"), State.OK)

    def test_a_stale_snapshot_is_critical_and_blocks_everything_downstream(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(), directory, now=self.NOW)
            # Re-run an hour later without the desk having rewritten the file.
            report = self._run(None, directory, now=self.NOW + 3_600)
            self.assertEqual(self._state_of(report, "readiness_freshness"), State.CRITICAL)
            # Reporting the last known-good values as current would manufacture
            # confidence exactly where visibility was lost.
            for name in ("safety_live_lock", "feed_yellowstone", "data_market_observations"):
                self.assertEqual(self._state_of(report, name), State.DATA_BLOCKED)

    def test_a_stale_but_present_snapshot_escalates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir(parents=True)
            path = root / "state" / "readiness.json"
            path.write_text(json.dumps(self._readiness()))
            os.utime(path, (self.NOW - 3_600, self.NOW - 3_600))
            report = run_health_checks(path, root / "models",
                                       root / "state" / "exec.jsonl", root, now=self.NOW)
            freshness = next(c for c in report.checks if c.name == "readiness_freshness")
            self.assertEqual(freshness.state, State.CRITICAL)
            self.assertTrue(freshness.escalate)

    def test_an_unlocked_live_path_is_surfaced_and_escalated(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(
                self._readiness(live_submission_locked=False,
                                execution={"dry_run": False}, mode="LIVE"), directory)
            check = next(c for c in report.checks if c.name == "safety_live_lock")
            self.assertEqual(check.state, State.WARN)
            self.assertTrue(check.escalate)

    def test_an_active_kill_switch_is_critical(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(
                portfolio={"kill_switch_active": True, "daily_pnl": -900.0,
                           "max_daily_loss": 1000.0}), directory)
            check = next(c for c in report.checks if c.name == "safety_kill_switch")
            self.assertEqual(check.state, State.CRITICAL)
            self.assertTrue(check.escalate)

    def test_a_dead_feed_is_critical(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(
                self._readiness(yellowstone={"status": "DISCONNECTED"}), directory)
            self.assertEqual(self._state_of(report, "feed_yellowstone"), State.CRITICAL)

    def test_a_feed_that_never_started_is_blocked_not_broken(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(
                self._readiness(yellowstone={"status": "NOT_STARTED"}), directory)
            self.assertEqual(self._state_of(report, "feed_yellowstone"), State.DATA_BLOCKED)

    def test_a_moat_that_stopped_growing_is_critical(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(
                dataset={"market_observed_at": self.NOW - 7_200}), directory)
            check = next(c for c in report.checks if c.name == "data_market_observations")
            self.assertEqual(check.state, State.CRITICAL)
            self.assertTrue(check.escalate)

    def test_an_untrained_model_is_blocked_not_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(prediction="DATA_BLOCKED"), directory)
            # Not yet trained is a known, expected state; it is not a defect.
            self.assertEqual(self._state_of(report, "model_prediction"), State.DATA_BLOCKED)

    def test_execution_failure_rate_needs_a_minimum_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            few = [{"timestamp": self.NOW - 10, "success": False} for _ in range(3)]
            report = self._run(self._readiness(), directory, execution_rows=few)
            check = next(c for c in report.checks if c.name == "execution_failure_rate")
            # Three failures out of four is four observations, not a 75%
            # failure rate, and acting on it retires a working route on noise.
            self.assertEqual(check.state, State.DATA_BLOCKED)

    def test_a_sustained_execution_failure_rate_is_critical(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = ([{"timestamp": self.NOW - 10, "success": False,
                      "error": "bundle_not_landed"} for _ in range(30)]
                    + [{"timestamp": self.NOW - 10, "success": True} for _ in range(10)])
            report = self._run(self._readiness(), directory, execution_rows=rows)
            check = next(c for c in report.checks if c.name == "execution_failure_rate")
            self.assertEqual(check.state, State.CRITICAL)
            self.assertEqual(check.evidence["top_reasons"]["bundle_not_landed"], 30)

    def test_old_execution_attempts_fall_outside_the_window(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"timestamp": self.NOW - 100_000, "success": False} for _ in range(50)]
            report = self._run(self._readiness(), directory, execution_rows=rows)
            self.assertEqual(self._state_of(report, "execution_failure_rate"),
                             State.DATA_BLOCKED)

    def test_degraded_sources_warn_without_halting(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(
                social={"data_status": {"telegram": "OK_PUSH",
                                        "youtube": "DATA_BLOCKED: quota"}}), directory)
            check = next(c for c in report.checks if c.name == "source_social")
            self.assertEqual(check.state, State.WARN)
            self.assertIn("youtube", check.evidence["degraded"])

    def test_decaying_champions_with_no_successor_warn(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(
                champions={"live_champions": 1, "decaying_champions": 2,
                           "shadow_models": 0}), directory)
            self.assertEqual(self._state_of(report, "promotion_pipeline"), State.WARN)

    def test_the_report_ranks_worst_state_and_lists_escalations(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(self._readiness(
                yellowstone={"status": "DISCONNECTED"}), directory)
            self.assertEqual(report.worst, State.CRITICAL)
            self.assertIn("feed_yellowstone", [c.name for c in report.escalations])
            payload = report.to_dict()
            self.assertEqual(payload["worst_state"], "CRITICAL")


class TestMonitorEntryPoint(unittest.TestCase):
    """Exit codes have to be actionable without parsing anything."""

    def _run(self, directory, readiness=None):
        root = Path(directory)
        (root / "data" / "state").mkdir(parents=True, exist_ok=True)
        if readiness is not None:
            (root / "data" / "state" / "readiness.json").write_text(json.dumps(readiness))
        return monitor_main(["--root", str(root), "--quiet"])

    def test_a_missing_snapshot_exits_critical(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self._run(directory), 2)

    def test_the_snapshot_is_written_atomically_and_history_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run(directory)
            root = Path(directory)
            health = json.loads((root / "data" / "state" / "health.json").read_text())
            self.assertIn("worst_state", health)
            # No temporary file is left behind for a reader to trip over.
            self.assertEqual(list((root / "data" / "state").glob("*.tmp")), [])
            history = (root / "data" / "state" / "health_history.jsonl").read_text()
            self.assertEqual(len(history.strip().splitlines()), 1)
            self._run(directory)
            history = (root / "data" / "state" / "health_history.jsonl").read_text()
            self.assertEqual(len(history.strip().splitlines()), 2)


class TestAuditPack(unittest.TestCase):
    """The point of the pack is what it leaves out."""

    def _health(self):
        return HealthReport(generated_at=1.0, checks=[
            Check("ok_check", State.OK, "fine"),
            Check("bad_check", State.CRITICAL, "broken", {"x": 1}, escalate=True),
        ])

    def test_missing_inputs_produce_stated_sections_not_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = build_audit_pack(Path(directory), run_tests=False)
            sections = pack.to_dict()["sections"]
            # Silently dropping an empty section makes the pack read as though
            # what remains is the whole picture.
            for name in ("wealth_leaks", "rug_defence", "execution", "edge_health",
                         "moat_growth", "system_integrity"):
                self.assertEqual(sections[name]["status"], "DATA_BLOCKED", name)
                self.assertTrue(sections[name]["summary"])

    def test_system_integrity_reports_only_what_is_not_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = build_audit_pack(Path(directory), health=self._health(), run_tests=False)
            section = pack.to_dict()["sections"]["system_integrity"]
            self.assertEqual(section["status"], "CRITICAL")
            self.assertEqual([e["check"] for e in section["entries"]], ["bad_check"])

    def test_truncation_is_recorded_rather_than_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            leaks = {"total_forgone_log_growth": 5.0,
                     "worst_tokens": [{"token": f"t{i}", "forgone_log_growth": 1.0}
                                      for i in range(40)]}
            pack = build_audit_pack(Path(directory), leak_report=leaks, run_tests=False)
            section = pack.to_dict()["sections"]["wealth_leaks"]
            self.assertEqual(len(section["entries"]), 10)
            # An audit that unknowingly sees the top 10 of 4,000 findings will
            # confidently rank the wrong work first.
            self.assertEqual(section["truncated_entries"], 30)

    def test_the_pack_is_capped_and_records_what_it_trimmed(self):
        with tempfile.TemporaryDirectory() as directory:
            leaks = {
                "total_forgone_log_growth": 5.0,
                "worst_tokens": [{"token": f"t{i}", "forgone_log_growth": 1.0,
                                  "detail": "y" * 300} for i in range(40)],
                "by_leak": {f"leak_{i}": float(i) for i in range(200)},
                "top_causes": [{"leak": "missed_monster", "reason": "z" * 300}
                               for _ in range(50)],
            }
            pack = build_audit_pack(Path(directory), leak_report=leaks, run_tests=False)
            text = pack.serialise(max_bytes=4_000)
            self.assertLessEqual(len(text.encode()), 4_000)
            payload = json.loads(text)
            # Details go before entries: a finding a reviewer cannot see at all
            # is worse than one they see without its supporting blob.
            self.assertTrue(payload["trimmed_to_fit"])
            self.assertIn("wealth_leaks.detail", payload["trimmed_to_fit"])

    def test_an_uncapped_pack_is_returned_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = build_audit_pack(Path(directory), run_tests=False)
            payload = json.loads(pack.serialise())
            self.assertNotIn("trimmed_to_fit", payload)

    def test_false_rug_alarms_are_included_alongside_rugs_entered(self):
        with tempfile.TemporaryDirectory() as directory:
            events = [
                {"token": "entered", "entered": True, "rugged": True,
                 "realized_multiple": 0.05},
                {"token": "rejected", "entered": False, "rugged": False,
                 "rejection_reason": "rug_risk_too_high", "max_feasible_multiple": 20.0},
            ]
            pack = build_audit_pack(Path(directory), rug_events=events, run_tests=False)
            section = pack.to_dict()["sections"]["rug_defence"]
            kinds = {entry["kind"] for entry in section["entries"]}
            # A detector tightened without a view of what it wrongly rejected
            # converges on refusing to trade.
            self.assertEqual(kinds, {"rug_entered", "false_alarm"})

    def test_edge_health_maps_decay_status_to_a_keep_kill_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            decay = {
                "creator_dna": {"status": "HEALTHY", "sample": 400},
                "telegram_lead": {"status": "DEGRADED", "sample": 400},
                "swarm": {"status": "WEAKENING", "sample": 400},
                "new_thing": {"status": "MEASURING", "sample": 4},
            }
            pack = build_audit_pack(Path(directory), decay=decay, run_tests=False)
            verdicts = {e["mechanism"]: e["verdict"]
                        for e in pack.to_dict()["sections"]["edge_health"]["entries"]}
            self.assertEqual(verdicts, {"creator_dna": "EXPAND", "telegram_lead": "HIBERNATE",
                                        "swarm": "SHADOW", "new_thing": "KEEP"})

    def test_recent_changes_reads_the_repository_history(self):
        # Resolved from this file rather than hardcoded: the absolute path
        # this test used to name exists on one machine, so on CI `git log` ran
        # with a cwd that did not exist and the pack correctly reported
        # DATA_BLOCKED. The pack was right and the test was wrong.
        pack = build_audit_pack(Path(__file__).resolve().parents[1],
                                period_days=3650, run_tests=False)
        section = pack.to_dict()["sections"]["recent_changes"]
        self.assertEqual(section["status"], "OK")
        self.assertIn("commits", section["summary"])


class TestMegaEventReserve(unittest.TestCase):
    """Detection is worthless if every dollar is committed when the event lands."""

    def test_a_quiet_week_withholds_almost_nothing(self):
        reserve = MegaEventReserve(baseline_fraction=0.0, max_fraction=0.35)
        decision = reserve.decide(event_probability=0.01)
        # A fixed cash buffer is a tax paid every week for an event that
        # happens twice a year.
        self.assertEqual(decision.reserve_fraction, 0.0)
        self.assertEqual(reserve.deployable_equity(10_000.0, decision), 10_000.0)

    def test_an_elevated_verified_event_withholds_a_lot(self):
        reserve = MegaEventReserve(max_fraction=0.35)
        decision = reserve.decide(event_probability=0.9, authenticated=True)
        self.assertGreater(decision.reserve_fraction, 0.25)
        self.assertLessEqual(decision.reserve_fraction, 0.35)

    def test_an_unverified_event_withholds_less_than_a_verified_one(self):
        reserve = MegaEventReserve(max_fraction=0.35)
        verified = reserve.decide(0.9, authenticated=True).reserve_fraction
        rumoured = reserve.decide(0.9, authenticated=False).reserve_fraction
        # Most viral stories never produce a token worth funding.
        self.assertLess(rumoured, verified)
        self.assertGreater(rumoured, 0.0)

    def test_an_unmeasured_probability_holds_only_the_baseline(self):
        reserve = MegaEventReserve(baseline_fraction=0.05, max_fraction=0.35)
        decision = reserve.decide(event_probability=None)
        self.assertEqual(decision.status, "DATA_BLOCKED")
        self.assertEqual(decision.reserve_fraction, 0.05)

    def test_the_reserve_never_exceeds_its_ceiling(self):
        reserve = MegaEventReserve(max_fraction=0.20)
        for probability in (0.5, 0.9, 1.0, 5.0):
            self.assertLessEqual(
                reserve.decide(probability, authenticated=True).reserve_fraction, 0.20)

    def test_the_reserve_only_ever_withholds(self):
        reserve = MegaEventReserve(max_fraction=0.35)
        for probability in (0.0, 0.3, 1.0):
            decision = reserve.decide(probability, authenticated=True)
            # There is no path that frees capital beyond the ordinary limits.
            self.assertLessEqual(reserve.deployable_equity(10_000.0, decision), 10_000.0)


class TestRemainingAudience(unittest.TestCase):
    """+10x and finished, versus +10x with 95% of the audience still ahead."""

    def test_early_reach_leaves_most_of_the_audience_ahead(self):
        report = remaining_audience([AudienceTier.TRENCHES])
        self.assertEqual(report.status, "OK")
        self.assertGreater(report.remaining_share, 0.9)
        self.assertFalse(report.exhausted)

    def test_broad_reach_is_exhausted(self):
        report = remaining_audience(list(AudienceTier))
        self.assertLess(report.remaining_share, 0.05)
        self.assertTrue(report.exhausted)

    def test_seeing_nothing_is_blocked_not_untouched(self):
        report = remaining_audience([])
        # "Nobody has heard of it" and "we have not looked" justify opposite
        # position sizes, and only one is knowable from having seen nothing.
        self.assertEqual(report.status, "DATA_BLOCKED")
        self.assertFalse(report.exhausted)

    def test_repeated_tiers_are_not_double_counted(self):
        once = remaining_audience([AudienceTier.TRENCHES]).remaining_share
        twice = remaining_audience(
            [AudienceTier.TRENCHES, AudienceTier.TRENCHES]).remaining_share
        self.assertAlmostEqual(once, twice)

    def test_the_prior_basis_is_stated_rather_than_implied(self):
        self.assertIn("prior-based", remaining_audience([AudienceTier.TRENCHES]).detail)


class TestCapacityEscalation(unittest.TestCase):
    """The life-changing trade rarely bet maximum size on the first observation."""

    LADDER = {"probe": 0.005, "authenticated": 0.02, "independent_demand": 0.05,
              "liquidity_expanding": 0.10, "mass_adoption": 0.20}

    def _plan(self, evidence, held=0.0, executable=1.0):
        return plan_capacity_escalation(evidence, held, self.LADDER, executable)

    def test_size_grows_only_as_evidence_is_proven(self):
        steps = []
        evidence = {}
        for _, requirement in ESCALATION_LADDER:
            evidence[requirement] = True
            steps.append(self._plan(dict(evidence)).target_fraction)
        self.assertEqual(steps, sorted(steps))
        self.assertAlmostEqual(steps[0], 0.005)
        self.assertAlmostEqual(steps[-1], 0.20)

    def test_the_ladder_is_strictly_ordered(self):
        """One impressive signal must not justify a large position."""
        plan = self._plan({"detected": True, "audience_still_ahead": True})
        # Skipping to the top rung on a single fact is the failure mode.
        self.assertEqual(plan.step, "probe")
        self.assertAlmostEqual(plan.target_fraction, 0.005)

    def test_no_evidence_means_no_position(self):
        plan = self._plan({})
        self.assertEqual(plan.step, "none")
        self.assertEqual(plan.target_fraction, 0.0)

    def test_the_target_is_capped_by_what_the_venue_can_absorb(self):
        plan = self._plan({req: True for _, req in ESCALATION_LADDER}, executable=0.03)
        # A size the market cannot fill is not a position, it is a slippage
        # estimate.
        self.assertAlmostEqual(plan.target_fraction, 0.03)
        self.assertTrue(plan.capacity_capped)

    def test_unmeasured_depth_blocks_rather_than_assuming_unlimited(self):
        plan = self._plan({req: True for _, req in ESCALATION_LADDER}, executable=None)
        self.assertEqual(plan.status, "DATA_BLOCKED")
        self.assertEqual(plan.target_fraction, 0.0)

    def test_escalation_never_shrinks_an_existing_position(self):
        plan = self._plan({"detected": True}, held=0.08)
        # Banking is the exit policy's decision; letting two components both
        # move size down is how a runner gets sold twice.
        self.assertAlmostEqual(plan.target_fraction, 0.08)


class TestMegaEventReserveWiring(unittest.IsolatedAsyncioTestCase):
    """Withheld capital must be invisible to every sizing ceiling at once."""

    def _desk(self, probability=None, authenticated=False):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), max_position_pct=0.05,
                             max_total_exposure_pct=0.30)
        desk = SimpleNamespace(
            elogw_engine=engine, dry_run=True, offline=False,
            wallet_equity_usd=0.0, sol_price_usd=0.0, equity_status="",
            total_pnl=0.0, global_config={"paper_equity_usd": 10_000.0},
            mega_event_reserve=MegaEventReserve(baseline_fraction=0.0, max_fraction=0.40),
            mega_event_probability=probability,
            mega_event_authenticated=authenticated,
            mega_event_reserve_state={},
            jupiter=SimpleNamespace(_session=object(),
                                    get_quote=lambda *a, **k: _async_value(
                                        SimpleNamespace(output_amount=150_000_000,
                                                        price_impact_pct=0.001))),
        )
        return desk

    async def test_a_quiet_week_deploys_the_whole_book(self):
        desk = self._desk(probability=0.0)
        await MemecoinQuantDesk._refresh_portfolio_state(desk)
        self.assertAlmostEqual(desk.elogw_engine.portfolio_value, desk.wallet_equity_usd)
        self.assertEqual(desk.mega_event_reserve_state["fraction"], 0.0)

    async def test_an_elevated_verified_event_shrinks_deployable_capital(self):
        desk = self._desk(probability=0.9, authenticated=True)
        await MemecoinQuantDesk._refresh_portfolio_state(desk)
        self.assertLess(desk.elogw_engine.portfolio_value, desk.wallet_equity_usd)
        # Reported equity is unchanged: the reserve withholds deployment, it
        # does not pretend the money is gone.
        self.assertGreater(desk.wallet_equity_usd, 0.0)
        self.assertGreater(desk.mega_event_reserve_state["fraction"], 0.25)

    async def test_every_ceiling_shrinks_together(self):
        """The reserve reaches all limits at once and cannot be forgotten by one."""
        quiet = self._desk(probability=0.0)
        armed = self._desk(probability=0.9, authenticated=True)
        await MemecoinQuantDesk._refresh_portfolio_state(quiet)
        await MemecoinQuantDesk._refresh_portfolio_state(armed)

        for engine in (quiet.elogw_engine, armed.elogw_engine):
            engine.max_position_usd = 1e12  # take the USD cap out of the way

        quiet_cap = quiet.elogw_engine.exposure_cap(1e12) * quiet.elogw_engine.portfolio_value
        armed_cap = armed.elogw_engine.exposure_cap(1e12) * armed.elogw_engine.portfolio_value
        self.assertLess(armed_cap, quiet_cap)

    async def test_an_unmeasured_event_holds_only_the_baseline(self):
        desk = self._desk(probability=None)
        await MemecoinQuantDesk._refresh_portfolio_state(desk)
        self.assertEqual(desk.mega_event_reserve_state["status"], "DATA_BLOCKED")
        # An unmeasured event must not silently tax the book every week.
        self.assertEqual(desk.mega_event_reserve_state["fraction"], 0.0)
        self.assertAlmostEqual(desk.elogw_engine.portfolio_value, desk.wallet_equity_usd)


class TestSourceDNA(unittest.TestCase):
    """"Does following it pay" and "does it predict the crowd" are different questions."""

    @staticmethod
    def _post_outcome(source="grp", token="t", posted=0.0, lag=0.5, returns=None,
                 flow=None, pre=None, sell=None, rugged=None, feasible=None,
                 edited=False, deleted=False):
        post = SourcePost(source_id=source, token=token, posted_at=posted,
                          observed_at=posted + lag, edited=edited, deleted=deleted)
        return PostOutcome(post=post, executable_returns=returns or {},
                           flow_acceleration=flow, pre_post_accumulation_usd=pre,
                           post_sell_usd=sell, rugged=rugged,
                           max_feasible_multiple=feasible)

    def test_a_thin_history_is_measuring_not_scored(self):
        dna = build_source_dna("grp", [self._post_outcome(returns={1.0: 0.5})
                                       for _ in range(6)])
        # A rate from six calls is noise, and promoting on it is how a lucky
        # channel gets capital.
        self.assertEqual(dna.status, "MEASURING")
        self.assertIsNone(dna.best_horizon_return)
        self.assertFalse(dna.tradeable_directly)

    def test_no_posts_at_all_is_blocked(self):
        self.assertEqual(build_source_dna("grp", []).status, "DATA_BLOCKED")

    def test_a_source_with_bad_calls_can_still_predict_flow(self):
        """The whole reason the two questions are kept apart."""
        outcomes = [self._post_outcome(returns={1.0: -0.20}, flow=3.0) for _ in range(30)]
        dna = build_source_dna("pump_group", outcomes)
        self.assertEqual(dna.status, "MEASURED")
        self.assertFalse(dna.tradeable_directly)
        self.assertTrue(dna.useful_as_flow_signal)

    def test_a_distributor_is_identified_by_its_shape(self):
        outcomes = [self._post_outcome(returns={1.0: -0.3}, flow=2.5,
                                  pre=5_000.0, sell=6_000.0) for _ in range(30)]
        dna = build_source_dna("insider_group", outcomes)
        # Linked wallets long beforehand and selling afterwards: excellent at
        # predicting flow, terrible to hold alongside.
        self.assertTrue(dna.is_distributor)
        self.assertTrue(dna.useful_as_flow_signal)
        self.assertFalse(dna.tradeable_directly)

    def test_accumulation_without_selling_is_not_distribution(self):
        outcomes = [self._post_outcome(returns={1.0: 0.4}, pre=5_000.0, sell=0.0)
                    for _ in range(30)]
        self.assertFalse(build_source_dna("early_group", outcomes).is_distributor)

    def test_the_best_horizon_is_the_one_measured_as_best(self):
        outcomes = [self._post_outcome(returns={0.5: 0.02, 3.0: 0.31, 30.0: 0.05})
                    for _ in range(30)]
        dna = build_source_dna("grp", outcomes)
        self.assertEqual(dna.best_horizon, 3.0)
        self.assertAlmostEqual(dna.best_horizon_return, 0.31)
        self.assertTrue(dna.tradeable_directly)

    def test_edit_and_delete_behaviour_is_recorded(self):
        outcomes = ([self._post_outcome(deleted=True) for _ in range(10)]
                    + [self._post_outcome() for _ in range(10)])
        self.assertAlmostEqual(build_source_dna("grp", outcomes).edit_delete_rate, 0.5)

    def test_a_slow_source_scores_below_a_fast_mediocre_one(self):
        excellent_but_late = build_source_dna(
            "late", [self._post_outcome(lag=30.0, returns={1.0: 0.50}) for _ in range(30)])
        mediocre_but_instant = build_source_dna(
            "fast", [self._post_outcome(lag=0.05, returns={1.0: 0.08}) for _ in range(30)])
        # A signal that reaches us after the move is not actionable however
        # accurate it was.
        self.assertLess(source_value(excellent_but_late),
                        source_value(mediocre_but_instant))

    def test_an_unmeasured_source_has_no_value_number(self):
        self.assertIsNone(source_value(build_source_dna("grp", [self._post_outcome()])))

    def test_ranking_separates_tradeable_from_flow_only(self):
        good = build_source_dna("good", [self._post_outcome(returns={1.0: 0.3}, flow=1.1)
                                         for _ in range(30)])
        flow = build_source_dna("flow", [self._post_outcome(returns={1.0: -0.2}, flow=3.0,
                                                       pre=100.0, sell=200.0)
                                         for _ in range(30)])
        thin = build_source_dna("thin", [self._post_outcome()])
        ranked = rank_sources([good, flow, thin])
        self.assertEqual([e["source"] for e in ranked["tradeable"]], ["good"])
        self.assertEqual([e["source"] for e in ranked["flow_signal_only"]], ["flow"])
        self.assertEqual(ranked["distributors"], ["flow"])
        self.assertEqual(ranked["measuring"], ["thin"])


class TestSourceGenealogy(unittest.TestCase):
    """Ranking sources by audience finds the repeaters."""

    @staticmethod
    def _post(source, token, at):
        return SourcePost(source_id=source, token=token, posted_at=at, observed_at=at)

    def _graph(self, launches=8):
        graph = SourceGenealogy(min_shared_tokens=5)
        for index in range(launches):
            base = index * 10_000.0
            # An obscure regional account, then the channel everyone watches.
            graph.record(self._post("regional", f"t{index}", base))
            graph.record(self._post("big_channel", f"t{index}", base + 45.0))
            graph.record(self._post("aggregator", f"t{index}", base + 120.0))
        return graph

    def test_the_first_publisher_is_identified_as_the_leader(self):
        pairs = self._graph().lead_lag()
        self.assertTrue(all(pair.leader != "aggregator" for pair in pairs))
        direct = next(pair for pair in pairs
                      if (pair.leader, pair.follower) == ("regional", "big_channel"))
        self.assertAlmostEqual(direct.median_lead_seconds, 45.0)
        self.assertEqual(direct.lead_rate, 1.0)

    def test_upstream_recursion_finds_the_earliest_node(self):
        graph = self._graph()
        upstream_of_big = graph.upstream_of("big_channel")
        self.assertEqual([item.leader for item in upstream_of_big], ["regional"])
        # Recurse: nothing publishes before the regional account, so it is the
        # earliest lawfully observable node.
        self.assertEqual(graph.upstream_of("regional"), [])

    def test_a_thin_overlap_is_not_a_lead_relationship(self):
        graph = SourceGenealogy(min_shared_tokens=5)
        graph.record(self._post("a", "t0", 0.0))
        graph.record(self._post("b", "t0", 10.0))
        self.assertEqual(graph.lead_lag(), [])

    def test_posts_far_apart_are_not_the_same_information_event(self):
        graph = SourceGenealogy(min_shared_tokens=3, max_lead_seconds=60.0)
        for index in range(6):
            base = index * 100_000.0
            graph.record(self._post("a", f"t{index}", base))
            graph.record(self._post("b", f"t{index}", base + 3_600.0))
        # Crediting an hour-later post as "following" would credit a
        # coincidence.
        self.assertEqual(graph.lead_lag(), [])

    def test_only_the_first_post_per_source_per_token_counts(self):
        graph = SourceGenealogy(min_shared_tokens=3)
        for index in range(5):
            base = index * 10_000.0
            graph.record(self._post("a", f"t{index}", base))
            graph.record(self._post("a", f"t{index}", base + 5.0))
            graph.record(self._post("b", f"t{index}", base + 20.0))
        pairs = graph.lead_lag()
        self.assertTrue(pairs)
        self.assertAlmostEqual(pairs[0].median_lead_seconds, 20.0)


class TestObservationLag(unittest.TestCase):
    def test_the_lag_is_publication_to_observation(self):
        post = SourcePost("s", "t", posted_at=100.0, observed_at=106.5)
        # The source's own delivery latency, which no local speed recovers.
        self.assertAlmostEqual(post.observation_lag, 6.5)

    def test_an_out_of_order_timestamp_never_reports_a_negative_lag(self):
        post = SourcePost("s", "t", posted_at=100.0, observed_at=99.0)
        self.assertEqual(post.observation_lag, 0.0)


class TestRustPythonQuoteParity(unittest.TestCase):
    """The Rust hot path and the Python path must price identically.

    Two implementations of the same curve is two chances to be wrong, and a
    divergence between them is the worst kind: the fast path fills at one
    price while every label, counterfactual and E[log W] in the research lake
    was computed against another. The fixture is generated by the Rust crate
    (`cargo run --example parity`) and checked against Python here, so a change
    to either side that moves a quote fails this test rather than silently
    splitting the two.
    """

    FIXTURE = Path(__file__).parent / "fixtures" / "rust_curve_parity.txt"

    @staticmethod
    def _curve(vt, vs, rt, rs):
        return BondingCurveState(
            virtual_token_reserves=int(vt), virtual_sol_reserves=int(vs),
            real_token_reserves=int(rt), real_sol_reserves=int(rs),
            token_total_supply=1_000_000_000_000_000, complete=False, creator="")

    def _rows(self):
        rows = []
        for line in self.FIXTURE.read_text().strip().splitlines():
            parts = line.split()
            rows.append((parts[0], *[int(value) for value in parts[1:]]))
        return rows

    def test_the_fixture_covers_both_sides_and_several_curve_states(self):
        rows = self._rows()
        self.assertGreaterEqual(len(rows), 20)
        self.assertEqual({row[0] for row in rows}, {"buy", "sell"})
        self.assertGreaterEqual(len({row[1:5] for row in rows}), 3)

    def test_every_rust_quote_matches_python_exactly(self):
        for side, vt, vs, rt, rs, amount, expected_out, expected_fee in self._rows():
            state = self._curve(vt, vs, rt, rs)
            quote = (quote_buy(state, amount) if side == "buy"
                     else quote_sell(state, amount))
            self.assertEqual(quote.data_status, "OK", (side, amount))
            # Exact, not approximate: both sides are integer arithmetic, so any
            # difference at all is a real divergence rather than rounding.
            self.assertEqual(quote.output_amount, expected_out,
                             f"{side} output diverged at amount={amount}")
            self.assertEqual(quote.fee_amount, expected_fee,
                             f"{side} fee diverged at amount={amount}")

    def test_the_shared_discriminator_is_the_same_constant_on_both_sides(self):
        # sha256("account:BondingCurve")[:8], recomputed rather than trusted.
        digest = hashlib.sha256(b"account:BondingCurve").digest()[:8]
        self.assertEqual(bytes(BONDING_CURVE_DISCRIMINATOR), digest)

    def test_the_pump_instruction_discriminators_are_recomputed_not_copied(self):
        for name, expected_len in (("buy_v2", 8), ("sell_v2", 8)):
            digest = hashlib.sha256(f"global:{name}".encode()).digest()[:8]
            self.assertEqual(len(digest), expected_len)
        self.assertNotEqual(hashlib.sha256(b"global:buy_v2").digest()[:8],
                            hashlib.sha256(b"global:sell_v2").digest()[:8])


class TestActionValuePolicy(unittest.TestCase):
    """A monster detector and an unrelated exit rule is how a 30x sells at 2x.

    These tests pin that every action prices the SAME forward distribution, so
    two components cannot disagree about a number they both read from one
    place.
    """

    @staticmethod
    def _state(**kwargs):
        base = dict(
            held_fraction=0.05, current_multiple=3.0,
            forward_bins=((0.55, -0.40), (0.30, 1.00), (0.15, 6.00)),
            exit_cost=0.02, entry_cost=0.02, exit_capacity_ratio=1.0,
            # Explicitly measured as fully escapable. Both this and capacity
            # are required inputs -- there is no permissive default.
            escape_probability=1.0,
        )
        base.update(kwargs)
        return PositionState(**base)

    def _policy(self, **kwargs):
        return ActionValuePolicy(**kwargs)

    def test_holding_is_the_baseline_and_scores_exactly_zero(self):
        decision = self._policy().score(self._state())
        self.assertEqual(decision.status, "OK")
        self.assertEqual(decision.score_of(Action.HOLD), 0.0)

    def test_a_strong_forward_distribution_holds(self):
        decision = self._policy().score(self._state(
            forward_bins=((0.20, -0.30), (0.40, 2.00), (0.40, 9.00))))
        self.assertEqual(decision.action, Action.HOLD)
        # Nothing about the current multiple appears in that verdict.
        for action in (Action.BANK_50, Action.BANK_75, Action.EXIT):
            self.assertLess(decision.score_of(action), 0.0)

    def test_a_collapsed_forward_distribution_banks_or_exits(self):
        decision = self._policy().score(self._state(
            forward_bins=((0.90, -0.60), (0.09, 0.10), (0.01, 1.00))))
        self.assertIn(decision.action,
                      {Action.BANK_50, Action.BANK_75, Action.EXIT})
        self.assertGreater(decision.q, 0.0)

    def test_the_same_position_at_thirty_x_still_holds_on_a_strong_distribution(self):
        """The failure this whole module exists to prevent."""
        low = self._policy().score(self._state(
            current_multiple=1.2,
            forward_bins=((0.30, -0.40), (0.30, 3.00), (0.40, 20.00))))
        high = self._policy().score(self._state(
            current_multiple=30.0,
            forward_bins=((0.30, -0.40), (0.30, 3.00), (0.40, 20.00))))
        # A ratchet sells the 30x hardest. This does not, because the multiple
        # is not an input to the decision -- the forward distribution is.
        self.assertEqual(low.action, Action.HOLD)
        self.assertEqual(high.action, Action.HOLD)

    def test_banking_is_never_free(self):
        """A free bank always beats holding and turns a runner into fees."""
        costless = self._policy().score(self._state(exit_cost=0.0))
        costly = self._policy().score(self._state(exit_cost=0.05))
        self.assertGreater(costless.score_of(Action.BANK_50),
                           costly.score_of(Action.BANK_50))
        # And the sold slice stops participating in the upside, so even at
        # zero cost banking is not automatically better than holding.
        self.assertLess(costless.score_of(Action.BANK_50), 0.0)

    def test_upside_that_cannot_be_sold_is_not_priced_as_upside(self):
        policy = self._policy()
        liquid = policy._hold_value(self._state(exit_capacity_ratio=1.0))
        trapped = policy._hold_value(self._state(exit_capacity_ratio=0.05))
        # Every action reads the same capacity, so none of them can be priced
        # on a return the position could not have realised.
        self.assertGreater(liquid, trapped)

        # And a trapped position is more willing to leave than a liquid one.
        liquid_exit = policy.score(self._state(exit_capacity_ratio=1.0))
        trapped_exit = policy.score(self._state(exit_capacity_ratio=0.05))
        self.assertGreater(trapped_exit.score_of(Action.EXIT),
                           liquid_exit.score_of(Action.EXIT))

    def test_a_low_escape_probability_discounts_the_hold(self):
        safe = self._policy()._hold_value(self._state(escape_probability=0.95))
        trapped = self._policy()._hold_value(self._state(escape_probability=0.02))
        self.assertGreater(safe, trapped)

    def test_unmeasured_escape_blocks_rather_than_reading_as_fully_escapable(self):
        """The most flattering assumption available in this module."""
        decision = self._policy().score(self._state(escape_probability=None))
        # Reading unknown escape as 1.0 makes every trapped position look
        # liquid, most strongly on the tokens where escape is hardest to
        # measure -- which is exactly where it matters.
        self.assertEqual(decision.status, "DATA_BLOCKED")
        self.assertIn("escape", decision.detail)

    def test_replace_is_scored_when_a_candidate_is_named(self):
        unnamed = self._policy().score(self._state())
        self.assertIsNone(unnamed.score_of(Action.REPLACE))

        named = self._policy().score(self._state(
            forward_bins=((0.85, -0.50), (0.15, 0.20)),
            replacement_fraction=0.05,
            replacement_bins=((0.20, -0.30), (0.40, 2.00), (0.40, 9.00))))
        # REPLACE existed in the enum and was never scored, so the allocator's
        # displacement path and this policy could disagree about one pair.
        self.assertIsNotNone(named.score_of(Action.REPLACE))
        self.assertEqual(named.action, Action.REPLACE)

    def test_replace_pays_the_new_candidates_entry_cost(self):
        cheap = self._policy().score(self._state(
            entry_cost=0.0, replacement_fraction=0.05,
            replacement_bins=((0.30, -0.30), (0.70, 3.00))))
        dear = self._policy().score(self._state(
            entry_cost=0.10, replacement_fraction=0.05,
            replacement_bins=((0.30, -0.30), (0.70, 3.00))))
        # Otherwise REPLACE is a free upgrade over EXIT.
        self.assertGreater(cheap.score_of(Action.REPLACE), dear.score_of(Action.REPLACE))

    def test_a_replacement_beyond_available_capital_is_refused(self):
        # Nearly all capital is in the position and it is down: the exit frees
        # 0.45 and only 0.10 cash exists beside it, so a 0.99 replacement is
        # not fundable at any price.
        decision = self._policy().score(self._state(
            held_fraction=0.90, current_multiple=0.5, replacement_fraction=0.99,
            replacement_bins=((1.0, 5.0),)))
        self.assertIsNone(decision.score_of(Action.REPLACE))

    def test_a_loaded_model_is_actually_consulted(self):
        class Model:
            def predict(self, state):
                return {action: 0.0 for action in Action} | {Action.BANK_50: 5.0}

        policy = self._policy()
        self.assertTrue(policy.load_model(Model(), "v7"))
        decision = policy.score(self._state())
        # A model marked loaded but never consulted reports trained: true
        # while every decision stays analytic, so the promotion ladder can
        # advance on a model that has never priced a trade.
        self.assertEqual(decision.action, Action.BANK_50)
        self.assertIn("v7", decision.detail)

    def test_a_model_that_omits_an_action_falls_back_rather_than_half_answering(self):
        class Partial:
            def predict(self, state):
                return {Action.HOLD: 0.0, Action.ADD: 1.0}

        policy = self._policy()
        policy.load_model(Partial(), "v8")
        decision = policy.score(self._state())
        # A Q table missing EXIT would silently make exiting impossible.
        self.assertNotIn("v8", decision.detail)
        self.assertEqual(decision.score_of(Action.HOLD), 0.0)

    def test_a_model_that_raises_falls_back_to_analytic_scoring(self):
        class Broken:
            def predict(self, state):
                raise RuntimeError("model is corrupt")

        policy = self._policy()
        policy.load_model(Broken(), "v9")
        decision = policy.score(self._state())
        self.assertEqual(decision.status, "OK")
        self.assertNotIn("v9", decision.detail)

    def test_adding_needs_a_size_and_respects_the_capacity_ceiling(self):
        no_size = self._policy().score(self._state())
        self.assertIsNone(no_size.score_of(Action.ADD))

        sized = self._policy().score(self._state(
            add_fraction=0.01, add_capacity_fraction=0.02,
            forward_bins=((0.20, -0.30), (0.40, 2.00), (0.40, 9.00))))
        self.assertIsNotNone(sized.score_of(Action.ADD))

        over = self._policy().score(self._state(
            add_fraction=0.05, add_capacity_fraction=0.001))
        self.assertIsNone(over.score_of(Action.ADD))

    def test_adding_wins_only_on_a_strong_distribution(self):
        strong = self._policy().score(self._state(
            add_fraction=0.02, add_capacity_fraction=0.05, held_fraction=0.01,
            current_multiple=1.1,
            forward_bins=((0.15, -0.30), (0.35, 3.00), (0.50, 12.00))))
        weak = self._policy().score(self._state(
            add_fraction=0.02, add_capacity_fraction=0.05, held_fraction=0.01,
            current_multiple=1.1,
            forward_bins=((0.85, -0.50), (0.14, 0.20), (0.01, 1.00))))
        self.assertEqual(strong.action, Action.ADD)
        self.assertLess(weak.score_of(Action.ADD), 0.0)

    def test_opportunity_cost_can_take_the_capital(self):
        patient = self._policy().score(self._state(
            alternative_growth_per_second=0.0, expected_remaining_seconds=300.0))
        contested = self._policy().score(self._state(
            alternative_growth_per_second=0.05, expected_remaining_seconds=300.0))
        self.assertGreater(contested.score_of(Action.EXIT),
                           patient.score_of(Action.EXIT))
        self.assertEqual(contested.action, Action.EXIT)

    def test_reentry_is_scored_on_its_own_distribution(self):
        # Still holding: re-entry is not the question being asked.
        open_position = self._policy().score(self._state(
            add_fraction=0.02, reentry_bins=((0.5, -0.3), (0.5, 4.0))))
        self.assertIsNone(open_position.score_of(Action.REENTER))

        flat = self._policy().score(self._state(
            held_fraction=0.0, current_multiple=1.0, add_fraction=0.02,
            forward_bins=((1.0, 0.0),),
            reentry_bins=((0.30, -0.40), (0.30, 2.00), (0.40, 8.00))))
        self.assertEqual(flat.action, Action.REENTER)
        self.assertGreater(flat.score_of(Action.REENTER), 0.0)

    def test_a_weak_second_wave_does_not_re_enter(self):
        flat = self._policy().score(self._state(
            held_fraction=0.0, current_multiple=1.0, add_fraction=0.02,
            forward_bins=((1.0, 0.0),),
            reentry_bins=((0.90, -0.50), (0.10, 0.30))))
        # A flat book does nothing by IGNORING, which is the same decision as
        # HOLD seen from zero exposure -- and recording which one it was is
        # what makes a rejected launch scoreable against what it went on to do.
        self.assertEqual(flat.action, Action.IGNORE)

    def test_hold_wins_ties_and_anything_inside_the_noise_margin(self):
        policy = self._policy(min_edge=0.01)
        # A distribution where banking is a hair better than holding.
        decision = policy.score(self._state(
            exit_cost=0.0, forward_bins=((0.500001, -0.0001), (0.499999, 0.0001))))
        self.assertEqual(decision.action, Action.HOLD)

    def test_a_malformed_distribution_blocks_rather_than_renormalising(self):
        for bins in ((), ((0.5, 1.0), (0.2, 2.0))):
            decision = self._policy().score(self._state(forward_bins=bins))
            # Quietly renormalising a distribution that does not sum to one
            # hides whichever producer is dropping mass.
            self.assertEqual(decision.status, "DATA_BLOCKED")

    def test_unmeasured_exit_capacity_blocks_every_action(self):
        decision = self._policy().score(self._state(exit_capacity_ratio=None))
        self.assertEqual(decision.status, "DATA_BLOCKED")
        self.assertIn("capacity", decision.detail)

    def test_bank_fractions_and_capital_release_are_declared_on_the_action(self):
        self.assertEqual(Action.BANK_25.bank_fraction, 0.25)
        self.assertEqual(Action.EXIT.bank_fraction, 1.0)
        self.assertEqual(Action.HOLD.bank_fraction, 0.0)
        # A partial bank frees cash but keeps the slot; only a full exit
        # releases it, which is what the opportunity-cost term prices.
        self.assertFalse(Action.BANK_75.frees_capital)
        self.assertTrue(Action.EXIT.frees_capital)
        self.assertTrue(Action.REPLACE.frees_capital)

    def test_a_model_without_predict_is_refused_at_load(self):
        policy = self._policy()
        self.assertFalse(policy.load_model(object(), "v1"))
        self.assertFalse(policy.is_trained)


class TestDeskWiringIsComplete(unittest.TestCase):
    """Catch a component that was built but never actually attached.

    Every desk collaborator is used inside a method, so a missing import or a
    missing `__init__` assignment does not fail at import time -- it fails on
    the VPS, at the moment the code path first runs. And because the unit
    tests bind their own fakes onto a SimpleNamespace desk, they cannot catch
    it either: they exercise the logic while the real desk has no such
    attribute. Both of those held simultaneously here, and four components
    were reported as wired while their imports and constructor lines had
    silently failed to apply.

    These two checks close that gap from opposite sides.
    """

    def test_no_name_used_in_main_is_undefined(self):
        """A static check that a runtime-only NameError cannot hide behind."""
        import builtins

        source = Path(__file__).resolve().parents[1] / "src" / "main.py"
        tree = ast.parse(source.read_text())
        defined = set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)

        used = {node.id for node in ast.walk(tree)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        undefined = sorted(name for name in used - defined if name[0].isupper())
        self.assertEqual(undefined, [], f"names used but never imported: {undefined}")

    def test_every_wired_component_is_constructed_on_the_real_desk(self):
        """The fakes prove the logic; this proves the desk actually has them."""
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        # Each of these was reported as wired at some point; each is only
        # reachable if it is both imported and assigned in __init__.
        for attribute in (
            "action_policy", "wallet_independence", "independence_report",
            "edge_decay", "hot_state", "mega_event_reserve", "opportunity_allocator",
            "distribution_detector", "monster_machine", "min_escape_probability",
        ):
            self.assertIn(f"self.{attribute} = ", source,
                          f"{attribute} is used but never constructed on the desk")

    def test_the_readiness_surface_exposes_the_new_components(self):
        """A component absent from readiness is invisible to the monitor."""
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        for key in ("action_policy", "actor_graph", "hot_state", "mega_event_reserve"):
            self.assertIn(f'"{key}":', source,
                          f"{key} is not reported in readiness, so nothing can monitor it")


class TestLifecycleReplay(unittest.TestCase):
    """A launch reduced to one row with a final return throws away the question.

    "Did this token go up" is not the question. "At each instant, what could we
    have bought, how much, and what could we then actually have sold" is, and
    only the second one has money in it.
    """

    T0 = 1_800_000_000.0

    def _lifecycle(self, marks=None, **kwargs):
        marks = marks if marks is not None else [
            Mark(self.T0 + 0.0, 1.0, executable_sol=5.0),
            Mark(self.T0 + 0.5, 1.4, executable_sol=5.0),
            Mark(self.T0 + 2.0, 3.0, executable_sol=5.0),
            Mark(self.T0 + 10.0, 8.0, executable_sol=5.0),
            Mark(self.T0 + 60.0, 2.0, executable_sol=5.0),
        ]
        return Lifecycle(token="t", created_at=self.T0, marks=marks, **kwargs)

    def test_entry_fills_at_the_last_mark_at_or_before_the_delay(self):
        life = self._lifecycle()
        # A buyer at T+1s cannot fill at a price that printed at T+2s, however
        # much closer it is.
        self.assertAlmostEqual(life.entry_mark(1.0).multiple, 1.4)
        self.assertAlmostEqual(life.entry_mark(0.0).multiple, 1.0)
        self.assertAlmostEqual(life.entry_mark(5.0).multiple, 3.0)

    def test_arriving_before_the_first_mark_is_blocked(self):
        life = Lifecycle("t", self.T0, [Mark(self.T0 + 5.0, 1.0, executable_sol=1.0)])
        cell = replay_cell(life, 0.0, 1.0, "hold", hold_to_end)
        self.assertEqual(cell.status, "DATA_BLOCKED")
        self.assertFalse(cell.ok)

    def test_being_earlier_is_worth_more_on_a_rising_launch(self):
        life = self._lifecycle()
        early = replay_cell(life, 0.0, 1.0, "hold", hold_to_end)
        late = replay_cell(life, 5.0, 1.0, "hold", hold_to_end)
        self.assertGreater(early.net_sol, late.net_sol)

    def test_a_fill_is_capped_by_observed_depth(self):
        thin = self._lifecycle(marks=[
            Mark(self.T0, 1.0, executable_sol=0.2),
            Mark(self.T0 + 10, 20.0, executable_sol=0.2),
        ])
        cell = replay_cell(thin, 0.0, 5.0, "hold", hold_to_end)
        # A theoretical 20x on size that could never have been filled is not a
        # 20x, and crediting it ranks the thinnest tokens highest.
        self.assertAlmostEqual(cell.filled_sol, 0.2)
        self.assertLess(cell.net_sol, 5.0)

    def test_exit_is_credited_only_for_what_the_curve_could_absorb(self):
        deep_in_thin_out = self._lifecycle(marks=[
            Mark(self.T0, 1.0, executable_sol=5.0),
            Mark(self.T0 + 10, 20.0, executable_sol=0.1),
        ])
        liquid = self._lifecycle(marks=[
            Mark(self.T0, 1.0, executable_sol=5.0),
            Mark(self.T0 + 10, 20.0, executable_sol=5.0),
        ])
        trapped = replay_cell(deep_in_thin_out, 0.0, 5.0, "hold", hold_to_end)
        clean = replay_cell(liquid, 0.0, 5.0, "hold", hold_to_end)
        self.assertLess(trapped.net_sol, clean.net_sol)

    def test_unobserved_depth_blocks_rather_than_assuming_liquidity(self):
        unknown = self._lifecycle(marks=[
            Mark(self.T0, 1.0, executable_sol=None),
            Mark(self.T0 + 10, 20.0, executable_sol=1.0),
        ])
        cell = replay_cell(unknown, 0.0, 1.0, "hold", hold_to_end)
        self.assertEqual(cell.status, "DATA_BLOCKED")
        self.assertIn("depth", cell.detail)

    def test_replay_is_point_in_time(self):
        """Checked structurally, not merely intended."""
        life = self._lifecycle()
        before = replay_cell(life, 1.0, 1.0, "hold", hold_to_end)

        contaminated = self._lifecycle(marks=life.marks + [
            Mark(self.T0 + 0.9, 99.0, executable_sol=99.0),
        ])
        # A mark at T+0.9s IS visible to a T+1s decision, so this one must
        # move the result -- proving the boundary is real and not vacuous.
        self.assertNotEqual(
            replay_cell(contaminated, 1.0, 1.0, "hold", hold_to_end).entry_multiple,
            before.entry_multiple)

        future = self._lifecycle(marks=life.marks + [
            Mark(self.T0 + 0.99, 1.4, executable_sol=5.0),
        ])
        entry_only = replay_cell(future, 0.5, 1.0, "hold", hold_to_end)
        # A mark after the delay must not change the ENTRY.
        self.assertAlmostEqual(entry_only.entry_multiple, 1.4)

    def test_exit_rules_are_pure_and_replay_deterministically(self):
        life = self._lifecycle()
        first = replay_lifecycle(life, delays=(0.0,), sizes=(1.0,))
        second = replay_lifecycle(life, delays=(0.0,), sizes=(1.0,))
        self.assertEqual([cell.net_sol for cell in first],
                         [cell.net_sol for cell in second])

    def test_a_take_profit_caps_the_runner_and_the_grid_shows_it(self):
        life = self._lifecycle()
        held = replay_cell(life, 0.0, 1.0, "hold", DEFAULT_EXIT_RULES["hold"])
        capped = replay_cell(life, 0.0, 1.0, "tp_2x", DEFAULT_EXIT_RULES["tp_2x"])
        self.assertGreater(capped.exit_multiple, held.exit_multiple)
        self.assertLess(capped.tail_capture, 1.0)

    def test_tail_capture_is_against_maximum_feasible(self):
        life = self._lifecycle()
        cell = replay_cell(life, 0.0, 1.0, "tp_2x", DEFAULT_EXIT_RULES["tp_2x"])
        self.assertAlmostEqual(cell.max_feasible_multiple, 8.0)
        self.assertAlmostEqual(cell.tail_capture, cell.exit_multiple / 8.0)

    def test_the_grid_covers_every_combination(self):
        cells = replay_lifecycle(self._lifecycle(), delays=(0.0, 1.0),
                                 sizes=(0.1, 1.0), exit_rules=DEFAULT_EXIT_RULES)
        self.assertEqual(len(cells), 2 * 2 * len(DEFAULT_EXIT_RULES))

    def test_the_default_grid_starts_below_one_second(self):
        # On a newborn launch the edge is won or lost sub-second; a grid that
        # starts at 1s cannot see it.
        self.assertLess(min(DEFAULT_DELAYS_S), 0.1)
        self.assertIn(0.0, DEFAULT_DELAYS_S)


class TestSniperScoreboard(unittest.TestCase):
    """Blocked cells are excluded, not scored zero."""

    def _cells(self):
        return [
            Cell("a", 0.0, 1.0, "hold", "OK", entry_multiple=1.0, exit_multiple=8.0,
                 filled_sol=1.0, net_sol=6.8, max_feasible_multiple=10.0),
            Cell("b", 1.0, 1.0, "hold", "OK", entry_multiple=1.0, exit_multiple=2.0,
                 filled_sol=1.0, net_sol=0.9, max_feasible_multiple=10.0),
            Cell("c", 0.0, 1.0, "hold", "DATA_BLOCKED", detail="no depth"),
        ]

    def test_blocked_cells_are_excluded_and_counted(self):
        board = sniper_scoreboard(self._cells(), launches_observed=3)
        self.assertEqual(board["priced"], 2)
        self.assertEqual(board["blocked"], 1)
        # Scoring it zero would make a launch nobody could trade look like one
        # that was traded and broke even.
        self.assertAlmostEqual(board["net_sol_per_priced_cell"], (6.8 + 0.9) / 2)

    def test_nothing_priceable_blocks_the_whole_board(self):
        blocked = [Cell("a", 0.0, 1.0, "hold", "DATA_BLOCKED")]
        self.assertEqual(sniper_scoreboard(blocked, 1)["status"], "DATA_BLOCKED")

    def test_opportunity_extraction_is_per_launch_observed(self):
        board = sniper_scoreboard(self._cells(), launches_observed=100)
        self.assertAlmostEqual(board["net_sol_per_100_launches"], 6.8 + 0.9)

    def test_the_board_reports_the_share_of_tens_actually_captured(self):
        board = sniper_scoreboard(self._cells(), launches_observed=3)
        # One of the two 10x-feasible cells exited above 5x.
        self.assertAlmostEqual(board["share_of_10x_captured_above_5x"], 0.5)

    def test_the_board_is_not_a_hedge_fund_report(self):
        board = sniper_scoreboard(self._cells(), launches_observed=3)
        for absent in ("sharpe", "sortino", "annualised_return", "volatility"):
            self.assertNotIn(absent, board)
        for present in ("net_sol_per_100_launches", "tail_capture_mean",
                        "net_sol_by_delay", "net_sol_by_size"):
            self.assertIn(present, board)

    def test_delay_decay_shows_where_the_edge_goes(self):
        cells = [
            Cell("a", 0.0, 1.0, "hold", "OK", net_sol=1.0, exit_multiple=2.0,
                 max_feasible_multiple=2.0),
            Cell("a", 1.0, 1.0, "hold", "OK", net_sol=0.4, exit_multiple=1.4,
                 max_feasible_multiple=2.0),
            Cell("a", 10.0, 1.0, "hold", "OK", net_sol=0.05, exit_multiple=1.05,
                 max_feasible_multiple=2.0),
        ]
        decay = delay_decay(sniper_scoreboard(cells, 1))
        self.assertAlmostEqual(decay[0.0], 1.0)
        self.assertAlmostEqual(decay[1.0], 0.4)
        self.assertLess(decay[10.0], decay[1.0])

    def test_delay_decay_refuses_a_ratio_against_a_non_positive_base(self):
        cells = [Cell("a", 0.0, 1.0, "hold", "OK", net_sol=-1.0),
                 Cell("a", 1.0, 1.0, "hold", "OK", net_sol=-0.5)]
        # With no positive edge at T0 there is nothing to decay from.
        self.assertIsNone(delay_decay(sniper_scoreboard(cells, 1)))


class TestLifecycleFromEpisode(unittest.TestCase):
    def test_an_episode_without_observations_yields_nothing(self):
        self.assertIsNone(lifecycle_from_episode(
            {"token": "t", "created_at": 1.0, "market_observations": []}))

    def test_observations_that_priced_nothing_contribute_nothing(self):
        life = lifecycle_from_episode({
            "token": "t", "created_at": 100.0,
            "market_observations": [
                {"timestamp": 101.0, "price_multiple": 2.0, "executable_sol": 1.0},
                {"timestamp": 102.0},
                {"timestamp": 103.0, "price_multiple": 0.0},
                {"timestamp": 104.0, "price_multiple": "not a number"},
            ],
        })
        # A blank observation contributes nothing rather than a zero.
        self.assertEqual(len(life.marks), 1)
        self.assertAlmostEqual(life.marks[0].multiple, 2.0)

    def test_marks_are_sorted_and_outcome_is_carried(self):
        life = lifecycle_from_episode({
            "token": "t", "created_at": 100.0,
            "market_observations": [
                {"timestamp": 105.0, "price_multiple": 3.0},
                {"timestamp": 101.0, "price_multiple": 1.0},
            ],
            "final_outcome": {"migrated": True, "rugged": True, "rug_time": 42.0},
        })
        self.assertEqual([mark.timestamp for mark in life.marks], [101.0, 105.0])
        self.assertTrue(life.migrated and life.rugged)
        self.assertAlmostEqual(life.rug_time, 42.0)

    def test_depth_is_none_where_it_was_not_recorded(self):
        life = lifecycle_from_episode({
            "token": "t", "created_at": 100.0,
            "market_observations": [{"timestamp": 101.0, "price_multiple": 2.0}],
        })
        self.assertIsNone(life.marks[0].executable_sol)


class TestActionValueTrainerGate(unittest.TestCase):
    """The gate exists to refuse the trade a search will otherwise make.

    Raise the win rate, cut the drawdown, quietly stop holding the trades that
    pay for everything. Every aggregate improves while the book gets worse.
    """

    @staticmethod
    def _metrics(launches=100, growth=0.05, tail=0.80, premature=0.10, monsters=25):
        return PolicyMetrics(
            launches=launches, priced_cells=launches, mean_net_sol=1.0,
            mean_log_growth=growth, tail_capture_on_monsters=tail,
            premature_exit_rate=premature, monster_launches=monsters)

    def test_a_genuine_improvement_passes(self):
        gate = tail_preservation_gate(
            self._metrics(growth=0.05, tail=0.80, premature=0.10),
            self._metrics(growth=0.08, tail=0.82, premature=0.08))
        self.assertTrue(gate.passed, gate.reasons)

    def test_better_growth_that_kills_the_tail_is_rejected(self):
        """The exact trade the gate exists to refuse."""
        gate = tail_preservation_gate(
            self._metrics(growth=0.05, tail=0.80, premature=0.10),
            self._metrics(growth=0.20, tail=0.30, premature=0.10))
        self.assertFalse(gate.passed)
        self.assertTrue(any("tail capture fell" in reason for reason in gate.reasons))

    def test_no_aggregate_score_buys_past_the_tail_check(self):
        gate = tail_preservation_gate(
            self._metrics(growth=0.05, tail=0.90, premature=0.05),
            self._metrics(growth=99.0, tail=0.10, premature=0.90))
        self.assertFalse(gate.passed)

    def test_more_premature_exits_are_rejected_even_at_stable_tail_capture(self):
        gate = tail_preservation_gate(
            self._metrics(growth=0.05, tail=0.80, premature=0.05),
            self._metrics(growth=0.09, tail=0.79, premature=0.40))
        self.assertFalse(gate.passed)
        self.assertTrue(any("premature exits rose" in reason for reason in gate.reasons))

    def test_growth_that_did_not_improve_is_rejected(self):
        gate = tail_preservation_gate(
            self._metrics(growth=0.05), self._metrics(growth=0.05))
        self.assertFalse(gate.passed)
        self.assertTrue(any("did not improve" in reason for reason in gate.reasons))

    def test_a_thin_out_of_sample_set_is_a_sample_not_a_measurement(self):
        gate = tail_preservation_gate(
            self._metrics(launches=5), self._metrics(launches=5, growth=0.5))
        self.assertFalse(gate.passed)
        self.assertTrue(any("not a measurement" in reason for reason in gate.reasons))

    def test_no_monster_evidence_is_unproven_not_proven(self):
        """A candidate that met no monsters has shown nothing about the tail."""
        gate = tail_preservation_gate(
            self._metrics(tail=None, monsters=0),
            self._metrics(growth=0.20, tail=None, monsters=0))
        self.assertFalse(gate.passed)
        self.assertTrue(any("unproven, not proven" in reason for reason in gate.reasons))


class TestActionValueTrainerSelection(unittest.TestCase):
    T0 = 1_800_000_000.0

    def _monster(self, index, exit_early: bool):
        """A launch where a 20x was feasible."""
        marks = [
            Mark(self.T0 + index * 1_000 + 0.0, 1.0, executable_sol=5.0),
            Mark(self.T0 + index * 1_000 + 5.0, 2.0, executable_sol=5.0),
            Mark(self.T0 + index * 1_000 + 30.0, 20.0, executable_sol=5.0),
            Mark(self.T0 + index * 1_000 + 90.0, 12.0, executable_sol=5.0),
        ]
        return Lifecycle(f"m{index}", self.T0 + index * 1_000, marks)

    def test_chronological_split_orders_by_launch_time(self):
        lives = [self._monster(index, False) for index in range(10)]
        train, oos = chronological_split(list(reversed(lives)), 0.7)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(oos), 3)
        # Whole launches, ordered: a random split puts the same regime on both
        # sides and every candidate looks like it generalises.
        self.assertLess(max(life.created_at for life in train),
                        min(life.created_at for life in oos))

    def test_a_take_profit_that_caps_monsters_is_rejected(self):
        lives = [self._monster(index, True) for index in range(60)]
        gate, report = evaluate_candidate(
            lives, "hold", "tp_2x", DEFAULT_EXIT_RULES)
        self.assertFalse(gate.passed)
        self.assertEqual(report["oos_launches"], len(lives) - int(len(lives) * 0.7))

    def test_shipping_nothing_is_a_valid_outcome(self):
        lives = [self._monster(index, True) for index in range(60)]
        result = select_policy(lives, DEFAULT_EXIT_RULES, "hold")
        self.assertEqual(result["status"], "OK")
        # A search that always finds a winner has found overfitting.
        self.assertIsNone(result["shipped"])
        self.assertTrue(result["evaluated"])

    def test_too_few_launches_blocks_rather_than_shipping(self):
        lives = [self._monster(index, True) for index in range(5)]
        result = select_policy(lives, DEFAULT_EXIT_RULES, "hold")
        self.assertEqual(result["status"], "DATA_BLOCKED")
        self.assertIsNone(result["shipped"])

    def test_growth_is_measured_in_log_wealth_not_profit(self):
        cells = [Cell("a", 0.0, 1.0, "hold", "OK", net_sol=1.0, filled_sol=1.0,
                      exit_multiple=2.0, max_feasible_multiple=2.0)]
        metrics = measure_policy(cells, 1)
        # log(1 + 1/1) = log 2. Summing profit would rank a policy that risks
        # everything above one that risks a sensible fraction.
        self.assertAlmostEqual(metrics.mean_log_growth, math.log(2.0))

    def test_a_rejected_run_is_still_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_report(Path(directory), {"status": "OK", "shipped": None})
            saved = json.loads(path.read_text())
            # The more informative record: it stops the same candidate being
            # re-proposed next week as though it were new.
            self.assertIsNone(saved["shipped"])
            self.assertIn("generated_at", saved)


class TestBackfillFactory(unittest.TestCase):
    """Backfill buys history. It must never buy the illusion of observation."""

    T0 = 1_700_000_000.0

    def _raw(self, trades=None, **kwargs):
        trades = trades if trades is not None else [
            {"timestamp": self.T0 + index, "side": "buy", "wallet": f"w{index}",
             "price_sol_per_token": 1e-9 * (1 + index), "notional_sol": 0.5}
            for index in range(8)
        ]
        base = dict(token="mint", created_at=self.T0, creator="dev", trades=trades)
        base.update(kwargs)
        return RawLaunch(**base)

    def test_a_reconstruction_is_stamped_and_lists_its_limitations(self):
        result = reconstruct(self._raw())
        self.assertEqual(result.status, "OK")
        provenance = result.episode[PROVENANCE_KEY]
        self.assertEqual(provenance["source"], BACKFILL_PROVENANCE)
        # The differences that flatter a reconstruction are stated, not implied.
        self.assertIn(Limitation.SURVIVORSHIP.value, provenance["limitations"])
        self.assertIn(Limitation.NO_OBSERVATION_LATENCY.value, provenance["limitations"])

    def test_inferred_depth_is_flagged_when_no_trade_recorded_reserves(self):
        result = reconstruct(self._raw())
        self.assertIn(Limitation.INFERRED_DEPTH.value,
                      result.episode[PROVENANCE_KEY]["limitations"])
        # And depth is simply absent, so the replay harness reads it as
        # DATA_BLOCKED rather than as a fillable measurement.
        self.assertNotIn("executable_sol", result.episode["market_observations"][0])

    def test_recorded_depth_is_carried_and_not_flagged(self):
        trades = [
            {"timestamp": self.T0 + index, "side": "buy", "wallet": f"w{index}",
             "price_sol_per_token": 1e-9 * (1 + index), "executable_sol": 2.0}
            for index in range(8)
        ]
        result = reconstruct(self._raw(trades=trades))
        self.assertNotIn(Limitation.INFERRED_DEPTH.value,
                         result.episode[PROVENANCE_KEY]["limitations"])
        self.assertEqual(result.episode["market_observations"][0]["executable_sol"], 2.0)

    def test_thin_material_is_refused_rather_than_reconstructed(self):
        thin = self._raw(trades=[
            {"timestamp": self.T0, "side": "buy", "wallet": "w",
             "price_sol_per_token": 1e-9}])
        result = reconstruct(thin, min_trades=5)
        # Three trades is three trades, not a lifecycle, and treating it as one
        # puts noise into the moat wearing the shape of evidence.
        self.assertEqual(result.status, "DATA_BLOCKED")
        self.assertIsNone(result.episode)

    def test_trades_without_a_usable_price_block(self):
        result = reconstruct(self._raw(trades=[
            {"timestamp": self.T0 + i, "side": "buy", "wallet": f"w{i}"}
            for i in range(8)]))
        self.assertEqual(result.status, "DATA_BLOCKED")

    def test_a_collapse_is_recorded_as_observed_not_as_a_rug(self):
        trades = [
            {"timestamp": self.T0, "side": "buy", "wallet": "a", "price_sol_per_token": 1e-9},
            {"timestamp": self.T0 + 10, "side": "buy", "wallet": "b", "price_sol_per_token": 2e-8},
            {"timestamp": self.T0 + 20, "side": "sell", "wallet": "a", "price_sol_per_token": 1e-8},
            {"timestamp": self.T0 + 30, "side": "sell", "wallet": "b", "price_sol_per_token": 5e-9},
            {"timestamp": self.T0 + 40, "side": "sell", "wallet": "c", "price_sol_per_token": 1e-11},
        ]
        outcome = reconstruct(self._raw(trades=trades)).episode["final_outcome"]
        self.assertTrue(outcome["collapsed"])
        # Reconstruction cannot see WHO caused it. Labelling intent from price
        # teaches a rug model to predict drawdowns instead of rugs.
        self.assertIsNone(outcome["rugged"])

    def test_first_buyers_are_ordered_and_deduplicated(self):
        trades = [
            {"timestamp": self.T0 + 3, "side": "buy", "wallet": "second",
             "price_sol_per_token": 1e-9},
            {"timestamp": self.T0 + 1, "side": "buy", "wallet": "first",
             "price_sol_per_token": 1e-9},
            {"timestamp": self.T0 + 5, "side": "buy", "wallet": "first",
             "price_sol_per_token": 1e-9},
            {"timestamp": self.T0 + 7, "side": "sell", "wallet": "third",
             "price_sol_per_token": 1e-9},
            {"timestamp": self.T0 + 9, "side": "buy", "wallet": "third",
             "price_sol_per_token": 1e-9},
        ]
        buyers = reconstruct(self._raw(trades=trades))
        names = [entry["wallet"] for entry in buyers.episode["first_buyers"]]
        # Order matters: the sequence and the set mean different things.
        self.assertEqual(names, ["first", "second", "third"])

    def test_a_partial_buyer_set_is_flagged(self):
        result = reconstruct(self._raw(), buyer_depth=25)
        self.assertIn(Limitation.PARTIAL_BUYER_SET.value,
                      result.episode[PROVENANCE_KEY]["limitations"])

    def test_an_unstamped_episode_is_treated_as_reconstructed(self):
        # The pessimistic default: mistaking a reconstruction for an
        # observation inflates every downstream result; the reverse only
        # wastes data.
        self.assertTrue(is_reconstructed({"token": "t"}))
        self.assertTrue(is_reconstructed({"token": "t", PROVENANCE_KEY: "nonsense"}))
        self.assertFalse(is_reconstructed(stamp_live({"token": "t"})))

    def test_partitioning_separates_the_two_populations(self):
        observed = [stamp_live({"token": "live"})]
        rebuilt = [reconstruct(self._raw()).episode]
        live, back = partition_by_provenance(observed + rebuilt + [{"token": "bare"}])
        self.assertEqual([item["token"] for item in live], ["live"])
        self.assertEqual(len(back), 2)

    def test_the_stamp_survives_a_json_round_trip(self):
        episode = reconstruct(self._raw()).episode
        # It has to survive into every downstream dataset or the separation is
        # only true in memory.
        revived = json.loads(json.dumps(episode, default=str))
        self.assertTrue(is_reconstructed(revived))

    def test_a_batch_reports_yield_and_why_it_lost_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            good = [self._raw(token=f"g{index}") for index in range(3)]
            bad = [self._raw(token=f"b{index}", trades=[]) for index in range(2)]
            report = run_backfill(good + bad, Path(directory))
            payload = report.to_dict()
            self.assertEqual((payload["attempted"], payload["reconstructed"],
                              payload["blocked"]), (5, 3, 2))
            self.assertAlmostEqual(payload["yield"], 0.6)
            self.assertTrue(payload["blocked_reasons"])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 3)

    def test_an_empty_batch_reports_no_yield_rather_than_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = run_backfill([], Path(directory)).to_dict()
            # Zero would read as "we tried and reconstructed nothing".
            self.assertIsNone(payload["yield"])


class TestSourceMesh(unittest.IsolatedAsyncioTestCase):
    """A dead feed and a quiet feed produce the same event count."""

    MINT = "So11111111111111111111111111111111111111112"
    NOW = 1_800_000_000.0

    class _Fake(EventSource):
        def __init__(self, source_id, events=None, raises=False):
            super().__init__(source_id, SourceClass.CHAT)
            self._events = events or []
            self._raises = raises

        async def poll(self):
            if self._raises:
                raise RuntimeError("upstream is down")
            return list(self._events)

    def _event(self, source_id, text, source_at=None, observed_at=None):
        return Event(source_id=source_id, source_class=SourceClass.CHAT,
                     source_at=source_at if source_at is not None else self.NOW,
                     observed_at=observed_at if observed_at is not None else self.NOW,
                     text=text, token_addresses=tuple(extract_mints(text)))

    async def test_a_failing_source_does_not_take_down_the_mesh(self):
        good = self._Fake("good", [self._event("good", "hello")])
        bad = self._Fake("bad", raises=True)
        mesh = SourceMesh([bad, good])
        events = await mesh.collect(self.NOW)
        # A mesh that stops on the first failure has coverage equal to its
        # least reliable member.
        self.assertEqual([event.source_id for event in events], ["good"])

    async def test_a_never_polled_source_is_not_reported_ok(self):
        mesh = SourceMesh([self._Fake("cold")])
        health = mesh.health(self.NOW)
        self.assertEqual(health["by_state"], {"NEVER_STARTED": 1})
        self.assertEqual(health["coverage"], 0.0)

    async def test_silence_becomes_degraded_then_dead(self):
        source = self._Fake("s")
        await source.collect(self.NOW)
        self.assertEqual(source.health(self.NOW).state, SourceState.OK)
        self.assertEqual(source.health(self.NOW + 400).state, SourceState.DEGRADED)
        self.assertEqual(source.health(self.NOW + 1_000).state, SourceState.DEAD)

    async def test_a_quiet_source_is_healthy_and_a_dead_one_is_not(self):
        quiet = self._Fake("quiet", [])
        await quiet.collect(self.NOW)
        report = quiet.health(self.NOW)
        # Zero events is not failure; a successful poll returning nothing is a
        # working source with nothing to say.
        self.assertEqual(report.state, SourceState.OK)
        self.assertIsNone(report.last_event_at)
        self.assertIsNotNone(report.last_poll_ok_at)

    async def test_unhealthy_sources_are_named_not_just_counted(self):
        alive = self._Fake("alive")
        silent = self._Fake("silent")
        await alive.collect(self.NOW)
        await silent.collect(self.NOW - 5_000)
        health = SourceMesh([alive, silent]).health(self.NOW)
        # The failure mode is six adapters going silent while the dashboard
        # stays green.
        self.assertEqual([item["source_id"] for item in health["unhealthy"]], ["silent"])
        self.assertAlmostEqual(health["coverage"], 0.5)

    async def test_repeats_are_deduplicated_but_the_repeater_is_recorded(self):
        text = f"official token {self.MINT}"
        first = self._Fake("regional", [self._event("regional", text)])
        second = self._Fake("big_channel", [self._event("big_channel", text)])
        mesh = SourceMesh([first, second])
        events = await mesh.collect(self.NOW)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_id, "regional")
        # Dropping the repeat outright would throw away exactly the lead-lag
        # evidence that says which source is upstream.
        self.assertEqual(mesh.repeaters_of(events[0].content_hash),
                         ["regional", "big_channel"])

    async def test_content_hash_ignores_the_source(self):
        one = self._event("a", f"buy {self.MINT} now")
        two = self._event("b", f"buy {self.MINT} now")
        self.assertEqual(one.content_hash, two.content_hash)
        self.assertNotEqual(one.content_hash, self._event("a", "different").content_hash)

    async def test_the_dedupe_window_expires(self):
        text = "same text"
        mesh = SourceMesh([self._Fake("a", [self._event("a", text)])], dedupe_window=60.0)
        self.assertEqual(len(await mesh.collect(self.NOW)), 1)
        self.assertEqual(len(await mesh.collect(self.NOW + 10)), 0)
        self.assertEqual(len(await mesh.collect(self.NOW + 1_000)), 1)

    def test_the_two_timestamps_are_never_collapsed(self):
        event = self._event("s", "text", source_at=self.NOW, observed_at=self.NOW + 6.5)
        # The source's own delivery latency, which no local speed recovers.
        self.assertAlmostEqual(event.observation_lag, 6.5)

    def test_a_clock_artefact_never_reports_a_negative_lag(self):
        event = self._event("s", "text", source_at=self.NOW, observed_at=self.NOW - 30)
        # Otherwise a source appears to reach us before it published, and
        # ranks first.
        self.assertEqual(event.observation_lag, 0.0)


class TestSourceAdapters(unittest.IsolatedAsyncioTestCase):
    MINT = "So11111111111111111111111111111111111111112"

    async def _collect(self, source):
        return await source.collect(1_800_000_000.0)

    async def test_a_telegram_message_normalises_with_its_mint(self):
        async def fetch():
            return [{"message": f"official CA {self.MINT}", "date": 1_700_000_000,
                     "id": 42, "sender_id": 1001}]
        events = await self._collect(telegram_source("tg:alpha", fetch, channel="alpha"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].token_addresses, (self.MINT,))
        self.assertEqual(events[0].source_class, SourceClass.CHAT)
        self.assertEqual(events[0].author_id, "1001")

    async def test_youtube_uses_title_and_description_not_the_transcript(self):
        async def fetch():
            return [{"title": f"my coin {self.MINT}", "description": "launching now",
                     "published": 1_700_000_000, "channel_id": "UC1", "video_id": "v1"}]
        events = await self._collect(youtube_websub_source("yt:UC1", fetch))
        # Waiting for a transcript to confirm what the title states spends the
        # entire lead the push notification bought.
        self.assertEqual(events[0].token_addresses, (self.MINT,))
        self.assertEqual(events[0].metadata["video_id"], "v1")

    async def test_rss_carries_language_and_host(self):
        async def fetch():
            return [{"title": "haber", "summary": "detay",
                     "link": "https://ornek.com.tr/a", "published_epoch": 1_700_000_000}]
        events = await self._collect(rss_source("rss:tr", fetch, language="tr"))
        self.assertEqual(events[0].language, "tr")
        self.assertEqual(events[0].metadata["host"], "ornek.com.tr")

    async def test_metadata_artifacts_carry_a_structured_mint(self):
        async def fetch():
            return [{"name": "Doge", "symbol": "DOGE", "mint": self.MINT,
                     "uri": "https://ipfs.example/x", "uploaded_at": 1_700_000_000}]
        events = await self._collect(metadata_artifact_source("meta", fetch))
        # The mint is a field here, not text, and must still reach the event.
        self.assertIn(self.MINT, events[0].token_addresses)

    async def test_a_malformed_record_does_not_lose_the_batch(self):
        async def fetch():
            return [{"message": "fine", "date": 1_700_000_000},
                    {"date": "not a number", "message": "also fine"},
                    {"message": "third", "date": 1_700_000_001}]
        events = await self._collect(telegram_source("tg", fetch))
        self.assertEqual(len(events), 2)

    async def test_empty_text_produces_no_event(self):
        async def fetch():
            return [{"message": "", "date": 1_700_000_000}]
        self.assertEqual(await self._collect(telegram_source("tg", fetch)), [])

    async def test_adapters_do_not_judge_relevance(self):
        """A source that filters silently becomes the model."""
        async def fetch():
            return [{"message": "gm", "date": 1_700_000_000},
                    {"message": "totally irrelevant chatter", "date": 1_700_000_001}]
        events = await self._collect(telegram_source("tg", fetch))
        self.assertEqual(len(events), 2)

    def test_uncovered_networks_are_stated_rather_than_hidden(self):
        report = coverage_report({"sources": 4, "coverage": 1.0, "by_state": {"OK": 4}})
        # 100% healthy across four adapters is still blind to most of the
        # world, and coverage alone would not say so.
        self.assertIn("x_twitter", report["uncovered_networks"])
        self.assertTrue(report["coverage_is_over_connected_sources_only"])

    async def test_every_adapter_produces_the_same_canonical_shape(self):
        async def fetch_one(payload):
            async def fetch():
                return [payload]
            return fetch

        builders = [
            (telegram_source, {"message": "x", "date": 1.0}),
            (bluesky_source, {"commit": {"record": {"text": "x"}}, "time_us": 1e6}),
            (nostr_source, {"content": "x", "created_at": 1.0}),
            (farcaster_source, {"text": "x", "timestamp": 1.0}),
            (mastodon_source, {"content": "x", "created_at_epoch": 1.0}),
        ]
        for builder, payload in builders:
            source = builder("s", await fetch_one(payload))
            events = await self._collect(source)
            self.assertEqual(len(events), 1, builder.__name__)
            event = events[0]
            self.assertIsInstance(event, Event)
            self.assertIn("schema_version", event.to_dict())
            self.assertGreaterEqual(event.observation_lag, 0.0)


class TestSourceRegistry(unittest.TestCase):
    """Adapters are not coverage. Twelve adapters and four feeds is blindness."""

    def _declaration(self, **kwargs):
        base = dict(source_id="rss:kr", kind="rss", language="ko", region="kr")
        base.update(kwargs)
        return SourceDeclaration(**base)

    async def _fetch(self):
        return []

    def _fetchers(self, *ids):
        async def fetch():
            return []
        return {source_id: fetch for source_id in ids}

    def test_a_declaration_is_instantiated_when_ready(self):
        declaration = self._declaration()
        sources, report = build_sources([declaration], self._fetchers("rss:kr"))
        self.assertEqual(len(sources), 1)
        self.assertEqual(report.ready, 1)
        self.assertEqual(report.by_state["READY"], 1)

    def test_a_missing_credential_is_named_not_silently_skipped(self):
        declaration = self._declaration(source_id="tg:a", kind="telegram",
                                        requires_env=("NO_SUCH_KEY_12345",))
        _, report = build_sources([declaration], self._fetchers("tg:a"))
        payload = report.to_dict()
        # A source that vanishes for a missing key is a coverage hole nobody
        # sees.
        self.assertEqual(payload["unconfigured"], ["tg:a"])
        self.assertEqual(payload["ready"], 0)

    def test_credentials_are_checked_by_presence_never_by_value(self):
        with patch.dict("os.environ", {"SECRET_TOKEN_X": "hunter2"}, clear=False):
            declaration = self._declaration(source_id="tg:b", kind="telegram",
                                            requires_env=("SECRET_TOKEN_X",))
            self.assertEqual(declaration.missing_credentials(), [])
            _, report = build_sources([declaration], self._fetchers("tg:b"))
            # Nothing in the report can carry the value.
            self.assertNotIn("hunter2", json.dumps(report.to_dict()))

    def test_an_unknown_kind_is_a_reported_problem(self):
        _, report = build_sources([self._declaration(kind="carrier_pigeon")],
                                  self._fetchers("rss:kr"))
        self.assertEqual(report.by_state["UNKNOWN_KIND"], 1)
        self.assertIn("carrier_pigeon", report.problems[0][1])

    def test_a_declaration_with_no_fetcher_is_reported_not_dropped(self):
        _, report = build_sources([self._declaration()], {})
        self.assertEqual(report.by_state["NO_FETCHER"], 1)
        self.assertEqual(report.ready, 0)

    def test_the_report_breaks_coverage_down_by_language_and_region(self):
        declarations = [
            self._declaration(source_id="rss:kr", language="ko", region="kr"),
            self._declaration(source_id="rss:jp", language="ja", region="jp"),
            self._declaration(source_id="rss:kr2", language="ko", region="kr"),
        ]
        _, report = build_sources(declarations, self._fetchers("rss:kr", "rss:jp", "rss:kr2"))
        payload = report.to_dict()
        self.assertEqual(payload["by_language"], {"ja": 1, "ko": 2})
        self.assertEqual(payload["by_region"], {"jp": 1, "kr": 2})
        self.assertAlmostEqual(payload["ready_share"], 1.0)

    def test_nothing_declared_reports_no_share_rather_than_zero(self):
        _, report = build_sources([], {})
        self.assertIsNone(report.to_dict()["ready_share"])

    def test_the_shipped_registry_parses_and_covers_many_languages(self):
        path = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
        declarations = load_declarations(str(path))
        self.assertGreater(len(declarations), 20)
        languages = {item.language for item in declarations if item.language}
        regions = {item.region for item in declarations if item.region}
        # Native-language detection, not English-first: a Korean story should
        # not wait for its English repost.
        for expected in ("ko", "ja", "zh", "ru", "tr", "ar", "pt", "es"):
            self.assertIn(expected, languages)
        self.assertGreaterEqual(len(regions), 10)

    def test_every_declared_kind_has_an_adapter(self):
        path = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
        for declaration in load_declarations(str(path)):
            self.assertIn(declaration.kind, ADAPTER_KINDS, declaration.source_id)

    def test_one_malformed_entry_does_not_take_the_file_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.yaml"
            path.write_text(yaml.safe_dump({"sources": [
                {"id": "good", "kind": "rss"},
                {"kind": "rss"},
                {"id": "also_good", "kind": "rss"},
            ]}))
            declarations = load_declarations(str(path))
            # One bad line in a 400-source file must not lose the other 399.
            self.assertEqual([item.source_id for item in declarations],
                             ["good", "also_good"])

    def test_an_unreadable_registry_yields_nothing_rather_than_raising(self):
        self.assertEqual(load_declarations("/nonexistent/sources.yaml"), [])


class TestSourceDiscovery(unittest.TestCase):
    """Promoting on two coincidences fills the mesh with noise."""

    KNOWN = ("big_channel", "aggregator", "other_known")

    def _discovery(self, **kwargs):
        base = dict(min_observations=8, min_lead_rate=0.6, min_distinct_followers=2)
        base.update(kwargs)
        return SourceDiscovery(**base)

    def test_a_consistent_upstream_source_is_promoted(self):
        discovery = self._discovery()
        for _ in range(10):
            discovery.observe(["regional", "big_channel", "aggregator"], self.KNOWN)
        promotable = discovery.promotable()
        self.assertEqual([item.source_id for item in promotable], ["regional"])
        self.assertAlmostEqual(promotable[0].lead_rate, 1.0)

    def test_a_thin_history_is_not_promoted(self):
        discovery = self._discovery()
        for _ in range(3):
            discovery.observe(["lucky", "big_channel", "aggregator"], self.KNOWN)
        self.assertEqual(discovery.promotable(), [])

    def test_leading_one_channel_repeatedly_is_one_relationship(self):
        discovery = self._discovery(min_distinct_followers=2)
        for _ in range(30):
            discovery.observe(["echo", "big_channel"], self.KNOWN)
        # Leading the same channel thirty times is one relationship, not
        # thirty, and a mirror of one feed is not an upstream source.
        self.assertEqual(discovery.promotable(), [])

    def test_a_source_already_in_the_mesh_is_not_a_discovery(self):
        discovery = self._discovery()
        for _ in range(20):
            discovery.observe(["big_channel", "aggregator"], self.KNOWN)
        self.assertEqual(discovery.report()["tracked"], 0)

    def test_a_source_that_usually_follows_is_not_promoted(self):
        discovery = self._discovery()
        for _ in range(3):
            discovery.observe(["sometimes", "big_channel", "aggregator"], self.KNOWN)
        for _ in range(17):
            discovery.observe(["sometimes"], self.KNOWN)
            discovery.observe(["sometimes", "unknown_other"], self.KNOWN)
        candidates = {item.source_id for item in discovery.promotable()}
        self.assertNotIn("sometimes", candidates)

    def test_a_single_source_observation_teaches_nothing(self):
        discovery = self._discovery()
        for _ in range(20):
            discovery.observe(["alone"], self.KNOWN)
        self.assertEqual(discovery.report()["tracked"], 0)

    def test_the_report_states_the_gate_it_applied(self):
        report = self._discovery().report()
        self.assertEqual(report["gate"]["min_observations"], 8)
        self.assertEqual(report["gate"]["min_distinct_followers"], 2)


class TestPromotionGate(unittest.TestCase):
    """The bar has to be written before the numbers arrive.

    Every system reaches the moment where forward results are almost good
    enough and the bar quietly becomes whatever they are. It never feels like
    dishonesty; each adjustment is defensible in isolation.
    """

    def _criteria(self, **kwargs):
        base = dict(stage=Stage.CANARY, min_decisions=100, min_real_fills=50,
                    min_launch_cohorts=20, min_regimes=2, min_net_log_growth=0.0,
                    max_rug_loss_share=0.15, min_monster_enrichment=2.0,
                    min_execution_success=0.6, max_catastrophic_failures=0)
        base.update(kwargs)
        return PromotionCriteria(**base)

    def _evidence(self, **kwargs):
        base = dict(stage=Stage.CANARY, decisions=500, real_fills=200,
                    launch_cohorts=60, regimes_covered=3, net_log_growth=0.02,
                    rug_loss_share=0.10, monster_enrichment=2.4,
                    execution_success=0.72, catastrophic_failures=0)
        base.update(kwargs)
        return Evidence(**base)

    def test_evidence_that_clears_every_bar_passes(self):
        verdict = evaluate(self._criteria(), self._evidence())
        self.assertTrue(verdict.passed, verdict.failures)

    def test_an_unmeasured_criterion_fails_rather_than_passing(self):
        """A gate that treats unmeasured as satisfied is decorative."""
        verdict = evaluate(self._criteria(), self._evidence(rug_loss_share=None))
        self.assertFalse(verdict.passed)
        self.assertIn("rug_loss_share", verdict.unmeasured)
        # And it fails in the direction that does NOT promote.
        self.assertTrue(any("not measured" in reason for reason in verdict.failures))

    def test_unmeasured_growth_fails(self):
        verdict = evaluate(self._criteria(), self._evidence(net_log_growth=None))
        self.assertFalse(verdict.passed)
        self.assertIn("net_log_growth", verdict.unmeasured)

    def test_flat_growth_does_not_clear_a_zero_bar(self):
        # "Did not lose money" is not "made money".
        self.assertFalse(evaluate(self._criteria(),
                                  self._evidence(net_log_growth=0.0)).passed)

    def test_each_shortfall_is_reported_individually(self):
        verdict = evaluate(self._criteria(),
                           self._evidence(real_fills=1, monster_enrichment=1.0,
                                          rug_loss_share=0.9))
        self.assertFalse(verdict.passed)
        self.assertGreaterEqual(len(verdict.failures), 3)

    def test_one_catastrophic_failure_is_disqualifying(self):
        self.assertFalse(evaluate(self._criteria(),
                                  self._evidence(catastrophic_failures=1)).passed)

    def test_evidence_for_the_wrong_stage_is_refused(self):
        verdict = evaluate(self._criteria(stage=Stage.CANARY),
                           self._evidence(stage=Stage.FORWARD_SHADOW))
        self.assertFalse(verdict.passed)
        self.assertIn("evidence is for", verdict.failures[0])

    def test_a_criterion_set_to_zero_needs_no_measurement(self):
        criteria = self._criteria(min_real_fills=0)
        verdict = evaluate(criteria, self._evidence(real_fills=None))
        self.assertTrue(verdict.passed, verdict.failures)

    def test_the_fingerprint_changes_when_a_threshold_changes(self):
        strict = self._criteria(min_real_fills=1_000)
        loose = self._criteria(min_real_fills=10)
        # Moving a bar produces a NEW bar with a new identity, rather than
        # silently redefining the old verdict.
        self.assertNotEqual(strict.fingerprint, loose.fingerprint)
        self.assertEqual(strict.fingerprint, self._criteria(min_real_fills=1_000).fingerprint)

    def test_the_verdict_records_which_bar_it_was_judged_against(self):
        criteria = self._criteria()
        verdict = evaluate(criteria, self._evidence())
        self.assertEqual(verdict.criteria_fingerprint, criteria.fingerprint)

    def test_a_pass_advances_exactly_one_stage(self):
        verdict = evaluate(self._criteria(), self._evidence())
        self.assertTrue(can_advance(Stage.CANARY, verdict))
        self.assertEqual(next_stage(Stage.CANARY), Stage.LIVE)
        # Skipping is how a promising backtest reaches real money without ever
        # producing a real fill.
        self.assertFalse(can_advance(Stage.FORWARD_SHADOW, verdict))
        self.assertIsNone(next_stage(Stage.LIVE))

    def test_a_failed_verdict_advances_nothing(self):
        verdict = evaluate(self._criteria(), self._evidence(real_fills=0))
        self.assertFalse(can_advance(Stage.CANARY, verdict))

    def test_the_shipped_bars_require_real_fills_before_canary_and_live(self):
        self.assertEqual(DEFAULT_CRITERIA[Stage.FORWARD_SHADOW].min_real_fills, 0)
        self.assertGreaterEqual(DEFAULT_CRITERIA[Stage.CANARY].min_real_fills, 1_000)
        self.assertGreater(DEFAULT_CRITERIA[Stage.LIVE].min_real_fills,
                           DEFAULT_CRITERIA[Stage.CANARY].min_real_fills)

    def test_the_shipped_bars_tighten_monotonically(self):
        canary = DEFAULT_CRITERIA[Stage.CANARY]
        live = DEFAULT_CRITERIA[Stage.LIVE]
        self.assertGreater(live.min_decisions, canary.min_decisions)
        self.assertLess(live.max_rug_loss_share, canary.max_rug_loss_share)
        self.assertGreater(live.min_monster_enrichment, canary.min_monster_enrichment)


class TestPromotionLedger(unittest.TestCase):
    """Rejections are the record that makes a moved bar visible."""

    def _row(self, directory, min_fills):
        criteria = PromotionCriteria(stage=Stage.CANARY, min_real_fills=min_fills,
                                     min_net_log_growth=0.0)
        evidence = Evidence(stage=Stage.CANARY, real_fills=100, net_log_growth=0.01,
                            rug_loss_share=0.0, catastrophic_failures=0,
                            regimes_covered=3, monster_enrichment=2.0)
        verdict = evaluate(criteria, evidence)
        ledger = PromotionLedger(Path(directory) / "promotions.jsonl")
        ledger.record(verdict, evidence, criteria)
        return ledger, verdict

    def test_verdicts_including_failures_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, verdict = self._row(directory, min_fills=1_000)
            self.assertFalse(verdict.passed)
            history = ledger.history()
            self.assertEqual(len(history), 1)
            self.assertFalse(history[0]["verdict"]["passed"])

    def test_a_moved_bar_is_visible_in_the_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger, strict = self._row(directory, min_fills=1_000)
            self.assertFalse(strict.passed)
            ledger, loose = self._row(directory, min_fills=10)
            self.assertTrue(loose.passed)
            # A moved bar and a cleared bar look identical in a verdict alone.
            self.assertTrue(ledger.bar_moved(Stage.CANARY))
            self.assertEqual(len(ledger.fingerprints_for(Stage.CANARY)), 2)

    def test_a_stable_bar_does_not_read_as_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            self._row(directory, min_fills=1_000)
            ledger, _ = self._row(directory, min_fills=1_000)
            self.assertFalse(ledger.bar_moved(Stage.CANARY))

    def test_an_absent_ledger_reads_as_empty_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PromotionLedger(Path(directory) / "missing.jsonl")
            self.assertEqual(ledger.history(), [])
            self.assertFalse(ledger.bar_moved(Stage.LIVE))

    def test_the_gate_does_not_touch_the_live_capital_lock(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "research"
                  / "promotion_gate.py").read_text()
        # A PASS says the evidence cleared a pre-declared bar. It is not an
        # instruction to trade, and this module has no path to becoming one.
        self.assertNotIn("ALLOW_LIVE_TRADING =", source)
        self.assertNotIn("os.environ[", source)


class TestSnapshotCaptureIsolation(unittest.IsolatedAsyncioTestCase):
    """One broken feature group must not cost a whole point-in-time row.

    A snapshot is unrecoverable once its instant has passed, so losing one to
    an AttributeError in a single feature is the most expensive way this
    repository can lose data -- and it was silent.
    """

    def test_wallet_score_has_no_is_insider_field(self):
        """The bug, pinned as a contract.

        Insider status lives on the genealogy WalletProfile. Reading it off
        WalletScore raised inside the capture handler and abandoned the entire
        snapshot, wherever a scored buyer appeared.
        """
        fields = {field.name for field in dataclasses.fields(WalletScore)}
        self.assertNotIn("is_insider", fields)
        self.assertIn("is_insider",
                      {field.name for field in dataclasses.fields(WalletProfile)})

    def test_insider_count_is_read_from_the_genealogy_profile(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "research"
                  / "dataset_builder.py").read_text()
        self.assertNotIn("ws.is_insider", source)
        self.assertIn("self.genealogy.get_wallet_profile", source)

    async def test_a_failing_group_is_data_blocked_and_the_rest_still_land(self):
        builder = PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
        builder.chain_config = SimpleNamespace(name="solana")
        builder.active_episodes = {}
        builder._snapshot_inflight = set()

        async def ok(episode, as_of):
            return {"status": "OK"}

        async def boom(episode, as_of):
            raise AttributeError("'WalletScore' object has no attribute 'is_insider'")

        builder._capture_deployer_features = ok
        builder._capture_wallet_features = boom
        for name in ("flow", "liquidity", "social", "token", "market", "entity_graph"):
            setattr(builder, f"_capture_{name}_features", ok)

        episode = LaunchEpisode(token="mint", chain="solana", created_at=1.0,
                                deployer="dev", factory="", pair="", base_token="")
        builder.active_episodes["mint"] = episode
        await PointInTimeDatasetBuilder._capture_snapshot(
            builder, "mint", SnapshotTimepoint.T0)

        snapshot = episode.snapshots[SnapshotTimepoint.T0]
        # The broken group says why, and the other seven survive.
        self.assertEqual(snapshot.wallet_features["status"], "DATA_BLOCKED")
        self.assertIn("is_insider", snapshot.wallet_features["reason"])
        self.assertEqual(snapshot.deployer_features["status"], "OK")
        self.assertEqual(snapshot.flow_features["status"], "OK")
        self.assertEqual(snapshot.entity_graph_features["status"], "OK")


class TestDecisionSnapshot(unittest.TestCase):
    """A decision must execute against the state that produced it.

    The bug this prevents does not look like a bug: the policy scores ADD from
    one plan_scale_in, the executor refreshes state and calls plan_scale_in
    again, and both calls are correct. Neither is the decision. What executes
    is a size computed from a state the policy never evaluated -- and the two
    diverge exactly when the market is moving, which is the only time the size
    matters. The decision and the fill are both logged, referring to different
    instants.
    """

    NOW = 1_800_000_000.0

    def _snapshot(self, seq=1, **kwargs):
        base = dict(token="mint", action="add", state_seq=seq, size_base_units=1_000,
                    protective_limit=1_010, feature_hash="f", model_hash="m",
                    created_at=self.NOW, expiry_seconds=1.5)
        base.update(kwargs)
        return DecisionSnapshot(**base)

    def _sequencer(self, token="mint", seq=1):
        sequencer = StateSequencer()
        for _ in range(seq):
            sequencer.bump(token)
        return sequencer

    def test_a_matching_decision_executes(self):
        outcome = decision_guard(self._snapshot(), self._sequencer(), self.NOW)
        self.assertTrue(outcome.executed)
        self.assertEqual(outcome.status, DecisionStatus.VALID)

    def test_state_advancing_makes_the_decision_stale(self):
        outcome = decision_guard(self._snapshot(seq=1), self._sequencer(seq=2), self.NOW)
        self.assertFalse(outcome.executed)
        self.assertEqual(outcome.status, DecisionStatus.STALE)
        self.assertIn("reprice", outcome.detail)

    def test_the_newest_decision_can_still_be_too_old(self):
        """Expiry is independent of staleness on purpose."""
        outcome = decision_guard(self._snapshot(), self._sequencer(), self.NOW + 10)
        self.assertFalse(outcome.executed)
        # Nothing superseded it; on a launch moving in hundreds of
        # milliseconds a 10-second-old size is still wrong.
        self.assertEqual(outcome.status, DecisionStatus.EXPIRED)

    def test_a_decision_executes_exactly_once(self):
        snapshot = self._snapshot()
        sequencer = self._sequencer()
        self.assertTrue(decision_guard(snapshot, sequencer, self.NOW).executed)
        snapshot.consume()
        second = decision_guard(snapshot, sequencer, self.NOW)
        # A retry loop resubmitting is a double position on a buy and an
        # oversell on a sell.
        self.assertFalse(second.executed)
        self.assertEqual(second.status, DecisionStatus.CONSUMED)

    def test_sequences_distinguish_updates_inside_one_clock_tick(self):
        sequencer = StateSequencer()
        first = sequencer.bump("mint")
        second = sequencer.bump("mint")
        # Two updates in one tick are indistinguishable by time and perfectly
        # distinguishable by sequence, and under load they arrive together.
        self.assertEqual((first, second), (1, 2))
        self.assertEqual(sequencer.current("other"), 0)

    def test_the_snapshot_carries_size_and_limit_not_the_inputs(self):
        payload = self._snapshot().to_dict()
        for key in ("size_base_units", "protective_limit", "state_seq",
                    "feature_hash", "model_hash"):
            self.assertIn(key, payload)
        # Anything recomputable at execution time will be recomputed, and then
        # the executed trade is not the decided trade.
        self.assertNotIn("liquidity_usd", payload)
        self.assertNotIn("prediction", payload)

    def test_identical_inputs_hash_identically_and_different_ones_do_not(self):
        self.assertEqual(state_hash({"a": 1, "b": 2}), state_hash({"b": 2, "a": 1}))
        self.assertNotEqual(state_hash({"a": 1}), state_hash({"a": 2}))


class TestScaleInExecutesTheDecidedSize(unittest.IsolatedAsyncioTestCase):
    """The wiring, not just the object."""

    def _desk(self, result, sequencer):
        engine = ElogwEngine(SimpleNamespace(_is_trained=True), min_edge_bps=-1)
        engine.portfolio_value = 10_000.0
        replans = []
        engine.plan_scale_in = lambda *a, **k: replans.append(1) or (0.5, 0.5)

        desk = SimpleNamespace(
            predictor=SimpleNamespace(_is_trained=True), dry_run=True,
            champion_challenger=SimpleNamespace(is_live=lambda _id: True),
            elogw_engine=engine, execution_engine=FakeExecutionEngineForExit(result),
            dataset_builder=FakeDatasetBuilderForExit(),
            fee_optimizer=SimpleNamespace(get_optimal_fee=lambda *a: 5_000,
                                          get_jito_tip=lambda *a: 100_000),
            wallet_equity_usd=10_000.0, sol_price_usd=150.0,
            state_sequencer=sequencer, global_config={}, replans=replans,
            ops_events=[],
        )
        desk._record_ops_event = (
            lambda stream, payload: desk.ops_events.append((stream, payload)))

        async def refresh():
            return None

        desk._refresh_portfolio_state = refresh
        return desk

    @staticmethod
    def _position():
        return {
            "size_tokens": 1_000, "initial_size_tokens": 1_000,
            "remaining_cost_usd": 100.0, "initial_cost_usd": 100.0,
            "risk_contribution": 0.02, "decision_id": "d1",
            "candidate": SimpleNamespace(base_token=None),
            "risk_object": SimpleNamespace(), "liquidity_usd": 500_000.0,
            "prediction_object": MultiHeadPrediction("mint", "solana", 0, p_2x=0.8,
                                                     p_5x=0.5, p_rug_30s=0.01,
                                                     p_rug_5m=0.02),
        }

    async def test_the_frozen_size_is_what_executes(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED,
                                 simulated=True, quoted_output_amount=500,
                                 actual_input_amount=1)
        sequencer = StateSequencer()
        sequencer.bump("mint")
        desk = self._desk(result, sequencer)
        snapshot = DecisionSnapshot(
            token="mint", action="add", state_seq=1, size_base_units=777_000_000,
            protective_limit=800_000_000, feature_hash="f", model_hash="m",
            q_value=0.5, evidence={"fraction": 0.01, "slippage_bps": 100})

        await MemecoinQuantDesk._consider_scale_in(
            desk, "mint", self._position(), 1.5, snapshot)

        self.assertEqual(desk.execution_engine.buys[0][2], 777_000_000)
        # The executor must not re-derive a size the policy never evaluated.
        self.assertEqual(desk.replans, [])

    async def test_a_stale_decision_is_refused_and_recorded(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED,
                                 simulated=True, quoted_output_amount=500)
        sequencer = StateSequencer()
        sequencer.bump("mint")
        sequencer.bump("mint")
        desk = self._desk(result, sequencer)
        snapshot = DecisionSnapshot(
            token="mint", action="add", state_seq=1, size_base_units=777_000_000,
            protective_limit=800_000_000, feature_hash="f", model_hash="m")

        await MemecoinQuantDesk._consider_scale_in(
            desk, "mint", self._position(), 1.5, snapshot)

        self.assertEqual(desk.execution_engine.buys, [])
        # How often decisions go stale measures how fast state moves relative
        # to how fast we decide -- a latency number worth keeping.
        streams = [stream for stream, _ in desk.ops_events]
        self.assertIn("stale_decisions", streams)

    async def test_without_a_snapshot_the_legacy_path_still_works(self):
        result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED,
                                 simulated=True, quoted_output_amount=500,
                                 actual_input_amount=1)
        desk = self._desk(result, StateSequencer())
        await MemecoinQuantDesk._consider_scale_in(desk, "mint", self._position(), 1.5)
        self.assertEqual(len(desk.execution_engine.buys), 1)
        self.assertEqual(desk.replans, [1])


class TestConcurrentSourceMesh(unittest.IsolatedAsyncioTestCase):
    """One slow endpoint must not delay every source behind it."""

    NOW = 1_800_000_000.0

    class _Slow(EventSource):
        def __init__(self, source_id, delay, events=None):
            super().__init__(source_id, SourceClass.FEED)
            self.delay = delay
            self._events = events or []

        async def poll(self):
            await asyncio.sleep(self.delay)
            return list(self._events)

    def _event(self, source_id, text):
        return Event(source_id=source_id, source_class=SourceClass.FEED,
                     source_at=self.NOW, observed_at=self.NOW, text=text)

    async def test_a_slow_source_does_not_hold_up_the_others(self):
        slow = self._Slow("slow_rss", 0.30, [self._event("slow_rss", "late")])
        fast = [self._Slow(f"fast{index}", 0.0, [self._event(f"fast{index}", f"x{index}")])
                for index in range(20)]
        mesh = SourceMesh([slow, *fast])

        started = time.monotonic()
        events = await mesh.collect(self.NOW)
        elapsed = time.monotonic() - started

        self.assertEqual(len(events), 21)
        # Serial would be the SUM of every source's latency; concurrent is the
        # max. With hundreds of sources the serial worst case is the sum of
        # every timeout.
        self.assertLess(elapsed, 0.30 * 3)

    async def test_a_hanging_source_is_timed_out_and_marked_unhealthy(self):
        hanging = self._Slow("hangs", 5.0)
        alive = self._Slow("alive", 0.0, [self._event("alive", "here")])
        mesh = SourceMesh([hanging, alive], poll_timeout=0.05)

        started = time.monotonic()
        events = await mesh.collect(self.NOW)
        elapsed = time.monotonic() - started

        self.assertEqual([event.source_id for event in events], ["alive"])
        self.assertLess(elapsed, 1.0)
        # A source that answers too late and one that never answers are the
        # same problem from the mesh's point of view.
        self.assertGreater(hanging.health(self.NOW).consecutive_failures, 0)

    async def test_the_queue_is_bounded_and_drops_are_counted(self):
        many = self._Slow("flood", 0.0,
                          [self._event("flood", f"msg{index}") for index in range(50)])
        mesh = SourceMesh([many], max_queue=10)
        events = await mesh.collect(self.NOW)
        self.assertEqual(len(events), 10)
        self.assertEqual(mesh.health(self.NOW)["dropped_events"], 40)

    async def test_arrival_order_is_still_recorded_for_lead_lag(self):
        text = "same story"
        first = self._Slow("regional", 0.0, [self._event("regional", text)])
        second = self._Slow("big", 0.05, [self._event("big", text)])
        mesh = SourceMesh([first, second])
        events = await mesh.collect(self.NOW)
        self.assertEqual(len(events), 1)
        self.assertEqual(mesh.repeaters_of(events[0].content_hash), ["regional", "big"])

    async def test_an_empty_mesh_returns_nothing_without_erroring(self):
        self.assertEqual(await SourceMesh([]).collect(self.NOW), [])


class TestPerSourceCadence(unittest.IsolatedAsyncioTestCase):
    """One universal clock calls a healthy feed dead or lets a dead one pass."""

    NOW = 1_800_000_000.0

    class _Fake(EventSource):
        async def poll(self):
            return []

    async def test_a_push_source_can_be_held_to_a_tighter_clock(self):
        push = self._Fake("telegram", SourceClass.CHAT,
                          degraded_after_seconds=30, dead_after_seconds=90)
        await push.collect(self.NOW)
        self.assertEqual(push.health(self.NOW + 10).state, SourceState.OK)
        # Half a minute without a Telegram push connection is a real problem.
        self.assertEqual(push.health(self.NOW + 45).state, SourceState.DEGRADED)
        self.assertEqual(push.health(self.NOW + 120).state, SourceState.DEAD)

    async def test_a_slow_feed_keeps_a_looser_clock(self):
        feed = self._Fake("regional_rss", SourceClass.FEED,
                          degraded_after_seconds=3_600, dead_after_seconds=21_600)
        await feed.collect(self.NOW)
        # An hour without a regional story is normal, and the tight clock
        # would have called this dead four times over.
        self.assertEqual(feed.health(self.NOW + 1_800).state, SourceState.OK)
        self.assertEqual(feed.health(self.NOW + 7_200).state, SourceState.DEGRADED)

    async def test_the_thresholds_are_stated_in_the_health_detail(self):
        source = self._Fake("s", SourceClass.CHAT,
                            degraded_after_seconds=30, dead_after_seconds=90)
        await source.collect(self.NOW)
        detail = source.health(self.NOW + 5).detail
        self.assertIn("degraded at 30s", detail)
        self.assertIn("dead at 90s", detail)


class TestActorIntelligenceIsLiveWired(unittest.TestCase):
    """A module that exists without reaching the decision is not intelligence.

    BuyerDNA, SmartFlow and SwarmPredictor were built and tested while main
    imported only Entry, IndependenceReport and WalletIndependence -- so the
    strongest actor work sat entirely outside the decision path and 555 tests
    passed anyway.
    """

    def _desk(self, entries=None, report=None):
        desk = SimpleNamespace(
            buyer_dna=BuyerDNA(depth=25, min_corpus=50),
            swarm_predictor=SwarmPredictor(),
            independence_report=report or IndependenceReport(status="DATA_BLOCKED"),
            _actor_entries={"mint": list(entries or [])},
        )
        return desk

    def _entry(self, wallet, timestamp, skill=0.9, capital=100.0):
        return Entry(token="mint", wallet=wallet, timestamp=timestamp,
                     skill=skill, capital_usd=capital)

    def test_main_imports_the_full_actor_surface(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        for name in ("BuyerDNA", "SwarmPredictor", "aggregate_smart_flow",
                     "build_fingerprint"):
            self.assertIn(name, source, f"{name} is built but never reaches main")

    def test_a_token_with_no_scored_buyers_is_blocked_not_zero(self):
        result = MemecoinQuantDesk.actor_intelligence(self._desk(), "mint", 100.0)
        # A launch nobody scored must not read as one whose buyers scored zero.
        self.assertEqual(result["status"], "DATA_BLOCKED")
        self.assertEqual(result["observed_buyers"], 0)

    def test_smart_flow_and_swarm_reach_the_result(self):
        entries = [self._entry(f"W{i}", 100.0 - i) for i in range(5)]
        report = IndependenceReport(status="OK",
                                    scores={f"W{i}": 1.0 for i in range(5)})
        result = MemecoinQuantDesk.actor_intelligence(
            self._desk(entries, report), "mint", 100.0)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["smart_flow"]["status"], "OK")
        self.assertGreater(result["smart_flow"]["evidence"], 0)
        self.assertEqual(result["swarm"]["independent_skilled"], 5)

    def test_a_sybil_cluster_is_discounted_in_the_wired_result(self):
        entries = [self._entry(f"S{i}", 100.0 - i) for i in range(10)]
        sybil = IndependenceReport(status="OK", scores={f"S{i}": 0.0 for i in range(10)})
        clean = IndependenceReport(status="OK", scores={f"S{i}": 1.0 for i in range(10)})
        discounted = MemecoinQuantDesk.actor_intelligence(
            self._desk(entries, sybil), "mint", 100.0)
        honest = MemecoinQuantDesk.actor_intelligence(
            self._desk(entries, clean), "mint", 100.0)
        self.assertLess(discounted["smart_flow"]["evidence"],
                        honest["smart_flow"]["evidence"])

    def test_an_uncalibrated_swarm_reports_evidence_and_no_probability(self):
        entries = [self._entry(f"W{i}", 100.0 - i) for i in range(5)]
        report = IndependenceReport(status="OK", scores={f"W{i}": 1.0 for i in range(5)})
        result = MemecoinQuantDesk.actor_intelligence(
            self._desk(entries, report), "mint", 100.0)
        self.assertEqual(result["swarm"]["status"], "DATA_BLOCKED")
        self.assertIsNone(result["swarm"]["probability"])

    def test_a_thin_dna_corpus_blocks_rather_than_labelling(self):
        entries = [self._entry(f"W{i}", 100.0 - i) for i in range(5)]
        result = MemecoinQuantDesk.actor_intelligence(self._desk(entries), "mint", 100.0)
        self.assertEqual(result["buyer_dna"]["status"], "DATA_BLOCKED")
        self.assertIsNone(result["buyer_dna"]["label"])


class TestSourceIntelligenceIsLiveWired(unittest.IsolatedAsyncioTestCase):
    """src.collectors was not imported by main at all."""

    MINT = "So11111111111111111111111111111111111111112"

    class _Fake(EventSource):
        def __init__(self, source_id, events):
            super().__init__(source_id, SourceClass.CHAT)
            self._events = events

        async def poll(self):
            return list(self._events)

    def _event(self, source_id, at, lag=0.5, language="en"):
        return Event(source_id=source_id, source_class=SourceClass.CHAT,
                     source_at=at, observed_at=at + lag,
                     text=f"call {self.MINT}", language=language,
                     token_addresses=(self.MINT,))

    def _desk(self, sources=()):
        desk = SimpleNamespace(
            source_mesh=SourceMesh(list(sources)),
            _source_events={},
            source_genealogy=SourceGenealogy(),
            hot_state=HotState(HotStateBudget(), archive_root=Path("/tmp")),
        )
        return desk

    def test_main_imports_the_collectors(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        self.assertIn("src.collectors.event_source", source)
        self.assertIn("src.collectors.registry", source)

    async def test_events_are_indexed_by_token_and_reach_the_decision(self):
        desk = self._desk([self._Fake("regional", [self._event("regional", 100.0)])])
        desk.hot_state.touch_token(self.MINT)
        collected = await MemecoinQuantDesk._poll_sources(desk)
        self.assertEqual(collected, 1)

        result = MemecoinQuantDesk.source_intelligence(desk, self.MINT)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["first_source"], "regional")
        # How stale their information already was when it reached us is the
        # whole signal.
        self.assertAlmostEqual(result["first_observation_lag_s"], 0.5)

    def test_a_token_nobody_mentioned_is_blocked_not_silent_agreement(self):
        result = MemecoinQuantDesk.source_intelligence(self._desk(), self.MINT)
        self.assertEqual(result["status"], "DATA_BLOCKED")
        self.assertIn("no public source", result["detail"])

    async def test_an_empty_mesh_degrades_the_decision_without_stopping_it(self):
        desk = self._desk()
        # A dead mesh must degrade evidence, not gate the money path.
        self.assertEqual(await MemecoinQuantDesk._poll_sources(desk), 0)
        self.assertEqual(
            MemecoinQuantDesk.source_intelligence(desk, self.MINT)["status"],
            "DATA_BLOCKED")

    async def test_per_token_observations_are_bounded(self):
        many = [self._event("spam", 100.0 + index) for index in range(200)]
        desk = self._desk([self._Fake("spam", many)])
        desk.hot_state.touch_token(self.MINT)
        await MemecoinQuantDesk._poll_sources(desk)
        # A viral mint attracts thousands of posts and only the earliest few
        # carry lead information.
        self.assertLessEqual(len(desk._source_events[self.MINT]), 50)

    def test_the_registry_reports_declared_sources_without_transport(self):
        path = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
        declarations = load_declarations(str(path))
        _, report = build_sources(declarations, {})
        payload = report.to_dict()
        # "We have adapters" is not "those signals reach T0 decisions", and
        # this is the number that says which.
        self.assertGreater(payload["declared"], 20)
        self.assertEqual(payload["ready"], 0)
        self.assertEqual(payload["by_state"].get("NO_FETCHER", 0)
                         + payload["by_state"].get("UNCONFIGURED", 0),
                         payload["declared"])


class TestReentryBook(unittest.TestCase):
    """Re-entry has to be a decision, not an enum value."""

    @staticmethod
    def _bins(upside_probability=0.5, upside=3.0):
        return [(upside_probability, upside), (1.0 - upside_probability, -0.5)]

    def _price(self, book, token, **overrides):
        kwargs = dict(
            bins=self._bins(), size_fraction=0.05, capital_usd=500.0,
            expected_hold_seconds=120.0, liquidity_usd=50_000.0,
            exit_capacity_ratio=1.0, escape_probability=1.0,
            prediction_at=2_000.0, entry_cost=0.02, exit_cost=0.02, now=2_000.0,
        )
        kwargs.update(overrides)
        return book.price(token, **kwargs)

    def test_exit_reasons_map_onto_dispositions(self):
        self.assertIs(classify_exit("rug_hazard_critical"), ExitDisposition.HAZARD_ESCAPE)
        self.assertIs(classify_exit("monster_catastrophic_collapse"), ExitDisposition.CATASTROPHIC)
        self.assertIs(classify_exit("profit_ratchet_5x"), ExitDisposition.BANKED_STRENGTH)
        self.assertIs(classify_exit("action_bank_25"), ExitDisposition.BANKED_STRENGTH)
        self.assertIs(classify_exit("displaced_by_challenger"), ExitDisposition.DISPLACED)
        self.assertIs(classify_exit("distribution_detected"), ExitDisposition.DISTRIBUTION)
        self.assertIs(classify_exit("something_new"), ExitDisposition.UNKNOWN)

    def test_a_token_never_held_is_not_a_reentry(self):
        book = ReentryBook()
        verdict = book.admits("fresh", now=1_000.0)
        self.assertTrue(verdict.admitted)
        self.assertIn("not a post-exit candidate", verdict.detail)

    def test_catastrophic_exits_are_barred_not_priced(self):
        book = ReentryBook()
        book.record_exit("mint", "monster_catastrophic_collapse", exited_at=1_000.0)
        verdict = book.admits("mint", now=1_000_000.0)
        # Even long past the cooldown, and even though the window would have
        # dropped it, the bar is on the reason, not the clock.
        self.assertEqual(verdict.status, "REJECTED")
        self.assertIs(verdict.disposition, ExitDisposition.CATASTROPHIC)
        self.assertIn(ExitDisposition.CATASTROPHIC, BARRED_DISPOSITIONS)

    def test_cooldown_blocks_buying_back_into_our_own_exit_impact(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=90.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        early = book.admits("mint", now=1_030.0)
        self.assertEqual(early.status, "REJECTED")
        self.assertIn("own exit impact", early.detail)
        self.assertTrue(book.admits("mint", now=1_100.0).admitted)

    def test_hazard_exit_requires_the_hazard_to_have_actually_fallen(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0, min_hazard_improvement=0.25))
        book.record_exit("mint", "rug_hazard_critical", exited_at=1_000.0, hazard_at_exit=0.80)
        # Unchanged: the reason we left still holds.
        stale = book.admits("mint", now=1_100.0, hazard_now=0.79)
        self.assertEqual(stale.status, "REJECTED")
        self.assertIn("has not fallen", stale.detail)
        # Materially lower: now it is a question worth asking.
        self.assertTrue(book.admits("mint", now=1_100.0, hazard_now=0.55).admitted)

    def test_an_unmeasured_hazard_is_not_an_improved_one(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "rug_hazard_critical", exited_at=1_000.0, hazard_at_exit=0.8)
        blocked = book.admits("mint", now=1_100.0, hazard_now=None)
        self.assertEqual(blocked.status, "DATA_BLOCKED")
        self.assertFalse(blocked.admitted)

    def test_an_exit_taken_without_a_hazard_reading_cannot_be_cleared_later(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "rug_hazard_critical", exited_at=1_000.0, hazard_at_exit=None)
        verdict = book.admits("mint", now=1_100.0, hazard_now=0.01)
        self.assertEqual(verdict.status, "DATA_BLOCKED")
        self.assertIn("never quantified", verdict.detail)

    def test_unescapable_mechanisms_bar_reentry_from_either_side(self):
        unescapable = next(iter(UNESCAPABLE_MECHANISMS))
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("a", "profit_ratchet_5x", exited_at=1_000.0,
                         mechanism_at_exit=unescapable)
        self.assertEqual(book.admits("a", now=1_100.0).status, "REJECTED")
        book.record_exit("b", "profit_ratchet_5x", exited_at=1_000.0)
        self.assertEqual(
            book.admits("b", now=1_100.0, mechanism_now=unescapable).status, "REJECTED")

    def test_window_expiry_makes_it_an_ordinary_candidate_again(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0, window_seconds=600.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        verdict = book.admits("mint", now=1_000.0 + 601.0)
        self.assertTrue(verdict.admitted)
        self.assertIsNone(book.get("mint"))

    def test_a_stale_prediction_cannot_be_reused_across_the_exit(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=2_000.0)
        verdict = self._price(book, "mint", prediction_at=1_999.0)
        self.assertEqual(verdict.status, "DATA_BLOCKED")
        self.assertIn("predates the exit", verdict.detail)

    def test_unmeasured_capacity_or_escape_blocks_rather_than_defaults(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        for field in ("exit_capacity_ratio", "escape_probability"):
            verdict = self._price(book, "mint", **{field: None})
            self.assertEqual(verdict.status, "DATA_BLOCKED", field)

    def test_a_hazard_exit_costs_more_to_undo_than_a_banked_one(self):
        banked = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        banked.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        fled = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        fled.record_exit("mint", "rug_hazard_high", exited_at=1_000.0, hazard_at_exit=0.5)
        cheap = self._price(banked, "mint")
        dear = self._price(fled, "mint")
        self.assertIsNotNone(cheap.required_q)
        self.assertIsNotNone(dear.required_q)
        self.assertGreater(dear.required_q, cheap.required_q)
        # The gross edge is identical; only the bar moved.
        self.assertAlmostEqual(cheap.q, dear.q)

    def test_the_premium_is_at_least_the_round_trip_it_costs(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        verdict = self._price(book, "mint", size_fraction=0.05,
                              entry_cost=0.02, exit_cost=0.02)
        self.assertGreaterEqual(verdict.required_q, 0.05 * (0.02 + 0.02))

    def test_a_marginal_reentry_is_rejected_and_a_strong_one_admitted(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        thin = self._price(book, "mint", bins=[(0.5, 0.05), (0.5, -0.04)])
        self.assertEqual(thin.status, "REJECTED")
        self.assertIn("does not clear", thin.detail)
        strong = self._price(book, "mint", bins=[(0.5, 4.0), (0.5, -0.4)])
        self.assertEqual(strong.status, "OK")
        self.assertGreater(strong.q, strong.required_q)

    def test_each_completed_reentry_raises_the_bar_for_the_next(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0, max_reentries=5))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        first = self._price(book, "mint")
        book.note_reentry("mint")
        second = self._price(book, "mint")
        self.assertGreater(second.required_q, first.required_q)

    def test_the_reentry_count_survives_the_next_exit(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        book.note_reentry("mint")
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_100.0)
        # A token that has already cycled us once must not look like a
        # first-timer merely because it was exited again.
        self.assertEqual(book.get("mint").reentries, 1)

    def test_a_token_that_has_farmed_us_enough_times_is_cut_off(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0, max_reentries=2))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        book.note_reentry("mint")
        book.note_reentry("mint")
        verdict = book.admits("mint", now=1_100.0)
        self.assertEqual(verdict.status, "REJECTED")
        self.assertIn("already re-entered", verdict.detail)

    def test_an_admitted_reentry_contests_for_capital_net_of_its_premium(self):
        book = ReentryBook(ReentryPolicy(cooldown_seconds=0.0))
        book.record_exit("mint", "profit_ratchet_5x", exited_at=1_000.0)
        verdict = self._price(book, "mint", bins=[(0.5, 4.0), (0.5, -0.4)])
        opportunity = verdict.opportunity
        self.assertIsNotNone(opportunity)
        self.assertIsNone(opportunity.blocked_reason)
        self.assertEqual(opportunity.sleeve, "reentry")
        # The allocator ranks the number the trade actually adds, not the
        # gross edge -- otherwise a re-entry scraping past its own bar would
        # outrank a fresh launch that cleared a higher one.
        self.assertAlmostEqual(opportunity.elogw, verdict.q - verdict.required_q)
        self.assertTrue(opportunity.metadata["reentry"])

    def test_the_book_is_bounded_and_drops_the_oldest_exits_first(self):
        book = ReentryBook(ReentryPolicy(window_seconds=10_000.0), capacity=3)
        for index in range(6):
            book.record_exit(f"mint{index}", "profit_ratchet_5x", exited_at=1_000.0 + index)
        self.assertEqual(len(book.candidates(now=1_010.0)), 3)
        self.assertIsNone(book.get("mint0"))
        self.assertIsNotNone(book.get("mint5"))

    def test_report_is_serialisable_and_counts_by_disposition(self):
        book = ReentryBook(ReentryPolicy(window_seconds=10_000.0))
        book.record_exit("a", "profit_ratchet_5x", exited_at=1_000.0)
        book.record_exit("b", "rug_hazard_critical", exited_at=1_000.0, hazard_at_exit=0.7)
        report = book.report(now=1_010.0)
        json.dumps(report)
        self.assertEqual(report["by_disposition"]["banked_strength"], 1)
        self.assertEqual(report["by_disposition"]["hazard_escape"], 1)


class TestReentryIsLiveWired(unittest.TestCase):
    """The last time re-entry was 'wired' it was an unreachable enum member."""

    def test_reenter_is_unreachable_from_an_open_position(self):
        # The reason the enum member was dead: every caller of the action
        # policy in the position loop holds a positive fraction, and REENTER
        # is only scorable at zero. This is asserted so that a future change
        # which appears to "enable" REENTER on open positions has to confront
        # the fact that it would be pricing a flat book.
        policy = ActionValuePolicy()
        state = ActionState(
            held_fraction=0.4, current_multiple=2.0,
            forward_bins=((0.5, 1.0), (0.5, -0.5)),
            exit_capacity_ratio=1.0, escape_probability=1.0,
            reentry_bins=((0.5, 4.0), (0.5, -0.4)), add_fraction=0.05,
        )
        decision = policy.score(state)
        reenter = next(score for score in decision.scores
                       if score.action is ActionValue.REENTER)
        self.assertFalse(reenter.feasible)
        self.assertIn("still open", reenter.detail)

    def test_desk_constructs_a_reentry_book_and_reports_it(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        assigned = {
            node.targets[0].attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and node.targets
            and isinstance(node.targets[0], ast.Attribute)
        }
        self.assertIn("reentry_book", assigned)
        self.assertIn('"reentry": self.reentry_book.report()', source)

    def test_full_exits_are_recorded_and_partial_ones_are_not(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        exit_fn = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_execute_exit")
        calls = [node for node in ast.walk(exit_fn)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "record_exit"]
        self.assertEqual(len(calls), 1)
        # A partial bank leaves the position open, so recording it would let
        # ADD be re-litigated through a path that believes the book is flat.
        # The one call must sit inside the remaining<=0 branch.
        closed_branch = next(
            node for node in ast.walk(exit_fn)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "remaining"
        )
        self.assertTrue(any(call in ast.walk(closed_branch) for call in calls))

    def test_the_entry_path_gates_and_prices_reentries(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        evaluate = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_evaluate_candidate")
        called = {node.func.attr for node in ast.walk(evaluate)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn("admits", called)
        self.assertIn("_price_reentry", called)
        self.assertIn("note_reentry", called)


class TestExecuteExitFeedsTheReentryBook(unittest.IsolatedAsyncioTestCase):
    async def test_a_full_exit_lands_in_the_book_with_its_reason_and_hazard(self):
        result = ExecutionResult(
            success=True, status=TransactionStatus.SIMULATED, simulated=True,
            quoted_output_amount=150_000_000, native_balance_delta_lamports=0,
            actual_input_amount=1_000,
        )
        desk = TestPartialExitAccounting()._desk(result)
        desk.rug_hazard = SimpleNamespace(
            get_hazard=lambda token: SimpleNamespace(hazard_30s=0.62))
        position = TestPartialExitAccounting._position()
        position["high_water_multiple"] = 6.0
        desk.elogw_engine.update_position("mint", position)

        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 1.0, "rug_hazard_critical")

        record = desk.reentry_book.get("mint")
        self.assertIsNotNone(record)
        self.assertIs(record.disposition, ExitDisposition.HAZARD_ESCAPE)
        self.assertAlmostEqual(record.hazard_at_exit, 0.62)
        self.assertAlmostEqual(record.exit_multiple, 6.0)

    async def test_a_partial_exit_leaves_the_book_empty(self):
        result = ExecutionResult(
            success=True, status=TransactionStatus.SIMULATED, simulated=True,
            quoted_output_amount=150_000_000, native_balance_delta_lamports=0,
            actual_input_amount=500,
        )
        desk = TestPartialExitAccounting()._desk(result)
        position = TestPartialExitAccounting._position()
        desk.elogw_engine.update_position("mint", position)

        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 0.5, "profit_ratchet_5x")

        self.assertIsNone(desk.reentry_book.get("mint"))


class TestPartialExitIsNotAFinalOutcome(unittest.IsolatedAsyncioTestCase):
    """A bank of half or more used to be recorded as a closed trade.

    `reduce_position` applies the sale to the position dict, and `_execute_exit`
    then subtracted the same sale again. At 50% the remainder computed to zero
    and at 75% it went negative, so both were treated as closes: log growth was
    attributed to capital still at risk, the outcome row carried a
    `realized_multiple` for a live position, and `_closed_pnl` stayed empty so
    the eventual real close banked the same PnL twice.
    """

    async def _bank(self, fraction: float, sold_tokens: int):
        result = ExecutionResult(
            success=True, status=TransactionStatus.SIMULATED, simulated=True,
            quoted_output_amount=150_000_000, native_balance_delta_lamports=0,
            actual_input_amount=sold_tokens,
        )
        desk = TestPartialExitAccounting()._desk(result)
        position = TestPartialExitAccounting._position()
        desk.elogw_engine.update_position("mint", position)
        await MemecoinQuantDesk._execute_exit(desk, "mint", position, fraction,
                                              "profit_ratchet_5x")
        return desk

    async def test_banking_three_quarters_leaves_the_trade_open(self):
        desk = await self._bank(0.75, 750)
        self.assertIn("mint", desk.elogw_engine.open_positions)
        outcomes = [payload for stream, payload in desk.ops_events
                    if stream == "trade_outcomes"]
        self.assertEqual(outcomes, [])
        self.assertEqual(desk._mechanism_growth, {})
        # The banked PnL is carried, not attributed, so the eventual close
        # reports the whole trade once rather than the last slice twice.
        self.assertAlmostEqual(desk._closed_pnl["mint"], 75.0)
        self.assertIsNone(desk.reentry_book.get("mint"))

    async def test_the_closing_slice_reports_the_whole_trade_once(self):
        desk = await self._bank(0.75, 750)
        position = desk.elogw_engine.open_positions["mint"]
        desk.execution_engine = FakeExecutionEngineForExit(ExecutionResult(
            success=True, status=TransactionStatus.SIMULATED, simulated=True,
            quoted_output_amount=37_500_000, native_balance_delta_lamports=0,
            actual_input_amount=250,
        ))
        await MemecoinQuantDesk._execute_exit(desk, "mint", position, 1.0,
                                              "profit_ratchet_10x")

        self.assertNotIn("mint", desk.elogw_engine.open_positions)
        outcomes = [payload for stream, payload in desk.ops_events
                    if stream == "trade_outcomes"]
        self.assertEqual(len(outcomes), 1)
        # $150.00 banked against a $75.00 basis, then $37.50 against $25.00.
        self.assertAlmostEqual(outcomes[0]["realized_pnl_usd"], 87.5)
        self.assertEqual(len(desk._mechanism_growth["t0_sniper"]), 1)
        self.assertIsNotNone(desk.reentry_book.get("mint"))


class TestOrphanIntelligence(unittest.IsolatedAsyncioTestCase):
    """One launch through the whole chain, with every module made to answer.

    Four components were once reported wired and were not: the edits that
    should have added their imports silently failed to match, the tests bound
    fakes into every collaborator slot so the real constructor was never
    exercised, and the modules sat fully built and completely unreachable
    while the suite stayed green. Nothing raises when a module is simply never
    called, which is why nothing caught it.

    This is the detector. A real desk is constructed -- no fakes in the slots
    that matter -- one launch is pushed through it, and every contributor the
    manifest declares must show up in the decision. MISSING fails. DATA_BLOCKED
    passes, because a source mesh with no transport and a fee table published
    as an image are honest silences, and demanding numbers from them is how
    fabricated inputs get in.
    """

    MINT = "So11111111111111111111111111111111111111112"

    async def _desk(self):
        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True, offline=True)
        await desk.initialize()
        return desk

    @staticmethod
    def _candidate(mint):
        return TokenCandidate(
            address=mint, chain="solana", source=DetectionSource.FACTORY, block_number=0,
            deployer="Deployer1111111111111111111111111111111",
            factory="pump", pair="Pair111111111111111111111111111111111111",
            base_token=WSOL_MINT, timestamp=time.time(),
        )

    async def test_a_bare_launch_leaves_no_module_unconsulted(self):
        desk = await self._desk()
        intelligence = desk._entry_intelligence(
            self.MINT, self._candidate(self.MINT), None, None, {}, 0.0)
        report = audit_intelligence("entry", intelligence)

        # The assertion that matters. Every declared contributor answered, even
        # if the answer was "I cannot measure that".
        self.assertEqual(report.orphans, [], f"orphaned intelligence: {report.orphans}")
        self.assertEqual(report.unknown_keys, [])
        self.assertEqual(report.coverage, 1.0)

    async def test_every_declared_contributor_has_a_slot(self):
        desk = await self._desk()
        intelligence = desk._entry_intelligence(
            self.MINT, self._candidate(self.MINT), None, None, {}, 0.0)
        for contributor in ENTRY_CONTRIBUTORS:
            self.assertIn(contributor.key, intelligence,
                          f"{contributor.module} has no slot: {contributor.why}")

    async def test_evidence_actually_flows_when_the_modules_have_something_to_say(self):
        """Coverage without evidence would be a manifest of empty boxes."""
        desk = await self._desk()
        candidate = self._candidate(self.MINT)

        # Public sources named the token, and one of them was first.
        desk.hot_state.touch_token(self.MINT)
        for index, source in enumerate(("regional-a", "regional-b")):
            event = Event(source_id=source, source_class=SourceClass.CHAT,
                          source_at=1_000.0 + index, observed_at=1_000.5 + index,
                          text=f"call {self.MINT}", language="en",
                          token_addresses=(self.MINT,))
            desk._source_events.setdefault(self.MINT, []).append(event)
            desk.source_genealogy.record(SourcePost(
                source_id=source, token=self.MINT,
                posted_at=event.source_at, observed_at=event.observed_at))

        # Buyers arrived, and some of them are scored.
        desk.rug_hazard.register_token(self.MINT, {"deployer": candidate.deployer})
        for index in range(6):
            desk.public_coordination.record_trade(self.MINT, {
                "wallet": f"buyer{index}", "side": "buy", "notional_usd": 100.0,
                "timestamp": 1_000.0 + index})
            desk._record_actor_entry(
                self.MINT, {"wallet": f"buyer{index}", "side": "buy",
                            "timestamp": 1_000.0 + index},
                {"notional_usd": 100.0})

        prediction = MultiHeadPrediction(self.MINT, "solana", 0, p_2x=0.6, p_5x=0.4, p_10x=0.2)
        trade_info = {"position_size_sol": 0.5, "position_value_usd": 75.0,
                      "risk_contribution": 0.01}
        intelligence = desk._entry_intelligence(
            self.MINT, candidate, None, prediction, trade_info, 50_000.0)
        report = audit_intelligence("entry", intelligence)

        self.assertEqual(report.orphans, [])
        for key in ("prediction", "sources", "coordination", "sizing", "cost_model"):
            self.assertIn(key, report.contributing,
                          f"{key} was consulted but contributed nothing: {intelligence[key]}")
        # The source mesh knows who spoke first and how late we were to it.
        self.assertEqual(intelligence["sources"]["first_source"], "regional-a")
        self.assertGreater(intelligence["sources"]["first_observation_lag_s"], 0.0)

    async def test_the_record_is_serialisable_because_the_audit_pack_reads_it(self):
        desk = await self._desk()
        intelligence = desk._entry_intelligence(
            self.MINT, self._candidate(self.MINT), None, None, {}, 0.0)
        json.dumps(_jsonable(intelligence))

    async def test_an_unwired_module_is_reported_as_an_orphan_not_as_blocked(self):
        """The distinction the whole manifest exists to draw."""
        desk = await self._desk()
        intelligence = desk._entry_intelligence(
            self.MINT, self._candidate(self.MINT), None, None, {}, 0.0)
        # Disconnecting a module looks like this: the slot is simply absent.
        intelligence.pop("actors")
        report = audit_intelligence("entry", intelligence)
        self.assertEqual(report.orphans, ["actors"])
        self.assertLess(report.coverage, 1.0)
        # Whereas a module that ran and could not answer is not an orphan.
        intelligence["actors"] = {"status": "DATA_BLOCKED", "reason": "no scored buyers"}
        self.assertEqual(audit_intelligence("entry", intelligence).orphans, [])

    async def test_position_decisions_declare_their_contributors_too(self):
        desk = await self._desk()
        position = {"size_tokens": 1_000, "remaining_cost_usd": 100.0,
                    "entry_time": time.time(), "high_water_multiple": 1.0}
        intelligence = desk._position_intelligence(self.MINT, position)
        report = audit_intelligence("position", intelligence)
        self.assertEqual(report.orphans, [])
        for contributor in POSITION_CONTRIBUTORS:
            self.assertIn(contributor.key, intelligence, contributor.why)

    async def test_coverage_is_tracked_across_decisions_for_the_audit_pack(self):
        desk = await self._desk()
        intelligence = desk._entry_intelligence(
            self.MINT, self._candidate(self.MINT), None, None, {}, 0.0)
        tracker = CoverageTracker("entry")
        tracker.record(intelligence)
        intelligence.pop("cost_model")
        tracker.record(intelligence)
        report = tracker.report()
        self.assertEqual(report["status"], "ORPHANED")
        self.assertEqual(report["orphaned"], ["cost_model"])
        # Half the decisions reached it; the trend is the signal, not the state.
        self.assertAlmostEqual(report["rates"]["cost_model"]["orphaned"], 0.5)

    async def test_the_fee_engine_prices_the_round_trip_rather_than_assuming_it(self):
        """Two config constants used to stand in for a schedule that changes."""
        desk = await self._desk()
        legacy = desk._cost_model(self.MINT, at_utc=1_760_000_000.0)
        self.assertEqual(legacy["status"], "OK")
        self.assertFalse(legacy["assumed"])
        self.assertEqual(legacy["round_trip_bps"], 2 * LEGACY_TOTAL_FEE_BPS)
        # Past the dynamic activation the tier table is published only as an
        # image, so the constant is used AND LABELLED, never quoted as measured.
        dynamic = desk._cost_model(self.MINT, at_utc=DYNAMIC_FEE_ACTIVATION_UTC + 1.0)
        self.assertEqual(dynamic["status"], "DATA_BLOCKED_FEE_SCHEDULE")
        self.assertTrue(dynamic["assumed"])

    async def test_an_empty_entity_registry_blocks_rather_than_clears(self):
        """No watched entities means 'we cannot tell', not 'nothing is a copycat'."""
        desk = await self._desk()
        verdict = desk._authenticity(self.MINT, self._candidate(self.MINT))
        self.assertEqual(verdict["status"], "DATA_BLOCKED")
        self.assertEqual(verdict["registry_size"], 0)


class TestIntelligenceCoverageHealthCheck(unittest.TestCase):
    """An orphaned module is a failure, not a warning."""

    def test_an_orphan_is_critical_because_capital_moves_without_it(self):
        checks = check_intelligence_coverage({"intelligence_coverage": {
            "entry": {"decisions": 40, "orphaned": ["actors", "cost_model"]},
            "position": {"decisions": 40, "orphaned": []},
        }})
        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["intelligence_coverage_entry"].state, State.CRITICAL)
        self.assertEqual(by_name["intelligence_coverage_position"].state, State.OK)

    def test_no_decisions_yet_is_blocked_not_healthy(self):
        checks = check_intelligence_coverage({"intelligence_coverage": {
            "entry": {"decisions": 0}, "position": {"decisions": 0}}})
        self.assertTrue(all(check.state is State.DATA_BLOCKED for check in checks))

    def test_a_desk_that_reports_nothing_is_blocked(self):
        checks = check_intelligence_coverage({})
        self.assertEqual(len(checks), 1)
        self.assertIs(checks[0].state, State.DATA_BLOCKED)


class TestSubSecondPointInTime(unittest.IsolatedAsyncioTestCase):
    """A dataset whose first post-launch row is at one second is not a sniper's.

    Everything that decides a launch happens inside the first half-second: the
    funding burst, the first twenty-five buyers, whether the deployer bought
    his own mint. Asking "was it better to enter at 100ms than at 500ms" needs
    rows at 100ms and 500ms, and until now the nearest evidence was a full
    second later -- by which time the answer has already been decided.
    """

    def _builder(self):
        builder = PointInTimeDatasetBuilder.__new__(PointInTimeDatasetBuilder)
        builder.chain_config = SimpleNamespace(name="solana")
        builder.active_episodes = {}
        builder._snapshot_inflight = set()
        builder.snapshot_times = dict(SNAPSHOT_OFFSETS_S)
        builder._fast_interval = 0.01
        builder._slow_interval = 1.0
        builder._subsecond_horizon = 0.75

        async def ok(episode, as_of):
            return {"status": "OK", "as_of": as_of}

        for name in ("deployer", "wallet", "flow", "liquidity", "social",
                     "token", "market", "entity_graph"):
            setattr(builder, f"_capture_{name}_features", ok)
        return builder

    def test_the_subsecond_rungs_exist_and_are_ordered(self):
        offsets = [SNAPSHOT_OFFSETS_S[point] for point in (
            SnapshotTimepoint.T0, SnapshotTimepoint.T50MS, SnapshotTimepoint.T100MS,
            SnapshotTimepoint.T250MS, SnapshotTimepoint.T500MS, SnapshotTimepoint.T1S)]
        self.assertEqual(offsets, [0, 0.05, 0.1, 0.25, 0.5, 1])
        self.assertEqual(sorted(offsets), offsets)

    async def test_the_cutoff_is_the_instant_the_row_claims_not_the_wall_clock(self):
        """The leak that made every labelled horizon longer than its label.

        `as_of` used to be `time.time()` at capture. With a loop that woke once
        a second, a row labelled T1S routinely carried whatever had arrived by
        1.4s -- and at 50ms that overshoot would be several times the horizon
        itself, making every sub-second row a fiction.
        """
        builder = self._builder()
        episode = LaunchEpisode(token="mint", chain="solana", created_at=1_000.0,
                                deployer="dev", factory="", pair="", base_token="")
        builder.active_episodes["mint"] = episode

        await PointInTimeDatasetBuilder._capture_snapshot(
            builder, "mint", SnapshotTimepoint.T100MS)

        snapshot = episode.snapshots[SnapshotTimepoint.T100MS]
        self.assertAlmostEqual(snapshot.timestamp, 1_000.1)
        # Every feature group was asked for state as of that instant, not now.
        for group in ("deployer_features", "flow_features", "market_features"):
            self.assertAlmostEqual(getattr(snapshot, group)["as_of"], 1_000.1)

    async def test_capture_lag_is_recorded_rather_than_hidden(self):
        builder = self._builder()
        episode = LaunchEpisode(token="mint", chain="solana", created_at=1_000.0,
                                deployer="dev", factory="", pair="", base_token="")
        builder.active_episodes["mint"] = episode
        await PointInTimeDatasetBuilder._capture_snapshot(
            builder, "mint", SnapshotTimepoint.T50MS)
        snapshot = episode.snapshots[SnapshotTimepoint.T50MS]
        # The row was materialised long after the instant it describes. That
        # does not corrupt it -- the cutoff protects the contents -- but it is
        # the number that says whether the loop is keeping up.
        self.assertIsNotNone(snapshot.capture_lag_s)
        self.assertGreater(snapshot.capture_lag_s, 0.0)

    def test_the_loop_runs_hot_only_while_a_subsecond_target_is_pending(self):
        builder = self._builder()
        now = time.time()
        builder.active_episodes["young"] = LaunchEpisode(
            token="young", chain="solana", created_at=now, deployer="d",
            factory="", pair="", base_token="")
        self.assertEqual(
            PointInTimeDatasetBuilder._loop_interval(builder), builder._fast_interval)

        builder.active_episodes["young"].created_at = now - 5.0
        self.assertEqual(
            PointInTimeDatasetBuilder._loop_interval(builder), builder._slow_interval)

    def test_a_one_second_sweep_could_never_have_taken_a_fifty_millisecond_row(self):
        """Why the cadence had to change, stated as an assertion."""
        builder = self._builder()
        self.assertLess(builder._fast_interval,
                        SNAPSHOT_OFFSETS_S[SnapshotTimepoint.T50MS])
        self.assertGreater(builder._subsecond_horizon,
                           SNAPSHOT_OFFSETS_S[SnapshotTimepoint.T500MS])

    def test_the_shadow_trainer_reads_the_subsecond_rows(self):
        for point in ("t50ms", "t100ms", "t250ms", "t500ms"):
            self.assertIn(point, SNAPSHOT_ORDER)
        # Earliest first, so a model is trained from the state it will be
        # asked about rather than from ten seconds after the decision.
        self.assertEqual(SNAPSHOT_ORDER[0], "t50ms")


class TestExtendedTailLabels(unittest.TestCase):
    def test_every_survival_rung_has_a_label(self):
        for _, multiple in SURVIVAL_LEVELS:
            self.assertIn(int(multiple), TAIL_THRESHOLDS)

    def test_labels_are_written_for_the_whole_tail(self):
        snapshot = LaunchSnapshot(token="mint", chain="solana",
                                  snapshot_time=SnapshotTimepoint.T0, timestamp=0.0)
        for threshold in TAIL_THRESHOLDS:
            self.assertTrue(hasattr(snapshot, f"label_{threshold}x"),
                            f"no label_{threshold}x field")

    def test_a_freak_outcome_is_distinguishable_from_an_ordinary_good_one(self):
        snapshot = LaunchSnapshot(token="mint", chain="solana",
                                  snapshot_time=SnapshotTimepoint.T0, timestamp=0.0)
        freak = LaunchSnapshot(token="other", chain="solana",
                               snapshot_time=SnapshotTimepoint.T0, timestamp=0.0)
        for threshold in TAIL_THRESHOLDS:
            setattr(snapshot, f"label_{threshold}x", 60 >= threshold)
            setattr(freak, f"label_{threshold}x", 600 >= threshold)
        # Under the old label set both were simply "label_50x = True".
        self.assertTrue(snapshot.label_50x)
        self.assertTrue(freak.label_50x)
        self.assertFalse(snapshot.label_500x)
        self.assertTrue(freak.label_500x)

    def test_the_feasible_multiple_target_is_no_longer_capped_at_fifty(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "research"
                  / "shadow_trainer.py").read_text()
        self.assertNotIn("np.clip(feasible, 0.02, 50)", source)
        self.assertIn("SURVIVAL_LEVELS[-1][1]", source)


class TestAgeBandedBrains(unittest.TestCase):
    """A launch at 100ms and a launch at five minutes are different objects.

    The pooled model was trained on every horizon at once, so it learned the
    average launch -- and the average was dominated by whichever horizon
    produced the most rows. The decisions that matter most were being priced
    by a model fitted mostly to states that arrive long after the decision.
    """

    def test_bands_partition_the_whole_timeline_without_gaps(self):
        self.assertEqual(AGE_BANDS[0][1], 0.0)
        self.assertEqual(AGE_BANDS[-1][2], float("inf"))
        for (_, _, high), (_, low, _) in zip(AGE_BANDS, AGE_BANDS[1:]):
            self.assertEqual(high, low)

    def test_the_subsecond_rungs_all_land_in_one_band(self):
        """Otherwise the sub-second rows would be split across two brains."""
        for offset in (0.0, 0.05, 0.1, 0.25, 0.4):
            self.assertEqual(band_for(offset), "flash")
        self.assertNotEqual(band_for(1.0), "flash")

    def test_band_boundaries_are_closed_below_and_open_above(self):
        self.assertEqual(band_for(0.5), "early")
        self.assertEqual(band_for(0.499), "flash")
        self.assertEqual(band_for(60.0), "mature")
        self.assertEqual(band_for(59.9), "forming")

    def test_age_and_regime_reach_the_feature_array(self):
        """Both lived on the dataclass and never got into the vector."""
        predictor = MultiHeadPredictor()
        self.assertIn("time_since_launch", predictor.feature_names)
        for name in ("regime_bull", "regime_bear", "regime_chop", "regime_euphoria"):
            self.assertIn(name, predictor.feature_names)
        young = PredictionFeatures("m", "solana", 0, time_since_launch=0.1).to_array()
        old = PredictionFeatures("m", "solana", 0, time_since_launch=3600).to_array()
        self.assertEqual(len(young), len(predictor.feature_names))
        self.assertNotEqual(young.tolist(), old.tolist())

    def test_an_unrecognised_regime_lights_nothing(self):
        vector = PredictionFeatures("m", "solana", 0, regime="something_new").to_array()
        names = MultiHeadPredictor().feature_names
        for name in ("regime_bull", "regime_bear", "regime_chop", "regime_euphoria"):
            self.assertEqual(vector[names.index(name)], 0.0)

    def test_each_band_gets_its_own_directory(self):
        """Sharing one would let load_latest pick up a neighbour's artifact."""
        directories = {band_model_dir("models", band) for band in BAND_NAMES}
        self.assertEqual(len(directories), len(BAND_NAMES))
        for directory in directories:
            self.assertTrue(directory.startswith(os.path.join("models", "bands")))

    def test_an_untrained_band_answers_nothing_rather_than_borrowing_a_neighbour(self):
        predictor = AgeBandedPredictor("/nonexistent-models", allow_pooled_fallback=False)
        features = PredictionFeatures("m", "solana", 0, time_since_launch=0.1)
        self.assertIsNone(predictor.predict(features))
        self.assertFalse(predictor._is_trained)

    def test_a_pooled_answer_is_always_labelled_as_one(self):
        predictor = AgeBandedPredictor("/nonexistent-models", allow_pooled_fallback=True)
        predictor.pooled._is_trained = True
        predictor.pooled.predict = lambda features: MultiHeadPrediction(
            features.token, features.chain, features.timestamp, p_2x=0.5)
        prediction = predictor.predict(
            PredictionFeatures("m", "solana", 0, time_since_launch=0.1))
        self.assertIsNotNone(prediction)
        # Shadow evaluation may run on the bridge; promotion must be able to
        # tell that it did.
        self.assertEqual(prediction.band_status, "POOLED_FALLBACK")
        self.assertEqual(prediction.age_band, "flash")

    def test_a_band_answer_is_labelled_as_its_own(self):
        predictor = AgeBandedPredictor("/nonexistent-models")
        flash = predictor.bands["flash"]
        flash._is_trained = True
        flash.predict = lambda features: MultiHeadPrediction(
            features.token, features.chain, features.timestamp, p_2x=0.9)
        prediction = predictor.predict(
            PredictionFeatures("m", "solana", 0, time_since_launch=0.2))
        self.assertEqual(prediction.band_status, "OWN_BAND")
        self.assertEqual(prediction.age_band, "flash")
        self.assertAlmostEqual(prediction.p_2x, 0.9)

    def test_one_trained_band_does_not_unblock_the_others(self):
        predictor = AgeBandedPredictor("/nonexistent-models", allow_pooled_fallback=False)
        predictor.bands["flash"]._is_trained = True
        predictor.bands["flash"].predict = lambda features: MultiHeadPrediction(
            features.token, features.chain, features.timestamp)
        self.assertTrue(predictor._is_trained)
        self.assertIsNotNone(predictor.predict(
            PredictionFeatures("m", "solana", 0, time_since_launch=0.1)))
        # A mature launch has no brain, and gets no answer.
        self.assertIsNone(predictor.predict(
            PredictionFeatures("m", "solana", 0, time_since_launch=600)))

    def test_the_report_names_which_ages_are_covered(self):
        predictor = AgeBandedPredictor("/nonexistent-models")
        report = predictor.report()
        json.dumps(report)
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertEqual(report["trained_bands"], [])
        predictor.bands["flash"]._is_trained = True
        self.assertEqual(predictor.report()["status"], "PARTIAL")
        for band in BAND_NAMES:
            predictor.bands[band]._is_trained = True
        self.assertEqual(predictor.report()["status"], "OK")

    def test_every_band_is_shown_the_same_columns(self):
        predictor = AgeBandedPredictor("/nonexistent-models")
        for band in BAND_NAMES:
            self.assertEqual(predictor.bands[band].feature_names, predictor.feature_names)


class TestAgeBandTraining(unittest.TestCase):
    def test_a_band_without_enough_rows_is_blocked_not_topped_up(self):
        def sample(age):
            features = PredictionFeatures("m", "solana", age, time_since_launch=age)
            return (features, {}, {})

        train = [sample(0.1) for _ in range(5)]
        oos = [sample(0.1)]
        with tempfile.TemporaryDirectory() as tmp:
            report = train_age_bands(train, oos, Path(tmp), min_band_samples=60)
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertEqual(report["bands"]["flash"]["status"], "DATA_BLOCKED")
        self.assertIn("need_at_least_60", report["bands"]["flash"]["reason"])
        # And the bands with no rows at all are blocked too, not skipped.
        for band in BAND_NAMES:
            self.assertIn(band, report["bands"])

    def test_rows_are_routed_to_the_band_that_owns_their_age(self):
        def sample(age):
            return (PredictionFeatures("m", "solana", age, time_since_launch=age), {}, {})

        train = [sample(0.1), sample(0.2), sample(2.0), sample(30.0), sample(600.0)]
        with tempfile.TemporaryDirectory() as tmp:
            report = train_age_bands(train, [], Path(tmp), min_band_samples=1000)
        self.assertEqual(report["bands"]["flash"]["train_samples"], 2)
        self.assertEqual(report["bands"]["early"]["train_samples"], 1)
        self.assertEqual(report["bands"]["forming"]["train_samples"], 1)
        self.assertEqual(report["bands"]["mature"]["train_samples"], 1)


class TestMechanismDecomposition(unittest.TestCase):
    """The race saw two mechanisms; the hazard model detects seven.

    Both of the two were built from the same aggregate hazard under different
    names, so the same number was fed in twice. Of the five that never
    arrived, three are unescapable -- which meant the mechanisms being dropped
    were exactly the ones that decide whether running is even the right move.
    """

    @staticmethod
    def _signal(trigger, strength=0.8, confidence=0.9):
        return SimpleNamespace(trigger=SimpleNamespace(value=trigger),
                               strength=strength, confidence=confidence)

    def test_every_unescapable_mechanism_is_reachable_from_a_live_trigger(self):
        reachable = {mechanism for mechanism, _ in TRIGGER_MECHANISMS.values()}
        reachable.add(HazardMechanism.AUTHORITY_ABUSE)  # from the safety report
        for mechanism in UNESCAPABLE_MECHANISMS:
            self.assertIn(mechanism, reachable, mechanism.value)

    def test_every_mechanism_the_enum_declares_can_actually_be_produced(self):
        producible = {mechanism for mechanism, _ in TRIGGER_MECHANISMS.values()}
        producible.add(HazardMechanism.AUTHORITY_ABUSE)
        self.assertEqual(producible, set(HazardMechanism))

    def test_decay_triggers_are_not_mistaken_for_ways_of_dying(self):
        """A token can fade for an hour and stay sellable the whole way down."""
        decomposition = mechanisms_from_signals([
            self._signal("buy_deceleration"), self._signal("volume_collapse"),
            self._signal("social_velocity_collapse"), self._signal("sell_acceleration"),
        ])
        self.assertEqual(decomposition.mechanisms, {})
        # Not silently dropped either -- unattributed hazard is reported.
        self.assertEqual(len(decomposition.unattributed_triggers), 4)
        self.assertIn("sell_acceleration", decomposition.unattributed_triggers)

    def test_repeated_evidence_for_one_mechanism_compounds_rather_than_maxes(self):
        one = mechanisms_from_signals([self._signal("liquidity_withdrawal", 0.5, 0.8)])
        two = mechanisms_from_signals([self._signal("liquidity_withdrawal", 0.5, 0.8),
                                       self._signal("liquidity_withdrawal", 0.5, 0.8)])
        single = one.mechanisms[HazardMechanism.LIQUIDITY_REMOVAL][0]
        double = two.mechanisms[HazardMechanism.LIQUIDITY_REMOVAL][0]
        self.assertGreater(double, single)
        # Two independent observations of liquidity leaving are more than one.
        self.assertAlmostEqual(double, 1 - (1 - single) ** 2)

    def test_the_shortest_horizon_wins_for_a_mechanism(self):
        decomposition = mechanisms_from_signals([
            self._signal("insider_sell"),        # 300s
            self._signal("smart_wallet_exit"),   # 300s
            self._signal("creator_transfer"),    # 30s
        ])
        self.assertEqual(
            decomposition.mechanisms[HazardMechanism.CREATOR_SELLING][1], 30.0)

    def test_a_live_authority_is_a_standing_mechanism_with_no_trigger(self):
        """No signal fires for a capability that is merely present."""
        blocked = mechanisms_from_signals([], authority_live=True)
        self.assertIn(HazardMechanism.AUTHORITY_ABUSE, blocked.mechanisms)
        self.assertIn(HazardMechanism.AUTHORITY_ABUSE, UNESCAPABLE_MECHANISMS)
        clean = mechanisms_from_signals([], authority_live=False)
        self.assertNotIn(HazardMechanism.AUTHORITY_ABUSE, clean.mechanisms)
        # Unmeasured is not clean.
        unknown = mechanisms_from_signals([], authority_live=None)
        self.assertNotIn(HazardMechanism.AUTHORITY_ABUSE, unknown.mechanisms)

    def test_the_report_is_serialisable(self):
        json.dumps(mechanisms_from_signals(
            [self._signal("route_degradation")], authority_live=True).report())


class TestLearnedExitLatency(unittest.TestCase):
    """The escape race was run against a config constant."""

    def test_below_the_sample_floor_it_refuses_rather_than_guesses(self):
        latency = LandingLatency()
        for _ in range(5):
            latency.record(400, landed=True)
        estimate = latency.estimate()
        self.assertEqual(estimate.status, "DATA_BLOCKED")
        self.assertIsNone(estimate.seconds)
        self.assertIn("have 5", estimate.detail)

    def test_a_paper_fill_teaches_it_nothing(self):
        latency = LandingLatency()
        for _ in range(50):
            self.assertFalse(latency.record(400, landed=True, simulated=True))
        self.assertEqual(latency.estimate().observations, 0)

    def test_a_submission_that_never_landed_has_no_latency(self):
        """Counting a timeout would make a failing relay look merely slow."""
        latency = LandingLatency()
        for _ in range(50):
            self.assertFalse(latency.record(30_000, landed=False))
        self.assertEqual(latency.estimate().observations, 0)

    def test_it_prices_the_slow_race_not_the_typical_one(self):
        latency = LandingLatency()
        for value in list(range(100, 1_100, 50)):
            latency.record(value, landed=True)
        estimate = latency.estimate()
        self.assertEqual(estimate.status, "OK")
        median = np.median([value / 1000 for value in range(100, 1_100, 50)])
        # The race that matters is the one run while something is collapsing.
        self.assertGreater(estimate.seconds, median)

    def test_nonsense_measurements_are_refused(self):
        latency = LandingLatency()
        for value in (0, -5, None, "slow", float("inf")):
            self.assertFalse(latency.record(value, landed=True))
        self.assertEqual(latency.estimate().observations, 0)

    def test_the_window_is_bounded_so_old_conditions_age_out(self):
        latency = LandingLatency(capacity=20)
        for _ in range(100):
            latency.record(400, landed=True)
        self.assertEqual(latency.estimate().observations, 20)


class TestEscapeUsesEveryMechanism(unittest.IsolatedAsyncioTestCase):
    def _desk(self, signals, *, risk=None, latency=None):
        desk = SimpleNamespace(
            rug_hazard=SimpleNamespace(observations={}),
            global_config={"acceptable_exit_impact": 0.10,
                           "expected_exit_latency_s": 0.4},
            _latest_curve_state={}, _latest_pool_state={},
            landing_latency=latency or LandingLatency(),
        )
        desk.hazard = SimpleNamespace(signals=signals)
        return desk

    def test_all_seven_mechanisms_reach_the_curve(self):
        desk = self._desk([])
        position = {"size_tokens": 1_000, "risk_object": SimpleNamespace(
            data_status="OK", can_mint=True, can_freeze=False)}
        signals = [SimpleNamespace(trigger=SimpleNamespace(value=name),
                                   strength=0.6, confidence=0.9)
                   for name in TRIGGER_MECHANISMS]
        hazard = SimpleNamespace(signals=signals)
        MemecoinQuantDesk._estimate_escape(desk, "mint", position, hazard)
        reported = position["hazard_mechanisms"]["mechanisms"]
        for mechanism in HazardMechanism:
            self.assertIn(mechanism.value, reported, mechanism.value)

    def test_a_measured_latency_replaces_the_configured_one(self):
        measured = LandingLatency()
        for value in range(1_000, 2_200, 50):
            measured.record(value, landed=True)
        desk = self._desk([], latency=measured)
        position = {"size_tokens": 1_000}
        MemecoinQuantDesk._estimate_escape(
            desk, "mint", position,
            SimpleNamespace(signals=[SimpleNamespace(
                trigger=SimpleNamespace(value="creator_transfer"),
                strength=0.5, confidence=0.9)]))
        self.assertEqual(position["exit_latency"]["status"], "OK")
        # The measured tail is seconds, not the configured 0.4.
        self.assertGreater(position["exit_latency"]["seconds"], 1.0)

    def test_an_unmeasured_latency_falls_back_but_says_so(self):
        desk = self._desk([])
        position = {"size_tokens": 1_000}
        MemecoinQuantDesk._estimate_escape(
            desk, "mint", position,
            SimpleNamespace(signals=[SimpleNamespace(
                trigger=SimpleNamespace(value="creator_transfer"),
                strength=0.5, confidence=0.9)]))
        self.assertEqual(position["exit_latency"]["status"], "DATA_BLOCKED")


class TestWalletObservationPath(unittest.IsolatedAsyncioTestCase):
    """Elite wallets were watched by walking a hundred HTTP calls every 2s.

    Sequentially. Even at fifty milliseconds a request that is five seconds of
    work inside a two-second loop, so the hundredth wallet was never on a
    two-second delay -- it was on whatever the queue happened to be, and
    nothing measured which. Meanwhile the geyser stream was already carrying
    every trade on the programs we subscribe to, in tens of milliseconds, and
    the poll was duplicating it.
    """

    def _engine(self):
        engine = WalletIntelligenceEngine.__new__(WalletIntelligenceEngine)
        engine._live_watch_wallets = set()
        engine._stream_seen_at = {}
        engine._observation_lag = {"stream": deque(maxlen=512), "poll": deque(maxlen=512)}
        engine._recent_buys = deque(maxlen=100)
        engine._recent_sells = deque(maxlen=100)
        engine._queued_history_wallets = set()
        engine._history_candidates = deque(maxlen=100)
        engine._history_evaluated_at = {}
        engine.stream_coverage_seconds = 120.0
        engine.max_reconcile_per_pass = 25
        engine.reconcile_concurrency = 8
        engine.data_status = {}
        return engine

    def test_a_stream_covered_wallet_is_not_polled(self):
        engine = self._engine()
        engine._live_watch_wallets = {"a", "b"}
        engine._stream_seen_at["a"] = time.time()
        self.assertEqual(engine.stale_watch_wallets(), ["b"])

    def test_the_budget_goes_to_the_wallets_we_know_least_about(self):
        engine = self._engine()
        now = time.time()
        engine._live_watch_wallets = {"fresh", "stale", "ancient"}
        engine._stream_seen_at = {"fresh": now - 200, "stale": now - 600,
                                  "ancient": now - 5_000}
        self.assertEqual(engine.stale_watch_wallets(now),
                         ["ancient", "stale", "fresh"])

    def test_a_wallet_never_seen_on_the_stream_sorts_first(self):
        engine = self._engine()
        now = time.time()
        engine._live_watch_wallets = {"seen", "never"}
        engine._stream_seen_at = {"seen": now - 300}
        self.assertEqual(engine.stale_watch_wallets(now)[0], "never")

    def test_a_stream_trade_marks_the_wallet_covered(self):
        engine = self._engine()
        engine._live_watch_wallets = {"w"}
        self.assertEqual(engine.stale_watch_wallets(), ["w"])
        WalletIntelligenceEngine.record_live_trade(engine, "mint", {
            "wallet": "w", "side": "buy", "amount": 1.0, "price": 0.001,
            "timestamp": time.time()})
        self.assertEqual(engine.stale_watch_wallets(), [])

    def test_observation_lag_is_measured_per_path_not_assumed(self):
        engine = self._engine()
        WalletIntelligenceEngine.record_live_trade(engine, "mint", {
            "wallet": "w", "side": "buy", "amount": 1.0, "price": 0.001,
            "timestamp": time.time() - 0.05})
        report = engine.coverage_report()
        self.assertEqual(report["observation_lag"]["stream"]["observations"], 1)
        self.assertLess(report["observation_lag"]["stream"]["median_s"], 1.0)
        # The poll path has said nothing, and reports nothing rather than zero.
        self.assertIsNone(report["observation_lag"]["poll"]["median_s"])

    def test_a_nonsense_timestamp_does_not_enter_the_latency_record(self):
        engine = self._engine()
        for stamp in (None, "soon", time.time() + 500, 0):
            engine._note_observation_lag("stream", stamp)
        self.assertEqual(len(engine._observation_lag["stream"]), 0)

    def test_the_coverage_split_is_reported_rather_than_assumed(self):
        engine = self._engine()
        now = time.time()
        engine._live_watch_wallets = {"a", "b", "c"}
        engine._stream_seen_at = {"a": now, "b": now - 10}
        report = engine.coverage_report(now)
        json.dumps(report)
        self.assertEqual(report["watched"], 3)
        self.assertEqual(report["stream_covered"], 2)
        self.assertEqual(report["awaiting_reconciliation"], 1)

    async def test_reconciliation_is_concurrent_and_bounded(self):
        engine = self._engine()
        engine.helius_key = "k"
        engine.max_reconcile_per_pass = 4
        engine._live_watch_wallets = {f"w{index}" for index in range(20)}
        engine._helius_base = "http://local"
        calls = []
        active = {"now": 0, "peak": 0}

        class _Response:
            status = 500

            async def __aenter__(self_inner):
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
                await asyncio.sleep(0.01)
                return self_inner

            async def __aexit__(self_inner, *args):
                active["now"] -= 1
                return False

        engine._session = SimpleNamespace(
            get=lambda url, params=None: (calls.append(url), _Response())[1])

        await WalletIntelligenceEngine._watch_live_wallets(engine)

        # Bounded per pass, so one pass cannot outlast its own interval.
        self.assertEqual(len(calls), 4)
        self.assertGreater(active["peak"], 1)
        self.assertIn("reconciled 4 of 20", engine.data_status["live_wallet_watch"])

    async def test_full_stream_coverage_skips_the_poll_entirely(self):
        engine = self._engine()
        engine.helius_key = "k"
        engine._live_watch_wallets = {"a", "b"}
        now = time.time()
        engine._stream_seen_at = {"a": now, "b": now}
        engine._session = SimpleNamespace(
            get=lambda *args, **kwargs: self.fail("polled a stream-covered wallet"))
        await WalletIntelligenceEngine._watch_live_wallets(engine)
        self.assertIn("stream covers every watched wallet",
                      engine.data_status["live_wallet_watch"])


class TestDecisionContribution(unittest.TestCase):
    """Coverage says a module ran. This says whether it mattered.

    A component that is disconnected and one that is connected but inert both
    look like trades that would have happened anyway, and only one of them is
    fixed by wiring.
    """

    @staticmethod
    def _state(**overrides):
        base = dict(
            held_fraction=0.3, current_multiple=4.0,
            forward_bins=((0.4, 2.0), (0.6, -0.6)),
            exit_capacity_ratio=0.5, escape_probability=0.4,
            alternative_growth_per_second=1e-5, expected_remaining_seconds=120.0,
            add_fraction=0.02, exit_cost=0.02, entry_cost=0.02,
        )
        base.update(overrides)
        return ActionState(**base)

    def test_measuring_escape_can_only_remove_optimism(self):
        """The baseline is 1.0, which is what the desk assumed before it measured.

        So the contribution is the amount of optimism the measurement took
        away, and it can never be negative -- if it were, measuring escape
        would be making positions look MORE liquid than assuming they were
        perfectly escapable, which is arithmetically impossible.
        """
        result = action_value_contributions(ActionValuePolicy(), self._state())
        escape = next(item for item in result.contributions
                      if item.component == "escape_probability")
        self.assertEqual(escape.status, "OK")
        self.assertGreater(escape.delta_q, 0.0)

    def test_an_input_absent_from_a_decision_is_reported_not_omitted(self):
        """'Did not contribute' and 'was not present' are different sentences."""
        result = action_value_contributions(ActionValuePolicy(), self._state())
        replacement = next(item for item in result.contributions
                           if item.component == "replacement_bins")
        self.assertEqual(replacement.status, "NOT_MEASURED")
        self.assertIsNone(replacement.delta_q)

    def test_a_decisive_input_is_named_as_such(self):
        """The strongest claim: without it the desk would have done something else."""
        policy = ActionValuePolicy()
        # A position worth holding, but only because a replacement is on offer.
        state = self._state(
            forward_bins=((0.5, 0.6), (0.5, -0.2)),
            escape_probability=1.0, exit_capacity_ratio=1.0,
            alternative_growth_per_second=None, add_fraction=None,
            replacement_bins=((0.6, 4.0), (0.4, -0.5)), replacement_fraction=0.2)
        decision = policy.score(state)
        result = action_value_contributions(policy, state)
        if decision.action is Action.REPLACE:
            self.assertIn("replacement_bins", result.decisive)

    def test_an_unpriceable_ablation_is_blocked_not_scored_as_zero(self):
        policy = ActionValuePolicy()
        result = action_value_contributions(policy, self._state())
        for item in result.contributions:
            if item.status == "OK":
                self.assertIsNotNone(item.delta_q)
            else:
                self.assertIsNone(item.delta_q)

    def test_a_blocked_state_produces_no_attribution_rather_than_a_fake_one(self):
        blocked = self._state(escape_probability=None)
        self.assertIsNone(action_value_contributions(ActionValuePolicy(), blocked))

    def test_the_ledger_finds_a_component_that_never_moves_anything(self):
        ledger = ContributionLedger()
        for _ in range(30):
            ledger.record(DecisionContribution(
                token="m", action="hold", q=0.0, contributions=[
                    Contribution("escape_probability", 0.02),
                    Contribution("exit_capacity_ratio", 0.0),
                ]))
        report = ledger.report()
        json.dumps(report)
        # Consulted every time, never changed anything. Not disconnected, and
        # not working either.
        self.assertEqual(report["inert_components"], ["exit_capacity_ratio"])
        self.assertAlmostEqual(
            report["components"]["escape_probability"]["share_nonzero"], 1.0)

    def test_sign_is_preserved_because_direction_is_the_finding(self):
        ledger = ContributionLedger()
        ledger.record(DecisionContribution("m", "exit", 0.1, [
            Contribution("escape_probability", -0.5)]))
        ledger.record(DecisionContribution("m", "exit", 0.1, [
            Contribution("escape_probability", 0.5)]))
        stats = ledger.report()["components"]["escape_probability"]
        # Averaging absolute values would hide an input that consistently
        # pushes toward worse actions.
        self.assertAlmostEqual(stats["mean_delta_q"], 0.0)
        self.assertAlmostEqual(stats["share_nonzero"], 1.0)

    def test_a_gate_that_vetoes_is_counted_even_though_it_shifts_no_q(self):
        ledger = ContributionLedger()
        ledger.record_gate(GateFlip("reentry_premium", "m", before=True, after=False))
        ledger.record_gate(GateFlip("reentry_premium", "m", before=True, after=True))
        gates = ledger.report()["gates"]
        self.assertEqual(gates["reentry_premium"]["evaluated"], 2)
        self.assertEqual(gates["reentry_premium"]["flipped"], 1)

    def test_an_empty_ledger_is_blocked_not_healthy(self):
        report = ContributionLedger().report()
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertEqual(report["decisions"], 0)

    def test_the_ledger_is_bounded(self):
        ledger = ContributionLedger(capacity=10)
        for _ in range(100):
            ledger.record(DecisionContribution("m", "hold", 0.0, [
                Contribution("escape_probability", 0.01)]))
        self.assertEqual(
            ledger.report()["components"]["escape_probability"]["observations"], 10)

    def test_it_is_wired_into_the_position_decision(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        scorer = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "_score_actions")
        called = {node.func.id for node in ast.walk(scorer)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("action_value_contributions", called)


class TestNativePumpRoute(unittest.TestCase):
    """Entry required two Jupiter round trips before a byte of the transaction existed.

    On the one path where latency is the entire product. Both were avoidable:
    the curve state already arrives on the stream, the pricing already runs
    locally with Rust parity, and every account buy_v2 and sell_v2 need is a
    fixed program, a derivable PDA, or an associated token account.
    """

    MINT = "So11111111111111111111111111111111111111112"
    OTHER = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def _route(self):
        return NativePumpRoute()

    def test_the_derivation_reproduces_an_address_the_docs_state_outright(self):
        """The one check that says the seeds and the program id are both right.

        pump-public-docs states the global config is
        4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf and separately that it is
        PDA-derived from ["global"]. Deriving one and getting the other is an
        independent confirmation of both readings.
        """
        self.assertEqual(ROUTE_STATUS, "OK")
        self.assertEqual(derived_global(), PUBLISHED_GLOBAL)

    def test_every_program_address_comes_from_its_own_idl(self):
        self.assertEqual(program_id(PUMP_IDL), PUMP_PROGRAM)
        self.assertEqual(program_id(PUMP_AMM_IDL),
                         "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA")
        self.assertEqual(program_id(PUMP_FEES_IDL),
                         "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ")

    def test_discriminators_agree_with_anchors_derivation(self):
        """A mismatch means the IDL is for a different program or the name is wrong."""
        for idl, name in ((PUMP_IDL, "buy_v2"), (PUMP_IDL, "sell_v2"),
                          (PUMP_AMM_IDL, "buy"), (PUMP_AMM_IDL, "sell")):
            expected = hashlib.sha256(f"global:{name}".encode()).digest()[:8]
            self.assertEqual(discriminator(idl, name), expected, f"{idl}.{name}")

    def test_the_idl_contradicts_the_prose_on_three_flags(self):
        """The reason the account lists are generated rather than transcribed.

        docs/instructions/BUY.md presents fee_recipient and
        buyback_fee_recipient as non-writable and global_volume_accumulator as
        writable. The program says the opposite on all three, and a
        transaction that declares the wrong mutability fails without pointing
        at why.
        """
        route = self._route()
        buy = route.build_buy(self.MINT, self.MINT, self.OTHER, 1_000, 5_000)
        self.assertEqual(buy.status, "OK")
        by_name = dict(zip(account_names(PUMP_IDL, "buy_v2"), buy.accounts))
        self.assertTrue(by_name["fee_recipient"].is_writable)
        self.assertTrue(by_name["buyback_fee_recipient"].is_writable)
        self.assertFalse(by_name["global_volume_accumulator"].is_writable)

    def test_sharing_config_is_derived_under_the_fees_program_not_pump(self):
        route = self._route()
        buy = route.build_buy(self.MINT, self.MINT, self.OTHER, 1_000, 5_000)
        by_name = dict(zip(account_names(PUMP_IDL, "buy_v2"), buy.accounts))
        under_fees, _ = Pubkey.find_program_address(
            [b"sharing-config", bytes(Pubkey.from_string(self.MINT))],
            Pubkey.from_string(program_id(PUMP_FEES_IDL)))
        under_pump, _ = Pubkey.find_program_address(
            [b"sharing-config", bytes(Pubkey.from_string(self.MINT))],
            Pubkey.from_string(PUMP_PROGRAM))
        self.assertEqual(by_name["sharing_config"].pubkey, str(under_fees))
        self.assertNotEqual(by_name["sharing_config"].pubkey, str(under_pump))

    def test_buy_and_sell_take_the_number_of_accounts_the_idl_declares(self):
        route = self._route()
        buy = route.build_buy(self.MINT, self.MINT, self.OTHER, 1_000, 5_000)
        sell = route.build_sell(self.MINT, self.MINT, self.OTHER, 1_000, 500)
        self.assertEqual(buy.status, "OK")
        self.assertEqual(sell.status, "OK")
        self.assertEqual(len(buy.accounts), len(account_names(PUMP_IDL, "buy_v2")))
        self.assertEqual(len(sell.accounts), len(account_names(PUMP_IDL, "sell_v2")))
        self.assertEqual(len(buy.accounts), 27)
        self.assertEqual(len(sell.accounts), 26)

    def test_only_the_buy_carries_the_global_volume_accumulator(self):
        self.assertIn("global_volume_accumulator", account_names(PUMP_IDL, "buy_v2"))
        self.assertNotIn("global_volume_accumulator", account_names(PUMP_IDL, "sell_v2"))

    def test_exactly_one_signer_and_it_is_the_user(self):
        route = self._route()
        for prepared, name in ((route.build_buy(self.MINT, self.MINT, self.OTHER, 10, 20), "buy_v2"),
                               (route.build_sell(self.MINT, self.MINT, self.OTHER, 10, 5), "sell_v2")):
            signers = [meta for meta in prepared.accounts if meta.is_signer]
            self.assertEqual(len(signers), 1, name)
            self.assertEqual(signers[0].pubkey, self.OTHER)
            # Writable on BOTH, which the prose tables also got wrong.
            self.assertTrue(signers[0].is_writable, name)

    def test_instruction_data_is_discriminator_then_two_little_endian_u64(self):
        route = self._route()
        buy = route.build_buy(self.MINT, self.MINT, self.OTHER, 1_234, 5_678)
        self.assertEqual(buy.data[:8], discriminator(PUMP_IDL, "buy_v2"))
        self.assertEqual(int.from_bytes(buy.data[8:16], "little"), 1_234)
        self.assertEqual(int.from_bytes(buy.data[16:24], "little"), 5_678)
        self.assertEqual(len(buy.data), 24)

    def test_the_wrong_number_of_arguments_is_refused_not_encoded(self):
        with self.assertRaises(IdlError):
            encode_u64_args(PUMP_IDL, "buy_v2", (1,))
        with self.assertRaises(IdlError):
            encode_u64_args(PUMP_IDL, "buy_v2", (1, 2, 3))

    def test_an_account_that_is_neither_a_pda_nor_supplied_raises(self):
        """A silently substituted account touches something nobody chose."""
        with self.assertRaises(IdlError):
            build_accounts(PUMP_IDL, "buy_v2", {})
        self.assertIn("user", unresolvable(PUMP_IDL, "buy_v2", {}))
        self.assertNotIn("bonding_curve", unresolvable(PUMP_IDL, "buy_v2", {}))

    def test_an_unbounded_trade_is_refused(self):
        route = self._route()
        self.assertEqual(route.build_buy(self.MINT, self.MINT, self.OTHER, 100, 0).status,
                         "REJECTED")
        self.assertEqual(route.build_sell(self.MINT, self.MINT, self.OTHER, 100, 0).status,
                         "REJECTED")
        self.assertEqual(route.build_buy(self.MINT, self.MINT, self.OTHER, 0, 100).status,
                         "REJECTED")

    def test_a_curve_without_a_creator_blocks_rather_than_deriving_a_wrong_vault(self):
        blocked = self._route().build_buy(self.MINT, "", self.OTHER, 10, 20)
        self.assertEqual(blocked.status, "DATA_BLOCKED")
        self.assertIn("creator", blocked.detail)

    def test_all_twenty_four_published_recipients_are_parsed(self):
        groups = fee_recipients()
        self.assertEqual(len(groups["normal"]), 8)
        self.assertEqual(len(groups["reserved"]), 8)
        self.assertEqual(len(groups["buyback"]), 8)
        # The three sets are disjoint; picking from the wrong one is rejected
        # on chain, so they must never be conflated.
        self.assertEqual(len(set(groups["normal"]) | set(groups["reserved"])
                             | set(groups["buyback"])), 24)

    def test_recipient_choice_is_deterministic_in_the_mint(self):
        """So a failing transaction reproduces, and two trades on one coin agree."""
        first = select_fee_recipient(self.MINT)
        self.assertEqual(first, select_fee_recipient(self.MINT))
        self.assertIn(first, fee_recipients()["normal"])
        self.assertIn(select_buyback_recipient(self.MINT), fee_recipients()["buyback"])
        # Mayhem coins draw from the reserved set, never the normal one.
        self.assertIn(select_fee_recipient(self.MINT, mayhem=True),
                      fee_recipients()["reserved"])

    def test_the_token_program_is_part_of_the_ata_seed(self):
        """A Token-2022 base and an SPL quote give the same owner different ATAs."""
        base = associated_token_address(self.MINT, self.MINT, TOKEN_2022_PROGRAM)
        quote = associated_token_address(self.MINT, self.MINT, TOKEN_PROGRAM)
        self.assertNotEqual(base, quote)

    def test_the_route_is_no_longer_blocked_on_unpublished_addresses(self):
        report = self._route().report()
        json.dumps(report)
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["buy_accounts"], 27)
        self.assertEqual(report["sell_accounts"], 26)


class TestNativeRouteIsTheCanonicalPath(unittest.TestCase):
    MINT = "So11111111111111111111111111111111111111112"
    OTHER = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def _engine(self, route=None, curve=None):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.pump_route = route
        engine.curve_state_provider = (lambda token: curve) if curve else None
        engine.pumpswap_route = None
        engine.pool_state_provider = None
        engine.pool_account_provider = None
        engine.tx_builder = SimpleNamespace(public_key=self.OTHER)
        engine.native_route_attempts = defaultdict(int)
        engine.stream_confirmations = 0
        engine.poll_confirmations = 0
        engine._signature_waiters = {}
        engine.reconcile_min_interval = 0.005
        engine.landing_model = LandingModel()
        engine.last_bid = {}
        return engine

    @staticmethod
    def _curve(creator=""):
        return BondingCurveState(
            virtual_token_reserves=1_000_000_000_000,
            virtual_sol_reserves=30_000_000_000,
            real_token_reserves=800_000_000_000,
            real_sol_reserves=0, token_total_supply=1_000_000_000_000_000,
            complete=False, creator=creator)

    def test_a_post_migration_swap_is_not_forced_down_the_curve_route(self):
        """Building against a curve that has graduated is worse than routing."""
        engine = self._engine(route=NativePumpRoute(), curve=None)
        self.assertIsNone(engine.prepare_native_route(self.MINT, self.OTHER, 100, 100))

    def test_a_curve_without_a_creator_blocks_rather_than_deriving_a_wrong_vault(self):
        engine = self._engine(route=NativePumpRoute(), curve=self._curve(creator=""))
        prepared = engine.prepare_native_route(self.MINT, self.OTHER, 1_000_000, 100)
        self.assertEqual(prepared.status, "DATA_BLOCKED")
        self.assertIn("creator", prepared.detail)

    def test_a_buy_is_built_from_the_streamed_curve_with_no_quote_call(self):
        engine = self._engine(route=NativePumpRoute(),
                              curve=self._curve(creator=self.MINT))
        prepared = engine.prepare_native_route(self.MINT, self.OTHER, 1_000_000_000, 100)
        self.assertEqual(prepared.status, "OK")
        self.assertEqual(len(prepared.accounts), 27)
        self.assertEqual(prepared.data[:8], discriminator(PUMP_IDL, "buy_v2"))

    def test_a_sell_carries_a_protective_floor_derived_from_the_local_quote(self):
        engine = self._engine(route=NativePumpRoute(),
                              curve=self._curve(creator=self.MINT))
        prepared = engine.prepare_native_route(self.OTHER, self.MINT, 1_000_000_000, 500)
        self.assertEqual(prepared.status, "OK")
        self.assertEqual(len(prepared.accounts), 26)
        floor = int.from_bytes(prepared.data[16:24], "little")
        self.assertGreater(floor, 0)

    def test_a_native_route_that_never_runs_is_visible_rather_than_silent(self):
        """A builder that exists and is never used looks finished and is not."""
        engine = self._engine(route=None)
        report = engine.native_route_report()
        json.dumps(report)
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertIsNone(report["prepared_share"])
        engine.native_route_attempts["prepared"] = 3
        engine.native_route_attempts["blocked:DATA_BLOCKED"] = 1
        self.assertAlmostEqual(engine.native_route_report()["prepared_share"], 0.75)


class TestOnChainFeeConfig(unittest.TestCase):
    """The fee block was lifted by reading an account, not by squinting at a picture.

    The engine reported DATA_BLOCKED for every trade after 2026-09-01 on the
    grounds that Pump publishes the tier table only as docs/fees.png. That was
    the right refusal and the wrong conclusion: the image is a courtesy
    snapshot of the on-chain FeeConfig account, and the docs say outright that
    a correct implementation reads the account and is therefore unaffected by
    future tier changes.
    """

    @staticmethod
    def _fees(lp, protocol, creator):
        return struct.pack("<QQQ", lp, protocol, creator)

    @classmethod
    def _tier(cls, threshold, lp, protocol, creator):
        return threshold.to_bytes(16, "little") + cls._fees(lp, protocol, creator)

    @classmethod
    def _account(cls, tiers=((0, 5, 50, 45), (10 ** 9, 5, 30, 25)), stable=()):
        data = FEE_CONFIG_DISCRIMINATOR + b"\x01" + bytes(32) + cls._fees(1, 2, 3)
        data += len(tiers).to_bytes(4, "little")
        for row in tiers:
            data += cls._tier(*row)
        data += len(stable).to_bytes(4, "little")
        for row in stable:
            data += cls._tier(*row)
        return data

    def test_the_account_decodes_to_the_published_layout(self):
        config = parse_fee_config(self._account())
        self.assertTrue(config.ok)
        self.assertEqual(len(config.fee_tiers), 2)
        self.assertEqual(config.fee_tiers[0].fees.protocol_fee_bps, 50)
        self.assertEqual(config.fee_tiers[1].market_cap_lamports_threshold, 10 ** 9)
        json.dumps(config.to_dict())

    def test_the_total_charges_all_three_legs(self):
        """Summing only protocol and creator understates every pool trade."""
        config = parse_fee_config(self._account())
        self.assertEqual(config.fee_tiers[0].fees.total_bps, 5 + 50 + 45)

    def test_a_wrong_discriminator_is_refused_rather_than_decoded(self):
        """A same-length account decodes into plausible-looking pubkeys."""
        data = bytearray(self._account())
        data[:8] = bytes(8)
        config = parse_fee_config(bytes(data))
        self.assertEqual(config.status, "DATA_BLOCKED")
        self.assertIn("not a FeeConfig", config.detail)

    def test_an_implausible_tier_count_does_not_allocate(self):
        """A vec length read at the wrong offset is a very large number."""
        data = FEE_CONFIG_DISCRIMINATOR + b"\x01" + bytes(32) + self._fees(0, 0, 0)
        # Padded past the length guard so the count itself is what is refused,
        # not the account being short.
        data += (2 ** 31).to_bytes(4, "little") + bytes(64)
        config = parse_fee_config(data)
        self.assertEqual(config.status, "DATA_BLOCKED")
        self.assertIn("implausible", config.detail)

    def test_a_config_with_no_tiers_is_blocked_not_empty(self):
        config = parse_fee_config(self._account(tiers=()))
        self.assertEqual(config.status, "DATA_BLOCKED")
        self.assertIn("no tiers", config.detail)

    def test_tier_selection_matches_the_published_algorithm(self):
        """Thresholds are floors on an ascending list, scanned in reverse."""
        config = parse_fee_config(self._account())
        tiers = config.fee_tiers
        # Below the first threshold the first tier applies.
        self.assertEqual(calculate_fee_tier(tiers, 0).protocol_fee_bps, 50)
        # At and above a threshold, the highest tier reached applies.
        self.assertEqual(calculate_fee_tier(tiers, 10 ** 9 - 1).protocol_fee_bps, 50)
        self.assertEqual(calculate_fee_tier(tiers, 10 ** 9).protocol_fee_bps, 30)
        self.assertEqual(calculate_fee_tier(tiers, 10 ** 15).protocol_fee_bps, 30)

    def test_a_gap_in_the_table_resolves_to_the_highest_tier_reached(self):
        """Where a forward scan over ceilings would give a different answer."""
        config = parse_fee_config(self._account(
            tiers=((100, 0, 90, 0), (200, 0, 80, 0), (10_000, 0, 10, 0))))
        tiers = config.fee_tiers
        self.assertEqual(calculate_fee_tier(tiers, 50).protocol_fee_bps, 90)
        self.assertEqual(calculate_fee_tier(tiers, 5_000).protocol_fee_bps, 80)
        self.assertEqual(calculate_fee_tier(tiers, 10_000).protocol_fee_bps, 10)

    def test_the_market_cap_formulas_are_integer_arithmetic(self):
        """Floating point drifts across exactly the boundaries that matter."""
        self.assertEqual(bonding_curve_market_cap(10 ** 15, 30 * 10 ** 9, 10 ** 12),
                         (30 * 10 ** 9 * 10 ** 15) // 10 ** 12)
        self.assertEqual(pool_market_cap(1_000, 3, 7), (7 * 1_000) // 3)
        self.assertIsNone(bonding_curve_market_cap(10 ** 15, 30 * 10 ** 9, 0))
        self.assertIsNone(pool_market_cap(1_000, 0, 7))

    def test_adopting_the_chain_config_lifts_the_post_activation_block(self):
        schedule = PumpFeeSchedule()
        blocked = schedule.quote(market_cap_lamports=5 * 10 ** 8,
                                 at_utc=DYNAMIC_FEE_ACTIVATION_UTC + 1)
        self.assertEqual(blocked.status, "DATA_BLOCKED")

        self.assertTrue(schedule.adopt_chain_config(parse_fee_config(self._account())))
        quote = schedule.quote(market_cap_lamports=5 * 10 ** 8,
                               at_utc=DYNAMIC_FEE_ACTIVATION_UTC + 1)
        self.assertEqual(quote.status, "OK")
        self.assertEqual(quote.total_bps, 100)
        self.assertIn("chain", quote.schedule_version)
        # And the tier actually moves with the market cap.
        higher = schedule.quote(market_cap_lamports=2 * 10 ** 9,
                                at_utc=DYNAMIC_FEE_ACTIVATION_UTC + 1)
        self.assertEqual(higher.total_bps, 60)

    def test_an_unmeasured_market_cap_still_blocks_after_adoption(self):
        """The tiers being readable does not make the market cap known."""
        schedule = PumpFeeSchedule()
        schedule.adopt_chain_config(parse_fee_config(self._account()))
        quote = schedule.quote(at_utc=DYNAMIC_FEE_ACTIVATION_UTC + 1)
        self.assertEqual(quote.status, "DATA_BLOCKED")
        with self.assertRaises(ValueError):
            quote.fee_lamports(1_000_000_000)

    def test_a_blocked_config_is_not_adopted(self):
        schedule = PumpFeeSchedule()
        self.assertFalse(schedule.adopt_chain_config(parse_fee_config(b"short")))
        self.assertFalse(schedule.adopt_chain_config(None))

    def test_the_legacy_era_is_untouched_by_adoption(self):
        schedule = PumpFeeSchedule()
        schedule.adopt_chain_config(parse_fee_config(self._account()))
        legacy = schedule.quote(at_utc=DYNAMIC_FEE_ACTIVATION_UTC - 1)
        self.assertEqual(legacy.status, "OK")
        self.assertEqual(legacy.total_bps, LEGACY_TOTAL_FEE_BPS)



def _encode_pumpswap_trade_event(is_buy: bool, *, pool: bytes, user: bytes,
                                 base_amount: int, quote_amount: int,
                                 pool_base_reserves: int, pool_quote_reserves: int,
                                 lp_fee_bps: int, protocol_fee_bps: int,
                                 coin_creator: bytes, coin_creator_fee_bps: int,
                                 cashback_fee_bps: int, buyback_fee_bps: int,
                                 virtual_quote_reserves: int, base_supply: int,
                                 user_quote_amount: int = 0,
                                 ix_name: str = "buy") -> bytes:
    """Encode a BuyEvent / SellEvent exactly as idl/pump_amm.json declares it.

    Written field by field from the IDL rather than as a hex blob so that a
    layout change upstream breaks this encoder and the decoder together,
    instead of leaving a fixture that agrees with a decoder both of which are
    wrong.
    """
    out = bytearray()
    out += PumpSwapMonitor.BUY_EVENT if is_buy else PumpSwapMonitor.SELL_EVENT
    out += struct.pack("<q", 1_700_000_000)
    out += struct.pack("<QQQQ", base_amount, 0, 0, 0)     # amounts + user reserves
    out += struct.pack("<QQ", pool_base_reserves, pool_quote_reserves)
    out += struct.pack("<Q", quote_amount)
    out += struct.pack("<QQ", lp_fee_bps, 0)
    out += struct.pack("<QQ", protocol_fee_bps, 0)
    out += struct.pack("<QQ", 0, user_quote_amount)       # with/without lp fee, user leg
    out += pool + user
    out += bytes(32) * 4                                  # the four token accounts
    out += coin_creator
    out += struct.pack("<QQ", coin_creator_fee_bps, 0)
    if is_buy:
        out += b"\x01"                                    # track_volume
        out += struct.pack("<QQQqQ", 0, 0, 0, 0, 0)       # accumulators + min_base_out
        encoded = ix_name.encode()
        out += struct.pack("<I", len(encoded)) + encoded
    out += struct.pack("<QQQQ", cashback_fee_bps, 0, buyback_fee_bps, 0)
    out += int(virtual_quote_reserves).to_bytes(16, "little", signed=True)
    out += b"\x00"                                        # can_boost
    out += struct.pack("<Q", base_supply)
    return bytes(out)


class TestPumpSwapEventTailIsRead(unittest.TestCase):
    """The fee schedule was on the wire the whole time.

    Every PumpSwap trade event carries the basis points that trade actually
    paid. Quoting a pool against an assumed fee table when the protocol is
    telling us the real one on every swap is choosing to be wrong.
    """

    def _event(self, is_buy=True, **overrides):
        pool, user, creator = (bytes([value]) * 32 for value in (5, 6, 7))
        args = dict(pool=pool, user=user, base_amount=1_000_000,
                    quote_amount=500_000_000, pool_base_reserves=10_000_000,
                    pool_quote_reserves=20_000_000_000, lp_fee_bps=20,
                    protocol_fee_bps=5, coin_creator=creator,
                    coin_creator_fee_bps=5, cashback_fee_bps=1,
                    buyback_fee_bps=2, virtual_quote_reserves=0,
                    base_supply=1_000_000_000_000_000,
                    user_quote_amount=501_650_000)
        args.update(overrides)
        return _encode_pumpswap_trade_event(is_buy, **args), args

    def test_the_buy_tail_yields_the_fee_bps_actually_charged(self):
        monitor = PumpSwapMonitor(DummyYellowstone(), lambda _: None)
        data, args = self._event(True)
        event = monitor._decode_program_event(data, "sig", 1)
        self.assertEqual(event["tail_data_status"], "OK")
        self.assertEqual(event["lp_fee_bps"], 20)
        self.assertEqual(event["protocol_fee_bps"], 5)
        self.assertEqual(event["coin_creator_fee_bps"], 5)
        self.assertEqual(event["cashback_fee_bps"], 1)
        self.assertEqual(event["buyback_fee_bps"], 2)
        # Every component, LP fee included. Summing only protocol and creator
        # understates the cost of the swap.
        self.assertEqual(event["total_fee_bps"], 33)
        self.assertEqual(event["base_supply"], args["base_supply"])
        self.assertEqual(event["pool_base_reserves"], args["pool_base_reserves"])
        self.assertEqual(event["pool_quote_reserves"], args["pool_quote_reserves"])

    def test_the_sell_tail_has_no_ix_name_and_still_decodes(self):
        """SellEvent is fixed layout; BuyEvent carries a borsh string."""
        monitor = PumpSwapMonitor(DummyYellowstone(), lambda _: None)
        data, _ = self._event(False, virtual_quote_reserves=777)
        event = monitor._decode_program_event(data, "sig", 1)
        self.assertEqual(event["tail_data_status"], "OK")
        self.assertEqual(event["side"], "sell")
        self.assertEqual(event["virtual_quote_reserves"], 777)
        self.assertEqual(event["total_fee_bps"], 33)
        self.assertNotIn("ix_name", event)

    def test_a_variable_length_ix_name_does_not_shift_the_tail(self):
        monitor = PumpSwapMonitor(DummyYellowstone(), lambda _: None)
        short = monitor._decode_program_event(self._event(True, ix_name="buy")[0], "s", 1)
        long = monitor._decode_program_event(
            self._event(True, ix_name="buy_exact_quote_in_v2")[0], "s", 1)
        self.assertEqual(short["total_fee_bps"], long["total_fee_bps"])
        self.assertEqual(short["base_supply"], long["base_supply"])
        self.assertEqual(long["ix_name"], "buy_exact_quote_in_v2")

    def test_a_truncated_tail_blocks_rather_than_reporting_a_zero_fee(self):
        """A fee defaulted to zero quotes every trade as free."""
        monitor = PumpSwapMonitor(DummyYellowstone(), lambda _: None)
        data, _ = self._event(True)
        event = monitor._decode_program_event(data[:200], "sig", 1)
        self.assertTrue(event["tail_data_status"].startswith("DATA_BLOCKED"))
        self.assertNotIn("total_fee_bps", event)
        # The reserves are before the truncation and are still reported.
        self.assertEqual(event["pool_base_reserves"], 10_000_000)


class TestPumpSwapPoolQuoting(unittest.TestCase):
    """Graduation should change the venue, not whether we can price at all."""

    POOL = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    MINT = "So11111111111111111111111111111111111111112"

    def _state(self, **overrides):
        args = dict(pool=self.POOL, base_mint=self.MINT, quote_mint=self.MINT,
                    base_reserves=1_000_000_000_000, quote_reserves=100_000_000_000,
                    total_fee_bps=30, lp_fee_bps=20, protocol_fee_bps=5,
                    coin_creator_fee_bps=5, updated_at=time.time(), source="test")
        args.update(overrides)
        return PumpSwapPoolState(**args)

    def test_a_pool_whose_fee_schedule_was_never_observed_is_blocked(self):
        blocked = pool_quote_buy(self._state(total_fee_bps=None), 1_000_000_000)
        self.assertEqual(blocked.data_status, "DATA_BLOCKED")
        self.assertEqual(blocked.reason, "fee_schedule_unobserved")
        self.assertEqual(blocked.output_amount, 0)

    def test_quotes_price_against_effective_quote_reserves(self):
        """The docs are explicit: raw vault balance plus virtual reserves."""
        plain = pool_quote_buy(self._state(), 1_000_000_000)
        virtual = pool_quote_buy(
            self._state(virtual_quote_reserves=100_000_000_000), 1_000_000_000)
        self.assertEqual(plain.data_status, "OK")
        # Twice the effective quote depth means materially fewer tokens for
        # the same spend. Reading only the raw balance would report the two
        # pools as identical.
        self.assertLess(virtual.output_amount, plain.output_amount)

    def test_the_buy_budget_is_what_leaves_the_wallet(self):
        """Sizing chose the spend; the fee comes out of it, not on top of it."""
        budget = 1_000_000_000
        quote = pool_quote_buy(self._state(total_fee_bps=100), budget)
        self.assertEqual(quote.input_amount, budget)
        self.assertGreater(quote.fee_amount, 0)
        # amm_leg + fee is exactly the budget: no lamport is created or lost.
        amm_leg = (budget * 10_000) // 10_100
        self.assertEqual(quote.fee_amount, budget - amm_leg)

    def test_a_higher_fee_yields_strictly_fewer_tokens(self):
        cheap = pool_quote_buy(self._state(total_fee_bps=10), 1_000_000_000)
        dear = pool_quote_buy(self._state(total_fee_bps=500), 1_000_000_000)
        self.assertLess(dear.output_amount, cheap.output_amount)

    def test_the_sell_fee_rounds_against_us(self):
        """Rounding the fee down overstates proceeds in the flattering direction."""
        quote = pool_quote_sell(self._state(total_fee_bps=1), 3)
        gross = (3 * 100_000_000_000) // (1_000_000_000_000 + 3)
        self.assertEqual(quote.fee_amount, -((-gross * 1) // 10_000))
        self.assertEqual(quote.output_amount, gross - quote.fee_amount)

    def test_stale_state_refuses_rather_than_pricing_a_memory(self):
        stale = self._state(updated_at=time.time() - 600)
        self.assertEqual(pool_quote_sell(stale, 1_000).data_status, "DATA_BLOCKED")
        # ...and the same state quotes fine when staleness is not the question.
        self.assertEqual(pool_quote_sell(stale, 1_000, max_age_s=0).data_status, "OK")

    def test_a_buy_can_never_be_quoted_the_pool_entire_inventory(self):
        """Constant product approaches the inventory; it never delivers it."""
        state = self._state()
        huge = pool_quote_buy(state, 10 ** 18)
        self.assertEqual(huge.data_status, "OK")
        self.assertLess(huge.output_amount, state.base_reserves)
        # And the price paid for trying says so.
        self.assertGreater(huge.price_impact_pct, 100.0)

    def test_capacity_is_the_largest_sale_within_the_impact_bound(self):
        state = self._state()
        capacity = sell_capacity_base(state, 0.05)
        self.assertGreater(capacity, 0)
        self.assertLessEqual(pool_quote_sell(state, capacity).price_impact_pct, 0.05)
        self.assertGreater(pool_quote_sell(state, capacity * 4).price_impact_pct, 0.05)

    def test_a_deeper_pool_absorbs_a_larger_exit(self):
        shallow = sell_capacity_base(self._state(), 0.05)
        deep = sell_capacity_base(self._state(base_reserves=10_000_000_000_000,
                                              quote_reserves=1_000_000_000_000), 0.05)
        self.assertGreater(deep, shallow)


class TestGraduationKeepsNativeExecution(unittest.IsolatedAsyncioTestCase):
    """A migrated coin is the same coin.

    The route existed and was constructed, and the engine was never given it,
    so graduation silently demoted every position to a router round trip at
    exactly the point where the position is largest.
    """

    MINT = "So11111111111111111111111111111111111111112"
    OTHER = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def _pool_account(self, **overrides):
        base = dict(status="OK", pool=self.OTHER, base_mint=self.MINT,
                    quote_mint=self.MINT, pool_base_token_account=self.OTHER,
                    pool_quote_token_account=self.OTHER, creator=self.OTHER,
                    coin_creator=self.MINT)
        base.update(overrides)
        return PoolState(**base)

    def _reserves(self, **overrides):
        args = dict(pool=self.OTHER, base_mint=self.MINT, quote_mint=self.MINT,
                    base_reserves=1_000_000_000_000, quote_reserves=100_000_000_000,
                    total_fee_bps=30, updated_at=time.time(), source="test")
        args.update(overrides)
        return PumpSwapPoolState(**args)

    def _engine(self, *, reserves=None, account=None, curve=None):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.pump_route = NativePumpRoute()
        engine.pumpswap_route = PumpSwapRoute()
        engine.curve_state_provider = (lambda token: curve) if curve else None
        engine.pool_state_provider = (lambda token: reserves) if reserves else None
        engine.pool_account_provider = (lambda token: account) if account else None
        engine.tx_builder = SimpleNamespace(public_key=self.OTHER)
        engine.native_route_attempts = defaultdict(int)
        engine.native_compute_unit_limit = 400_000
        engine.dry_run = True
        engine.landing_model = LandingModel()
        engine.last_bid = {}
        engine.stream_confirmations = 0
        engine.poll_confirmations = 0
        engine._signature_waiters = {}
        engine.reconcile_min_interval = 0.005
        engine.reconcile_max_interval = 0.05
        engine.execution_history = deque(maxlen=16)
        engine.route_performance = defaultdict(
            lambda: {"total": 0, "landed": 0, "filled": 0, "failed": 0, "avg_latency": 0})
        engine.counterfactual_lab = None
        return engine

    def test_a_migrated_coin_still_builds_its_own_instruction(self):
        engine = self._engine(reserves=self._reserves(), account=self._pool_account())
        prepared = engine.prepare_native_route(self.MINT, self.OTHER, 1_000_000_000, 100)
        self.assertEqual(prepared.status, "OK")
        self.assertEqual(prepared.venue, "pumpswap")
        self.assertEqual(len(prepared.accounts), 23)

    def test_the_sell_side_is_native_too(self):
        engine = self._engine(reserves=self._reserves(), account=self._pool_account())
        prepared = engine.prepare_native_route(self.OTHER, self.MINT, 1_000_000, 200)
        self.assertEqual(prepared.status, "OK")
        self.assertEqual(prepared.venue, "pumpswap")
        self.assertEqual(len(prepared.accounts), 21)

    def test_the_max_quote_in_bounds_the_amm_leg_not_the_whole_budget(self):
        """The protocol takes its fee on top of the leg it is given."""
        budget = 1_000_000_000
        engine = self._engine(reserves=self._reserves(total_fee_bps=100),
                              account=self._pool_account())
        prepared = engine.prepare_native_route(self.MINT, self.OTHER, budget, 0)
        bound = struct.unpack_from("<Q", prepared.data, 16)[0]
        self.assertLess(bound, budget)

    def test_an_undecoded_pool_account_blocks_rather_than_guessing(self):
        """The mayhem flag picks the published fee-recipient set."""
        engine = self._engine(reserves=self._reserves())
        prepared = engine.prepare_native_route(self.MINT, self.OTHER, 1_000_000_000, 100)
        self.assertEqual(prepared.status, "DATA_BLOCKED")
        self.assertEqual(prepared.venue, "pumpswap")
        self.assertIn("not decoded", prepared.detail)

    def test_an_unknown_token_is_still_no_native_trade_at_all(self):
        self.assertIsNone(
            self._engine().prepare_native_route(self.MINT, self.OTHER, 1_000, 100))

    def test_the_curve_wins_while_the_coin_is_still_on_it(self):
        """Both venues answering means the coin has not graduated yet."""
        curve = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000, virtual_sol_reserves=30_000_000_000,
            real_token_reserves=800_000_000_000, real_sol_reserves=10_000_000_000,
            token_total_supply=1_000_000_000_000, complete=False, creator=self.OTHER)
        engine = self._engine(curve=curve, reserves=self._reserves(),
                              account=self._pool_account())
        prepared = engine.prepare_native_route(self.MINT, self.OTHER, 1_000_000_000, 100)
        self.assertEqual(prepared.venue, "pump_curve")

    def test_the_quote_follows_the_venue_the_instruction_was_built_for(self):
        """A curve quote on a pool instruction looks valid right up to the fill."""
        engine = self._engine(reserves=self._reserves(), account=self._pool_account())
        pool_quote = engine._native_quote(self.MINT, self.OTHER, 1_000_000_000, "pumpswap")
        self.assertIsNotNone(pool_quote)
        self.assertIsNone(engine._native_quote(self.MINT, self.OTHER, 1_000_000_000))

    async def test_a_pool_fill_is_reported_as_a_pool_fill(self):
        engine = self._engine(reserves=self._reserves(), account=self._pool_account())
        result = await engine.execute_swap(self.MINT, self.OTHER, 1_000_000_000,
                                           slippage_bps=100)
        self.assertTrue(result.success)
        self.assertEqual(result.route_type, RouteType.PUMPSWAP_NATIVE)
        self.assertEqual(engine.native_route_attempts["prepared:pumpswap"], 1)

    def test_the_report_says_whether_the_pool_side_is_wired(self):
        engine = self._engine(reserves=self._reserves(), account=self._pool_account())
        report = engine.native_route_report()
        self.assertEqual(report["pumpswap_route"]["status"], "OK")
        self.assertTrue(report["pool_state_wired"])
        self.assertTrue(report["pool_account_wired"])
        self.assertFalse(self._engine().native_route_report()["pool_state_wired"])


class TestDeskMaintainsPoolState(unittest.IsolatedAsyncioTestCase):
    """The desk owns the pool view, exactly as it owns the curve view.

    Two views of the price we are about to trade at is one view too many, so
    the engine reads this through a provider rather than subscribing itself.
    """

    TOKEN = "So11111111111111111111111111111111111111112"
    POOL = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def _desk(self):
        desk = SimpleNamespace(
            _latest_pool_state={}, _pool_accounts={}, _pool_account_pending=set(),
            solana_rpc=None, state_sequencer=SimpleNamespace(bump=lambda token: None),
        )
        desk.redecisions = []
        desk.request_redecision = desk.redecisions.append
        desk.backgrounded = []

        def spawn(coroutine):
            desk.backgrounded.append(coroutine)
            coroutine.close()

        desk._spawn_background = spawn
        desk._fetch_pool_account = (
            lambda token, pool: MemecoinQuantDesk._fetch_pool_account(desk, token, pool))
        return desk

    def _created(self, **overrides):
        event = {"pool": self.POOL, "base_mint": self.TOKEN, "quote_mint": self.TOKEN,
                 "initial_base_amount": 1_000_000_000_000,
                 "initial_quote_amount": 100_000_000_000,
                 "base_mint_decimals": 6, "timestamp": time.time(), "slot": 5}
        event.update(overrides)
        return event

    def _traded(self, **overrides):
        event = {"pool": self.POOL, "token": self.TOKEN, "quote_mint": self.TOKEN,
                 "pool_base_reserves": 900_000_000_000,
                 "pool_quote_reserves": 110_000_000_000,
                 "tail_data_status": "OK", "lp_fee_bps": 20, "protocol_fee_bps": 5,
                 "coin_creator_fee_bps": 5, "total_fee_bps": 30,
                 "virtual_quote_reserves": 0, "base_supply": 1_000_000_000_000_000,
                 "coin_creator": self.POOL, "timestamp": time.time(), "slot": 6}
        event.update(overrides)
        return event

    def test_migration_opens_pool_state_without_inventing_a_fee(self):
        desk = self._desk()
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        state = desk._latest_pool_state[self.TOKEN]
        self.assertEqual(state.pool, self.POOL)
        self.assertEqual(state.base_reserves, 1_000_000_000_000)
        # Not zero, and not a table lookup: unobserved.
        self.assertIsNone(state.total_fee_bps)
        self.assertEqual(state.blocked_reason(), "fee_schedule_unobserved")
        # And the account decode is queued off the hot path.
        self.assertEqual(len(desk.backgrounded), 1)

    def test_the_first_trade_supplies_the_schedule_the_pool_actually_charges(self):
        desk = self._desk()
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded())
        state = desk._latest_pool_state[self.TOKEN]
        self.assertEqual(state.total_fee_bps, 30)
        self.assertEqual(state.base_reserves, 900_000_000_000)
        self.assertIsNone(state.blocked_reason())
        # Reserves moved, so anything priced against the old ones is stale.
        self.assertIn(self.TOKEN, desk.redecisions)

    def test_an_unreadable_tail_carries_the_last_measurement_forward(self):
        """Not a default -- a measurement, just not of this trade."""
        desk = self._desk()
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded())
        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded(
            tail_data_status="DATA_BLOCKED: truncated", pool_base_reserves=800_000_000_000))
        state = desk._latest_pool_state[self.TOKEN]
        self.assertEqual(state.total_fee_bps, 30)
        self.assertEqual(state.base_reserves, 800_000_000_000)

    def test_a_first_trade_we_could_not_read_leaves_the_pool_unpriceable(self):
        desk = self._desk()
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded(
            tail_data_status="DATA_BLOCKED: truncated"))
        self.assertEqual(desk._latest_pool_state[self.TOKEN].blocked_reason(),
                         "fee_schedule_unobserved")

    def test_an_empty_trade_event_does_not_erase_live_reserves(self):
        desk = self._desk()
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded())
        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded(
            pool_base_reserves=0, pool_quote_reserves=0))
        self.assertEqual(desk._latest_pool_state[self.TOKEN].base_reserves, 900_000_000_000)

    async def test_the_pool_account_is_decoded_once_and_cached(self):
        desk = self._desk()
        from solders.pubkey import Pubkey
        body = (b"\x01" + (7).to_bytes(2, "little")
                + bytes(Pubkey.from_string(self.POOL)) * 6
                + (123).to_bytes(8, "little")
                + bytes(Pubkey.from_string(self.TOKEN))
                + b"\x00\x00" + (0).to_bytes(16, "little", signed=True))
        encoded = base64.b64encode(POOL_DISCRIMINATOR + body).decode()
        calls = []

        async def request(method, params):
            calls.append(params[0])
            return {"value": {"data": [encoded, "base64"]}}

        desk.solana_rpc = SimpleNamespace(request=request)
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        self.assertTrue(await MemecoinQuantDesk._fetch_pool_account(desk, self.TOKEN, self.POOL))
        self.assertEqual(desk._pool_accounts[self.TOKEN].coin_creator, self.TOKEN)
        self.assertEqual(calls, [self.POOL])

    async def test_an_account_that_is_not_a_pool_is_not_adopted(self):
        desk = self._desk()
        encoded = base64.b64encode(bytes(300)).decode()

        async def request(method, params):
            return {"value": {"data": [encoded, "base64"]}}

        desk.solana_rpc = SimpleNamespace(request=request)
        self.assertFalse(await MemecoinQuantDesk._fetch_pool_account(desk, self.TOKEN, self.POOL))
        self.assertNotIn(self.TOKEN, desk._pool_accounts)

    def test_the_report_separates_wired_from_executable(self):
        """A route blocked on every coin looks exactly like one never wired."""
        desk = self._desk()
        MemecoinQuantDesk._seed_pool_state(desk, self.TOKEN, self._created())
        blocked = MemecoinQuantDesk.pool_route_report(desk)
        self.assertEqual(blocked["status"], "DATA_BLOCKED")
        self.assertEqual(blocked["pools_tracked"], 1)
        self.assertEqual(blocked["pools_executable"], 0)
        self.assertEqual(blocked["reasons"]["fee_schedule_unobserved"], 1)

        MemecoinQuantDesk._update_pool_state(desk, self.TOKEN, self._traded())
        desk._pool_accounts[self.TOKEN] = PoolState(
            status="OK", pool=self.POOL, base_mint=self.TOKEN, quote_mint=self.TOKEN,
            pool_base_token_account=self.POOL, pool_quote_token_account=self.POOL,
            creator=self.POOL, coin_creator=self.TOKEN)
        live = MemecoinQuantDesk.pool_route_report(desk)
        self.assertEqual(live["status"], "OK")
        self.assertEqual(live["pools_executable"], 1)
        self.assertEqual(live["executable_share"], 1.0)


class FakeHttpClient:
    """Canned responses in the shape HttpClient.get returns."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def get(self, url, headers=None):
        self.requests.append((url, dict(headers or {})))
        if not self.responses:
            raise TransportError("no more canned responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestSourceTransports(unittest.IsolatedAsyncioTestCase):
    """The mesh was an adapter library, not a source of signal.

    source_fetchers was empty, so build_sources ran from nothing and every
    declaration reported NO_FETCHER -- which reads as an operator task and was
    actually a missing layer.
    """

    RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
      <item><title>First &amp; only</title><description>&lt;b&gt;body&lt;/b&gt;</description>
        <link>https://example.test/a</link><guid>a</guid>
        <pubDate>Tue, 10 Jun 2025 09:00:00 GMT</pubDate></item>
      <item><title>Second</title><description>more</description>
        <link>https://example.test/b</link><guid>b</guid></item>
    </channel></rss>"""

    ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>urn:1</id><title>Atom title</title><summary>Atom summary</summary>
        <link href="https://example.test/atom"/>
        <published>2025-06-10T09:00:00Z</published></entry>
    </feed>"""

    def test_rss_parses_both_formats_and_decodes_entities(self):
        client = FakeHttpClient([(200, self.RSS, {}), (200, self.ATOM, {})])
        rss = RssTransport("rss:test", "https://example.test/feed", client, language="en")
        records = asyncio.run(rss.fetch())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["title"], "First & only")
        # Tags stripped, not left as markup for the mint extractor to trip on.
        self.assertEqual(records[0]["summary"], "body")
        self.assertEqual(records[0]["language"], "en")
        self.assertAlmostEqual(records[0]["published_epoch"], 1749546000.0, places=0)

        atom = RssTransport("rss:atom", "https://example.test/atom", client)
        entries = asyncio.run(atom.fetch())
        self.assertEqual(entries[0]["title"], "Atom title")
        self.assertEqual(entries[0]["link"], "https://example.test/atom")

    def test_a_feed_that_reserves_the_same_items_emits_them_once(self):
        """Without this, every poll re-emits the page and one story is fifty."""
        client = FakeHttpClient([(200, self.RSS, {}), (200, self.RSS, {})])
        rss = RssTransport("rss:test", "https://example.test/feed", client)
        self.assertEqual(len(asyncio.run(rss.fetch())), 2)
        self.assertEqual(len(asyncio.run(rss.fetch())), 0)

    def test_conditional_get_is_sent_and_a_304_is_not_a_failure(self):
        client = FakeHttpClient([
            (200, self.RSS, {"ETag": "abc", "Last-Modified": "Tue, 10 Jun 2025 09:00:00 GMT"}),
            (304, "", {})])
        rss = RssTransport("rss:test", "https://example.test/feed", client)
        asyncio.run(rss.fetch())
        self.assertEqual(asyncio.run(rss.fetch()), [])
        self.assertEqual(client.requests[1][1]["If-None-Match"], "abc")
        self.assertIn("If-Modified-Since", client.requests[1][1])

    def test_an_error_status_raises_rather_than_returning_empty(self):
        """A silent empty batch turns a dead feed into a quiet one."""
        rss = RssTransport("rss:test", "https://example.test/feed",
                           FakeHttpClient([(503, "", {})]))
        with self.assertRaises(TransportError):
            asyncio.run(rss())
        self.assertEqual(rss.failures, 1)

    def test_rate_limiting_is_named_as_such(self):
        rss = RssTransport("rss:test", "https://example.test/feed",
                           FakeHttpClient([(429, "", {})]))
        with self.assertRaises(TransportError) as raised:
            asyncio.run(rss.fetch())
        self.assertIn("rate limited", str(raised.exception))

    def test_a_missing_publication_time_is_not_backfilled_with_now(self):
        """A stale item stamped 'now' manufactures lead time we did not have."""
        client = FakeHttpClient([(200, self.RSS, {})])
        records = asyncio.run(
            RssTransport("rss:test", "https://example.test/feed", client).fetch())
        self.assertIsNone(records[1]["published_epoch"])

    def test_timestamps_parse_from_every_shape_feeds_actually_use(self):
        self.assertAlmostEqual(parse_timestamp(1749546000), 1749546000.0)
        self.assertAlmostEqual(parse_timestamp(1749546000000), 1749546000.0)
        self.assertAlmostEqual(parse_timestamp(1749546000000000), 1749546000.0)
        self.assertAlmostEqual(parse_timestamp("2025-06-10T09:00:00Z"), 1749546000.0, places=0)
        self.assertAlmostEqual(
            parse_timestamp("Tue, 10 Jun 2025 09:00:00 GMT"), 1749546000.0, places=0)
        self.assertIsNone(parse_timestamp("not a date"))
        self.assertIsNone(parse_timestamp(None))

    def test_the_mastodon_timeline_advances_its_cursor(self):
        page = json.dumps([
            {"id": "12", "content": "<p>hello</p>", "created_at": "2025-06-10T09:00:00Z",
             "account": {"acct": "someone"}, "language": "en"}])
        client = FakeHttpClient([(200, page, {}), (200, "[]", {})])
        transport = MastodonTimelineTransport("m", "https://example.test", client)
        records = asyncio.run(transport.fetch())
        self.assertEqual(records[0]["content"], "hello")
        self.assertEqual(records[0]["account"]["acct"], "someone")
        asyncio.run(transport.fetch())
        self.assertIn("since_id=12", client.requests[1][0])

    def test_official_site_emits_only_on_change_and_never_on_first_read(self):
        client = FakeHttpClient([(200, "<p>one</p>", {}), (200, "<p>one</p>", {}),
                                 (200, "<p>two</p>", {})])
        site = OfficialSiteTransport("o", "https://example.test/page", client)
        self.assertEqual(asyncio.run(site.fetch()), [])
        self.assertEqual(asyncio.run(site.fetch()), [])
        changed = asyncio.run(site.fetch())
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["changed_text"], "two")

    def test_github_commits_become_code_records(self):
        payload = json.dumps([
            {"sha": "deadbeef",
             "commit": {"message": "add mint", "author": {"name": "dev", "date": "2025-06-10T09:00:00Z"}}}])
        client = FakeHttpClient([(200, payload, {}), (200, payload, {})])
        transport = GithubRepoTransport("c", "owner/repo", client)
        records = asyncio.run(transport.fetch())
        self.assertEqual(records[0]["sha"], "deadbeef")
        self.assertEqual(records[0]["repo"], "owner/repo")
        # And the same commit is not reported twice.
        self.assertEqual(asyncio.run(transport.fetch()), [])

    def test_a_non_json_body_is_a_transport_error_not_a_crash(self):
        transport = JsonPollTransport("j", "https://example.test/api",
                                      FakeHttpClient([(200, "<html>", {})]))
        with self.assertRaises(TransportError):
            asyncio.run(transport.fetch())

    def test_the_queue_transport_drops_the_oldest_under_pressure(self):
        queue = QueueTransport("q", capacity=3)
        for index in range(5):
            queue.push({"n": index})
        drained = asyncio.run(queue.fetch())
        self.assertEqual([record["n"] for record in drained], [2, 3, 4])
        self.assertEqual(queue.report()["dropped"], 2)

    def test_the_jetstream_transport_ignores_frames_with_no_text(self):
        transport = BlueskyJetstreamTransport("b")
        self.assertIsNone(transport.on_message({"commit": {"operation": "delete"}}))
        self.assertIsNone(transport.on_message({"commit": {"record": {}}}))
        kept = transport.on_message(
            {"did": "did:plc:x", "commit": {"operation": "create",
                                            "record": {"text": "gm"}}})
        self.assertIsNotNone(kept)

    def test_the_nostr_subscription_names_its_authors_when_it_has_them(self):
        transport = NostrRelayTransport("n", "wss://relay.test", authors=("abc",))
        frame = json.loads(transport.on_open()[0])
        self.assertEqual(frame[0], "REQ")
        self.assertEqual(frame[2]["authors"], ["abc"])
        self.assertEqual(frame[2]["kinds"], [1])
        # Relay frames are arrays; anything else is not an event.
        self.assertIsNone(transport.on_message({"content": "x"}))
        self.assertEqual(
            transport.on_message(["EVENT", "sub", {"content": "gm"}])["content"], "gm")

    def test_telegram_names_the_variables_it_needs_and_never_reads_a_value(self):
        transport = TelegramChannelTransport("t", "somechannel",
                                             api_id_env="NOT_SET_API_ID",
                                             api_hash_env="NOT_SET_API_HASH")
        with self.assertRaises(TransportError) as raised:
            asyncio.run(transport.start())
        message = str(raised.exception)
        self.assertIn("NOT_SET_API_ID", message)
        self.assertIn("NOT_SET_API_HASH", message)


class TestTransportsAreBuiltFromDeclarations(unittest.TestCase):
    """A declaration with no transport is a coverage hole with a name."""

    def _declaration(self, **overrides):
        args = dict(source_id="s", kind="rss", options={"url": "https://example.test/f"})
        args.update(overrides)
        return SourceDeclaration(**args)

    def test_every_supported_kind_builds(self):
        client = FakeHttpClient([])
        kinds = {
            "rss": {"url": "https://example.test/f"},
            "official_site": {"url": "https://example.test/p"},
            "mastodon": {"instance": "https://example.test"},
            "code_repo": {"repo": "o/r"},
            "youtube": {"channel_id": "UC123"},
            "bluesky": {},
            "nostr": {"relay": "wss://relay.test"},
            "farcaster": {"hub_url": "https://hub.test/v1/casts"},
            "telegram": {"channel": "chan"},
            "metadata": {}, "twitch": {}, "discord": {},
        }
        for kind, options in kinds.items():
            transport = build_transport(
                self._declaration(source_id=kind, kind=kind, options=options), client)
            self.assertIsNotNone(transport, kind)

    def test_a_missing_option_is_reported_by_name_not_raised_as_a_type_error(self):
        with self.assertRaises(TransportError) as raised:
            build_transport(self._declaration(options={}), FakeHttpClient([]))
        self.assertIn("url", str(raised.exception))

    def test_the_three_reasons_a_source_has_no_transport_are_kept_apart(self):
        """Three different problems with three different owners."""
        declarations = [
            self._declaration(source_id="ready"),
            self._declaration(source_id="pending", options={}),
            self._declaration(source_id="keyless", requires_env=("NOT_SET_ANYWHERE",)),
            self._declaration(source_id="odd", kind="carrier_pigeon", options={}),
        ]
        transports, report, _client = build_transports(declarations, FakeHttpClient([]))
        self.assertEqual(sorted(transports), ["ready"])
        self.assertEqual([source for source, _ in report.pending_endpoint], ["pending"])
        self.assertEqual([source for source, _ in report.unconfigured], ["keyless"])
        self.assertEqual([source for source, _ in report.unsupported], ["odd"])
        self.assertEqual(len(report.unbuilt), 3)

    def test_the_shipped_registry_builds_the_transports_it_declares(self):
        """The seed is a seed, and the report says exactly which part is which."""
        declarations = load_declarations("config/sources.yaml")
        self.assertGreater(len(declarations), 20)
        transports, report, _client = build_transports(declarations, FakeHttpClient([]))
        self.assertGreater(report.built, 0)
        self.assertEqual(report.declared, len(declarations))
        # Nothing declared is unsupported: every kind in the registry has a
        # transport, and what is left is endpoints and credentials.
        self.assertEqual(report.unsupported, [])
        self.assertEqual(report.built + len(report.unbuilt), report.declared)

    def test_the_report_names_a_wired_source_that_has_never_answered(self):
        transport = QueueTransport("q")
        report = transport_report({"q": transport})
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertEqual(report["transports"], 1)
        transport.push({"any": "thing"})
        asyncio.run(transport())
        self.assertEqual(transport_report({"q": transport})["status"], "OK")


class TestEntityProvenanceIsRequired(unittest.TestCase):
    """An identity assertion nobody can trace is the expensive error.

    A missing entity makes an official token look unverified and costs a
    trade. A WRONG entity makes an impersonator look verified and costs the
    position -- so the parse is strict about exactly the fields that assert
    identity.
    """

    def _write(self, entries) -> str:
        path = os.path.join(tempfile.mkdtemp(), "entities.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump({"entities": entries}, handle)
        return path

    def _entry(self, **overrides):
        entry = {"entity_id": "figure", "display_name": "Some Figure",
                 "official_domains": ["figure.com"],
                 "accounts": {"telegram": ["1001"]},
                 "verified_from": "https://figure.com/press",
                 "verified_at": "2026-06-01"}
        entry.update(overrides)
        return entry

    def test_an_entity_with_provenance_loads(self):
        entities = load_entities(self._write([self._entry()]))
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].verified_from, "https://figure.com/press")
        self.assertGreater(entities[0].verified_at, 0)

    def test_an_entity_without_a_source_is_refused_not_flagged(self):
        """A flag on a record that still confers proof is not a control."""
        entry = self._entry()
        del entry["verified_from"]
        self.assertEqual(load_entities(self._write([entry])), [])

    def test_an_entity_without_a_date_is_refused(self):
        entry = self._entry()
        del entry["verified_at"]
        self.assertEqual(load_entities(self._write([entry])), [])

    def test_an_unreadable_date_is_refused_rather_than_stamped_now(self):
        """Defaulting to now makes every entry permanently fresh."""
        self.assertEqual(
            load_entities(self._write([self._entry(verified_at="whenever")])), [])

    def test_provenance_is_accepted_from_the_metadata_block_the_schema_documents(self):
        entry = self._entry()
        del entry["verified_from"]
        del entry["verified_at"]
        entry["metadata"] = {"verified_from": "https://figure.com/about",
                             "verified_at": "2026-06-01"}
        self.assertEqual(len(load_entities(self._write([entry]))), 1)

    def test_one_bad_entry_does_not_take_the_registry_offline(self):
        bad = self._entry(entity_id="bad")
        del bad["verified_from"]
        entities = load_entities(self._write([self._entry(), bad]))
        self.assertEqual([entity.entity_id for entity in entities], ["figure"])

    def test_a_verified_overlay_wins_over_the_seed(self):
        seed = self._write([self._entry(display_name="Stale Name")])
        overlay = self._write([self._entry(display_name="Current Name")])
        entities = load_entities(f"{seed},{overlay}")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].display_name, "Current Name")

    def test_a_missing_overlay_is_not_an_error_when_there_are_several_paths(self):
        seed = self._write([self._entry()])
        entities = load_entities(f"{seed},/nonexistent/entities.yaml")
        self.assertEqual(len(entities), 1)

    def test_the_shipped_registry_is_empty_and_says_why(self):
        """Empty is not 'nothing is a copycat'. It is 'we cannot tell'."""
        self.assertEqual(load_entities("config/entities.yaml"), [])
        report = EntityRegistry([]).report()
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertIn("cannot tell", report["detail"])
        self.assertEqual(report["entities"], 0)


class TestStaleEntitiesLoseTheirProof(unittest.TestCase):
    """A handle verified two years ago is a claim about the past."""

    MINT = "So11111111111111111111111111111111111111112"
    WALLET = "WaLLeT1111111111111111111111111111111111111"

    def _entity(self, age_days: float):
        return WatchedEntity(
            entity_id="figure", display_name="Some Figure",
            accounts={"telegram": {"1001"}}, official_domains={"figure.com"},
            known_wallets={self.WALLET},
            verified_from="https://figure.com/press",
            verified_at=time.time() - age_days * 86_400)

    def _resolver(self, age_days: float):
        return AuthenticityResolver(EntityRegistry([self._entity(age_days)]))

    def _signal(self, text: str):
        return SourceSignal(platform="telegram", account_id="1001", text=text,
                            timestamp=time.time())

    def test_a_fresh_entity_still_gets_the_strongest_proof(self):
        verdict = self._resolver(1).resolve_signal(self._signal(f"launching {self.MINT}"))
        self.assertEqual(verdict.level, ProofLevel.DIRECT_MINT)
        self.assertEqual(verdict.mint, self.MINT)

    def test_a_stale_entity_is_capped_at_name_only_and_carries_no_mint(self):
        verdict = self._resolver(400).resolve_signal(self._signal(f"launching {self.MINT}"))
        self.assertEqual(verdict.level, ProofLevel.NAME_ONLY)
        self.assertIsNone(verdict.mint)
        self.assertFalse(verdict.tradeable)
        self.assertIn("re-verified", verdict.detail)

    def test_a_stale_wallet_claim_does_not_authorise_a_position(self):
        verdict = self._resolver(400).resolve_creator(self.MINT, self.WALLET)
        self.assertEqual(verdict.level, ProofLevel.NAME_ONLY)
        self.assertFalse(verdict.tradeable)
        fresh = self._resolver(1).resolve_creator(self.MINT, self.WALLET)
        self.assertEqual(fresh.level, ProofLevel.CREATOR_WALLET)

    def test_a_stale_official_domain_link_is_capped_too(self):
        signal = self._signal("announcement at https://figure.com/token")
        published = {"figure.com": self.MINT}
        stale = self._resolver(400).resolve_signal(signal, published)
        self.assertEqual(stale.level, ProofLevel.NAME_ONLY)
        self.assertIsNone(stale.mint)
        fresh = self._resolver(1).resolve_signal(signal, published)
        self.assertEqual(fresh.level, ProofLevel.OFFICIAL_DOMAIN)
        self.assertEqual(fresh.mint, self.MINT)

    def test_an_entity_with_no_verification_date_is_treated_as_stale(self):
        registry = EntityRegistry([WatchedEntity(
            entity_id="x", display_name="X", accounts={"telegram": {"1001"}})])
        verdict = AuthenticityResolver(registry).resolve_signal(
            self._signal(f"launching {self.MINT}"))
        self.assertEqual(verdict.level, ProofLevel.NAME_ONLY)

    def test_the_report_names_which_entities_have_gone_stale(self):
        registry = EntityRegistry([self._entity(400)])
        report = registry.report()
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["stale"], ["figure"])
        self.assertEqual(report["fresh"], 0)
        self.assertGreater(report["oldest_verification_days"], 180)


class TestEntityVerifierReadsPublishedPages(unittest.TestCase):
    """Filling the registry from memory is the failure the empty file prevents."""

    PAGE = """<html><body>
      <a href="https://t.me/example_official">Telegram</a>
      <a href="https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv">YouTube</a>
      <a href="https://bsky.app/profile/example.org">Bluesky</a>
      <a href="https://github.com/exampleorg">Code</a>
      <a href="https://t.me/example_official/4321">a post, not a profile</a>
    </body></html>"""

    def test_only_profile_links_become_handles(self):
        found = verify_entities.handles_in(self.PAGE)
        self.assertEqual(found["telegram"], ["example_official"])
        self.assertEqual(found["youtube"], ["UCabcdefghijklmnopqrstuv"])
        self.assertEqual(found["bluesky"], ["example.org"])
        self.assertEqual(found["github"], ["exampleorg"])

    def test_a_page_that_links_nothing_yields_nothing(self):
        self.assertEqual(verify_entities.handles_in("<p>no links here</p>"), {})

    def test_handles_are_emitted_as_comments_never_as_account_ids(self):
        """`accounts` holds stable ids; a display handle there is impersonable."""
        result = {"domain": "example.org", "reachable": True,
                  "pages": [{"url": "https://example.org/", "sha256": "abc",
                             "detail": "HTTP 200"}],
                  "handles": {"telegram": ["example_official"]}}
        lines = verify_entities.declaration("example-org", "Example Org", result)
        rendered = "\n".join(lines)
        self.assertIn("# telegram: example_official", rendered)
        self.assertIn("verified_from: \"https://example.org/\"", rendered)
        self.assertIn("verified_at:", rendered)
        # The emitted accounts block must not carry the handle uncommented.
        account_lines = [line for line in lines
                         if line.strip().startswith("telegram:")]
        self.assertEqual(account_lines, [])

    def test_what_it_emits_is_refused_until_a_person_fills_in_the_ids(self):
        """Provenance is present, so the refusal is about ids, not paperwork."""
        result = {"domain": "example.org", "reachable": True,
                  "pages": [{"url": "https://example.org/", "sha256": "abc",
                             "detail": "HTTP 200"}],
                  "handles": {"telegram": ["example_official"]}}
        rendered = "entities:\n" + "\n".join(
            verify_entities.declaration("example-org", "Example Org", result))
        path = os.path.join(tempfile.mkdtemp(), "emitted.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        entities = load_entities(path)
        self.assertEqual(len(entities), 1)
        # It loads -- provenance is there -- and it vouches for no account, so
        # it can never make an impersonator look verified.
        self.assertEqual(entities[0].accounts, {})
        self.assertEqual(entities[0].known_wallets, set())
        self.assertEqual(entities[0].official_domains, {"example.org"})


class TestBlockhashIsNotFetchedOnTheHotPath(unittest.IsolatedAsyncioTestCase):
    """An RPC round trip sat between the decision and the signature.

    Caching it naively would be worse than the round trip: a stale blockhash
    is a transaction the cluster silently refuses, and that looks exactly like
    a transaction that lost a race. So the cache refuses what it cannot vouch
    for, and a refusal costs the round trip rather than the fill.
    """

    HASH = "GfVcyD4kkTrj4bKc7WA9sZCin9JDbdT4Zkd3o4uGFvNn"
    OTHER = "11111111111111111111111111111111"

    def _rpc(self, *responses, fail=False):
        calls = []

        async def request(method, params):
            calls.append(method)
            if fail:
                raise ConnectionError("rpc down")
            if responses:
                return responses[min(len(calls) - 1, len(responses) - 1)]
            return {"value": {"blockhash": self.HASH, "lastValidBlockHeight": 300}}

        return SimpleNamespace(request=request, calls=calls)

    async def test_a_fresh_cache_serves_without_an_rpc_call(self):
        rpc = self._rpc()
        cache = BlockhashCache(rpc)
        self.assertTrue(await cache.refresh())
        self.assertEqual(len(rpc.calls), 1)
        for _ in range(50):
            state = cache.current()
            self.assertTrue(state.ok)
            self.assertEqual(state.blockhash, self.HASH)
        # Fifty signings, one call. That is the whole point.
        self.assertEqual(len(rpc.calls), 1)
        self.assertEqual(cache.report()["cache_hit_rate"], 1.0)

    async def test_a_stalled_refresher_refuses_rather_than_serving_a_stale_hash(self):
        cache = BlockhashCache(self._rpc(), max_age_s=5.0)
        await cache.refresh()
        state = cache.current(now=time.time() + 60)
        self.assertFalse(state.ok)
        self.assertIn("stalled", state.detail)
        # The hash is still reported, so an operator can see what was rejected.
        self.assertEqual(state.blockhash, self.HASH)

    async def test_a_hash_near_expiry_refuses_before_the_age_bound_bites(self):
        """Two different failures; one is not a substitute for the other."""
        cache = BlockhashCache(self._rpc(), max_age_s=300.0, min_slots_remaining=30)
        await cache.refresh()
        # 50s at ~0.4s/slot consumes 125 of 150 slots: 25 remain, below 30.
        state = cache.current(now=time.time() + 50)
        self.assertFalse(state.ok)
        self.assertIn("slots of validity", state.detail)
        self.assertLess(state.slots_remaining, 30)

    async def test_nothing_fetched_yet_is_blocked_not_an_empty_hash(self):
        state = BlockhashCache(self._rpc()).current()
        self.assertFalse(state.ok)
        self.assertEqual(state.blockhash, "")

    async def test_a_failed_refresh_keeps_the_previous_hash(self):
        """A transient hiccup must not become a round trip on every trade."""
        rpc = self._rpc()
        cache = BlockhashCache(rpc)
        await cache.refresh()
        cache.rpc = self._rpc(fail=True)
        self.assertFalse(await cache.refresh())
        self.assertTrue(cache.current().ok)
        self.assertEqual(cache.failures, 1)

    async def test_a_response_with_no_blockhash_is_a_failure_not_an_empty_state(self):
        cache = BlockhashCache(self._rpc({"value": {}}))
        self.assertFalse(await cache.refresh())
        self.assertIn("no blockhash", cache.last_error)

    async def test_the_builder_signs_from_cache_and_falls_back_when_refused(self):
        rpc = self._rpc()
        cache = BlockhashCache(rpc)
        await cache.refresh()
        builder = SolanaTransactionBuilder.__new__(SolanaTransactionBuilder)
        builder.rpc = rpc
        builder.blockhash_cache = cache
        builder.blockhash_fallbacks = 0
        builder.last_blockhash_status = ""

        from solders.hash import Hash as SoldersHash

        served = await builder._recent_blockhash()
        self.assertIsInstance(served, SoldersHash)
        self.assertEqual(len(rpc.calls), 1)
        self.assertEqual(builder.blockhash_fallbacks, 0)

        # Expire it, and the round trip is paid rather than a rejection risked.
        cache._fetched_at = time.time() - 600
        await builder._recent_blockhash()
        self.assertEqual(builder.blockhash_fallbacks, 1)
        self.assertEqual(len(rpc.calls), 2)

    async def test_the_report_says_whether_the_cache_is_doing_its_job(self):
        """A cache serving 40% is a hot path still paying for the other 60%."""
        cache = BlockhashCache(self._rpc(), max_age_s=5.0)
        await cache.refresh()
        cache.current()
        cache.current(now=time.time() + 600)
        report = cache.report()
        self.assertEqual(report["served_from_cache"], 1)
        self.assertEqual(report["refused"], 1)
        self.assertEqual(report["cache_hit_rate"], 0.5)
        self.assertEqual(report["refreshes"], 1)

    async def test_start_awaits_the_first_fetch_then_refreshes_in_the_background(self):
        rpc = self._rpc()
        cache = BlockhashCache(rpc, refresh_interval_s=0.1)
        self.assertTrue(await cache.start())
        # Already usable before any trade has been decided.
        self.assertTrue(cache.current().ok)
        await asyncio.sleep(0.35)
        await cache.stop()
        self.assertGreater(cache.refreshes, 1)


class TestWalletValueIsMeasuredNotWeighted(unittest.TestCase):
    """The ranking was five chosen weights and two invented multipliers.

    Nothing in the system could say whether following the resulting list made
    or lost money. This asks the only question with an answer: what did
    following this wallet do to our capital, at fills we could have got.
    """

    def _follow(self, wallet, multiple, **overrides):
        args = dict(wallet=wallet, token="mint", observed_at=time.time(),
                    executable_multiple=multiple)
        args.update(overrides)
        return FollowOutcome(**args)

    def _fill(self, model, wallet, multiples):
        for multiple in multiples:
            model.record(self._follow(wallet, multiple))
        return model

    def test_a_wallet_below_the_sample_floor_has_no_value_not_a_low_one(self):
        model = WalletValueModel(min_samples=12)
        self._fill(model, "w", [2.0] * 5)
        value = model.value("w")
        self.assertEqual(value.status, "DATA_BLOCKED")
        self.assertIn("5 followed outcomes", value.detail)
        self.assertFalse(value.followable)

    def test_the_value_is_expected_log_growth_not_a_composite(self):
        model = WalletValueModel(min_samples=4, shrinkage_strength=0.0)
        self._fill(model, "w", [2.0] * 20)
        value = model.value("w")
        self.assertAlmostEqual(value.mean_log_return, math.log(2.0), places=6)

    def test_a_rug_is_a_loss_in_the_number_not_a_separate_penalty(self):
        model = WalletValueModel(min_samples=4, shrinkage_strength=0.0)
        for _ in range(10):
            model.record(self._follow("w", 2.0))
        for _ in range(10):
            model.record(self._follow("w", 0.0001, rugged=True))
        value = model.value("w")
        self.assertLess(value.mean_log_return, 0.0)
        self.assertAlmostEqual(value.rug_rate, 0.5)
        self.assertFalse(value.followable)

    def test_six_lucky_trades_do_not_outrank_three_hundred_steady_ones(self):
        """Ranking on the mean is how a system ends up following noise."""
        model = WalletValueModel(min_samples=6, shrinkage_strength=10.0)
        self._fill(model, "lucky", [50.0, 40.0, 0.01, 0.01, 0.01, 0.01])
        self._fill(model, "steady", [1.6] * 300)
        ranked = [value.wallet for value in model.rank(limit=5)]
        self.assertEqual(ranked[0], "steady")
        self.assertGreater(model.value("steady").lower_bound,
                           model.value("lucky").lower_bound)

    def test_the_bound_is_below_the_mean_and_tightens_with_evidence(self):
        model = WalletValueModel(min_samples=6, shrinkage_strength=0.0)
        self._fill(model, "few", [3.0, 0.5, 4.0, 0.4, 2.0, 0.6])
        self._fill(model, "many", [3.0, 0.5, 4.0, 0.4, 2.0, 0.6] * 40)
        few, many = model.value("few"), model.value("many")
        self.assertLess(few.lower_bound, few.mean_log_return)
        # Same distribution, more evidence: the bound moves up toward the mean.
        self.assertGreater(many.lower_bound, few.lower_bound)

    def test_only_positive_growth_wallets_are_followable(self):
        model = WalletValueModel(min_samples=4, shrinkage_strength=0.0)
        self._fill(model, "good", [2.0] * 30)
        self._fill(model, "bad", [0.5] * 30)
        self.assertTrue(model.value("good").followable)
        self.assertFalse(model.value("bad").followable)
        self.assertEqual([value.wallet for value in model.rank()], ["good"])

    def test_an_unmeasurable_outcome_is_refused_not_recorded_as_break_even(self):
        """Averaging it in as 1.0 pulls every wallet toward break-even."""
        model = WalletValueModel(min_samples=2)
        self.assertFalse(model.record(self._follow(
            "w", 1.0, data_status="DATA_BLOCKED: no exit quote")))
        self.assertEqual(model.value("w").samples, 0)
        self.assertEqual(model.report()["rejected_outcomes"], 1)

    def test_a_total_loss_is_floored_rather_than_infinite(self):
        """One rug must not dominate any amount of evidence."""
        model = WalletValueModel(min_samples=2, shrinkage_strength=0.0)
        model.record(self._follow("w", 0.0, rugged=True))
        model.record(self._follow("w", 3.0))
        value = model.value("w")
        self.assertTrue(math.isfinite(value.mean_log_return))

    def test_regimes_are_scored_separately_when_asked(self):
        model = WalletValueModel(min_samples=4, shrinkage_strength=0.0)
        for _ in range(10):
            model.record(self._follow("w", 3.0, regime="ultra_early"))
            model.record(self._follow("w", 0.4, regime="post_migration"))
        early = model.value("w", "ultra_early")
        late = model.value("w", "post_migration")
        self.assertTrue(early.followable)
        self.assertFalse(late.followable)
        # And unfiltered, the two average out to something in between.
        self.assertLess(model.value("w").mean_log_return, early.mean_log_return)

    def test_history_is_bounded_so_a_wallet_is_judged_on_recent_trades(self):
        model = WalletValueModel(min_samples=4, history=50, shrinkage_strength=0.0)
        self._fill(model, "w", [10.0] * 50)
        self._fill(model, "w", [0.5] * 50)
        self.assertEqual(model.value("w").samples, 50)
        self.assertLess(model.value("w").mean_log_return, 0.0)

    def test_the_report_says_nothing_is_ranked_rather_than_showing_an_empty_top(self):
        model = WalletValueModel(min_samples=12)
        self._fill(model, "w", [2.0] * 3)
        report = model.report()
        self.assertEqual(report["status"], "DATA_BLOCKED")
        self.assertIn("none with 12", report["detail"])
        self.assertEqual(report["wallets_estimable"], 0)
        self.assertEqual(report["observations"], 3)


class TestExecutableMultipleIsOurFillNotTheirs(unittest.TestCase):
    def test_the_entry_that_counts_is_ours(self):
        # They got in at 1.0, we could only get in at 3.0, and it ended at 6.0.
        multiple, status = executable_multiple(1.0, 3.0, 6.0)
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(multiple, 2.0)

    def test_fees_come_out_of_the_number(self):
        plain, _ = executable_multiple(1.0, 1.0, 2.0)
        with_fee, _ = executable_multiple(1.0, 1.0, 2.0, fee_bps=100)
        self.assertLess(with_fee, plain)

    def test_a_partial_exit_with_no_mark_for_the_rest_is_blocked(self):
        """A 10x on a tenth of the position is not a 10x."""
        multiple, status = executable_multiple(1.0, 1.0, 10.0, capacity_ratio=0.1)
        self.assertTrue(status.startswith("DATA_BLOCKED"))
        self.assertEqual(multiple, 0.0)

    def test_a_partial_exit_with_a_mark_blends_the_two(self):
        multiple, status = executable_multiple(
            1.0, 1.0, 10.0, capacity_ratio=0.1, remainder_multiple=1.0)
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(multiple, 0.1 * 10.0 + 0.9 * 1.0)

    def test_unmeasured_prices_are_blocked_not_treated_as_one(self):
        for args in ((0.0, 1.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, -1.0)):
            _multiple, status = executable_multiple(*args)
            self.assertTrue(status.startswith("DATA_BLOCKED"), args)


class TestTheDeskMeasuresWhatFollowingReturns(unittest.IsolatedAsyncioTestCase):
    """A watch list nobody has scored is a list of wallets, not intelligence."""

    TOKEN = "So11111111111111111111111111111111111111112"

    def _curve(self, sol=30_000_000_000, tokens=1_000_000_000_000):
        return BondingCurveState(
            virtual_token_reserves=tokens, virtual_sol_reserves=sol,
            real_token_reserves=tokens, real_sol_reserves=sol,
            token_total_supply=tokens, complete=False, creator="c")

    def _desk(self, watched=("w",)):
        model = WalletValueModel(min_samples=2, shrinkage_strength=0.0)
        intel = SimpleNamespace(
            wallet_value=model,
            is_watched=lambda wallet: wallet in set(watched),
            record_follow_outcome=model.record)
        desk = SimpleNamespace(
            _latest_curve_state={self.TOKEN: self._curve()},
            _latest_pool_state={}, _follow_candidates={},
            _follow_resolved=0, _follow_unresolved=0,
            global_config={"follow_reference_sol": 0.5, "follow_horizon_seconds": 300.0},
            wallet_intel=intel,
            rug_hazard=SimpleNamespace(get_hazard=lambda token: None))
        desk._follow_quote = (
            lambda token, lamports: MemecoinQuantDesk._follow_quote(desk, token, lamports))
        desk._follow_exit_quote = (
            lambda token, size: MemecoinQuantDesk._follow_exit_quote(desk, token, size))
        return desk

    def _event(self, wallet="w", side="buy"):
        return {"wallet": wallet, "side": side, "timestamp": time.time()}

    def test_a_watched_wallet_buy_opens_a_follow_priced_after_their_trade(self):
        desk = self._desk()
        self.assertTrue(
            MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event()))
        candidate = desk._follow_candidates[self.TOKEN][0]
        self.assertEqual(candidate["wallet"], "w")
        self.assertGreater(candidate["size_tokens"], 0)
        self.assertEqual(candidate["cost_lamports"], 500_000_000)

    def test_an_unwatched_wallet_is_not_followed(self):
        """Measuring every wallet would rank the market, not our watch list."""
        desk = self._desk(watched=())
        self.assertFalse(
            MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event()))
        self.assertEqual(desk._follow_candidates, {})

    def test_a_sell_is_not_a_follow(self):
        desk = self._desk()
        self.assertFalse(MemecoinQuantDesk._open_follow_candidate(
            desk, self.TOKEN, self._event(side="sell")))

    def test_adding_to_a_position_is_one_decision_to_follow_not_three(self):
        desk = self._desk()
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        self.assertEqual(len(desk._follow_candidates[self.TOKEN]), 1)

    def test_a_token_we_cannot_quote_opens_no_follow(self):
        desk = self._desk()
        desk._latest_curve_state = {}
        self.assertFalse(
            MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event()))

    def test_nothing_resolves_before_the_horizon(self):
        desk = self._desk()
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        self.assertEqual(MemecoinQuantDesk._resolve_follow_candidates(desk), 0)
        self.assertEqual(len(desk._follow_candidates[self.TOKEN]), 1)

    def test_a_price_that_ran_resolves_as_a_gain_net_of_impact(self):
        desk = self._desk()
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        # The curve moved decisively in our favour.
        desk._latest_curve_state[self.TOKEN] = self._curve(
            sol=300_000_000_000, tokens=1_000_000_000_000)
        resolved = MemecoinQuantDesk._resolve_follow_candidates(
            desk, now=time.time() + 400)
        self.assertEqual(resolved, 1)
        outcome = list(desk.wallet_intel.wallet_value._outcomes["w"])[0]
        self.assertGreater(outcome.executable_multiple, 1.0)
        self.assertFalse(outcome.rugged)
        self.assertEqual(desk._follow_candidates, {})

    def test_a_token_whose_state_vanished_resolves_as_a_total_loss(self):
        """For a position we could not have quoted an exit for, that is what it was."""
        desk = self._desk()
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        desk._latest_curve_state = {}
        MemecoinQuantDesk._resolve_follow_candidates(desk, now=time.time() + 400)
        outcome = list(desk.wallet_intel.wallet_value._outcomes["w"])[0]
        self.assertTrue(outcome.rugged)
        self.assertEqual(outcome.executable_multiple, 0.0)

    def test_a_round_trip_at_a_flat_price_loses_the_spread(self):
        """Buying and selling the same curve is not break-even, and should not read as one."""
        desk = self._desk()
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        MemecoinQuantDesk._resolve_follow_candidates(desk, now=time.time() + 400)
        outcome = list(desk.wallet_intel.wallet_value._outcomes["w"])[0]
        self.assertLess(outcome.executable_multiple, 1.0)

    def test_the_report_distinguishes_open_follows_from_measured_value(self):
        desk = self._desk()
        MemecoinQuantDesk._open_follow_candidate(desk, self.TOKEN, self._event())
        report = MemecoinQuantDesk.follow_report(desk)
        self.assertEqual(report["open_follows"], 1)
        self.assertEqual(report["resolved"], 0)
        self.assertEqual(report["model"]["status"], "DATA_BLOCKED")


class TestAgeBandsDoNotSplitOnIntuition(unittest.TestCase):
    """Every extra band looks like more precision. It usually is not.

    Splitting halves the rows each model is fitted on, and any partition of a
    finite sample shows some difference between its halves -- so a split that
    looks justified usually is not.
    """

    def test_a_split_with_too_few_rows_a_side_is_blocked_however_different(self):
        """A side fed from its neighbour is the pooled model with a band's name."""
        warrant = split_warrant([0.0] * 10, [1.0] * 500, min_side_samples=60)
        self.assertEqual(warrant.status, "DATA_BLOCKED")
        self.assertIn("neighbour", warrant.detail)
        self.assertFalse(warrant.warranted)

    def test_two_halves_of_one_distribution_are_not_a_split(self):
        rng = random.Random(7)
        left = [rng.gauss(0.3, 0.2) for _ in range(400)]
        right = [rng.gauss(0.3, 0.2) for _ in range(400)]
        warrant = split_warrant(left, right)
        self.assertEqual(warrant.status, "NOT_WARRANTED")
        self.assertGreater(warrant.p_value, 0.01)
        self.assertIn("two copies of one model", warrant.detail)

    def test_a_real_step_with_enough_rows_is_warranted(self):
        rng = random.Random(11)
        left = [rng.gauss(0.15, 0.1) for _ in range(400)]
        right = [rng.gauss(0.45, 0.1) for _ in range(400)]
        warrant = split_warrant(left, right)
        self.assertEqual(warrant.status, "WARRANTED")
        self.assertLess(warrant.p_value, 0.01)
        self.assertGreater(warrant.difference, 0.2)

    def test_a_real_difference_still_needs_rows_on_both_sides(self):
        """Neither condition alone is enough."""
        rng = random.Random(13)
        big = [rng.gauss(0.15, 0.1) for _ in range(400)]
        tiny = [rng.gauss(0.45, 0.1) for _ in range(20)]
        self.assertEqual(split_warrant(big, tiny).status, "DATA_BLOCKED")

    def test_identical_constants_are_not_a_difference(self):
        warrant = split_warrant([0.5] * 100, [0.5] * 100)
        self.assertEqual(warrant.status, "NOT_WARRANTED")
        self.assertEqual(warrant.p_value, 1.0)

    def test_a_step_between_two_constants_is(self):
        warrant = split_warrant([0.0] * 100, [1.0] * 100)
        self.assertEqual(warrant.status, "WARRANTED")

    def test_testing_many_cuts_corrects_the_threshold(self):
        """Eight cuts at p<0.01, best reported, is p<0.08 called 0.01."""
        rng = random.Random(3)
        rows = [(age, rng.gauss(0.3, 0.2))
                for age in [rng.uniform(0, 10) for _ in range(1_000)]]
        report = evaluate_cuts(rows, band="early", cuts=(2.0, 5.0, 8.0), target="p_2x")
        self.assertAlmostEqual(report["corrected_alpha"], 0.01 / 3)
        self.assertEqual(report["status"], "NOT_WARRANTED")
        self.assertIsNone(report["recommended_cut"])
        self.assertIn("leave it as one model", report["detail"])

    def test_a_genuine_boundary_is_found_and_named_once(self):
        rng = random.Random(5)
        rows = []
        for _ in range(1_000):
            age = rng.uniform(0, 10)
            outcome = rng.gauss(0.15 if age < 5 else 0.5, 0.1)
            rows.append((age, outcome))
        report = evaluate_cuts(rows, band="early", cuts=(2.5, 5.0, 7.5), target="p_2x")
        self.assertEqual(report["status"], "WARRANTED")
        # Singular: a band that "could" be split three ways at once is a band
        # whose evidence has not been read.
        self.assertEqual(report["recommended_cut"], 5.0)

    def test_the_shipped_table_is_four_bands_and_covers_every_age(self):
        """The table does not grow on intuition; it grows on a recorded warrant."""
        self.assertEqual(len(AGE_BANDS), 4)
        self.assertEqual(AGE_BANDS[0][1], 0.0)
        self.assertEqual(AGE_BANDS[-1][2], float("inf"))
        for (_, _, high), (_, low, _) in zip(AGE_BANDS, AGE_BANDS[1:]):
            # No gap and no overlap: every age belongs to exactly one brain.
            self.assertEqual(high, low)
        for age in (0.0, 0.4, 0.5, 4.9, 5.0, 59.9, 60.0, 10_000.0):
            self.assertIn(band_for(age), {name for name, _, _ in AGE_BANDS})

    def test_the_band_table_carries_the_rule_for_changing_it(self):
        """A convention nobody wrote down is a convention that lapses."""
        source = Path(multihead_predictor_module.__file__).read_text()
        self.assertIn("band_split", source)
        self.assertIn("split_warrants", source)


class TestTheRunbookMatchesTheSystem(unittest.TestCase):
    """A runbook whose commands error is worse than no runbook.

    Every key these snippets read has to exist in the payload the desk
    actually serves, and the unit has to point at a file that is there.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _runbook(self) -> str:
        return (self.ROOT / "RUNBOOK.md").read_text(encoding="utf-8")

    def test_the_runbook_the_unit_documents_exists(self):
        unit = (self.ROOT / "deploy" / "systemd" / "memecoin-shadow.service").read_text()
        self.assertIn("RUNBOOK.md", unit)
        self.assertTrue((self.ROOT / "RUNBOOK.md").exists())

    def test_every_status_key_the_runbook_reads_is_a_key_the_desk_serves(self):
        text = self._runbook()
        # The top-level sections the snippets index into.
        for key in ("source_mesh", "entity_registry", "forward_evidence",
                    "native_route"):
            self.assertIn(f"['{key}']", text, key)
        source = (self.ROOT / "src" / "main.py").read_text(encoding="utf-8")
        for key in ("source_mesh", "entity_registry", "forward_evidence",
                    "native_route"):
            self.assertIn(f'"{key}":', source, key)

    def test_the_forward_evidence_shape_the_runbook_reads_is_the_shape_reported(self):
        ledger = ForwardEvidence(os.path.join(tempfile.mkdtemp(), "evidence.json"))
        report = ledger.report()
        self.assertIn("evidence", report)
        distance = report["distance"]
        for key in ("stage", "next_stage", "progress", "slowest", "verdict"):
            self.assertIn(key, distance, key)
        for row in distance["progress"].values():
            self.assertEqual(set(row), {"have", "need", "fraction"})
        self.assertIn("failures", distance["verdict"])

    def test_the_native_route_shape_the_runbook_reads_is_the_shape_reported(self):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.pump_route = None
        engine.pumpswap_route = None
        engine.pool_state_provider = None
        engine.pool_account_provider = None
        engine.native_route_attempts = defaultdict(int)
        engine.landing_model = LandingModel()
        engine.last_bid = {}
        engine.stream_confirmations = 0
        engine.poll_confirmations = 0
        engine._signature_waiters = {}
        engine.reconcile_min_interval = 0.01
        engine.tx_builder = SimpleNamespace()
        report = engine.native_route_report()
        for key in ("prepared_share", "blockhash", "outcomes",
                    "pool_state_wired", "pool_account_wired"):
            self.assertIn(key, report, key)

    def test_the_ports_agree_across_the_unit_the_installer_and_the_runbook(self):
        """A runbook curling the wrong port reads as a dead desk."""
        unit = (self.ROOT / "deploy" / "systemd" / "memecoin-shadow.service").read_text()
        installer = (self.ROOT / "deploy" / "install_shadow.sh").read_text()
        self.assertIn("HEALTH_PORT=18080", unit)
        self.assertIn("18080", installer)
        self.assertIn("localhost:18080", self._runbook())

    def test_the_status_endpoint_binds_loopback_by_default(self):
        """/status serves the desk's interior; 0.0.0.0 publishes all of it."""
        source = (self.ROOT / "src" / "main.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("HEALTH_HOST", "127.0.0.1")', source)
        self.assertNotIn('web.TCPSite(self._web_runner, "0.0.0.0"', source)
        unit = (self.ROOT / "deploy" / "systemd" / "memecoin-shadow.service").read_text()
        self.assertIn("HEALTH_HOST=127.0.0.1", unit)

    def test_a_reinstall_does_not_delete_what_was_verified_on_the_host(self):
        """They are produced here, are not in the repo, and --delete removes them.

        Losing them silently un-configures the source mesh and empties the
        entity registry, and the desk keeps running and reports less coverage
        with nothing saying why.
        """
        installer = (self.ROOT / "deploy" / "install_shadow.sh").read_text()
        self.assertIn("--delete", installer)
        self.assertIn("--exclude 'config/*.verified.yaml'", installer)
        # And the same files are out of version control, so the exclusion is
        # the only thing protecting them.
        ignored = (self.ROOT / ".gitignore").read_text()
        self.assertIn("config/sources.verified.yaml", ignored)
        self.assertIn("config/entities.verified.yaml", ignored)

    def test_the_shadow_unit_never_carries_a_live_acknowledgement(self):
        unit = (self.ROOT / "deploy" / "systemd" / "memecoin-shadow.service").read_text()
        # Cleared AFTER the environment file, so a stale env cannot promote a
        # shadow run into a live one.
        self.assertLess(unit.index("EnvironmentFile"),
                        unit.index("Environment=ALLOW_LIVE_TRADING="))
        self.assertIn("Environment=SOLANA_PRIVATE_KEY=", unit)
        self.assertIn("--dry-run", unit)

    def test_the_start_limit_directives_are_where_systemd_reads_them(self):
        """Under [Service] systemd silently ignores them."""
        unit = (self.ROOT / "deploy" / "systemd" / "memecoin-shadow.service").read_text()
        # The SECTION HEADER, not the comment above it that names the trap.
        head = unit[:unit.index("\n[Service]\n")]
        self.assertIn("StartLimitIntervalSec", head)
        self.assertIn("StartLimitBurst", head)


class TestTheRustKernelIsOnTheCanonicalPath(unittest.TestCase):
    """A kernel that exists and is never called is a kernel never written.

    Connecting it by swapping the call would be worse: the Rust path has
    never decided anything in production, and promoting an unproven
    implementation onto the money path because its unit tests pass is the
    move this codebase refuses everywhere else. So it is promoted the way a
    model is -- on evidence, with an automatic and loud demotion.
    """

    LEVELS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)

    def _survival(self, **overrides):
        args = dict(levels=self.LEVELS, p_rug_30s=0.1, p_rug_5m=0.2,
                    expected_feasible_multiple=3.0)
        args.update(overrides)
        return SurvivalInputs(**args)

    def _bins(self, survival):
        prediction = MultiHeadPrediction(token="t", chain="solana", timestamp=0.0)
        for (target, _multiple), value in zip(SURVIVAL_LEVELS, survival.levels):
            setattr(prediction, target.value, float(value))
        prediction.p_rug_30s = survival.p_rug_30s
        prediction.p_rug_5m = survival.p_rug_5m
        prediction.expected_feasible_multiple = survival.expected_feasible_multiple
        return [(probability, gross) for _, probability, gross
                in ElogwEngine.probability_bins(prediction)]

    def _state(self, survival, **overrides):
        args = dict(held_fraction=0.02, current_multiple=1.4,
                    forward_bins=tuple(self._bins(survival)),
                    exit_cost=0.02, entry_cost=0.02,
                    exit_capacity_ratio=0.8, escape_probability=0.7)
        args.update(overrides)
        return PositionState(**args)

    def _kernel(self, **overrides):
        args = dict(mode="shadow", promote_after=3)
        args.update(overrides)
        return T0Kernel(ActionValuePolicy(min_edge=1e-4, max_add_fraction=0.5), **args)

    def test_the_kernel_is_wired_even_when_the_extension_is_absent(self):
        """A missing toolchain must not stop the desk from deciding."""
        kernel = self._kernel()
        survival = self._survival()
        decision = kernel.score(self._state(survival), survival=survival)
        self.assertEqual(decision.status, "OK")
        self.assertIsNotNone(decision.kernel)
        if not kernel.rust_available:
            self.assertEqual(decision.kernel["source"], "python")
            self.assertIn("unavailable", kernel.report()["native"])

    def test_a_state_the_kernel_cannot_express_goes_to_python_and_is_counted(self):
        """'Rust was not asked' and 'Rust agreed' are different facts."""
        kernel = self._kernel()
        survival = self._survival()
        state = self._state(survival, held_fraction=0.0,
                            reentry_bins=tuple(self._bins(survival)))
        decision = kernel.score(state, survival=survival)
        self.assertEqual(decision.kernel["source"], "python")
        report = kernel.report()
        self.assertEqual(report["not_expressible_in_kernel"], 1)
        self.assertEqual(report["agreements"], 0)

    def test_a_caller_with_no_survival_inputs_is_counted_separately(self):
        kernel = self._kernel()
        decision = kernel.score(self._state(self._survival()))
        self.assertEqual(decision.kernel["source"], "python")
        self.assertEqual(kernel.report()["without_survival_inputs"], 1)

    def test_off_never_consults_rust(self):
        kernel = self._kernel(mode="off")
        survival = self._survival()
        kernel.score(self._state(survival), survival=survival)
        self.assertEqual(kernel.report()["compared"], 0)
        self.assertFalse(kernel.rust_authoritative)

    def test_an_unknown_mode_falls_back_to_shadow_rather_than_to_rust(self):
        kernel = T0Kernel(ActionValuePolicy(), mode="turbo")
        self.assertIs(kernel.mode, KernelMode.SHADOW)
        self.assertFalse(kernel.rust_authoritative)


@unittest.skipUnless(_NATIVE_KERNEL, "solana_fastpath is not importable")
class TestKernelParityWithTheNativeExtension(unittest.TestCase):
    """The comparison is POLICY to POLICY.

    Rust is called with permissive limits so its safety layer cannot bind,
    because the desk's own safety layer is unchanged and still authoritative.
    Comparing policy against policy-plus-safety would report a divergence
    every time safety correctly refused.
    """

    LEVELS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)

    def _survival(self, **overrides):
        args = dict(levels=self.LEVELS, p_rug_30s=0.1, p_rug_5m=0.2,
                    expected_feasible_multiple=3.0)
        args.update(overrides)
        return SurvivalInputs(**args)

    def _bins(self, survival):
        prediction = MultiHeadPrediction(token="t", chain="solana", timestamp=0.0)
        for (target, _multiple), value in zip(SURVIVAL_LEVELS, survival.levels):
            setattr(prediction, target.value, float(value))
        prediction.p_rug_30s = survival.p_rug_30s
        prediction.p_rug_5m = survival.p_rug_5m
        prediction.expected_feasible_multiple = survival.expected_feasible_multiple
        return [(probability, gross) for _, probability, gross
                in ElogwEngine.probability_bins(prediction)]

    def _state(self, survival, **overrides):
        args = dict(held_fraction=0.02, current_multiple=1.4,
                    forward_bins=tuple(self._bins(survival)),
                    exit_cost=0.02, entry_cost=0.02,
                    exit_capacity_ratio=0.8, escape_probability=0.7)
        args.update(overrides)
        return PositionState(**args)

    def _kernel(self, **overrides):
        args = dict(mode="shadow", promote_after=3)
        args.update(overrides)
        return T0Kernel(ActionValuePolicy(min_edge=1e-4, max_add_fraction=0.5), **args)

    RESERVES = {"virtual_sol": 30_000_000_000, "virtual_token": 1_000_000_000_000}

    def test_the_two_implementations_build_the_same_bins(self):
        """The parity has to start here, or nothing above it means anything."""
        import solana_fastpath

        survival = self._survival()
        python_bins = [(round(p, 12), round(g, 12)) for p, g in self._bins(survival)]
        rust_bins = [(round(p, 12), round(g, 12)) for p, g in solana_fastpath.survival_bins(
            list(survival.levels), survival.p_rug_30s, survival.p_rug_5m,
            survival.expected_feasible_multiple)]
        self.assertEqual(python_bins, rust_bins)

    def test_they_agree_across_a_range_of_positions(self):
        kernel = self._kernel(promote_after=10_000)
        survival = self._survival()
        for held in (0.0, 0.01, 0.05, 0.2):
            for multiple in (0.5, 1.0, 2.5, 12.0):
                for capacity in (0.1, 0.5, 1.0):
                    state = self._state(survival, held_fraction=held,
                                        current_multiple=multiple,
                                        exit_capacity_ratio=capacity,
                                        add_fraction=0.01 if held else None,
                                        probe_fraction=None if held else 0.01)
                    decision = kernel.score(state, survival=survival, **self.RESERVES)
                    self.assertEqual(decision.status, "OK")
        report = kernel.report()
        self.assertGreater(report["compared"], 40)
        self.assertEqual(report["divergences"], 0, report["divergence_examples"])
        self.assertEqual(report["rust_errors"], 0)

    def test_shadow_never_lets_rust_move_capital(self):
        kernel = self._kernel(mode="shadow", promote_after=1)
        survival = self._survival()
        for _ in range(10):
            decision = kernel.score(self._state(survival), survival=survival, **self.RESERVES)
            self.assertEqual(decision.kernel["source"], "python")
        self.assertFalse(kernel.rust_authoritative)
        self.assertGreater(kernel.report()["agreements"], 0)

    def test_auto_promotes_only_after_a_run_of_measured_agreement(self):
        kernel = self._kernel(mode="auto", promote_after=5)
        survival = self._survival()
        sources = [kernel.score(self._state(survival), survival=survival, **self.RESERVES).kernel["source"]
                   for _ in range(8)]
        # Python decides until the run is established, then Rust does.
        self.assertEqual(sources[:5], ["python"] * 5)
        self.assertEqual(sources[5:], ["rust"] * 3)
        self.assertTrue(kernel.rust_authoritative)
        self.assertEqual(kernel.report()["decisions_by_rust"], 3)

    def test_one_disagreement_demotes_it_for_the_session(self):
        """Demotion is automatic and permanent; re-promotion needs a human."""
        kernel = self._kernel(mode="auto", promote_after=2)
        survival = self._survival()
        for _ in range(4):
            kernel.score(self._state(survival), survival=survival, **self.RESERVES)
        self.assertTrue(kernel.rust_authoritative)

        # Force a disagreement by making the Python policy answer differently.
        class Contrarian:
            min_edge = 1e-4
            max_add_fraction = 0.5

            def score(self, state):
                return Decision(status="OK", action=ActionValue.EXIT, q=99.0)

        kernel.policy = Contrarian()
        kernel.score(self._state(survival), survival=survival, **self.RESERVES)
        self.assertTrue(kernel.demoted_reason)
        self.assertFalse(kernel.rust_authoritative)
        # And it stays demoted however many agreements follow.
        kernel.policy = ActionValuePolicy(min_edge=1e-4, max_add_fraction=0.5)
        for _ in range(20):
            decision = kernel.score(self._state(survival), survival=survival, **self.RESERVES)
            self.assertEqual(decision.kernel["source"], "python")
        self.assertFalse(kernel.rust_authoritative)

    def test_a_kernel_that_raises_does_not_take_the_desk_with_it(self):
        kernel = self._kernel(mode="auto", promote_after=1)
        survival = self._survival()

        class Exploding:
            def t0_decide(self, *args, **kwargs):
                raise RuntimeError("boom")

        kernel.native = Exploding()
        decision = kernel.score(self._state(survival), survival=survival, **self.RESERVES)
        self.assertEqual(decision.status, "OK")
        self.assertEqual(decision.kernel["source"], "python")
        self.assertEqual(kernel.report()["rust_errors"], 1)
        self.assertFalse(kernel.rust_authoritative)

    def test_the_report_says_whether_the_money_path_is_on_rust(self):
        kernel = self._kernel(mode="auto", promote_after=2)
        survival = self._survival()
        for _ in range(6):
            kernel.score(self._state(survival), survival=survival, **self.RESERVES)
        report = kernel.report()
        self.assertTrue(report["rust_authoritative"])
        self.assertGreater(report["rust_share"], 0.0)
        self.assertEqual(report["status"], "OK")


class TestLiveMarkingIsLocal(unittest.IsolatedAsyncioTestCase):
    """The desk paid for its latency and handed it back at the moment of deciding.

    `_mark_position` used the streamed mark only under `if self.dry_run`, so
    every LIVE redecision waited on a router quote after an event that had
    already arrived instantly.
    """

    TOKEN = "So11111111111111111111111111111111111111112"

    def _curve(self, sol=30_000_000_000, tokens=1_000_000_000_000):
        return BondingCurveState(
            virtual_token_reserves=tokens, virtual_sol_reserves=sol,
            real_token_reserves=tokens, real_sol_reserves=sol,
            token_total_supply=tokens, complete=False, creator="c")

    def _desk(self, *, dry_run=False, curve=True, stream=None, quote=None):
        calls = []

        async def get_quote(*args, **kwargs):
            calls.append(args)
            return quote

        desk = SimpleNamespace(
            dry_run=dry_run, offline=False, sol_price_usd=150.0,
            _latest_curve_state={self.TOKEN: self._curve()} if curve else {},
            _latest_pool_state={},
            _latest_stream_mark={self.TOKEN: stream} if stream else {},
            _marks_local=0, _marks_router=0, _mark_checked_at={},
            _mark_checks=0, _mark_checks_blocked=0, _mark_checks_diverged=0,
            _mark_drift_total=0.0, _mark_divergences=[],
            global_config={},
            jupiter=SimpleNamespace(get_quote=get_quote),
            rug_hazard=SimpleNamespace(record_observation=lambda *a: None),
            dataset_builder=SimpleNamespace(record_market_observation=lambda *a: None),
            counterfactual_lab=SimpleNamespace(record_market_observation=lambda *a: None),
        )
        desk.quote_calls = calls
        desk.backgrounded = []

        def spawn(coroutine):
            desk.backgrounded.append(coroutine)
            coroutine.close()

        desk._spawn_background = spawn
        for name in ("_local_mark", "_follow_exit_quote", "_maybe_cross_check_mark",
                     "_cross_check_mark"):
            setattr(desk, name, functools.partial(getattr(MemecoinQuantDesk, name), desk))
        return desk

    def _position(self, size=1_000_000):
        return {"size_tokens": size, "remaining_cost_usd": 100.0}

    async def test_a_live_desk_marks_without_asking_the_router(self):
        desk = self._desk(dry_run=False)
        marked = await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position())
        self.assertIsNotNone(marked)
        self.assertEqual(desk.quote_calls, [])
        self.assertEqual(desk._marks_local, 1)
        self.assertEqual(desk._marks_router, 0)

    async def test_the_local_mark_carries_its_own_price_impact(self):
        """What the position is worth and what it would fetch are different."""
        desk = self._desk()
        small = MemecoinQuantDesk._local_mark(desk, self.TOKEN, self._position(1_000))
        large = MemecoinQuantDesk._local_mark(desk, self.TOKEN,
                                              self._position(100_000_000_000))
        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        # Per token, the big exit fetches strictly less.
        self.assertLess(large[1] / 100_000_000_000, small[1] / 1_000)

    async def test_a_recent_stream_ratio_is_used_when_no_local_quote_can_be_made(self):
        desk = self._desk(curve=False,
                          stream={"multiple": 2.5, "timestamp": time.time()})
        marked = await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position())
        self.assertEqual(marked[0], 2.5)
        self.assertEqual(desk.quote_calls, [])

    async def test_a_stale_stream_ratio_is_not_used(self):
        """A price from a minute ago is not a mark."""
        desk = self._desk(curve=False,
                          stream={"multiple": 2.5, "timestamp": time.time() - 60},
                          quote=SimpleNamespace(output_amount=250_000_000,
                                                price_impact_pct=0.01))
        marked = await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position())
        self.assertIsNotNone(marked)
        self.assertEqual(len(desk.quote_calls), 1)
        self.assertEqual(desk._marks_router, 1)

    async def test_with_nothing_local_the_round_trip_is_paid_rather_than_skipped(self):
        """A decision priced on no mark at all is worse than a slow one."""
        desk = self._desk(curve=False, quote=None)
        self.assertIsNone(
            await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position()))
        self.assertEqual(len(desk.quote_calls), 1)

    async def test_the_router_is_still_asked_afterwards_as_a_cross_check(self):
        desk = self._desk()
        await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position())
        # Off the decision path: scheduled, not awaited.
        self.assertEqual(len(desk.backgrounded), 1)
        self.assertEqual(desk.quote_calls, [])

    async def test_the_cross_check_is_sampled_not_run_on_every_mark(self):
        """Running it every time reintroduces the quote storm one step later."""
        desk = self._desk()
        for _ in range(5):
            await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position())
        self.assertEqual(len(desk.backgrounded), 1)
        self.assertEqual(desk._marks_local, 5)

    async def test_a_cross_check_that_disagrees_is_recorded_and_moves_no_capital(self):
        desk = self._desk(quote=SimpleNamespace(output_amount=500_000_000,
                                                price_impact_pct=0.01))
        await MemecoinQuantDesk._cross_check_mark(desk, self.TOKEN, 1_000_000, 1.0, 100.0)
        self.assertEqual(desk._mark_checks, 1)
        self.assertEqual(desk._mark_checks_diverged, 1)
        self.assertEqual(len(desk._mark_divergences), 1)

    async def test_a_cross_check_that_agrees_is_not_a_divergence(self):
        desk = self._desk(quote=SimpleNamespace(output_amount=100_000_000,
                                                price_impact_pct=0.01))
        await MemecoinQuantDesk._cross_check_mark(desk, self.TOKEN, 1_000_000, 1.0, 100.0)
        self.assertEqual(desk._mark_checks, 1)
        self.assertEqual(desk._mark_checks_diverged, 0)

    async def test_a_router_that_cannot_answer_is_not_counted_as_agreement(self):
        desk = self._desk(quote=None)
        await MemecoinQuantDesk._cross_check_mark(desk, self.TOKEN, 1_000_000, 1.0, 100.0)
        self.assertEqual(desk._mark_checks, 0)
        self.assertEqual(desk._mark_checks_blocked, 1)

    async def test_the_report_separates_being_fast_from_being_right(self):
        """Marking 100% locally and drifting 40% is fast and wrong."""
        desk = self._desk()
        await MemecoinQuantDesk._mark_position(desk, self.TOKEN, self._position())
        report = MemecoinQuantDesk.mark_report(desk)
        self.assertEqual(report["local_share"], 1.0)
        self.assertEqual(report["status"], "OK")
        self.assertIsNone(report["mean_drift"])


class TestEveryRaceBidsOnItsOwnEconomics(unittest.TestCase):
    """A $500 position is not $500 of expected value -- and nor is a $500 exit."""

    def _desk(self, equity=10_000.0):
        desk = SimpleNamespace(wallet_equity_usd=equity)
        desk._exit_edge_usd = functools.partial(MemecoinQuantDesk._exit_edge_usd, desk)
        return desk

    def _position(self, **overrides):
        position = {"remaining_cost_usd": 200.0, "current_multiple": 3.0,
                    "action_value": {"q": 0.002}}
        position.update(overrides)
        return position

    def test_an_ordinary_bank_bids_the_edge_the_decision_priced(self):
        desk = self._desk()
        edge = desk._exit_edge_usd(self._position(), 0.25, "action_bank_25")
        self.assertAlmostEqual(edge, 0.002 * 10_000)

    def test_an_escape_bids_the_slice_because_missing_it_loses_the_slice(self):
        desk = self._desk()
        edge = desk._exit_edge_usd(self._position(), 1.0, "rug_hazard_critical")
        # 200 cost at 3x, all of it.
        self.assertAlmostEqual(edge, 600.0)
        # And that is far more than the routine bank would have bid.
        self.assertGreater(edge, desk._exit_edge_usd(self._position(), 1.0, "action_exit"))

    def test_a_partial_escape_bids_only_the_part_being_escaped(self):
        desk = self._desk()
        self.assertAlmostEqual(
            desk._exit_edge_usd(self._position(), 0.5, "escape_urgent"), 300.0)

    def test_an_explicitly_supplied_edge_wins(self):
        desk = self._desk()
        self.assertAlmostEqual(
            desk._exit_edge_usd(self._position(), 1.0, "rug_hazard_high", 12.5), 12.5)

    def test_an_unpriced_ordinary_exit_bids_nothing_rather_than_the_notional(self):
        """Bidding the notional overpays for a marginal trade, every time."""
        desk = self._desk()
        self.assertEqual(
            desk._exit_edge_usd(self._position(action_value={}), 1.0, "action_exit"), 0.0)

    def test_the_scale_in_passes_its_marginal_elogw_to_the_bid(self):
        """An ADD fell back to the fixed ladder while the entry beside it bid."""
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        block = source[source.index("attempt = {**_jsonable(result), \"scale_in\": True")
                       - 2_000:source.index("attempt = {**_jsonable(result), \"scale_in\": True")]
        self.assertIn("expected_edge_usd=max(0.0, gain * max(self.wallet_equity_usd, 0.0))",
                      block)
        self.assertIn("sol_price_usd=self.sol_price_usd", block)

class TestPumpSwapConstruction(unittest.TestCase):
    """The last DATA_BLOCKED that was never about missing information.

    PumpSwap's account lists are absent from Pump's prose docs and present in
    idl/pump_amm.json, in the same repository. The block was looking in the
    wrong file.
    """

    MINT = "So11111111111111111111111111111111111111112"
    OTHER = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def _pool(self, **overrides):
        base = dict(status="OK", pool=self.OTHER, base_mint=self.MINT,
                    quote_mint=self.MINT, pool_base_token_account=self.OTHER,
                    pool_quote_token_account=self.OTHER, creator=self.OTHER,
                    coin_creator=self.MINT)
        base.update(overrides)
        return PoolState(**base)

    def test_the_account_lists_come_from_the_idl(self):
        self.assertEqual(len(account_names(PUMP_AMM_IDL, "buy")), 23)
        self.assertEqual(len(account_names(PUMP_AMM_IDL, "sell")), 21)
        route = PumpSwapRoute()
        buy = route.build_buy(self._pool(), self.OTHER, 1_000, 5_000)
        sell = route.build_sell(self._pool(), self.OTHER, 1_000, 500)
        self.assertEqual(buy.status, "OK")
        self.assertEqual(sell.status, "OK")
        self.assertEqual(len(buy.accounts), 23)
        self.assertEqual(len(sell.accounts), 21)

    def test_only_the_buy_tracks_volume(self):
        self.assertIn("global_volume_accumulator", account_names(PUMP_AMM_IDL, "buy"))
        self.assertNotIn("global_volume_accumulator", account_names(PUMP_AMM_IDL, "sell"))

    def test_the_buy_carries_the_option_bool_the_sell_does_not(self):
        """`track_volume` is an OptionBool, encoded explicitly as absent."""
        route = PumpSwapRoute()
        buy = route.build_buy(self._pool(), self.OTHER, 1_000, 5_000)
        sell = route.build_sell(self._pool(), self.OTHER, 1_000, 500)
        self.assertEqual(len(buy.data), 8 + 8 + 8 + 1)
        self.assertEqual(buy.data[-1], 0)
        self.assertEqual(len(sell.data), 8 + 8 + 8)

    def test_an_account_declared_before_its_own_seed_still_resolves(self):
        """coin_creator_vault_ata is account 18; its authority is account 19."""
        route = PumpSwapRoute()
        buy = route.build_buy(self._pool(), self.OTHER, 1_000, 5_000)
        names = account_names(PUMP_AMM_IDL, "buy")
        authority = buy.accounts[names.index("coin_creator_vault_authority")].pubkey
        ata = buy.accounts[names.index("coin_creator_vault_ata")].pubkey
        self.assertTrue(authority and ata and authority != ata)

    def test_the_creator_vault_follows_coin_creator_not_creator(self):
        """The two sit adjacent in the layout with the same type."""
        route = PumpSwapRoute()
        names = account_names(PUMP_AMM_IDL, "buy")
        index = names.index("coin_creator_vault_authority")
        as_built = route.build_buy(self._pool(), self.OTHER, 10, 20).accounts[index].pubkey
        swapped = route.build_buy(
            self._pool(coin_creator=self.OTHER), self.OTHER, 10, 20).accounts[index].pubkey
        self.assertNotEqual(as_built, swapped)

    def test_a_mayhem_pool_draws_from_the_reserved_recipient_set(self):
        route = PumpSwapRoute()
        names = account_names(PUMP_AMM_IDL, "buy")
        index = names.index("protocol_fee_recipient")
        normal = route.build_buy(self._pool(), self.OTHER, 10, 20).accounts[index].pubkey
        mayhem = route.build_buy(
            self._pool(is_mayhem_mode=True), self.OTHER, 10, 20).accounts[index].pubkey
        self.assertIn(normal, fee_recipients()["normal"])
        self.assertIn(mayhem, fee_recipients()["reserved"])

    def test_a_pool_address_is_an_input_never_derived_from_the_mint(self):
        """A guessed index derives a real address that is not this pool."""
        blocked = PumpSwapRoute().build_buy(self._pool(pool=""), self.OTHER, 10, 20)
        self.assertEqual(blocked.status, "DATA_BLOCKED")
        self.assertIn("never derived", blocked.detail)

    def test_the_pool_index_is_part_of_the_address(self):
        first = derive_pool(0, self.OTHER, self.MINT, self.MINT)
        second = derive_pool(1, self.OTHER, self.MINT, self.MINT)
        self.assertNotEqual(first, second)

    def test_an_undecoded_pool_blocks_construction(self):
        blocked = PumpSwapRoute().build_buy(
            PoolState(status="DATA_BLOCKED", detail="nope"), self.OTHER, 10, 20)
        self.assertEqual(blocked.status, "DATA_BLOCKED")

    def test_an_unbounded_trade_is_refused(self):
        route = PumpSwapRoute()
        self.assertEqual(route.build_buy(self._pool(), self.OTHER, 10, 0).status, "REJECTED")
        self.assertEqual(route.build_sell(self._pool(), self.OTHER, 0, 10).status, "REJECTED")

    def test_the_pool_account_decodes_and_a_wrong_discriminator_does_not(self):
        from solders.pubkey import Pubkey
        body = (b"\x01" + (7).to_bytes(2, "little")
                + bytes(Pubkey.from_string(self.OTHER)) * 6
                + (123).to_bytes(8, "little")
                + bytes(Pubkey.from_string(self.MINT))
                + b"\x01\x00" + (-5).to_bytes(16, "little", signed=True))
        good = parse_pool(POOL_DISCRIMINATOR + body, address=self.OTHER)
        self.assertTrue(good.ok)
        self.assertEqual(good.index, 7)
        self.assertEqual(good.lp_supply, 123)
        self.assertEqual(good.coin_creator, self.MINT)
        self.assertTrue(good.is_mayhem_mode)
        self.assertFalse(good.is_cashback_coin)
        self.assertEqual(good.virtual_quote_reserves, -5)
        self.assertEqual(parse_pool(bytes(8) + body).status, "DATA_BLOCKED")


class TestGeneratedFlagsCannotDrift(unittest.TestCase):
    """The Rust builder carried three wrong flags for as long as it existed.

    Because they were transcribed from prose. Fixing them by hand fixes them
    once; generating them fixes them permanently, and makes the next upstream
    change a regeneration rather than a re-reading.
    """

    def test_the_checked_in_rust_table_matches_the_idl(self):
        result = subprocess.run(
            [sys.executable, "tools/gen_account_flags.py", "--check"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_generated_table_agrees_with_the_python_route(self):
        """Two builders, one source. They must agree account for account."""
        generated = (Path(__file__).resolve().parents[1] / "native" / "solana_fastpath"
                     / "src" / "generated_flags.rs").read_text()
        for prefix, idl, name in (("BUY_V2", PUMP_IDL, "buy_v2"),
                                  ("SELL_V2", PUMP_IDL, "sell_v2"),
                                  ("PUMPSWAP_BUY", PUMP_AMM_IDL, "buy"),
                                  ("PUMPSWAP_SELL", PUMP_AMM_IDL, "sell")):
            names = account_names(idl, name)
            self.assertIn(f"pub const {prefix}_ACCOUNT_COUNT: usize = {len(names)};",
                          generated)
            accounts = instruction(idl, name)["accounts"]
            writable = [index for index, account in enumerate(accounts, 1)
                        if account.get("writable")]
            self.assertIn(f"pub const {prefix}_WRITABLE: [usize; {len(writable)}] = "
                          f"{writable};".replace("'", ""), generated)

    def test_the_prose_and_the_program_disagree_and_the_program_wins(self):
        """Documented here so the next reader does not re-transcribe the prose."""
        accounts = instruction(PUMP_IDL, "buy_v2")["accounts"]
        by_name = {account["name"]: account for account in accounts}
        self.assertTrue(by_name["fee_recipient"].get("writable"))
        self.assertTrue(by_name["buyback_fee_recipient"].get("writable"))
        self.assertFalse(by_name["global_volume_accumulator"].get("writable", False))
        # And `user` is writable on the sell too, which the prose denied.
        sell = {account["name"]: account
                for account in instruction(PUMP_IDL, "sell_v2")["accounts"]}
        self.assertTrue(sell["user"].get("writable"))
        self.assertTrue(sell["user"].get("signer"))


class TestEventDrivenRuntime(unittest.IsolatedAsyncioTestCase):
    """T0 ran on a 500ms clock, which is not a sniper's control loop.

    One pass dequeued a single candidate and then slept. On a chain where the
    first four buyers decide the trade, a candidate detected just after a pass
    began waited out the whole cycle before anything looked at it, and a burst
    of ten launches took five seconds to start evaluating. The detection queue
    already blocks until something arrives, so the sleep bought nothing: it
    was latency with no corresponding saving.
    """

    def _desk(self):
        desk = SimpleNamespace(
            _running=True,
            global_config={},
            elogw_engine=SimpleNamespace(open_positions={}),
            _candidate_pipelines={},
            _background_tasks=set(),
            _redecide=asyncio.Queue(maxsize=4),
            _redecide_pending=set(),
            _redecision_drops=0,
            _candidate_drops=0,
            position_coverage=CoverageTracker("position"),
            managed=[],
        )
        desk.request_redecision = (
            lambda token: MemecoinQuantDesk.request_redecision(desk, token))

        async def manage_one(token, position):
            desk.managed.append(token)

        desk._manage_one_position = manage_one
        desk._position_intelligence = (
            lambda token, position: MemecoinQuantDesk._position_intelligence(
                desk, token, position))
        desk.rug_hazard = SimpleNamespace(get_hazard=lambda token: None)
        return desk

    def test_the_dispatcher_never_sleeps_between_candidates(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_candidate_dispatch_loop")
        sleeps = [node for node in ast.walk(loop)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "sleep"]
        # The only sleep is the error backoff, and it is inside an except.
        self.assertLessEqual(len(sleeps), 1)
        self.assertNotIn("wait_for", ast.dump(loop))

    def test_the_old_clocked_loop_is_gone(self):
        source = Path("src/main.py").read_text()
        self.assertNotIn("await self._process_new_tokens()\n                await self._manage_positions()",
                         source)
        tree = ast.parse(source)
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, ast.AsyncFunctionDef)}
        self.assertIn("_candidate_dispatch_loop", names)
        self.assertIn("_redecision_loop", names)
        # The sweep survives as a backstop, not as the decision path.
        self.assertIn("_safety_sweep_loop", names)

    def test_a_trade_on_an_open_position_requests_an_immediate_redecision(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        handler = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_on_pump_event")
        called = {node.func.attr for node in ast.walk(handler)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn("request_redecision", called)

    def test_requests_are_coalesced_per_token(self):
        """Forty trades in one slot describe one state, so one decision."""
        desk = self._desk()
        desk.elogw_engine.open_positions["mint"] = {"size_tokens": 1}
        self.assertTrue(desk.request_redecision("mint"))
        for _ in range(39):
            self.assertFalse(desk.request_redecision("mint"))
        self.assertEqual(desk._redecide.qsize(), 1)

    def test_a_token_we_do_not_hold_is_not_queued(self):
        desk = self._desk()
        self.assertFalse(desk.request_redecision("unheld"))
        self.assertEqual(desk._redecide.qsize(), 0)

    def test_the_queue_is_bounded_and_drops_are_counted(self):
        """A queue silently shedding work looks exactly like a quiet market."""
        desk = self._desk()
        for index in range(8):
            token = f"mint{index}"
            desk.elogw_engine.open_positions[token] = {"size_tokens": 1}
            desk.request_redecision(token)
        self.assertEqual(desk._redecide.qsize(), 4)
        self.assertEqual(desk._redecision_drops, 4)
        # A dropped request leaves no pending marker behind, so the token can
        # be requested again rather than being wedged out forever.
        self.assertEqual(len(desk._redecide_pending), 4)

    async def test_the_worker_redecides_and_clears_the_coalescing_marker(self):
        desk = self._desk()
        desk.elogw_engine.open_positions["mint"] = {"size_tokens": 1}
        desk.request_redecision("mint")
        desk._running = True

        worker = asyncio.create_task(MemecoinQuantDesk._redecision_loop(desk))
        for _ in range(200):
            if desk.managed:
                break
            await asyncio.sleep(0.001)
        desk._running = False
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        self.assertEqual(desk.managed, ["mint"])
        self.assertNotIn("mint", desk._redecide_pending)
        # And it can be requested again immediately afterwards.
        self.assertTrue(desk.request_redecision("mint"))

    async def test_a_position_closed_before_its_turn_is_skipped(self):
        desk = self._desk()
        desk.elogw_engine.open_positions["mint"] = {"size_tokens": 1}
        desk.request_redecision("mint")
        desk.elogw_engine.open_positions.pop("mint")
        desk._running = True

        worker = asyncio.create_task(MemecoinQuantDesk._redecision_loop(desk))
        await asyncio.sleep(0.02)
        desk._running = False
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        self.assertEqual(desk.managed, [])

    def test_dispatch_is_synchronous_so_it_cannot_block_the_queue(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        dispatch = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_dispatch_candidate")
        self.assertNotIsInstance(dispatch, ast.AsyncFunctionDef)
        self.assertEqual([], [node for node in ast.walk(dispatch)
                              if isinstance(node, ast.Await)])


class TestActionValueIsAuthoritative(unittest.TestCase):
    """One Q function owns ordinary actions, or three components disagree.

    Monster banking used to execute BEFORE the action-value engine was
    consulted, so the component that owns the objective was routinely told
    what had already been done. Worse, when Q explicitly chose HOLD, the
    ratchet could subsequently sell and scale-in could subsequently add -- a
    position the objective had just decided to leave alone got traded anyway,
    by two components reasoning from different quantities.
    """

    def test_the_action_set_covers_entry_as_well_as_exit(self):
        values = {action.value for action in Action}
        self.assertIn("ignore", values)
        self.assertIn("probe", values)
        self.assertTrue(Action.IGNORE.is_entry and Action.PROBE.is_entry)
        self.assertFalse(Action.HOLD.is_entry or Action.ADD.is_entry)

    def test_a_flat_book_ignores_rather_than_holds(self):
        """Recording which 'nothing' it was makes a rejection scoreable."""
        policy = ActionValuePolicy()
        weak = ActionState(
            held_fraction=0.0, current_multiple=1.0,
            forward_bins=((0.2, 1.0), (0.8, -0.6)),
            exit_capacity_ratio=1.0, escape_probability=0.9,
            probe_fraction=0.02, entry_cost=0.02, exit_cost=0.02)
        decision = policy.score(weak)
        self.assertIs(decision.action, Action.IGNORE)
        self.assertEqual(decision.q, 0.0)

    def test_a_worthwhile_flat_book_probes(self):
        policy = ActionValuePolicy()
        strong = ActionState(
            held_fraction=0.0, current_multiple=1.0,
            forward_bins=((0.45, 3.0), (0.55, -0.5)),
            exit_capacity_ratio=1.0, escape_probability=0.9,
            probe_fraction=0.02, entry_cost=0.02, exit_cost=0.02)
        decision = policy.score(strong)
        self.assertIs(decision.action, Action.PROBE)
        self.assertGreater(decision.q, 0.0)

    def test_probing_is_unavailable_once_a_position_is_open(self):
        policy = ActionValuePolicy()
        held = ActionState(
            held_fraction=0.3, current_multiple=2.0,
            forward_bins=((0.5, 2.0), (0.5, -0.5)),
            exit_capacity_ratio=1.0, escape_probability=0.9,
            probe_fraction=0.02, entry_cost=0.02, exit_cost=0.02)
        decision = policy.score(held)
        probe = next(score for score in decision.scores if score.action is Action.PROBE)
        ignore = next(score for score in decision.scores if score.action is Action.IGNORE)
        self.assertFalse(probe.feasible)
        self.assertFalse(ignore.feasible)

    def test_a_probe_is_not_credited_with_the_information_it_buys(self):
        """Crediting an unmeasured benefit justifies any speculative position."""
        policy = ActionValuePolicy()
        state = ActionState(
            held_fraction=0.0, current_multiple=1.0,
            forward_bins=((0.5, 1.0), (0.5, -1.0)),
            exit_capacity_ratio=1.0, escape_probability=1.0,
            probe_fraction=0.05, entry_cost=0.0, exit_cost=0.0)
        decision = policy.score(state)
        # A coin flip that can lose everything is never worth probing, however
        # much watching our own fill would teach us.
        self.assertIs(decision.action, Action.IGNORE)

    def test_a_priced_hold_ends_the_cycle(self):
        """The ratchet and scale-in used to run after a chosen HOLD."""
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        body = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_manage_one_position")
        text = ast.unparse(body)
        # The HOLD branch returns rather than falling through.
        self.assertIn("action-value priced this state and chose HOLD", text)
        hold_index = text.index("action-value priced this state and chose HOLD")
        ratchet_index = text.index("evaluate_exit")
        self.assertLess(hold_index, ratchet_index)

    def test_only_a_catastrophic_reading_bypasses_the_objective(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        body = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_manage_one_position")
        text = ast.unparse(body)
        # The bank branch records; only the emergency branch executes.
        self.assertIn("suppressed_monster_banks", text)
        bank_index = text.index("monster.action == 'bank'")
        emergency_index = text.index("monster.action == 'emergency_exit'")
        self.assertLess(emergency_index, bank_index)
        after_bank = text[bank_index:bank_index + 700]
        self.assertNotIn("_execute_exit", after_bank)

    def test_the_entry_path_prices_ignore_against_probe(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        evaluate = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.AsyncFunctionDef)
                        and node.name == "_evaluate_candidate")
        called = {node.func.attr for node in ast.walk(evaluate)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn("_score_entry", called)
        self.assertIn("entry_action", ast.unparse(evaluate))

    def test_an_unmeasurable_exit_capacity_blocks_entry_rather_than_assuming_liquid(self):
        """An entry we do not know how to leave is not an entry."""
        policy = ActionValuePolicy()
        blocked = ActionState(
            held_fraction=0.0, current_multiple=1.0,
            forward_bins=((0.5, 3.0), (0.5, -0.5)),
            exit_capacity_ratio=None, escape_probability=1.0,
            probe_fraction=0.02)
        self.assertEqual(policy.score(blocked).status, "DATA_BLOCKED")

    def test_authority_counters_reach_readiness(self):
        source = Path("src/main.py").read_text()
        for key in ("priced_holds", "unpriced_cycles", "suppressed_monster_banks"):
            self.assertIn(f'"{key}"', source)
        self.assertIn('"action_authority"', source)


class TestNativeRouteIsActuallyTaken(unittest.IsolatedAsyncioTestCase):
    """Preparing an instruction and then submitting someone else's transaction
    is the worst of both: it pays the construction cost, keeps the round trips,
    and looks finished.

    `execute_swap` built the native Pump instruction and then unconditionally
    called Jupiter for a quote, asked Jupiter to build a transaction, signed
    that, and submitted it. The native instruction was telemetry.
    """

    MINT = "So11111111111111111111111111111111111111112"
    OTHER = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    @staticmethod
    def _curve(creator=""):
        return BondingCurveState(
            virtual_token_reserves=1_000_000_000_000,
            virtual_sol_reserves=30_000_000_000,
            real_token_reserves=800_000_000_000,
            real_sol_reserves=0, token_total_supply=1_000_000_000_000_000,
            complete=False, creator=creator)

    def _engine(self, curve=None):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.pump_route = NativePumpRoute()
        engine.curve_state_provider = (lambda token: curve) if curve else None
        engine.pumpswap_route = None
        engine.pool_state_provider = None
        engine.pool_account_provider = None
        engine.tx_builder = SimpleNamespace(public_key=self.OTHER)
        engine.native_route_attempts = defaultdict(int)
        engine.native_compute_unit_limit = 400_000
        engine.stream_confirmations = 0
        engine.poll_confirmations = 0
        engine._signature_waiters = {}
        engine.reconcile_min_interval = 0.005
        engine.reconcile_max_interval = 0.05
        engine.landing_model = LandingModel()
        engine.last_bid = {}
        engine.dry_run = True
        engine.execution_history = deque(maxlen=16)
        engine.route_performance = defaultdict(
            lambda: {"total": 0, "landed": 0, "filled": 0, "failed": 0, "avg_latency": 0})
        engine.counterfactual_lab = SimpleNamespace(
            record_execution=lambda *args, **kwargs: None)
        engine.jupiter = SimpleNamespace(
            get_quote=lambda *args, **kwargs: self.fail("asked Jupiter for a quote"))
        return engine

    async def test_a_buildable_native_route_never_calls_jupiter(self):
        engine = self._engine(curve=self._curve(creator=self.MINT))
        result = await ExecutionEngine.execute_swap(
            engine, self.MINT, self.OTHER, 1_000_000_000, slippage_bps=100)
        self.assertTrue(result.success)
        self.assertIs(result.route_type, RouteType.PUMP_NATIVE)
        self.assertTrue(result.simulated)
        self.assertEqual(engine.native_route_attempts["simulated"], 1)

    async def test_an_unbuildable_native_route_falls_back_rather_than_failing(self):
        """Jupiter keeps the jobs it is better at, including being the fallback."""
        engine = self._engine(curve=None)
        engine.jupiter = SimpleNamespace(
            get_quote=self._quote_stub())
        result = await ExecutionEngine.execute_swap(
            engine, self.MINT, self.OTHER, 1_000_000_000, slippage_bps=100)
        self.assertTrue(result.success)
        self.assertIs(result.route_type, RouteType.JUPITER_V1)

    @staticmethod
    def _quote_stub():
        async def get_quote(*args, **kwargs):
            return SimpleNamespace(output_amount=123, route_type=RouteType.JUPITER_V1)
        return get_quote

    async def test_the_dry_run_gate_guards_the_native_path_too(self):
        """A path whose safety gate is only on the other branch will one day
        be the branch taken."""
        source = Path("src/execution/jupiter_jito.py").read_text()
        tree = ast.parse(source)
        native = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "_execute_native")
        text = ast.unparse(native)
        self.assertIn("self.dry_run", text)
        self.assertIn("ALLOW_LIVE_TRADING", text)

    async def test_hard_invariants_still_run_before_any_route(self):
        engine = self._engine(curve=self._curve(creator=self.MINT))
        for amount, slippage in ((0, 100), (100, 0), (100, 5_000)):
            result = await ExecutionEngine.execute_swap(
                engine, self.MINT, self.OTHER, amount, slippage_bps=slippage)
            self.assertIs(result.status, TransactionStatus.REJECTED)


class TestStreamedCurveStateIsBuildable(unittest.TestCase):
    """The native route blocked itself: streamed state carried no creator.

    A trade event carries virtual reserves only. Static facts about the curve
    -- who created it, how large the supply is -- arrive once, on the creation
    event, and never change. Not carrying them forward meant the route refused
    every trade for want of a creator it had already been told.
    """

    def test_the_creation_event_records_what_trade_events_lack(self):
        source = Path("src/main.py").read_text()
        self.assertIn("_curve_static[token]", source)
        tree = ast.parse(source)
        handler = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_on_pump_event")
        text = ast.unparse(handler)
        # The static record is written on creation and read on every trade.
        self.assertIn("_curve_static", text)
        self.assertIn('static.get(\'creator\'', text.replace('"', "'"))

    def test_an_account_update_replaces_rather_than_merges(self):
        """Mixing a measured field into a reconstructed record produces a row
        that is neither, and nothing downstream can tell which parts to trust."""
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        ingest = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "ingest_curve_account")
        text = ast.unparse(ingest)
        self.assertIn("parse_bonding_curve", text)
        self.assertIn("_latest_curve_state[token] = state", text)
        self.assertIn("request_redecision", text)

    def test_static_facts_are_pruned_against_the_hot_state(self):
        desk = SimpleNamespace(
            _curve_static={"live": {}, "stale": {}},
            hot_state=SimpleNamespace(active_tokens={"live"}))
        dropped = MemecoinQuantDesk._prune_curve_static(desk)
        self.assertEqual(dropped, 1)
        self.assertEqual(set(desk._curve_static), {"live"})


class TestSourceMeshStreams(unittest.IsolatedAsyncioTestCase):
    """A gather barrier makes every source as slow as the slowest one.

    `collect()` waited for all polls before returning anything, so a source
    taking four seconds held the 5ms chat event behind it -- on every cycle,
    not just slow ones. The docstring even claimed completion order, which
    gather does not do.
    """

    class _Fake(EventSource):
        def __init__(self, source_id, events, delay=0.0, interval=0.01):
            super().__init__(source_id, SourceClass.CHAT,
                             poll_interval_seconds=interval)
            self._events = list(events)
            self._delay = delay
            self.polls = 0

        async def poll(self):
            self.polls += 1
            if self._delay:
                await asyncio.sleep(self._delay)
            return [self._events.pop(0)] if self._events else []

    @staticmethod
    def _event(source_id, text, at=100.0):
        return Event(source_id=source_id, source_class=SourceClass.CHAT,
                     source_at=at, observed_at=at + 0.1, text=text,
                     token_addresses=("mint",))

    async def test_a_fast_source_is_not_held_behind_a_slow_one(self):
        fast = self._Fake("fast", [self._event("fast", "quick")], delay=0.0)
        slow = self._Fake("slow", [self._event("slow", "eventually")], delay=5.0)
        mesh = SourceMesh([fast, slow])
        await mesh.start()
        try:
            event = await asyncio.wait_for(mesh.next_event(), timeout=1.0)
        finally:
            await mesh.stop()
        # Under the old barrier this took five seconds.
        self.assertEqual(event.source_id, "fast")

    async def test_each_source_keeps_its_own_cadence(self):
        """A chat channel and a daily regulatory feed do not share a clock."""
        brisk = self._Fake("brisk", [], interval=0.01)
        languid = self._Fake("languid", [], interval=5.0)
        mesh = SourceMesh([brisk, languid])
        await mesh.start()
        await asyncio.sleep(0.1)
        await mesh.stop()
        self.assertGreater(brisk.polls, 2)
        self.assertLessEqual(languid.polls, 1)

    async def test_a_repeat_is_suppressed_but_its_source_is_recorded(self):
        """Dropping the repeat would throw away the lead-lag evidence."""
        first = self._Fake("first", [self._event("first", "same")])
        second = self._Fake("second", [self._event("second", "same")])
        mesh = SourceMesh([first, second])
        await mesh.start()
        await asyncio.sleep(0.1)
        await mesh.stop()
        events = mesh.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(len(mesh.repeaters_of(events[0].content_hash)), 2)

    async def test_the_fan_in_is_bounded_and_drops_the_oldest(self):
        mesh = SourceMesh([], max_queue=3)
        for index in range(6):
            mesh._publish(self._event("s", f"text{index}"), 100.0)
        self.assertEqual(mesh.pending, 3)
        self.assertEqual(mesh.dropped, 3)
        # The newest survive, because they are what a decision might still need.
        texts = [event.text for event in mesh.drain()]
        self.assertEqual(texts, ["text3", "text4", "text5"])

    async def test_a_dead_source_does_not_stop_the_others(self):
        class _Broken(EventSource):
            def __init__(self):
                super().__init__("broken", SourceClass.CHAT, poll_interval_seconds=0.01)

            async def poll(self):
                raise RuntimeError("upstream is down")

        alive = self._Fake("alive", [self._event("alive", "still here")])
        mesh = SourceMesh([_Broken(), alive])
        await mesh.start()
        try:
            event = await asyncio.wait_for(mesh.next_event(), timeout=1.0)
        finally:
            await mesh.stop()
        self.assertEqual(event.source_id, "alive")

    async def test_health_reports_that_it_is_streaming(self):
        mesh = SourceMesh([self._Fake("a", [])])
        await mesh.start()
        health = mesh.health()
        await mesh.stop()
        json.dumps(health)
        self.assertTrue(health["streaming"])
        self.assertEqual(health["producers"], 1)

    def test_the_runtime_consumes_the_mesh(self):
        """The architecture existed and the runtime never called it."""
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        names = {node.name for node in ast.walk(tree)
                 if isinstance(node, ast.AsyncFunctionDef)}
        self.assertIn("_source_consumer_loop", names)
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_source_consumer_loop")
        text = ast.unparse(loop)
        self.assertIn("source_mesh.start", text)
        self.assertIn("next_event", text)


class TestStreamFillReconciliation(unittest.IsolatedAsyncioTestCase):
    """A fixed 500ms poll finds a 400ms fill somewhere between 100 and 500ms late.

    Every time, on the one path where a position open and unaccounted for is
    the expensive kind.
    """

    def _engine(self, statuses=None):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine._signature_waiters = {}
        engine.stream_confirmations = 0
        engine.poll_confirmations = 0
        engine.reconcile_min_interval = 0.01
        engine.reconcile_max_interval = 0.05
        engine.confirmation_timeout = 1.0
        calls = []

        async def request(method, params):
            calls.append(method)
            return (statuses or {}).get(method)

        engine.rpc = SimpleNamespace(request=request)
        engine.rpc_calls = calls
        return engine

    async def test_the_stream_resolves_a_landing_without_any_polling(self):
        engine = self._engine()
        deadline = time.monotonic() + 1.0
        task = asyncio.create_task(
            ExecutionEngine._await_landing(engine, "sig", deadline))
        for _ in range(200):
            if "sig" in engine._signature_waiters:
                break
            await asyncio.sleep(0.001)
        self.assertTrue(ExecutionEngine.observe_signature(engine, "sig", 42))
        self.assertTrue(await task)
        self.assertEqual(engine.stream_confirmations, 1)
        self.assertEqual(engine.poll_confirmations, 0)

    async def test_the_poll_is_the_backstop_when_the_stream_misses_it(self):
        engine = self._engine(statuses={"getSignatureStatuses": {
            "value": [{"confirmationStatus": "confirmed"}]}})
        landed = await ExecutionEngine._await_landing(
            engine, "sig", time.monotonic() + 1.0)
        self.assertTrue(landed)
        self.assertEqual(engine.poll_confirmations, 1)
        self.assertEqual(engine.stream_confirmations, 0)

    async def test_a_missing_status_endpoint_falls_back_rather_than_denying(self):
        """Otherwise a node that does not serve statuses looks like a lost fill."""
        engine = self._engine(statuses={"getTransaction": {"slot": 7}})
        self.assertTrue(await ExecutionEngine._signature_landed(engine, "sig"))
        self.assertIn("getTransaction", engine.rpc_calls)

    async def test_an_unlanded_signature_times_out_rather_than_hanging(self):
        engine = self._engine()
        engine.confirmation_timeout = 0.05
        landed = await ExecutionEngine._await_landing(
            engine, "sig", time.monotonic() + 0.05)
        self.assertFalse(landed)
        # And the waiter is cleaned up, so a long run does not accumulate them.
        self.assertEqual(engine._signature_waiters, {})

    async def test_observing_an_unknown_signature_is_harmless(self):
        engine = self._engine()
        self.assertFalse(ExecutionEngine.observe_signature(engine, "someone-elses"))
        self.assertEqual(engine.stream_confirmations, 0)

    def test_the_decode_path_reports_our_signatures(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        handler = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_on_pump_event")
        self.assertIn("observe_signature", ast.unparse(handler))


class TestLandingModel(unittest.TestCase):
    """Tip selection was a lookup table consulted backwards.

    The optimiser kept a fee history, filtered it to fees with an 80% landing
    rate, and then took the CHEAPEST of them -- which is the wrong end: the
    cheapest fee clearing 80% is the one closest to failing.
    """

    def _fitted(self, curve=((0, 0.05), (100_000, 0.5), (1_000_000, 0.95))):
        model = LandingModel(min_bucket_attempts=10)
        for bid, rate in curve:
            landed = int(round(rate * 40))
            for index in range(40):
                model.record(Attempt(bid_lamports=bid, landed=index < landed,
                                     congestion=0.5, route="jito"))
        return model

    def test_it_refuses_before_it_has_a_curve(self):
        """A landing curve fitted to four attempts recommends whatever they paid."""
        model = LandingModel()
        estimate = model.probability(500_000)
        self.assertEqual(estimate.status, "DATA_BLOCKED")
        recommendation = model.recommend(1_000, 150)
        self.assertEqual(recommendation.status, "DATA_BLOCKED")
        self.assertTrue(recommendation.fallback)

    def test_failures_are_evidence_too(self):
        """A model fed only successes learns that everything lands."""
        model = LandingModel(min_bucket_attempts=4)
        for _ in range(10):
            model.record(Attempt(bid_lamports=0, landed=False))
        estimate = model.probability(0)
        self.assertEqual(estimate.status, "OK")
        self.assertEqual(estimate.probability, 0.0)

    def test_the_curve_rises_with_the_bid(self):
        model = self._fitted()
        low = model.probability(0, 0.5).probability
        mid = model.probability(100_000, 0.5).probability
        high = model.probability(1_000_000, 0.5).probability
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_a_large_edge_bids_up_and_a_worthless_one_does_not_bid(self):
        model = self._fitted()
        big = model.recommend(10_000.0, 150.0, congestion=0.5)
        self.assertEqual(big.status, "OK")
        self.assertGreater(big.bid_lamports, 0)
        # At an edge smaller than the bid itself, not bidding is the answer.
        tiny = model.recommend(0.0001, 150.0, congestion=0.5)
        self.assertIn(tiny.status, {"OK", "REJECTED"})
        self.assertEqual(tiny.bid_lamports, 0)

    def test_no_edge_is_rejected_rather_than_bid_on(self):
        model = self._fitted()
        self.assertEqual(model.recommend(0.0, 150.0).status, "REJECTED")
        self.assertEqual(model.recommend(100.0, 0.0).status, "REJECTED")

    def test_it_falls_back_from_conditioned_to_pooled_before_refusing(self):
        """'Never traded at this bid in a contested slot' is a narrow gap."""
        model = LandingModel(min_bucket_attempts=10)
        for index in range(20):
            model.record(Attempt(bid_lamports=500_000, landed=index < 15,
                                 congestion=0.1))
        # Asked about a congestion bucket with no samples of its own.
        estimate = model.probability(500_000, congestion=0.9)
        self.assertEqual(estimate.status, "OK")
        self.assertEqual(estimate.congestion, "pooled")

    def test_buckets_are_geometric_because_bids_respond_to_magnitude(self):
        self.assertEqual(bid_bucket(0), 0)
        self.assertEqual(bid_bucket(99_999), 50_000)
        self.assertEqual(bid_bucket(100_000), 100_000)
        self.assertEqual(bid_bucket(10 ** 9), BID_BUCKETS[-1])
        ratios = [BID_BUCKETS[i + 1] / BID_BUCKETS[i]
                  for i in range(1, len(BID_BUCKETS) - 1)]
        self.assertTrue(all(ratio >= 2 for ratio in ratios))

    def test_congestion_is_bucketed_and_unknown_is_its_own_bucket(self):
        self.assertEqual(congestion_bucket(None), "unknown")
        self.assertEqual(congestion_bucket(0.1), "calm")
        self.assertEqual(congestion_bucket(0.9), "contested")

    def test_the_window_is_bounded_and_counts_stay_consistent(self):
        model = LandingModel(capacity=50, min_bucket_attempts=5)
        for index in range(500):
            model.record(Attempt(bid_lamports=100_000, landed=index % 2 == 0))
        report = model.report()
        json.dumps(report)
        self.assertEqual(report["attempts"], 50)
        total = sum(row["attempts"] for row in report["curve"])
        self.assertEqual(total, 50)

    def test_the_engine_says_whether_a_bid_was_measured_or_guessed(self):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine.landing_model = LandingModel()
        guessed = ExecutionEngine.choose_bid(engine, 1_000.0, 150.0, 250_000)
        self.assertFalse(guessed["measured"])
        self.assertEqual(guessed["lamports"], 250_000)

        engine.landing_model = self._fitted()
        measured = ExecutionEngine.choose_bid(engine, 10_000.0, 150.0, 250_000,
                                              congestion=0.5)
        self.assertTrue(measured["measured"])
        self.assertGreater(measured["lamports"], 0)



class TestCapacitySurvivesGraduation(unittest.TestCase):
    """An absent answer is not a cautious one.

    Exit capacity read only the bonding curve, so the moment a coin migrated
    every position in it became permanently unmeasurable -- and the exit
    policy cannot tell DATA_BLOCKED-because-we-stopped-looking from
    DATA_BLOCKED-because-nothing-is-liquid.
    """

    TOKEN = "So11111111111111111111111111111111111111112"
    POOL = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def _desk(self, pool=None):
        return SimpleNamespace(
            _latest_curve_state={}, global_config={},
            _latest_pool_state={self.TOKEN: pool} if pool else {})

    def _pool(self, **overrides):
        args = dict(pool=self.POOL, base_mint=self.TOKEN, quote_mint=self.TOKEN,
                    base_reserves=1_000_000_000_000, quote_reserves=100_000_000_000,
                    total_fee_bps=30, updated_at=time.time(), source="test")
        args.update(overrides)
        return PumpSwapPoolState(**args)

    def test_a_graduated_position_is_still_measured(self):
        desk = self._desk(self._pool())
        position = {"size_tokens": 1_000_000}
        status, ratio = MemecoinQuantDesk._exit_capacity(desk, self.TOKEN, position)
        self.assertEqual(status, "OK")
        self.assertGreater(ratio, 0.0)
        self.assertIn("tradeability", position)

    def test_a_position_too_large_for_the_pool_is_measured_as_such(self):
        desk = self._desk(self._pool())
        small = MemecoinQuantDesk._exit_capacity(desk, self.TOKEN, {"size_tokens": 1_000})[1]
        huge = MemecoinQuantDesk._exit_capacity(
            desk, self.TOKEN, {"size_tokens": 900_000_000_000})[1]
        self.assertEqual(small, 1.0)
        self.assertLess(huge, small)

    def test_no_pool_and_no_curve_is_still_blocked_never_one(self):
        status, ratio = MemecoinQuantDesk._exit_capacity(
            self._desk(), self.TOKEN, {"size_tokens": 1_000})
        self.assertEqual(status, "DATA_BLOCKED_NO_CURVE_STATE")
        self.assertEqual(ratio, 0.0)

    def test_a_pool_whose_fee_was_never_observed_does_not_answer(self):
        """An unpriceable pool is not a measured capacity."""
        status, _ratio = MemecoinQuantDesk._exit_capacity(
            self._desk(self._pool(total_fee_bps=None)), self.TOKEN, {"size_tokens": 1_000})
        self.assertEqual(status, "DATA_BLOCKED_NO_CURVE_STATE")

class TestLocalLiquidity(unittest.TestCase):
    """A T0 decision paid a Jupiter round trip to learn what the curve states."""

    def _desk(self, state=None, sol_price=150.0):
        return SimpleNamespace(_latest_curve_state={"mint": state} if state else {},
                               _latest_pool_state={},
                               sol_price_usd=sol_price)

    def test_depth_comes_from_the_streamed_reserves(self):
        state = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000,
            virtual_sol_reserves=30_000_000_000,
            real_token_reserves=800_000_000_000,
            real_sol_reserves=12_000_000_000,
            token_total_supply=10 ** 15, complete=False, creator="c")
        desk = self._desk(state)
        # Real reserves preferred: 12 SOL at $150.
        self.assertAlmostEqual(
            MemecoinQuantDesk._local_liquidity(desk, "mint"), 12.0 * 150.0)

    def test_a_reconstruction_falls_back_to_the_virtual_reserve(self):
        state = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000,
            virtual_sol_reserves=30_000_000_000,
            real_token_reserves=0, real_sol_reserves=0,
            token_total_supply=0, complete=False, creator="c")
        desk = self._desk(state)
        self.assertAlmostEqual(
            MemecoinQuantDesk._local_liquidity(desk, "mint"), 30.0 * 150.0)

    def test_an_untradeable_or_unknown_curve_reports_zero(self):
        self.assertEqual(MemecoinQuantDesk._local_liquidity(self._desk(), "mint"), 0.0)
        done = BondingCurveState(
            virtual_token_reserves=0, virtual_sol_reserves=0,
            real_token_reserves=0, real_sol_reserves=0,
            token_total_supply=0, complete=True, creator="c")
        self.assertEqual(
            MemecoinQuantDesk._local_liquidity(self._desk(done), "mint"), 0.0)

    def test_the_local_read_is_tried_before_any_network_call(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        resolve = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_resolve_liquidity")
        text = ast.unparse(resolve)
        self.assertLess(text.index("_local_liquidity"), text.index("get_quote"))


class TestAccountPrewarming(unittest.TestCase):
    """Twenty-seven PDA derivations is ~2ms, and it was being paid at execution.

    For a given (mint, creator, wallet) those accounts are derivations of
    constants: they never change. Deriving them inside the window the whole
    system is optimised for is avoidable work, and detection is a moment when
    nothing is waiting.
    """

    MINT = "So11111111111111111111111111111111111111112"
    OTHER = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

    def test_warming_populates_both_instructions(self):
        route = NativePumpRoute()
        self.assertEqual(route.warm(self.MINT, self.MINT, self.OTHER), 2)
        # Warming twice caches nothing new.
        self.assertEqual(route.warm(self.MINT, self.MINT, self.OTHER), 0)

    def test_a_warmed_build_hits_the_cache(self):
        route = NativePumpRoute()
        route.warm(self.MINT, self.MINT, self.OTHER)
        prepared = route.build_buy(self.MINT, self.MINT, self.OTHER, 100, 200)
        self.assertEqual(prepared.status, "OK")
        self.assertEqual(route.warm_hits, 1)
        self.assertEqual(route.warm_misses, 0)

    def test_the_cached_list_is_identical_to_a_cold_derivation(self):
        cold = NativePumpRoute().build_buy(self.MINT, self.MINT, self.OTHER, 1, 2)
        warm_route = NativePumpRoute()
        warm_route.warm(self.MINT, self.MINT, self.OTHER)
        warm = warm_route.build_buy(self.MINT, self.MINT, self.OTHER, 1, 2)
        self.assertEqual([meta.to_dict() for meta in cold.accounts],
                         [meta.to_dict() for meta in warm.accounts])

    def test_the_key_separates_everything_that_changes_the_answer(self):
        """A wrong key builds a transaction against another token's accounts."""
        route = NativePumpRoute()
        first = route.build_buy(self.MINT, self.MINT, self.OTHER, 1, 2)
        second = route.build_buy(self.OTHER, self.MINT, self.OTHER, 1, 2)
        self.assertNotEqual([m.pubkey for m in first.accounts],
                            [m.pubkey for m in second.accounts])
        # Different user, different accounts.
        third = route.build_buy(self.MINT, self.MINT, self.MINT, 1, 2)
        self.assertNotEqual([m.pubkey for m in first.accounts],
                            [m.pubkey for m in third.accounts])
        # Different creator, different vault.
        fourth = route.build_buy(self.MINT, self.OTHER, self.OTHER, 1, 2)
        self.assertNotEqual([m.pubkey for m in first.accounts],
                            [m.pubkey for m in fourth.accounts])

    def test_buy_and_sell_do_not_share_a_cache_entry(self):
        route = NativePumpRoute()
        buy = route.build_buy(self.MINT, self.MINT, self.OTHER, 1, 2)
        sell = route.build_sell(self.MINT, self.MINT, self.OTHER, 1, 2)
        self.assertEqual(len(buy.accounts), 27)
        self.assertEqual(len(sell.accounts), 26)

    def test_amounts_are_not_part_of_the_key(self):
        """They are instruction data, not accounts."""
        route = NativePumpRoute()
        route.build_buy(self.MINT, self.MINT, self.OTHER, 1, 2)
        route.build_buy(self.MINT, self.MINT, self.OTHER, 999, 888)
        self.assertEqual(route.warm_hits, 1)

    def test_the_cache_is_bounded(self):
        route = NativePumpRoute(cache_size=4)
        for index in range(20):
            mint = str(Pubkey.from_bytes(bytes([index % 251] * 32)))
            route.warm(mint, self.MINT, self.OTHER)
        self.assertLessEqual(len(route._accounts), 4)

    def test_warming_is_measurably_cheaper_than_deriving(self):
        """Timed over several samples, because one sample is noise on a runner."""
        colds, warms = [], []
        for index in range(12):
            # A distinct mint each round, so every cold measurement is cold.
            mint = str(Pubkey.from_bytes(bytes([(index + 3) % 251] * 32)))
            route = NativePumpRoute()
            started = time.perf_counter()
            route.build_buy(mint, self.MINT, self.OTHER, 1, 2)
            colds.append(time.perf_counter() - started)
            started = time.perf_counter()
            route.build_buy(mint, self.MINT, self.OTHER, 1, 2)
            warms.append(time.perf_counter() - started)
        cold = sorted(colds)[len(colds) // 2]
        warm = sorted(warms)[len(warms) // 2]
        # An order of magnitude, asserted rather than the two seen in practice,
        # so a loaded runner does not fail a working cache.
        self.assertLess(warm * 10, cold, f"cold={cold:.6f}s warm={warm:.6f}s")

    def test_the_hit_rate_is_reported_so_cold_warming_is_visible(self):
        """A cache that never hits means warming is not reaching execution."""
        route = NativePumpRoute()
        self.assertIsNone(route.report()["prewarm"]["hit_rate"])
        route.warm(self.MINT, self.MINT, self.OTHER)
        route.build_buy(self.MINT, self.MINT, self.OTHER, 1, 2)
        self.assertEqual(route.report()["prewarm"]["hit_rate"], 1.0)

    def test_detection_warms_the_accounts(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        handler = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "_on_pump_event")
        self.assertIn("pump_route.warm", ast.unparse(handler))


class TestNoTestDependsOnOneMachine(unittest.TestCase):
    """A suite green on the author's box and red on CI is not a green suite.

    Every commit for the life of the workflow failed on one test that named an
    absolute path existing only on the machine it was written on. The product
    was correct throughout; the test was asserting against a directory the
    runner does not have.
    """

    def test_no_absolute_home_paths_are_hardcoded(self):
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for path in list(root.glob("tests/*.py")) + list(root.rglob("src/**/*.py")) \
                + list(root.glob("ops/*.py")) + list(root.glob("tools/*.py")):
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), 1):
                # Assembled rather than written out, so this check does not
                # match its own source and report itself.
                if any(("/" + part + "/") in line
                       for part in ("home" + "/user", "Users")):
                    offenders.append(f"{path.relative_to(root)}:{number}")
        self.assertEqual(offenders, [], f"machine-specific paths: {offenders}")

    def test_the_repo_root_is_derived_from_the_module(self):
        """So the pack reports on the repository it is actually inside."""
        from ops.audit_pack import changes_section

        section = changes_section(Path(__file__).resolve().parents[1], since_days=3650)
        self.assertEqual(section.status, "OK")


class TestForwardEvidence(unittest.TestCase):
    """Every audit said the forward proof was insufficient, and nothing counted.

    "Insufficient" and "not started" are the same sentence, and only one of
    them improves by waiting. This is the thing that makes the number go up.
    """

    @staticmethod
    def _row(**overrides):
        # NOT named `_outcome`: unittest assigns `self._outcome` a `_Outcome`
        # instance before each test, which would shadow this helper and make
        # every call raise "not callable" from somewhere that looks unrelated.
        base = dict(token="mint", entered=True, regime="bull",
                    realized_pnl_usd=10.0, equity_at_decision_usd=10_000.0)
        base.update(overrides)
        return Outcome(**base)

    def test_declines_are_recorded_too(self):
        """A ledger fed only on entries hides the trades we passed on."""
        ledger = ForwardEvidence()
        ledger.record(self._row(entered=False, token="passed"))
        ledger.record(self._row(entered=True, token="taken"))
        evidence = ledger.evidence()
        self.assertEqual(evidence.decisions, 2)
        self.assertEqual(evidence.launch_cohorts, 2)
        self.assertEqual(ledger.entered, 1)

    def test_cohorts_are_a_set_not_a_total(self):
        """5,000 decisions about one launch is one cohort."""
        ledger = ForwardEvidence()
        for _ in range(500):
            ledger.record(self._row(token="same"))
        self.assertEqual(ledger.evidence().decisions, 500)
        self.assertEqual(ledger.evidence().launch_cohorts, 1)

    def test_unknown_is_not_a_regime(self):
        """Otherwise a desk that never measured passes with one bucket."""
        ledger = ForwardEvidence()
        for _ in range(50):
            ledger.record(self._row(regime="unknown"))
        self.assertIsNone(ledger.evidence().regimes_covered)
        ledger.record(self._row(regime="bear"))
        self.assertEqual(ledger.evidence().regimes_covered, 1)

    def test_growth_is_logged_against_the_book_at_the_time(self):
        """Summing percentages over a changing book overstates a winning run."""
        ledger = ForwardEvidence()
        ledger.record(self._row(realized_pnl_usd=1_000.0,
                                    equity_at_decision_usd=10_000.0))
        self.assertAlmostEqual(ledger.net_log_growth, math.log(1.1))

    def test_a_trade_that_takes_the_book_is_catastrophic_not_negative_infinity(self):
        ledger = ForwardEvidence()
        ledger.record(self._row(realized_pnl_usd=-10_000.0,
                                    equity_at_decision_usd=10_000.0))
        self.assertEqual(ledger.catastrophic_failures, 1)
        self.assertTrue(math.isfinite(ledger.net_log_growth))

    def test_enrichment_is_none_against_a_zero_base_rate(self):
        """Reporting infinite enrichment is the most flattering possible number."""
        ledger = ForwardEvidence()
        for _ in range(30):
            ledger.record(self._row(max_multiple=1.1))
        self.assertIsNone(ledger.evidence().monster_enrichment)

    def test_unmeasured_fields_stay_none_rather_than_zero(self):
        """The gate treats unmeasured as failing; zero would claim a pass."""
        evidence = ForwardEvidence().evidence()
        self.assertIsNone(evidence.net_log_growth)
        self.assertIsNone(evidence.rug_loss_share)
        self.assertIsNone(evidence.execution_success)
        self.assertIsNone(evidence.monster_enrichment)

    def test_rug_share_is_of_losses_not_of_everything(self):
        ledger = ForwardEvidence()
        ledger.record(self._row(realized_pnl_usd=-100.0, rugged=True))
        ledger.record(self._row(realized_pnl_usd=-100.0, rugged=False))
        ledger.record(self._row(realized_pnl_usd=500.0))
        self.assertAlmostEqual(ledger.evidence().rug_loss_share, 0.5)

    def test_distance_reports_ratios_and_names_the_slowest_requirement(self):
        """A FAIL cannot distinguish a week away from a year away."""
        ledger = ForwardEvidence()
        for index in range(60):
            ledger.record(self._row(token=f"mint{index}",
                                        regime="bull" if index % 2 else "bear"))
        distance = ledger.distance()
        json.dumps(distance)
        self.assertEqual(distance["status"], "OK")
        self.assertFalse(distance["verdict"]["passed"])
        self.assertEqual(distance["slowest"], "decisions")
        self.assertLess(distance["progress"]["decisions"]["fraction"], 0.05)
        self.assertEqual(distance["progress"]["launch_cohorts"]["have"], 60)

    def test_it_survives_a_restart(self):
        """A counter that resets never reaches five thousand, however long it runs."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            first = ForwardEvidence(path)
            for index in range(40):
                first.record(self._row(token=f"mint{index}"))
            self.assertTrue(first.save())

            second = ForwardEvidence(path)
            self.assertEqual(second.decisions, 40)
            self.assertEqual(second.evidence().launch_cohorts, 40)
            self.assertAlmostEqual(second.net_log_growth, first.net_log_growth)

    def test_a_corrupt_ledger_does_not_crash_the_desk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text("{not json")
            ledger = ForwardEvidence(path)
            self.assertEqual(ledger.decisions, 0)

    def test_saving_is_atomic(self):
        """A half-written ledger parses to nothing and silently restarts the count."""
        source = Path("src/research/forward_evidence.py").read_text()
        self.assertIn("os.replace", source)
        self.assertIn("NamedTemporaryFile", source)

    def test_the_desk_feeds_it_from_trade_outcomes(self):
        source = Path("src/main.py").read_text()
        tree = ast.parse(source)
        recorder = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_record_ops_event")
        text = ast.unparse(recorder)
        self.assertIn("trade_outcomes", text)
        self.assertIn("_record_forward_evidence", text)

    def test_the_regime_label_is_unknown_without_measurements(self):
        desk = SimpleNamespace(global_research=None, global_config={})
        self.assertEqual(MemecoinQuantDesk.current_regime.fget(desk), "unknown")
        desk.global_research = SimpleNamespace(get_stats=lambda: {
            "meme_launch_rate_1h": 500, "sol_change_24h": 5.0})
        self.assertEqual(MemecoinQuantDesk.current_regime.fget(desk), "euphoria")
        desk.global_research = SimpleNamespace(get_stats=lambda: {
            "meme_launch_rate_1h": 10, "sol_change_24h": -5.0})
        self.assertEqual(MemecoinQuantDesk.current_regime.fget(desk), "bear")


def _fastpath():
    """The compiled extension, or None when it has not been built here.

    Skipped rather than failed when absent: the Python path is the reference
    implementation and must work without Rust, which is the whole reason the
    logic lives behind a feature gate.
    """
    try:
        import solana_fastpath  # noqa: F401
    except ImportError:
        return None
    return solana_fastpath


class TestRustPythonPolicyParity(unittest.TestCase):
    """Two implementations of one objective that disagree are worse than one.

    The disagreement does not show up as an error; it shows up as trades
    nobody can explain. So the Rust T0 core is a deliberate mirror of
    `action_value.py`, and this drives both from the same inputs and requires
    the same answer. Any change to either has to be made to both, and this is
    what says so.
    """

    LEVELS = [
        [0.45, 0.30, 0.20, 0.12, 0.05, 0.00, 0.00, 0.000],
        [0.90, 0.70, 0.50, 0.30, 0.10, 0.03, 0.01, 0.004],
        [0.20, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.000],
        [0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02, 0.010],
        [0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.000],
    ]

    def setUp(self):
        self.rust = _fastpath()
        if self.rust is None:
            self.skipTest("solana_fastpath extension not built in this environment")

    @staticmethod
    def _prediction(levels, p_rug_30s=0.0, p_rug_5m=0.0, feasible=0.0):
        return MultiHeadPrediction(
            "mint", "solana", 0,
            p_2x=levels[0], p_5x=levels[1], p_10x=levels[2], p_20x=levels[3],
            p_50x=levels[4], p_100x=levels[5], p_250x=levels[6], p_500x=levels[7],
            p_rug_30s=p_rug_30s, p_rug_5m=p_rug_5m,
            expected_feasible_multiple=feasible)

    def test_the_bins_agree_bin_for_bin(self):
        for levels in self.LEVELS:
            for rug in (0.0, 0.3):
                for feasible in (0.0, 3.0):
                    python = ElogwEngine.probability_bins(
                        self._prediction(levels, rug, rug, feasible))
                    native = self.rust.survival_bins(list(levels), rug, rug, feasible)
                    self.assertEqual(len(python), len(native),
                                     f"{levels} rug={rug} feasible={feasible}")
                    for (_, probability, gross), (rust_p, rust_g) in zip(python, native):
                        self.assertAlmostEqual(probability, rust_p, places=12)
                        self.assertAlmostEqual(gross, rust_g, places=12)

    def test_the_chosen_action_agrees_across_a_grid(self):
        checked = 0
        for levels in self.LEVELS:
            for held, multiple in ((0.0, 1.0), (0.3, 2.0), (0.5, 8.0)):
                for capacity, escape in ((1.0, 1.0), (0.5, 0.8), (0.2, 0.3)):
                    for rug in (0.0, 0.4):
                        probe = 0.02 if held == 0.0 else None
                        python_state = ActionState(
                            held_fraction=held, current_multiple=multiple,
                            forward_bins=tuple(
                                (probability, gross) for _, probability, gross
                                in ElogwEngine.probability_bins(
                                    self._prediction(levels, rug, rug))),
                            exit_cost=0.02, entry_cost=0.02,
                            exit_capacity_ratio=capacity, escape_probability=escape,
                            probe_fraction=probe)
                        python = ActionValuePolicy(min_edge=1e-4,
                                                   max_add_fraction=0.05).score(python_state)
                        native = self.rust.t0_decide(
                            0.1, 30_000_000_000, 1_000_000_000_000,
                            list(levels), rug, rug, 0.0,
                            held, multiple, 0.02, 0.02, capacity, escape,
                            None, None, None, probe,
                            1e-4, 0.05, False, 0.25, 0.05, 0.0005, 0.10, False)
                        self.assertEqual(python.status, "OK")
                        self.assertEqual(
                            python.action.value, native[0],
                            f"levels={levels} held={held} cap={capacity} "
                            f"escape={escape} rug={rug}")
                        self.assertAlmostEqual(python.q, native[1], places=9)
                        checked += 1
        self.assertGreater(checked, 50)

    def test_both_block_on_the_same_unmeasured_inputs(self):
        for capacity, escape in ((None, 0.9), (0.9, None), (None, None)):
            python = ActionValuePolicy().score(ActionState(
                held_fraction=0.3, current_multiple=2.0,
                forward_bins=((0.5, 2.0), (0.5, -0.5)),
                exit_capacity_ratio=capacity, escape_probability=escape))
            native = self.rust.t0_decide(
                0.1, 30_000_000_000, 1_000_000_000_000,
                [0.5, 0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0,
                0.3, 2.0, 0.02, 0.02, capacity, escape,
                None, None, None, None,
                1e-4, 0.05, False, 0.25, 0.05, 0.0005, 0.10, False)
            self.assertEqual(python.status, "DATA_BLOCKED")
            self.assertIsNotNone(native[4])

    def test_the_age_bands_agree(self):
        for age in (0.0, 0.05, 0.49, 0.5, 4.9, 5.0, 59.9, 60.0, 3_600.0):
            self.assertEqual(band_for(age), self.rust.t0_age_band(age), f"age={age}")

    def test_the_survival_levels_are_the_same_ladder(self):
        native = self.rust.survival_bins(
            [1.0] * 8, 0.0, 0.0, 0.0)
        # Every rung present, so a level added on one side and not the other
        # shows up here rather than as a quietly different distribution.
        grosses = sorted(gross for _, gross in native)
        expected = sorted([-0.98, -0.35] + [multiple - 1.0 for _, multiple in SURVIVAL_LEVELS])
        self.assertEqual(len(grosses), len(expected))
        for observed, want in zip(grosses, expected):
            self.assertAlmostEqual(observed, want, places=9)


class TestRustT0Safety(unittest.TestCase):
    """The safety layer is arithmetic about the account, not a model of it."""

    def setUp(self):
        self.rust = _fastpath()
        if self.rust is None:
            self.skipTest("solana_fastpath extension not built in this environment")

    def _decide(self, **overrides):
        args = dict(
            age_seconds=0.1, virtual_sol=30_000_000_000,
            virtual_token=1_000_000_000_000,
            levels=[0.45, 0.30, 0.20, 0.12, 0.05, 0.0, 0.0, 0.0],
            p_rug_30s=0.0, p_rug_5m=0.0, expected_feasible_multiple=0.0,
            held_fraction=0.0, current_multiple=1.0, exit_cost=0.02,
            entry_cost=0.02, exit_capacity_ratio=0.9, escape_probability=0.9,
            alternative_growth_per_second=None, expected_remaining_seconds=None,
            add_fraction=None, probe_fraction=0.02, min_edge=1e-4,
            max_add_fraction=0.05, live=False, max_position_fraction=0.25,
            max_single_commit_fraction=0.05, min_commit_fraction=0.0005,
            min_exit_capacity=0.10, live_unlocked=False)
        args.update(overrides)
        return self.rust.t0_decide(*[args[name] for name in (
            "age_seconds", "virtual_sol", "virtual_token", "levels", "p_rug_30s",
            "p_rug_5m", "expected_feasible_multiple", "held_fraction",
            "current_multiple", "exit_cost", "entry_cost", "exit_capacity_ratio",
            "escape_probability", "alternative_growth_per_second",
            "expected_remaining_seconds", "add_fraction", "probe_fraction",
            "min_edge", "max_add_fraction", "live", "max_position_fraction",
            "max_single_commit_fraction", "min_commit_fraction",
            "min_exit_capacity", "live_unlocked")])

    def test_a_locked_desk_decides_and_then_refuses(self):
        """Deciding and refusing are different states; only one is a bug."""
        action, _q, _band, allowed, blocked, refused, _commit, _scores = \
            self._decide(live=True)
        self.assertEqual(action, "probe")
        self.assertFalse(allowed)
        self.assertIsNone(blocked)
        self.assertEqual(refused, "live submission is locked")

    def test_the_acknowledgement_unlocks_it(self):
        _a, _q, _b, allowed, _bl, refused, _c, _s = self._decide(
            live=True, live_unlocked=True)
        self.assertTrue(allowed)
        self.assertIsNone(refused)

    def test_an_untradeable_curve_blocks_before_pricing(self):
        _a, _q, _b, allowed, blocked, _r, _c, scores = self._decide(
            virtual_sol=0, virtual_token=0)
        self.assertFalse(allowed)
        self.assertEqual(blocked, "curve is not tradeable")
        self.assertEqual(scores, [])

    def test_an_oversized_commit_is_refused_by_safety(self):
        action, _q, _b, allowed, _bl, refused, commit, _s = self._decide(
            probe_fraction=0.4)
        self.assertEqual(action, "probe")
        self.assertFalse(allowed)
        self.assertIn("single-action", refused)
        self.assertAlmostEqual(commit, 0.4)

    def test_dust_never_reaches_safety_because_the_objective_declines_first(self):
        """A better outcome than refusal: the objective already knows.

        A commit of 1e-9 cannot clear its own costs, so Q puts IGNORE ahead of
        PROBE and safety is never asked. Asserting a refusal here would be
        asserting that the policy chose something it should not have.
        """
        action, _q, _b, allowed, _bl, refused, commit, _s = self._decide(
            probe_fraction=1e-9)
        self.assertEqual(action, "ignore")
        self.assertTrue(allowed)
        self.assertIsNone(refused)
        self.assertEqual(commit, 0.0)

    def test_a_thin_book_is_refused_when_the_edge_would_otherwise_carry_it(self):
        """Capacity enters Q as well, so the floor only bites on strong views.

        On an ordinary distribution a thin book makes PROBE unattractive and
        the objective declines. The floor exists for the case the edge looks
        good enough to override that -- which is exactly when a hard limit
        should not be a matter of opinion.
        """
        strong = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        action, _q, _b, allowed, _bl, refused, _c, _s = self._decide(
            levels=strong, exit_capacity_ratio=0.09)
        self.assertEqual(action, "probe")
        self.assertFalse(allowed)
        self.assertIn("exit capacity", refused)

    def test_an_unmeasured_capacity_blocks_before_safety_is_consulted(self):
        _a, _q, _b, allowed, blocked, refused, _c, _s = self._decide(
            exit_capacity_ratio=None)
        self.assertFalse(allowed)
        self.assertIsNotNone(blocked)
        self.assertIsNone(refused)

    def test_selling_is_never_refused_even_live_and_locked(self):
        action, _q, _b, allowed, _bl, refused, _c, _s = self._decide(
            held_fraction=0.3, current_multiple=2.0, probe_fraction=None,
            levels=[0.2, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            p_rug_30s=0.5, p_rug_5m=0.6, live=True)
        self.assertIn(action, {"exit", "bank_75", "bank_50", "bank_25", "bank_10"})
        self.assertTrue(allowed, f"safety refused an exit: {refused}")
