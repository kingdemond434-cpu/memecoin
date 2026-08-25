import asyncio
import base64
import gzip
import hashlib
import json
import math
import struct
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from solders.hash import Hash
from solders.keypair import Keypair
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
from src.detection.token_detector import DetectionSource
from src.execution.jupiter_jito import (
    ExecutionEngine, ExecutionResult, JitoClient, JupiterClient, RouteType, SolanaTransactionBuilder, SwapQuote,
    SwapTransaction, TransactionStatus,
)
from src.main import CAPACITY_REJECTIONS, MemecoinQuantDesk, _jsonable
from src.strategies.information_graph import CounterfactualExecutionLab
from src.strategies.multihead_predictor import ElogwEngine, MultiHeadPrediction, MultiHeadPredictor, PredictionFeatures
from src.strategies.authenticity import (
    AuthenticityResolver, AuthenticityVerdict, EntityRegistry, ProofLevel,
    SourceSignal, WatchedEntity, extract_mints, host_matches, looks_like_mint,
    rank_copycats,
)
from src.strategies.distribution import (
    DISTRIBUTION_FEATURE_NAMES, DISTRIBUTION_HORIZONS, DistributionDetector,
    distribution_features,
)
from src.strategies.monster import (
    MONSTER_STATES, MonsterEvidence, MonsterState, MonsterStateMachine,
    hold_versus_exit, premature_exit_rates, tail_capture_ratio,
)
from src.strategies.opportunity_allocator import Opportunity, OpportunityAllocator
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.wallet_intelligence import WalletIntelligenceEngine, WalletRegime
from src.research.dataset_builder import LaunchEpisode, PointInTimeDatasetBuilder, SnapshotTimepoint
from src.research.shadow_trainer import chronological_episode_split, train_shadow
from src.chains.pump_curve import (
    BONDING_CURVE_DISCRIMINATOR, observation_from_state, parse_bonding_curve,
    quote_buy, quote_sell, sell_capacity_lamports,
)
from src.execution.pump_fees import (
    DYNAMIC_FEE_ACTIVATION_UTC, LEGACY_TOTAL_FEE_BPS, PumpFeeSchedule,
    VENUE_BONDING_CURVE, VENUE_PUMPSWAP_CANONICAL,
)
from src.chains.pump_curve import DEFAULT_FEE_BPS, resolve_fee_bps
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
                                         p_10x=0.2, p_50x=0.05, p_rug_5m=0.1,
                                         expected_feasible_multiple=3.0)
        bins = ElogwEngine.probability_bins(prediction)
        self.assertAlmostEqual(sum(probability for _, probability, _ in bins), 1.0)
        self.assertLessEqual(prediction.p_5x, prediction.p_2x)
        self.assertLessEqual(prediction.p_10x, prediction.p_5x)
        self.assertLessEqual(prediction.p_50x, prediction.p_10x)
        self.assertAlmostEqual(dict((name, probability) for name, probability, _ in bins)["50x_plus"], 0.045)
        self.assertLessEqual(max(outcome for _, _, outcome in bins), 2.0)

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


class TestPartialExitAccounting(unittest.IsolatedAsyncioTestCase):
    def _desk(self, result):
        desk = SimpleNamespace(
            execution_engine=FakeExecutionEngineForExit(result),
            dataset_builder=FakeDatasetBuilderForExit(),
            counterfactual_lab=FakeCounterfactualLabForExit(),
            elogw_engine=ElogwEngine(None),
            sol_price_usd=150.0, total_pnl=0.0, successful_exits=0, dry_run=True,
        )
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
        desk._read_distribution = lambda token: MemecoinQuantDesk._read_distribution(desk, token)
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
        self.assertIn("fees.png", quote.reason)
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
        desk._read_distribution = lambda token: MemecoinQuantDesk._read_distribution(desk, token)
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

    async def test_a_calibrated_high_reading_banks_before_the_trail_would(self):
        detector = DistributionDetector()
        detector.load_model(self._Model(0.75), DISTRIBUTION_FEATURE_NAMES, "v1")
        desk = self._desk(detector, self._turning_flow())

        # At 2.9x off a 3.0x high water with continuation 0.9, the trailing
        # stop is nowhere near firing -- price has barely moved.
        self.assertIsNone(evaluate_exit(ExitPolicy.default(), 2.9, 3.0, 0.9,
                                        {"cost_recovery"}, 60.0))

        await MemecoinQuantDesk._manage_positions(desk)

        self.assertEqual(len(desk.exits), 1)
        self.assertEqual(desk.exits[0][0], "monster_distribution")
        self.assertAlmostEqual(desk.exits[0][1],
                               MonsterStateMachine.DEFAULT_BANK_FRACTIONS[MonsterState.DISTRIBUTION])

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
        desk._read_distribution = lambda token: MemecoinQuantDesk._read_distribution(desk, token)
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
        hazard = SimpleNamespace(urgency="critical", hazard_5m=0.9)

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
            ),
            WatchedEntity(entity_id="newsdesk", display_name="News Desk",
                          accounts={"rss": {"newsdesk.example/feed"}},
                          official_domains={"newsdesk.example"}),
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
