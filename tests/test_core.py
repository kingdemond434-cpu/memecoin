import asyncio
import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from src.chains.rpc_manager import ChainConfig, ChainRegistry, ChainType, RPCEndpointConfig, RPCManager
from src.chains.yellowstone_grpc import (
    PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream, YellowstoneClient,
    create_combined_subscription, enrich_trade_balances,
)
from src.detection.rug_detector import TOKEN_2022_PROGRAM, TOKEN_PROGRAM, RugDetector
from src.execution.jupiter_jito import (
    ExecutionEngine, JupiterClient, RouteType, SolanaTransactionBuilder, SwapQuote, TransactionStatus,
)
from src.main import MemecoinQuantDesk
from src.strategies.information_graph import CounterfactualExecutionLab
from src.strategies.multihead_predictor import ElogwEngine, MultiHeadPrediction, MultiHeadPredictor, PredictionFeatures
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.wallet_intelligence import WalletIntelligenceEngine, WalletRegime
from src.research.dataset_builder import LaunchEpisode, PointInTimeDatasetBuilder, SnapshotTimepoint
from src.research.shadow_trainer import chronological_episode_split, train_shadow
from src.research.global_research_miner import GlobalResearchMiner, ResearchLead
from src.strategies.champion_challenger import ChampionChallengerFramework, HypothesisSpec, TrialResult
from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.genealogy_graph import GenealogyGraph, RelationshipType
from src.strategies.social_intelligence import SocialAccount, SocialIntelligenceEngine, SocialPlatform


def solana_chain():
    return ChainConfig(
        name="Solana Mainnet", chain_id="solana", chain_type=ChainType.SOLANA,
        rpc_endpoints=[RPCEndpointConfig("https://example.invalid")], explorer_api="", explorer_key="",
        native_token="SOL", decimals=9, block_time=0.4, factories={}, routers={}, base_tokens=[],
        min_liquidity_usd=2_000, max_tax=0, honeypot_check=False, programs={},
    )


class DummyYellowstone:
    def __init__(self):
        self.handlers = {}

    def on(self, event_type, handler):
        self.handlers[event_type] = handler


class TestSolanaParsing(unittest.IsolatedAsyncioTestCase):
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

    def test_insufficient_history_remains_explicitly_data_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = train_shadow(root / "episodes", root / "models", min_samples=10)
            self.assertEqual(report["status"], "DATA_BLOCKED")
            persisted = json.loads((root / "models" / "last_training_report.json").read_text())
            self.assertEqual(persisted["status"], "DATA_BLOCKED")


class TestApplicationStartup(unittest.IsolatedAsyncioTestCase):
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


class TestExecution(unittest.IsolatedAsyncioTestCase):
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


class FakeGenealogy:
    wallets = {}

    def get_deployer_profile(self, address):
        return None


class TestWalletAndCoordination(unittest.TestCase):
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
        self.assertIsNone(engine._classify_regime({"timestamp": 11, "multiple": 5.0}))
        self.assertEqual(engine._classify_regime({"regime": "post_migration"}), WalletRegime.POST_MIGRATION)

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


class TestRpcProtocol(unittest.TestCase):
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
        await stream.poll_once()
        self.assertEqual(events, [])
        await stream.poll_once()
        self.assertEqual(events, [{"signature": "new"}])


class TestShadowMarketObservation(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(outcome["realized_pnl"], 12.5)

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
