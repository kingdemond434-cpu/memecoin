import asyncio
import base64
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from src.chains.rpc_manager import ChainConfig, ChainType, RPCEndpointConfig, RPCManager
from src.chains.yellowstone_grpc import (
    PumpFunMonitor, PumpSwapMonitor, RaydiumMonitor, SolanaRpcProgramStream, YellowstoneClient,
    create_combined_subscription,
)
from src.detection.rug_detector import TOKEN_2022_PROGRAM, TOKEN_PROGRAM, RugDetector
from src.execution.jupiter_jito import (
    ExecutionEngine, RouteType, SolanaTransactionBuilder, SwapQuote, TransactionStatus,
)
from src.strategies.information_graph import CounterfactualExecutionLab
from src.strategies.multihead_predictor import ElogwEngine, MultiHeadPrediction, MultiHeadPredictor
from src.strategies.public_coordination import PublicCoordinationMiner
from src.strategies.wallet_intelligence import WalletIntelligenceEngine
from src.research.dataset_builder import LaunchEpisode, PointInTimeDatasetBuilder, SnapshotTimepoint
from src.strategies.rug_hazard import ContinuousRugHazardModel


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
        events = []
        monitor = PumpFunMonitor(DummyYellowstone(), events.append)
        await monitor._on_transaction(fixture)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["side"], "sell")
        self.assertEqual(events[0]["token"], "FySyjuXTts9mTz2wjyuSXAz4bEBv6v5qxCTcLAMd4mVX")
        self.assertEqual(events[0]["wallet"], "V4SyF9Vv3EgzPyUEQQckN8b5eWSRndu9j2584WpnkCg")
        self.assertEqual(events[0]["slot"], 441417557)
        self.assertEqual(events[0]["signature"], fixture["signature"])

    async def test_raydium_v4_initialize2_layout(self):
        keys = [f"key{i}" for i in range(18)]
        accounts = list(range(18))
        payload = struct.pack("<BQQQ", 7, 1234, 50_000, 75_000)
        event = RaydiumMonitor._decode_v4(1, keys, accounts, payload, "sig", 99)
        self.assertEqual(event["pool"], "key4")
        self.assertEqual(event["mint_a"], "key8")
        self.assertEqual(event["mint_b"], "key9")
        self.assertEqual(event["creator"], "key17")

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
                                         p_10x=0.2, p_50x=0.05, p_rug_5m=0.1)
        bins = ElogwEngine.probability_bins(prediction)
        self.assertAlmostEqual(sum(probability for _, probability, _ in bins), 1.0)
        self.assertLessEqual(prediction.p_5x, prediction.p_2x)
        self.assertLessEqual(prediction.p_10x, prediction.p_5x)
        self.assertLessEqual(prediction.p_50x, prediction.p_10x)
        self.assertAlmostEqual(dict((name, probability) for name, probability, _ in bins)["50x_plus"], 0.045)

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
