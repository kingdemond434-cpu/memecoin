import asyncio
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Any
from aiohttp import web

import yaml

from src.chains.rpc_manager import ChainRegistry, ChainConfig
from src.chains.yellowstone_grpc import YellowstoneClient, create_combined_subscription, PumpFunMonitor, RaydiumMonitor
from src.detection.token_detector import TokenDetectionEngine
from src.detection.rug_detector import RugDetector
from src.strategies.genealogy_graph import GenealogyGraph
from src.strategies.wallet_intelligence import WalletIntelligenceEngine
from src.strategies.social_intelligence import SocialIntelligenceEngine
from src.strategies.prelaunch_intent import PrelaunchIntentModel
from src.strategies.information_graph import InformationLeadGraph, CounterfactualExecutionLab, AdversarialAdaptationDetector
from src.strategies.rug_hazard import ContinuousRugHazardModel
from src.strategies.multihead_predictor import MultiHeadPredictor, ElogwEngine
from src.strategies.champion_challenger import ChampionChallengerFramework
from src.execution.jupiter_jito import JupiterClient, JitoClient, SolanaTransactionBuilder, ExecutionEngine, PriorityFeeOptimizer
from src.research.dataset_builder import PointInTimeDatasetBuilder
from nacl.signing import SigningKey

logger = logging.getLogger(__name__)


class MemecoinQuantDesk:
    def __init__(self, config_path: str = "config/chains.yaml"):
        self.config_path = config_path
        self.config = None
        self.chain_registry: Optional[ChainRegistry] = None
        
        self.yellowstone: Optional[YellowstoneClient] = None
        self.pump_monitor: Optional[PumpFunMonitor] = None
        self.raydium_monitor: Optional[RaydiumMonitor] = None
        
        self.detection_engine: Optional[TokenDetectionEngine] = None
        
        self.genealogy: Optional[GenealogyGraph] = None
        self.wallet_intel: Optional[WalletIntelligenceEngine] = None
        self.social_intel: Optional[SocialIntelligenceEngine] = None
        self.prelaunch: Optional[PrelaunchIntentModel] = None
        self.info_graph: Optional[InformationLeadGraph] = None
        self.counterfactual_lab: Optional[CounterfactualExecutionLab] = None
        self.adversarial: Optional[AdversarialAdaptationDetector] = None
        self.rug_hazard: Optional[ContinuousRugHazardModel] = None
        self.predictor: Optional[MultiHeadPredictor] = None
        self.elogw_engine: Optional[ElogwEngine] = None
        self.champion_challenger: Optional[ChampionChallengerFramework] = None
        self.dataset_builder: Optional[PointInTimeDatasetBuilder] = None
        
        self.jupiter: Optional[JupiterClient] = None
        self.jito: Optional[JitoClient] = None
        self.tx_builder: Optional[SolanaTransactionBuilder] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.fee_optimizer: Optional[PriorityFeeOptimizer] = None
        
        self.keypair: Optional[SigningKey] = None
        self._running = False
        self._main_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._web_app: Optional[web.Application] = None
        self._web_runner: Optional[web.AppRunner] = None
        
        self.start_time = time.time()
        self.trade_count = 0
        self.successful_trades = 0
        self.total_pnl = 0.0

    async def initialize(self):
        logger.info("Initializing Memecoin Quant Desk...")
        
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        global_config = self.config.get("global", {})
        self.dry_run = global_config.get("dry_run", True)
        
        await self._setup_keys()
        await self._setup_chains()
        await self._setup_yellowstone()
        await self._setup_detection()
        await self._setup_intelligence()
        await self._setup_prediction()
        await self._setup_execution()
        await self._setup_research()
        await self._setup_risk()
        
        logger.info("Initialization complete")

    async def _setup_keys(self):
        private_key_b64 = os.getenv("SOLANA_PRIVATE_KEY")
        if not private_key_b64:
            logger.warning("SOLANA_PRIVATE_KEY not set, generating ephemeral key for testing")
            self.keypair = SigningKey.generate()
        else:
            import base64
            private_key = base64.b64decode(private_key_b64)
            self.keypair = SigningKey(private_key[:32])
        
        logger.info(f"Wallet: {self.keypair.verify_key.encode().decode()}")

    async def _setup_chains(self):
        self.chain_registry = ChainRegistry(self.config_path)
        await self.chain_registry.start_all()
        
        solana_chain = self.chain_registry.get_chain("solana")
        solana_rpc = self.chain_registry.get_rpc("solana")
        
        if not solana_chain or not solana_rpc:
            raise RuntimeError("Solana chain configuration missing")
        
        self.solana_config = solana_chain
        self.solana_rpc = solana_rpc

    async def _setup_yellowstone(self):
        yellowstone_url = os.getenv("YELLOWSTONE_GRPC_URL", "https://yellowstone.helius-rpc.com")
        helius_key = os.getenv("HELIUS_API_KEY", "")
        
        self.yellowstone = YellowstoneClient(yellowstone_url, helius_key)
        await self.yellowstone.connect()
        
        subscription = create_combined_subscription()
        await self.yellowstone.subscribe(subscription)
        
        self.pump_monitor = PumpFunMonitor(self.yellowstone, self._on_pump_event)
        self.raydium_monitor = RaydiumMonitor(self.yellowstone, self._on_raydium_event)
        
        logger.info("Yellowstone gRPC connected")

    async def _setup_detection(self):
        self.detection_engine = TokenDetectionEngine(self.chain_registry)
        self.detection_engine.add_chain("solana", enable_mempool=False, enable_factory=True)
        await self.detection_engine.start()
        
        self.rug_detector = RugDetector(self.solana_config, self.solana_rpc)
        
        logger.info("Token detection engine started")

    async def _setup_intelligence(self):
        helius_key = os.getenv("HELIUS_API_KEY", "")
        api_keys = {
            "helius": helius_key,
            "x_bearer": os.getenv("X_BEARER_TOKEN", ""),
            "telegram": os.getenv("TELEGRAM_API_KEY", "")
        }
        
        self.genealogy = GenealogyGraph(self.solana_config, self.solana_rpc, helius_key)
        await self.genealogy.start()
        
        self.wallet_intel = WalletIntelligenceEngine(
            self.solana_config, self.solana_rpc, self.genealogy, helius_key
        )
        await self.wallet_intel.start()
        
        self.social_intel = SocialIntelligenceEngine(
            self.solana_config, self.solana_rpc, self.genealogy, self.wallet_intel, api_keys
        )
        self.social_intel.on_mention(self._on_social_mention)
        await self.social_intel.start()
        
        self.prelaunch = PrelaunchIntentModel(
            self.solana_config, self.solana_rpc, self.genealogy, self.wallet_intel, helius_key
        )
        await self.prelaunch.start()
        
        self.counterfactual_lab = CounterfactualExecutionLab()
        self.adversarial = AdversarialAdaptationDetector()
        self._setup_fakeability_scores()
        
        self.info_graph = InformationLeadGraph(
            self.solana_config, self.solana_rpc, self.genealogy,
            self.wallet_intel, self.social_intel, self.prelaunch
        )
        await self.info_graph.start()
        
        self.rug_hazard = ContinuousRugHazardModel(
            self.solana_config, self.solana_rpc, self.genealogy,
            self.wallet_intel, self.adversarial
        )
        await self.rug_hazard.start()
        
        logger.info("Intelligence engines started")

    def _setup_fakeability_scores(self):
        self.adversarial.set_fakeability("buyer_count", 0.9)
        self.adversarial.set_fakeability("fresh_wallet_volume", 0.8)
        self.adversarial.set_fakeability("social_engagement", 0.85)
        self.adversarial.set_fakeability("wallet_history_6m", 0.2)
        self.adversarial.set_fakeability("independent_funding", 0.3)
        self.adversarial.set_fakeability("creator_genealogy", 0.15)
        self.adversarial.set_fakeability("narrative_acceleration", 0.4)
        self.adversarial.set_fakeability("real_liquidity", 0.1)

    async def _setup_prediction(self):
        self.predictor = MultiHeadPredictor()
        self.predictor.initialize_models()
        
        self.elogw_engine = ElogwEngine(
            self.predictor,
            risk_aversion=1.0,
            max_position_pct=0.05,
            max_portfolio_risk=0.1,
            min_edge_bps=50
        )
        
        self.champion_challenger = ChampionChallengerFramework()
        await self.champion_challenger.start()
        
        logger.info("Prediction engine started")

    async def _setup_execution(self):
        self.jupiter = JupiterClient()
        self.jito = JitoClient()
        self.tx_builder = SolanaTransactionBuilder(self.solana_rpc, self.keypair)
        
        self.execution_engine = ExecutionEngine(
            self.solana_config, self.solana_rpc,
            self.jupiter, self.jito, self.tx_builder,
            self.counterfactual_lab
        )
        await self.execution_engine.start()
        
        self.fee_optimizer = PriorityFeeOptimizer()
        
        logger.info("Execution engine started")

    async def _setup_research(self):
        self.dataset_builder = PointInTimeDatasetBuilder(
            self.solana_config, self.solana_rpc, self.genealogy,
            self.wallet_intel, self.social_intel, self.prelaunch,
            self.info_graph, self.rug_hazard, self.champion_challenger
        )
        await self.dataset_builder.start()
        
        logger.info("Research dataset builder started")

    async def _setup_risk(self):
        logger.info("Risk systems initialized")

    async def start(self):
        self._running = True
        await self._setup_health_server()
        self._main_task = asyncio.create_task(self._main_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("Memecoin Quant Desk STARTED")

    async def stop(self):
        logger.info("Shutting down Memecoin Quant Desk...")
        self._running = False
        
        for task in [self._main_task, self._health_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await self._close_health_server()
        
        components = [
            self.dataset_builder, self.execution_engine, self.rug_hazard,
            self.info_graph, self.prelaunch, self.social_intel,
            self.wallet_intel, self.genealogy, self.detection_engine,
            self.champion_challenger, self.chain_registry
        ]
        
        for comp in components:
            if comp and hasattr(comp, 'stop'):
                try:
                    await comp.stop()
                except Exception as e:
                    logger.error(f"Error stopping {comp.__class__.__name__}: {e}")
        
        if self.yellowstone:
            await self.yellowstone.close()
        
        logger.info("Shutdown complete")

    async def _main_loop(self):
        while self._running:
            try:
                await self._process_new_tokens()
                await self._manage_positions()
                await self._update_intelligence()
            except Exception as e:
                logger.error(f"Main loop error: {e}")
            await asyncio.sleep(0.5)

    async def _process_new_tokens(self):
        try:
            candidate = await asyncio.wait_for(self.detection_engine.get_candidate(), timeout=0.1)
            await self._evaluate_candidate(candidate)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Candidate processing error: {e}")

    async def _evaluate_candidate(self, candidate):
        token = candidate.address
        chain = candidate.chain
        
        self.dataset_builder.start_episode(
            token, candidate.deployer or "", 
            candidate.factory or "", candidate.pair or "",
            candidate.base_token or ""
        )
        
        if chain != "solana":
            return
        
        risk_report = await self.rug_detector.analyze(
            token, candidate.pair, candidate.base_token
        )
        
        if risk_report.risk_level.value in ["critical", "honeypot", "rugged"]:
            logger.info(f"Token {token} rejected: {risk_report.risk_level.value}")
            return
        
        features = await self._build_prediction_features(candidate, risk_report)
        if not features:
            return
        
        prediction = self.predictor.predict(features)
        if not prediction:
            return
        
        should_trade, trade_info = self.elogw_engine.should_trade(
            prediction, 1.0, risk_report.liquidity_usd or 0,
            self.elogw_engine.portfolio_value
        )
        
        if not should_trade:
            return
        
        priority_fee = self.fee_optimizer.get_optimal_fee(
            trade_info.get("position_value_usd", 0), 0.5
        )
        jito_tip = self.fee_optimizer.get_jito_tip(
            trade_info.get("position_value_usd", 0), "MEDIUM"
        )
        
        result = await self.execution_engine.execute_swap(
            candidate.base_token or "So11111111111111111111111111111111111111112",
            token,
            int(trade_info["position_size_sol"] * 1e9),
            slippage_bps=100,
            priority_fee=priority_fee,
            jito_tip=jito_tip,
            use_jito=True
        )
        
        if result.success:
            self.trade_count += 1
            position = {
                "token": token,
                "entry_price": 1.0,
                "size_sol": trade_info["position_size_sol"],
                "size_tokens": result.output_amount,
                "entry_time": time.time(),
                "prediction": prediction.__dict__,
                "risk_report": risk_report.__dict__,
                "trade_info": trade_info
            }
            self.elogw_engine.update_position(token, position)
            self.dataset_builder.record_execution_attempt(token, result.__dict__)
            
            logger.info(f"BOUGHT {token}: {trade_info['position_size_sol']:.4f} SOL, sig: {result.signature}")

    async def _build_prediction_features(self, candidate, risk_report):
        from src.strategies.multihead_predictor import PredictionFeatures
        from src.strategies.wallet_intelligence import WalletRegime
        
        features = PredictionFeatures(
            token=candidate.address,
            chain=candidate.chain,
            timestamp=time.time()
        )
        
        dp = self.genealogy.get_deployer_profile(candidate.deployer or "")
        if dp:
            features.deployer_rug_rate = dp.rug_rate
            features.deployer_success_rate = dp.success_rate
            features.deployer_avg_multiple = dp.avg_max_multiple
        
        risk_assessment = self.genealogy.assess_launch_risk(
            candidate.deployer or "", 
            candidate.metadata.get("funding_wallets", []),
            candidate.metadata.get("initial_buyers", [])
        )
        features.deployer_cluster_risk = risk_assessment.get("risk_score", 0)
        
        social_signal = self.social_intel.get_token_social_signal(candidate.address)
        features.social_velocity = social_signal.get("avg_velocity", 0)
        features.social_acceleration = social_signal.get("acceleration", 0)
        features.social_credibility = social_signal.get("avg_credibility", 0)
        features.chain_before_social = social_signal.get("chain_before_pct", 0)
        features.cross_platform = social_signal.get("cross_platform", False)
        
        return features

    async def _manage_positions(self):
        for token, position in list(self.elogw_engine.open_positions.items()):
            hazard_state = self.rug_hazard.get_hazard(token)
            if hazard_state:
                should_exit, urgency, exit_pct = self.rug_hazard.should_exit(token, position)
                if should_exit:
                    await self._execute_exit(token, position, exit_pct, f"rug_hazard_{urgency}")
                    continue
            
            hold_time = time.time() - position["entry_time"]
            if hold_time > 3600:
                await self._execute_exit(token, position, 1.0, "time_stop")
                continue

    async def _execute_exit(self, token: str, position: Dict, exit_pct: float, reason: str):
        size = int(position["size_tokens"] * exit_pct)
        result = await self.execution_engine.execute_sell(token, size, slippage_bps=500, use_jito=True)
        
        if result.success:
            pnl = (result.output_amount / 1e6) * 0.001
            self.total_pnl += pnl
            if pnl > 0:
                self.successful_trades += 1
            
            self.elogw_engine.close_position(token)
            self.dataset_builder.record_execution_attempt(token, result.__dict__)
            
            logger.info(f"EXIT {token}: {exit_pct*100:.0f}%, reason: {reason}, PnL: {pnl:.4f}")

    async def _update_intelligence(self):
        pass

    async def _health_loop(self):
        while self._running:
            try:
                await self._log_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            await asyncio.sleep(60)

    async def _log_health(self):
        uptime = time.time() - self.start_time
        rpc_stats = self.chain_registry.get_all_stats() if self.chain_registry else {}
        
        logger.info(f"""
=== HEALTH CHECK ===
Uptime: {uptime/3600:.1f}h
Trades: {self.trade_count} (Wins: {self.successful_trades})
Total PnL: {self.total_pnl:.4f}
Win Rate: {self.successful_trades/max(self.trade_count,1)*100:.1f}%
Open Positions: {len(self.elogw_engine.open_positions) if self.elogw_engine else 0}
Portfolio: {self.elogw_engine.get_portfolio_state() if self.elogw_engine else {}}
RPC: {rpc_stats}
Wallet Intel: {self.wallet_intel.get_stats() if self.wallet_intel else {}}
Social Intel: {self.social_intel.get_stats() if self.social_intel else {}}
Prelaunch: {self.prelaunch.get_stats() if self.prelaunch else {}}
Info Graph: {self.info_graph.get_stats() if self.info_graph else {}}
Rug Hazard: {self.rug_hazard.get_stats() if self.rug_hazard else {}}
Execution: {self.execution_engine.get_route_stats() if self.execution_engine else {}}
Dataset: {self.dataset_builder.get_stats() if self.dataset_builder else {}}
Champions: {self.champion_challenger.get_stats() if self.champion_challenger else {}}
====================
        """)

    async def _on_pump_event(self, event: Dict):
        if event["type"] == "token_created":
            self.info_graph.record_event(
                event["token"], 
                self.info_graph.LeadEventType.DEPLOYER_ACTIVITY,
                event["creator"], "deployer", event["timestamp"]
            )
        elif event["type"] == "token_trade":
            if event["side"] == "buy":
                self.info_graph.record_event(
                    event["token"],
                    self.info_graph.LeadEventType.ELITE_WALLET_BUY,
                    event["wallet"], "wallet", event["timestamp"]
                )
            else:
                self.info_graph.record_event(
                    event["token"],
                    self.info_graph.LeadEventType.SMART_WALLET_EXIT,
                    event["wallet"], "wallet", event["timestamp"]
                )

    async def _on_raydium_event(self, event: Dict):
        pass

    async def _on_social_mention(self, signal: Dict):
        if signal.get("type") == "new_mention":
            self.info_graph.record_event(
                signal["token"],
                self.info_graph.LeadEventType.OBSCURE_X_MENTION,
                signal["account"], "social", signal["timestamp"]
            )

    async def _setup_health_server(self):
        self._web_app = web.Application()
        self._web_app.router.add_get('/health', self._health_endpoint)
        self._web_app.router.add_get('/metrics', self._metrics_endpoint)
        self._web_app.router.add_get('/status', self._status_endpoint)
        
        self._web_runner = web.AppRunner(self._web_app)
        await self._web_runner.setup()
        site = web.TCPSite(self._web_runner, '0.0.0.0', 8080)
        await site.start()
        logger.info("Health server started on port 8080")

    async def _health_endpoint(self, request):
        return web.json_response({
            "status": "healthy" if self._running else "stopping",
            "uptime_seconds": time.time() - self.start_time,
            "trades": self.trade_count,
            "win_rate": self.successful_trades / max(self.trade_count, 1),
            "open_positions": len(self.elogw_engine.open_positions) if self.elogw_engine else 0
        })

    async def _metrics_endpoint(self, request):
        portfolio = self.elogw_engine.get_portfolio_state() if self.elogw_engine else {}
        return web.json_response({
            "portfolio_value": portfolio.get("portfolio_value", 0),
            "daily_pnl": portfolio.get("daily_pnl", 0),
            "open_positions": portfolio.get("open_positions", 0),
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "successful_trades": self.successful_trades
        })

    async def _status_endpoint(self, request):
        return web.json_response({
            "wallet_intel": self.wallet_intel.get_stats() if self.wallet_intel else {},
            "social_intel": self.social_intel.get_stats() if self.social_intel else {},
            "prelaunch": self.prelaunch.get_stats() if self.prelaunch else {},
            "rug_hazard": self.rug_hazard.get_stats() if self.rug_hazard else {},
            "execution": self.execution_engine.get_route_stats() if self.execution_engine else {},
            "dataset": self.dataset_builder.get_stats() if self.dataset_builder else {},
            "champions": self.champion_challenger.get_stats() if self.champion_challenger else {},
            "rpc": self.chain_registry.get_all_stats() if self.chain_registry else {}
        })

    async def _close_health_server(self):
        if self._web_runner:
            await self._web_runner.cleanup()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    desk = MemecoinQuantDesk()
    
    try:
        await desk.initialize()
        await desk.start()
        
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await desk.stop()


if __name__ == "__main__":
    asyncio.run(main())