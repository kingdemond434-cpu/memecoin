import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json

import aiohttp
import networkx as nx
import numpy as np

from src.chains.rpc_manager import ChainConfig, RPCManager

logger = logging.getLogger(__name__)


class EntityType(Enum):
    WALLET = "wallet"
    DEPLOYER = "deployer"
    FUNDING_SOURCE = "funding_source"
    CASHOUT_DEST = "cashout_dest"
    CLUSTER = "cluster"


class RelationshipType(Enum):
    FUNDED = "funded"
    DEPLOYED = "deployed"
    CO_BOUGHT = "co_bought"
    CO_SOLD = "co_sold"
    SAME_CASHOUT = "same_cashout"
    TRANSFERRED = "transferred"
    CREATED_CLUSTER = "created_cluster"


@dataclass
class WalletProfile:
    address: str
    entity_type: EntityType
    first_seen: float
    last_seen: float
    tx_count: int = 0
    sol_balance: float = 0.0
    token_balances: Dict[str, float] = field(default_factory=dict)
    
    launches_participated: List[str] = field(default_factory=list)
    launches_deployed: List[str] = field(default_factory=list)
    
    win_rate_2x: float = 0.0
    win_rate_5x: float = 0.0
    win_rate_10x: float = 0.0
    rug_exposure: float = 0.0
    median_entry_pct: float = 50.0
    median_exit_pct: float = 50.0
    realized_pnl: float = 0.0
    holding_time_median: float = 0.0
    
    funding_sources: Set[str] = field(default_factory=set)
    cashout_destinations: Set[str] = field(default_factory=set)
    related_wallets: Set[str] = field(default_factory=set)
    cluster_id: Optional[str] = None
    
    is_smart_money: bool = False
    is_insider: bool = False
    is_rugged_deployer: bool = False
    trust_score: float = 0.5


@dataclass
class DeployerProfile:
    address: str
    wallet_profile: WalletProfile
    tokens_created: List[str] = field(default_factory=list)
    tokens_rugged: List[str] = field(default_factory=list)
    tokens_migrated: List[str] = field(default_factory=list)
    tokens_successful: List[str] = field(default_factory=list)
    
    avg_liquidity_usd: float = 0.0
    avg_tax_buy: float = 0.0
    avg_tax_sell: float = 0.0
    ownership_renounced_rate: float = 0.0
    liquidity_locked_rate: float = 0.0
    
    rug_rate: float = 0.0
    success_rate: float = 0.0
    avg_max_multiple: float = 0.0
    median_time_to_peak: float = 0.0
    
    funding_wallets: Set[str] = field(default_factory=set)
    operational_wallets: Set[str] = field(default_factory=set)


@dataclass
class ClusterProfile:
    cluster_id: str
    wallets: Set[str] = field(default_factory=set)
    deployers: Set[str] = field(default_factory=set)
    funding_sources: Set[str] = field(default_factory=set)
    cashout_destinations: Set[str] = field(default_factory=set)
    
    total_launches: int = 0
    rugged_launches: int = 0
    successful_launches: int = 0
    migrated_launches: int = 0
    
    avg_rug_rate: float = 0.0
    avg_success_rate: float = 0.0
    
    creation_pattern: str = "unknown"
    activity_level: str = "low"
    risk_level: str = "unknown"


class GenealogyGraph:
    def __init__(self, chain_config: ChainConfig, rpc: RPCManager, helius_key: str):
        self.chain_config = chain_config
        self.rpc = rpc
        self.helius_key = helius_key
        
        self.graph = nx.MultiDiGraph()
        self.wallets: Dict[str, WalletProfile] = {}
        self.deployers: Dict[str, DeployerProfile] = {}
        self.clusters: Dict[str, ClusterProfile] = {}
        
        self._wallet_to_cluster: Dict[str, str] = {}
        self._cluster_counter = 0
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._helius_base = "https://api.helius.xyz/v0"
        
        self._update_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing = False
        self.outcome_provider = None
        self.data_status: Dict[str, str] = {}

    async def start(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=50)
        )
        self._processing = True
        asyncio.create_task(self._process_updates())
        await self._load_historical_data()

    async def stop(self):
        self._processing = False
        if self._session:
            await self._session.close()

    async def _load_historical_data(self):
        self.data_status["historical_genealogy"] = (
            "DATA_BLOCKED: no versioned historical genealogy artifact; building from observed launches"
        )

    def set_outcome_provider(self, provider):
        self.outcome_provider = provider

    def record_token_creation(self, token: str, deployer: str, metadata: Optional[Dict] = None):
        update = {
            "type": "token_created",
            "token": token,
            "deployer": deployer,
            "timestamp": (metadata or {}).get("timestamp", time.time()),
            "funding_wallets": (metadata or {}).get("funding_wallets", []),
            "initial_buyers": (metadata or {}).get("initial_buyers", []),
        }
        try:
            self._update_queue.put_nowait(update)
        except asyncio.QueueFull:
            self.data_status["update_queue"] = "DATA_BLOCKED: genealogy update queue full"

    async def _process_updates(self):
        while self._processing:
            try:
                update = await asyncio.wait_for(self._update_queue.get(), timeout=1)
                await self._apply_update(update)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Graph update error: {e}")

    async def _apply_update(self, update: Dict):
        update_type = update.get("type")
        if update_type == "token_created":
            await self._process_token_creation(update)
        elif update_type == "wallet_activity":
            await self._process_wallet_activity(update)
        elif update_type == "transfer":
            await self._process_transfer(update)
        elif update_type == "trade":
            await self._process_trade(update)

    async def _process_token_creation(self, data: Dict):
        deployer = data.get("deployer")
        token = data.get("token")
        funding_wallets = data.get("funding_wallets", [])
        initial_buyers = data.get("initial_buyers", [])
        timestamp = data.get("timestamp", time.time())
        
        if deployer not in self.wallets:
            self.wallets[deployer] = WalletProfile(
                address=deployer,
                entity_type=EntityType.DEPLOYER,
                first_seen=timestamp,
                last_seen=timestamp
            )
        wallet = self.wallets[deployer]
        wallet.last_seen = timestamp
        wallet.launches_deployed.append(token)
        wallet.entity_type = EntityType.DEPLOYER
        
        if deployer not in self.deployers:
            self.deployers[deployer] = DeployerProfile(
                address=deployer,
                wallet_profile=wallet
            )
        deployer_profile = self.deployers[deployer]
        deployer_profile.tokens_created.append(token)
        
        for funder in funding_wallets:
            self._add_relationship(funder, deployer, RelationshipType.FUNDED, timestamp)
            if funder not in self.wallets:
                self.wallets[funder] = WalletProfile(
                    address=funder,
                    entity_type=EntityType.FUNDING_SOURCE,
                    first_seen=timestamp,
                    last_seen=timestamp
                )
            self.wallets[funder].last_seen = timestamp
            self.wallets[funder].entity_type = EntityType.FUNDING_SOURCE
            deployer_profile.funding_wallets.add(funder)
            self.wallets[deployer].funding_sources.add(funder)
        
        for buyer in initial_buyers:
            buyer_addr = buyer.get("address")
            if buyer_addr:
                self._add_relationship(deployer, buyer_addr, RelationshipType.CO_BOUGHT, timestamp)
                if buyer_addr not in self.wallets:
                    self.wallets[buyer_addr] = WalletProfile(
                        address=buyer_addr,
                        entity_type=EntityType.WALLET,
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                self.wallets[buyer_addr].last_seen = timestamp
                self.wallets[buyer_addr].launches_participated.append(token)

    async def _process_wallet_activity(self, data: Dict):
        wallet = data.get("wallet")
        activity = data.get("activity")
        timestamp = data.get("timestamp", time.time())
        
        if wallet not in self.wallets:
            self.wallets[wallet] = WalletProfile(
                address=wallet,
                entity_type=EntityType.WALLET,
                first_seen=timestamp,
                last_seen=timestamp
            )
        self.wallets[wallet].last_seen = timestamp
        self.wallets[wallet].tx_count += 1

    async def _process_transfer(self, data: Dict):
        from_addr = data.get("from")
        to_addr = data.get("to")
        amount = data.get("amount", 0)
        timestamp = data.get("timestamp", time.time())
        
        if from_addr and to_addr:
            self._add_relationship(from_addr, to_addr, RelationshipType.TRANSFERRED, timestamp, amount)
            
            for addr in [from_addr, to_addr]:
                if addr not in self.wallets:
                    self.wallets[addr] = WalletProfile(
                        address=addr,
                        entity_type=EntityType.WALLET,
                        first_seen=timestamp,
                        last_seen=timestamp
                    )
                self.wallets[addr].last_seen = timestamp

    async def _process_trade(self, data: Dict):
        wallet = data.get("wallet")
        token = data.get("token")
        side = data.get("side")
        amount = data.get("amount", 0)
        price = data.get("price", 0)
        timestamp = data.get("timestamp", time.time())
        
        if wallet not in self.wallets:
            self.wallets[wallet] = WalletProfile(
                address=wallet,
                entity_type=EntityType.WALLET,
                first_seen=timestamp,
                last_seen=timestamp
            )
        w = self.wallets[wallet]
        w.last_seen = timestamp
        w.tx_count += 1
        
        if token not in w.launches_participated:
            w.launches_participated.append(token)

    def _add_relationship(self, from_addr: str, to_addr: str, rel_type: RelationshipType, 
                          timestamp: float, weight: float = 1.0):
        self.graph.add_edge(from_addr, to_addr, 
                           type=rel_type.value, timestamp=timestamp, weight=weight)

    async def update_token_outcome(self, token: str, outcome: Dict):
        deployer = outcome.get("deployer")
        if deployer and deployer in self.deployers:
            dp = self.deployers[deployer]
            if outcome.get("rugged"):
                dp.tokens_rugged.append(token)
            elif outcome.get("migrated"):
                dp.tokens_migrated.append(token)
            elif outcome.get("max_multiple", 0) >= 5:
                dp.tokens_successful.append(token)
            
            self._recalculate_deployer_stats(deployer)
            await self._recalculate_wallet_stats(deployer)

    def _recalculate_deployer_stats(self, deployer: str):
        dp = self.deployers[deployer]
        total = len(dp.tokens_created)
        if total == 0:
            return
        dp.rug_rate = len(dp.tokens_rugged) / total
        dp.success_rate = len(dp.tokens_successful) / total
        
        if dp.wallet_profile:
            dp.wallet_profile.is_rugged_deployer = dp.rug_rate > 0.5

    async def _recalculate_wallet_stats(self, wallet: str):
        if wallet not in self.wallets:
            return
        w = self.wallets[wallet]
        
        if w.launches_participated:
            outcomes = await self._get_launch_outcomes(w.launches_participated)
            if outcomes:
                w.win_rate_2x = sum(1 for o in outcomes if o.get("max_multiple", 0) >= 2) / len(outcomes)
                w.win_rate_5x = sum(1 for o in outcomes if o.get("max_multiple", 0) >= 5) / len(outcomes)
                w.win_rate_10x = sum(1 for o in outcomes if o.get("max_multiple", 0) >= 10) / len(outcomes)
                w.rug_exposure = sum(1 for o in outcomes if o.get("rugged")) / len(outcomes)
                w.realized_pnl = sum(o.get("realized_pnl", 0) for o in outcomes)
                
                entry_pcts = [o.get("entry_pct", 50) for o in outcomes if "entry_pct" in o]
                exit_pcts = [o.get("exit_pct", 50) for o in outcomes if "exit_pct" in o]
                if entry_pcts:
                    w.median_entry_pct = np.median(entry_pcts)
                if exit_pcts:
                    w.median_exit_pct = np.median(exit_pcts)
                
                w.is_smart_money = (w.win_rate_5x > 0.3 and w.rug_exposure < 0.3 and w.realized_pnl > 0)
                w.trust_score = self._calculate_trust_score(w)

    def _calculate_trust_score(self, w: WalletProfile) -> float:
        score = 0.5
        score += (w.win_rate_5x - 0.2) * 0.5
        score -= w.rug_exposure * 0.3
        score += min(w.realized_pnl / 10000, 0.2)
        score -= max(0, w.median_entry_pct - 30) * 0.002
        return max(0, min(1, score))

    async def _get_launch_outcomes(self, tokens: List[str]) -> List[Dict]:
        if not self.outcome_provider:
            self.data_status["launch_outcomes"] = "DATA_BLOCKED: PIT outcome provider unavailable"
            return []
        outcomes = []
        for token in tokens:
            outcome = self.outcome_provider(token)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if outcome and outcome.get("status") != "DATA_BLOCKED":
                outcomes.append(outcome)
        self.data_status["launch_outcomes"] = "OK" if outcomes else "DATA_BLOCKED: no finalized outcomes"
        return outcomes

    def get_wallet_profile(self, address: str) -> Optional[WalletProfile]:
        return self.wallets.get(address)

    def get_deployer_profile(self, address: str) -> Optional[DeployerProfile]:
        return self.deployers.get(address)

    def find_cluster(self, address: str) -> Optional[ClusterProfile]:
        cluster_id = self._wallet_to_cluster.get(address)
        if cluster_id:
            return self.clusters.get(cluster_id)
        return None

    async def build_clusters(self, min_connections: int = 3):
        # Rebuild deterministically from current evidence. Wallet profiles may
        # exist before their first graph edge, and graph nodes may exist before
        # their wallet profile is hydrated.
        self.clusters.clear()
        self._wallet_to_cluster.clear()
        self._cluster_counter = 0
        for profile in self.wallets.values():
            profile.cluster_id = None
        visited = set()
        for wallet in list(self.wallets):
            if wallet in visited:
                continue
            cluster_wallets = self._find_connected_component(wallet, min_connections)
            visited.update(cluster_wallets)
            if len(cluster_wallets) >= min_connections:
                cluster_id = f"cluster_{self._cluster_counter}"
                self._cluster_counter += 1
                cluster = ClusterProfile(cluster_id=cluster_id, wallets=set(cluster_wallets))
                
                for w_addr in cluster_wallets:
                    self._wallet_to_cluster[w_addr] = cluster_id
                    self.wallets[w_addr].cluster_id = cluster_id
                    if w_addr in self.deployers:
                        cluster.deployers.add(w_addr)
                
                self._analyze_cluster(cluster)
                self.clusters[cluster_id] = cluster

    def _find_connected_component(self, start: str, min_connections: int) -> Set[str]:
        if start not in self.wallets:
            return set()
        if not self.graph.has_node(start):
            return {start}
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            if not self.graph.has_node(node):
                continue
            neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
            for n in neighbors:
                if n in self.wallets and n not in component:
                    edge_data = self.graph.get_edge_data(node, n) or self.graph.get_edge_data(n, node)
                    if edge_data:
                        total_weight = sum(d.get('weight', 1) for d in edge_data.values())
                        if total_weight >= min_connections:
                            queue.append(n)
        return component

    def _analyze_cluster(self, cluster: ClusterProfile):
        total_launches = 0
        rugged = 0
        successful = 0
        migrated = 0
        
        for deployer_addr in cluster.deployers:
            dp = self.deployers.get(deployer_addr)
            if dp:
                total_launches += len(dp.tokens_created)
                rugged += len(dp.tokens_rugged)
                successful += len(dp.tokens_successful)
                migrated += len(dp.tokens_migrated)
        
        cluster.total_launches = total_launches
        cluster.rugged_launches = rugged
        cluster.successful_launches = successful
        cluster.migrated_launches = migrated
        
        if total_launches > 0:
            cluster.avg_rug_rate = rugged / total_launches
            cluster.avg_success_rate = successful / total_launches
        
        if cluster.avg_rug_rate > 0.7:
            cluster.risk_level = "critical"
        elif cluster.avg_rug_rate > 0.4:
            cluster.risk_level = "high"
        elif cluster.avg_rug_rate > 0.2:
            cluster.risk_level = "medium"
        else:
            cluster.risk_level = "low"

    def assess_launch_risk(self, deployer: str, funding_wallets: List[str], 
                           initial_buyers: List[str]) -> Dict[str, Any]:
        risk_factors = []
        risk_score = 0.0
        
        dp = self.deployers.get(deployer)
        if dp:
            if dp.rug_rate > 0.5:
                risk_factors.append(f"Deployer rug rate: {dp.rug_rate:.1%}")
                risk_score += 0.4
            if dp.rug_rate > 0.2:
                risk_factors.append(f"Deployer historical rug rate: {dp.rug_rate:.1%}")
                risk_score += 0.2
            if dp.success_rate > 0.3:
                risk_score -= 0.15
        
        for funder in funding_wallets:
            cluster = self.find_cluster(funder)
            if cluster:
                if cluster.risk_level == "critical":
                    risk_factors.append(f"Funding from critical-risk cluster {cluster.cluster_id}")
                    risk_score += 0.3
                elif cluster.risk_level == "high":
                    risk_factors.append(f"Funding from high-risk cluster {cluster.cluster_id}")
                    risk_score += 0.15
        
        smart_buyers = 0
        insider_buyers = 0
        for buyer in initial_buyers:
            wp = self.wallets.get(buyer.get("address", ""))
            if wp:
                if wp.is_smart_money:
                    smart_buyers += 1
                if wp.is_insider:
                    insider_buyers += 1
        
        if smart_buyers == 0 and len(initial_buyers) > 5:
            risk_factors.append("No smart money in initial buyers")
            risk_score += 0.1
        if insider_buyers > len(initial_buyers) * 0.5:
            risk_factors.append("High insider concentration in initial buyers")
            risk_score += 0.25
        
        return {
            "risk_score": min(1.0, max(0.0, risk_score)),
            "risk_factors": risk_factors,
            "deployer_profile": dp.__dict__ if dp else None,
            "smart_buyers": smart_buyers,
            "insider_buyers": insider_buyers
        }

    def get_smart_money_wallets(self, min_trust: float = 0.7, min_win_rate: float = 0.3) -> List[WalletProfile]:
        return [
            w for w in self.wallets.values()
            if w.trust_score >= min_trust and w.win_rate_5x >= min_win_rate and w.is_smart_money
        ]

    def get_entity_graph_stats(self) -> Dict:
        return {
            "total_wallets": len(self.wallets),
            "total_deployers": len(self.deployers),
            "total_clusters": len(self.clusters),
            "smart_money_wallets": len(self.get_smart_money_wallets()),
            "rugged_deployers": sum(1 for d in self.deployers.values() if d.wallet_profile.is_rugged_deployer),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
            ,"data_status": dict(self.data_status)
        }

    def serialize(self) -> Dict:
        return {
            "wallets": {addr: {k: v for k, v in wp.__dict__.items() if not k.startswith('_')} 
                       for addr, wp in self.wallets.items()},
            "deployers": {addr: {k: v for k, v in dp.__dict__.items() if not k.startswith('_')} 
                         for addr, dp in self.deployers.items()},
            "clusters": {cid: {k: v for k, v in cp.__dict__.items() if not k.startswith('_')} 
                        for cid, cp in self.clusters.items()},
            "wallet_to_cluster": self._wallet_to_cluster
        }

    @classmethod
    def deserialize(cls, data: Dict, chain_config: ChainConfig, rpc: RPCManager, helius_key: str):
        graph = cls(chain_config, rpc, helius_key)
        for addr, wp_data in data.get("wallets", {}).items():
            graph.wallets[addr] = WalletProfile(**wp_data)
        for addr, dp_data in data.get("deployers", {}).items():
            wallet = graph.wallets.get(addr)
            graph.deployers[addr] = DeployerProfile(address=addr, wallet_profile=wallet or WalletProfile(addr, EntityType.DEPLOYER, 0, 0), **{k:v for k,v in dp_data.items() if k != 'wallet_profile'})
        for cid, cp_data in data.get("clusters", {}).items():
            graph.clusters[cid] = ClusterProfile(**cp_data)
        graph._wallet_to_cluster = data.get("wallet_to_cluster", {})
        return graph
