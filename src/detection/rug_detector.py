import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from functools import lru_cache

from web3 import AsyncWeb3
from web3.contract import AsyncContract

from src.chains.rpc_manager import ChainConfig, RPCManager

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    HONEYPOT = "honeypot"
    RUGGED = "rugged"


@dataclass
class TokenRiskReport:
    token_address: str
    chain: str
    risk_level: RiskLevel
    score: float
    checks: Dict[str, Any]
    warnings: List[str]
    timestamp: float = field(default_factory=time.time)
    liquidity_usd: Optional[float] = None
    liquidity_locked: bool = False
    ownership_renounced: bool = False
    max_tx_limit: Optional[int] = None
    max_wallet_limit: Optional[int] = None
    buy_tax: float = 0
    sell_tax: float = 0
    transfer_tax: float = 0
    can_mint: bool = False
    can_freeze: bool = False
    is_proxy: bool = False
    holder_count: int = 0
    top_10_pct: float = 0
    deployer_balance_pct: float = 0


class RugDetector:
    ERC20_ABI = [
        {"inputs": [], "name": "name", "outputs": [{"type": "string"}], "type": "function"},
        {"inputs": [], "name": "symbol", "outputs": [{"type": "string"}], "type": "function"},
        {"inputs": [], "name": "decimals", "outputs": [{"type": "uint8"}], "type": "function"},
        {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [], "name": "owner", "outputs": [{"type": "address"}], "type": "function"},
        {"inputs": [], "name": "renouncedOwnership", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "isExcludedFromFee", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "isExcludedFromReward", "outputs": [{"type": "bool"}], "type": "function"},
    ]

    EXTENDED_ABI = ERC20_ABI + [
        {"inputs": [], "name": "buyTax", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [], "name": "sellTax", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [], "name": "transferTax", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [], "name": "maxTxAmount", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [], "name": "maxWalletAmount", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [], "name": "tradingOpen", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [], "name": "swapAndLiquify", "outputs": [], "type": "function"},
        {"inputs": [], "name": "isOpen", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "isBlacklisted", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "isWhitelisted", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [], "name": "mintingFinished", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "mint", "outputs": [], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "freezeAccount", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "increaseAllowance", "outputs": [{"type": "bool"}], "type": "function"},
        {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "decreaseAllowance", "outputs": [{"type": "bool"}], "type": "function"},
    ]

    PAIR_ABI = [
        {"inputs": [], "name": "getReserves", "outputs": [{"type": "uint112"}, {"type": "uint112"}, {"type": "uint32"}], "type": "function"},
        {"inputs": [], "name": "token0", "outputs": [{"type": "address"}], "type": "function"},
        {"inputs": [], "name": "token1", "outputs": [{"type": "address"}], "type": "function"},
        {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "type": "function"},
        {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "type": "function"},
    ]

    def __init__(self, chain_config: ChainConfig, rpc: RPCManager):
        self.chain_config = chain_config
        self.rpc = rpc
        self._web3: Optional[AsyncWeb3] = None
        self._token_contracts: Dict[str, AsyncContract] = {}
        self._pair_contracts: Dict[str, AsyncContract] = {}
        self._cache: Dict[str, Tuple[TokenRiskReport, float]] = {}
        self._cache_ttl = 60

    async def _get_web3(self) -> AsyncWeb3:
        if not self._web3:
            self._web3 = self.rpc.get_web3()
        return self._web3

    def _get_token_contract(self, address: str) -> AsyncContract:
        if address not in self._token_contracts:
            w3 = await self._get_web3()
            self._token_contracts[address] = w3.eth.contract(
                address=w3.to_checksum_address(address),
                abi=self.EXTENDED_ABI
            )
        return self._token_contracts[address]

    def _get_pair_contract(self, address: str) -> AsyncContract:
        if address not in self._pair_contracts:
            w3 = await self._get_web3()
            self._pair_contracts[address] = w3.eth.contract(
                address=w3.to_checksum_address(address),
                abi=self.PAIR_ABI
            )
        return self._pair_contracts[address]

    async def analyze(self, token_address: str, pair_address: Optional[str] = None,
                      base_token: Optional[str] = None) -> TokenRiskReport:
        cache_key = f"{self.chain_config.name}:{token_address.lower()}"
        if cache_key in self._cache:
            report, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return report

        token_addr = self.chain_config.chain_id != "solana" and token_address.lower() or token_address
        w3 = await self._get_web3()
        token_contract = self._get_token_contract(token_addr)

        checks = {}
        warnings = []
        score = 100.0

        try:
            basic_info = await self._get_basic_info(token_contract)
            checks.update(basic_info)
        except Exception as e:
            warnings.append(f"Basic info failed: {e}")

        try:
            tax_info = await self._check_taxes(token_contract)
            checks["taxes"] = tax_info
            if tax_info["buy_tax"] > self.chain_config.max_tax:
                warnings.append(f"High buy tax: {tax_info['buy_tax']}%")
                score -= 20
            if tax_info["sell_tax"] > self.chain_config.max_tax:
                warnings.append(f"High sell tax: {tax_info['sell_tax']}%")
                score -= 30
            if tax_info["sell_tax"] > 50:
                return TokenRiskReport(
                    token_address=token_address, chain=self.chain_config.name,
                    risk_level=RiskLevel.HONEYPOT, score=0, checks=checks,
                    warnings=warnings + ["Extreme sell tax - likely honeypot"],
                    buy_tax=tax_info["buy_tax"], sell_tax=tax_info["sell_tax"]
                )
        except Exception as e:
            warnings.append(f"Tax check failed: {e}")

        try:
            ownership = await self._check_ownership(token_contract)
            checks["ownership"] = ownership
            if not ownership["renounced"]:
                warnings.append("Ownership not renounced")
                score -= 15
            if ownership["can_mint"]:
                warnings.append("Can mint new tokens")
                score -= 25
            if ownership["can_freeze"]:
                warnings.append("Can freeze accounts")
                score -= 20
        except Exception as e:
            warnings.append(f"Ownership check failed: {e}")

        try:
            limits = await self._check_limits(token_contract)
            checks["limits"] = limits
        except Exception as e:
            warnings.append(f"Limits check failed: {e}")

        liquidity_usd = None
        liquidity_locked = False
        if pair_address:
            try:
                liq_info = await self._check_liquidity(pair_address, base_token, token_address)
                checks["liquidity"] = liq_info
                liquidity_usd = liq_info.get("liquidity_usd")
                liquidity_locked = liq_info.get("locked", False)
                if liquidity_usd and liquidity_usd < self.chain_config.min_liquidity_usd:
                    warnings.append(f"Low liquidity: ${liquidity_usd:.0f}")
                    score -= 20
                if not liquidity_locked:
                    warnings.append("Liquidity not locked")
                    score -= 15
            except Exception as e:
                warnings.append(f"Liquidity check failed: {e}")

        try:
            holder_info = await self._check_holders(token_address)
            checks["holders"] = holder_info
            if holder_info["top_10_pct"] > 80:
                warnings.append(f"Top 10 holders control {holder_info['top_10_pct']:.1f}%")
                score -= 20
            if holder_info["deployer_pct"] > 10:
                warnings.append(f"Deployer holds {holder_info['deployer_pct']:.1f}%")
                score -= 15
        except Exception as e:
            warnings.append(f"Holder check failed: {e}")

        if pair_address:
            try:
                honeypot_result = await self._honeypot_check(token_address, pair_address, base_token)
                checks["honeypot"] = honeypot_result
                if honeypot_result["is_honeypot"]:
                    return TokenRiskReport(
                        token_address=token_address, chain=self.chain_config.name,
                        risk_level=RiskLevel.HONEYPOT, score=0, checks=checks,
                        warnings=warnings + ["HONEYPOT DETECTED - Cannot sell"],
                        liquidity_usd=liquidity_usd, liquidity_locked=liquidity_locked
                    )
            except Exception as e:
                warnings.append(f"Honeypot check failed: {e}")

        risk_level = self._calculate_risk_level(score, warnings)
        
        report = TokenRiskReport(
            token_address=token_address,
            chain=self.chain_config.name,
            risk_level=risk_level,
            score=max(0, score),
            checks=checks,
            warnings=warnings,
            liquidity_usd=liquidity_usd,
            liquidity_locked=liquidity_locked,
            ownership_renounced=checks.get("ownership", {}).get("renounced", False),
            max_tx_limit=checks.get("limits", {}).get("max_tx"),
            max_wallet_limit=checks.get("limits", {}).get("max_wallet"),
            buy_tax=checks.get("taxes", {}).get("buy_tax", 0),
            sell_tax=checks.get("taxes", {}).get("sell_tax", 0),
            transfer_tax=checks.get("taxes", {}).get("transfer_tax", 0),
            can_mint=checks.get("ownership", {}).get("can_mint", False),
            can_freeze=checks.get("ownership", {}).get("can_freeze", False),
            holder_count=checks.get("holders", {}).get("count", 0),
            top_10_pct=checks.get("holders", {}).get("top_10_pct", 0),
            deployer_balance_pct=checks.get("holders", {}).get("deployer_pct", 0),
        )

        self._cache[cache_key] = (report, time.time())
        return report

    async def _get_basic_info(self, contract: AsyncContract) -> Dict:
        try:
            name = await contract.functions.name().call()
        except Exception:
            name = "Unknown"
        try:
            symbol = await contract.functions.symbol().call()
        except Exception:
            symbol = "Unknown"
        try:
            decimals = await contract.functions.decimals().call()
        except Exception:
            decimals = 18
        try:
            total_supply = await contract.functions.totalSupply().call()
        except Exception:
            total_supply = 0
        return {"name": name, "symbol": symbol, "decimals": decimals, "total_supply": total_supply}

    async def _check_taxes(self, contract: AsyncContract) -> Dict:
        result = {"buy_tax": 0, "sell_tax": 0, "transfer_tax": 0}
        for tax_type in ["buyTax", "sellTax", "transferTax"]:
            try:
                val = await getattr(contract.functions, tax_type)().call()
                result[f"{tax_type.lower().replace('tax', '_tax')}"] = val / 100 if val > 100 else val
            except Exception:
                pass
        return result

    async def _check_ownership(self, contract: AsyncContract) -> Dict:
        result = {"owner": None, "renounced": False, "can_mint": False, "can_freeze": False, "is_proxy": False}
        try:
            owner = await contract.functions.owner().call()
            result["owner"] = owner
            if owner == "0x0000000000000000000000000000000000000000":
                result["renounced"] = True
        except Exception:
            pass
        try:
            result["can_mint"] = await contract.functions.mintingFinished().call() is False
        except Exception:
            pass
        try:
            await contract.functions.freezeAccount("0x0000000000000000000000000000000000000000").call()
            result["can_freeze"] = True
        except Exception:
            pass
        return result

    async def _check_limits(self, contract: AsyncContract) -> Dict:
        result = {"max_tx": None, "max_wallet": None, "trading_open": True}
        try:
            result["max_tx"] = await contract.functions.maxTxAmount().call()
        except Exception:
            pass
        try:
            result["max_wallet"] = await contract.functions.maxWalletAmount().call()
        except Exception:
            pass
        try:
            result["trading_open"] = await contract.functions.tradingOpen().call()
        except Exception:
            try:
                result["trading_open"] = await contract.functions.isOpen().call()
            except Exception:
                pass
        return result

    async def _check_liquidity(self, pair_address: str, base_token: str, token_address: str) -> Dict:
        pair_contract = self._get_pair_contract(pair_address)
        reserves = await pair_contract.functions.getReserves().call()
        token0 = await pair_contract.functions.token0().call()
        token1 = await pair_contract.functions.token1().call()
        
        token0_addr = token0.lower()
        token1_addr = token1.lower()
        token_addr = token_address.lower()
        base_addr = base_token.lower() if base_token else None
        
        if token0_addr == token_addr:
            token_reserve, base_reserve = reserves[0], reserves[1]
        else:
            token_reserve, base_reserve = reserves[1], reserves[0]
        
        w3 = await self._get_web3()
        token_contract = self._get_token_contract(token_address)
        decimals = await token_contract.functions.decimals().call()
        
        token_amount = token_reserve / (10 ** decimals)
        
        base_decimals = 18
        if base_addr:
            base_contract = self._get_token_contract(base_token)
            try:
                base_decimals = await base_contract.functions.decimals().call()
            except Exception:
                pass
        base_amount = base_reserve / (10 ** base_decimals)
        
        base_price_usd = await self._get_base_token_price_usd(base_token) if base_token else 1.0
        liquidity_usd = base_amount * base_price_usd * 2
        
        total_supply = await pair_contract.functions.totalSupply().call()
        lp_balance = await pair_contract.functions.balanceOf(pair_address).call()
        
        return {
            "token_reserve": token_amount,
            "base_reserve": base_amount,
            "liquidity_usd": liquidity_usd,
            "locked": lp_balance == 0 or lp_balance / total_supply > 0.9,
            "lp_burned_pct": (lp_balance / total_supply * 100) if total_supply > 0 else 0
        }

    async def _check_holders(self, token_address: str) -> Dict:
        return {"count": 0, "top_10_pct": 0, "deployer_pct": 0}

    async def _honeypot_check(self, token_address: str, pair_address: str, base_token: str) -> Dict:
        if not self.chain_config.honeypot_check:
            return {"is_honeypot": False, "simulated": False}
        
        try:
            w3 = await self._get_web3()
            token_contract = self._get_token_contract(token_address)
            pair_contract = self._get_pair_contract(pair_address)
            
            reserves = await pair_contract.functions.getReserves().call()
            token0 = await pair_contract.functions.token0().call()
            token_addr = token_address.lower()
            
            token_reserve = reserves[0] if token0.lower() == token_addr else reserves[1]
            base_reserve = reserves[1] if token0.lower() == token_addr else reserves[0]
            
            test_amount = token_reserve // 10000
            if test_amount == 0:
                return {"is_honeypot": False, "simulated": False}
            
            router_addr = list(self.chain_config.routers.values())[0]
            router_contract = w3.eth.contract(
                address=w3.to_checksum_address(router_addr),
                abi=[{"inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}], "name": "getAmountsOut", "outputs": [{"name": "amounts", "type": "uint256[]"}], "type": "function"}]
            )
            
            path = [w3.to_checksum_address(token_address), w3.to_checksum_address(base_token)]
            try:
                amounts = await router_contract.functions.getAmountsOut(test_amount, path).call()
                expected_out = amounts[-1]
                
                min_out = int(expected_out * 0.5)
                if min_out > 0:
                    return {"is_honeypot": False, "simulated": True, "expected_out": expected_out}
            except Exception:
                pass
            
            return {"is_honeypot": True, "simulated": True, "reason": "Swap simulation failed"}
        except Exception as e:
            return {"is_honeypot": False, "simulated": False, "error": str(e)}

    async def _get_base_token_price_usd(self, base_token: str) -> float:
        stablecoins = ["usdc", "usdt", "dai", "busd", "tusd"]
        base_lower = base_token.lower()
        for stable in stablecoins:
            if stable in base_lower:
                return 1.0
        return 1.0

    def _calculate_risk_level(self, score: float, warnings: List[str]) -> RiskLevel:
        if score >= 80:
            return RiskLevel.SAFE
        elif score >= 60:
            return RiskLevel.LOW
        elif score >= 40:
            return RiskLevel.MEDIUM
        elif score >= 20:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL