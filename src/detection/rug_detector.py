"""Chain-native token safety checks.

Solana assets are inspected as SPL Token or Token-2022 mints.  The detector does
not make ERC-20 ABI calls for Solana and treats unavailable safety evidence as
``DATA_BLOCKED`` rather than silently substituting zeroes.
"""

import base64
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.chains.rpc_manager import ChainConfig, ChainType, RPCManager

logger = logging.getLogger(__name__)

TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    HONEYPOT = "honeypot"
    RUGGED = "rugged"


TOKEN_2022_EXTENSIONS = {
    0: "uninitialized",
    1: "transfer_fee_config",
    2: "transfer_fee_amount",
    3: "mint_close_authority",
    4: "confidential_transfer_mint",
    5: "confidential_transfer_account",
    6: "default_account_state",
    7: "immutable_owner",
    8: "memo_transfer",
    9: "non_transferable",
    10: "interest_bearing_config",
    11: "cpi_guard",
    12: "permanent_delegate",
    13: "non_transferable_account",
    14: "transfer_hook",
    15: "transfer_hook_account",
    16: "confidential_transfer_fee_config",
    17: "confidential_transfer_fee_amount",
    18: "metadata_pointer",
    19: "token_metadata",
    20: "group_pointer",
    21: "token_group",
    22: "group_member_pointer",
    23: "token_group_member",
    24: "confidential_mint_burn",
    25: "scaled_ui_amount",
    26: "pausable",
    27: "pausable_account",
}

HIGH_RISK_EXTENSIONS = {
    "transfer_fee_config": 18,
    "default_account_state": 12,
    "non_transferable": 60,
    "permanent_delegate": 35,
    "transfer_hook": 30,
    "confidential_transfer_mint": 30,
    "confidential_transfer_fee_config": 30,
    "confidential_mint_burn": 35,
    "pausable": 35,
}


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
    holder_count: Optional[int] = None
    top_10_pct: float = 0
    top_20_pct: Optional[float] = None
    # These require token-account owner/entity enrichment.  They stay None
    # until measured; zero would falsely assert that no developer, insider,
    # bundler, fresh wallet, whale or connected cluster holds supply.
    deployer_balance_pct: Optional[float] = None
    insider_pct: Optional[float] = None
    bundler_pct: Optional[float] = None
    fresh_wallet_pct: Optional[float] = None
    whale_pct: Optional[float] = None
    connected_cluster_pct: Optional[float] = None
    token_program: str = ""
    token_extensions: List[str] = field(default_factory=list)
    extension_risk: float = 0.0
    sell_route_feasible: Optional[bool] = None
    data_status: str = "OK"
    blocked_checks: List[str] = field(default_factory=list)


class RugDetector:
    def __init__(self, chain_config: ChainConfig, rpc: RPCManager, quote_provider: Any = None):
        self.chain_config = chain_config
        self.rpc = rpc
        self.quote_provider = quote_provider
        self._cache: Dict[str, Tuple[TokenRiskReport, float]] = {}
        self._cache_ttl = 30

    def set_quote_provider(self, quote_provider: Any):
        self.quote_provider = quote_provider

    async def analyze(
        self,
        token_address: str,
        pair_address: Optional[str] = None,
        base_token: Optional[str] = None,
    ) -> TokenRiskReport:
        cache_key = f"{self.chain_config.name}:{token_address}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]
        if self.chain_config.chain_type == ChainType.SOLANA:
            report = await self._analyze_solana(token_address)
        else:
            report = TokenRiskReport(
                token_address=token_address,
                chain=self.chain_config.name,
                risk_level=RiskLevel.CRITICAL,
                score=0,
                checks={"evm_analyzer": {"status": "DATA_BLOCKED"}},
                warnings=["EVM safety analyzer is not enabled in this Solana-focused build"],
                data_status="DATA_BLOCKED",
                blocked_checks=["evm_analyzer"],
            )
        self._cache[cache_key] = (report, time.time())
        return report

    async def _analyze_solana(self, mint: str) -> TokenRiskReport:
        warnings: List[str] = []
        blocked: List[str] = []
        checks: Dict[str, Any] = {}
        score = 100.0
        try:
            result = await self.rpc.request(
                "getAccountInfo", [mint, {"encoding": "base64", "commitment": "confirmed"}]
            )
            account = (result or {}).get("value")
        except Exception as exc:
            account = None
            checks["mint_account"] = {"status": "DATA_BLOCKED", "error": str(exc)}
        if not account:
            return TokenRiskReport(
                token_address=mint,
                chain=self.chain_config.name,
                risk_level=RiskLevel.CRITICAL,
                score=0,
                checks=checks,
                warnings=["Mint account unavailable or does not exist"],
                data_status="DATA_BLOCKED",
                blocked_checks=["mint_account"],
            )

        owner = account.get("owner", "")
        encoded = account.get("data", [""])[0] if isinstance(account.get("data"), list) else ""
        try:
            raw = base64.b64decode(encoded, validate=True)
            mint_state = self.parse_spl_mint(raw, owner)
        except (ValueError, struct.error) as exc:
            return TokenRiskReport(
                token_address=mint,
                chain=self.chain_config.name,
                risk_level=RiskLevel.CRITICAL,
                score=0,
                checks={"mint_account": {"status": "INVALID", "error": str(exc)}},
                warnings=["Malformed SPL mint account"],
                token_program=owner,
            )
        checks["mint"] = mint_state
        if owner not in {TOKEN_PROGRAM, TOKEN_2022_PROGRAM}:
            warnings.append(f"Mint owned by unexpected program {owner}")
            score -= 80
        if mint_state["mint_authority_present"]:
            warnings.append("Mint authority is active")
            score -= 30
        if mint_state["freeze_authority_present"]:
            warnings.append("Freeze authority is active")
            score -= 25
        if not mint_state["initialized"]:
            warnings.append("Mint is not initialized")
            score -= 100

        extensions = mint_state["extensions"]
        for name in extensions:
            penalty = HIGH_RISK_EXTENSIONS.get(name, 0)
            if penalty:
                warnings.append(f"Token-2022 extension requires review: {name}")
                score -= penalty

        holders = await self._solana_holder_concentration(mint, mint_state["supply"])
        checks["holders"] = holders
        if holders.get("status") == "DATA_BLOCKED":
            blocked.append("holders")
        top10 = float(holders.get("top_10_pct", 0))
        top20 = holders.get("top_20_pct")
        if top10 > 80:
            warnings.append(f"Top token accounts hold {top10:.1f}% of supply")
            score -= 25
        elif top10 > 50:
            warnings.append(f"Top token accounts hold {top10:.1f}% of supply")
            score -= 10

        route = await self._solana_sell_route(mint, mint_state)
        checks["sell_route"] = route
        route_feasible = route.get("feasible")
        if route.get("status") == "DATA_BLOCKED":
            blocked.append("sell_route")
        elif not route_feasible:
            warnings.append("No executable token-to-USDC route")
            score -= 60
        elif float(route.get("price_impact_pct", 0)) > 0.20:
            warnings.append("Sell route price impact exceeds 20%")
            score -= 25

        risk_level = self._calculate_risk_level(score)
        return TokenRiskReport(
            token_address=mint,
            chain=self.chain_config.name,
            risk_level=risk_level,
            score=max(0.0, score),
            checks=checks,
            warnings=warnings,
            ownership_renounced=not mint_state["mint_authority_present"],
            can_mint=mint_state["mint_authority_present"],
            can_freeze=mint_state["freeze_authority_present"],
            # getTokenLargestAccounts returns at most twenty accounts. Its
            # list length is not the token's holder count.
            holder_count=None,
            top_10_pct=top10,
            top_20_pct=float(top20) if top20 is not None else None,
            token_program=owner,
            token_extensions=extensions,
            extension_risk=min(1.0, sum(HIGH_RISK_EXTENSIONS.get(name, 0) for name in extensions) / 100.0),
            sell_route_feasible=route_feasible,
            data_status="DATA_BLOCKED" if blocked else "OK",
            blocked_checks=blocked,
        )

    @staticmethod
    def parse_spl_mint(data: bytes, owner: str) -> Dict[str, Any]:
        if len(data) < 82:
            raise ValueError(f"SPL mint is {len(data)} bytes; expected at least 82")
        mint_authority_tag = struct.unpack_from("<I", data, 0)[0]
        supply = struct.unpack_from("<Q", data, 36)[0]
        decimals = data[44]
        initialized = data[45] == 1
        freeze_authority_tag = struct.unpack_from("<I", data, 46)[0]
        if mint_authority_tag not in {0, 1} or freeze_authority_tag not in {0, 1}:
            raise ValueError("invalid SPL COption authority tag")
        extensions: List[str] = []
        if owner == TOKEN_2022_PROGRAM and len(data) > 83:
            offset = 83  # byte 82 is AccountType::Mint in Token-2022
            while offset + 4 <= len(data):
                extension_type, length = struct.unpack_from("<HH", data, offset)
                offset += 4
                if length < 0 or offset + length > len(data):
                    raise ValueError("malformed Token-2022 TLV extension")
                extensions.append(TOKEN_2022_EXTENSIONS.get(extension_type, f"unknown_{extension_type}"))
                offset += length
        return {
            "program": owner,
            "supply": supply,
            "decimals": decimals,
            "initialized": initialized,
            "mint_authority_present": mint_authority_tag != 0,
            "freeze_authority_present": freeze_authority_tag != 0,
            "extensions": sorted(set(extensions)),
        }

    async def _solana_holder_concentration(self, mint: str, supply: int) -> Dict[str, Any]:
        if supply <= 0:
            return {"status": "OK", "largest_account_count": 0,
                    "holder_count_status": "DATA_BLOCKED",
                    "top_10_pct": 100.0,
                    "top_20_pct": 100.0}
        try:
            result = await self.rpc.request("getTokenLargestAccounts", [mint, {"commitment": "confirmed"}])
            values = (result or {}).get("value", [])
            amounts = [int(item.get("amount", 0) or 0) for item in values[:20]]
            return {
                "status": "OK",
                "largest_account_count": len(values),
                "holder_count_status": "DATA_BLOCKED",
                "top_10_pct": 100.0 * sum(amounts[:10]) / supply,
                "top_20_pct": 100.0 * sum(amounts) / supply,
                "owner_enrichment_status": "DATA_BLOCKED",
                "note": ("largest token-account concentration; account owners, "
                         "developer share and entity-cluster concentration are separate evidence"),
            }
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "error": str(exc), "count": 0}

    async def _solana_sell_route(self, mint: str, mint_state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.quote_provider or not getattr(self.quote_provider, "_session", None):
            return {"status": "DATA_BLOCKED", "feasible": None, "reason": "quote provider unavailable"}
        amount = min(max(1, mint_state["supply"] // 10_000), 1_000 * 10 ** mint_state["decimals"])
        try:
            quote = await self.quote_provider.get_quote(mint, USDC_MINT, amount, slippage_bps=500)
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "feasible": None, "error": str(exc)}
        if not quote:
            return {"status": "OK", "feasible": False}
        return {
            "status": "OK",
            "feasible": quote.output_amount > 0,
            "test_amount": amount,
            "output_usdc_raw": quote.output_amount,
            "price_impact_pct": quote.price_impact_pct,
            "min_output_amount": quote.min_output_amount,
        }

    @staticmethod
    def _calculate_risk_level(score: float) -> RiskLevel:
        if score >= 80:
            return RiskLevel.SAFE
        if score >= 65:
            return RiskLevel.LOW
        if score >= 45:
            return RiskLevel.MEDIUM
        if score >= 20:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL
