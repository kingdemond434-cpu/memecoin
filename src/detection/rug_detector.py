"""Chain-native token safety checks.

Solana assets are inspected as SPL Token or Token-2022 mints.  The detector does
not make ERC-20 ABI calls for Solana and treats unavailable safety evidence as
``DATA_BLOCKED`` rather than silently substituting zeroes.
"""

import asyncio
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
    def __init__(self, chain_config: ChainConfig, rpc: RPCManager, quote_provider: Any = None,
                 curve_state_provider: Any = None):
        self.chain_config = chain_config
        self.rpc = rpc
        self.quote_provider = quote_provider
        #: Returns the streamed bonding-curve state for a mint, or None.
        #: Consulted BEFORE the router, because a token still on its curve
        #: has a sell route by construction and the router has never heard
        #: of it -- see _check_sell_route.
        self.curve_state_provider = curve_state_provider
        self._cache: Dict[str, Tuple[TokenRiskReport, float]] = {}
        self._cache_ttl = 30

    def set_quote_provider(self, quote_provider: Any):
        self.quote_provider = quote_provider

    def set_curve_state_provider(self, provider: Any):
        self.curve_state_provider = provider

    async def analyze(
        self,
        token_address: str,
        pair_address: Optional[str] = None,
        base_token: Optional[str] = None,
        deployer_address: Optional[str] = None,
    ) -> TokenRiskReport:
        # Developer ownership is part of the report, so a report produced
        # without an identified creator must never satisfy a later request
        # which does identify one.
        cache_key = f"{self.chain_config.name}:{token_address}:{deployer_address or ''}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]
        if self.chain_config.chain_type == ChainType.SOLANA:
            report = await self._analyze_solana(token_address, deployer_address)
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

    async def _analyze_solana(self, mint: str,
                              deployer_address: Optional[str] = None) -> TokenRiskReport:
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

        holder_job = self._solana_holder_concentration(mint, mint_state["supply"])
        route_job = self._solana_sell_route(mint, mint_state)
        developer_job = self._solana_owner_token_share(
            mint, deployer_address, mint_state["supply"])
        # These are independent reads. Running them concurrently removes the
        # extra owner-enrichment calls from the serial candidate latency path.
        holders, route, developer = await asyncio.gather(
            holder_job, route_job, developer_job)
        checks["holders"] = holders
        checks["developer_balance"] = developer
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

        checks["sell_route"] = route
        route_feasible = route.get("feasible")
        if route.get("status") == "DATA_BLOCKED":
            blocked.append("sell_route")
        elif not route_feasible:
            warnings.append("No executable token-to-USDC route")
            score -= 60
        elif float(route.get("price_impact_pct") or 0.0) > 0.20:
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
            deployer_balance_pct=(float(developer["supply_pct"])
                                  if developer.get("status") == "OK" else None),
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
        if owner == TOKEN_2022_PROGRAM and len(data) > 82:
            # Token-2022 deliberately places extension TLV data after the
            # 165-byte legacy token-account boundary, even for an 82-byte
            # Mint.  Bytes 82..164 are zero padding, byte 165 is
            # AccountType::Mint (1), and TLV starts at 166.  Starting at 83
            # interprets the padding/account-type as a TLV header and falsely
            # rejects valid extended mints.
            if len(data) < 166:
                raise ValueError("truncated Token-2022 extension header")
            if any(data[82:165]):
                raise ValueError("non-zero Token-2022 mint padding")
            if data[165] != 1:
                raise ValueError("invalid Token-2022 mint account type")
            offset = 166
            while offset < len(data):
                if offset + 2 > len(data):
                    raise ValueError("malformed Token-2022 TLV extension type")
                extension_type = struct.unpack_from("<H", data, offset)[0]
                # ExtensionType::Uninitialized terminates the used TLV area.
                # Allocation padding after it must stay zeroed.
                if extension_type == 0:
                    if any(data[offset:]):
                        raise ValueError("non-zero data after Token-2022 TLV terminator")
                    break
                if offset + 4 > len(data):
                    raise ValueError("malformed Token-2022 TLV extension header")
                extension_type, length = struct.unpack_from("<HH", data, offset)
                offset += 4
                if offset + length > len(data):
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
            addresses = [str(item.get("address", "") or "") for item in values[:20]]
            owners = await self._solana_token_account_owners(addresses)
            accounts = []
            resolved_amount = 0
            for address, amount in zip(addresses, amounts):
                owner = owners.get("owners", {}).get(address)
                if owner:
                    resolved_amount += amount
                accounts.append({
                    "token_account": address,
                    "owner": owner,
                    "amount_raw": amount,
                    "supply_pct": 100.0 * amount / supply,
                })
            return {
                "status": "OK",
                "largest_account_count": len(values),
                "holder_count_status": "DATA_BLOCKED",
                "top_10_pct": 100.0 * sum(amounts[:10]) / supply,
                "top_20_pct": 100.0 * sum(amounts) / supply,
                "owner_enrichment_status": owners.get("status", "DATA_BLOCKED"),
                "owner_enrichment_error": owners.get("error"),
                "owner_resolved_accounts": sum(1 for item in accounts if item["owner"]),
                "owner_resolved_supply_pct": 100.0 * resolved_amount / supply,
                "accounts": accounts,
                "note": ("largest token-account concentration with native SPL owner "
                         "resolution; entity labels remain separate evidence"),
            }
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "error": str(exc), "count": 0}

    async def _solana_token_account_owners(self, addresses: List[str]) -> Dict[str, Any]:
        """Resolve SPL token-account addresses to controlling wallet owners."""
        usable = [address for address in addresses if address]
        if not usable:
            return {"status": "OK", "owners": {}}
        try:
            result = await self.rpc.request("getMultipleAccounts", [
                usable, {"encoding": "jsonParsed", "commitment": "confirmed"},
            ])
            values = (result or {}).get("value", [])
            owners = {}
            for address, account in zip(usable, values):
                owner = self.parse_spl_token_account_owner(account)
                if owner:
                    owners[address] = owner
            if len(owners) == len(usable):
                status = "OK"
            elif owners:
                status = "PARTIAL"
            else:
                status = "DATA_BLOCKED"
            return {"status": status, "owners": owners,
                    "resolved": len(owners), "requested": len(usable)}
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "owners": {}, "error": str(exc)}

    async def _solana_owner_token_share(self, mint: str, owner: Optional[str],
                                        supply: int) -> Dict[str, Any]:
        """Measure an owner's complete balance for a mint using native RPC."""
        if not owner:
            return {"status": "DATA_BLOCKED", "reason": "deployer unavailable"}
        if supply <= 0:
            return {"status": "OK", "owner": owner, "amount_raw": 0,
                    "supply_pct": 0.0, "token_accounts": 0}
        try:
            result = await self.rpc.request("getTokenAccountsByOwner", [
                owner, {"mint": mint},
                {"encoding": "jsonParsed", "commitment": "confirmed"},
            ])
            values = (result or {}).get("value", [])
            amounts = [self.parse_spl_token_account_amount(item.get("account") or {})
                       for item in values]
            if any(amount is None for amount in amounts):
                return {"status": "DATA_BLOCKED", "owner": owner,
                        "reason": "one or more developer token accounts did not decode",
                        "token_accounts": len(values)}
            total = sum(int(amount or 0) for amount in amounts)
            return {"status": "OK", "owner": owner, "amount_raw": total,
                    "supply_pct": 100.0 * total / supply,
                    "token_accounts": len(values)}
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "owner": owner, "error": str(exc)}

    @staticmethod
    def parse_spl_token_account_owner(account: Optional[Dict[str, Any]]) -> Optional[str]:
        if not account:
            return None
        data = account.get("data")
        if isinstance(data, dict):
            info = ((data.get("parsed") or {}).get("info") or {})
            owner = info.get("owner")
            return str(owner) if owner else None
        if isinstance(data, list) and data:
            try:
                raw = base64.b64decode(data[0], validate=True)
            except (ValueError, TypeError):
                return None
            if len(raw) >= 64:
                return RugDetector._base58_encode(raw[32:64])
        return None

    @staticmethod
    def parse_spl_token_account_amount(account: Optional[Dict[str, Any]]) -> Optional[int]:
        if not account:
            return None
        data = account.get("data")
        if isinstance(data, dict):
            info = ((data.get("parsed") or {}).get("info") or {})
            token_amount = info.get("tokenAmount") or {}
            try:
                return int(token_amount.get("amount"))
            except (TypeError, ValueError):
                return None
        if isinstance(data, list) and data:
            try:
                raw = base64.b64decode(data[0], validate=True)
            except (ValueError, TypeError):
                return None
            if len(raw) >= 72:
                return struct.unpack_from("<Q", raw, 64)[0]
        return None

    @staticmethod
    def _base58_encode(raw: bytes) -> str:
        alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        number = int.from_bytes(raw, "big")
        encoded = ""
        while number:
            number, remainder = divmod(number, 58)
            encoded = alphabet[remainder] + encoded
        zeros = len(raw) - len(raw.lstrip(b"\0"))
        return "1" * zeros + encoded

    async def _solana_sell_route(self, mint: str, mint_state: Dict[str, Any]) -> Dict[str, Any]:
        # The curve first. A pump.fun mint that has not migrated is sold
        # back to its own bonding curve, which this desk executes natively --
        # so a sell route EXISTS by construction and its impact is exact
        # arithmetic on the reserves.
        #
        # Asking the router instead was the single largest source of
        # rejection on this desk: measured 2026-08-29, every one of 419
        # decided launches was rejected by the hard safety veto and never
        # reached the economic layer at all, with sell_route_unavailable in
        # 240 of them and catastrophic_exit_price_impact in 181. Jupiter has
        # not indexed a mint that is seconds old, and the old code turned
        # that ignorance into {"status": "OK", "feasible": False} -- a
        # CONFIDENT false rather than a gap, which is why it hard-vetoed
        # instead of degrading to uncertainty.
        curve = None
        if self.curve_state_provider is not None:
            try:
                curve = self.curve_state_provider(mint)
            except Exception:
                curve = None
        if curve is not None and getattr(curve, "tradeable", False):
            return {
                "status": "OK",
                "feasible": True,
                "venue": "bonding_curve",
                # Selling back to the curve is the route. Impact is a
                # function of size against reserves and is priced by the
                # sizing engine, which knows the size; asserting a number
                # here without one would be inventing the position.
                "price_impact_pct": None,
                "detail": "native bonding-curve exit; router not consulted",
            }
        if not self.quote_provider or not getattr(self.quote_provider, "_session", None):
            return {"status": "DATA_BLOCKED", "feasible": None, "reason": "quote provider unavailable"}
        amount = min(max(1, mint_state["supply"] // 10_000), 1_000 * 10 ** mint_state["decimals"])
        try:
            quote = await self.quote_provider.get_quote(mint, USDC_MINT, amount, slippage_bps=500)
        except Exception as exc:
            return {"status": "DATA_BLOCKED", "feasible": None, "error": str(exc)}
        if not quote:
            # The router has no route. For a mint it has never indexed that
            # is ignorance, not a property of the token, and the difference
            # decides whether this hard-vetoes or merely adds uncertainty.
            return {"status": "DATA_BLOCKED", "feasible": None,
                    "reason": "router returned no route; it may not have "
                              "indexed this mint yet"}
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
