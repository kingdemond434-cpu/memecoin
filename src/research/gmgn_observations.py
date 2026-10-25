"""Authorized, key-gated GMGN read-only research observations.

The owner explicitly authorized public token/wallet/deployer address queries
after the initial automatic review rejection on 25 October 2026. This adapter
cannot sign, place an order, load a wallet key or call a caller-supplied URL.
Protocol: GMGNAI/gmgn-skills b0f4ac71355bb565b592c48061ecb2663549fe4b.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from urllib.parse import urlencode

from src.research.data_miners import Enriches, MinerSpec, RateLimited, rotating_selector

ORIGIN = "https://openapi.gmgn.ai"
READS = {
    "token_info": ("/v1/token/info", "mint", "address", Enriches.TOKEN_METADATA),
    "token_security": ("/v1/token/security", "mint", "address", Enriches.SUPPLY_CONTROL),
    "token_pools": ("/v1/token/pool_info", "mint", "address", Enriches.VENUE_LIQUIDITY),
    "token_holders": ("/v1/market/token_top_holders", "mint", "address", Enriches.HOLDER_STRUCTURE),
    "token_traders": ("/v1/market/token_top_traders", "mint", "address", Enriches.WALLET_HISTORY),
    "wallet_activity": ("/v1/user/wallet_activity", "address", "wallet_address", Enriches.WALLET_HISTORY),
    "wallet_stats": ("/v1/user/wallet_stats", "address", "wallet_address", Enriches.WALLET_HISTORY),
    "creator_launches": ("/v1/user/created_tokens", "address", "wallet_address", Enriches.WALLET_HISTORY),
}


def _redacted(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redacted(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redacted(key, secret): ("[REDACTED]" if key.lower() in {
            "api_key", "apikey", "x-apikey", "private_key", "privatekey", "secret",
            "seed", "seed_phrase", "mnemonic", "authorization"}
            else _redacted(item, secret)) for key, item in value.items()}
    return value


class GMGNReadClient:
    def __init__(self, http, *, credential=lambda: os.getenv("GMGN_API_KEY", "")):
        self.http, self.credential = http, credential
        self.guard = threading.Lock()
        self.next_request = self.cooldown = 0.0

    async def observe(self, operation, target):
        if operation not in READS:
            raise ValueError("unreviewed GMGN read operation")
        key = self.credential()
        if not key:
            raise RuntimeError("GMGN_API_KEY missing")
        from src.chains.yellowstone_grpc import b58decode
        try:
            valid = isinstance(target, str) and len(b58decode(target)) == 32
        except (ValueError, UnicodeError):
            valid = False
        if not valid:
            raise ValueError("GMGN Solana public address required")
        path, field, query_field, _ = READS[operation]
        with self.guard:
            now = time.monotonic()
            if now < self.cooldown:
                raise RateLimited("GMGN provider cooldown")
            reserved = max(now, self.next_request)
            self.next_request = reserved + 0.5
        await asyncio.sleep(max(0, reserved - time.monotonic()))
        with self.guard:
            if time.monotonic() < self.cooldown:
                raise RateLimited("GMGN provider cooldown")
        query = {"chain": "sol", query_field: target,
                 "timestamp": int(time.time()), "client_id": str(uuid.uuid4())}
        if operation == "wallet_stats":
            query["period"] = "7d"
        try:
            status, body, headers = await self.http.get(ORIGIN + path + "?" + urlencode(query),
                headers={"X-APIKEY": key, "Content-Type": "application/json"}, allow_redirects=False)
        except Exception:
            raise RuntimeError("GMGN read transport unavailable") from None
        try:
            envelope = json.loads(body)
        except (ValueError, TypeError):
            envelope = None
        provider_error = envelope.get("error") if isinstance(envelope, dict) else None
        if status == 429 or (isinstance(provider_error, str) and provider_error in {"RATE_LIMIT_EXCEEDED", "RATE_LIMIT_BANNED"}):
            normalized = {k.lower():v for k,v in headers.items()}
            try:
                reset = float(normalized.get("x-ratelimit-reset", 0))
                if reset > 10**12:
                    reset /= 1000
                delay = max(60.0, float(normalized.get("retry-after", 0)), reset - time.time() + 1)
                if not 0 <= delay < float("inf"):
                    delay = 60.0
            except (ValueError, TypeError):
                delay = 60.0
            with self.guard:
                self.cooldown = max(self.cooldown, time.monotonic() + delay)
            raise RateLimited("GMGN provider cooldown")
        if (status != 200 or not isinstance(envelope, dict)
                or type(envelope.get("code")) is not int or envelope["code"] != 0):
            raise RuntimeError("GMGN read unavailable: HTTP " + str(status))
        return {field: target, "chain": "solana", "provider": "gmgn", "provider_operation": operation,
                "provider_payload": _redacted(envelope.get("data"), key),
                "observed_at": time.time(), "publication_time_known": False,
                "claim_status": "PROVIDER_CLAIM_UNVERIFIED", "execution_authority": False,
                "history_coverage": "provider response; pagination not exhausted",
                "data_status": "OBSERVED_PROVIDER_RESPONSE"}


def register_gmgn_miners(pool, *, http, watched_tokens, watched_wallets, deployers):
    client = GMGNReadClient(http)
    result = {}
    for operation, (path, field, _, enriches) in READS.items():
        provider = watched_tokens if field == "mint" else deployers if operation == "creator_launches" else watched_wallets
        select = rotating_selector(provider, 1)
        async def fetch(operation=operation, select=select):
            return [await client.observe(operation, target) for target in select()]
        spec = MinerSpec("provider:gmgn:" + operation, enriches, 15,
            endpoint=ORIGIN + path, requires_env=("GMGN_API_KEY",),
            detail="authorized read-only public address queries; provider claims; no signing or orders")
        result[spec.miner_id] = pool.register(spec, fetch)
    return result
