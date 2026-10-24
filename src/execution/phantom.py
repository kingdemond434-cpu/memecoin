"""Interactive Phantom signer. No seed, private key, or submission endpoint.

The browser signs the exact transaction requested by the existing execution
engine. Its existing promotion and submission boundaries still apply. Paper
runs never call this signer. Connecting a wallet does not arm live trading.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction
from solders.message import from_bytes_versioned, to_bytes_versioned

from src.execution.signer import SignerPolicy, TransactionInspector


class PublicWallet:
    """A public address with deliberately no signing material."""
    def __init__(self, address: str):
        self._address = Pubkey.from_string(address)

    def pubkey(self):
        return self._address


class PhantomBridge:
    def __init__(self, *, timeout: float = 45, policy=None, live_authority=None):
        self.timeout = timeout
        self.policy = policy or SignerPolicy()
        self.inspector = TransactionInspector(self.policy)
        self.live_authority = live_authority or (lambda: False)
        self.connected = ""
        self.challenges: dict[str, tuple[str, bytes, float]] = {}
        self.pending: dict[str, dict] = {}
        self.signed = 0
        self.refused = 0
        self.last_signature_at = 0.0
        self.signature_times: list[float] = []

    def challenge(self, address: str) -> dict:
        Pubkey.from_string(address)
        now = time.time()
        self.challenges = {k: v for k, v in self.challenges.items() if v[2] > now}
        if len(self.challenges) >= 8:
            raise ValueError("too many wallet connection attempts")
        nonce = secrets.token_urlsafe(24)
        message = ("Memecoin research wallet connection\n"
                   "This proves ownership only. It authorizes no trades or transfers.\n"
                   f"Address: {address}\nNonce: {nonce}\nExpires: {int(now + 120)}").encode()
        self.challenges[nonce] = (address, message, now + 120)
        return {"nonce": nonce, "message_base64": base64.b64encode(message).decode()}

    def connect(self, nonce: str, signature_base64: str) -> str:
        item = self.challenges.pop(nonce, None)
        if not item or item[2] < time.time():
            raise ValueError("wallet challenge expired or already used")
        address, message, _ = item
        signature = Signature.from_bytes(base64.b64decode(signature_base64, validate=True))
        if not signature.verify(Pubkey.from_string(address), message):
            raise ValueError("wallet ownership signature does not verify")
        self.disconnect()
        self.connected = address
        return address

    def disconnect(self):
        self.connected = ""
        for item in self.pending.values():
            if not item["future"].done():
                item["future"].set_exception(PermissionError("Phantom disconnected"))

    def _authorise(self, message: bytes, address: str, *, count_limit=True):
        if not self.live_authority():
            raise PermissionError("live capital has not been authorised by the promotion ledger")
        if os.getenv("ALLOW_LIVE_TRADING") != "yes-i-understand":
            raise PermissionError("live trading acknowledgement absent")
        if self.policy.kill_file.exists():
            raise PermissionError("HALT_SIGNING is present")
        if not self.connected or self.connected != address:
            raise PermissionError("expected Phantom wallet is not connected")
        now = time.time()
        self.signature_times = [t for t in self.signature_times if now - t < 60]
        if count_limit and len(self.signature_times) >= self.policy.rate_limit_per_minute:
            raise PermissionError("Phantom signing rate limit reached")
        decision = self.inspector.inspect(message, expected_payer=address)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        parsed = from_bytes_versioned(message)
        if parsed.header.num_required_signatures != 1:
            raise PermissionError("Phantom bridge currently supports exactly one required signer")
        return parsed, decision

    async def sign(self, message: bytes, address: str) -> bytes:
        if len(self.pending) >= 4:
            raise PermissionError("Phantom approval queue is full")
        try:
            parsed, decision = self._authorise(message, address)
            tx = VersionedTransaction.populate(parsed, [Signature.default()])
            request_id = secrets.token_urlsafe(24)
            future = asyncio.get_running_loop().create_future()
            self.signature_times.append(time.time())
            self.pending[request_id] = {
                "id": request_id, "message": message, "address": address,
                "transaction_base64": base64.b64encode(bytes(tx)).decode(),
                "message_sha256": hashlib.sha256(message).hexdigest(),
                "expires_at": time.time() + self.timeout,
                "review": decision.to_dict(), "future": future,
            }
            try:
                return await asyncio.wait_for(future, self.timeout)
            finally:
                self.pending.pop(request_id, None)
        except Exception:
            self.refused += 1
            raise

    def requests(self) -> list[dict]:
        return [{k: v for k, v in item.items() if k not in {"message", "future"}}
                for item in self.pending.values() if not item["future"].done()]

    def complete(self, request_id: str, signed_base64: str = "", *, reject=False):
        item = self.pending.get(request_id)
        if not item or item["future"].done() or item["expires_at"] < time.time():
            raise ValueError("signing request expired or already answered")
        if reject:
            item["future"].set_exception(PermissionError("wallet owner rejected transaction"))
            return
        self._authorise(item["message"], item["address"], count_limit=False)
        tx = VersionedTransaction.from_bytes(base64.b64decode(signed_base64, validate=True))
        if to_bytes_versioned(tx.message) != item["message"]:
            raise ValueError("Phantom returned a different transaction message")
        if len(tx.signatures) != 1 or not tx.verify_with_results()[0]:
            raise ValueError("Phantom transaction signature does not verify")
        item["future"].set_result(bytes(tx.signatures[0]))
        self.signed += 1
        self.last_signature_at = time.time()

    def report(self):
        return {"mode": "phantom_interactive", "connected": bool(self.connected),
                "public_key": self.connected, "pending": len(self.pending),
                "signed": self.signed, "refused": self.refused,
                "live_authority": bool(self.live_authority()),
                "requires_wallet_owner_approval": True, "private_key_loaded": False}


class PhantomSignerClient:
    isolated = True

    def __init__(self, address: str, url: str, token_file: Path, *, timeout=50):
        Pubkey.from_string(address)
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("Phantom signer must use a loopback HTTP origin")
        self.public_key, self.url = address, url.rstrip("/")
        self.token_file, self.timeout = Path(token_file), timeout
        self.signed = self.refused = 0

    async def sign_message(self, message: bytes) -> bytes:
        token = self.token_file.read_text(encoding="utf-8").strip()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(self.url + "/api/phantom/sign", headers={"Authorization": "Bearer " + token},
                                    json={"address": self.public_key,
                                          "message_base64": base64.b64encode(message).decode()}) as response:
                result = await response.json()
                if response.status != 200:
                    self.refused += 1
                    raise PermissionError(result.get("error", "Phantom signing refused"))
        raw = base64.b64decode(result["signature_base64"], validate=True)
        if not Signature.from_bytes(raw).verify(Pubkey.from_string(self.public_key), message):
            raise ValueError("bridge signature verification failed")
        self.signed += 1
        return raw

    def report(self) -> dict[str, Any]:
        return {"mode": "phantom_interactive", "isolated": True,
                "public_key": self.public_key, "signed": self.signed, "refused": self.refused,
                "requires_wallet_owner_approval": True}
