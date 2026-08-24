"""Validated Yellowstone client plus Pump.fun and Raydium decoders.

The decoders accept both Yellowstone protobuf updates and canonical Solana
``getTransaction`` JSON. This makes historical-fixture tests use the exact same
path as the live feed, including inner CPI instructions.
"""

import asyncio
import base64
import hashlib
import importlib
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import grpc

logger = logging.getLogger(__name__)

B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(raw: bytes) -> str:
    number = int.from_bytes(raw, "big")
    encoded = bytearray()
    while number:
        number, remainder = divmod(number, 58)
        encoded.append(B58_ALPHABET[remainder])
    zeros = len(raw) - len(raw.lstrip(b"\0"))
    return (B58_ALPHABET[:1] * zeros + bytes(reversed(encoded or b""))).decode("ascii")


def b58decode(value: str) -> bytes:
    number = 0
    for char in value.encode("ascii"):
        number = number * 58 + B58_ALPHABET.index(char)
    raw = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    zeros = len(value) - len(value.lstrip("1"))
    return b"\0" * zeros + raw


class SolanaCommitment(Enum):
    PROCESSED = "processed"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"


@dataclass
class TransactionFilter:
    accounts_include: List[str] = field(default_factory=list)
    accounts_exclude: List[str] = field(default_factory=list)
    accounts_required: List[str] = field(default_factory=list)
    vote: bool = False
    failed: bool = False


@dataclass
class SubscribeRequest:
    transactions: Dict[str, TransactionFilter] = field(default_factory=dict)
    commitment: SolanaCommitment = SolanaCommitment.PROCESSED


class YellowstoneClient:
    def __init__(self, endpoint: str, x_token: str = ""):
        self.endpoint = endpoint
        self.x_token = x_token
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Any = None
        self._proto: Any = None
        self._stream: Any = None
        self._stream_task: Optional[asyncio.Task] = None
        self._running = False
        self._handlers: Dict[str, List[Callable]] = {}
        self._subscribe_request: Optional[SubscribeRequest] = None
        self._reconnect_attempts = 0
        self.status = "NOT_STARTED"
        self.status_detail = ""

    @property
    def available(self) -> bool:
        return self._stub is not None and self.status not in {"DATA_BLOCKED", "INVALID"}

    def validate_setup(self) -> Dict[str, Any]:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            return {"status": "INVALID", "detail": "YELLOWSTONE_GRPC_URL must be an http(s) URL"}
        for pb2_name, grpc_name in (
            ("src.chains.generated.geyser_pb2", "src.chains.generated.geyser_pb2_grpc"),
            ("geyser_pb2", "geyser_pb2_grpc"),
            ("yellowstone_grpc.geyser_pb2", "yellowstone_grpc.geyser_pb2_grpc"),
        ):
            try:
                proto = importlib.import_module(pb2_name)
                proto_grpc = importlib.import_module(grpc_name)
                return {"status": "OK", "proto": proto, "proto_grpc": proto_grpc}
            except ImportError:
                continue
        return {
            "status": "DATA_BLOCKED",
            "detail": "generated Yellowstone geyser_pb2/geyser_pb2_grpc modules are not installed",
        }

    async def connect(self) -> bool:
        validation = self.validate_setup()
        self.status = validation["status"]
        self.status_detail = validation.get("detail", "")
        if self.status != "OK":
            logger.warning("Yellowstone %s: %s", self.status, self.status_detail)
            return False
        self._proto = validation["proto"]
        parsed = urlparse(self.endpoint)
        target = parsed.netloc or self.endpoint
        options = [
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
        ]
        self._channel = (
            grpc.aio.secure_channel(target, grpc.ssl_channel_credentials(), options=options)
            if parsed.scheme == "https"
            else grpc.aio.insecure_channel(target, options=options)
        )
        self._stub = validation["proto_grpc"].GeyserStub(self._channel)
        metadata = (("x-token", self.x_token),) if self.x_token else ()
        try:
            await asyncio.wait_for(
                self._stub.GetVersion(self._proto.GetVersionRequest(), metadata=metadata), timeout=10
            )
            self.status = "READY"
            return True
        except Exception as exc:
            self.status = "DATA_BLOCKED"
            self.status_detail = f"Yellowstone handshake failed: {exc}"
            await self._channel.close()
            self._channel = None
            self._stub = None
            return False

    async def subscribe(self, request: SubscribeRequest) -> bool:
        if not self.available:
            return False
        self._subscribe_request = request
        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        return True

    async def _request_iterator(self) -> AsyncIterator[Any]:
        yield self._build_grpc_request(self._subscribe_request)
        while self._running:
            await asyncio.sleep(15)
            ping = self._proto.SubscribeRequest()
            if hasattr(ping, "ping"):
                ping.ping.id = int(time.time()) & 0x7FFFFFFF
            yield ping

    async def _stream_loop(self):
        while self._running:
            try:
                metadata = (("x-token", self.x_token),) if self.x_token else ()
                self._stream = self._stub.Subscribe(self._request_iterator(), metadata=metadata)
                self.status = "STREAMING"
                self._reconnect_attempts = 0
                async for response in self._stream:
                    await self._handle_response(response)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status = "DEGRADED"
                self.status_detail = str(exc)
                self._reconnect_attempts += 1
                if self._reconnect_attempts > 10:
                    self.status = "DATA_BLOCKED"
                    return
                await asyncio.sleep(min(30, 2 ** (self._reconnect_attempts - 1)))

    def _build_grpc_request(self, request: SubscribeRequest) -> Any:
        grpc_request = self._proto.SubscribeRequest()
        grpc_request.commitment = {
            SolanaCommitment.PROCESSED: 0,
            SolanaCommitment.CONFIRMED: 1,
            SolanaCommitment.FINALIZED: 2,
        }[request.commitment]
        for name, tx_filter in request.transactions.items():
            target = grpc_request.transactions[name]
            target.vote = tx_filter.vote
            target.failed = tx_filter.failed
            target.account_include.extend(tx_filter.accounts_include)
            target.account_exclude.extend(tx_filter.accounts_exclude)
            if hasattr(target, "account_required"):
                target.account_required.extend(tx_filter.accounts_required)
        return grpc_request

    async def _handle_response(self, response: Any):
        for event_type in ("account", "transaction", "block", "block_meta", "slot", "entry", "ping"):
            data = getattr(response, event_type, None)
            if data:
                await self._dispatch(event_type, data)
                return

    async def _dispatch(self, event_type: str, data: Any):
        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Yellowstone %s handler failed: %s", event_type, exc)

    def on(self, event_type: str, handler: Callable):
        self._handlers.setdefault(event_type, []).append(handler)

    async def close(self):
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        if self._stream:
            self._stream.cancel()
        if self._channel:
            await self._channel.close()
        self.status = "CLOSED"

    def get_status(self) -> Dict[str, str]:
        return {"status": self.status, "detail": self.status_detail}


class SolanaRpcProgramStream:
    """Confirmed-transaction fallback using the same decoders as Yellowstone."""

    def __init__(self, rpc: Any, programs: Iterable[str], poll_interval: float = 2.0):
        self.rpc = rpc
        self.programs = list(dict.fromkeys(programs))
        self.poll_interval = poll_interval
        self._handlers: Dict[str, List[Callable]] = {}
        self._seen: Set[str] = set()
        self._primed_programs: Set[str] = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.status = "NOT_STARTED"
        self.status_detail = ""

    def on(self, event_type: str, handler: Callable):
        self._handlers.setdefault(event_type, []).append(handler)

    async def start(self):
        self._running = True
        self.status = "RPC_FALLBACK"
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                return

    async def _loop(self):
        while self._running:
            try:
                await self.poll_once()
                self.status = "RPC_FALLBACK"
                self.status_detail = "confirmed polling; lower timeliness than Yellowstone"
            except Exception as exc:
                self.status = "DEGRADED"
                self.status_detail = str(exc)
            await asyncio.sleep(self.poll_interval)

    async def poll_once(self):
        for program in self.programs:
            signatures = await self.rpc.request(
                "getSignaturesForAddress", [program, {"limit": 50, "commitment": "confirmed"}]
            )
            if program not in self._primed_programs:
                # Establish a high-water mark without replaying an arbitrary
                # startup backlog as if it were newly detected point-in-time.
                self._seen.update(item.get("signature") for item in signatures or [] if item.get("signature"))
                self._primed_programs.add(program)
                continue
            for item in reversed(signatures or []):
                signature = item.get("signature")
                if not signature or signature in self._seen or item.get("err") is not None:
                    continue
                self._seen.add(signature)
                transaction = await self.rpc.request(
                    "getTransaction",
                    [signature, {"encoding": "json", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}],
                )
                if transaction:
                    for handler in self._handlers.get("transaction", []):
                        result = handler(transaction)
                        if asyncio.iscoroutine(result):
                            await result
        if len(self._seen) > 100_000:
            self._seen = set(list(self._seen)[-50_000:])

    def get_status(self) -> Dict[str, str]:
        return {"status": self.status, "detail": self.status_detail}


def _json_transaction_parts(tx_data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], str, int]:
    result = tx_data.get("result", tx_data)
    transaction = result.get("transaction", {})
    message = transaction.get("message", {})
    keys = [item.get("pubkey") if isinstance(item, dict) else item for item in message.get("accountKeys", [])]
    meta = result.get("meta") or {}
    loaded = meta.get("loadedAddresses") or {}
    keys.extend(loaded.get("writable", []))
    keys.extend(loaded.get("readonly", []))
    instructions = [dict(item, _inner=False) for item in message.get("instructions", [])]
    for group in meta.get("innerInstructions", []) or []:
        instructions.extend(dict(item, _inner=True, _outer_index=group.get("index")) for item in group.get("instructions", []))
    signatures = transaction.get("signatures", [])
    return keys, instructions, signatures[0] if signatures else "", int(result.get("slot", 0))


def _proto_transaction_parts(tx_data: Any) -> Tuple[List[str], List[Any], str, int]:
    info = getattr(tx_data, "transaction", tx_data)
    tx = getattr(info, "transaction", info)
    message = getattr(tx, "message", tx)
    keys = [b58encode(bytes(key)) for key in getattr(message, "account_keys", [])]
    meta = getattr(info, "meta", None)
    if meta:
        keys.extend(b58encode(bytes(key)) for key in getattr(meta, "loaded_writable_addresses", []))
        keys.extend(b58encode(bytes(key)) for key in getattr(meta, "loaded_readonly_addresses", []))
    instructions: List[Any] = list(getattr(message, "instructions", []))
    for group in getattr(meta, "inner_instructions", []) if meta else []:
        instructions.extend(getattr(group, "instructions", []))
    sig_raw = getattr(info, "signature", b"") or (getattr(tx, "signatures", [b""])[0])
    signature = b58encode(bytes(sig_raw)) if sig_raw else ""
    return keys, instructions, signature, int(getattr(tx_data, "slot", 0))


def transaction_parts(tx_data: Any) -> Tuple[List[str], List[Any], str, int]:
    return _json_transaction_parts(tx_data) if isinstance(tx_data, dict) else _proto_transaction_parts(tx_data)


def transaction_block_time(tx_data: Any) -> Optional[float]:
    if not isinstance(tx_data, dict):
        return None
    result = tx_data.get("result", tx_data)
    value = result.get("blockTime", result.get("block_time"))
    return float(value) if value is not None else None


def apply_event_timing(event: Dict[str, Any], tx_data: Any, received_ns: int) -> Dict[str, Any]:
    block_time = transaction_block_time(tx_data)
    event["block_time"] = block_time
    event["received_ns"] = received_ns
    event["decoded_ns"] = time.time_ns()
    event["timestamp"] = block_time if block_time is not None else received_ns / 1_000_000_000
    return event


def enrich_trade_balances(event: Dict[str, Any], tx_data: Any, keys: List[str]) -> Dict[str, Any]:
    """Attach observed wallet/token balance deltas; never treat instruction limits as fills."""
    if event.get("type") != "token_trade":
        return event
    if isinstance(tx_data, dict):
        result = tx_data.get("result", tx_data)
        meta = result.get("meta") or {}
    else:
        info = getattr(tx_data, "transaction", tx_data)
        meta = getattr(info, "meta", None)
    if not meta:
        event["fill_data_status"] = "DATA_BLOCKED: transaction metadata missing"
        return event

    def field(container: Any, camel: str, snake: str, default: Any = None) -> Any:
        if isinstance(container, dict):
            return container.get(camel, container.get(snake, default))
        return getattr(container, snake, default)

    def token_totals(name_camel: str, name_snake: str) -> Tuple[int, int]:
        total = 0
        decimals = 0
        for item in field(meta, name_camel, name_snake, []) or []:
            if field(item, "mint", "mint", "") != event.get("token"):
                continue
            if field(item, "owner", "owner", "") != event.get("wallet"):
                continue
            ui = field(item, "uiTokenAmount", "ui_token_amount", {}) or {}
            total += int(field(ui, "amount", "amount", 0) or 0)
            decimals = int(field(ui, "decimals", "decimals", decimals) or decimals)
        return total, decimals

    pre_token, pre_decimals = token_totals("preTokenBalances", "pre_token_balances")
    post_token, post_decimals = token_totals("postTokenBalances", "post_token_balances")
    token_delta = post_token - pre_token
    decimals = post_decimals or pre_decimals

    wallet = event.get("wallet")
    native_delta: Optional[int] = None
    if wallet in keys:
        index = keys.index(wallet)
        pre_native = field(meta, "preBalances", "pre_balances", []) or []
        post_native = field(meta, "postBalances", "post_balances", []) or []
        if index < len(pre_native) and index < len(post_native):
            native_delta = int(post_native[index]) - int(pre_native[index])
    fee = int(field(meta, "fee", "fee", 0) or 0)
    side = event.get("side")
    if native_delta is None or token_delta == 0:
        event["fill_data_status"] = "DATA_BLOCKED: owner balance deltas unavailable"
        return event
    notional_lamports = (
        max(0, -native_delta - fee) if side == "buy"
        else max(0, native_delta + fee) if side == "sell"
        else 0
    )
    token_amount_ui = abs(token_delta) / (10 ** decimals) if decimals >= 0 else 0
    event.update({
        "fill_data_status": "OBSERVED_WALLET_BALANCE_DELTA",
        "actual_token_delta_raw": token_delta,
        "actual_token_amount_ui": token_amount_ui,
        "token_decimals": decimals,
        "wallet_native_delta_lamports": native_delta,
        "network_fee_lamports": fee,
        "notional_sol": notional_lamports / 1_000_000_000,
        "price_sol_per_token": (notional_lamports / 1_000_000_000 / token_amount_ui) if token_amount_ui else None,
    })
    return event


def instruction_fields(instruction: Any) -> Tuple[int, List[int], bytes]:
    if isinstance(instruction, dict):
        data = instruction.get("data", "")
        return int(instruction.get("programIdIndex", -1)), list(instruction.get("accounts", [])), b58decode(data) if data else b""
    return int(instruction.program_id_index), list(instruction.accounts), bytes(instruction.data)


class PumpFunMonitor:
    PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    PUMP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
    DISCRIMINATORS = {
        hashlib.sha256(f"global:{name}".encode()).digest()[:8]: name
        for name in ("create", "create_v2", "buy", "buy_v2", "sell", "sell_v2", "migrate")
    }

    def __init__(self, yellowstone: YellowstoneClient, callback: Callable):
        self.yellowstone = yellowstone
        self.callback = callback
        self._seen: Set[Tuple[str, str]] = set()
        yellowstone.on("transaction", self._on_transaction)

    async def _on_transaction(self, tx_data: Any):
        received_ns = time.time_ns()
        keys, instructions, signature, slot = transaction_parts(tx_data)
        for instruction_index, instruction in enumerate(instructions):
            try:
                program_index, accounts, data = instruction_fields(instruction)
                if program_index < 0 or program_index >= len(keys) or len(data) < 8:
                    continue
                program = keys[program_index]
                name = self.DISCRIMINATORS.get(data[:8])
                if not name or program != self.PUMP_FUN_PROGRAM:
                    continue
                dedupe = (signature, instruction_index)
                if dedupe in self._seen:
                    continue
                self._seen.add(dedupe)
                if len(self._seen) > 100_000:
                    self._seen = set(list(self._seen)[-50_000:])
                event = self._decode_instruction(name, keys, accounts, data[8:], signature, slot)
                if event:
                    apply_event_timing(event, tx_data, received_ns)
                    enrich_trade_balances(event, tx_data, keys)
                    result = self.callback(event)
                    if asyncio.iscoroutine(result):
                        await result
            except (ValueError, IndexError, struct.error) as exc:
                logger.debug("Pump instruction parse rejected: %s", exc)

    def _decode_instruction(
        self, name: str, keys: List[str], accounts: List[int], payload: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        account = lambda index: keys[accounts[index]] if index < len(accounts) and accounts[index] < len(keys) else ""
        base = {"chain": "solana", "program": self.PUMP_FUN_PROGRAM, "timestamp": time.time(),
                "signature": signature, "slot": slot, "instruction": name}
        if name in {"create", "create_v2"}:
            token_name, symbol, uri, offset = self._parse_create_strings(payload)
            creator = b58encode(payload[offset:offset + 32]) if len(payload) >= offset + 32 else ""
            mint_index, curve_index = (0, 2) if name == "create" else (0, 2)
            return {
                **base,
                "type": "token_created",
                "token": account(mint_index),
                "bonding_curve": account(curve_index),
                "creator": creator,
                "name": token_name,
                "symbol": symbol,
                "uri": uri,
                "data_status": "OK" if creator else "DATA_BLOCKED",
            }
        if name in {"buy", "sell", "buy_v2", "sell_v2"}:
            v2 = name.endswith("_v2")
            mint = account(1 if v2 else 2)
            wallet = account(13 if v2 else 6)
            amount, limit_amount = struct.unpack_from("<QQ", payload, 0) if len(payload) >= 16 else (0, 0)
            return {
                **base,
                "type": "token_trade",
                "token": mint,
                "wallet": wallet,
                "side": "buy" if name.startswith("buy") else "sell",
                "token_amount": amount,
                "quote_limit_amount": limit_amount,
                "data_status": "OK" if mint and wallet and amount else "DATA_BLOCKED",
            }
        if name == "migrate":
            token = account(2)
            return {
                **base,
                "type": "token_migrated",
                "token": token,
                "pool": account(9),
                "wallet": account(5),
                "data_status": "OK" if token else "DATA_BLOCKED",
            }
        return None

    @staticmethod
    def _parse_create_strings(data: bytes) -> Tuple[str, str, str, int]:
        values: List[str] = []
        offset = 0
        for _ in range(3):
            if offset + 4 > len(data):
                raise ValueError("truncated Pump create string")
            length = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            if length > 4_096 or offset + length > len(data):
                raise ValueError("invalid Pump create string length")
            values.append(data[offset:offset + length].decode("utf-8", errors="replace"))
            offset += length
        return values[0], values[1], values[2], offset


class PumpSwapMonitor:
    """Decoder for the official PumpSwap AMM IDL.

    PumpSwap deliberately has its own account layouts even though its buy and
    sell discriminators match the legacy bonding-curve program. Keeping the
    decoders separate prevents a valid transaction from being assigned the
    wrong mint or wallet.
    """

    PUMP_AMM_PROGRAM = PumpFunMonitor.PUMP_AMM_PROGRAM
    DISCRIMINATORS = {
        bytes((233, 146, 209, 142, 207, 104, 64, 188)): "create_pool",
        bytes((102, 6, 61, 18, 1, 218, 235, 234)): "buy",
        bytes((51, 230, 133, 164, 1, 127, 131, 173)): "sell",
    }

    def __init__(self, yellowstone: YellowstoneClient, callback: Callable):
        self.yellowstone = yellowstone
        self.callback = callback
        self._seen: Set[Tuple[str, int]] = set()
        yellowstone.on("transaction", self._on_transaction)

    async def _on_transaction(self, tx_data: Any):
        received_ns = time.time_ns()
        keys, instructions, signature, slot = transaction_parts(tx_data)
        for instruction_index, instruction in enumerate(instructions):
            try:
                program_index, accounts, data = instruction_fields(instruction)
                if program_index < 0 or program_index >= len(keys) or len(data) < 8:
                    continue
                if keys[program_index] != self.PUMP_AMM_PROGRAM:
                    continue
                name = self.DISCRIMINATORS.get(data[:8])
                if not name:
                    continue
                dedupe = (signature, instruction_index)
                if dedupe in self._seen:
                    continue
                self._seen.add(dedupe)
                if len(self._seen) > 100_000:
                    self._seen = set(list(self._seen)[-50_000:])
                event = self._decode_instruction(name, keys, accounts, data[8:], signature, slot)
                if event:
                    apply_event_timing(event, tx_data, received_ns)
                    enrich_trade_balances(event, tx_data, keys)
                    result = self.callback(event)
                    if asyncio.iscoroutine(result):
                        await result
            except (ValueError, IndexError, struct.error) as exc:
                logger.debug("PumpSwap instruction parse rejected: %s", exc)

    @staticmethod
    def _decode_instruction(
        name: str, keys: List[str], accounts: List[int], payload: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        account = lambda index: keys[accounts[index]] if index < len(accounts) and accounts[index] < len(keys) else ""
        base = {
            "chain": "solana",
            "program": PumpSwapMonitor.PUMP_AMM_PROGRAM,
            "timestamp": time.time(),
            "signature": signature,
            "slot": slot,
            "instruction": name,
        }
        if name == "create_pool":
            if len(payload) < 18:
                raise ValueError("truncated PumpSwap create_pool payload")
            index, base_amount, quote_amount = struct.unpack_from("<HQQ", payload, 0)
            token = account(3)
            return {
                **base,
                "type": "pool_created",
                "pool": account(0),
                "creator": account(2),
                "token": token,
                "base_mint": token,
                "quote_mint": account(4),
                "pool_index": index,
                "initial_base_amount": base_amount,
                "initial_quote_amount": quote_amount,
                "data_status": "OK" if token and account(0) else "DATA_BLOCKED",
            }
        if name in {"buy", "sell"}:
            if len(payload) < 16:
                raise ValueError(f"truncated PumpSwap {name} payload")
            amount, quote_limit = struct.unpack_from("<QQ", payload, 0)
            token, wallet = account(3), account(1)
            return {
                **base,
                "type": "token_trade",
                "pool": account(0),
                "token": token,
                "quote_mint": account(4),
                "wallet": wallet,
                "side": name,
                "token_amount": amount,
                "quote_limit_amount": quote_limit,
                "data_status": "OK" if token and wallet and amount else "DATA_BLOCKED",
            }
        return None


class RaydiumMonitor:
    """Decode pool creation across the supported native Solana AMMs.

    The historical class name is retained for API compatibility. Every layout
    below is keyed by the deployed program and official Anchor IDL
    discriminator; unknown instructions are ignored rather than guessed.
    """

    RAYDIUM_AMM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    RAYDIUM_CPMM = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
    RAYDIUM_CLMM = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
    METEORA_DLMM = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo"
    METEORA_DYNAMIC_AMM = "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB"
    ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
    CPMM_INITIALIZE = bytes((175, 175, 109, 31, 13, 152, 155, 237))
    CLMM_CREATE_POOL = bytes((233, 146, 209, 142, 207, 104, 64, 188))
    METEORA_DLMM_INITIALIZE = bytes((45, 154, 237, 210, 221, 15, 166, 92))
    METEORA_DLMM_CUSTOM_INITIALIZE = bytes((46, 39, 41, 135, 111, 183, 200, 64))
    METEORA_AMM_INITIALIZERS = {
        bytes((118, 173, 41, 157, 173, 72, 97, 103)): (0, 2, 3, 17),
        bytes((6, 135, 68, 147, 229, 82, 169, 113)): (0, 2, 3, 17),
        bytes((7, 166, 138, 171, 206, 171, 236, 244)): (0, 3, 4, 18),
        bytes((48, 149, 220, 130, 61, 11, 9, 178)): (0, 3, 4, 18),
        bytes((145, 24, 172, 194, 219, 125, 3, 190)): (0, 2, 3, 17),
    }
    ORCA_INITIALIZE_POOL = bytes((95, 180, 10, 172, 84, 174, 232, 40))
    ORCA_INITIALIZE_POOL_V2 = bytes((207, 45, 87, 242, 27, 63, 204, 67))

    def __init__(self, yellowstone: YellowstoneClient, callback: Callable):
        self.yellowstone = yellowstone
        self.callback = callback
        self._seen: Set[Tuple[str, int]] = set()
        yellowstone.on("transaction", self._on_transaction)

    async def _on_transaction(self, tx_data: Any):
        received_ns = time.time_ns()
        keys, instructions, signature, slot = transaction_parts(tx_data)
        for instruction_index, instruction in enumerate(instructions):
            try:
                program_index, accounts, data = instruction_fields(instruction)
                if program_index < 0 or program_index >= len(keys) or not data:
                    continue
                program = keys[program_index]
                supported = {
                    self.RAYDIUM_AMM_V4, self.RAYDIUM_CPMM, self.RAYDIUM_CLMM,
                    self.METEORA_DLMM, self.METEORA_DYNAMIC_AMM, self.ORCA_WHIRLPOOL,
                }
                if program not in supported:
                    continue
                dedupe = (signature, instruction_index)
                if dedupe in self._seen:
                    continue
                self._seen.add(dedupe)
                if program == self.RAYDIUM_AMM_V4:
                    event = self._decode_v4(data[0], keys, accounts, data[1:], signature, slot)
                elif program == self.RAYDIUM_CPMM:
                    event = self._decode_cpmm(keys, accounts, data, signature, slot)
                elif program == self.RAYDIUM_CLMM:
                    event = self._decode_clmm(keys, accounts, data, signature, slot)
                elif program == self.METEORA_DLMM:
                    event = self._decode_meteora_dlmm(keys, accounts, data, signature, slot)
                elif program == self.METEORA_DYNAMIC_AMM:
                    event = self._decode_meteora_amm(keys, accounts, data, signature, slot)
                else:
                    event = self._decode_orca(keys, accounts, data, signature, slot)
                if event:
                    apply_event_timing(event, tx_data, received_ns)
                    result = self.callback(event)
                    if asyncio.iscoroutine(result):
                        await result
            except (IndexError, struct.error, ValueError):
                continue

    @staticmethod
    def _decode_v4(
        tag: int, keys: List[str], accounts: List[int], payload: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        account = lambda index: keys[accounts[index]] if index < len(accounts) and accounts[index] < len(keys) else ""
        base = {"chain": "solana", "program": RaydiumMonitor.RAYDIUM_AMM_V4,
                "signature": signature, "slot": slot, "timestamp": time.time(), "data_status": "OK"}
        if tag == 1 and len(payload) >= 25:  # Initialize2
            nonce, open_time, quote_amount, base_amount = struct.unpack_from("<BQQQ", payload, 0)
            return {
                **base,
                "type": "pool_created",
                "pool": account(4),
                "mint_a": account(8),
                "mint_b": account(9),
                "creator": account(17),
                "nonce": nonce,
                "open_time": open_time,
                "initial_quote_amount": quote_amount,
                "initial_base_amount": base_amount,
            }
        if tag in {9, 11} and len(payload) >= 16:  # SwapBaseIn / SwapBaseOut
            amount_a, amount_b = struct.unpack_from("<QQ", payload, 0)
            return {
                **base,
                "type": "pool_swap",
                "pool": account(1),
                "wallet": account(17),
                "swap_mode": "base_in" if tag == 9 else "base_out",
                "amount_specified": amount_a,
                "limit_amount": amount_b,
            }
        return None

    @staticmethod
    def _decode_cpmm(
        keys: List[str], accounts: List[int], data: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        """Decode Raydium CPMM initialize using the official Anchor IDL layout."""
        if not data.startswith(RaydiumMonitor.CPMM_INITIALIZE) or len(data) < 32:
            return None
        account = lambda index: keys[accounts[index]] if index < len(accounts) and accounts[index] < len(keys) else ""
        amount_0, amount_1, open_time = struct.unpack_from("<QQQ", data, 8)
        required = (account(3), account(4), account(5), account(0))
        return {
            "chain": "solana",
            "program": RaydiumMonitor.RAYDIUM_CPMM,
            "venue": "raydium_cpmm",
            "signature": signature,
            "slot": slot,
            "timestamp": time.time(),
            "type": "pool_created",
            "pool": account(3),
            "mint_a": account(4),
            "mint_b": account(5),
            "creator": account(0),
            "initial_base_amount": amount_0,
            "initial_quote_amount": amount_1,
            "open_time": open_time,
            "data_status": "OK" if all(required) and amount_0 and amount_1 else "DATA_BLOCKED",
        }

    @staticmethod
    def _decode_clmm(
        keys: List[str], accounts: List[int], data: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        """Decode Raydium CLMM create_pool using the official Anchor IDL layout."""
        if not data.startswith(RaydiumMonitor.CLMM_CREATE_POOL) or len(data) < 32:
            return None
        account = lambda index: keys[accounts[index]] if index < len(accounts) and accounts[index] < len(keys) else ""
        sqrt_price_x64 = int.from_bytes(data[8:24], "little")
        open_time = struct.unpack_from("<Q", data, 24)[0]
        required = (account(2), account(3), account(4), account(0))
        return {
            "chain": "solana",
            "program": RaydiumMonitor.RAYDIUM_CLMM,
            "venue": "raydium_clmm",
            "signature": signature,
            "slot": slot,
            "timestamp": time.time(),
            "type": "pool_created",
            "pool": account(2),
            "mint_a": account(3),
            "mint_b": account(4),
            "creator": account(0),
            "sqrt_price_x64": sqrt_price_x64,
            "open_time": open_time,
            "data_status": "OK" if all(required) and sqrt_price_x64 else "DATA_BLOCKED",
        }

    @staticmethod
    def _pool_event(
        program: str, venue: str, keys: List[str], accounts: List[int], signature: str, slot: int,
        pool_index: int, mint_a_index: int, mint_b_index: int, creator_index: int, **fields: Any,
    ) -> Dict[str, Any]:
        def account(index: int) -> str:
            return keys[accounts[index]] if index < len(accounts) and accounts[index] < len(keys) else ""

        required = (account(pool_index), account(mint_a_index), account(mint_b_index), account(creator_index))
        return {
            "chain": "solana", "program": program, "venue": venue, "signature": signature,
            "slot": slot, "timestamp": time.time(), "type": "pool_created",
            "pool": account(pool_index), "mint_a": account(mint_a_index),
            "mint_b": account(mint_b_index), "creator": account(creator_index),
            "data_status": "OK" if all(required) else "DATA_BLOCKED", **fields,
        }

    @staticmethod
    def _decode_meteora_dlmm(
        keys: List[str], accounts: List[int], data: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        if len(data) < 8 or data[:8] not in {
            RaydiumMonitor.METEORA_DLMM_INITIALIZE, RaydiumMonitor.METEORA_DLMM_CUSTOM_INITIALIZE,
        }:
            return None
        fields: Dict[str, Any] = {}
        if data[:8] == RaydiumMonitor.METEORA_DLMM_INITIALIZE and len(data) >= 14:
            fields["active_id"], fields["bin_step"] = struct.unpack_from("<iH", data, 8)
        return RaydiumMonitor._pool_event(
            RaydiumMonitor.METEORA_DLMM, "meteora_dlmm", keys, accounts, signature, slot,
            0, 2, 3, 8, **fields,
        )

    @staticmethod
    def _decode_meteora_amm(
        keys: List[str], accounts: List[int], data: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        layout = RaydiumMonitor.METEORA_AMM_INITIALIZERS.get(data[:8])
        if not layout:
            return None
        return RaydiumMonitor._pool_event(
            RaydiumMonitor.METEORA_DYNAMIC_AMM, "meteora_dynamic_amm", keys, accounts,
            signature, slot, *layout,
        )

    @staticmethod
    def _decode_orca(
        keys: List[str], accounts: List[int], data: bytes, signature: str, slot: int
    ) -> Optional[Dict[str, Any]]:
        if data.startswith(RaydiumMonitor.ORCA_INITIALIZE_POOL_V2) and len(data) >= 26:
            tick_spacing = struct.unpack_from("<H", data, 8)[0]
            sqrt_price_x64 = int.from_bytes(data[10:26], "little")
            layout = (6, 1, 2, 5)
        elif data.startswith(RaydiumMonitor.ORCA_INITIALIZE_POOL) and len(data) >= 27:
            tick_spacing = struct.unpack_from("<H", data, 9)[0]
            sqrt_price_x64 = int.from_bytes(data[11:27], "little")
            layout = (4, 1, 2, 3)
        else:
            return None
        return RaydiumMonitor._pool_event(
            RaydiumMonitor.ORCA_WHIRLPOOL, "orca_whirlpool", keys, accounts, signature, slot,
            *layout, tick_spacing=tick_spacing, sqrt_price_x64=sqrt_price_x64,
        )


def create_combined_subscription() -> SubscribeRequest:
    programs = [
        PumpFunMonitor.PUMP_FUN_PROGRAM,
        PumpSwapMonitor.PUMP_AMM_PROGRAM,
        RaydiumMonitor.RAYDIUM_AMM_V4,
        RaydiumMonitor.RAYDIUM_CPMM,
        RaydiumMonitor.RAYDIUM_CLMM,
        RaydiumMonitor.METEORA_DLMM,
        RaydiumMonitor.METEORA_DYNAMIC_AMM,
        RaydiumMonitor.ORCA_WHIRLPOOL,
    ]
    return SubscribeRequest(
        transactions={
            "memecoin_programs": TransactionFilter(accounts_include=programs, vote=False, failed=False)
        },
        commitment=SolanaCommitment.PROCESSED,
    )
