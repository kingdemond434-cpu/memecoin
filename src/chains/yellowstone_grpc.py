import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set
import base64

import grpc
from google.protobuf import descriptor_pb2

logger = logging.getLogger(__name__)


class SolanaCommitment(Enum):
    PROCESSED = "processed"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"


@dataclass
class AccountFilter:
    owner: Optional[str] = None
    memcmp: Optional[Dict] = None
    data_size: Optional[int] = None


@dataclass
class TransactionFilter:
    accounts_include: List[str] = field(default_factory=list)
    accounts_exclude: List[str] = field(default_factory=list)
    vote: bool = False
    failed: bool = False


@dataclass
class BlockFilter:
    include_transactions: bool = True
    include_accounts: bool = False
    include_entries: bool = False


@dataclass
class SubscribeRequest:
    accounts: Dict[str, AccountFilter] = field(default_factory=dict)
    transactions: Dict[str, TransactionFilter] = field(default_factory=dict)
    blocks: Dict[str, BlockFilter] = field(default_factory=dict)
    blocks_meta: bool = False
    slots: Dict[str, bool] = field(default_factory=dict)
    commitment: SolanaCommitment = SolanaCommitment.PROCESSED


class YellowstoneClient:
    def __init__(self, endpoint: str, x_token: str = ""):
        self.endpoint = endpoint
        self.x_token = x_token
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub = None
        self._stream = None
        self._running = False
        self._handlers: Dict[str, List[Callable]] = {}
        self._subscribe_request: Optional[SubscribeRequest] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_reconnect_delay = 1.0

    async def connect(self):
        self._channel = grpc.aio.secure_channel(
            self.endpoint,
            grpc.ssl_channel_credentials(),
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
            ]
        )
        
        try:
            from geyser_pb2_grpc import GeyserStub
            from geyser_pb2 import SubscribeRequest as GrpcSubscribeRequest
            self._stub = GeyserStub(self._channel)
            self._GrpcSubscribeRequest = GrpcSubscribeRequest
        except ImportError:
            logger.warning("geyser_pb2 not installed, using generic gRPC")
            self._stub = None

    async def subscribe(self, request: SubscribeRequest):
        self._subscribe_request = request
        self._running = True
        self._reconnect_attempts = 0
        asyncio.create_task(self._stream_loop())

    async def _stream_loop(self):
        while self._running:
            try:
                await self._establish_stream()
            except Exception as e:
                logger.error(f"Yellowstone stream error: {e}")
                if not await self._handle_reconnect():
                    break

    async def _establish_stream(self):
        if not self._stub:
            raise RuntimeError("Geyser stub not available. Install geyser-proto-py")
        
        grpc_request = self._build_grpc_request(self._subscribe_request)
        
        metadata = []
        if self.x_token:
            metadata.append(('x-token', self.x_token))
        
        self._stream = self._stub.Subscribe(grpc_request, metadata=metadata)
        
        async for response in self._stream:
            await self._handle_response(response)

    def _build_grpc_request(self, request: SubscribeRequest) -> Any:
        grpc_req = self._GrpcSubscribeRequest()
        
        if request.commitment == SolanaCommitment.PROCESSED:
            grpc_req.commitment = 0
        elif request.commitment == SolanaCommitment.CONFIRMED:
            grpc_req.commitment = 1
        else:
            grpc_req.commitment = 2
        
        for key, acc_filter in request.accounts.items():
            acc = grpc_req.accounts[key]
            if acc_filter.owner:
                acc.owner.append(acc_filter.owner)
            if acc_filter.memcmp:
                memcmp = acc.memcmp.add()
                memcmp.offset = acc_filter.memcmp.get("offset", 0)
                memcmp.bytes = base64.b64decode(acc_filter.memcmp.get("bytes", ""))
                if "encoding" in acc_filter.memcmp:
                    memcmp.encoding = acc_filter.memcmp["encoding"]
            if acc_filter.data_size:
                acc.data_size = acc_filter.data_size
        
        for key, tx_filter in request.transactions.items():
            tx = grpc_req.transactions[key]
            tx.account_include.extend(tx_filter.accounts_include)
            tx.account_exclude.extend(tx_filter.accounts_exclude)
            tx.vote = tx_filter.vote
            tx.failed = tx_filter.failed
        
        for key, block_filter in request.blocks.items():
            block = grpc_req.blocks[key]
            block.include_transactions = block_filter.include_transactions
            block.include_accounts = block_filter.include_accounts
            block.include_entries = block_filter.include_entries
        
        grpc_req.blocks_meta = request.blocks_meta
        
        for key, enabled in request.slots.items():
            grpc_req.slots[key] = enabled
        
        return grpc_req

    async def _handle_response(self, response: Any):
        if hasattr(response, 'account') and response.account:
            await self._dispatch('account', response.account)
        elif hasattr(response, 'transaction') and response.transaction:
            await self._dispatch('transaction', response.transaction)
        elif hasattr(response, 'block') and response.block:
            await self._dispatch('block', response.block)
        elif hasattr(response, 'block_meta') and response.block_meta:
            await self._dispatch('block_meta', response.block_meta)
        elif hasattr(response, 'slot') and response.slot:
            await self._dispatch('slot', response.slot)
        elif hasattr(response, 'entry') and response.entry:
            await self._dispatch('entry', response.entry)
        elif hasattr(response, 'ping') and response.ping:
            await self._dispatch('ping', response.ping)

    async def _dispatch(self, event_type: str, data: Any):
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Handler error for {event_type}: {e}")

    def on(self, event_type: str, handler: Callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def _handle_reconnect(self) -> bool:
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnect attempts reached")
            return False
        
        self._reconnect_attempts += 1
        delay = self._base_reconnect_delay * (2 ** (self._reconnect_attempts - 1))
        delay += asyncio.get_event_loop().time() % 1
        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)
        return True

    async def close(self):
        self._running = False
        if self._stream:
            self._stream.cancel()
        if self._channel:
            await self._channel.close()


class PumpFunMonitor:
    PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    PUMP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
    
    CREATE_IX_DISCRIMINATOR = bytes.fromhex("1c7c2c7c8c8c8c8c")
    BUY_IX_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
    SELL_IX_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")
    MIGRATE_IX_DISCRIMINATOR = bytes.fromhex("9c9c9c9c9c9c9c9c")

    def __init__(self, yellowstone: YellowstoneClient, callback: Callable):
        self.yellowstone = yellowstone
        self.callback = callback
        self._seen_signatures: Set[str] = set()
        self._setup_handlers()

    def _setup_handlers(self):
        self.yellowstone.on('transaction', self._on_transaction)

    async def _on_transaction(self, tx_data: Any):
        try:
            tx = tx_data.transaction.transaction
            message = tx.message
            
            for ix in message.instructions:
                program_id = message.account_keys[ix.program_id_index]
                
                if str(program_id) == self.PUMP_FUN_PROGRAM:
                    await self._process_pump_fun_ix(tx_data, ix, message)
                elif str(program_id) == self.PUMP_AMM_PROGRAM:
                    await self._process_pump_amm_ix(tx_data, ix, message)
                    
        except Exception as e:
            logger.debug(f"TX process error: {e}")

    async def _process_pump_fun_ix(self, tx_data: Any, ix: Any, message: Any):
        data = bytes(ix.data) if hasattr(ix, 'data') else b''
        if not data:
            return
        
        discriminator = data[:8]
        sig = self._get_signature(tx_data)
        
        if sig in self._seen_signatures:
            return
        self._seen_signatures.add(sig)
        if len(self._seen_signatures) > 100000:
            self._seen_signatures.clear()
        
        if discriminator == self.CREATE_IX_DISCRIMINATOR:
            await self._handle_create(tx_data, ix, message, data)
        elif discriminator == self.BUY_IX_DISCRIMINATOR:
            await self._handle_buy(tx_data, ix, message, data)
        elif discriminator == self.SELL_IX_DISCRIMINATOR:
            await self._handle_sell(tx_data, ix, message, data)

    async def _process_pump_amm_ix(self, tx_data: Any, ix: Any, message: Any):
        data = bytes(ix.data) if hasattr(ix, 'data') else b''
        if not data:
            return
        
        discriminator = data[:8]
        if discriminator == self.MIGRATE_IX_DISCRIMINATOR:
            await self._handle_migrate(tx_data, ix, message, data)

    def _get_signature(self, tx_data: Any) -> str:
        if hasattr(tx_data, 'signature'):
            return str(tx_data.signature)
        return ""

    def _get_account(self, message: Any, index: int) -> str:
        if hasattr(message, 'account_keys') and index < len(message.account_keys):
            return str(message.account_keys[index])
        return ""

    async def _handle_create(self, tx_data: Any, ix: Any, message: Any, data: bytes):
        try:
            mint = self._get_account(message, ix.accounts[0]) if ix.accounts else ""
            bonding_curve = self._get_account(message, ix.accounts[1]) if len(ix.accounts) > 1 else ""
            creator = self._get_account(message, ix.accounts[2]) if len(ix.accounts) > 2 else ""
            
            name, symbol, uri = self._parse_create_data(data[8:])
            
            await self.callback({
                "type": "token_created",
                "chain": "solana",
                "token": mint,
                "creator": creator,
                "bonding_curve": bonding_curve,
                "name": name,
                "symbol": symbol,
                "uri": uri,
                "timestamp": time.time(),
                "signature": self._get_signature(tx_data),
                "slot": getattr(tx_data, 'slot', 0)
            })
        except Exception as e:
            logger.error(f"Create parse error: {e}")

    async def _handle_buy(self, tx_data: Any, ix: Any, message: Any, data: bytes):
        try:
            mint = self._get_account(message, ix.accounts[0]) if ix.accounts else ""
            buyer = self._get_account(message, ix.accounts[1]) if len(ix.accounts) > 1 else ""
            amount = int.from_bytes(data[8:16], 'little') if len(data) >= 16 else 0
            sol_amount = int.from_bytes(data[16:24], 'little') if len(data) >= 24 else 0
            
            await self.callback({
                "type": "token_trade",
                "chain": "solana",
                "token": mint,
                "wallet": buyer,
                "side": "buy",
                "token_amount": amount,
                "sol_amount": sol_amount,
                "timestamp": time.time(),
                "signature": self._get_signature(tx_data),
                "slot": getattr(tx_data, 'slot', 0)
            })
        except Exception as e:
            logger.error(f"Buy parse error: {e}")

    async def _handle_sell(self, tx_data: Any, ix: Any, message: Any, data: bytes):
        try:
            mint = self._get_account(message, ix.accounts[0]) if ix.accounts else ""
            seller = self._get_account(message, ix.accounts[1]) if len(ix.accounts) > 1 else ""
            amount = int.from_bytes(data[8:16], 'little') if len(data) >= 16 else 0
            sol_amount = int.from_bytes(data[16:24], 'little') if len(data) >= 24 else 0
            
            await self.callback({
                "type": "token_trade",
                "chain": "solana",
                "token": mint,
                "wallet": seller,
                "side": "sell",
                "token_amount": amount,
                "sol_amount": sol_amount,
                "timestamp": time.time(),
                "signature": self._get_signature(tx_data),
                "slot": getattr(tx_data, 'slot', 0)
            })
        except Exception as e:
            logger.error(f"Sell parse error: {e}")

    async def _handle_migrate(self, tx_data: Any, ix: Any, message: Any, data: bytes):
        try:
            mint = self._get_account(message, ix.accounts[0]) if ix.accounts else ""
            
            await self.callback({
                "type": "token_migrated",
                "chain": "solana",
                "token": mint,
                "timestamp": time.time(),
                "signature": self._get_signature(tx_data),
                "slot": getattr(tx_data, 'slot', 0)
            })
        except Exception as e:
            logger.error(f"Migrate parse error: {e}")

    def _parse_create_data(self, data: bytes) -> Tuple[str, str, str]:
        try:
            offset = 0
            name_len = int.from_bytes(data[offset:offset+4], 'little')
            offset += 4
            name = data[offset:offset+name_len].decode('utf-8', errors='ignore')
            offset += name_len
            
            symbol_len = int.from_bytes(data[offset:offset+4], 'little')
            offset += 4
            symbol = data[offset:offset+symbol_len].decode('utf-8', errors='ignore')
            offset += symbol_len
            
            uri_len = int.from_bytes(data[offset:offset+4], 'little')
            offset += 4
            uri = data[offset:offset+uri_len].decode('utf-8', errors='ignore')
            
            return name, symbol, uri
        except Exception:
            return "", "", ""


class RaydiumMonitor:
    RAYDIUM_AMM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    RAYDIUM_CPMM = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
    RAYDIUM_CLMM = "CAMMCzo5YL8w4VSA8KLpfqZrUQfM6NWzrgZt5Jg3u6R"

    def __init__(self, yellowstone: YellowstoneClient, callback: Callable):
        self.yellowstone = yellowstone
        self.callback = callback
        self._seen_pools: Set[str] = set()
        self.yellowstone.on('transaction', self._on_transaction)

    async def _on_transaction(self, tx_data: Any):
        try:
            tx = tx_data.transaction.transaction
            message = tx.message
            
            for ix in message.instructions:
                program_id = str(message.account_keys[ix.program_id_index])
                
                if program_id in [self.RAYDIUM_AMM_V4, self.RAYDIUM_CPMM, self.RAYDIUM_CLMM]:
                    await self._process_raydium_ix(tx_data, ix, message, program_id)
        except Exception as e:
            logger.debug(f"Raydium TX error: {e}")

    async def _process_raydium_ix(self, tx_data: Any, ix: Any, message: Any, program_id: str):
        data = bytes(ix.data) if hasattr(ix, 'data') else b''
        if len(data) < 8:
            return
        
        discriminator = data[:8]
        
        if discriminator == bytes.fromhex("09d5a4e6c5a4c5a4"):
            await self._handle_initialize_pool(tx_data, ix, message, data, program_id)
        elif discriminator == bytes.fromhex("a5a5a5a5a5a5a5a5"):
            await self._handle_swap(tx_data, ix, message, data, program_id)

    async def _handle_initialize_pool(self, tx_data: Any, ix: Any, message: Any, data: bytes, program_id: str):
        try:
            pool = self._get_account(message, ix.accounts[0]) if ix.accounts else ""
            mint_a = self._get_account(message, ix.accounts[1]) if len(ix.accounts) > 1 else ""
            mint_b = self._get_account(message, ix.accounts[2]) if len(ix.accounts) > 2 else ""
            creator = self._get_account(message, ix.accounts[3]) if len(ix.accounts) > 3 else ""
            
            if pool in self._seen_pools:
                return
            self._seen_pools.add(pool)
            
            await self.callback({
                "type": "pool_created",
                "chain": "solana",
                "pool": pool,
                "program": program_id,
                "mint_a": mint_a,
                "mint_b": mint_b,
                "creator": creator,
                "timestamp": time.time(),
                "signature": self._get_signature(tx_data),
                "slot": getattr(tx_data, 'slot', 0)
            })
        except Exception as e:
            logger.error(f"Pool init parse error: {e}")

    async def _handle_swap(self, tx_data: Any, ix: Any, message: Any, data: bytes, program_id: str):
        pass

    def _get_account(self, message: Any, index: int) -> str:
        if hasattr(message, 'account_keys') and index < len(message.account_keys):
            return str(message.account_keys[index])
        return ""

    def _get_signature(self, tx_data: Any) -> str:
        if hasattr(tx_data, 'signature'):
            return str(tx_data.signature)
        return ""


def create_pump_fun_subscription() -> SubscribeRequest:
    return SubscribeRequest(
        transactions={
            "pump_fun": TransactionFilter(
                accounts_include=[PumpFunMonitor.PUMP_FUN_PROGRAM],
                vote=False,
                failed=False
            ),
            "pump_amm": TransactionFilter(
                accounts_include=[PumpFunMonitor.PUMP_AMM_PROGRAM],
                vote=False,
                failed=False
            )
        },
        commitment=SolanaCommitment.PROCESSED
    )


def create_raydium_subscription() -> SubscribeRequest:
    return SubscribeRequest(
        transactions={
            "raydium_v4": TransactionFilter(
                accounts_include=[RaydiumMonitor.RAYDIUM_AMM_V4],
                vote=False,
                failed=False
            ),
            "raydium_cpmm": TransactionFilter(
                accounts_include=[RaydiumMonitor.RAYDIUM_CPMM],
                vote=False,
                failed=False
            ),
            "raydium_clmm": TransactionFilter(
                accounts_include=[RaydiumMonitor.RAYDIUM_CLMM],
                vote=False,
                failed=False
            )
        },
        commitment=SolanaCommitment.PROCESSED
    )


def create_combined_subscription() -> SubscribeRequest:
    req = SubscribeRequest(commitment=SolanaCommitment.PROCESSED)
    req.transactions.update(create_pump_fun_subscription().transactions)
    req.transactions.update(create_raydium_subscription().transactions)
    return req