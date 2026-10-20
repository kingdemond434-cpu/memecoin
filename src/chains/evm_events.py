"""Strict EVM event decoding and the shared PIT chain-event envelope.

The PairCreated layout comes from Uniswap's published factory interface:
https://github.com/Uniswap/v2-core/blob/master/contracts/interfaces/IUniswapV2Factory.sol
Indexed tokens occupy topics 1/2; the pair and ordinal occupy data words 0/1.
These are observation records and never grant execution authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from eth_utils import keccak

from src.collectors.event_source import Event, SourceClass


NATIVE_EVENT_SCHEMA_VERSION = "1"
PAIR_CREATED_ABI_SOURCE = "https://github.com/Uniswap/v2-core/blob/master/contracts/interfaces/IUniswapV2Factory.sol"
PAIR_CREATED_SIGNATURE = "PairCreated(address,address,address,uint256)"
PAIR_CREATED_TOPIC = "0x" + keccak(text=PAIR_CREATED_SIGNATURE).hex()


def hex_data(value: Any, length: Optional[int] = None) -> bytes:
    if isinstance(value, bytes):
        decoded = bytes(value)
    elif isinstance(value, str) and value.startswith("0x") and len(value) % 2 == 0:
        decoded = bytes.fromhex(value[2:])
    else:
        raise ValueError("native data must be 0x-prefixed bytes")
    if length is not None and len(decoded) != length:
        raise ValueError("native data length does not match the reviewed ABI")
    return decoded


def native_address(value: Any) -> str:
    return "0x" + hex_data(value, 20).hex()


def quantity(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an EVM quantity")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.startswith("0x"):
        number = int(value, 16)
    else:
        raise ValueError("native quantity must be an integer or hex quantity")
    if number < 0:
        raise ValueError("native quantity is negative")
    return number


def _address_word(word: bytes) -> str:
    if len(word) != 32 or any(word[:12]):
        raise ValueError("noncanonical ABI address word")
    return "0x" + word[-20:].hex()


@dataclass(frozen=True)
class PairCreated:
    token0: str
    token1: str
    pair: str
    pair_count: int


def decode_pair_created(log: Mapping[str, Any]) -> Optional[PairCreated]:
    """Unknown topic/layout/removed log yields no tradeable interpretation."""
    try:
        topics = log.get("topics")
        if log.get("removed") or not isinstance(topics, (list, tuple)) or len(topics) != 3:
            return None
        if hex_data(topics[0], 32).hex() != PAIR_CREATED_TOPIC[2:]:
            return None
        data = hex_data(log.get("data"), 64)
        token0 = _address_word(hex_data(topics[1], 32))
        token1 = _address_word(hex_data(topics[2], 32))
        pair = _address_word(data[:32])
        count = int.from_bytes(data[32:], "big")
        # The official factory sorts nonzero tokens and emits a positive count.
        if int(token0, 16) == 0 or int(token0, 16) >= int(token1, 16) or int(pair, 16) == 0 or count < 1:
            return None
        return PairCreated(token0, token1, pair, count)
    except (ValueError, TypeError, OverflowError):
        return None


def select_quote_pair(event: PairCreated, base_tokens: Sequence[str]) -> Optional[Tuple[str, str]]:
    bases = {native_address(token) for token in base_tokens}
    a, b = event.token0 in bases, event.token1 in bases
    if a == b:  # neither known quote, or both quotes: do not invent a new token
        return None
    return (event.token1, event.token0) if a else (event.token0, event.token1)


@dataclass(frozen=True)
class NativeChainEvent:
    """Canonical chain observation retaining native identity and PIT clocks.

    ``source_at`` is native block time; ``observed_at`` is this desk's first
    receipt time. Replays retain their actual receipt time instead of claiming
    that historical data was available at the historical block timestamp.
    ``native`` is retained in the existing Event lake, not a new raw store.
    """

    chain: str
    chain_id: str
    chain_family: str
    protocol: Optional[str]
    protocol_version: Optional[str]
    event_type: str
    block_number: int
    block_hash: str
    transaction_hash: Optional[str]
    event_index: Optional[int]
    source_at: float
    observed_at: float
    finality: str
    source_ref: str
    decoder_ref: Optional[str] = None
    token_addresses: Tuple[str, ...] = ()
    pool_address: Optional[str] = None
    actor_addresses: Tuple[str, ...] = ()
    native: Dict[str, Any] = field(default_factory=dict)
    reverts_event_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in (self.chain, self.chain_id, self.chain_family,
                                                    self.event_type, self.source_ref)):
            raise ValueError("native event identity and provenance are required")
        if self.block_number < 0 or not self.block_hash:
            raise ValueError("native block identity is required")
        for name in ("source_at", "observed_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("native PIT clocks must be measured finite timestamps")
        if self.finality not in ("FINALIZED", "SAFE", "UNCONFIRMED", "REVERTED"):
            raise ValueError("unknown native finality semantics")
        if self.event_type == "REORG" and not self.reverts_event_id:
            raise ValueError("a reorg correction must identify the observation it retracts")

    @property
    def event_id(self) -> str:
        identity = [self.chain_family, self.chain_id, self.block_hash, self.transaction_hash,
                    self.event_index, self.event_type, self.reverts_event_id]
        return "native-" + hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row.update(schema_version=NATIVE_EVENT_SCHEMA_VERSION, event_id=self.event_id,
                   observation_only=True, execution_eligible=False,
                   data_status="UNKNOWN" if self.event_type == "UNKNOWN" else "OBSERVED")
        return row

    def to_event(self) -> Event:
        """Use the desk's existing canonical Event/lake interface."""
        return Event(
            source_id=f"chain:{self.chain_id}:{self.protocol or 'unknown'}", source_class=SourceClass.CHAIN,
            source_at=self.source_at, observed_at=self.observed_at,
            text=f"{self.event_type}:{self.event_id}",
            token_addresses=self.token_addresses, entity_ids=self.actor_addresses,
            urls=(self.source_ref,), metadata=self.to_dict())
