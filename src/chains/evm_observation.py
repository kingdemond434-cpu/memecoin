"""Bounded, observation-only EVM backfill over the existing RPC manager.

No task, transport, signer or trading callback is created. The caller's
research loop polls finalized/safe blocks and persists NativeChainEvent in
the existing canonical lake before this transactional checkpoint advances.
Delivery is at least once with stable event IDs. A sink must deduplicate those
IDs; a provider/sink failure leaves the block available for the next pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence
import uuid

from src.chains.evm_events import (
    NativeChainEvent, decode_pair_created,
    hex_data, native_address, quantity,
)


@dataclass(frozen=True)
class FactoryContract:
    protocol_id: str
    address: str
    version: str = "UNKNOWN"
    decoder: str = "UNKNOWN"
    abi_ref: Optional[str] = None
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        native_address(self.address)
        if not self.protocol_id or not self.version:
            raise ValueError("factory protocol identity/version are required")
        if self.decoder not in ("UNKNOWN", "uniswap_v2_pair_created"):
            raise ValueError("unreviewed native decoder")
        if self.decoder != "UNKNOWN" and not (self.abi_ref and self.source_ref):
            raise ValueError("a known decoder requires source and official ABI provenance")


class EVMObservationAdapter:
    """Native finality/checkpoint index sharing the research SQLite file."""

    def __init__(self, rpc: Any, *, chain_name: str, chain_id: Any,
                 factories: Sequence[FactoryContract], start_block: int,
                 checkpoint_path: Any = "data/research/frontier.sqlite",
                 finality_tag: str = "finalized", max_blocks_per_pass: int = 8,
                 max_logs_per_block: int = 1000, max_reorg_depth: int = 64,
                 max_reorg_events: int = 1000, lease_seconds: float = 120):
        if finality_tag not in ("finalized", "safe"):
            raise ValueError("observation requires native finalized/safe semantics; no latest fallback")
        self.rpc = rpc
        self.chain_name = str(chain_name)
        self.chain_id = str(quantity(chain_id)) if isinstance(chain_id, int) else str(chain_id)
        if not self.chain_name or not self.chain_id.isdecimal():
            raise ValueError("EVM chain identity must be explicit")
        self.factories = {native_address(spec.address): spec for spec in factories}
        if not self.factories or len(self.factories) != len(factories):
            raise ValueError("explicit distinct factory contracts are required")
        self.start_block = quantity(start_block)
        for name, value, bound in (("max_blocks_per_pass", max_blocks_per_pass, 1000),
                                   ("max_logs_per_block", max_logs_per_block, 10000),
                                   ("max_reorg_depth", max_reorg_depth, 1000),
                                   ("max_reorg_events", max_reorg_events, 10000)):
            if isinstance(value, bool) or int(value) != value or not 1 <= value <= bound:
                raise ValueError(f"{name} must be a bounded positive integer")
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("observation pass lease must be bounded")
        self.finality_tag = finality_tag
        self.max_blocks_per_pass = int(max_blocks_per_pass)
        self.max_logs_per_block = int(max_logs_per_block)
        self.max_reorg_depth = int(max_reorg_depth)
        self.max_reorg_events = int(max_reorg_events)
        self.lease_seconds = float(lease_seconds)
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        contract_identity = [vars(spec) for spec in sorted(factories, key=lambda item: item.address)]
        self.adapter_id = hashlib.sha256(json.dumps(
            [self.chain_id, finality_tag, self.start_block, contract_identity], sort_keys=True).encode()).hexdigest()
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS evm_observer_state(
                    adapter_id TEXT PRIMARY KEY, chain_id TEXT NOT NULL, next_block INTEGER NOT NULL,
                    last_height INTEGER, last_hash TEXT, status TEXT NOT NULL,
                    observed_events INTEGER NOT NULL, unknown_events INTEGER NOT NULL,
                    reorgs INTEGER NOT NULL, updated_at REAL, lease_token TEXT, lease_until REAL);
                CREATE TABLE IF NOT EXISTS evm_observer_blocks(
                    adapter_id TEXT NOT NULL, height INTEGER NOT NULL, hash TEXT NOT NULL,
                    parent_hash TEXT NOT NULL, source_at REAL NOT NULL, canonical INTEGER NOT NULL,
                    PRIMARY KEY(adapter_id,height,hash));
                CREATE UNIQUE INDEX IF NOT EXISTS evm_canonical_height
                    ON evm_observer_blocks(adapter_id,height) WHERE canonical=1;
                CREATE TABLE IF NOT EXISTS evm_observer_refs(
                    adapter_id TEXT NOT NULL, height INTEGER NOT NULL, block_hash TEXT NOT NULL,
                    event_id TEXT NOT NULL, evidence_ref TEXT NOT NULL, canonical INTEGER NOT NULL,
                    PRIMARY KEY(adapter_id,event_id));
                CREATE TABLE IF NOT EXISTS evm_observer_reversions(
                    adapter_id TEXT NOT NULL, event_id TEXT NOT NULL, evidence_ref TEXT NOT NULL,
                    PRIMARY KEY(adapter_id,event_id));
                CREATE TABLE IF NOT EXISTS evm_observer_pending_refs(
                    adapter_id TEXT NOT NULL, event_id TEXT NOT NULL, height INTEGER NOT NULL,
                    block_hash TEXT NOT NULL, source_at REAL NOT NULL, evidence_ref TEXT NOT NULL,
                    status TEXT NOT NULL, PRIMARY KEY(adapter_id,event_id));
            """)
            db.execute("INSERT OR IGNORE INTO evm_observer_state VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (self.adapter_id, self.chain_id, self.start_block, None, None,
                        "NOT_RUN", 0, 0, 0, None, None, None))

    @classmethod
    def from_registry(cls, registry: Any, chain_name: str, *, start_block: int,
                      factory_contracts: Optional[Sequence[FactoryContract]] = None,
                      **kwargs: Any) -> "EVMObservationAdapter":
        config, rpc = registry.get_chain(chain_name), registry.get_rpc(chain_name)
        if config is None or rpc is None or getattr(config.chain_type, "value", None) != "evm":
            raise ValueError("a started existing EVM ChainRegistry client is required")
        # A configured address does not establish ABI/source parity. Retain
        # UNKNOWN unless a versioned reviewed contract is supplied explicitly.
        contracts = factory_contracts if factory_contracts is not None else [
            FactoryContract(name, address) for name, address in config.factories.items()]
        return cls(rpc, chain_name=chain_name, chain_id=config.chain_id,
                   factories=contracts, start_block=start_block, **kwargs)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def report(self) -> Dict[str, Any]:
        with self._db() as db:
            row = dict(db.execute("SELECT * FROM evm_observer_state WHERE adapter_id=?",
                                  (self.adapter_id,)).fetchone())
        row.pop("lease_token", None)
        row.pop("lease_until", None)
        total = row["observed_events"]
        row.update(chain=self.chain_name, finality_tag=self.finality_tag,
                   unknown_event_rate=row["unknown_events"] / total if total else None,
                   observation_only=True, execution_eligible=False, evidence_scope="LOCAL_OBSERVATION_INDEX")
        return row

    def _claim(self, at: float) -> Optional[str]:
        token = uuid.uuid4().hex
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT lease_until FROM evm_observer_state WHERE adapter_id=?",
                             (self.adapter_id,)).fetchone()
            if row[0] is not None and row[0] > at:
                return None
            db.execute("UPDATE evm_observer_state SET lease_token=?,lease_until=? WHERE adapter_id=?",
                       (token, at + self.lease_seconds, self.adapter_id))
        return token

    def _owner(self, db: sqlite3.Connection, token: str) -> None:
        row = db.execute("SELECT lease_token,lease_until FROM evm_observer_state WHERE adapter_id=?",
                         (self.adapter_id,)).fetchone()
        if row[0] != token or row[1] <= time.time():
            raise PermissionError("observation pass lease expired")

    async def _block(self, height: Any) -> Mapping[str, Any]:
        tag = hex(height) if isinstance(height, int) else height
        block = await self.rpc.request("eth_getBlockByNumber", [tag, False])
        if not isinstance(block, Mapping):
            raise ValueError("native block unavailable")
        number = quantity(block.get("number"))
        if isinstance(height, int) and number != height:
            raise ValueError("native block height mismatch")
        hex_data(block.get("hash"), 32)
        hex_data(block.get("parentHash"), 32)
        quantity(block.get("timestamp"))
        return block

    async def _initialize_anchor(self, token: str, head: Mapping[str, Any]) -> None:
        with self._db() as db:
            state = db.execute("SELECT last_height FROM evm_observer_state WHERE adapter_id=?",
                               (self.adapter_id,)).fetchone()
        if state[0] is not None or self.start_block == 0 or quantity(head["number"]) < self.start_block:
            return
        # A measured predecessor lets a reorg of the first requested block
        # rewind safely too; no source timestamp is manufactured from parentHash.
        anchor = await self._block(self.start_block - 1)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owner(db, token)
            db.execute("INSERT OR IGNORE INTO evm_observer_blocks VALUES(?,?,?,?,?,1)",
                       (self.adapter_id, self.start_block - 1, str(anchor["hash"]).lower(),
                        str(anchor["parentHash"]).lower(), float(quantity(anchor["timestamp"]))))
            db.execute("UPDATE evm_observer_state SET last_height=?,last_hash=? WHERE adapter_id=?",
                       (self.start_block - 1, str(anchor["hash"]).lower(), self.adapter_id))

    def _pending_ref(self, token: str, event: NativeChainEvent, reference: str) -> None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owner(db, token)
            db.execute("INSERT OR IGNORE INTO evm_observer_pending_refs VALUES(?,?,?,?,?,?,?)",
                       (self.adapter_id, event.event_id, event.block_number, event.block_hash,
                        event.source_at, reference, "PENDING"))

    async def _reconcile_pending(self, token: str, sink: Callable) -> int:
        # A block may have persisted some evidence before its provider/sink
        # failed. Retain those references independently of block completion
        # so a later reorg can retract the partial batch after restart.
        with self._db() as db:
            rows = db.execute("SELECT * FROM evm_observer_pending_refs WHERE adapter_id=? AND status='PENDING' "
                              "ORDER BY height,event_id LIMIT ?",
                              (self.adapter_id, self.max_reorg_events + 1)).fetchall()
        if len(rows) > self.max_reorg_events:
            raise ValueError("pending correction exceeds this pass's evidence budget")
        current_headers = {}
        corrected = []
        for row in rows:
            if row["height"] not in current_headers:
                current_headers[row["height"]] = await self._block(row["height"])
            current = current_headers[row["height"]]
            if str(current["hash"]).lower() == row["block_hash"]:
                continue
            correction = NativeChainEvent(
                chain=self.chain_name, chain_id=self.chain_id, chain_family="EVM", protocol=None,
                protocol_version=None, event_type="REORG", block_number=row["height"],
                block_hash=row["block_hash"], transaction_hash=None, event_index=None,
                source_at=row["source_at"], observed_at=time.time(), finality="REVERTED",
                source_ref=row["evidence_ref"], reverts_event_id=row["event_id"],
                native={"reverted_evidence_ref": row["evidence_ref"],
                        "replacement_head_hash": current["hash"], "partial_batch": True})
            corrected.append((row["event_id"], correction.event_id, await self._persist(sink, correction)))
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owner(db, token)
            for old_id, correction_id, reference in corrected:
                db.execute("UPDATE evm_observer_pending_refs SET status='REVERTED' WHERE adapter_id=? AND event_id=?",
                           (self.adapter_id, old_id))
                db.execute("INSERT OR IGNORE INTO evm_observer_reversions VALUES(?,?,?)",
                           (self.adapter_id, correction_id, reference))
        return len(corrected)

    @staticmethod
    async def _persist(sink: Callable, event: NativeChainEvent) -> str:
        reference = sink(event)
        if inspect.isawaitable(reference):
            reference = await reference
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("canonical evidence sink did not acknowledge a durable reference")
        return reference

    def _event(self, log: Mapping[str, Any], header: Mapping[str, Any],
               observed_at: float) -> NativeChainEvent:
        address = native_address(log.get("address"))
        factory = self.factories.get(address)
        if factory is None:
            raise ValueError("provider returned an unrequested event emitter")
        if log.get("removed") or str(log.get("blockHash", "")).lower() != str(header["hash"]).lower():
            raise ValueError("removed or reorg-raced native log")
        if quantity(log.get("blockNumber")) != quantity(header["number"]):
            raise ValueError("log height differs from its native block")
        tx_hash = "0x" + hex_data(log.get("transactionHash"), 32).hex()
        event_index = quantity(log.get("logIndex"))
        pair = decode_pair_created(log) if factory.decoder == "uniswap_v2_pair_created" else None
        native = {"log": dict(log), "block_parent_hash": header["parentHash"],
                  "factory_address": address, "configured_protocol_identity": factory.protocol_id,
                  "protocol_identity_status": "OBSERVATION_ONLY",
                  "decoder_status": "DECODED_OFFICIAL_ABI" if pair else "UNKNOWN_REFUSED",
                  "abi_source": factory.source_ref}
        if pair:
            native["pair_created"] = vars(pair)
        return NativeChainEvent(
            chain=self.chain_name, chain_id=self.chain_id, chain_family="EVM",
            protocol=factory.protocol_id, protocol_version=factory.version,
            event_type="POOL_CREATED" if pair else "UNKNOWN",
            block_number=quantity(header["number"]), block_hash=str(header["hash"]).lower(),
            transaction_hash=tx_hash, event_index=event_index,
            source_at=float(quantity(header["timestamp"])), observed_at=observed_at,
            finality=self.finality_tag.upper(),
            source_ref=f"chain:{self.chain_id}:block:{header['hash']}:log:{event_index}",
            decoder_ref=factory.abi_ref if pair else None,
            token_addresses=(pair.token0, pair.token1) if pair else (),
            pool_address=pair.pair if pair else None, native=native)

    async def _reconcile(self, token: str, sink: Callable, head: Mapping[str, Any]) -> int:
        """Retract orphaned lake observations before replaying replacements."""
        with self._db() as db:
            state = db.execute("SELECT * FROM evm_observer_state WHERE adapter_id=?",
                               (self.adapter_id,)).fetchone()
            if state["last_height"] is None:
                return 0
            if quantity(head["number"]) < state["last_height"]:
                raise ValueError("native finality head regressed")
            blocks = db.execute("SELECT * FROM evm_observer_blocks WHERE adapter_id=? AND canonical=1 "
                                "ORDER BY height DESC LIMIT ?", (self.adapter_id, self.max_reorg_depth)).fetchall()
        ancestor = None
        for block in blocks:
            current = await self._block(block["height"])
            if str(current["hash"]).lower() == block["hash"]:
                ancestor = block
                break
        if ancestor is not None and ancestor["height"] == state["last_height"]:
            return 0
        if ancestor is None:
            # With no common stored block, never guess the old source time or
            # throw away unseen orphan references. An explicit deeper repair
            # is required, leaving the index and canonical evidence intact.
            raise ValueError("reorg exceeds retained common-ancestor evidence")
        with self._db() as db:
            orphans = db.execute("SELECT r.*,b.source_at FROM evm_observer_refs r "
                                 "JOIN evm_observer_blocks b ON b.adapter_id=r.adapter_id "
                                 "AND b.height=r.height AND b.hash=r.block_hash "
                                 "WHERE r.adapter_id=? AND r.canonical=1 AND r.height>? "
                                 "ORDER BY r.height,r.event_id LIMIT ?",
                                 (self.adapter_id, ancestor["height"], self.max_reorg_events + 1)).fetchall()
        if len(orphans) > self.max_reorg_events:
            raise ValueError("reorg correction exceeds this pass's event budget")
        corrections = []
        for orphan in orphans:
            event = NativeChainEvent(
                chain=self.chain_name, chain_id=self.chain_id, chain_family="EVM", protocol=None,
                protocol_version=None, event_type="REORG", block_number=orphan["height"],
                block_hash=orphan["block_hash"], transaction_hash=None, event_index=None,
                source_at=orphan["source_at"], observed_at=time.time(), finality="REVERTED",
                source_ref=orphan["evidence_ref"], reverts_event_id=orphan["event_id"],
                native={"reverted_evidence_ref": orphan["evidence_ref"],
                        "replacement_head_hash": head["hash"], "common_ancestor_hash": ancestor["hash"]})
            corrections.append((event.event_id, await self._persist(sink, event)))
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owner(db, token)
            db.execute("UPDATE evm_observer_blocks SET canonical=0 WHERE adapter_id=? AND height>?",
                       (self.adapter_id, ancestor["height"]))
            db.execute("UPDATE evm_observer_refs SET canonical=0 WHERE adapter_id=? AND height>?",
                       (self.adapter_id, ancestor["height"]))
            db.executemany("INSERT OR IGNORE INTO evm_observer_reversions VALUES(?,?,?)",
                           [(self.adapter_id, event_id, ref) for event_id, ref in corrections])
            db.execute("UPDATE evm_observer_state SET next_block=?,last_height=?,last_hash=?,reorgs=reorgs+1 "
                       "WHERE adapter_id=?", (ancestor["height"] + 1, ancestor["height"],
                                               ancestor["hash"], self.adapter_id))
        return len(corrections)

    async def poll_once(self, sink: Callable[[NativeChainEvent], Any]) -> Dict[str, Any]:
        """Observe at most the declared block/log budgets; no trading path."""
        token = self._claim(time.time())
        if token is None:
            return {**self.report(), "pass_status": "LEASED_BY_ANOTHER_PASS"}
        processed = emitted = unknown = reverted = 0
        status = "OBSERVED"
        failure_type = None
        try:
            actual_chain = quantity(await self.rpc.request("eth_chainId", []))
            if str(actual_chain) != self.chain_id:
                raise ValueError("provider chain differs from configured native chain")
            head = await self._block(self.finality_tag)
            await self._initialize_anchor(token, head)
            reverted = await self._reconcile_pending(token, sink)
            reverted += await self._reconcile(token, sink, head)
            state = self.report()
            end = min(quantity(head["number"]), state["next_block"] + self.max_blocks_per_pass - 1)
            previous_hash = state["last_hash"]
            for height in range(state["next_block"], end + 1):
                header = await self._block(height)
                if previous_hash is not None and str(header["parentHash"]).lower() != previous_hash:
                    raise ValueError("native parent continuity failed")
                logs = await self.rpc.request("eth_getLogs", [{
                    "blockHash": header["hash"], "address": list(self.factories)}])
                if not isinstance(logs, list) or len(logs) > self.max_logs_per_block:
                    raise ValueError("provider log response exceeds declared block budget")
                references = []
                seen = set()
                block_unknown = 0
                received_at = time.time()
                for log in logs:
                    event = self._event(log, header, received_at)
                    if event.event_id in seen:
                        continue
                    seen.add(event.event_id)
                    reference = await self._persist(sink, event)
                    self._pending_ref(token, event, reference)
                    references.append((event, reference))
                    block_unknown += event.event_type == "UNKNOWN"
                # A provider can reorganize between the log and header calls.
                # Such evidence remains provisional in the lake, with an
                # explicit correction; the checkpoint never advances blindly.
                confirmed = await self._block(height)
                if str(confirmed["hash"]).lower() != str(header["hash"]).lower():
                    for event, reference in references:
                        correction = NativeChainEvent(
                            chain=self.chain_name, chain_id=self.chain_id, chain_family="EVM", protocol=None,
                            protocol_version=None, event_type="REORG", block_number=height,
                            block_hash=event.block_hash, transaction_hash=None, event_index=None,
                            source_at=event.source_at, observed_at=time.time(), finality="REVERTED",
                            source_ref=reference, reverts_event_id=event.event_id,
                            native={"reverted_evidence_ref": reference, "replacement_head_hash": confirmed["hash"]})
                        await self._persist(sink, correction)
                    with self._db() as db:
                        db.execute("UPDATE evm_observer_pending_refs SET status='REVERTED' "
                                   "WHERE adapter_id=? AND height=? AND block_hash=?",
                                   (self.adapter_id, height, str(header["hash"]).lower()))
                    raise ValueError("native block changed during observation")
                with self._db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    self._owner(db, token)
                    block_hash = str(header["hash"]).lower()
                    db.execute("INSERT OR REPLACE INTO evm_observer_blocks VALUES(?,?,?,?,?,1)",
                               (self.adapter_id, height, block_hash, str(header["parentHash"]).lower(),
                                float(quantity(header["timestamp"]))))
                    db.executemany("INSERT OR IGNORE INTO evm_observer_refs VALUES(?,?,?,?,?,1)", [
                        (self.adapter_id, height, block_hash, event.event_id, reference)
                        for event, reference in references])
                    db.execute("UPDATE evm_observer_pending_refs SET status='COMMITTED' "
                               "WHERE adapter_id=? AND height=? AND block_hash=?",
                               (self.adapter_id, height, block_hash))
                    db.execute("UPDATE evm_observer_state SET next_block=?,last_height=?,last_hash=?,"
                               "observed_events=observed_events+?,unknown_events=unknown_events+?,updated_at=? "
                               "WHERE adapter_id=?", (height + 1, height, block_hash,
                                                       len(references), block_unknown, time.time(), self.adapter_id))
                previous_hash = str(header["hash"]).lower()
                processed += 1
                emitted += len(references)
                unknown += block_unknown
            status = "CAUGHT_UP" if end >= quantity(head["number"]) else "BACKFILL_PENDING"
        except Exception as exc:
            # Provider exception text may contain a credential-bearing URL.
            status, failure_type = "DATA_BLOCKED", type(exc).__name__
        finally:
            with self._db() as db:
                db.execute("UPDATE evm_observer_state SET status=?,updated_at=?,lease_token=NULL,lease_until=NULL "
                           "WHERE adapter_id=? AND lease_token=?", (status, time.time(), self.adapter_id, token))
        return {**self.report(), "pass_status": status, "processed_blocks": processed,
                "emitted_events": emitted, "unknown_events_this_pass": unknown,
                "reverted_events_this_pass": reverted, "failure_type": failure_type}
