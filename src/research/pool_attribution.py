"""Recover observed PumpSwap trade identities without inventing old knowledge.

Raw orphan events survive restart. Pool ownership/layout and mint decimals are
read through the existing RPC manager, then canonically retained before use.
Late attribution goes to research/history, never backwards into the hot price.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import re
from pathlib import Path
import sqlite3
import threading
import time

from src.chains.pumpswap_route import PUMPSWAP_PROGRAM, parse_pool
from src.chains.pump_route import TOKEN_PROGRAM, TOKEN_2022_PROGRAM, WSOL_MINT
from src.chains.rpc_manager import RPCError
from src.research.data_miners import Enriches, MinerSpec

MINER = 'chain:pool_attribution'
# Solana's getMultipleAccounts protocol limit, not a lifetime source ceiling.
RPC_ACCOUNT_BATCH = 100
logger = logging.getLogger(__name__)


def mint_decimals(value):
    if not isinstance(value, dict) or value.get('owner') not in {TOKEN_PROGRAM, TOKEN_2022_PROGRAM}:
        return None
    try:
        data = base64.b64decode(value['data'][0], validate=True)
    except (ValueError, KeyError, TypeError):
        return None
    if value['owner'] == TOKEN_PROGRAM and len(data) != 82:
        return None
    if value['owner'] == TOKEN_2022_PROGRAM and len(data) != 82 and not (len(data) >= 166 and data[165] == 1):
        return None
    if len(data) < 82 or data[45] != 1 or int.from_bytes(data[:4], 'little') not in {0, 1} or int.from_bytes(data[46:50], 'little') not in {0, 1}:
        return None
    return data[44]


def bind_trade(event, mapping, *, available_at):
    if (event.get('program') != PUMPSWAP_PROGRAM or event.get('pool') != mapping['pool']
            or mapping.get('status') != 'OWNER_AND_LAYOUT_VERIFIED'):
        raise ValueError('pool identity does not match native event')
    if (not math.isfinite(float(available_at)) or not math.isfinite(float(mapping['observed_at']))
            or available_at < mapping['observed_at']):
        raise ValueError('pool identity was unavailable at observation time')
    decimals = mapping.get('base_decimals')
    amount = event.get('token_amount')
    ui = amount / 10 ** decimals if type(amount) is int and type(decimals) is int else None
    notional = event.get('quote_amount_gross')
    sol = notional / 1e9 if type(notional) is int and mapping['quote_mint'] == WSOL_MINT else None
    return {**event, 'token': mapping['base_mint'], 'quote_mint': mapping['quote_mint'],
            'actual_token_amount_ui': ui, 'notional_sol': sol,
            'price_sol_per_token': sol / ui if sol is not None and ui else None,
            'identity_available_at': mapping['observed_at'], 'available_at': available_at,
            'identity_source_ref': mapping.get('source_ref'),
            'data_status': 'OBSERVED_PROGRAM_EVENT_WITH_VERIFIED_POOL_IDENTITY'}


class PoolAttribution:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=3)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,pool TEXT,body TEXT,first_seen REAL,ref TEXT,derived_ref TEXT);
          CREATE INDEX IF NOT EXISTS pending_ref ON events(ref);
          CREATE INDEX IF NOT EXISTS pending_pool ON events(pool);
          CREATE TABLE IF NOT EXISTS identities(pool TEXT PRIMARY KEY,body TEXT NOT NULL,ref TEXT);
          CREATE TABLE IF NOT EXISTS jobs(pool TEXT PRIMARY KEY,due REAL,last_error TEXT,attempts INTEGER);
          CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY,value INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS capabilities(name TEXT PRIMARY KEY,value INTEGER NOT NULL,observed REAL NOT NULL);
        ''')
        self._identities = {pool: {**json.loads(body), 'source_ref': ref}
                            for pool, body, ref in self.db.execute('SELECT pool,body,ref FROM identities')}
        self._delivery_tasks = set()
        self.delivery_failures = 0
        self.capture_failures = 0
        row = self.db.execute("SELECT value,observed FROM capabilities WHERE name='account_batch'").fetchone()
        self._batch_limit, self._batch_observed = (int(row[0]), float(row[1])) if row else (RPC_ACCOUNT_BATCH, 0.)

    def lookup(self, pool):
        with self.lock:
            return self._identities.get(pool)

    def _capture(self, event):
        if event.get('program') != PUMPSWAP_PROGRAM or not event.get('pool') or not event.get('signature'):
            raise ValueError('unattributed event lacks native pool provenance')
        identity = {k: event.get(k) for k in ('signature', 'pool', 'slot', 'wallet', 'side',
                    'token_amount', 'quote_amount_gross', 'program_event_hash', 'log_index')}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        raw = json.dumps(event, allow_nan=False)
        with self.lock, self.db:
            self.db.execute('INSERT OR IGNORE INTO events VALUES(?,?,?,?,NULL,NULL)', (key, event['pool'], raw, time.time()))
            self.db.execute('INSERT OR IGNORE INTO jobs VALUES(?,0,\'\',0)', (event['pool'],))

    async def capture(self, event):
        task = asyncio.create_task(asyncio.to_thread(self._capture, dict(event)))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        except Exception:
            self.capture_failures += 1
            raise

    def _remember_capacity(self, limit, observed):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO capabilities VALUES('account_batch',?,?)", (limit, observed))
            self._batch_limit, self._batch_observed = limit, observed

    async def _read_accounts(self, rpc, addresses, rows):
        while True:
            try:
                result = await rpc.request('getMultipleAccounts', [addresses, {'encoding': 'base64', 'commitment': 'confirmed'}])
                if (len(addresses) > self._batch_limit and isinstance(result, dict)
                        and isinstance(result.get('value'), list) and len(result['value']) == len(addresses)):
                    # A later successful larger probe releases the old bound.
                    # 100 remains the protocol ceiling, not a provider promise.
                    self._remember_capacity(RPC_ACCOUNT_BATCH, time.time())
                return result, addresses
            except RPCError as exc:
                error = exc.error if isinstance(exc.error, dict) else {}
                message = str(error.get('message', ''))
                match = re.search(r'\bmaximum\s+of\s+(\d+)\s+accounts?\b', message, re.IGNORECASE)
                limit = int(match[1]) if match and error.get('code') == -32602 else 0
                if not 0 < limit < len(addresses):
                    raise
                # Never turn HTTP/JSON-RPC rate limits into split retries.
                observed = time.time()
                self._remember_capacity(limit, observed)
                rows.append({'provider': 'native_pool_attribution', 'attribution_kind': 'provider_capacity',
                    'method': 'getMultipleAccounts', 'account_limit': limit, 'observed_at': observed,
                    'error_code': -32602, 'error_message_sha256': hashlib.sha256(message.encode()).hexdigest(),
                    'execution_eligible': False})
                addresses = addresses[:limit]  # the remainder stays durably queued

    async def fetch(self, rpc):
        now = time.time()
        with self.lock:
            pending = list(self.db.execute('SELECT id,body,first_seen FROM events WHERE ref IS NULL ORDER BY first_seen LIMIT 250'))
            recovered = list(self.db.execute("SELECT e.id,e.body,e.first_seen,e.ref FROM events e JOIN identities i ON e.pool=i.pool WHERE e.ref IS NOT NULL AND json_extract(i.body,'$.base_decimals') IS NOT NULL ORDER BY e.first_seen LIMIT 250"))
            # Re-probe capacity periodically so a learned free-provider bound
            # does not permanently cap a later, more capable endpoint.
            batch_limit = RPC_ACCOUNT_BATCH if now - self._batch_observed >= 3600 else self._batch_limit
            pools = [row[0] for row in self.db.execute("SELECT j.pool FROM jobs j LEFT JOIN identities i ON j.pool=i.pool WHERE (i.pool IS NULL OR json_extract(i.body,'$.base_decimals') IS NULL) AND j.due<=? ORDER BY j.due,j.pool LIMIT ?", (now, batch_limit))]
        rows = [{'provider': 'native_pool_attribution', 'attribution_kind': 'raw_orphan',
                 'event_id': key, 'original_event': json.loads(body), 'first_seen': at,
                 'available_at': now, 'execution_eligible': False} for key, body, at in pending]
        for key, body, first_seen, ref in recovered:
            event = json.loads(body)
            mapping = self.lookup(event['pool'])
            rows.append({'provider': 'native_pool_attribution', 'attribution_kind': 'recovered_trade',
                'event_id': key, 'original_source_ref': ref, 'original_first_seen': first_seen,
                'event': bind_trade(event, mapping, available_at=now), 'observed_at': now,
                'execution_eligible': False, 'late_attribution': True})
        if not pools:
            return rows
        try:
            result, pools = await self._read_accounts(rpc, pools, rows)
            values = result.get('value') if isinstance(result, dict) else None
            if not isinstance(values, list) or len(values) != len(pools):
                raise ValueError('pool response alignment unavailable')
            valid = []
            for pool, value in zip(pools, values):
                try:
                    if not value or value.get('owner') != PUMPSWAP_PROGRAM:
                        raise ValueError('pool owner unavailable or different')
                    state = parse_pool(base64.b64decode(value['data'][0], validate=True), pool)
                    if not state.ok:
                        raise ValueError('pool layout unavailable')
                    valid.append((pool, value, state))
                except (ValueError, KeyError, TypeError):
                    rows.append(self.gap(pool, 'OWNER_OR_LAYOUT_UNVERIFIED', now))
            mints = list(dict.fromkeys(s.base_mint for _, _, s in valid))
            decimals, mint_proofs, mint_context = {}, {}, None
            if mints:
                try:
                    mint_result, mints = await self._read_accounts(rpc, mints, rows)
                    mint_context = mint_result.get('context') if isinstance(mint_result, dict) else None
                    mint_values = mint_result.get('value') if isinstance(mint_result, dict) else None
                    if isinstance(mint_values, list) and len(mint_values) == len(mints):
                        for mint, value in zip(mints, mint_values):
                            decimals[mint], mint_proofs[mint] = mint_decimals(value), value
                except Exception:
                    pass  # mint identities remain valid, UI amounts unavailable
            for pool, proof, state in valid:
                rows.append({'provider': 'native_pool_attribution', 'attribution_kind': 'identity',
                    'pool': pool, 'base_mint': state.base_mint, 'quote_mint': state.quote_mint,
                    'base_decimals': decimals.get(state.base_mint), 'creator': state.creator,
                    'coin_creator': state.coin_creator, 'observed_at': time.time(),
                    'status': 'OWNER_AND_LAYOUT_VERIFIED', 'pool_account_proof': proof,
                    'base_mint_proof': mint_proofs.get(state.base_mint),
                    'base_mint_slot_context': mint_context if state.base_mint in mint_proofs else None,
                    'slot_context': result.get('context'), 'execution_eligible': False})
        except Exception as exc:
            rows.extend(self.gap(pool, 'RPC_' + type(exc).__name__, now) for pool in pools)
        return rows

    @staticmethod
    def gap(pool, reason, now):
        return {'provider': 'native_pool_attribution', 'attribution_kind': 'mapping_gap',
                'pool': pool, 'reason': reason, 'observed_at': now, 'execution_eligible': False}

    def acknowledge(self, records, ref):
        if not isinstance(ref, str) or not ref:
            raise ValueError('canonical attribution reference required')
        updates = {}
        with self.lock, self.db:
            for row in records:
                kind = row['attribution_kind']
                if kind == 'raw_orphan':
                    self.db.execute('UPDATE events SET ref=coalesce(ref,?) WHERE id=?', (ref, row['event_id']))
                elif kind == 'identity':
                    body = json.dumps(row, allow_nan=False)
                    self.db.execute('INSERT OR REPLACE INTO identities VALUES(?,?,?)', (row['pool'], body, ref))
                    if row.get('base_decimals') is None:
                        self.db.execute("UPDATE jobs SET due=?,last_error='MINT_DECIMALS_UNAVAILABLE',attempts=attempts+1 WHERE pool=?",
                                        (row['observed_at'] + 60, row['pool']))
                    else:
                        self.db.execute('DELETE FROM jobs WHERE pool=?', (row['pool'],))
                    updates[row['pool']] = {**row, 'source_ref': ref}
                elif kind == 'mapping_gap':
                    self.db.execute('UPDATE jobs SET due=?,last_error=?,attempts=attempts+1 WHERE pool=?',
                                    (row['observed_at'] + 60, row['reason'], row['pool']))
                elif kind == 'recovered_trade':
                    self.db.execute('UPDATE events SET derived_ref=coalesce(derived_ref,?) WHERE id=? AND ref IS NOT NULL', (ref, row['event_id']))
            # Publish in-memory identities only after the durable transaction
            # commits; a failed write cannot teach the live decoder a mapping.
        with self.lock:
            self._identities.update(updates)

    def delivered(self, identities):
        """Only the actual owner-loop consumer may release a retained outbox row."""
        with self.lock, self.db:
            for key in identities:
                deleted = self.db.execute('DELETE FROM events WHERE id=? AND derived_ref IS NOT NULL', (key,)).rowcount
                if deleted:
                    self.db.execute("INSERT INTO counters VALUES('recovered',1) ON CONFLICT(name) DO UPDATE SET value=value+1")

    def queue_delivery_ack(self, identities):
        task = asyncio.create_task(asyncio.to_thread(self.delivered, tuple(identities)))
        self._delivery_tasks.add(task)
        def done(value):
            self._delivery_tasks.discard(value)
            if value.cancelled() or value.exception() is not None:
                self.delivery_failures += 1
                logger.warning('pool attribution delivery ACK failed; durable outbox retained')
        task.add_done_callback(done)

    async def finish_delivery_acks(self):
        if self._delivery_tasks:
            await asyncio.gather(*tuple(self._delivery_tasks), return_exceptions=True)

    def register(self, pool, rpc):
        async def fetch():
            return await self.fetch(rpc)
        return {MINER: pool.register(MinerSpec(MINER, Enriches.WALLET_HISTORY, 15,
                  endpoint='rpc:getMultipleAccounts', detail='durable orphan recovery; owner/layout verified pool and mint facts'), fetch)}

    def report(self):
        with self.lock:
            pending = self.db.execute('SELECT count(*),min(first_seen) FROM events').fetchone()
            recovered = self.db.execute("SELECT value FROM counters WHERE name='recovered'").fetchone()
            gaps = self.db.execute("SELECT count(*) FROM jobs WHERE last_error<>''").fetchone()[0]
        return {'status': 'OBSERVATIONAL_ATTRIBUTION', 'resolved_pools': len(self._identities),
                'pending_events': pending[0], 'oldest_pending_age_seconds': time.time()-pending[1] if pending[1] else None,
                'mapping_gaps': gaps, 'recovered_events': recovered[0] if recovered else 0,
                'capture_failures': self.capture_failures, 'delivery_ack_failures': self.delivery_failures,
                'pending_delivery_acks': len(self._delivery_tasks),
                'pool_read_batch_limit': self._batch_limit, 'protocol_account_limit': RPC_ACCOUNT_BATCH,
                'historical_loss_before_capture': 'UNRECOVERED; NO_BACKDATED_COVERAGE', 'execution_eligible': False}

    def close(self):
        with self.lock:
            self.db.close()
