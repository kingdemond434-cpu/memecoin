"""Incremental launch-operator research and paper watch plans; no RPC or signer.

Relationships are evidenced associations, never authenticated insider identities.
Price excursions are descriptive benchmarks, never executable trade returns.
"""
from __future__ import annotations
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import threading

from src.research.actor_store import ActorStore, EdgeKind, native_node

KINDS = {'OPERATOR_ANALYSIS', 'OPERATOR_WATCH', 'OPERATOR_FUNDING',
         'OPERATOR_LAUNCH', 'OPERATOR_TRADE'}
HORIZONS = (2, 5, 10, 30, 60)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name+' requires an explicit finite measurement')
    return float(value)


def required(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(key+' required')
    return value.strip()


def references(values):
    if not isinstance(values, (tuple, list)) or not values or len(values) > 256 or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError('canonical evidence references required')
    return sorted(set(values))


class OperatorPlayService:
    def __init__(self, path=None, *, actor_store=None):
        self._temporary = tempfile.TemporaryDirectory(prefix='operator-play-') if path is None else None
        self.path = Path(path) if path is not None else Path(self._temporary.name)/'index.sqlite'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.actor_store = actor_store if actor_store is not None else ActorStore()
        self.lock = threading.RLock()
        with closing(self._connect()) as db, db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS receipts(id TEXT PRIMARY KEY,seal TEXT NOT NULL,result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS watches(id TEXT PRIMARY KEY,funder TEXT NOT NULL,registered REAL NOT NULL,body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS watched_funder ON watches(funder);
                CREATE TABLE IF NOT EXISTS armed(id TEXT PRIMARY KEY,watch TEXT NOT NULL,wallet TEXT NOT NULL,armed_at REAL NOT NULL,closed INTEGER NOT NULL DEFAULT 0,body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS armed_wallet ON armed(wallet,closed);
                CREATE TABLE IF NOT EXISTS consumed(arm TEXT NOT NULL,trigger TEXT NOT NULL,PRIMARY KEY(arm,trigger));
                CREATE TABLE IF NOT EXISTS plans(token TEXT PRIMARY KEY,id TEXT NOT NULL,created REAL NOT NULL,body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trades(id TEXT PRIMARY KEY,token TEXT NOT NULL,event_at REAL NOT NULL,available REAL NOT NULL,
                    eligible INTEGER NOT NULL,quote TEXT NOT NULL,price REAL NOT NULL,tx INTEGER,ix INTEGER,body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS trade_window ON trades(token,event_at,available);
                CREATE INDEX IF NOT EXISTS eligible_entry ON trades(token,eligible,event_at,available);
            ''')
            self._has_watches = db.execute('SELECT 1 FROM watches LIMIT 1').fetchone() is not None
            self._has_armed = db.execute('SELECT 1 FROM armed WHERE closed=0 LIMIT 1').fetchone() is not None

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        return db

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    @staticmethod
    def _node(p, value, entity='wallet'):
        family = required(p, 'chain_family').upper()
        if family not in {'SOLANA', 'EVM'}:
            raise ValueError('operator adapter requires explicit SOLANA or EVM semantics')
        if entity not in {'wallet', 'contract', 'token'}:
            raise ValueError('explicit wallet, contract or token entity type required')
        return native_node(required(p, 'chain_id'), entity, value, family)

    def consume(self, kind, payload, refs, now):
        try:
            if kind not in KINDS:
                raise ValueError('unsupported operator work kind')
            p = dict(payload)
            p['chain_id'] = required(p, 'chain_id').lower()
            p['chain_family'] = required(p, 'chain_family').upper()
            if len(encoded(p).encode()) > 131072:
                raise ValueError('compact operator fact exceeds operation budget')
            now = number(now, 'now')
            refs = references(refs)
            event = number(p.get('event_at'), 'event_at')
            available = number(p.get('observed_at'), 'observed_at')
            if not event <= available <= now:
                raise ValueError('event <= observation <= processing clock required')
            self._node(p, 'validation')
            key = identity([kind, p['chain_id'], required(p, 'observation_id')])
            seal = identity([p, refs])
            with self.lock, closing(self._connect()) as db, db:
                db.execute('BEGIN IMMEDIATE')
                old = db.execute('SELECT seal,result FROM receipts WHERE id=?', (key,)).fetchone()
                if old:
                    if old[0] != seal:
                        raise ValueError('operator observation identity conflicts; explicit correction required')
                    return json.loads(old[1])
                detail, children = getattr(self, '_'+kind.removeprefix('OPERATOR_').lower())(db, p, refs, now)
                result = {'disposition': 'OBSERVED', 'detail': {**detail, 'live_authority': False,
                          'strategy_family': 'launch_operator_genealogy'}, 'children': children}
                db.execute('INSERT INTO receipts VALUES(?,?,?)', (key, seal, encoded(result)))
                if kind in {'OPERATOR_WATCH', 'OPERATOR_FUNDING', 'OPERATOR_LAUNCH'}:
                    self._has_watches = db.execute('SELECT 1 FROM watches LIMIT 1').fetchone() is not None
                    self._has_armed = db.execute('SELECT 1 FROM armed WHERE closed=0 LIMIT 1').fetchone() is not None
                return result
        except (ValueError, KeyError, TypeError) as exc:
            return {'disposition': 'DATA_BLOCKED', 'detail': {'reason': str(exc), 'live_authority': False},
                    'enabling_change': 'operator-contract:'+kind, 'children': []}

    def _analysis(self, db, p, refs, now):
        moment = number(p.get('as_of'), 'as_of')
        if moment > now:
            raise ValueError('future analysis cutoff')
        seed = self._node(p, required(p, 'address'), p.get('entity_type', 'wallet'))
        creator_count, creator_node, creator_offset = 0, None, p.get('creator_offset', 0)
        if type(creator_offset) is not int or creator_offset < 0:
            raise ValueError('valid creator cursor required')
        creator_refs = set()
        if p.get('entity_type') == 'token':
            creators, creator_count = self.actor_store.observed_prefix(seed, EdgeKind.DEPLOYED, moment,
                limit=1, offset=creator_offset)
            if creators:
                creator_node = creators[0].source
                creator_refs.update(creators[0].evidence_refs)
        walk = self.actor_store.family(creator_node or seed, moment)
        launches, buyers, used_refs = {}, Counter(), set(refs)
        used_refs.update(creator_refs)
        nodes = sorted(n for n in walk.reached if n.startswith('native:') and json.loads(n[7:])[0] == p['chain_id'])
        node_offset, edge_offset = p.get('node_offset', 0), p.get('edge_offset', 0)
        limit = p.get('page_size', 128)
        if (any(type(v) is not int or v < 0 for v in (node_offset, edge_offset))
                or type(limit) is not int or not 1 <= limit <= 1024):
            raise ValueError('valid report cursor and page size required')
        counts = [self.actor_store.out_degree(n, EdgeKind.DEPLOYED, moment) for n in nodes]
        remaining, cursor = limit, None
        for index in range(node_offset, len(nodes)):
            node = nodes[index]
            offset = edge_offset if index == node_offset else 0
            edges, total = self.actor_store.observed_prefix(node, EdgeKind.DEPLOYED, moment,
                limit=remaining, offset=offset, outgoing=True)
            for edge in edges:
                item = launches.setdefault(edge.target, {'token_node': edge.target, 'deployers': [],
                    'event_at': edge.event_at, 'available_at': edge.observed_at, 'evidence_refs': []})
                item['deployers'].append(node)
                item['evidence_refs'] = sorted(set(item['evidence_refs']) | set(edge.evidence_refs))
                used_refs.update(edge.evidence_refs)
            remaining -= len(edges)
            if remaining == 0:
                cursor = ({'node_offset': index, 'edge_offset': offset+len(edges)}
                          if offset+len(edges) < total else
                          {'node_offset': index+1, 'edge_offset': 0} if index+1 < len(nodes) else None)
                break
        # Only the indexed observed prefix is examined; it is not claimed to
        # be blockchain transaction order or a complete early-buyer census.
        for token, item in launches.items():
            edges, count = self.actor_store.observed_prefix(token, EdgeKind.BOUGHT, moment, limit=25)
            cohort = sorted({e.source for e in edges})
            buyers.update(cohort)
            item.update(observed_buyers=cohort, available_buy_observations=count,
                        buyer_prefix_complete=count <= 25,
                        benchmark=self._benchmark(db, token, moment))
            for edge in edges:
                used_refs.update(edge.evidence_refs)
        ordered = sorted(launches.values(), key=lambda row: (row['available_at'], row['token_node']))
        detail = {'as_of': moment, 'processed_at': now, 'seed': seed,
            'association_semantics': 'OBSERVED_FUNDING_NETWORK_NOT_OWNER_IDENTITY',
            'resolved_creator_node': creator_node, 'observed_creator_edges': creator_count,
            'next_creator_offset': creator_offset+1 if creator_offset+1 < creator_count else None,
            'coverage': 'OBSERVED_ONLY', 'full_history_verified': False,
            'truncated': walk.truncated, 'hubs_skipped': walk.hubs_skipped,
            'known_deployment_observations_in_walk': sum(counts), 'known_launches_in_page': len(ordered),
            'launches': ordered, 'next_cursor': cursor,
            'recurrence_scope': 'CURRENT_REPORT_PAGE_OBSERVED_BUYER_PREFIXES',
            'recurring_observed_buyers': {k:v for k,v in sorted(buyers.items()) if v > 1},
            'evidence_refs': sorted(used_refs), 'forecast_probability': None,
            'executable_edge_verified': False}
        children = []
        continuation = dict(cursor) if cursor else (
            {'creator_offset': creator_offset+1, 'node_offset': 0, 'edge_offset': 0}
            if creator_offset+1 < creator_count else None)
        if continuation is not None:
            body = {**p, **continuation}
            body['observation_id'] = identity(['operator-report-page', p['observation_id'], continuation])
            children.append({'kind':'OPERATOR_ANALYSIS','payload':body,'axes':{'chain':p['chain_id']}})
        return detail, children

    def _watch(self, db, p, refs, now):
        funder = self._node(p, required(p, 'funder'), p.get('funder_type', 'wallet'))
        low, high = p.get('minimum_raw', 1), p.get('maximum_raw')
        if type(low) is not int or low < 1 or (high is not None and (type(high) is not int or high < low)):
            raise ValueError('funding bounds use positive native integers, never copied ETH/SOL floats')
        fresh = p.get('freshness', 'ANY')
        trigger = p.get('trigger', 'CREATION')
        if fresh not in {'ANY', 'VERIFIED_FIRST_TRANSACTION', 'HAS_HISTORY'} or trigger not in {'CREATION', 'BUY', 'BOTH'}:
            raise ValueError('explicit wallet-history and trigger policy required')
        for field in ('first_buy_only', 'first_launch_only'):
            if type(p.get(field, True)) is not bool:
                raise ValueError(field+' requires a boolean')
        policy = {'funder': funder, 'minimum_raw': low, 'maximum_raw': high, 'freshness': fresh,
            'trigger': trigger, 'first_buy_only': p.get('first_buy_only', True),
            'first_launch_only': p.get('first_launch_only', True),
            'exit_policy': p.get('exit_policy', {}), 'execution_authority': 'PAPER_PLAN_ONLY'}
        # Exit settings are research scenarios, not parameters sent to a venue.
        allowed = {'take_profit_bps', 'stop_loss_bps', 'trail_activation_bps', 'trail_drawdown_bps', 'timeout_seconds'}
        if not isinstance(policy['exit_policy'], dict) or set(policy['exit_policy'])-allowed:
            raise ValueError('unknown exit scenario setting')
        for field, value in policy['exit_policy'].items():
            number(value, field, positive=True)
        watch_id = identity(['operator-watch-v1', policy])
        old = db.execute('SELECT registered FROM watches WHERE id=?', (watch_id,)).fetchone()
        if not old:
            db.execute('INSERT INTO watches VALUES(?,?,?,?)',
                (watch_id, funder, now, encoded({**policy, 'evidence_refs': refs})))
        return {'watch_id': watch_id, 'registered_at': old[0] if old else now,
                'status': 'WATCH_REGISTERED', 'policy': policy}, []

    def _funding(self, db, p, refs, now):
        source = self._node(p, required(p, 'source'), p.get('source_type', 'wallet'))
        wallet = self._node(p, required(p, 'target'), p.get('target_type', 'wallet'))
        amount = p.get('amount_raw')
        if type(amount) is not int or amount <= 0:
            raise ValueError('positive raw native funding amount required')
        expected = 'lamports' if p['chain_family'].upper() == 'SOLANA' else 'wei'
        if p.get('amount_unit') != expected:
            raise ValueError('chain-specific raw funding unit mismatch')
        if source == wallet:
            raise ValueError('self transfer is not a new funded-wallet watch')
        armed, reasons = [], []
        for watch_id, registered, body in db.execute('SELECT id,registered,body FROM watches WHERE funder=?', (source,)):
            policy = json.loads(body)
            if registered > p['event_at']:
                reasons.append('FUNDING_PREDATES_WATCH'); continue
            if amount < policy['minimum_raw'] or (policy['maximum_raw'] is not None and amount > policy['maximum_raw']):
                continue
            fresh = p.get('wallet_history', 'UNKNOWN')
            if policy['freshness'] != 'ANY':
                if fresh != policy['freshness']:
                    reasons.append('WALLET_HISTORY_UNVERIFIED'); continue
                references(p.get('wallet_history_evidence_refs'))
            arm_id = identity([watch_id, p['observation_id'], wallet])
            data = {'arm_id': arm_id, 'watch_id': watch_id, 'wallet': wallet,
                'armed_at': now, 'funding_event_at': p['event_at'], 'funding_available_at': p['observed_at'],
                'amount_raw': amount, 'amount_unit': expected, 'policy': policy, 'evidence_refs': refs}
            db.execute('INSERT INTO armed(id,watch,wallet,armed_at,body) VALUES(?,?,?,?,?)',
                       (arm_id, watch_id, wallet, now, encoded(data)))
            armed.append(data)
        return {'status': 'PAPER_WATCH_ARMED' if armed else 'NO_ELIGIBLE_WATCH',
                'armed': armed, 'unavailable_reasons': sorted(set(reasons))}, []

    def _launch(self, db, p, refs, now):
        wallet = self._node(p, required(p, 'wallet'), p.get('wallet_type', 'wallet'))
        token = self._node(p, required(p, 'token'), 'token')
        trigger = p.get('trigger', 'CREATION')
        if trigger not in {'CREATION', 'BUY'}:
            raise ValueError('known creation or buy observation required')
        plans, missed = [], []
        rows = db.execute('SELECT id,armed_at,body FROM armed WHERE wallet=? AND closed=0', (wallet,)).fetchall()
        for arm_id, armed_at, body in rows:
            arm = json.loads(body); policy = arm['policy']
            if policy['trigger'] not in {trigger, 'BOTH'}:
                continue
            if db.execute('SELECT 1 FROM consumed WHERE arm=? AND trigger=?', (arm_id, trigger)).fetchone():
                continue
            if p['event_at'] < armed_at:
                missed.append({'arm_id': arm_id, 'reason': 'LAUNCH_PREDATES_ACTUAL_ARMING'})
            else:
                plan_id = identity(['operator-paper-plan-v1', token])
                plan = {'plan_id': plan_id, 'token_node': token, 'wallet_node': wallet,
                    'arm_id': arm_id, 'created_at': now, 'signal_event_at': p['event_at'],
                    'signal_available_at': p['observed_at'], 'watch_armed_at': armed_at,
                    'status': 'AWAITING_QUOTE_COST_SAFETY_AND_SELL_VALIDATION',
                    'first_buy_only': policy['first_buy_only'], 'token_deduplication': True,
                    'exit_scenario': policy['exit_policy'], 'evidence_refs': sorted(set(refs+arm['evidence_refs'])),
                    'live_authority': False, 'real_fill': False, 'filled': False}
                result = db.execute('INSERT OR IGNORE INTO plans VALUES(?,?,?,?)', (token, plan_id, now, encoded(plan)))
                if result.rowcount:
                    plans.append(plan)
            first_only = policy['first_launch_only'] if trigger == 'CREATION' else policy['first_buy_only']
            if first_only:
                db.execute('INSERT OR IGNORE INTO consumed VALUES(?,?)', (arm_id, trigger))
                if policy['trigger'] != 'BOTH' or (policy['first_launch_only'] and policy['first_buy_only']
                        and db.execute('SELECT COUNT(*) FROM consumed WHERE arm=?', (arm_id,)).fetchone()[0] == 2):
                    db.execute('UPDATE armed SET closed=1 WHERE id=?', (arm_id,))
        return {'status': 'PAPER_PLANS_PREPARED' if plans else 'NO_NEW_PLAN', 'plans': plans,
                'missed': missed, 'fills_created': 0}, []

    def _trade(self, db, p, refs, now):
        token = self._node(p, required(p, 'token'), 'token')
        required(p, 'wallet'); required(p, 'quote_asset')
        number(p.get('price'), 'price', positive=True)
        eligible = p.get('benchmark_eligible', False)
        if type(eligible) is not bool:
            raise ValueError('benchmark classification needs explicit provenance')
        if eligible:
            references(p.get('eligibility_evidence_refs'))
        if p.get('side') not in {'BUY', 'SELL'} or (eligible and p['side'] != 'BUY'):
            raise ValueError('an entry benchmark must be an observed buy')
        for field in ('transaction_index', 'instruction_index'):
            if p.get(field) is not None and (type(p[field]) is not int or p[field] < 0):
                raise ValueError('nonnegative transaction ordering required')
        row = {**p, 'evidence_refs': refs}
        db.execute('INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?)', (identity([token, p['observation_id']]),
                   token, p['event_at'], p['observed_at'], int(eligible), p['quote_asset'], p['price'],
                   p.get('transaction_index'), p.get('instruction_index'), encoded(row)))
        return {'status': 'OBSERVED_PRICE_SAMPLE', 'token_node': token, 'executable_fill': False}, []

    def _benchmark(self, db, token, as_of):
        result = {'semantics': 'OBSERVED_PRICE_EXCURSION_NOT_EXECUTABLE_PNL',
            'full_tape_verified': False, 'entry': None, 'peaks': {str(h):None for h in HORIZONS}}
        earliest = db.execute('SELECT MIN(event_at) FROM trades WHERE token=? AND eligible=1 AND available<=?',
                              (token, as_of)).fetchone()[0]
        if earliest is None:
            return {**result, 'status': 'NO_EVIDENCED_ENTRY_BENCHMARK'}
        tied = [json.loads(row[0]) for row in db.execute(
            'SELECT body FROM trades WHERE token=? AND eligible=1 AND available<=? AND event_at=? LIMIT 257',
            (token, as_of, earliest))]
        if len(tied) > 256:
            return {**result, 'status': 'ENTRY_ORDER_REQUIRES_DEEPER_TAPE'}
        if len(tied) > 1 and (any(row.get('transaction_index') is None or row.get('instruction_index') is None for row in tied)
                             or len({(r['transaction_index'],r['instruction_index']) for r in tied}) != len(tied)):
            return {**result, 'status': 'ENTRY_ORDER_AMBIGUOUS'}
        entry = min(tied, key=lambda r:(r.get('transaction_index',0),r.get('instruction_index',0)))
        result.update(status='DESCRIPTIVE_ONLY', entry={k:entry.get(k) for k in
            ('wallet','event_at','observed_at','price','quote_asset','evidence_refs','eligibility_evidence_refs')})
        # SQLite walks only this token's first 60 seconds. No lifetime tape is
        # copied or sorted in Python, even for an old high-volume token.
        columns = ','.join('MAX(CASE WHEN event_at<=? THEN price END)' for _ in HORIZONS)
        values = db.execute('SELECT '+columns+' FROM trades WHERE token=? AND available<=?'
            ' AND event_at>=? AND event_at<=? AND quote=?',
            (*[earliest+h for h in HORIZONS], token, as_of, earliest, earliest+60, entry['quote_asset'])).fetchone()
        for horizon, value in zip(HORIZONS, values):
            result['peaks'][str(horizon)] = value/entry['price']-1 if value is not None else None
        result['window_elapsed'] = {str(h): as_of >= earliest+h for h in HORIZONS}
        return result

    def actor_children(self, p, refs):
        relation = p.get('relation') or {}
        kind = relation.get('kind')
        if (kind not in {'funded','deployed','bought'} or not p.get('observation_id')
                or str(p.get('chain_family', '')).upper() not in {'SOLANA', 'EVM'}
                or (relation.get('target_chain_id') or p.get('chain_id')) != p.get('chain_id')):
            return []
        source_type = relation.get('source_type', 'wallet')
        if (source_type not in {'wallet','contract'}
                or (kind in {'deployed','bought'} and relation.get('target_type','wallet') != 'token')
                or (kind == 'funded' and relation.get('target_type','wallet') not in {'wallet','contract'})):
            return []
        node = self._node(p, p['address'], source_type)
        children = []
        count = self.actor_store.out_degree(node, EdgeKind(kind), p['observed_at'])
        if kind in {'funded','deployed'} and count >= 2 and count & (count-1) == 0 and not self.actor_store.is_hub(node, EdgeKind.FUNDED, p['observed_at']):
            body = {k:p[k] for k in ('chain_id','chain_family','address','event_at','observed_at','observation_id')}
            body.update(as_of=p['observed_at'], entity_type=source_type, auto_watch_candidates=False)
            children.append({'kind':'OPERATOR_ANALYSIS','payload':body,'axes':{'chain':p['chain_id']}})
        if not self._has_watches and not self._has_armed:
            return children
        with self.lock, closing(self._connect()) as db:
            if kind == 'funded' and db.execute('SELECT 1 FROM watches WHERE funder=? LIMIT 1',(node,)).fetchone():
                body = {**p, 'source':p['address'], 'target':relation['target'],
                        'source_type':source_type, 'target_type':relation.get('target_type', 'wallet')}
                children.append({'kind':'OPERATOR_FUNDING','payload':body,'axes':{'chain':p['chain_id']}})
            elif kind in {'deployed','bought'} and db.execute('SELECT 1 FROM armed WHERE wallet=? AND closed=0 LIMIT 1',(node,)).fetchone():
                body = {**p, 'wallet':p['address'], 'wallet_type':source_type,
                        'token':relation['target'], 'trigger':'CREATION' if kind=='deployed' else 'BUY'}
                children.append({'kind':'OPERATOR_LAUNCH','payload':body,'axes':{'chain':p['chain_id']}})
        return children

    @staticmethod
    def csv_report(report):
        fields = ['token_node','deployers','event_at','available_at','event_utc','available_utc',
                  'evidence_refs','benchmark_status','entry_price','quote_asset',
                  *['peak_'+str(h)+'s_price_excursion' for h in HORIZONS]]
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in report.get('launches',[]):
            benchmark=row['benchmark'];entry=benchmark.get('entry') or {}
            output={k:row.get(k) for k in ('token_node','event_at','available_at')}
            for field, destination in (('event_at','event_utc'),('available_at','available_utc')):
                value = row.get(field)
                output[destination] = datetime.fromtimestamp(value,timezone.utc).isoformat() if value is not None else ''
            output['evidence_refs'] = ';'.join(sorted(set(row.get('evidence_refs',[])) | set(entry.get('evidence_refs') or [])))
            output.update(deployers=';'.join(row['deployers']),benchmark_status=benchmark['status'],
                          entry_price=entry.get('price'),quote_asset=entry.get('quote_asset'))
            output.update({'peak_'+str(h)+'s_price_excursion':benchmark['peaks'][str(h)] for h in HORIZONS})
            for key,value in output.items():
                if isinstance(value,str) and value[:1] in ('=','+','-','@','\t','\r'):
                    output[key]="'"+value
            writer.writerow(output)
        return stream.getvalue()
