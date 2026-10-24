"""Permissionless native World candidate discovery, independent of HTTP access.

An inferred account layout and consistent mint/vault relations are observations;
they are not operator identity, market rules, a firm quote or signing authority.
"""
import asyncio
import base64
import hashlib
import json
import struct
import time

from solders.pubkey import Pubkey
from src.detection.rug_detector import RugDetector, TOKEN_2022_PROGRAM

PROGRAM_CANDIDATE = 'prediCtPZCttYMvm2W3PtxmMxLmT1dtN7riU6Cxh6tM'
PROGRAM_SOURCE = 'https://github.com/chainstacklabs/world-xyz-research'
CASH_MINT = 'CASHx9KJUStyftLFWGvEVf59SGeG9sh5FfcnZMVPCASH'
MARKET_DISCRIMINATOR = hashlib.sha256(b'account:Market').digest()[:8]


def _raw(account, owner):
    if not isinstance(account, dict) or account.get('owner') != owner or account.get('executable') is not False:
        raise ValueError('native non-executable account with expected owner required')
    data = account.get('data')
    if not isinstance(data, list) or len(data) != 2 or data[1] != 'base64':
        raise ValueError('native base64 account data required')
    return base64.b64decode(data[0], validate=True)


def _frame(response, *, minimum_slot=0):
    if not isinstance(response, dict):
        raise ValueError('context-bearing native response required')
    slot = response.get('context', {}).get('slot')
    if type(slot) is not int or slot < minimum_slot or not isinstance(response.get('value'), list):
        raise ValueError('valid native observation slot and aligned values required')
    return slot, response['value']


def decode_market_candidate(row, slot):
    market = str(Pubkey.from_string(row['pubkey']))
    raw = _raw(row['account'], PROGRAM_CANDIDATE)
    if len(raw) != 320 or raw[:8] != MARKET_DISCRIMINATOR:
        raise ValueError('unrecognized candidate market layout')
    offsets = {'operator_key': 8, 'cash_mint': 40, 'yes_mint': 72, 'no_mint': 104, 'cash_vault': 136}
    if any(not any(raw[offset:offset+32]) for offset in offsets.values()):
        raise ValueError('null candidate relationship key')
    decoded = {key: str(Pubkey.from_bytes(raw[offset:offset+32])) for key, offset in offsets.items()}
    if decoded['cash_mint'] != CASH_MINT or len({decoded['yes_mint'], decoded['no_mint'], CASH_MINT}) != 3:
        raise ValueError('unexpected collateral or duplicate outcome mints')
    return {'market': market, **decoded, 'context_slot': slot,
        'account_sha256': hashlib.sha256(raw).hexdigest(), 'layout_source': PROGRAM_SOURCE,
        'operator_identity': 'UNVERIFIED', 'executable_quote': False, 'trade_authority': False}


def _token_metadata(raw, mint):
    """Read TokenMetadata TLV strings without executing or fetching their URI."""
    if len(raw) <= 82:
        return None
    offset = 166
    while offset + 4 <= len(raw):
        kind, length = struct.unpack_from('<HH', raw, offset)
        offset += 4
        payload = raw[offset:offset+length]
        if len(payload) != length:
            raise ValueError('truncated mint TLV')
        offset += length
        if kind != 19:
            continue
        if len(payload) < 64 or str(Pubkey.from_bytes(payload[32:64])) != mint:
            raise ValueError('TokenMetadata mint binding mismatch')
        position = 64
        def text():
            nonlocal position
            if position + 4 > len(payload): raise ValueError('truncated metadata string length')
            size = struct.unpack_from('<I', payload, position)[0]
            position += 4
            if size > len(payload) - position: raise ValueError('truncated metadata string')
            value = payload[position:position+size].decode('utf-8')
            position += size
            return value
        name, symbol, uri = text(), text(), text()
        if position + 4 > len(payload): raise ValueError('truncated additional metadata')
        count = struct.unpack_from('<I', payload, position)[0]
        position += 4
        if count > (len(payload) - position) // 8: raise ValueError('invalid metadata pair count')
        pairs = [[text(), text()] for _ in range(count)]
        if position != len(payload): raise ValueError('unexpected metadata trailing bytes')
        return {'name': name, 'symbol': symbol, 'uri': uri, 'additional_metadata': pairs,
                'semantic_rules_verified': False}
    return None


def verify_market_relations(candidate, accounts, slot):
    """Check native outcome authority, collateral mint and vault owner/mint."""
    details = {}
    for side in ('yes_mint', 'no_mint'):
        mint = candidate[side]
        raw = _raw(accounts.get(mint), TOKEN_2022_PROGRAM)
        fields = RugDetector.parse_spl_mint(raw, TOKEN_2022_PROGRAM)
        if (not fields['initialized'] or int.from_bytes(raw[:4], 'little') != 1
                or str(Pubkey.from_bytes(raw[4:36])) != candidate['market']):
            raise ValueError('outcome mint authority does not match candidate market')
        details[side] = {'mint': mint, 'decimals': fields['decimals'], 'supply_raw': str(fields['supply']),
            'freeze_authority_present': fields['freeze_authority_present'], 'extensions': fields['extensions'],
            'metadata': _token_metadata(raw, mint), 'account_sha256': hashlib.sha256(raw).hexdigest()}
    cash = _raw(accounts.get(CASH_MINT), TOKEN_2022_PROGRAM)
    cash_fields = RugDetector.parse_spl_mint(cash, TOKEN_2022_PROGRAM)
    if not cash_fields['initialized'] or len({cash_fields['decimals'], details['yes_mint']['decimals'], details['no_mint']['decimals']}) != 1:
        raise ValueError('native collateral/outcome decimal relationship mismatch')
    vault = _raw(accounts.get(candidate['cash_vault']), TOKEN_2022_PROGRAM)
    if (len(vault) < 165 or str(Pubkey.from_bytes(vault[:32])) != CASH_MINT or vault[108] not in (1, 2)):
        raise ValueError('native CASH vault relationship mismatch')
    vault_owner = str(Pubkey.from_bytes(vault[32:64]))
    matches = vault_owner == candidate['market']
    return {'relationship_status': ('NATIVE_MINT_VAULT_RELATIONSHIPS_CONSISTENT' if matches
                                    else 'VAULT_AUTHORITY_DIFFERS_FROM_REFERENCE'), 'context_slot': slot,
        'reference_relationships_consistent': matches, 'cash_vault_authority_observed': vault_owner,
        'cash_vault_authority_matches_market': matches,
        'outcomes': details, 'collateral_decimals': cash_fields['decimals'],
        'cash_vault_amount_raw': str(struct.unpack_from('<Q', vault, 64)[0]),
        'cash_vault_frozen': vault[108] == 2, 'collateral_extensions': cash_fields['extensions'],
        'operator_identity': 'UNVERIFIED', 'market_rules_verified': False,
        'firm_quote_available': False, 'settlement_verified': False, 'trade_authority': False}


class NativeWorldCensus:
    def __init__(self, store):
        self.store = store
        store.db.execute('''CREATE TABLE IF NOT EXISTS native_world_markets(
          market TEXT PRIMARY KEY,payload TEXT NOT NULL,census_slot INTEGER NOT NULL,
          observed_at REAL NOT NULL,proof_status TEXT NOT NULL,proof TEXT)''')
        store.db.execute('CREATE INDEX IF NOT EXISTS world_pending ON native_world_markets(census_slot,proof_status,market)')
        store.db.execute('CREATE INDEX IF NOT EXISTS world_cursor ON native_world_markets(census_slot,market)')
        store.db.commit()
        saved = store.db.execute("SELECT value FROM phantom_meta WHERE key='native_world_census'").fetchone()
        self.state = json.loads(saved[0]) if saved else {'status': 'UNOBSERVED', 'census_slot': None, 'next_refresh': 0}

    def _save(self):
        with self.store.db:
            self.store.db.execute("INSERT OR REPLACE INTO phantom_meta VALUES('native_world_census',?)",
                                  (json.dumps(self.state, allow_nan=False),))

    async def poll(self, rpc):
        if rpc is None:
            self.state['status'] = 'RPC_CONNECTION_REQUIRED'
            return
        try:
            unfinished = self.store.db.execute("SELECT 1 FROM native_world_markets WHERE census_slot=? AND proof_status='PENDING' LIMIT 1",
                                              (self.state.get('census_slot'),)).fetchone()
            if not unfinished and time.time() >= self.state.get('next_refresh', 0):
                response = await rpc.request('getProgramAccounts', [PROGRAM_CANDIDATE,
                    {'encoding': 'base64', 'commitment': 'finalized', 'withContext': True, 'filters': [{'dataSize': 320}]}])
                slot, rows = _frame(response)
                candidates, rejected = {}, 0
                for row in rows:
                    try:
                        candidate = decode_market_candidate(row, slot)
                        if candidate['market'] in candidates: raise ValueError('duplicate native market')
                        candidates[candidate['market']] = candidate
                    except (ValueError, KeyError, TypeError):
                        rejected += 1
                self.store.event('world_native_market_census', {'response': response, 'observed_at': time.time(),
                    'layout_source': PROGRAM_SOURCE, 'candidate_count': len(candidates), 'rejected_rows': rejected,
                    'operator_identity': 'UNVERIFIED', 'trade_authority': False})
                with self.store.db:
                    self.store.db.executemany('''INSERT INTO native_world_markets VALUES(?,?,?,?,'PENDING',NULL)
                      ON CONFLICT(market) DO UPDATE SET payload=excluded.payload,census_slot=excluded.census_slot,
                      observed_at=excluded.observed_at,proof_status='PENDING',proof=NULL ''',
                      ((key, json.dumps(value), slot, time.time()) for key, value in candidates.items()))
                self.state.update(status='NATIVE_CENSUS_OBSERVED', census_slot=slot, next_refresh=time.time()+900,
                                  observed_at=time.time(), rejected_layout_rows=rejected)
                self._save()
            slot = self.state.get('census_slot')
            if slot is None: return
            while True:
                # 33 relationships fit the RPC protocol's 100-account batch;
                # no total-market ceiling or fixed target prefix is imposed.
                rows = self.store.db.execute("SELECT market,payload FROM native_world_markets WHERE census_slot=? AND proof_status='PENDING' ORDER BY market LIMIT 33", (slot,)).fetchall()
                if not rows: break
                candidates = [json.loads(row['payload']) for row in rows]
                keys = list(dict.fromkeys([CASH_MINT, *[c[key] for c in candidates for key in ('yes_mint','no_mint','cash_vault')]]))
                response = await rpc.request('getMultipleAccounts', [keys, {'encoding':'base64','commitment':'finalized','minContextSlot':slot}])
                observed_slot, values = _frame(response, minimum_slot=slot)
                if len(keys) != len(values): raise ValueError('native account response alignment mismatch')
                accounts = dict(zip(keys, values))
                self.store.event('world_native_relationship_frame', {'keys': keys, 'response': response,
                    'census_slot': slot, 'observed_at': time.time(), 'trade_authority': False})
                with self.store.db:
                    for candidate in candidates:
                        try:
                            proof = verify_market_relations(candidate, accounts, observed_slot)
                            status = 'CONSISTENT_NATIVE_RELATIONSHIPS' if proof['reference_relationships_consistent'] else 'UNVERIFIED_RELATIONSHIPS'
                        except (ValueError, KeyError, TypeError, struct.error) as exc:
                            proof, status = {'error_type': type(exc).__name__, 'context_slot': observed_slot, 'trade_authority': False}, 'UNVERIFIED_RELATIONSHIPS'
                        self.store.db.execute('UPDATE native_world_markets SET proof_status=?,proof=? WHERE market=?',
                            (status, json.dumps(proof, allow_nan=False), candidate['market']))
                await asyncio.sleep(0)
            self.state['status'] = 'NATIVE_CANDIDATE_SNAPSHOT_COMPLETE'
            self.state.pop('error_type', None)
        except Exception as exc:
            self.state.update(status='NATIVE_SNAPSHOT_PARTIAL_OR_BLOCKED', error_type=type(exc).__name__)
        self._save()

    def report(self):
        slot = self.state.get('census_slot')
        counts = dict(self.store.db.execute('SELECT proof_status,count(*) FROM native_world_markets WHERE census_slot=? GROUP BY proof_status', (slot,)))
        return {**self.state, 'current_candidates': sum(counts.values()), 'by_relationship_status': counts,
            'snapshot_age_seconds': max(0, time.time()-self.state['observed_at']) if 'observed_at' in self.state else None,
            'total_market_limit': None, 'snapshot_scope': 'CURRENT_PROGRAM_ACCOUNTS_NOT_ALL_HISTORICAL_MARKETS',
            'operator_identity_verified': False, 'firm_quotes_available': False, 'trade_authority': False}

    def page(self, after='', limit=100):
        if not isinstance(after, str) or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('invalid native market cursor')
        rows = self.store.db.execute('SELECT * FROM native_world_markets WHERE census_slot=? AND market>? ORDER BY market LIMIT ?',
                                    (self.state.get('census_slot'), after, limit)).fetchall()
        items = [{'candidate': json.loads(row['payload']), 'relationship_status': row['proof_status'],
                  'relationship_evidence': json.loads(row['proof']) if row['proof'] else None,
                  'census_observed_at': row['observed_at'], 'trade_authority': False} for row in rows]
        cursor = rows[-1]['market'] if rows else after
        more = bool(self.store.db.execute('SELECT 1 FROM native_world_markets WHERE census_slot=? AND market>? LIMIT 1',
                    (self.state.get('census_slot'), cursor)).fetchone())
        return {'items': items, 'next_cursor': cursor, 'has_more': more,
                'census_slot': self.state.get('census_slot'), 'trade_authority': False}
