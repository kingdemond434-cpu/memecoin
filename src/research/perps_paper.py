"""Causal perpetuals accounting in independent, explicitly synthetic accounts.

No account queries, exchange methods, wallet material or live authority.
Funding estimates and unobserved costs never become certified net profit.
"""
from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from pathlib import Path
from src.research.perps_snapshots import _marks


ZERO = Decimal(0)


def _d(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('finite perps accounting input required') from None
    if not result.is_finite():
        raise ValueError('finite perps accounting input required')
    return result


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=str)


def _account_view(db, account, plan, now):
    equity = _d(account['cash'])
    status = 'SCENARIO_MARKED_EQUITY'
    pending = db.execute("SELECT count(*) FROM pp_funding WHERE account=? AND status='UNPRICED'",(account['id'],)).fetchone()[0]
    for coin,p in account['positions'].items():
        q = _d(p['quantity'])
        if not q: continue
        row = db.execute('SELECT at,body FROM pp_marks WHERE coin=?',(coin,)).fetchone()
        if not row or not 0<=now-row[0]<=plan['mark_max_age_seconds']:
            status = 'UNMEASURED_STALE_MARK'
            break
        equity += _d(p['margin'])+q*_d(json.loads(row[1])['markPx'])-(1 if q>0 else -1)*_d(p['entry_notional'])
    if pending: status = 'UNMEASURED_FUNDING'
    known = status=='SCENARIO_MARKED_EQUITY'
    return {**account,'valuation_status':status,'unpriced_funding_events':pending,
        'model_equity_usdc':str(equity) if known else None,
        'model_pnl_usdc':str(equity-_d(account['seed'])) if known else None,
        'verified_profit':None,'fee_basis':plan['fee_basis']}


def _margin(notional, asset, metadata):
    """Continuous maintenance tiers from the observed venue metadata."""
    notional = abs(_d(notional))
    table_id = asset.get('marginTableId')
    tables = dict(metadata.get('marginTables', []))
    rows = tables.get(table_id, {}).get('marginTiers')
    if rows is None and type(table_id) is int and 0 < table_id < 50:
        rows = [{'lowerBound':'0', 'maxLeverage':table_id}]
    if not rows:
        raise ValueError('observed maintenance margin table required')
    previous_bound, previous_rate, deduction, selected = None, ZERO, ZERO, None
    for row in rows:
        bound, maximum = _d(row['lowerBound']), _d(row['maxLeverage'])
        if bound < 0 or maximum <= 0 or (previous_bound is None and bound != 0):
            raise ValueError('invalid margin tier')
        rate = Decimal(1)/(2*maximum)
        if previous_bound is not None and (bound <= previous_bound or rate < previous_rate):
            raise ValueError('invalid margin tier ordering')
        deduction += bound*(rate-previous_rate)
        if notional >= bound:
            selected = (notional*rate-deduction, maximum)
        previous_bound, previous_rate = bound, rate
    return selected


def _walk(book, signed_quantity, precision, *, limit=None):
    """Displayed taker IOC depth, including explicit partial execution."""
    quantity = _d(signed_quantity)
    if type(precision) is not int or not 0 <= precision <= 18 or not quantity:
        raise ValueError('valid quantity and precision required')
    quantum = Decimal(1).scaleb(-precision)
    if abs(quantity).quantize(quantum, rounding=ROUND_DOWN) != abs(quantity):
        raise ValueError('quantity exceeds observed asset precision')
    levels = book.get('levels')
    if not isinstance(levels, list) or len(levels) != 2 or not all(levels):
        raise ValueError('two-sided observed book required')
    parsed = [[(_d(row['px']),_d(row['sz'])) for row in side] for side in levels]
    if any(px <= 0 or size < 0 for side in parsed for px,size in side):
        raise ValueError('invalid book depth')
    bids, asks = parsed
    if (bids[0][0]>=asks[0][0] or bids!=sorted(bids,key=lambda x:-x[0])
            or asks!=sorted(asks,key=lambda x:x[0])):
        raise ValueError('unordered or crossed book')
    left, cash = abs(quantity), ZERO
    for px,size in asks if quantity>0 else bids:
        if limit is not None and ((quantity>0 and px>_d(limit)) or (quantity<0 and px<_d(limit))):
            break
        take = min(left,size).quantize(quantum,rounding=ROUND_DOWN)
        cash += take*px
        left -= take
        if not left: break
    filled = abs(quantity)-left
    return filled*(1 if quantity>0 else -1), cash


class PerpsPaperLedger:
    def __init__(self, state, *, now=None, controls=True):
        self.state, self.db = state, state.db
        self.controls = controls
        self._plan = {'version':1,'account_seed_usdc':'100','control_notional_usdc':'10.25',
            'control_leverage':'2','control_hold_seconds':600,'latency_seconds':.3,
            'fee_bps':'9.5','fee_basis':'UNVERIFIED_ACCOUNT_FEE_SCENARIO',
            'mark_max_age_seconds':30,'oracle_sample_max_age_seconds':10,
            'controls_enabled':controls,
            'control_scope':'independent long/short pipeline controls; not an alpha strategy'}
        self._spec = _dump(self._plan)
        source = hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
        self.study = hashlib.sha256(_dump([self._plan,source]).encode()).hexdigest()
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS pp_studies(id TEXT PRIMARY KEY,created REAL NOT NULL,spec TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS pp_accounts(id TEXT PRIMARY KEY,study TEXT NOT NULL,body TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS pp_accounts_study ON pp_accounts(study);
          CREATE TABLE IF NOT EXISTS pp_orders(id TEXT PRIMARY KEY,account TEXT NOT NULL,coin TEXT NOT NULL,
            created REAL NOT NULL,status TEXT NOT NULL,body TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS pp_orders_pending ON pp_orders(coin,status,created);
          CREATE TABLE IF NOT EXISTS pp_fills(id TEXT PRIMARY KEY,account TEXT NOT NULL,coin TEXT NOT NULL,
            at REAL NOT NULL,quantity TEXT NOT NULL,body TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS pp_fills_position ON pp_fills(account,coin,at);
          CREATE INDEX IF NOT EXISTS pp_fills_coin ON pp_fills(coin,account);
          CREATE TABLE IF NOT EXISTS pp_exposure(account TEXT NOT NULL,coin TEXT NOT NULL,
            at REAL NOT NULL,delta TEXT NOT NULL,kind TEXT NOT NULL,ref TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS pp_exposure_boundary ON pp_exposure(account,at);
          CREATE TABLE IF NOT EXISTS pp_marks(coin TEXT PRIMARY KEY,at REAL NOT NULL,body TEXT NOT NULL,ref TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS pp_oracles(coin TEXT NOT NULL,at REAL NOT NULL,price TEXT NOT NULL,ref TEXT NOT NULL,
            PRIMARY KEY(coin,at));
          CREATE TABLE IF NOT EXISTS pp_funding(account TEXT NOT NULL,coin TEXT NOT NULL,time INTEGER NOT NULL,
            quantity TEXT NOT NULL,status TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(account,coin,time));
          CREATE INDEX IF NOT EXISTS pp_funding_pending ON pp_funding(coin,status,account,time);
          CREATE TABLE IF NOT EXISTS pp_journal(seq INTEGER PRIMARY KEY,account TEXT NOT NULL,at REAL NOT NULL,
            kind TEXT NOT NULL,body TEXT NOT NULL,ref TEXT NOT NULL);
        ''')
        at = time.time() if now is None else now
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO pp_studies VALUES(?,?,?)',(self.study,at,_dump(self._plan)))
        self.started = self.db.execute('SELECT created FROM pp_studies WHERE id=?',(self.study,)).fetchone()[0]

    def _account(self, identity):
        row = self.db.execute('SELECT body FROM pp_accounts WHERE id=? AND study=?',(identity,self.study)).fetchone()
        if not row: raise ValueError('account belongs to no active paper study')
        return json.loads(row[0])

    def _save(self, account):
        self.db.execute('UPDATE pp_accounts SET body=? WHERE id=?',(_dump(account),account['id']))

    def _event(self, account, at, kind, body, ref):
        self.db.execute('INSERT INTO pp_journal(account,at,kind,body,ref) VALUES(?,?,?,?,?)',
                        (account['id'],at,kind,_dump(body),ref))

    def _create(self, identity, at, ref):
        account = {'id':identity,'cash':self._plan['account_seed_usdc'],'seed':self._plan['account_seed_usdc'],
            'positions':{},'fees':'0','realized_pnl':'0','funding_cashflow':'0','funding_scan':at,
            'kind':'PIPELINE_CONTROL_NOT_ALPHA','created':at,'last_control':0,'bad_debt':'0'}
        self.db.execute('INSERT OR IGNORE INTO pp_accounts VALUES(?,?,?)',(identity,self.study,_dump(account)))
        if self.db.execute('SELECT changes()').fetchone()[0]:
            self._event(account,at,'SYNTHETIC_SEED',{'usdc':account['seed']},ref)
        return self._account(identity)

    def _contract(self, coin):
        row = self.db.execute('SELECT dex,spec FROM markets WHERE coin=? AND active=1',(coin,)).fetchone()
        if not row: raise ValueError('observed active perps contract required')
        spec = json.loads(row[1])
        collateral = self.db.execute('SELECT metadata FROM collateral_tokens WHERE token_index=?',(spec['collateral_token'],)).fetchone()
        token = json.loads(collateral[0]) if collateral else {}
        if type(spec['collateral_token']) is not int or spec['collateral_token']!=0 or token.get('name')!='USDC' or token.get('isCanonical') is not True:
            raise ValueError('canonical USDC collateral identity required')
        meta = self.db.execute('SELECT metadata FROM catalogs WHERE dex=?',(row[0],)).fetchone()
        if not meta: raise ValueError('observed contract margin metadata required')
        return spec['asset'], json.loads(meta[0])

    def _risk(self, account, at):
        if self.db.execute("SELECT 1 FROM pp_funding WHERE account=? AND status='UNPRICED' LIMIT 1",(account['id'],)).fetchone():
            return {'status':'UNMEASURED_FUNDING'}
        equity, initial, maintenance = _d(account['cash']), ZERO, ZERO
        isolated = {}
        for coin,p in account['positions'].items():
            q = _d(p['quantity'])
            if not q: continue
            mark = self.db.execute('SELECT at,body FROM pp_marks WHERE coin=?',(coin,)).fetchone()
            if not mark or not 0<=at-mark[0]<=self._plan['mark_max_age_seconds']:
                return {'status':'UNMEASURED_STALE_MARK'}
            try: asset,meta = self._contract(coin)
            except ValueError: return {'status':'UNMEASURED_CONTRACT'}
            px = _d(json.loads(mark[1])['markPx'])
            mm,maximum = _margin(q*px,asset,meta)
            pnl = q*px-(1 if q>0 else -1)*_d(p['entry_notional'])
            if p['mode']=='isolated':
                isolated[coin] = {'equity':_d(p['margin'])+pnl,'maintenance':mm,'notional':abs(q*px)}
            else:
                equity += pnl
                maintenance += mm
                initial += abs(q*px)/min(_d(p['leverage']),maximum)
        return {'status':'OBSERVED_MARK_SCENARIO','cross_equity':equity,'initial':initial,
                'maintenance':maintenance,'isolated':isolated}

    def submit(self, account_id, coin, quantity, *, at, ref, leverage='2', mode='isolated',
               reduce_only=False, limit=None, reason='PIPELINE_CONTROL', initial_liquidation_partial=False):
        """Durable research IOC intent; matching requires a future observed book."""
        account = self._account(account_id)
        if _dump(self._plan)!=self._spec: raise ValueError('paper specification changed; new study required')
        asset,meta = self._contract(coin)
        q,lev = _d(quantity),_d(leverage)
        quantum = Decimal(1).scaleb(-asset['szDecimals'])
        if not q or q.quantize(quantum,rounding=ROUND_DOWN)!=q or lev<=0:
            raise ValueError('valid order size and leverage required')
        if mode not in {'isolated','cross'} or (mode=='cross' and asset.get('onlyIsolated')):
            raise ValueError('unsupported margin mode')
        if not ref or at<self.started or lev>_d(asset['maxLeverage']):
            raise ValueError('observed order authority/clock/leverage required')
        _d(at)
        pending = self.db.execute("SELECT 1 FROM pp_orders WHERE account=? AND coin=? AND status='PENDING'",(account_id,coin)).fetchone()
        if pending: return None
        p = account['positions'].get(coin)
        if reduce_only and (not p or not _d(p['quantity']) or q*_d(p['quantity'])>=0):
            raise ValueError('reduce-only intent must reduce an existing position')
        if limit is not None and _d(limit)<=0: raise ValueError('positive limit required')
        identity = hashlib.sha256(_dump([self.study,account_id,coin,at,ref,quantity,reason]).encode()).hexdigest()
        order = {'id':identity,'account':account_id,'coin':coin,'quantity':str(q),'leverage':str(lev),
            'mode':mode,'reduce_only':bool(reduce_only),'limit':str(limit) if limit is not None else None,
            'eligible_after':at+self._plan['latency_seconds'],'reason':reason,'decision_ref':ref}
        order['initial_liquidation_partial'] = bool(initial_liquidation_partial)
        self.db.execute('INSERT OR IGNORE INTO pp_orders VALUES(?,?,?,?,?,?)',
                        (identity,account_id,coin,at,'PENDING',_dump(order)))
        if self.db.execute('SELECT changes()').fetchone()[0]:
            self._event(account,at,'INTENT',order,ref)
        return identity

    def _fill(self, order, book, at, ref):
        account = self._account(order['account'])
        coin = order['coin']
        self._funding_due(account,at,ref)
        self._save(account)
        self._settle_funding(coin,at,ref)
        account = self._account(order['account'])
        asset,meta = self._contract(coin)
        p = account['positions'].get(coin)
        previous = _d(p['quantity']) if p else ZERO
        q = _d(order['quantity'])
        if order['reason']=='OBSERVED_MARK_LIQUIDATION' and previous:
            risk = self._risk(account,at)
            if risk['status']!='OBSERVED_MARK_SCENARIO': return 'UNMEASURED_LIQUIDATION_RISK'
            bucket = risk['isolated'][coin] if p['mode']=='isolated' else {'equity':risk['cross_equity'],'maintenance':risk['maintenance']}
            if bucket['equity']>=bucket['maintenance']: return 'CANCELLED_MARGIN_RECOVERED'
        if order['reduce_only']:
            if not previous or q*previous>=0: return 'CANCELLED_REDUCE_ONLY'
            q = min(abs(q),abs(previous))*(1 if q>0 else -1)
        elif previous and (q*previous<0 or p['mode']!=order['mode'] or _d(p['leverage'])!=_d(order['leverage'])):
            return 'REJECTED_POSITION_FLIP'  # close and reopen are separate causal intents
        filled,notional = _walk(book,q,asset['szDecimals'],limit=order['limit'])
        if not filled: return 'IOC_NO_DEPTH'
        px = notional/abs(filled)
        fee = notional*_d(self._plan['fee_bps'])/10000
        closing = bool(previous and filled*previous<0)
        if not closing and notional<10: return 'REJECTED_MIN_NOTIONAL'
        realized = ZERO
        if closing:
            allocated = (_d(p['entry_notional']) if abs(filled)==abs(previous)
                         else _d(p['entry_notional'])*abs(filled)/abs(previous))
            realized = (notional-allocated)*(1 if previous>0 else -1)
            p['entry_notional'] = str(_d(p['entry_notional'])-allocated)
            if p['mode']=='isolated':
                p['margin'] = str(_d(p['margin'])+realized-fee)
            else:
                account['cash'] = str(_d(account['cash'])+realized-fee)
            p['quantity'] = str(previous+filled)
            if not _d(p['quantity']) and p['mode']=='isolated':
                residual = _d(p['margin'])
                account['cash'] = str(_d(account['cash'])+max(ZERO,residual))
                if residual<0:
                    account['bad_debt'] = str(_d(account['bad_debt'])-residual)
                    self._event(account,at,'ISOLATED_INSOLVENCY',{'deficit_usdc':str(-residual),
                        'classification':'MODEL_DEFICIT_INSURANCE_UNVERIFIED'},ref)
                p['margin'] = '0'
        else:
            lev = _d(order['leverage'])
            mark = self.db.execute('SELECT at,body FROM pp_marks WHERE coin=?',(coin,)).fetchone()
            if not mark or not 0<=at-mark[0]<=self._plan['mark_max_age_seconds']: return 'UNMEASURED_MARK'
            mark_px = _d(json.loads(mark[1])['markPx'])
            _,maximum = _margin((abs(previous)+abs(q))*max(px,mark_px),asset,meta)
            if lev>maximum: return 'REJECTED_TIER_LEVERAGE'
            risk = self._risk(account,at)
            if risk['status']!='OBSERVED_MARK_SCENARIO': return 'UNMEASURED_MARK'
            required = abs(filled)*max(px,mark_px)/lev
            entry_pnl = filled*mark_px-(1 if filled>0 else -1)*notional
            full_ratio = abs(q/filled)
            if order['mode']=='isolated':
                required += max(ZERO,-entry_pnl)
                if (_d(account['cash'])<(required+fee)*full_ratio
                        or risk['cross_equity']-(required+fee)*full_ratio<risk['initial']):
                    return 'REJECTED_MARGIN'
                account['cash'] = str(_d(account['cash'])-required-fee)
            elif risk['cross_equity']+(entry_pnl-fee)*full_ratio<risk['initial']+required*full_ratio:
                return 'REJECTED_MARGIN'
            else:
                account['cash'] = str(_d(account['cash'])-fee)
            entry_notional = (_d(p['entry_notional']) if p and previous else ZERO)+notional
            basis = entry_notional/(abs(previous)+abs(filled))
            p = {'quantity':str(previous+filled),'basis':str(basis),'entry_notional':str(entry_notional),'margin':str(( _d(p['margin']) if p and previous else ZERO)+(required if order['mode']=='isolated' else ZERO)),
                'mode':order['mode'],'leverage':str(lev),'opened_at':p['opened_at'] if p and previous else at}
            account['positions'][coin] = p
        account['fees'] = str(_d(account['fees'])+fee)
        if order['initial_liquidation_partial']:
            account['last_partial_liquidation'] = at
        account['realized_pnl'] = str(_d(account['realized_pnl'])+realized)
        receipt = {'order':order['id'],'quantity':str(filled),'price':str(px),'notional_usdc':str(notional),'fee_usdc':str(fee),
            'realized_usdc':str(realized),'book_time':book['time'],'effective_at':at,'source_ref':ref,
            'fee_basis':self._plan['fee_basis'],'real_fill':False,'simulated':True,'capacity_scope':'independent account scenario'}
        self.db.execute('INSERT INTO pp_fills VALUES(?,?,?,?,?,?)',
                        (order['id'],account['id'],coin,at,str(filled),_dump(receipt)))
        self.db.execute('INSERT INTO pp_exposure VALUES(?,?,?,?,?,?)',
                        (account['id'],coin,at,str(filled),'SIMULATED_FILL',ref))
        self._event(account,at,'SIMULATED_IOC_FILL',receipt,ref)
        self._save(account)
        return 'FILLED' if abs(filled)==abs(q) else 'PARTIAL_IOC'

    def _funding_due(self, account, at, ref):
        # Reconstruct exposure at the hourly boundary from fills, including
        # positions already closed by the time history finally arrives.
        first = int(account['funding_scan']//3600+1)*3600
        end = int(at//3600)*3600
        for boundary in range(first,min(end,first+249*3600)+1,3600):
            exposures = {}
            for coin,quantity in self.db.execute('SELECT coin,delta FROM pp_exposure WHERE account=? AND at<?', (account['id'],boundary)):
                exposures[coin] = exposures.get(coin,ZERO)+_d(quantity)
            for coin,q in exposures.items():
                if q:
                    self.db.execute("INSERT OR IGNORE INTO pp_funding VALUES(?,?,?,?,'UNPRICED',?)",
                        (account['id'],coin,boundary*1000,str(q),_dump({'reason':'settled rate and prior oracle sample required','source_ref':ref})))
            account['funding_scan'] = boundary

    def _exposure_at(self, identity, coin, at):
        return sum((_d(row[0]) for row in self.db.execute(
            'SELECT delta FROM pp_exposure WHERE account=? AND coin=? AND at<?',(identity,coin,at))),ZERO)

    def _settle_funding(self, coin, at, ref, settlements=()):
        # Actual funding timestamps are usually a few milliseconds AFTER the
        # nominal hour. Match the hour, but reconstruct exposure at the actual
        # observed settlement instant, not a fabricated round-hour timestamp.
        accounts = None
        for event in settlements:
            instant = event['time']/1000
            if instant<self.started: continue
            boundary = event['time']//3600000*3600000
            if accounts is None:
                accounts = [json.loads(row[0]) for row in self.db.execute(
                    'SELECT DISTINCT a.body FROM pp_accounts a JOIN pp_fills f ON f.account=a.id WHERE a.study=? AND f.coin=?',(self.study,coin))]
            for account in accounts:
                if account['created']>instant: continue
                q = self._exposure_at(account['id'],coin,instant)
                if q:
                    self.db.execute("INSERT INTO pp_funding VALUES(?,?,?,?,'UNPRICED',?) ON CONFLICT(account,coin,time) DO UPDATE SET quantity=excluded.quantity WHERE pp_funding.status='UNPRICED'",
                        (account['id'],coin,boundary,str(q),_dump({'reason':'prior billing-oracle sample required','source_ref':ref,'actual_settlement_time_ms':event['time']})))
        rows = self.db.execute("SELECT f.account,f.time,h.time,h.body FROM pp_funding f JOIN pp_accounts a ON a.id=f.account JOIN funding h ON h.coin=f.coin AND h.time>=f.time AND h.time<f.time+3600000 WHERE a.study=? AND f.coin=? AND f.status='UNPRICED'",(self.study,coin)).fetchall()
        multiplicities = {}
        for identity,boundary,instant,raw in rows:
            multiplicities[identity,boundary] = multiplicities.get((identity,boundary),0)+1
        for identity,boundary,instant,raw in rows:
            if multiplicities[identity,boundary]!=1: continue
            effective = instant/1000
            q = self._exposure_at(identity,coin,effective)
            if not q:
                self.db.execute("UPDATE pp_funding SET quantity='0',status='NO_EXPOSURE',body=? WHERE account=? AND coin=? AND time=?",(_dump({'actual_settlement_time_ms':instant,'source_ref':ref}),identity,coin,boundary))
                continue
            before = self.db.execute('SELECT at,price,ref FROM pp_oracles WHERE coin=? AND at<=? ORDER BY at DESC LIMIT 1',(coin,effective)).fetchone()
            if not before or effective-before[0]>self._plan['oracle_sample_max_age_seconds']:
                continue  # never replace missing billing oracle with today's mark
            rate = _d(json.loads(raw)['fundingRate'])
            if abs(rate)>Decimal('.04'): continue
            cashflow = -_d(q)*_d(before[1])*rate
            account = self._account(identity)
            p = account['positions'].get(coin)
            # Corrections reach the account where the economic exposure was
            # held. No new position silently inherits an old isolated debit.
            if p and _d(p['quantity']) and p['mode']=='isolated' and p['opened_at']<effective:
                p['margin'] = str(_d(p['margin'])+cashflow)
            else:
                account['cash'] = str(_d(account['cash'])+cashflow)
            account['funding_cashflow'] = str(_d(account['funding_cashflow'])+cashflow)
            body = {'cashflow_usdc':str(cashflow),'rate':str(rate),'oracle':before[1],
                'actual_settlement_time_ms':instant,'settled_quantity':str(q),
                'oracle_sample_at':before[0],'oracle_source_ref':before[2],'source_ref':ref,
                'classification':'SAMPLED_ORACLE_FUNDING_SCENARIO_NOT_EXACT_BILLING'}
            self.db.execute("UPDATE pp_funding SET status='SAMPLED_SCENARIO',body=? WHERE account=? AND coin=? AND time=?",(_dump(body),identity,coin,boundary))
            self._event(account,at,'FUNDING_SCENARIO',body,ref)
            self._save(account)

    def _liquidations(self, account, at, ref):
        risk = self._risk(account,at)
        if risk['status']!='OBSERVED_MARK_SCENARIO': return
        for coin,p in account['positions'].items():
            if not _d(p['quantity']): continue
            bucket = risk['isolated'].get(coin) if p['mode']=='isolated' else {
                'equity':risk['cross_equity'],'maintenance':risk['maintenance'],
                'notional':abs(_d(p['quantity']))*_d(json.loads(self.db.execute('SELECT body FROM pp_marks WHERE coin=?',(coin,)).fetchone()[0])['markPx'])}
            if bucket['equity']>=bucket['maintenance']: continue
            q = -_d(p['quantity'])
            partial = False
            if bucket['notional']>100000 and at-account.get('last_partial_liquidation',0)>=30:
                asset,_ = self._contract(coin)
                q = (q*Decimal('.2')).quantize(Decimal(1).scaleb(-asset['szDecimals']),rounding=ROUND_DOWN)
                if not q: q = -_d(p['quantity'])
                else: partial = True
            self.submit(account['id'],coin,q,at=at,ref=ref,leverage=p['leverage'],
                mode=p['mode'],reduce_only=True,reason='OBSERVED_MARK_LIQUIDATION',initial_liquidation_partial=partial)

    def _backstop(self, account, coin, at, ref):
        risk = self._risk(account,at)
        if risk['status']!='OBSERVED_MARK_SCENARIO': return
        p = account['positions'].get(coin)
        if not p or not _d(p['quantity']): return
        bucket = risk['isolated'][coin] if p['mode']=='isolated' else {'equity':risk['cross_equity'],'maintenance':risk['maintenance']}
        if bucket['equity']>=bucket['maintenance']*Decimal(2)/3: return
        affected = [coin] if p['mode']=='isolated' else [c for c,v in account['positions'].items() if v['mode']=='cross']
        body = {'coins':affected,'pre_transfer_equity':str(bucket['equity']),
                'classification':'BACKSTOP_COLLATERAL_TRANSFER_SCENARIO_NOT_BOOK_FILL'}
        for c in affected:
            self.db.execute('INSERT INTO pp_exposure VALUES(?,?,?,?,?,?)',
                (account['id'],c,at,str(-_d(account['positions'][c]['quantity'])),'BACKSTOP_TRANSFER',ref))
            account['positions'][c].update(quantity='0',margin='0',entry_notional='0')
            self.db.execute("UPDATE pp_orders SET status='CANCELLED_BACKSTOP' WHERE account=? AND coin=? AND status='PENDING'",(account['id'],c))
        if p['mode']=='cross': account['cash']='0'
        account['bad_debt'] = str(_d(account['bad_debt'])+max(ZERO,-bucket['equity']))
        self._event(account,at,'BACKSTOP_MODEL',body,ref)
        self._save(account)

    def on_record(self, record, ref):
        if _dump(self._plan)!=self._spec: raise ValueError('paper specification changed; new study required')
        at,role = float(record['observed_at']),record['perps_role']
        if at<self.started: return
        affected = []
        if role=='catalog':
            for asset,context in zip(record['meta']['universe'],record['contexts']):
                coin = asset['name']
                if context.get('markPx') is None or context.get('oraclePx') is None: continue
                if _d(context['markPx'])<=0 or _d(context['oraclePx'])<=0: continue
                self.db.execute('INSERT INTO pp_marks VALUES(?,?,?,?) ON CONFLICT(coin) DO UPDATE SET at=excluded.at,body=excluded.body,ref=excluded.ref WHERE excluded.at>pp_marks.at', (coin,at,_dump(context),ref))
                self.db.execute('INSERT OR IGNORE INTO pp_oracles VALUES(?,?,?,?)',(coin,at,str(_d(context['oraclePx'])),ref))
                affected.append(coin)
        elif role in {'mark_snapshot','asset_context'}:
            if not 0<=time.time()-at<=10: return
            updates = ({coin:{'markPx':px} for coin,px in _marks(record['frame']).items()}
                       if role=='mark_snapshot' else {record['frame']['data']['coin']:record['frame']['data']['ctx']})
            for coin,context in updates.items():
                if not self.db.execute('SELECT 1 FROM markets WHERE coin=? AND active=1',(coin,)).fetchone(): continue
                if context.get('markPx') is None or _d(context['markPx'])<=0: continue
                self.db.execute('INSERT INTO pp_marks VALUES(?,?,?,?) ON CONFLICT(coin) DO UPDATE SET at=excluded.at,body=excluded.body,ref=excluded.ref WHERE excluded.at>pp_marks.at',(coin,at,_dump(context),ref))
                if role=='asset_context' and context.get('oraclePx') is not None and _d(context['oraclePx'])>0:
                    self.db.execute('INSERT OR IGNORE INTO pp_oracles VALUES(?,?,?,?)',(coin,at,str(_d(context['oraclePx'])),ref))
                affected.append(coin)
        elif role=='book' and 0<=time.time()-at<=10:
            coin,book = record['coin'],record['book']
            if not 0<=at-book['time']/1000<=10: return
            orders = self.db.execute("SELECT o.id,o.body FROM pp_orders o JOIN pp_accounts a ON a.id=o.account WHERE o.coin=? AND o.status='PENDING' AND a.study=? ORDER BY o.created,o.id",(coin,self.study)).fetchall()
            for identity,raw in orders:
                order = json.loads(raw)
                if book['time']/1000<=order['eligible_after']: continue
                status = self._fill(order,book,at,ref)
                self.db.execute('UPDATE pp_orders SET status=? WHERE id=?',(status,identity))
                if order['reason']=='OBSERVED_MARK_LIQUIDATION':
                    self._backstop(self._account(order['account']),coin,at,ref)
            if self.controls:
                self._controls(coin,book,at,ref)
            affected.append(coin)
        elif role=='funding': affected.append(record['coin'])
        if not affected: return
        placeholders = ','.join('?' for _ in affected)
        identities = [row[0] for row in self.db.execute('SELECT DISTINCT a.id FROM pp_accounts a JOIN pp_fills f ON f.account=a.id WHERE a.study=? AND f.coin IN ('+placeholders+')',(self.study,*affected))]
        for identity in identities:
            account = self._account(identity)
            self._funding_due(account,at,ref)
            self._save(account)
        for coin in affected:
            self._settle_funding(coin,at,ref,record['rows'] if role=='funding' else ())
        if role in {'catalog','mark_snapshot','asset_context'}:
            for identity in identities: self._liquidations(self._account(identity),at,ref)

    def _controls(self, coin, book, at, ref):
        try: asset,_ = self._contract(coin)
        except ValueError: return
        quantum = Decimal(1).scaleb(-asset['szDecimals'])
        for side in (1,-1):
            identity = self.study+':'+coin+':'+str(side)
            account = self._create(identity,at,ref)
            p = account['positions'].get(coin)
            if p and _d(p['quantity']):
                if at-p['opened_at']<self._plan['control_hold_seconds']: continue
                self.submit(identity,coin,-_d(p['quantity']),at=at,ref=ref,reduce_only=True,
                            mode=p['mode'],leverage=p['leverage'],reason='CONTROL_HORIZON_EXIT')
            else:
                quantity = (_d(self._plan['control_notional_usdc'])/_d(book['levels'][1 if side>0 else 0][0]['px'])).quantize(quantum,rounding=ROUND_UP)*side
                self.submit(identity,coin,quantity,at=at,ref=ref,leverage=min(_d(self._plan['control_leverage']),_d(asset['maxLeverage'])))

    def report(self):
        accounts = self.db.execute('SELECT count(*) FROM pp_accounts WHERE study=?',(self.study,)).fetchone()[0]
        fills = self.db.execute('SELECT count(*) FROM pp_fills f JOIN pp_accounts a ON a.id=f.account WHERE a.study=?',(self.study,)).fetchone()[0]
        orders = dict(self.db.execute('SELECT o.status,count(*) FROM pp_orders o JOIN pp_accounts a ON a.id=o.account WHERE a.study=? GROUP BY o.status',(self.study,)))
        funding = dict(self.db.execute('SELECT f.status,count(*) FROM pp_funding f JOIN pp_accounts a ON a.id=f.account WHERE a.study=? GROUP BY f.status',(self.study,)))
        opened = self.db.execute("SELECT count(*) FROM pp_accounts a,json_each(a.body,'$.positions') p WHERE a.study=? AND CAST(json_extract(p.value,'$.quantity') AS REAL)<>0",(self.study,)).fetchone()[0]
        now=time.time()
        fresh_marks=self.db.execute('SELECT count(*) FROM pp_marks WHERE at BETWEEN ? AND ?',(now-self._plan['mark_max_age_seconds'],now)).fetchone()[0]
        recent_ws=self.db.execute('SELECT count(*),COALESCE(sum(subscriptions),0) FROM pp_ws_attempts WHERE at>?',(now-60,)).fetchone()
        examples = [_account_view(self.db,json.loads(row[0]),self._plan,time.time()) for row in self.db.execute('SELECT body FROM pp_accounts WHERE study=? ORDER BY id LIMIT 8',(self.study,))]
        return {'study':self.study,'started_at':self.started,'accounts':accounts,'simulated_fills':fills,
            'open_positions':opened,'account_examples':examples,
            'fresh_contract_marks':fresh_marks,'ws_connections_last_minute':recent_ws[0],
            'ws_subscriptions_last_minute':recent_ws[1],
            'orders':orders,'funding':funding,'controls_enabled':self.controls,'paper_specification':json.loads(self._spec),
            'real_fills':0,'live_authority':False,'verified_net_profit':None,
            'scope':'independent account execution controls; not pooled capital or a validated alpha strategy',
            'missing_costs':['verified account/deployer fees','exact billing oracle','unobserved intervening mark/book states','dynamic impact and landing'],
            'past_studies_retained':True}
