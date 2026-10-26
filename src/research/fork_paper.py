"""Durable prospective SVM controls on the existing fork-probe observation lane.

No historical enrollment, extra RPC subscription, signer, live order path or
promotion hook. Inventory and its result outbox commit in the same transaction.
"""
import json
import time

from src.research.fork_simulation import ForkUnavailable, digest, run_isolated

STUDY = 'prospective-local-bank-control-v1'


class ForkPaperControls:
    def __init__(self, probe):
        self.probe = probe
        with probe.lock, probe.db:
            probe.db.executescript('''
              CREATE TABLE IF NOT EXISTS paper_bank_study(id TEXT PRIMARY KEY,body TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS paper_bank_inventory(job TEXT PRIMARY KEY,body TEXT NOT NULL,
                status TEXT NOT NULL,probe_id TEXT NOT NULL,due REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS paper_bank_due ON paper_bank_inventory(status,due);
            ''')
            declaration = {'study': STUDY, 'registered_at': time.time(),
                'virtual_funding_lamports_per_cohort': 2_000_000_000,
                'quote_budget_lamports': 1_000_000, 'minimum_holding_seconds': 60,
                'entry_rule': 'first usable public bank for targets queued after registration',
                'exit_rule': 'first sampled later bank after minimum holding; retry unresolved exits',
                'capital_scope': 'independent virtual account per target, not a deployable portfolio',
                'evidence_scope': 'unvalidated local SVM infrastructure control, no alpha promotion'}
            probe.db.execute('INSERT OR IGNORE INTO paper_bank_study VALUES(?,?)',
                             (STUDY, json.dumps(declaration, sort_keys=True)))
            self.declaration = json.loads(probe.db.execute(
                'SELECT body FROM paper_bank_study WHERE id=?', (STUDY,)).fetchone()[0])
            if {k:v for k,v in declaration.items() if k!='registered_at'} != {
                    k:v for k,v in self.declaration.items() if k!='registered_at'}:
                raise ValueError('paper bank study changed; declare a new version')

    async def prepare(self, frame, job):
        identity, first_seen = job[0], job[2]
        with self.probe.lock:
            row = self.probe.db.execute('SELECT body,status,probe_id,due FROM paper_bank_inventory WHERE job=?',
                                       (identity,)).fetchone()
            previous = self.probe.db.execute('SELECT body,ref,available FROM results WHERE id=?',
                                             (row[2],)).fetchone() if row else None
        now = frame['observed_at']
        if row:
            if row[1] == 'CLOSED' or now < row[3]:
                return None
            inventory = json.loads(row[0])
            if (not previous or not previous[1] or previous[2] > now or
                    json.loads(previous[0]).get('paper_control', {}).get('execution', {}).get('inventory') != inventory):
                raise ForkUnavailable('prior paper inventory lacks matching canonical ACK')
            request = {'phase': 'exit', 'inventory': inventory}
            prior_ref = previous[1]
        else:
            if (first_seen is None or first_seen < self.declaration['registered_at'] or
                    now < first_seen or frame['target']['quote_budget'] != self.declaration['quote_budget_lamports']):
                return None
            request, prior_ref = {'phase': 'entry'}, None
        execution = await run_isolated(frame, paper_request=request)
        return {'study': STUDY, 'declaration': self.declaration, 'declaration_sha256': digest(self.declaration),
            'phase': request['phase'], 'previous_canonical_ref': prior_ref,
            'source_frame_sha256': digest(frame), 'execution': execution,
            'real_fill': False, 'promotion_eligible': False}

    def commit_locked(self, identity, probe_id, result):
        if not result:
            return
        execution = result.get('execution', {})
        status = execution.get('status')
        if status not in {'PAPER_BANK_OPENED', 'PAPER_BANK_CLOSED', 'PAPER_BANK_EXIT_UNRESOLVED','PAPER_BANK_ENTRY_FAILED'}:
            return  # Unknown/failed entry cannot mint a paper holding.
        inventory = execution['inventory']
        closed = status in {'PAPER_BANK_CLOSED','PAPER_BANK_ENTRY_FAILED'}
        due = inventory['last_observed_at'] + self.declaration['minimum_holding_seconds']
        self.probe.db.execute('INSERT INTO paper_bank_inventory VALUES(?,?,?,?,?) '
            'ON CONFLICT(job) DO UPDATE SET body=excluded.body,status=excluded.status,'
            'probe_id=excluded.probe_id,due=excluded.due',
            (identity,json.dumps(inventory,sort_keys=True,allow_nan=False),'CLOSED' if closed else 'OPEN',probe_id,due))
        if not closed:
            self.probe.db.execute('UPDATE jobs SET due=? WHERE id=?', (due,identity))

    def report(self):
        with self.probe.lock:
            rows = self.probe.db.execute('SELECT p.status,count(*) FROM paper_bank_inventory p '
                'JOIN results r ON r.id=p.probe_id WHERE r.ref IS NOT NULL GROUP BY p.status').fetchall()
            pending = self.probe.db.execute('SELECT count(*) FROM paper_bank_inventory p '
                'JOIN results r ON r.id=p.probe_id WHERE r.ref IS NULL').fetchone()[0]
            outcomes = self.probe.db.execute("SELECT json_extract(r.body, '$.paper_control.execution.conditional_net_liquid_pnl_lamports') "
                "FROM paper_bank_inventory p JOIN results r ON r.id=p.probe_id WHERE p.status='CLOSED' AND r.ref IS NOT NULL").fetchall()
            failures = self.probe.db.execute("SELECT count(*) FROM paper_bank_inventory p JOIN results r ON r.id=p.probe_id "
                "WHERE r.ref IS NOT NULL AND json_extract(r.body,'$.paper_control.execution.status')='PAPER_BANK_ENTRY_FAILED'").fetchone()[0]
        counts = dict(rows)
        return {'status': 'PROSPECTIVE_LOCAL_SVM_CONTROL', 'study': self.declaration,
            'canonical_open_cohorts': counts.get('OPEN',0), 'canonical_closed_cohorts': counts.get('CLOSED',0),
            'canonical_failed_entries': failures,
            'pending_canonical_ack': pending, 'closed_conditional_pnl_lamports_sum':
                sum(row[0] for row in outcomes) if outcomes else None,
            'capital_scope': 'independent cohorts; summed conditional PnL is not a portfolio return',
            'live_costs_complete': False, 'real_fills': 0, 'promotion_eligible': False,
            'verified_live_profit': None}
