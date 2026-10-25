"""Read a perps paper study without constructing a writer or querying a wallet."""
import argparse
import json
import sqlite3
import time
from pathlib import Path

from src.research.perps_paper import _account_view


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=Path('data/state/phantom_perps.sqlite'))
    parser.add_argument('--study')
    parser.add_argument('--after',default='')
    parser.add_argument('--limit',type=int,default=100)
    args=parser.parse_args(argv)
    if not 1<=args.limit<=1000: parser.error('display page must contain 1..1000 accounts; continue with --after')
    with sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro',uri=True,timeout=5) as db:
        row=(db.execute('SELECT id,created,spec FROM pp_studies WHERE id=?',(args.study,)).fetchone() if args.study
             else db.execute('SELECT id,created,spec FROM pp_studies ORDER BY created DESC LIMIT 1').fetchone())
        if not row: raise ValueError('requested paper study not found')
        study,started,spec=row
        plan=json.loads(spec)
        rows=db.execute('SELECT id,body FROM pp_accounts WHERE study=? AND id>? ORDER BY id LIMIT ?',
                        (study,args.after,args.limit+1)).fetchall()
        accounts=[_account_view(db,json.loads(raw),plan,time.time()) for identity,raw in rows[:args.limit]]
        report={'study':study,'started_at':started,'paper_specification':plan,'accounts':accounts,
            'has_more':len(rows)>args.limit,'next_cursor':rows[args.limit-1][0] if len(rows)>args.limit else None,
            'total_accounts':db.execute('SELECT count(*) FROM pp_accounts WHERE study=?',(study,)).fetchone()[0],
            'scope':'independent synthetic accounts; do not sum as deployable portfolio capital',
            'real_fills':0,'trade_authority':False,'verified_profit':None}
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
