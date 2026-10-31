"""Export a retained operator analysis receipt; no database, RPC or trading IO."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.operator_play import OperatorPlayService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding='utf-8-sig'))
    report = receipt.get('detail', receipt)
    if (report.get('association_semantics') != 'OBSERVED_FUNDING_NETWORK_NOT_OWNER_IDENTITY'
            or report.get('live_authority') is not False):
        parser.error('an actual operator analysis receipt with no live authority is required')
    args.csv.write_text(OperatorPlayService.csv_report(report), encoding='utf-8')
    print(json.dumps({'status':'EXPORTED_DESCRIPTIVE_REPORT','launch_rows':len(report['launches']),
                      'more_launch_pages':report.get('next_cursor') is not None,
                      'complete_history_claimed':False,'fills_created':0}))


if __name__ == '__main__':
    main()
