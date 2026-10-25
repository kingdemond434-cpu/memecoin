"""Read-only, streaming fill/TCA/markout audit of retained episode files."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import time

from src.research.execution_measurement import clock, episode_measurements


def audit(episodes, output, *, as_of):
    source, target = Path(episodes).resolve(), Path(output).resolve()
    clock(as_of)
    if not source.is_dir():
        raise ValueError('episode directory required')
    if source == target or source in target.parents:
        raise ValueError('measurement output must be outside the source corpus')
    counts = Counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites; this is a dated measurement receipt, not corpus repair.
    with target.open('x', encoding='utf-8') as sink:
        for path in source.rglob('*.json.gz'):
            counts['episode_files'] += 1
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as handle:
                    data = json.load(handle)
                if not isinstance(data, dict) or not data.get('token'):
                    raise ValueError('episode identity missing')
                episode = SimpleNamespace(token=data['token'],
                    execution_attempts=data.get('execution_attempts') or [],
                    market_observations=data.get('market_observations') or [])
                report = episode_measurements(episode, as_of=as_of)
                counts['attempts'] += report['attempts']
                for row in report['results']:
                    counts[row['status']] += 1
                    if row.get('real_fill') is True:
                        counts['verified_real_fill_attempts'] += 1
                    for horizon, result in row.get('horizons', {}).items():
                        counts['horizon_' + horizon + ':' + result['status']] += 1
                sink.write(json.dumps({'source_file': str(path), 'token': episode.token, **report}, allow_nan=False)+'\n')
            except (OSError, ValueError, TypeError, KeyError) as exc:
                counts['blocked_episode_files'] += 1
                sink.write(json.dumps({'source_file': str(path), 'status': 'DATA_BLOCKED_EPISODE_SCHEMA',
                                       'error_type': type(exc).__name__})+'\n')
        summary = {'kind': 'summary', 'as_of': as_of, 'counts': dict(counts),
                   'authority': 'MEASUREMENT_ONLY', 'economic_edge': 'UNMEASURED'}
        sink.write(json.dumps(summary)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--as-of', type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.episodes, args.output, as_of=args.as_of or time.time())))


if __name__ == '__main__':
    main()
