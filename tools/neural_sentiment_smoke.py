"""Offline synthetic pipeline/weights smoke; never a trading accuracy claim."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from src.research.neural_sentiment import SentimentResearch, REVISION


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    queue = SentimentResearch(args.output/'synthetic.sqlite', python=args.python, model_root=args.model_root)
    examples = [('en', 'I love this project, it is wonderful.'),
        ('en', 'This is terrible. I hate it.'), ('zh', '这个项目太棒了，我很喜欢。'),
        ('ja', 'このプロジェクトは素晴らしいです。'), ('de', 'Dieses Projekt ist schrecklich.'),
        ('ar', 'هذا المشروع رائع وأنا سعيد به.'),
        ('en', ' '.join(['A public example for a long text.']*180)+' This ending must be retained.')]
    for index, (language, text) in enumerate(examples):
        queue.enqueue({'text': text, 'source_id': 'synthetic:fixture', 'language': language,
            'platform': 'synthetic', 'observed_at': time.time()},
            source_job='synthetic:'+str(index), source_ref='synthetic:input:'+str(index))
    started, windows = time.perf_counter(), []
    try:
        for index in range(30):
            rows = await queue.fetch()
            if not rows and queue.report()['pending_inputs']:
                raise RuntimeError('offline synthetic inference failed: '+str(queue.report()['last_error_type']))
            queue.acknowledge(rows, 'synthetic:output:'+str(index))
            windows.extend(rows)
            if not queue.report()['pending_inputs']:
                break
        report = {'scope': 'SYNTHETIC_OFFLINE_PIPELINE_SMOKE_ONLY', 'model_revision': REVISION,
            'elapsed_seconds': time.perf_counter()-started, 'summary': queue.report(),
            'windows': windows, 'trading_accuracy': 'UNMEASURED', 'production_integrated': False}
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({key: report[key] for key in ('scope','model_revision','elapsed_seconds','summary')}))
    finally:
        await queue.stop()
        queue.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
