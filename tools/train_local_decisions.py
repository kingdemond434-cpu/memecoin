"""Train our local typed classifier from evidenced chronological labels."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research.local_decisions import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--question", required=True)
    p.add_argument("--primitive", choices=["choice", "score", "noul"], required=True)
    p.add_argument("--options", nargs="+", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists(): p.error("refusing to overwrite a frozen model artifact")
    rows = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    artifact = train(rows, question_id=args.question, primitive=args.primitive, options=args.options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    print(json.dumps({"model_sha256": artifact["sha256"], "validation": artifact["validation"], "counts": artifact["counts"]}))


if __name__ == "__main__": main()
