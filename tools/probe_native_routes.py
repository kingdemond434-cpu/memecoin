"""Run bounded read-only native EVM probes from a deployment configuration.

This diagnostic writes a fresh report only. The running desk retains its own
receipts in its canonical ledger; a second process must not append there.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

import yaml

from src.chains.evm_native import EVMV2NativeProbe
from src.chains.rpc_manager import ChainRegistry, ChainType


async def run(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    declarations = config.get("global", {}).get("evm_native_probes", [])
    if not isinstance(declarations, list) or not 0 < len(declarations) <= 4:
        raise ValueError("one to four explicit native probes required")
    registry = ChainRegistry(str(config_path))
    results = []
    try:
        chains = {row["chain"] for row in declarations}
        if any(registry.get_chain(chain) is None or registry.get_chain(chain).chain_type != ChainType.EVM for chain in chains):
            raise ValueError("native EVM chains must be declared in the existing registry")
        await registry.start_all(chains)
        for row in declarations:
            chain = row["chain"]
            probe = EVMV2NativeProbe(registry.get_rpc(chain), chain=chain,
                chain_id=registry.get_chain(chain).chain_id,
                **{key: value for key, value in row.items() if key != "chain"})
            results.append(await probe.probe())
    finally:
        await registry.stop_all()
    return {"captured_at": time.time(), "native_probes": results,
            "authority": "none", "execution_eligible": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to overwrite a retained report")
    result = asyncio.run(run(args.config))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"output": str(args.output), "statuses": [row["status"] for row in result["native_probes"]]}))
    return 0 if all(row["status"] == "NATIVE_QUOTE_PARITY_VERIFIED" for row in result["native_probes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
