#!/usr/bin/env python3
"""Find one real pool-creation transaction per AMM program, no jq/sudo needed."""
import json
import os
import time
import urllib.request

RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
OUTDIR = "fixture_candidates"
os.makedirs(OUTDIR, exist_ok=True)

PROGRAMS = [
    ("CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C", "raydium_cpmm", ["instruction: initialize"]),
    ("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK", "raydium_clmm", ["instruction: createpool"]),
    ("LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo", "meteora_dlmm", ["instruction: initializelbpair"]),
    ("Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB", "meteora_dynamic_amm", ["instruction: initializepermissionless"]),
    ("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc", "orca_whirlpool", ["instruction: initializepool"]),
]


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


for program, label, patterns in PROGRAMS:
    print(f"=== {label} ({program}) ===")
    found = False
    before = None
    for page in range(1, 11):
        params = [program, {"limit": 100, **({"before": before} if before else {})}]
        try:
            sigs_resp = rpc("getSignaturesForAddress", params)
        except Exception as exc:
            print(f"  RPC error: {exc}")
            break
        sigs = [item["signature"] for item in (sigs_resp.get("result") or []) if not item.get("err")]
        if not sigs:
            print("  no more signatures, stopping")
            break
        before = sigs[-1]
        for sig in sigs:
            try:
                tx = rpc("getTransaction", [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}])
            except Exception as exc:
                print(f"  RPC error on {sig}: {exc}")
                time.sleep(0.2)
                continue
            result = tx.get("result") or {}
            logs = [line.lower() for line in ((result.get("meta") or {}).get("logMessages") or [])]
            if any(any(p in line for p in patterns) for line in logs):
                slot = result.get("slot")
                print(f"  MATCH: {sig} (slot {slot}) page={page}")
                with open(f"{OUTDIR}/{label}_{sig[:12]}.json", "w") as handle:
                    json.dump(tx, handle)
                found = True
                break
            time.sleep(0.15)
        if found:
            break
        print(f"  scanned page {page} ({len(sigs)} sigs), no match yet, paging back...")
    if not found:
        print(f"  !! no pool-creation tx found in scanned window for {label}")

print()
print(f"Saved candidates in {OUTDIR}/:")
for name in sorted(os.listdir(OUTDIR)):
    print(" ", name)
