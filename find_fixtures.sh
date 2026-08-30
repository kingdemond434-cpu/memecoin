#!/usr/bin/env bash
set -euo pipefail

RPC_URL="${SOLANA_RPC_URL:-https://api.mainnet-beta.solana.com}"
OUTDIR="./fixture_candidates"
mkdir -p "$OUTDIR"

rpc() {
  curl -sS --max-time 20 "$RPC_URL" -H "Content-Type: application/json" -d "$1"
}

PROGRAMS=(
  "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C|raydium_cpmm|Instruction: Initialize$"
  "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK|raydium_clmm|Instruction: CreatePool"
  "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo|meteora_dlmm|Instruction: InitializeLbPair"
  "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB|meteora_dynamic_amm|Instruction: InitializePermissionless"
  "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc|orca_whirlpool|Instruction: InitializePool"
)

for entry in "${PROGRAMS[@]}"; do
  IFS='|' read -r PROGRAM LABEL PATTERN <<< "$entry"
  echo "=== $LABEL ($PROGRAM) ==="
  FOUND=""
  BEFORE=""
  for page in $(seq 1 10); do
    if [ -z "$BEFORE" ]; then
      PARAMS="[\"$PROGRAM\", {\"limit\": 100}]"
    else
      PARAMS="[\"$PROGRAM\", {\"limit\": 100, \"before\": \"$BEFORE\"}]"
    fi
    SIGS_JSON=$(rpc "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getSignaturesForAddress\",\"params\":$PARAMS}")
    SIGS=$(echo "$SIGS_JSON" | jq -r '.result[]? | select(.err == null) | .signature')
    COUNT=$(echo "$SIGS" | grep -c . || true)
    if [ "$COUNT" -eq 0 ]; then
      echo "  no more signatures, stopping pagination"
      break
    fi
    BEFORE=$(echo "$SIGS" | tail -n1)
    for SIG in $SIGS; do
      TX=$(rpc "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getTransaction\",\"params\":[\"$SIG\", {\"encoding\":\"json\",\"maxSupportedTransactionVersion\":0}]}")
      LOGS=$(echo "$TX" | jq -r '.result.meta.logMessages[]?' 2>/dev/null || true)
      if echo "$LOGS" | grep -qiE "$PATTERN"; then
        echo "  MATCH: $SIG (slot $(echo "$TX" | jq -r '.result.slot')) page=$page"
        echo "$TX" > "$OUTDIR/${LABEL}_${SIG:0:12}.json"
        FOUND=1
        break 2
      fi
      sleep 0.15
    done
    echo "  scanned page $page (${COUNT} sigs), no match yet, paging back..."
  done
  if [ -z "$FOUND" ]; then
    echo "  !! no pool-creation tx found in scanned window for $LABEL"
  fi
done

echo
echo "Saved candidates in $OUTDIR/:"
ls -la "$OUTDIR/" 2>/dev/null || true
