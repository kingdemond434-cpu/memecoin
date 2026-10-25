"""Native LaunchLab creation capture, across every platform configuration.

Account positions and argument schemas come from the pinned IDL. This is a
discovery decoder, not a curve quote or an assertion of execution support.
See docs/DONOR_ADAPTATIONS.md for source, fixture and license provenance.
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Callable

from src.chains.idl import load_idl
from src.chains.yellowstone_grpc import (
    apply_event_timing, extract_compute_budget, extract_system_transfers,
    instruction_fields, transaction_parts,
)

LAUNCHLAB_PROGRAM = "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj"
CREATE_NAMES = ("initialize", "initialize_v2", "initialize_with_token_2022")
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


class _Arguments:
    """The subset of Borsh used by the reviewed initialization schemas."""
    def __init__(self, raw: bytes, types: dict):
        self.raw, self.types, self.offset = raw, types, 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.raw):
            raise ValueError("truncated LaunchLab arguments")
        value = self.raw[self.offset:self.offset + count]
        self.offset += count
        return value

    def read(self, kind: Any) -> Any:
        if isinstance(kind, str) and kind in ("u8", "u16", "u32", "u64"):
            return int.from_bytes(self.take(int(kind[1:]) // 8), "little")
        if kind == "string":
            return self.take(self.read("u32")).decode("utf-8", errors="strict")
        if isinstance(kind, dict) and "option" in kind:
            present = self.read("u8")
            if present not in (0, 1):
                raise ValueError("invalid Borsh option")
            return self.read(kind["option"]) if present else None
        if isinstance(kind, dict) and "defined" in kind:
            schema = self.types[kind["defined"]["name"]]
            if schema["kind"] == "struct":
                return {item["name"]: self.read(item["type"]) for item in schema["fields"]}
            if schema["kind"] == "enum":
                index = self.read("u8")
                if index >= len(schema["variants"]):
                    raise ValueError("unknown LaunchLab enum variant")
                variant = schema["variants"][index]
                return {"variant": variant["name"], **{
                    item["name"]: self.read(item["type"]) for item in variant.get("fields", [])}}
        raise ValueError("unsupported LaunchLab argument schema")


class LaunchLabMonitor:
    program_id = LAUNCHLAB_PROGRAM

    def __init__(self, stream: Any, callback: Callable):
        self.callback = callback
        idl = load_idl("raydium_launchlab")
        if idl["address"] != LAUNCHLAB_PROGRAM:
            raise ValueError("LaunchLab IDL address changed")
        self.schemas = {bytes(row["discriminator"]): row for row in idl["instructions"]
                        if row["name"] in CREATE_NAMES}
        self.types = {row["name"]: row["type"] for row in idl["types"]}
        self.matched, self.rejected = Counter(), Counter()
        stream.on("transaction", self._on_transaction)

    def decode(self, keys: list, accounts: list, data: bytes) -> dict | None:
        schema = self.schemas.get(data[:8])
        if schema is None:
            return None
        if len(accounts) < len(schema["accounts"]):
            raise ValueError("truncated LaunchLab account list")
        named = {}
        for position, account in enumerate(schema["accounts"]):
            index = accounts[position]
            if type(index) is not int or not 0 <= index < len(keys):
                raise ValueError("invalid LaunchLab account index")
            named[account["name"]] = keys[index]
        if named["program"] != LAUNCHLAB_PROGRAM:
            raise ValueError("LaunchLab event program mismatch")
        expected_base = TOKEN_2022_PROGRAM if schema["name"] == CREATE_NAMES[2] else TOKEN_PROGRAM
        if named["base_token_program"] != expected_base or named["quote_token_program"] not in {
                TOKEN_PROGRAM, TOKEN_2022_PROGRAM}:
            raise ValueError("unsupported LaunchLab token program")
        reader = _Arguments(data[8:], self.types)
        args = {row["name"].lstrip("_"): reader.read(row["type"]) for row in schema["args"]}
        trailing = reader.raw[reader.offset:]
        # The official SDK serializes ten zero bytes even for Option::None.
        # Accept that specific reviewed representation, never arbitrary tails.
        if trailing and not (schema["name"] == CREATE_NAMES[2]
                             and args.get("transfer_fee_extension_param") is None
                             and trailing == bytes(10)):
            raise ValueError("unreviewed LaunchLab argument tail")
        mint, curve = args["base_mint_param"], args["curve_param"]
        return {
            "type": "token_launch", "chain": "solana", "program": LAUNCHLAB_PROGRAM,
            "venue": "raydium_launchlab", "instruction": schema["name"],
            "token": named["base_mint"], "creator": named["creator"],
            "payer": named["payer"], "pool": named["pool_state"],
            "quote_mint": named["quote_mint"], "quote_decimals": None,
            "base_vault": named["base_vault"], "quote_vault": named["quote_vault"],
            "base_token_program": named["base_token_program"],
            "quote_token_program": named["quote_token_program"],
            "platform_config": named["platform_config"], "global_config": named["global_config"],
            "name": mint["name"], "symbol": mint["symbol"], "uri": mint["uri"],
            "token_decimals": mint["decimals"], "curve_parameters": curve,
            "token_total_supply": curve["data"]["supply"],
            "vesting_parameters": args["vesting_param"],
            "amm_fee_on": args.get("amm_fee_on"),
            "transfer_fee_extension": args.get("transfer_fee_extension_param"),
            "data_status": "OBSERVED_NATIVE_INITIALIZATION",
            "execution_support": "DATA_BLOCKED_NATIVE_CURVE_EXECUTION",
            "protocol_ref": "idl/raydium_launchlab.json",
        }

    async def _on_transaction(self, tx_data: Any):
        received_ns = time.time_ns()
        if isinstance(tx_data, dict):
            meta = tx_data.get("result", tx_data).get("meta")
            if not isinstance(meta, dict) or meta.get("err") is not None:
                self.rejected["failed_or_missing_transaction_meta"] += 1
                return
        else:
            info = getattr(tx_data, "transaction", tx_data)
            meta = getattr(info, "meta", None)
            if meta is None or (hasattr(meta, "HasField") and meta.HasField("err")):
                self.rejected["failed_or_missing_transaction_meta"] += 1
                return
        keys, instructions, signature, slot = transaction_parts(tx_data)
        emitted = set()  # Per transaction, retains every distinct launch.
        for index, instruction in enumerate(instructions):
            try:
                program_index, accounts, data = instruction_fields(instruction)
                if not 0 <= program_index < len(keys) or keys[program_index] != self.program_id:
                    continue
                event = self.decode(keys, accounts, data)
                if event is None or event["token"] in emitted:
                    continue
                emitted.add(event["token"])
                event.update(signature=signature, slot=slot, instruction_index=index)
                budget = extract_compute_budget(keys, instructions, tx_data=tx_data)
                if budget.stated:
                    event.update(priority_lamports=budget.priority_lamports,
                                 compute_unit_price=budget.unit_price_micro_lamports,
                                 compute_unit_limit=budget.unit_limit,
                                 absolute_priority_lamports=budget.absolute_priority_lamports)
                transfers = extract_system_transfers(keys, instructions)
                event["funding_transfers"] = transfers
                event["funding_wallets"] = sorted({row["from"] for row in transfers
                    if row["to"] == event["creator"] and row["from"] != event["creator"]})
                apply_event_timing(event, tx_data, received_ns)
                self.matched[event["instruction"]] += 1
            except (ValueError, IndexError, KeyError, TypeError, AttributeError) as exc:
                self.rejected[type(exc).__name__] += 1
                continue
            result = self.callback(event)
            if asyncio.iscoroutine(result):
                await result

    def decoder_report(self) -> dict:
        return {"status": "WIRED", "program": LAUNCHLAB_PROGRAM,
                "matched": dict(self.matched), "rejected": dict(self.rejected),
                "platform_scope": "all observed configurations; no platform allowlist",
                "execution_support": "DATA_BLOCKED_NATIVE_CURVE_EXECUTION"}
