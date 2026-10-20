"""Read-only Uniswap V2 native reserve, quote-parity and independent-leg probes.

References: Uniswap/v2-periphery/contracts/libraries/UniswapV2Library.sol;
EIP-1898 blockHash/requireCanonical. All state reads are pinned to one safe or
finalized block. No private keys, signing or transaction submission exist here.
Router quotes alone do not certify token transfers, gas costs or future exits.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from eth_utils import keccak

from src.chains.evm_events import hex_data, native_address, quantity

ABI_SOURCE = "https://github.com/Uniswap/v2-core/blob/master/contracts/interfaces/IUniswapV2Pair.sol"
MATH_SOURCE = "https://github.com/Uniswap/v2-periphery/blob/master/contracts/libraries/UniswapV2Library.sol"
UINT256_MAX = 2**256 - 1


def uint(value: Any, *, positive=False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value <= UINT256_MAX:
        raise ValueError("canonical uint256 required")
    return value


def address(value: Any) -> str:
    value = native_address(value)
    if int(value, 16) == 0:
        raise ValueError("nonzero native address required")
    return value


def selector(signature: str) -> str:
    return "0x" + keccak(text=signature)[:4].hex()


def word(value: int) -> str:
    return uint(value).to_bytes(32, "big").hex()


def address_word(value: str) -> str:
    return "0" * 24 + address(value)[2:]


def decode_address(raw: Any) -> str:
    data = hex_data(raw, 32)
    if any(data[:12]):
        raise ValueError("noncanonical ABI address")
    return address("0x" + data[-20:].hex())


def decode_amounts(raw: Any, *, length=2) -> list[int]:
    data = hex_data(raw, (length + 2) * 32)
    words = [int.from_bytes(data[i:i+32], "big") for i in range(0, len(data), 32)]
    if words[:2] != [32, length]:
        raise ValueError("noncanonical bounded ABI amounts array")
    return words[2:]


def amount_out(amount: int, reserve_in: int, reserve_out: int) -> int:
    """Exact 997/1000 core arithmetic, including Solidity SafeMath limits."""
    uint(amount, positive=True)
    uint(reserve_in, positive=True)
    uint(reserve_out, positive=True)
    with_fee = uint(amount * 997)
    numerator = uint(with_fee * reserve_out)
    denominator = uint(uint(reserve_in * 1000) + with_fee)
    result = numerator // denominator
    if result <= 0 or result >= reserve_out:
        raise ValueError("native quote has no executable output")
    return result


class EVMV2NativeProbe:
    """A registry-owned RPC client plus an explicitly declared native route.

    getPair, pair.factory and router.factory must agree; code must exist on all
    contracts. Both directions must match router getAmountsOut at the same
    canonical block hash. A probe is research evidence, never live authority.
    """
    def __init__(self, rpc, *, chain: str, chain_id: int, factory: str, router: str,
                 token_in: str, token_out: str, amount_in: int, source_ref: str,
                 finality_tag="finalized", wallet: str | None = None, timeout_seconds=20):
        if finality_tag not in {"safe", "finalized"} or not chain or not source_ref:
            raise ValueError("native chain, official source and safe/finalized policy required")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
            raise ValueError("bounded native probe timeout required")
        self.rpc, self.chain, self.chain_id = rpc, chain, uint(chain_id, positive=True)
        self.factory, self.router = address(factory), address(router)
        self.token_in, self.token_out = address(token_in), address(token_out)
        if self.token_in == self.token_out:
            raise ValueError("distinct route tokens required")
        self.amount_in, self.source_ref = uint(amount_in, positive=True), source_ref
        self.finality_tag, self.wallet = finality_tag, address(wallet) if wallet else None
        self.timeout_seconds = timeout_seconds

    async def probe(self) -> dict:
        self._stage = "CHAIN_FINALITY"
        try:
            return await asyncio.wait_for(self._probe(), timeout=self.timeout_seconds)
        except Exception as exc:
            # Provider exception strings may contain credential-bearing URLs.
            return {"status": "DATA_BLOCKED", "chain": self.chain,
                    "failure_type": type(exc).__name__, "observed_at": time.time(),
                    "stage": self._stage,
                    "observation_only": True, "execution_eligible": False, "authority": "none"}

    async def _probe(self) -> dict:
        chain_id, block = await asyncio.gather(self.rpc.request("eth_chainId", []),
            self.rpc.request("eth_getBlockByNumber", [self.finality_tag, False]))
        if quantity(chain_id) != self.chain_id or not isinstance(block, dict):
            raise ValueError("provider native chain mismatch or finality unavailable")
        block_hash = "0x" + hex_data(block["hash"], 32).hex()
        height, source_at = quantity(block["number"]), quantity(block["timestamp"])
        if source_at <= 0 or source_at > time.time() + 5:
            raise ValueError("future native block time")
        pinned = {"blockHash": block_hash, "requireCanonical": True}

        async def call(target, signature, args="", *, sender=None):
            tx = {"to": target, "data": selector(signature) + args}
            if sender:
                tx["from"] = sender
            return await self.rpc.request("eth_call", [tx, pinned])

        self._stage = "NATIVE_ROUTE_IDENTITY"
        pair_raw, router_factory = await asyncio.gather(
            call(self.factory, "getPair(address,address)", address_word(self.token_in) + address_word(self.token_out)),
            call(self.router, "factory()"))
        pair = decode_address(pair_raw)
        if decode_address(router_factory) != self.factory:
            raise ValueError("router factory mismatch")
        self._stage = "NATIVE_CODE_AND_RESERVES"
        codes = await asyncio.gather(*(self.rpc.request("eth_getCode", [contract, pinned])
                                      for contract in (self.factory, self.router, pair)))
        if any(not hex_data(code) for code in codes):
            raise ValueError("native contract bytecode unavailable")
        factory_raw, token0_raw, token1_raw, reserves_raw = await asyncio.gather(
            call(pair, "factory()"), call(pair, "token0()"), call(pair, "token1()"), call(pair, "getReserves()"))
        token0, token1 = decode_address(token0_raw), decode_address(token1_raw)
        if (decode_address(factory_raw) != self.factory or int(token0, 16) >= int(token1, 16)
                or {token0, token1} != {self.token_in, self.token_out}):
            raise ValueError("pair identity does not match native route")
        reserves = hex_data(reserves_raw, 96)
        reserve0, reserve1, last_update = [int.from_bytes(reserves[i:i+32], "big") for i in (0, 32, 64)]
        if reserve0 >= 2**112 or reserve1 >= 2**112 or last_update >= 2**32:
            raise ValueError("reserve ABI uint widths mismatch")
        reserve_in, reserve_out = (reserve0, reserve1) if self.token_in == token0 else (reserve1, reserve0)
        bought = amount_out(self.amount_in, reserve_in, reserve_out)
        # Exit is evaluated against the same observed state, without pretending
        # an entry changed that state or that future depth is already observed.
        sold = amount_out(bought, reserve_out, reserve_in)

        def quote_args(amount, path):
            return word(amount) + word(64) + word(2) + "".join(address_word(token) for token in path)

        self._stage = "NATIVE_QUOTE_PARITY"
        buy_raw, sell_raw = await asyncio.gather(
            call(self.router, "getAmountsOut(uint256,address[])", quote_args(self.amount_in, [self.token_in, self.token_out])),
            call(self.router, "getAmountsOut(uint256,address[])", quote_args(bought, [self.token_out, self.token_in])))
        if decode_amounts(buy_raw) != [self.amount_in, bought] or decode_amounts(sell_raw) != [bought, sold]:
            raise ValueError("router quote disagrees with exact native reserve arithmetic")
        simulations = []
        if self.wallet:
            self._stage = "INDEPENDENT_LEG_SIMULATION"
            for amount, path, output in ((self.amount_in, [self.token_in, self.token_out], bought),
                                         (bought, [self.token_out, self.token_in], sold)):
                balance_raw, allowance_raw = await asyncio.gather(
                    call(path[0], "balanceOf(address)", address_word(self.wallet)),
                    call(path[0], "allowance(address,address)", address_word(self.wallet) + address_word(self.router)))
                if min(int.from_bytes(hex_data(balance_raw, 32), "big"),
                       int.from_bytes(hex_data(allowance_raw, 32), "big")) < amount:
                    raise ValueError("independent-leg wallet balance or allowance unavailable")
                args = (word(amount) + word(output) + word(160) + address_word(self.wallet)
                        + word(source_at + 300) + word(2) + "".join(address_word(token) for token in path))
                data = selector("swapExactTokensForTokens(uint256,uint256,address[],address,uint256)") + args
                tx = {"from": self.wallet, "to": self.router, "data": data}
                simulated = await self.rpc.request("eth_call", [tx, pinned])
                if decode_amounts(simulated) != [amount, output]:
                    raise ValueError("native transfer simulation disagrees with quoted output")
                # Gas estimates remain estimates, explicitly independent of
                # realized gas or future route/fill reliability.
                gas = quantity(await self.rpc.request("eth_estimateGas", [tx, pinned]))
                if gas <= 0:
                    raise ValueError("native gas estimate unavailable")
                simulations.append({"input_amount": amount, "output_amount": output,
                    "estimated_gas_units": gas, "scope": "INDEPENDENT_LEG_ETH_CALL"})
        self._stage = "CANONICAL_BLOCK_RECHECK"
        current = await self.rpc.request("eth_getBlockByNumber", [hex(height), False])
        if not isinstance(current, dict) or str(current.get("hash", "")).lower() != block_hash:
            raise ValueError("native block changed during probe")
        return {"status": "NATIVE_QUOTE_PARITY_VERIFIED", "chain": self.chain, "chain_id": self.chain_id,
            "protocol": "uniswap_v2", "factory": self.factory, "router": self.router, "pair": pair,
            "token_in": self.token_in, "token_out": self.token_out,
            "input_amount": self.amount_in, "quoted_buy_output": bought, "quoted_sell_output": sold,
            "reserves": [reserve0, reserve1], "reserve_timestamp_uint32": last_update,
            "block_number": height, "block_hash": block_hash, "finality": self.finality_tag.upper(),
            "event_at": source_at, "observed_at": time.time(), "source_ref": self.source_ref,
            "abi_ref": ABI_SOURCE, "math_ref": MATH_SOURCE,
            "code_sha256": [hashlib.sha256(hex_data(code)).hexdigest() for code in codes],
            "simulations": simulations, "transfer_feasibility": "INDEPENDENT_LEGS_ONLY" if simulations else "UNMEASURED",
            "realized_cost": None, "future_exit_feasibility": "UNMEASURED",
            "observation_only": True, "execution_eligible": False, "authority": "none"}
