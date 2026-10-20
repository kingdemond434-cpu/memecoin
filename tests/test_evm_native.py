import asyncio
import copy
import time
from types import SimpleNamespace

from eth_abi import decode, encode
import pytest

from src.chains.evm_native import EVMV2NativeProbe, UINT256_MAX, amount_out, selector
from src.research.global_research_miner import GlobalResearchMiner

FACTORY, ROUTER, PAIR, TOKEN0, TOKEN1, WALLET = ["0x" + byte * 20 for byte in ("11", "22", "33", "44", "55", "66")]
HASH = "0x" + "ab" * 32
BLOCK = {"hash": HASH, "number": "0x100", "timestamp": hex(int(time.time()) - 30)}


class RPC:
    def __init__(self):
        self.calls = []
        self.reorg = self.disagree = self.fail_simulation = self.wrong_chain = self.empty_code = False

    async def request(self, method, params):
        self.calls.append((method, copy.deepcopy(params)))
        if method == "eth_chainId":
            return "0x2" if self.wrong_chain else "0x1"
        if method == "eth_getBlockByNumber":
            return {**BLOCK, "hash": "0x"+"cd"*32} if self.reorg and params[0] == "0x100" else BLOCK
        if method == "eth_getCode":
            return "0x" if self.empty_code else "0x60006000"
        if method == "eth_estimateGas":
            return "0x20000"
        assert method == "eth_call", "native probe attempted an unexpected or submitting RPC"
        tx, pinned = params
        assert pinned == {"blockHash": HASH, "requireCanonical": True}
        data, target = tx["data"], tx["to"]
        sig = data[:10]
        types, values = None, None
        if sig == selector("getPair(address,address)"):
            types, values = ["address"], [PAIR]
        elif sig == selector("factory()"):
            types, values = ["address"], [FACTORY]
        elif sig == selector("token0()"):
            types, values = ["address"], [TOKEN0]
        elif sig == selector("token1()"):
            types, values = ["address"], [TOKEN1]
        elif sig == selector("getReserves()"):
            types, values = ["uint112", "uint112", "uint32"], [1_000_000, 2_000_000, 1]
        elif sig in {selector("balanceOf(address)"), selector("allowance(address,address)")}:
            assert target in {TOKEN0, TOKEN1}
            types, values = ["uint256"], [UINT256_MAX]
        elif sig == selector("getAmountsOut(uint256,address[])"):
            amount, path = decode(["uint256", "address[]"], bytes.fromhex(data[10:]))
            reserves = (1_000_000, 2_000_000) if path[0] == TOKEN0 else (2_000_000, 1_000_000)
            types, values = ["uint256[]"], [[amount, amount_out(amount, *reserves) + int(self.disagree)]]
        elif sig == selector("swapExactTokensForTokens(uint256,uint256,address[],address,uint256)"):
            if self.fail_simulation:
                raise RuntimeError("sensitive endpoint with credential must never be printed")
            amount, output, path, wallet, deadline = decode(["uint256", "uint256", "address[]", "address", "uint256"], bytes.fromhex(data[10:]))
            assert wallet == WALLET and path[0] in {TOKEN0, TOKEN1} and deadline > int(BLOCK["timestamp"], 16)
            types, values = ["uint256[]"], [[amount, output]]
        else:
            raise AssertionError(sig)
        return "0x" + encode(types, values).hex()


def adapter(rpc, **kwargs):
    return EVMV2NativeProbe(rpc, chain="ethereum", chain_id=1, factory=FACTORY, router=ROUTER,
        token_in=TOKEN0, token_out=TOKEN1, amount_in=1000, source_ref="official-source-fixture", **kwargs)


def test_native_math_uses_exact_solidity_rounding_and_overflow():
    assert amount_out(1000, 1_000_000, 2_000_000) == 1992
    for values in ((True, 1, 2), (0, 1, 2), (1, 0, 2), (UINT256_MAX, 1, 2), (1, UINT256_MAX, 1)):
        with pytest.raises(ValueError):
            amount_out(*values)


def test_native_quote_parity_is_pinned_and_has_no_transfer_or_live_claim():
    rpc = RPC()
    result = asyncio.run(adapter(rpc).probe())
    assert result["status"] == "NATIVE_QUOTE_PARITY_VERIFIED"
    assert result["quoted_buy_output"] == 1992 and result["quoted_sell_output"] == 992
    assert result["transfer_feasibility"] == "UNMEASURED"
    assert result["realized_cost"] is None and not result["execution_eligible"]
    assert all(params[1] == {"blockHash": HASH, "requireCanonical": True}
               for method, params in rpc.calls if method in {"eth_call", "eth_getCode"})


def test_both_independent_legs_simulate_with_real_balance_allowance_checks():
    result = asyncio.run(adapter(RPC(), wallet=WALLET).probe())
    assert result["status"] == "NATIVE_QUOTE_PARITY_VERIFIED"
    assert len(result["simulations"]) == 2
    assert result["transfer_feasibility"] == "INDEPENDENT_LEGS_ONLY"
    assert result["future_exit_feasibility"] == "UNMEASURED"
    assert all(row["estimated_gas_units"] > 0 for row in result["simulations"])


@pytest.mark.parametrize("fault", ["reorg", "disagree", "wrong_chain", "empty_code", "fail_simulation"])
def test_native_mismatch_reorg_or_failed_exit_blocks_without_leaking_secret(fault):
    rpc = RPC()
    setattr(rpc, fault, True)
    result = asyncio.run(adapter(rpc, wallet=WALLET).probe())
    assert result["status"] == "DATA_BLOCKED" and not result["execution_eligible"]
    assert "credential" not in str(result) and "reason" not in result


def test_existing_collection_loop_retains_probes_and_rotates_without_another_task(tmp_path):
    receipts = []
    miner = GlobalResearchMiner(None, ledger_path=str(tmp_path/"leads"), native_probes=[adapter(RPC())],
        native_probe_sink=lambda receipt: receipts.append(receipt) or "canonical-ref")
    result = asyncio.run(miner._run_observation_pass())
    assert result["native_quote_probes"][0]["evidence_ref"] == "canonical-ref"
    assert receipts[0]["status"] == "NATIVE_QUOTE_PARITY_VERIFIED"
    assert not result["execution_eligible"]


def test_runtime_native_receipt_reaches_the_real_typed_data_consumer(tmp_path):
    from src.runtime.research_runtime import ResearchRuntime
    from test_research_workflows import create
    engine = create(tmp_path)
    desk = SimpleNamespace(unknown_engine=engine)
    receipt = asyncio.run(adapter(RPC()).probe())
    ref = ResearchRuntime._retain_native_probe(desk, receipt)
    for _ in range(3):
        engine.run_pass()
    data = engine.workflows.data.report()
    assert data["variables"] == {"native_v2_buy_output": 1, "native_v2_sell_output": 1}
    assert engine.evidence.verify(engine.evidence.path)[0] and ref.startswith("evidence:")
    assert not engine.framework.canary_models and not engine.framework.champions


def test_same_native_block_route_size_does_not_manufacture_samples_after_restart(tmp_path):
    from src.runtime.research_runtime import ResearchRuntime
    from test_research_workflows import create
    engine = create(tmp_path)
    receipt = asyncio.run(adapter(RPC()).probe())
    ResearchRuntime._retain_native_probe(SimpleNamespace(unknown_engine=engine), receipt)
    for _ in range(3):
        engine.run_pass()
    restored = create(tmp_path)
    repeated = {**receipt, "observed_at": time.time()}
    ResearchRuntime._retain_native_probe(SimpleNamespace(unknown_engine=restored), repeated)
    for _ in range(3):
        restored.run_pass()
    assert restored.workflows.data.report()["variables"] == {"native_v2_buy_output": 1, "native_v2_sell_output": 1}
