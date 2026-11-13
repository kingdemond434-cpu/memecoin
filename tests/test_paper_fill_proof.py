import json

from src.research.paper_fill_proof import PaperFillProof, main, run
from src.research.trade_evidence import TradeEvidenceLedger

NOW = 1_800_000_000.0


def _proof(tmp_path, **kw):
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("clock", lambda: NOW)
    return PaperFillProof(tmp_path / "trade_evidence.jsonl", tmp_path / "p.state.json", **kw)


def _buy(ledger, token, cost, at=NOW - 60, **extra):
    ledger.record("paper_fill", {"token": token, "side": "buy", "cost_usd": cost,
                                 "equity_usd": 1000.0, **extra}, timestamp=at)


def _sell(ledger, token, proceeds, allocated, closed, at=NOW - 30, pnl=None):
    ledger.record("paper_fill", {
        "token": token, "side": "sell", "proceeds_usd": proceeds, "native_delta_usd": -0.01,
        "allocated_cost_usd": allocated,
        "realized_pnl_usd": proceeds - 0.01 - allocated if pnl is None else pnl,
        "closed": closed, "equity_usd": 1000.0}, timestamp=at)


def test_round_trip_with_scale_in_and_partials_is_proven(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    ledger.record("candidate_decision", {"mint": "x"}, timestamp=NOW - 100)
    _buy(ledger, "A", 10.0)
    _buy(ledger, "A", 5.0, scale_in=True)
    _sell(ledger, "A", 20.0, 7.5, False)
    _sell(ledger, "A", 4.0, 7.5, True)
    proof = _proof(tmp_path)
    proof.scan()
    art = proof.artifact()
    assert art["verdict"] == "PROVEN", art["checks"]
    assert art["totals"]["round_trips"] == 1 and art["totals"]["buys"] == 2
    assert art["open_positions"] == 0
    assert abs(art["totals"]["realized_pnl_usd"] - (24.0 - 0.02 - 15.0)) < 1e-9
    assert art["totals"]["win_rate"] == 1.0
    assert art["evidence"]["rows_verified"] == 5


def test_pnl_that_does_not_reconcile_is_flagged(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    _buy(ledger, "A", 10.0)
    _sell(ledger, "A", 20.0, 10.0, True, pnl=99.0)
    proof = _proof(tmp_path)
    proof.scan()
    art = proof.artifact()
    assert art["verdict"] == "ACCOUNTING_MISMATCH"
    assert art["identity_examples"][0]["pnl"] == 99.0


def test_closed_position_must_allocate_its_whole_cost(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    _buy(ledger, "A", 10.0)
    _sell(ledger, "A", 20.0, 6.0, True)     # 4 USD of basis vanished
    proof = _proof(tmp_path)
    proof.scan()
    assert proof.artifact()["verdict"] == "ACCOUNTING_MISMATCH"


def test_verdict_order_no_fills_no_round_trip_stale(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    proof = _proof(tmp_path)
    proof.scan()
    assert proof.artifact()["verdict"] == "NO_FILLS"
    _buy(ledger, "A", 10.0, at=NOW - 3 * 86400)
    proof.scan()
    assert proof.artifact()["verdict"] == "NO_ROUND_TRIP"
    _sell(ledger, "A", 5.0, 10.0, True, at=NOW - 2 * 86400)
    proof.scan()
    art = proof.artifact()
    assert art["verdict"] == "STALE" and art["checks"]["accounting_identity"]


def test_incremental_scan_survives_restart_and_skips_partial_tail(tmp_path):
    path = tmp_path / "trade_evidence.jsonl"
    ledger = TradeEvidenceLedger(path)
    _buy(ledger, "A", 10.0)
    first = _proof(tmp_path)
    first.scan()
    offset = first.state["offset"]
    _sell(ledger, "A", 12.0, 10.0, True)
    with path.open("a") as handle:
        handle.write('{"partial":')                 # writer mid-append
    second = _proof(tmp_path)                       # restart from checkpoint
    assert second.state["offset"] == offset
    second.scan()
    art = second.artifact()
    assert art["totals"]["round_trips"] == 1 and art["totals"]["buys"] == 1
    assert art["checks"]["chain_ok"] and not art["checks"]["scan_complete"]


def test_tampered_or_shrunk_ledger_is_chain_broken(tmp_path):
    path = tmp_path / "trade_evidence.jsonl"
    ledger = TradeEvidenceLedger(path)
    _buy(ledger, "A", 10.0)
    _sell(ledger, "A", 12.0, 10.0, True)
    rows = path.read_text().splitlines()
    rows[1] = rows[1].replace('"proceeds_usd":12.0', '"proceeds_usd":120.0')
    path.write_text("\n".join(rows) + "\n")
    proof = _proof(tmp_path)
    proof.scan()
    assert proof.artifact()["verdict"] == "CHAIN_BROKEN"

    other = tmp_path / "b"
    other.mkdir()
    ledger = TradeEvidenceLedger(other / "trade_evidence.jsonl")
    _buy(ledger, "A", 10.0)
    proof = _proof(other)
    proof.scan()
    (other / "trade_evidence.jsonl").write_text("")
    proof.scan()
    assert proof.artifact()["verdict"] == "CHAIN_BROKEN"


def test_byte_budget_reports_catching_up(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    for i in range(50):
        ledger.record("candidate_decision", {"mint": str(i), "pad": "x" * 200})
    proof = _proof(tmp_path, max_bytes=2000)
    proof.scan()
    assert proof.artifact()["verdict"] == "CATCHING_UP"
    for _ in range(20):
        proof.scan()
    assert proof.artifact()["checks"]["scan_complete"]


def test_run_and_cli_write_separate_artifacts(tmp_path):
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    _buy(ledger, "A", 10.0, at=1.0)
    art = run(tmp_path)
    assert json.loads((tmp_path / "paper_fill_proof.json").read_text())["schema"] == art["schema"]
    assert main(["--state-dir", str(tmp_path)]) == 1          # not PROVEN, exit 1
    assert (tmp_path / "paper_fill_proof_check.json").exists()
    assert (tmp_path / "paper_fill_proof.state.json").exists()


def test_desk_paper_exits_reconcile_through_the_proof(tmp_path):
    import asyncio
    asyncio.run(_desk_round_trip(tmp_path))


async def _desk_round_trip(tmp_path):
    """The desk's own sell rows, after a partial and a closing exit, must
    satisfy the proof's accounting identity and close the round trip."""
    from tests.test_core import (ExecutionResult, FakeExecutionEngineForExit,
                                 MemecoinQuantDesk, TestPartialExitAccounting,
                                 TransactionStatus)
    ledger = TradeEvidenceLedger(tmp_path / "trade_evidence.jsonl")
    result = ExecutionResult(success=True, status=TransactionStatus.SIMULATED, simulated=True,
                             quoted_output_amount=150_000_000, native_balance_delta_lamports=0,
                             actual_input_amount=750)
    desk = TestPartialExitAccounting()._desk(result)
    desk.trade_evidence = ledger
    position = TestPartialExitAccounting._position()
    _buy(ledger, "mint", float(position["remaining_cost_usd"]), at=None)
    desk.elogw_engine.update_position("mint", position)
    await MemecoinQuantDesk._execute_exit(desk, "mint", position, 0.75, "profit_ratchet_5x")
    position = desk.elogw_engine.open_positions["mint"]
    desk.execution_engine = FakeExecutionEngineForExit(ExecutionResult(
        success=True, status=TransactionStatus.SIMULATED, simulated=True,
        quoted_output_amount=37_500_000, native_balance_delta_lamports=0,
        actual_input_amount=250))
    await MemecoinQuantDesk._execute_exit(desk, "mint", position, 1.0, "profit_ratchet_10x")
    proof = PaperFillProof(tmp_path / "trade_evidence.jsonl", tmp_path / "s.json",
                           sleep=lambda s: None)
    proof.scan()
    art = proof.artifact()
    assert art["verdict"] == "PROVEN", (art["checks"], art["identity_examples"])
    assert art["totals"]["sells"] == 2 and art["totals"]["round_trips"] == 1
    assert abs(art["totals"]["realized_pnl_usd"] - 87.5) < 1e-6


def test_break_latches_until_the_ledger_is_replaced(tmp_path):
    path = tmp_path / "trade_evidence.jsonl"
    ledger = TradeEvidenceLedger(path)
    _buy(ledger, "A", 10.0)
    _sell(ledger, "A", 12.0, 10.0, True)
    rows = path.read_text().splitlines()
    path.write_text(rows[0] + "\n" + rows[1].replace('"proceeds_usd":12.0', '"proceeds_usd":120.0') + "\n")
    proof = _proof(tmp_path)
    proof.scan()
    assert proof.artifact()["verdict"] == "CHAIN_BROKEN"
    with path.open("a") as handle:                  # appending does not heal it
        handle.write(rows[1] + "\n")
    proof.scan()
    assert proof.artifact()["verdict"] == "CHAIN_BROKEN"
    path.unlink()                                   # a new segment / ledger
    fresh = TradeEvidenceLedger(path)
    _buy(fresh, "B", 10.0)
    _sell(fresh, "B", 11.0, 10.0, True)
    proof.scan()
    art = proof.artifact()
    assert art["verdict"] == "PROVEN", art["checks"]
    assert len(art["chain_restarts"]) == 1 and "row 2" in art["chain_restarts"][0]["error"]
    assert art["totals"]["round_trips"] == 1


def test_health_loop_runs_the_proof_as_its_own_task(tmp_path, monkeypatch):
    import asyncio
    from src.runtime.reporting import ReportingSurface
    calls = []

    class Surface(ReportingSurface):
        _running = True
        memory = type("M", (), {"observe": staticmethod(lambda: None)})()
        global_config = {"ops_state_dir": str(tmp_path)}

        async def _readiness_snapshot(self):
            return {}

        async def _publish_readiness(self, snapshot):
            self._running = False

    async def fast(_):
        return None

    monkeypatch.setattr("src.runtime.reporting.asyncio.sleep", fast)

    async def go():
        surface = Surface()
        await surface._health_loop()
        assert surface._paper_proof_task is not None
        await surface._paper_proof_task
        calls.append(surface._paper_proof_verdict)

    asyncio.run(go())
    assert calls == ["NO_FILLS"]
    assert (tmp_path / "paper_fill_proof.json").exists()
