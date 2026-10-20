from src.runtime.latency import LatencyLedger


def test_all_unmeasured_budget_stages_are_blocked_not_healthy():
    report = LatencyLedger().budget_report({"dispatch_to_decide": 5000})
    assert report["status"] == "DATA_BLOCKED"
    assert report["unmeasured_stages"] == ["dispatch_to_decide"]
    assert report["stages"][0]["p99_us"] is None
    assert report["stages"][0]["over_budget"] is None


def test_one_measured_stage_cannot_make_an_unmeasured_stage_healthy():
    ledger = LatencyLedger()
    trace = ledger.open("fixture")
    ledger.mark("fixture", "dispatch_to_decide", trace.started_ns + 1000)
    ledger.close("fixture")
    report = ledger.budget_report({"dispatch_to_decide": 5000, "sign_to_submit": 5000})
    assert report["status"] == "DEGRADED"
    assert report["unmeasured_stages"] == ["sign_to_submit"]


def test_measured_breach_remains_a_breach_alongside_missing_measurements():
    ledger = LatencyLedger()
    trace = ledger.open("fixture")
    ledger.mark("fixture", "dispatch_to_decide", trace.started_ns + 10_000_000)
    ledger.close("fixture")
    report = ledger.budget_report({"dispatch_to_decide": 5000, "sign_to_submit": 5000})
    assert report["status"] == "DEGRADED"
    assert report["breaching"] == ["dispatch_to_decide"]


def test_only_measured_compliant_budgets_are_healthy():
    ledger = LatencyLedger()
    assert ledger.budget_report({})["status"] == "DATA_BLOCKED"
    trace = ledger.open("fixture")
    ledger.mark("fixture", "dispatch_to_decide", trace.started_ns + 1000)
    ledger.close("fixture")
    assert ledger.budget_report({"dispatch_to_decide": 5000})["status"] == "OK"
