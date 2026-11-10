"""Two-sided miss audit: a screen's discarded monsters are priced against the
rugs it avoided in the same cohort, so the audit cannot argue from survivors."""
from src.research.launch_census import LaunchCensus


def _census(tmp_path):
    return LaunchCensus(tmp_path / "census.json")


def test_screen_is_priced_on_both_sides(tmp_path):
    c = _census(tmp_path)
    for i in range(4):
        c.see(f"s{i}", at=1.0)
        c.screen(f"s{i}", "low_liquidity")
    c.resolve("s0", peak_multiple=25.0)             # monster the screen discarded
    for mint in ("s1", "s2", "s3"):                 # rugs the same screen avoided
        c.resolve(mint, peak_multiple=1.1, rugged=True, rug_mechanism="lp_pull")
    c.see("r0", at=1.0); c.reject("r0", "safety_veto:mint_authority_active")
    c.resolve("r0", peak_multiple=0.9, rugged=True, rug_mechanism="mint")
    c.see("e0", at=1.0); c.decide("e0", "ENTER"); c.enter("e0")
    c.resolve("e0", peak_multiple=12.0)
    c.see("u0", at=1.0)                             # unresolved: never counted as flat

    audit = c.miss_audit()
    assert audit["monsters_resolved"] == 2 and audit["rugs_resolved"] == 4
    assert audit["by_class"]["decision_screened_out"] == {"monsters": 1, "rugs": 3}
    assert audit["by_class"]["decision_declined"] == {"monsters": 0, "rugs": 1}
    assert audit["by_class"]["entered"] == {"monsters": 1, "rugs": 0}
    rows = {row["reason"]: row for row in audit["screens_two_sided"]}
    low = rows["low_liquidity"]
    assert (low["monsters_discarded"], low["rugs_avoided"], low["total_rejected"]) == (1, 3, 4)
    assert abs(low["monsters_per_rug_avoided"] - 1 / 3) < 1e-9
    veto = rows["safety_veto:mint_authority_active"]
    assert (veto["monsters_discarded"], veto["rugs_avoided"]) == (0, 1)
    assert audit["policy_change_authority"] is False
    assert "detection_latency" in audit["unmeasured"]
    assert c.report()["miss_audit"] == audit


def test_no_rugs_avoided_is_null_not_infinite_or_zero(tmp_path):
    c = _census(tmp_path)
    c.see("a", at=1.0); c.screen("a", "age_gate"); c.resolve("a", peak_multiple=50.0)
    row = c.miss_audit()["screens_two_sided"][0]
    assert row["rugs_avoided"] == 0 and row["monsters_per_rug_avoided"] is None


def test_rug_side_survives_save_and_load(tmp_path):
    c = _census(tmp_path)
    c.see("a", at=1.0); c.screen("a", "age_gate")
    c.resolve("a", peak_multiple=1.0, rugged=True, rug_mechanism="lp_pull")
    assert c.save()
    again = _census(tmp_path)
    assert again.load()
    assert again.miss_audit() == c.miss_audit()


def test_legacy_state_without_rug_side_loads(tmp_path):
    c = _census(tmp_path)
    c.see("a", at=1.0); c.screen("a", "age_gate"); c.resolve("a", peak_multiple=1.0, rugged=True)
    import json
    state = c.state()
    state["totals"].pop("rugs_by_stage"); state["totals"].pop("rugs_by_screen")
    c.path.write_text(json.dumps(state))
    again = _census(tmp_path)
    assert again.load()
    audit = again.miss_audit()
    assert audit["rugs_resolved"] == 1 and audit["rugs_attributed"] == 0
