"""A bonding curve is the venue, not a whale.

A pump.fun curve holds essentially the whole supply the instant a token is
created -- that is what the curve IS, the AMM every buy trades against.
Counting it as a holder made `top_10_pct` read near 100 on every healthy
launch, which is -25 on the risk score, permanently, for existing. Combined
with the sell-route veto that scored -60, a perfectly ordinary new Pump mint
landed at CRITICAL: 100 - 30 (mint authority, also normal on a live curve)
- 25 - 60 = -15, and `native_risk_level:critical` appeared in 262 of the 678
launches the desk rejected on 2026-09-04.

Excluding the curve alone is the opposite error. With 79% of supply in the
curve, every real holder's share divides by a denominator that is mostly
unbuyable, and genuine concentration reads as trivial. It comes out of BOTH
sides.
"""

from types import SimpleNamespace

import pytest

from src.detection.rug_detector import RugDetector

SUPPLY = 1_000_000_000_000_000
CURVE = "CurveAccount1111111111111111111111111111111"


def _detector(accounts, *, curve_account=CURVE, owners=None):
    """accounts: [(token_account, amount)] as getTokenLargestAccounts gives."""
    detector = RugDetector.__new__(RugDetector)
    detector.curve_account_provider = lambda _mint: curve_account

    class _RPC:
        async def request(self, method, params):
            return {"value": [{"address": address, "amount": str(amount)}
                              for address, amount in accounts]}

    detector.rpc = _RPC()

    async def resolve(addresses):
        return {"status": "OK", "owners": dict(owners or {})}

    detector._solana_token_account_owners = resolve
    return detector


@pytest.mark.asyncio
async def test_a_fresh_launch_is_unmeasured_not_maximally_concentrated():
    """The curve holds everything. Nobody holds anything, so nothing is
    concentrated -- and 0% would be reassurance about a measurement nobody
    made."""
    detector = _detector([(CURVE, SUPPLY)])
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    assert result["concentration_status"] == "DATA_BLOCKED"
    assert result["top_10_pct"] is None
    assert result["curve_held_pct"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_concentration_is_measured_against_circulating_supply():
    """79% in the curve, and one holder owns every circulating token."""
    detector = _detector([(CURVE, int(SUPPLY * 0.79)),
                          ("Whale1", int(SUPPLY * 0.21))])
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    assert result["concentration_status"] == "OK"
    assert result["top_10_pct"] == pytest.approx(100.0, abs=0.01)


@pytest.mark.asyncio
async def test_a_diffuse_book_reads_as_diffuse():
    holders = [(f"H{index}", int(SUPPLY * 0.001)) for index in range(20)]
    detector = _detector([(CURVE, int(SUPPLY * 0.98))] + holders)
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    # Ten of twenty equal holders is half the circulating supply.
    assert result["top_10_pct"] == pytest.approx(50.0, abs=1.0)


@pytest.mark.asyncio
async def test_the_curve_is_matched_by_owner_as_well_as_by_account():
    """Which of the two a launch event names varies by launchpad."""
    detector = _detector([("CurveATA", int(SUPPLY * 0.9)),
                          ("Whale1", int(SUPPLY * 0.1))],
                         owners={"CurveATA": CURVE})
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    assert result["curve_held_pct"] == pytest.approx(90.0)
    assert result["top_10_pct"] == pytest.approx(100.0, abs=0.01)


@pytest.mark.asyncio
async def test_without_a_curve_address_the_old_denominator_stands():
    """No provider means no venue to exclude; nothing silently changes."""
    detector = _detector([("Whale1", int(SUPPLY * 0.6))], curve_account=None)
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    assert result["top_10_pct"] == pytest.approx(60.0, abs=0.01)
    assert result["curve_held_pct"] is None


@pytest.mark.asyncio
async def test_the_venue_is_labelled_on_the_account_row():
    detector = _detector([(CURVE, int(SUPPLY * 0.9)),
                          ("Whale1", int(SUPPLY * 0.1))])
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    labelled = {row["token_account"]: row["is_launch_venue"]
                for row in result["accounts"]}
    assert labelled[CURVE] is True
    assert labelled["Whale1"] is False


@pytest.mark.asyncio
async def test_a_raising_provider_does_not_break_the_report():
    detector = _detector([("Whale1", SUPPLY)])
    detector.curve_account_provider = lambda _m: (_ for _ in ()).throw(
        RuntimeError("boom"))
    result = await detector._solana_holder_concentration("mint", SUPPLY)
    assert result["status"] == "OK"


def test_the_desk_records_the_curve_address_and_wires_the_provider():
    """The provider is useless without something populating it."""
    from pathlib import Path
    main = Path("src/main.py").read_text(encoding="utf-8")
    wiring = Path("src/runtime/wiring.py").read_text(encoding="utf-8")
    assert '["bonding_curve"] = str(' in main
    assert "curve_account_provider" in wiring
    assert '"bonding_curve"' in wiring
