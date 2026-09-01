"""The native receiver has to prove it misses nothing before it decides.

Same bargain the Rust transaction builder made: byte parity before trust. A
receiver that is faster on the events it catches and blind to one launch in
a thousand is not an improvement, and the only way to tell the two apart is
to run both and compare.
"""

from __future__ import annotations

import time
import unittest

from src.chains.native_ingress import (
    MODE_AUTO, MODE_OFF, MODE_RUST, MODE_SHADOW, PARITY_WARMUP_S,
    PARITY_WINDOW_S, IngressEvent, NativeIngress, canonical_signature_key,
    normalise_mode)


def _row(sig: bytes, slot: int = 1, at: float = None):
    # The wire timestamp defaults to NOW because the parity window is real
    # time: a fixture stamped in 2023 expires the moment it is drained.
    when = time.time() if at is None else at
    return (sig, b"P" * 32, b"D" * 8, b"F" * 32, b"data",
            [b"A" * 32, b"B" * 32], slot, int(when * 1e9), False)


class TheEventShapeIsPrimitivesNotObjects(unittest.TestCase):

    def test_a_row_decodes_into_named_fields(self):
        event = IngressEvent.from_tuple(_row(b"s" * 64, slot=77))
        self.assertEqual(77, event.slot)
        self.assertEqual(2, len(event.accounts))
        self.assertEqual(b"s" * 8, event.signature_key)

    def test_keys_stay_binary(self):
        # base58 is a division loop and an allocation per key, and a
        # transaction carries dozens. Only what the desk acts on is encoded.
        event = IngressEvent.from_tuple(_row(b"s" * 64))
        self.assertIsInstance(event.program, bytes)
        self.assertIsInstance(event.accounts[0], bytes)


class ItRefusesToStartRatherThanPretend(unittest.TestCase):

    def test_disabled_says_so(self):
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_OFF)
        self.assertFalse(ingress.start())
        self.assertIn("disabled", ingress.unavailable_reason)

    def test_no_endpoint_says_so(self):
        ingress = NativeIngress("", programs=("P",))
        self.assertFalse(ingress.start())
        self.assertIn("no endpoint", ingress.unavailable_reason)

    def test_no_programs_says_so(self):
        ingress = NativeIngress("http://x", programs=())
        self.assertFalse(ingress.start())
        self.assertIn("no programs", ingress.unavailable_reason)

    def test_a_report_from_a_dead_ingress_is_honest(self):
        report = NativeIngress("", programs=()).report()
        self.assertEqual("OFF", report["status"])
        self.assertFalse(report["available"])
        self.assertIsNone(report["agreement_rate"])


class _Fake:
    """Stands in for the Rust object so parity logic is testable offline."""

    def __init__(self):
        self.rows = []
        self.stopped = False

    def drain(self, _max):
        rows, self.rows = self.rows, []
        return rows

    def stop(self):
        self.stopped = True

    def report(self):
        return {"updates": 10, "matched": len(self.rows)}


class ParityDecidesWhetherItIsTrusted(unittest.TestCase):

    def _shadowing(self, mode=MODE_AUTO):
        ingress = NativeIngress("http://x", programs=("P",), promote_after=3,
                                mode=mode)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        return ingress

    def _age_everything(self, ingress):
        old = time.time() - PARITY_WINDOW_S - 1
        for key in list(ingress._native_seen):
            ingress._native_seen[key] = (old, old)
        for key in list(ingress._python_seen):
            ingress._python_seen[key] = old

    def test_agreement_promotes_it(self):
        ingress = self._shadowing()
        for index in range(3):
            signature = bytes([index]) * 64
            ingress._native.rows.append(_row(signature))
            ingress.note_python_event(signature)
            ingress.drain()
        self._age_everything(ingress)
        ingress.drain()
        self.assertEqual(3, ingress.agreements)
        self.assertEqual(MODE_AUTO, ingress.mode)

    def test_one_miss_demotes_it_permanently(self):
        ingress = self._shadowing()
        # Enough agreement to be promoted...
        for index in range(3):
            signature = bytes([index]) * 64
            ingress._native.rows.append(_row(signature))
            ingress.note_python_event(signature)
            ingress.drain()
        self._age_everything(ingress)
        ingress.drain()
        self.assertEqual(MODE_AUTO, ingress.mode)

        # ...then one event the reference saw and it did not.
        ingress.note_python_event(b"\xff" * 64)
        self._age_everything(ingress)
        ingress.drain()
        self.assertEqual(MODE_SHADOW, ingress.mode)
        self.assertIn("missed", ingress.demoted_reason)

        # And no amount of subsequent agreement wins it back.
        for index in range(50):
            signature = bytes([index]) * 63 + b"\x01"
            ingress._native.rows.append(_row(signature))
            ingress.note_python_event(signature)
            ingress.drain()
        self._age_everything(ingress)
        ingress.drain()
        self.assertEqual(MODE_SHADOW, ingress.mode)

    def test_seeing_an_event_first_is_not_a_fault(self):
        # The two subscribe with different filters at different instants, and
        # being EARLY looks exactly like "only the native side saw it".
        ingress = self._shadowing()
        ingress._native.rows.append(_row(b"z" * 64))
        ingress.drain()
        self._age_everything(ingress)
        ingress.drain()
        self.assertEqual(1, ingress.native_only)
        self.assertEqual("", ingress.demoted_reason)
        self.assertEqual(MODE_SHADOW, ingress.mode)

    def test_a_comparison_is_not_made_before_both_have_had_a_chance(self):
        # Compared immediately, every event the native side has not yet
        # delivered reads as a miss -- which would demote it on arrival-time
        # differences alone.
        ingress = self._shadowing()
        ingress.note_python_event(b"q" * 64)
        ingress.drain()
        self.assertEqual(0, ingress.python_only)
        self.assertEqual("", ingress.demoted_reason)

    def test_stopping_releases_the_native_object(self):
        ingress = self._shadowing()
        native = ingress._native
        ingress.stop()
        self.assertTrue(native.stopped)
        self.assertEqual(MODE_OFF, ingress.mode)


def _b58(raw: bytes) -> str:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    number = int.from_bytes(raw, "big")
    out = ""
    while number:
        number, remainder = divmod(number, 58)
        out = alphabet[remainder] + out
    return alphabet[0] * (len(raw) - len(raw.lstrip(b"\0"))) + out


class BothSidesMeanTheSameThingByASignature(unittest.TestCase):
    """The bug this closes made the parity ledger incapable of agreeing.

    Rust dedupes on the first eight bytes of the raw 64-byte signature. The
    Python Yellowstone decoder converts the same signature to base58 text
    before it reaches the desk. `b"5Kj7..."[:8]` and `raw[:8]` are different
    key spaces, so every event the reference client saw read as a native
    miss, the permanent-demotion latch fired inside the first window, and the
    fast path was disqualified before it had a chance to be wrong.
    """

    def test_base58_text_and_raw_bytes_reduce_to_one_key(self):
        raw = bytes(range(64))
        self.assertEqual(raw[:8], canonical_signature_key(raw))
        self.assertEqual(raw[:8], canonical_signature_key(_b58(raw)))
        self.assertEqual(raw[:8], canonical_signature_key(_b58(raw).encode()))

    def test_leading_zero_bytes_survive_the_round_trip(self):
        # base58 encodes leading zero bytes as literal '1's; dropping them
        # shifts every subsequent byte and silently changes the key.
        raw = b"\x00\x00" + bytes(range(62))
        self.assertEqual(raw[:8], canonical_signature_key(_b58(raw)))

    def test_undecodable_input_yields_no_key_rather_than_a_wrong_one(self):
        # A zero-length key would match every other failure and manufacture
        # agreements out of errors, which is worse than counting the loss.
        for bad in ("", None, "0OIl-not-base58", b"", b"\xff\xfe"):
            self.assertEqual(b"", canonical_signature_key(bad))

    def test_the_ledger_agrees_across_representations(self):
        ingress = NativeIngress("http://x", programs=("P",), promote_after=3,
                                mode=MODE_AUTO)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        raw = bytes(range(64))
        ingress._native.rows.append(_row(raw))
        ingress.drain()
        # The reference client hands over base58 text, as its decoder emits.
        ingress.note_python_event(_b58(raw))
        self.assertEqual(1, ingress.agreements)
        self.assertEqual(0, ingress.python_only)

    def test_a_signature_it_cannot_resolve_is_counted_not_swallowed(self):
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_AUTO)
        ingress._native = _Fake()
        ingress.note_python_event("!!!not-base58!!!")
        self.assertEqual(1, ingress.unresolvable_signatures)
        self.assertEqual(0, ingress.agreements)


class ModeMeansExactlyWhatItSays(unittest.TestCase):

    def _running(self, mode):
        ingress = NativeIngress("http://x", programs=("P",), promote_after=2,
                                mode=mode)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        return ingress

    def _agree(self, ingress, count):
        for index in range(count):
            signature = bytes([index]) * 64
            ingress._native.rows.append(_row(signature))
            ingress.drain()
            ingress.note_python_event(signature)

    def test_configured_shadow_never_promotes_itself(self):
        # The previous code set mode=SHADOW at start whatever was asked for
        # and then promoted out of it on agreement, so "SHADOW" in config did
        # not mean "stays a shadow" -- the setting was a lie in the one
        # direction that matters.
        ingress = self._running(MODE_SHADOW)
        self._agree(ingress, 20)
        self.assertEqual(20, ingress.agreements)
        self.assertEqual(MODE_SHADOW, ingress.mode)
        self.assertFalse(ingress.authoritative)
        self.assertFalse(ingress.report()["promotable"])

    def test_auto_promotes_on_sustained_agreement(self):
        ingress = self._running(MODE_AUTO)
        self._agree(ingress, 2)
        self.assertEqual(MODE_AUTO, ingress.mode)
        self.assertTrue(ingress.authoritative)

    def test_rust_is_authoritative_from_the_first_event(self):
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_RUST)
        ingress._native = object()  # start() is idempotent when already set
        self.assertTrue(ingress.start())
        self.assertEqual(MODE_RUST, ingress.requested_mode)

    def test_an_unrecognised_mode_is_off_not_a_silent_shadow(self):
        self.assertEqual(MODE_OFF, normalise_mode("fast"))
        self.assertEqual(MODE_OFF, normalise_mode(""))
        self.assertEqual(MODE_AUTO, normalise_mode("auto"))


class ItMeasuresHowMuchEarlierItSaw(unittest.TestCase):
    """Agreement proves it is CORRECT. Lead time is the only evidence that it
    is FASTER, and without a number there is nothing to promote on."""

    def test_lead_time_is_recorded_from_the_wire_timestamp(self):
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_AUTO,
                                promote_after=10_000)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        raw = bytes(range(64))
        native_at = time.time() - 0.020
        row = list(_row(raw))
        row[7] = int(native_at * 1e9)
        ingress._native.rows.append(tuple(row))
        ingress.drain()
        ingress.note_python_event(raw)
        lead = ingress.median_lead_ms
        self.assertIsNotNone(lead)
        # Roughly 20ms earlier, allowing for the test's own scheduling.
        self.assertGreater(lead, 10.0)
        self.assertLess(lead, 200.0)
        self.assertEqual(1, ingress.report()["lead_samples"])

    def test_a_native_path_that_is_behind_reads_as_negative_not_zero(self):
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_AUTO,
                                promote_after=10_000)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        raw = bytes(range(64))
        ingress.note_python_event(raw)
        row = list(_row(raw))
        row[7] = int((time.time() + 0.030) * 1e9)
        ingress._native.rows.append(tuple(row))
        ingress.drain()
        self.assertLess(ingress.median_lead_ms, 0.0)


class TheStartupSkewIsNotHeldAgainstIt(unittest.TestCase):

    def test_misses_inside_the_warmup_do_not_demote(self):
        # The reference client is already streaming when this subscribes, and
        # a gRPC subscription takes a moment to be served. Punishing that gap
        # with a permanent latch makes promotion unreachable by construction
        # -- indistinguishable from never having wired it at all.
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_AUTO)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        ingress.started_at = time.time()
        ingress.note_python_event(bytes(range(64)))
        for key in list(ingress._python_seen):
            ingress._python_seen[key] = time.time() - PARITY_WINDOW_S - 1
        ingress.drain()
        self.assertEqual(1, ingress.python_only)
        self.assertEqual("", ingress.demoted_reason)

    def test_a_miss_after_the_warmup_still_demotes(self):
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_AUTO)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        ingress.started_at = time.time() - PARITY_WARMUP_S - PARITY_WINDOW_S - 10
        ingress.note_python_event(bytes(range(64)))
        for key in list(ingress._python_seen):
            ingress._python_seen[key] = time.time() - PARITY_WINDOW_S - 1
        ingress.drain()
        self.assertIn("missed", ingress.demoted_reason)


class TheExtensionActuallyCarriesIt(unittest.TestCase):

    def test_the_built_extension_exposes_the_receiver(self):
        try:
            import solana_fastpath
        except ImportError:
            self.skipTest("extension not built")
        if not hasattr(solana_fastpath, "NativeIngress"):
            self.skipTest("extension built without the ingress feature")
        native = solana_fastpath.NativeIngress("http://127.0.0.1:1")
        report = native.report()
        self.assertFalse(report["running"])
        self.assertEqual(0, report["matched"])

    def test_a_malformed_program_id_is_refused_not_ignored(self):
        try:
            import solana_fastpath
        except ImportError:
            self.skipTest("extension not built")
        if not hasattr(solana_fastpath, "NativeIngress"):
            self.skipTest("extension built without the ingress feature")
        native = solana_fastpath.NativeIngress("http://127.0.0.1:1")
        with self.assertRaises(ValueError):
            native.start(["not-base58!!!"], None, 1)


if __name__ == "__main__":
    unittest.main()


class TheFeeConfigIsActuallyRead(unittest.IsolatedAsyncioTestCase):
    """`adopt_chain_config` had zero callers in the entire tree.

    Pump publishes the dynamic fee tiers only as an image, so the fee engine
    refused to transcribe them and answered DATA_BLOCKED for every quote
    after the schedule went dynamic. The decoder for the on-chain account
    those numbers describe was written and tested -- and never called. The
    consequence is not a missing field: costing is blocked, so no expected
    value can be computed net of cost, so nothing clears an entry bar. A
    refusal nobody lifts is indistinguishable from a decision never to trade.
    """

    def _desk(self, response):
        import base64
        import struct
        from src.chains.pump_fee_config import FEE_CONFIG_DISCRIMINATOR
        from src.execution.pump_fees import PumpFeeSchedule
        from src.runtime.maintenance import DeskMaintenance

        class _Rpc:
            def __init__(self):
                self.calls = []

            async def request(self, method, params):
                self.calls.append((method, params))
                return response

        class _Desk(DeskMaintenance):
            offline = False

            def __init__(self):
                self.solana_rpc = _Rpc()
                self.fee_schedule = PumpFeeSchedule.load()
                self._fee_config_refreshed_at = 0.0

        return _Desk()

    @staticmethod
    def _account_bytes():
        import struct
        from src.chains.pump_fee_config import FEE_CONFIG_DISCRIMINATOR

        def fees(lp, protocol, creator):
            return struct.pack("<QQQ", lp, protocol, creator)

        data = FEE_CONFIG_DISCRIMINATOR + b"\x01" + bytes(32) + fees(1, 2, 3)
        tiers = ((0, 5, 50, 45), (10 ** 9, 5, 30, 25))
        data += len(tiers).to_bytes(4, "little")
        for threshold, lp, protocol, creator in tiers:
            data += threshold.to_bytes(16, "little") + fees(lp, protocol, creator)
        data += (0).to_bytes(4, "little")
        return data

    async def test_the_tiers_come_off_the_account_and_unblock_costing(self):
        import base64
        encoded = base64.b64encode(self._account_bytes()).decode()
        desk = self._desk({"value": {"data": [encoded, "base64"]}})
        self.assertTrue(await desk._refresh_pump_fee_config())
        self.assertEqual(2, len(desk.fee_schedule.chain_tiers))
        # And the quote it was refusing now answers.
        quote = desk.fee_schedule.quote(
            market_cap_lamports=10 ** 10, at_utc=time.time())
        self.assertEqual("OK", quote.status)
        self.assertGreater(desk._fee_config_refreshed_at, 0.0)

    async def test_a_bad_read_keeps_the_tiers_it_already_had(self):
        # Losing a good table to one bad RPC response would re-block costing
        # for no reason, and a desk that intermittently cannot price a trade
        # is worse than one that consistently can.
        import base64
        desk = self._desk({"value": {"data": [
            base64.b64encode(self._account_bytes()).decode(), "base64"]}})
        self.assertTrue(await desk._refresh_pump_fee_config())

        async def broken(method, params):
            raise RuntimeError("429")

        desk.solana_rpc.request = broken
        self.assertFalse(await desk._refresh_pump_fee_config())
        self.assertEqual(2, len(desk.fee_schedule.chain_tiers))

    async def test_an_empty_account_is_reported_not_silently_accepted(self):
        desk = self._desk({"value": None})
        self.assertFalse(await desk._refresh_pump_fee_config())
        self.assertEqual((), desk.fee_schedule.chain_tiers)


class TheTwoDecodersMustAgreeOnTheSameBytes(unittest.TestCase):
    """A semantic decoder that is subtly wrong is worse than a raw tuple.

    A raw tuple is opaque and obviously needs decoding. A wrong reserve
    looks like a number, flows into `_latest_curve_state`, prices a position
    and sizes a trade. So the Rust decoder is held against the Python one on
    the same bytes, and the discriminators are checked rather than assumed
    to have been copied correctly.
    """

    def _payloads(self):
        import struct

        from src.chains.yellowstone_grpc import PumpFunMonitor

        create = bytearray(PumpFunMonitor.CREATE_EVENT)
        for text in (b"Name", b"SYM", b"https://example.invalid/x"):
            create += len(text).to_bytes(4, "little") + text
        create += bytes([1]) * 32 + bytes([2]) * 32 + bytes([3]) * 32 + bytes([4]) * 32
        create += struct.pack("<q", 1_700_000_000)

        trade = bytearray(PumpFunMonitor.TRADE_EVENT)
        trade += bytes([9]) * 32
        trade += struct.pack("<QQ", 5_000_000_000, 123_456)
        trade += bytes([1])
        trade += bytes([7]) * 32
        trade += struct.pack("<q", 1_700_000_001)
        trade += struct.pack("<QQ", 30_000_000_000, 1_073_000_000_000_000)

        migrate = bytearray(PumpFunMonitor.COMPLETE_EVENT)
        migrate += bytes([7]) * 32 + bytes([9]) * 32 + bytes([2]) * 32
        migrate += struct.pack("<q", 1_700_000_002)
        return bytes(create), bytes(trade), bytes(migrate)

    def _native(self):
        try:
            import solana_fastpath
        except ImportError:
            self.skipTest("extension not built")
        decode = getattr(solana_fastpath, "decode_pump_event", None)
        if decode is None:
            self.skipTest("extension built without the semantic decoder")
        return decode

    def test_the_two_decoders_agree_field_for_field_on_a_create(self):
        # The comparison, not a re-implementation of the layout in the test:
        # a fixture that encodes what it expects proves only that the test
        # agrees with itself.
        decode = self._native()
        from src.chains.yellowstone_grpc import PumpFunMonitor

        create, _, _ = self._payloads()
        native = decode(create)
        monitor = PumpFunMonitor.__new__(PumpFunMonitor)
        python = PumpFunMonitor._decode_program_event(monitor, create, "sig", 1)
        for field in ("type", "token", "bonding_curve", "wallet", "creator",
                      "name", "symbol", "uri"):
            self.assertEqual(python[field], native[field], field)

    def test_the_two_decoders_agree_field_for_field_on_a_trade(self):
        decode = self._native()
        from src.chains.yellowstone_grpc import PumpFunMonitor

        _, trade, _ = self._payloads()
        native = decode(trade)
        monitor = PumpFunMonitor.__new__(PumpFunMonitor)
        python = PumpFunMonitor._decode_program_event(monitor, trade, "sig", 1)
        self.assertEqual(python["token"], native["token"])
        self.assertEqual(python["side"], native["side"])
        self.assertEqual(python["virtual_sol_reserves"],
                         native["virtual_sol_reserves"])
        self.assertEqual(python["virtual_token_reserves"],
                         native["virtual_token_reserves"])
        # And the SOL amount, which the Python side reports already divided.
        self.assertAlmostEqual(python["notional_sol"],
                               native["sol_amount"] / 1e9)

    def test_a_create_decodes_to_the_same_mint_and_creator(self):
        decode = self._native()
        from src.chains.yellowstone_grpc import b58encode

        create, _, _ = self._payloads()
        native = decode(create)
        self.assertEqual("token_created", native["type"])
        self.assertEqual(b58encode(bytes([1]) * 32), native["token"])
        self.assertEqual(b58encode(bytes([4]) * 32), native["creator"])
        self.assertEqual("SYM", native["symbol"])

    def test_a_trade_decodes_to_the_same_reserves(self):
        decode = self._native()
        _, trade, _ = self._payloads()
        native = decode(trade)
        self.assertEqual("token_trade", native["type"])
        self.assertEqual("buy", native["side"])
        self.assertEqual(5_000_000_000, native["sol_amount"])
        self.assertEqual(30_000_000_000, native["virtual_sol_reserves"])
        self.assertEqual(1_073_000_000_000_000, native["virtual_token_reserves"])

    def test_a_migration_decodes(self):
        decode = self._native()
        _, _, migrate = self._payloads()
        native = decode(migrate)
        self.assertEqual("token_migrated", native["type"])

    def test_an_unrecognised_payload_decodes_to_nothing(self):
        # The stream carries plenty of events that are not ours, and
        # skipping them is normal rather than an error.
        decode = self._native()
        self.assertIsNone(decode(b"\x00" * 64))


class ADecodedEventSaysWhatItIs(unittest.TestCase):

    def test_an_event_with_no_decode_is_unknown_not_a_guess(self):
        event = IngressEvent.from_tuple(_row(b"s" * 64))
        self.assertEqual("unknown", event.kind)
        self.assertIsNone(event.decoded)

    def test_a_decoded_event_carries_its_kind(self):
        row = list(_row(b"s" * 64)) + [{"type": "token_created", "token": b"m" * 32}]
        event = IngressEvent.from_tuple(tuple(row))
        self.assertEqual("token_created", event.kind)

    def test_the_drain_counts_what_the_native_side_decoded(self):
        # The ratio is what says whether the semantic decoder is carrying
        # its weight or whether Python is still doing the work that matters.
        ingress = NativeIngress("http://x", programs=("P",), mode=MODE_AUTO)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        ingress._native.rows.append(
            tuple(list(_row(b"a" * 64)) + [{"type": "token_created"}]))
        ingress._native.rows.append(_row(b"b" * 64))
        ingress.drain()
        report = ingress.report()
        self.assertEqual(1, report["decoded_natively"])
        self.assertEqual(0.5, report["decoded_share"])
        self.assertEqual({"token_created": 1}, report["by_kind"])
