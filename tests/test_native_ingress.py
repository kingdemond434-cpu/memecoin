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
    MODE_AUTO, MODE_OFF, MODE_SHADOW, PARITY_WINDOW_S, IngressEvent,
    NativeIngress)


def _row(sig: bytes, slot: int = 1):
    return (sig, b"P" * 32, b"D" * 8, b"F" * 32, b"data",
            [b"A" * 32, b"B" * 32], slot, 1_700_000_000_000_000_000, False)


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

    def _shadowing(self):
        ingress = NativeIngress("http://x", programs=("P",), promote_after=3)
        ingress._native = _Fake()
        ingress.available = True
        ingress.mode = MODE_SHADOW
        return ingress

    def _age_everything(self, ingress):
        old = time.time() - PARITY_WINDOW_S - 1
        for store in (ingress._native_seen, ingress._python_seen):
            for key in list(store):
                store[key] = old

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
