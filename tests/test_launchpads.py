"""A registry that starts honest and earns its coverage by running."""

from __future__ import annotations

import hashlib
import unittest

from src.chains.launchpads import (
    OBSERVATIONS_TO_VERIFY, LaunchpadRegistry, LaunchpadSpec,
    anchor_discriminator, default_specs)


class DiscriminatorsAreDerivedNotTranscribed(unittest.TestCase):

    def test_it_is_anchor_s_own_rule(self):
        for name in ("create", "initialize", "buy", "sell"):
            expected = hashlib.sha256(f"global:{name}".encode()).digest()[:8]
            self.assertEqual(expected, anchor_discriminator(name))

    def test_it_is_eight_bytes(self):
        self.assertEqual(8, len(anchor_discriminator("create")))

    def test_different_instructions_do_not_collide(self):
        names = ("create", "initialize", "buy", "sell", "swap", "launch")
        self.assertEqual(len(names), len({anchor_discriminator(n) for n in names}))


class ProgramIdsAreHypothesesUntilObserved(unittest.TestCase):

    def test_everything_but_the_natively_decoded_venue_starts_unverified(self):
        specs = {spec.name: spec for spec in default_specs()}
        self.assertEqual("VERIFIED", specs["pump.fun"].status)
        for name, spec in specs.items():
            if name == "pump.fun":
                continue
            self.assertEqual("UNVERIFIED", spec.status,
                             f"{name} claims coverage it has not demonstrated")

    def test_coverage_counts_only_what_was_observed(self):
        registry = LaunchpadRegistry()
        report = registry.report()
        self.assertEqual(1, report["verified"])
        self.assertGreater(report["declared"], report["verified"])

    def test_unverified_programs_are_still_subscribed_to(self):
        # They can never be verified otherwise.
        registry = LaunchpadRegistry()
        self.assertGreater(len(registry.watched_programs),
                           len(registry.verified_programs))


class DecodingProducesOneShapeForEveryVenue(unittest.TestCase):

    def setUp(self):
        self.spec = LaunchpadSpec(
            name="testpad", program_id="PROG", create_instructions=("create",),
            mint_account_index=0, creator_account_index=1)
        self.registry = LaunchpadRegistry([self.spec])

    def _decode(self, signature="sig0", instruction="create"):
        data = anchor_discriminator(instruction) + b"\x00" * 16
        return self.registry.decode(
            "PROG", data, keys=["MINT", "CREATOR", "OTHER"],
            accounts=[0, 1, 2], signature=signature, slot=42, observed_at=100.0)

    def test_a_matching_instruction_becomes_a_canonical_launch(self):
        event = self._decode()
        self.assertIsNotNone(event)
        self.assertEqual("MINT", event.mint)
        self.assertEqual("CREATOR", event.creator)
        self.assertEqual("testpad", event.venue)

    def test_an_unverified_venue_yields_a_PROVISIONAL_event(self):
        # It is evidence about the registry, not a trading candidate.
        self.assertTrue(self._decode().provisional)

    def test_an_unknown_instruction_decodes_to_nothing(self):
        data = anchor_discriminator("something_else") + b"\x00" * 16
        self.assertIsNone(self.registry.decode(
            "PROG", data, ["MINT"], [0], "sig", 1))

    def test_an_unknown_program_decodes_to_nothing(self):
        data = anchor_discriminator("create") + b"\x00" * 16
        self.assertIsNone(self.registry.decode(
            "OTHER_PROG", data, ["MINT"], [0], "sig", 1))

    def test_a_launch_with_no_identifiable_mint_is_refused(self):
        # Emitting it would put a blank key into the census.
        spec = LaunchpadSpec(name="p", program_id="P",
                             create_instructions=("create",),
                             mint_account_index=None)
        registry = LaunchpadRegistry([spec])
        data = anchor_discriminator("create") + b"\x00" * 16
        self.assertIsNone(registry.decode("P", data, ["A"], [0], "sig", 1))

    def test_the_fee_payer_stands_in_for_an_unmapped_creator(self):
        spec = LaunchpadSpec(name="p", program_id="P",
                             create_instructions=("create",),
                             mint_account_index=0, creator_account_index=None)
        registry = LaunchpadRegistry([spec])
        data = anchor_discriminator("create") + b"\x00" * 16
        event = registry.decode("P", data, ["MINT"], [0], "sig", 1,
                                fee_payer="PAYER")
        self.assertEqual("PAYER", event.creator)


class VerificationIsEarnedByObservation(unittest.TestCase):

    def setUp(self):
        self.registry = LaunchpadRegistry([LaunchpadSpec(
            name="testpad", program_id="PROG", create_instructions=("create",),
            mint_account_index=0)])

    def _observe(self, index):
        data = anchor_discriminator("create") + b"\x00" * 16
        event = self.registry.decode("PROG", data, ["MINT"], [0],
                                     f"sig{index}", index)
        return self.registry.observe(event)

    def test_one_clean_decode_is_not_enough(self):
        self.assertFalse(self._observe(0))
        self.assertEqual("UNVERIFIED", self.registry.specs["PROG"].status)

    def test_enough_decodes_promote_the_program(self):
        promoted = [self._observe(i) for i in range(OBSERVATIONS_TO_VERIFY)]
        self.assertTrue(promoted[-1])
        self.assertEqual("VERIFIED", self.registry.specs["PROG"].status)
        self.assertIn("PROG", self.registry.verified_programs)

    def test_a_verified_venue_stops_marking_events_provisional(self):
        for i in range(OBSERVATIONS_TO_VERIFY):
            self._observe(i)
        data = anchor_discriminator("create") + b"\x00" * 16
        event = self.registry.decode("PROG", data, ["MINT"], [0], "s", 1)
        self.assertFalse(event.provisional)

    def test_promotion_is_announced_once_not_every_time(self):
        for i in range(OBSERVATIONS_TO_VERIFY):
            self._observe(i)
        self.assertFalse(self._observe(99), "already verified; no second promotion")

    def test_the_first_signature_is_kept_for_audit(self):
        self._observe(0)
        self.assertEqual("sig0", self.registry.specs["PROG"].first_seen_signature)


if __name__ == "__main__":
    unittest.main()
