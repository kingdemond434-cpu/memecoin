"""A registry that starts honest and earns its coverage by running."""

from __future__ import annotations

import hashlib
import unittest

from src.chains.launchpads import (
    OBSERVATIONS_TO_VERIFY, LaunchpadRegistry, LaunchpadSpec,
    anchor_discriminator, default_specs)


def reviewed_test_spec(**kwargs):
    """Local test schema attestation, never represented as mainnet evidence."""
    return LaunchpadSpec(
        protocol_version="local-test-v1", source_ref="fixture:source",
        idl_ref="fixture:idl", verification_provenance="fixture:review",
        fixture_refs=("fixture:decoder-parity",), layout_verified=True,
        decoder_parity=True, **kwargs)


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
        self.spec = reviewed_test_spec(
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
        spec = reviewed_test_spec(name="p", program_id="P",
                             create_instructions=("create",),
                             mint_account_index=None)
        registry = LaunchpadRegistry([spec])
        data = anchor_discriminator("create") + b"\x00" * 16
        self.assertIsNone(registry.decode("P", data, ["A"], [0], "sig", 1))

    def test_the_fee_payer_stands_in_for_an_unmapped_creator(self):
        spec = reviewed_test_spec(name="p", program_id="P",
                             create_instructions=("create",),
                             mint_account_index=0, creator_account_index=None,
                             fee_payer_is_creator=True)
        registry = LaunchpadRegistry([spec])
        data = anchor_discriminator("create") + b"\x00" * 16
        event = registry.decode("P", data, ["MINT"], [0], "sig", 1,
                                fee_payer="PAYER")
        self.assertEqual("PAYER", event.creator)


class VerificationIsEarnedByObservation(unittest.TestCase):

    def setUp(self):
        self.registry = LaunchpadRegistry([reviewed_test_spec(
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


class UnknownLayoutsCannotEarnTrust(unittest.TestCase):

    def test_unconfirmed_account_indexes_are_not_decoded(self):
        spec = LaunchpadSpec("hypothesis", "P", ("create",), mint_account_index=0)
        registry = LaunchpadRegistry([spec])
        self.assertIsNone(registry.decode("P", anchor_discriminator("create"), ["not-known-to-be-mint"], [0], "s", 1))
        self.assertEqual(1.0, registry.report()["unknown_event_rate"])

    def test_three_distinct_events_without_source_idl_parity_do_not_verify(self):
        from src.chains.launchpads import CanonicalLaunchEvent
        spec = LaunchpadSpec("hypothesis", "P", ("create",), mint_account_index=0)
        registry = LaunchpadRegistry([spec])
        for index in range(OBSERVATIONS_TO_VERIFY + 5):
            event = CanonicalLaunchEvent("hypothesis", "P", "mint", None, f"sig-{index}", index, 100,
                                         instruction="create", provisional=True)
            self.assertFalse(registry.observe(event))
        self.assertFalse(spec.trusted)

    def test_a_status_flag_without_source_contract_cannot_claim_verified_coverage(self):
        spec = LaunchpadSpec("hypothesis", "P", ("create",), mint_account_index=0, status="VERIFIED")
        registry = LaunchpadRegistry([spec])
        self.assertFalse(spec.trusted)
        self.assertEqual(0, registry.report()["verified"])
        self.assertEqual("DATA_BLOCKED_PROTOCOL_CONTRACT", registry.report()["venues"]["hypothesis"]["status"])

    def test_default_pump_positions_come_from_the_official_vendored_idl(self):
        from src.chains.idl import load_idl
        spec = next(item for item in default_specs() if item.name == "pump.fun")
        idl = load_idl("pump")
        create = next(item for item in idl["instructions"] if item["name"] == "create")
        self.assertEqual("mint", create["accounts"][spec.mint_account_index]["name"])
        self.assertEqual(idl["address"], spec.program_id)
        self.assertTrue(spec.verification_ready)
        self.assertIsNone(spec.creator_account_index)

    def test_fee_payer_is_not_assumed_to_be_creator(self):
        spec = reviewed_test_spec(name="p", program_id="P", create_instructions=("create",), mint_account_index=0)
        event = LaunchpadRegistry([spec]).decode("P", anchor_discriminator("create"), ["mint"], [0], "s", 1, fee_payer="payer")
        self.assertIsNone(event.creator)


if __name__ == "__main__":
    unittest.main()
