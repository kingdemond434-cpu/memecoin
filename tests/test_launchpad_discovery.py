"""Venues found by watching, because a program id cannot be checked any
other way.

An Anchor discriminator is `sha256("global:<name>")` -- arithmetic, and
therefore verifiable. A program id is an opaque 32-byte constant, and one
wrong character produces a registry entry that silently never fires, which
is indistinguishable from a venue that has no launches. That is exactly the
failure the UNVERIFIED status exists to prevent, reintroduced by the act of
transcribing an id to fix it.

So the desk finds them. `_note_launch_venue` used to see a launch from a
program it did not recognise and return on that line -- being shown new
venues and throwing them away.
"""

from __future__ import annotations

import unittest

from src.chains.launchpad_discovery import (
    MIN_MINTS_TO_PROPOSE, LaunchpadDiscovery)


class ItCountsDistinctMintsNotEvents(unittest.TestCase):

    def test_one_mint_seen_repeatedly_proposes_nothing(self):
        # One transaction redelivered by three racing feeds is one
        # observation, the same rule verification uses.
        discovery = LaunchpadDiscovery()
        for _ in range(MIN_MINTS_TO_PROPOSE * 4):
            discovery.observe("NewProgram", "SameMint")
        self.assertEqual([], discovery.proposals())

    def test_enough_distinct_mints_proposes_once(self):
        discovery = LaunchpadDiscovery()
        proposed = [discovery.observe("NewProgram", f"Mint{index}")
                    for index in range(MIN_MINTS_TO_PROPOSE)]
        self.assertEqual(1, sum(1 for value in proposed if value))
        self.assertTrue(proposed[-1])
        # And it does not re-announce on every launch afterwards.
        self.assertFalse(discovery.observe("NewProgram", "MintExtra"))

    def test_a_launch_with_no_mint_supports_nothing(self):
        # The claim being accumulated is "this program creates tokens", and
        # an observation that cannot name the token supports none of it.
        discovery = LaunchpadDiscovery()
        for _ in range(MIN_MINTS_TO_PROPOSE * 2):
            discovery.observe("NewProgram", "")
        self.assertEqual(0, discovery.observations)


class DeclaredVenuesAreNotCandidates(unittest.TestCase):

    def test_a_known_program_is_ignored(self):
        discovery = LaunchpadDiscovery(known=("pump",))
        for index in range(MIN_MINTS_TO_PROPOSE):
            discovery.observe("pump", f"Mint{index}")
        self.assertEqual([], discovery.proposals())
        self.assertEqual(MIN_MINTS_TO_PROPOSE, discovery.ignored_known)

    def test_adopting_a_candidate_stops_it_being_proposed(self):
        discovery = LaunchpadDiscovery()
        for index in range(MIN_MINTS_TO_PROPOSE):
            discovery.observe("NewProgram", f"Mint{index}")
        self.assertTrue(discovery.proposals())
        discovery.note_known("NewProgram")
        self.assertEqual([], discovery.proposals())
        self.assertNotIn("NewProgram", discovery.venues)


class ItStaysSmallOnAChainOfThousandsOfPrograms(unittest.TestCase):

    def test_the_single_sighting_tail_is_evicted(self):
        discovery = LaunchpadDiscovery(max_tracked=5)
        for index in range(200):
            discovery.observe(f"OneOff{index}", f"Mint{index}")
        self.assertLessEqual(len(discovery.venues), 5)
        self.assertGreater(discovery.evicted, 0)

    def test_a_repeatedly_seen_program_survives_eviction(self):
        # Twenty mints from one program is the shape a launchpad has; one
        # mint from a thousand programs is the shape a chain has.
        discovery = LaunchpadDiscovery(max_tracked=3)
        for index in range(20):
            discovery.observe("RealVenue", f"Mint{index}")
        for index in range(200):
            discovery.observe(f"OneOff{index}", f"Other{index}")
        self.assertIn("RealVenue", discovery.venues)


class AProposalCarriesWhatAnOperatorNeeds(unittest.TestCase):

    def _proposed(self):
        discovery = LaunchpadDiscovery()
        for index in range(MIN_MINTS_TO_PROPOSE):
            discovery.observe("NewProgram", f"Mint{index}",
                              signature=f"Sig{index}", instruction="create_v2",
                              discriminator="aabbccdd")
        return discovery

    def test_it_names_the_id_the_instruction_and_an_example(self):
        proposal = self._proposed().proposals()[0].as_dict()
        self.assertEqual("NewProgram", proposal["program_id"])
        self.assertEqual(MIN_MINTS_TO_PROPOSE, proposal["distinct_mints"])
        self.assertIn("create_v2", proposal["instructions"])
        self.assertIn("aabbccdd", proposal["discriminators"])
        self.assertTrue(proposal["example_signatures"])

    def test_examples_are_a_handful_not_a_log(self):
        proposal = self._proposed().proposals()[0]
        self.assertLessEqual(len(proposal.signatures), 5)

    def test_a_proposal_makes_the_report_ask_for_attention(self):
        # A proposal is a finding: the registry is demonstrably incomplete
        # and somebody should look. Reporting it as OK would bury it.
        self.assertEqual("ATTENTION", self._proposed().report()["status"])
        self.assertEqual("OK", LaunchpadDiscovery().report()["status"])

    def test_programs_short_of_the_bar_are_still_visible(self):
        discovery = LaunchpadDiscovery()
        for index in range(3):
            discovery.observe("Emerging", f"Mint{index}")
        report = discovery.report()
        self.assertEqual("OK", report["status"])
        self.assertEqual("Emerging", report["nearly_there"][0]["program_id"])
        self.assertFalse(report["nearly_there"][0]["proposable"])


class TheDeskFeedsIt(unittest.IsolatedAsyncioTestCase):

    async def test_a_launch_from_an_undeclared_program_is_recorded(self):
        import time as _time

        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        before = desk.launchpad_discovery.observations
        desk._note_launch_venue({
            "program": "SomeVenueNobodyDeclared", "token": "MintA",
            "signature": "Sig", "instruction": "create",
            "timestamp": _time.time()})
        self.assertEqual(before + 1, desk.launchpad_discovery.observations)

    async def test_a_declared_venue_is_not_proposed_as_unknown(self):
        from src.main import MemecoinQuantDesk

        desk = MemecoinQuantDesk("config/chains.yaml", dry_run_override=True,
                                 offline=True)
        await desk.initialize()
        declared = next(iter(desk.launchpads.specs))
        self.assertIn(declared, desk.launchpad_discovery.known)


if __name__ == "__main__":
    unittest.main()
