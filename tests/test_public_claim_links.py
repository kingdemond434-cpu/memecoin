import tempfile
import unittest
from pathlib import Path
from solders.keypair import Keypair
from src.research.public_claim_links import PublicClaimLinks, public_address


class PublicClaimTests(unittest.TestCase):
    def test_public_addresses_are_accepted_and_secret_lengths_are_not(self):
        self.assertEqual("evm", public_address("0x" + "a" * 40)[0])
        self.assertEqual("solana", public_address(str(Keypair().pubkey()))[0])
        self.assertIsNone(public_address("0x" + "a" * 64))
        self.assertIsNone(public_address(str(Keypair())))

    def test_source_proof_required_and_roles_are_never_invented(self):
        with tempfile.TemporaryDirectory() as temp:
            store = PublicClaimLinks(Path(temp) / "claims.sqlite")
            try:
                kwargs = dict(source_job="canonical:1", source_id="onion:public", addresses=["0x" + "a" * 40],
                              topics=["insider_claim", "launch_claim"], observed_at=100.)
                store.record(**kwargs)
                store.record(**kwargs)
                result = store.page()
                self.assertEqual(1, len(result["items"]))
                self.assertEqual("UNKNOWN", result["items"][0]["entity_role"])
                self.assertEqual("UNVERIFIED_PUBLIC_MENTION", result["items"][0]["claim_status"])
                self.assertFalse(result["items"][0]["trade_authority"])
                self.assertEqual([], store.page(result["next_cursor"])["items"])
                with self.assertRaises(ValueError): store.record(**{**kwargs, "source_job": ""})
                with self.assertRaises(ValueError): store.page(-1)
                self.assertEqual({"evm": 1}, store.report()["by_chain"])
            finally: store.close()

    def test_resume_preserves_cursor_without_exporting_posts_or_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "claims.sqlite"
            store = PublicClaimLinks(path)
            store.record(source_job="proof", source_id="telegram:public", addresses=[str(Keypair().pubkey())],
                         topics=["launch_claim"], observed_at=100.)
            before = store.page()
            store.close()
            store = PublicClaimLinks(path)
            try:
                self.assertEqual(before, store.page())
                self.assertNotIn("text", before["items"][0])
                self.assertNotIn("private_key", before["items"][0])
            finally: store.close()
