import asyncio
import base64
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from src.execution.phantom import PhantomBridge, PhantomSignerClient, PublicWallet
from src.execution.signer import SignerPolicy, signer_from_env
from src.research.prediction_markets import PaperBroker
from src.runtime.market_gateway import make_app, token_at


class PhantomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.key = Keypair()
        self.address = str(self.key.pubkey())
        self.bridge = PhantomBridge(timeout=.1, policy=SignerPolicy(kill_file=Path(self.temp.name) / "halt"), live_authority=lambda: True)
        challenge = self.bridge.challenge(self.address)
        proof = self.key.sign_message(base64.b64decode(challenge["message_base64"]))
        self.bridge.connect(challenge["nonce"], base64.b64encode(bytes(proof)).decode())
        self.msg = MessageV0.try_compile(self.key.pubkey(), [transfer(TransferParams(from_pubkey=self.key.pubkey(), to_pubkey=Keypair().pubkey(), lamports=1000))], [], Hash.default())
        self.payload = to_bytes_versioned(self.msg)
        self.env = patch.dict(os.environ, {"ALLOW_LIVE_TRADING": "yes-i-understand"})
        self.env.start()

    async def asyncTearDown(self):
        self.env.stop()
        self.temp.cleanup()

    async def test_exact_transaction_signature_round_trip(self):
        task = asyncio.create_task(self.bridge.sign(self.payload, self.address))
        await asyncio.sleep(0)
        request = self.bridge.requests()[0]
        signed = VersionedTransaction(self.msg, [self.key])
        self.bridge.complete(request["id"], base64.b64encode(bytes(signed)).decode())
        self.assertEqual(bytes(signed.signatures[0]), await task)
        self.assertEqual(1, self.bridge.signed)
        self.assertEqual([], self.bridge.requests())

    async def test_mutated_message_is_refused(self):
        task = asyncio.create_task(self.bridge.sign(self.payload, self.address))
        await asyncio.sleep(0)
        other = MessageV0.try_compile(self.key.pubkey(), [], [], Hash.default())
        with self.assertRaisesRegex(ValueError, "different transaction"):
            self.bridge.complete(self.bridge.requests()[0]["id"], base64.b64encode(bytes(VersionedTransaction(other, [self.key]))).decode())
        self.bridge.disconnect()
        with self.assertRaises(PermissionError): await task

    async def test_owner_rejection_propagates(self):
        task = asyncio.create_task(self.bridge.sign(self.payload, self.address))
        await asyncio.sleep(0)
        self.bridge.complete(self.bridge.requests()[0]["id"], reject=True)
        with self.assertRaisesRegex(PermissionError, "rejected"): await task

    async def test_disconnection_cancels_requests(self):
        task = asyncio.create_task(self.bridge.sign(self.payload, self.address))
        await asyncio.sleep(0)
        self.bridge.disconnect()
        with self.assertRaisesRegex(PermissionError, "disconnected"): await task

    async def test_timeout_cleans_up_queue(self):
        with self.assertRaises(asyncio.TimeoutError): await self.bridge.sign(self.payload, self.address)
        self.assertEqual([], self.bridge.requests())

    async def test_paper_authority_cannot_sign_even_with_ack(self):
        self.bridge.live_authority = lambda: False
        with self.assertRaisesRegex(PermissionError, "promotion ledger"): await self.bridge.sign(self.payload, self.address)

    async def test_kill_file_refuses(self):
        self.bridge.policy.kill_file.touch()
        with self.assertRaisesRegex(PermissionError, "HALT_SIGNING"): await self.bridge.sign(self.payload, self.address)

    async def test_ack_absence_refuses(self):
        with patch.dict(os.environ, {"ALLOW_LIVE_TRADING": ""}):
            with self.assertRaisesRegex(PermissionError, "acknowledgement"): await self.bridge.sign(self.payload, self.address)

    async def test_wrong_account_refuses(self):
        with self.assertRaisesRegex(PermissionError, "not connected"): await self.bridge.sign(self.payload, str(Keypair().pubkey()))

    async def test_ownership_challenge_cannot_replay_or_forge(self):
        challenge = self.bridge.challenge(self.address)
        signature = Keypair().sign_message(base64.b64decode(challenge["message_base64"]))
        with self.assertRaises(ValueError): self.bridge.connect(challenge["nonce"], base64.b64encode(bytes(signature)).decode())
        with self.assertRaisesRegex(ValueError, "already used"): self.bridge.connect(challenge["nonce"], base64.b64encode(bytes(signature)).decode())

    async def test_policy_refuses_excess_transfer(self):
        msg = MessageV0.try_compile(self.key.pubkey(), [transfer(TransferParams(from_pubkey=self.key.pubkey(), to_pubkey=Keypair().pubkey(), lamports=3_000_000_000))], [], Hash.default())
        with self.assertRaises(PermissionError): await self.bridge.sign(to_bytes_versioned(msg), self.address)

    async def test_signer_selection_does_not_fall_back_to_private_key(self):
        with patch.dict(os.environ, {"MEMECOIN_WALLET_MODE": "phantom", "MEMECOIN_PHANTOM_TOKEN_FILE": "unused"}):
            signer = signer_from_env(PublicWallet(self.address))
        self.assertIsInstance(signer, PhantomSignerClient)
        self.assertEqual(self.address, signer.public_key)

    async def test_remote_signer_url_is_refused(self):
        with self.assertRaises(ValueError): PhantomSignerClient(self.address, "http://example.com", Path("unused"))


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.token = token_at(Path(self.temp.name) / "gateway.token")
        broker = PaperBroker(Path(self.temp.name) / "paper.sqlite")
        self.client = TestClient(TestServer(make_app(broker, token=self.token, start_runner=False)))
        await self.client.start_server()
        self.headers = {"Host": "127.0.0.1:18082"}

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def test_health_is_paper_and_status_never_implies_edge(self):
        response = await self.client.get("/health", headers=self.headers)
        self.assertTrue((await response.json())["live_submission_locked"])
        response = await self.client.get("/status", headers=self.headers)
        self.assertEqual("UNMEASURED", (await response.json())["prediction_markets"]["production_performance"])

    async def test_mutation_requires_authentication(self):
        response = await self.client.post("/api/phantom/connect", headers=self.headers, json={})
        self.assertEqual(401, response.status)

    async def test_cross_origin_even_with_token_is_rejected(self):
        response = await self.client.post("/api/pair", headers={**self.headers, "Origin": "https://evil.example"}, json={"token": self.token})
        self.assertEqual(403, response.status)

    async def test_dns_rebinding_host_is_rejected(self):
        response = await self.client.get("/status", headers={"Host": "evil.example:18082"})
        self.assertEqual(403, response.status)

    async def test_bearer_paper_signing_stays_locked(self):
        response = await self.client.post("/api/phantom/sign", headers={**self.headers, "Authorization": "Bearer " + self.token},
                                          json={"address": str(Keypair().pubkey()), "message_base64": "AA=="})
        self.assertEqual(403, response.status)

    async def test_ui_is_self_contained_and_csp_disallows_remote_scripts(self):
        response = await self.client.get("/", headers=self.headers)
        self.assertEqual(200, response.status)
        self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
        self.assertNotIn("https://cdn", await response.text())
