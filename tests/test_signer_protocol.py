"""One held connection and binary frames, with the key still isolated.

Every signature used to open a new Unix socket, JSON-encode a dict and
base64 the message -- a connect, a round trip through the kernel's accept
path and two encodings, for a request that means "sign these 300 bytes".

What must NOT change: the key stays in its own process, every message is
authorised independently, and a refusal is still a refusal. Held connection
buys transport cost, not standing permission. These assert both halves.
"""

from __future__ import annotations

import asyncio
import struct
import unittest

from src.execution.signer_protocol import (
    HEADER_SIZE, MAX_FRAME, OP_PING, OP_PUBKEY, OP_SIGN, STATUS_OK,
    STATUS_REFUSED, decode_header, encode_request, encode_response)


class TheFrameIsLengthPrefixedNotLineDelimited(unittest.TestCase):

    def test_a_payload_containing_a_newline_survives(self):
        # A signature contains 0x0a about one byte in thirty. A line
        # protocol carrying binary is a bug waiting for the right signature.
        payload = bytes(range(256))
        self.assertIn(b"\n", payload)
        frame = encode_response(STATUS_OK, payload)
        length, status = decode_header(frame[:HEADER_SIZE])
        self.assertEqual(STATUS_OK, status)
        self.assertEqual(len(payload), length)
        self.assertEqual(payload, frame[HEADER_SIZE:])

    def test_an_empty_payload_round_trips(self):
        length, op = decode_header(encode_request(OP_PING)[:HEADER_SIZE])
        self.assertEqual(0, length)
        self.assertEqual(OP_PING, op)

    def test_an_oversized_frame_is_refused_at_encode(self):
        with self.assertRaises(ValueError):
            encode_request(OP_SIGN, b"\x00" * (MAX_FRAME + 1))

    def test_an_absurd_declared_length_is_refused_at_decode(self):
        # An unbounded length prefix is how a framing bug becomes memory
        # exhaustion.
        with self.assertRaises(ValueError):
            decode_header(struct.pack("<IB", 2 ** 31, OP_SIGN))

    def test_a_zero_length_frame_is_refused(self):
        with self.assertRaises(ValueError):
            decode_header(struct.pack("<IB", 0, OP_SIGN))


def _serve(service, path):
    from src.execution.signer import SignerServer

    return SignerServer(service, path)


class BothProtocolsWorkAgainstARealSocket(unittest.TestCase):
    """A signer and a desk are deployed separately; both must be servable."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        try:
            from solders.keypair import Keypair
        except ImportError:  # pragma: no cover
            self.skipTest("solders not installed")
        from src.execution.signer import SignerPolicy, SignerService

        # The signer refuses to sign for a desk that has not acknowledged
        # live trading -- correct, and it means exercising the SIGNING path
        # needs the acknowledgement present. Set for this process only, on a
        # freshly generated keypair that holds nothing and is discarded when
        # the test ends, and removed by cleanup so it cannot leak into
        # another test or a shell. No desk is running here and no capital is
        # reachable; this asserts the transport, not a decision to trade.
        import os

        previous = os.environ.get("ALLOW_LIVE_TRADING")
        os.environ["ALLOW_LIVE_TRADING"] = "yes-i-understand"

        def restore():
            if previous is None:
                os.environ.pop("ALLOW_LIVE_TRADING", None)
            else:
                os.environ["ALLOW_LIVE_TRADING"] = previous

        self.addCleanup(restore)
        self.keypair = Keypair()
        # The system program only. An empty allowlist refuses everything,
        # which is the right default and the wrong fixture for testing
        # transport.
        self.service = SignerService(self.keypair, SignerPolicy(
            allowed_programs={"11111111111111111111111111111111"},
            max_transfer_lamports=10 ** 12, rate_limit_per_minute=10_000))
        self.path = Path(tempfile.mkdtemp()) / "signer.sock"

    def _message(self, lamports: int = 1000) -> bytes:
        """A REAL compiled v0 message.

        The signer parses what it is asked to sign and refuses anything it
        cannot read -- so a test that signs random bytes tests the refusal
        path, not the transport. That refusal is a feature; this is how to
        get past it honestly.
        """
        from solders.hash import Hash
        from solders.message import MessageV0
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer

        instruction = transfer(TransferParams(
            from_pubkey=self.keypair.pubkey(), to_pubkey=Pubkey.default(),
            lamports=lamports))
        return bytes(MessageV0.try_compile(
            self.keypair.pubkey(), [instruction], [], Hash.default()))

    def _run(self, body):
        async def main():
            server = _serve(self.service, self.path)
            await server.start()
            try:
                return await body()
            finally:
                await server.stop()

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(main())
        finally:
            loop.close()

    def test_the_binary_client_signs_over_one_held_connection(self):
        from src.execution.signer import SignerClient

        async def body():
            client = SignerClient(self.path)
            key = await client.pubkey()
            signatures = [await client.sign_message(self._message(1000 + index))
                          for index in range(5)]
            await client.close()
            return key, signatures, client

        key, signatures, client = self._run(body)
        self.assertEqual(str(self.keypair.pubkey()), key)
        self.assertEqual(5, len(signatures))
        for signature in signatures:
            self.assertEqual(64, len(signature))
        self.assertEqual(5, client.signed)
        # Six frames (one pubkey, five signs) and NO reconnects: the
        # connection was opened once and held.
        self.assertEqual(6, client.frames)
        self.assertEqual(0, client.reconnects)

    def test_signatures_actually_verify(self):
        from src.execution.signer import SignerClient

        async def body():
            client = SignerClient(self.path)
            message = self._message()
            signature = await client.sign_message(message)
            await client.close()
            return message, signature

        message, signature = self._run(body)
        from solders.signature import Signature

        self.assertTrue(
            Signature.from_bytes(signature).verify(self.keypair.pubkey(), message))

    def test_the_old_json_protocol_still_answers(self):
        # The signer and the desk are separate units, deployed separately. A
        # protocol change that only works if both restart in the right order
        # strands a running desk.
        import base64
        import json

        async def body():
            reader, writer = await asyncio.open_unix_connection(str(self.path))
            writer.write((json.dumps({"op": "pubkey"}) + "\n").encode())
            await writer.drain()
            raw = await reader.readline()
            writer.close()
            return json.loads(raw)

        response = self._run(body)
        self.assertTrue(response["ok"])
        self.assertEqual(str(self.keypair.pubkey()), response["public_key"])

    def test_a_refusal_is_still_a_refusal_on_the_held_connection(self):
        from src.execution.signer import SignerClient, SignerPolicy, SignerService

        # A policy that allows nothing: the refusal must still arrive
        # cleanly over the held connection.
        self.service = SignerService(self.keypair, SignerPolicy(
            allowed_programs=set(), max_transfer_lamports=0,
            rate_limit_per_minute=0))

        async def body():
            client = SignerClient(self.path)
            try:
                await client.sign_message(self._message())
            except PermissionError as exc:
                return str(exc), client
            finally:
                await client.close()
            return "", client

        message, client = self._run(body)
        self.assertIn("refused", message)
        self.assertEqual(1, client.refusals)
        self.assertEqual(0, client.signed)

    def test_the_connection_is_re_established_after_the_signer_restarts(self):
        # A held socket dies on a signer restart or a deploy. The first
        # signature after that must not be the one that fails.
        from src.execution.signer import SignerClient

        async def body():
            client = SignerClient(self.path)
            await client.sign_message(self._message(1))
            client._drop()          # as if the signer had gone away
            await client.sign_message(self._message(2))
            await client.close()
            return client

        client = self._run(body)
        self.assertEqual(2, client.signed)


if __name__ == "__main__":
    unittest.main()
