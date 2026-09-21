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


class TheProtocolIsDeclaredNotGuessed(unittest.TestCase):
    """The first byte was a guess, and the guess was reachably wrong.

    The server told the two protocols apart by calling anything that was not
    `{` a binary frame. On a binary connection that byte is the LOW BYTE of a
    little-endian u32 length, and 0x7B is `{` -- so a frame whose body is
    123, 379, 635 or 891 bytes long opened with `{` and was handed to
    `json.loads`. A Solana message of 378 or 634 bytes is an ordinary
    transaction. This was not a theoretical collision on an exotic input; it
    was roughly one signature in 256 on the path where being wrong means an
    unsigned transaction or a stalled connection.
    """

    def test_the_lengths_that_used_to_look_like_json_really_do_occur(self):
        from src.execution.signer_protocol import encode_request, OP_SIGN

        # Every payload length whose frame header starts with `{`.
        collisions = [size for size in range(1, 1300)
                      if encode_request(OP_SIGN, b"\x00" * size)[0:1] == b"{"]
        self.assertEqual([122, 378, 634, 890, 1146], collisions)
        # And those are transaction-sized, which is the whole problem.
        self.assertTrue(any(200 <= size <= 1232 for size in collisions))

    def test_the_handshake_cannot_be_confused_with_json(self):
        from src.execution.signer_protocol import HANDSHAKE, MAGIC

        self.assertNotEqual(b"{", MAGIC[:1])
        self.assertEqual(4, len(HANDSHAKE))

    def test_a_wrong_version_is_stated_not_tolerated(self):
        from src.execution.signer_protocol import (
            MAGIC, PROTOCOL_VERSION, parse_handshake)

        self.assertEqual(PROTOCOL_VERSION, parse_handshake(MAGIC + b"\x01"))
        self.assertEqual(9, parse_handshake(MAGIC + b"\x09"))
        with self.assertRaises(ValueError):
            parse_handshake(b"HTTP")
        with self.assertRaises(ValueError):
            parse_handshake(b"MCS")


class ItSurvivesWhateverArrivesOnTheSocket(BothProtocolsWorkAgainstARealSocket):
    """A signer holds the only key. What reaches its socket is not always a
    client of ours: a port scan, a half-deployed desk, a truncated write from
    a process that died mid-frame. None of it may hang the server, crash the
    handler, or -- above all -- get something signed."""

    def _speak(self, payloads, read_reply=True):
        async def body():
            replies = []
            for payload in payloads:
                reader, writer = await asyncio.open_unix_connection(str(self.path))
                writer.write(payload)
                await writer.drain()
                # EOF immediately. A frame that declares more than it sends
                # must end the connection rather than leave the server
                # waiting for bytes that will never arrive -- and without the
                # EOF this test would itself sit through that wait.
                if writer.can_write_eof():
                    writer.write_eof()
                if read_reply:
                    try:
                        replies.append(await asyncio.wait_for(
                            reader.read(4096), timeout=1.0))
                    except asyncio.TimeoutError:
                        replies.append(None)
                writer.close()
            return replies

        return self._run(body)

    def test_garbage_first_bytes_are_refused_with_a_reason(self):
        replies = self._speak([b"GET / HTTP/1.1\r\n\r\n", b"\x00\x00\x00\x00",
                               b"\xff" * 16, b"MCX\x01"])
        for reply in replies:
            self.assertIsNotNone(reply, "the server hung on garbage")
            self.assertTrue(reply, "the server closed without saying why")

    def test_a_frame_length_that_never_arrives_does_not_hang_the_server(self):
        from src.execution.signer_protocol import HANDSHAKE

        # Declares 4096 bytes, sends three. The handler must give up on the
        # connection rather than block a slot for ever.
        truncated = HANDSHAKE + struct.pack("<IB", 4096, 2) + b"abc"
        self._speak([truncated], read_reply=False)
        # And the signer is still serving afterwards.
        from src.execution.signer import SignerClient

        async def body():
            client = SignerClient(self.path)
            try:
                self.assertEqual(str(self.keypair.pubkey()), await client.pubkey())
            finally:
                await client.close()

        self._run(body)

    def test_fuzzed_frames_never_produce_a_signature(self):
        import random

        from src.execution.signer_protocol import (
            HANDSHAKE, STATUS_OK, decode_header)

        random.seed(20260901)
        frames = []
        for _ in range(60):
            declared = random.choice([0, 1, 2, 7, 123, 379, 635,
                                      random.randint(1, 2048), 10 ** 6])
            op = random.randint(0, 255)
            body = bytes(random.getrandbits(8) for _ in range(random.randint(0, 64)))
            frames.append(HANDSHAKE + struct.pack("<IB", declared, op) + body)

        signed_before = self.service.signed
        replies = self._speak(frames)
        # Nothing random gets signed. The only op that signs is OP_SIGN, and
        # it signs only a message the policy parses and allows -- so a fuzzer
        # that happens to pick op 2 still gets a refusal, not a signature.
        self.assertEqual(signed_before, self.service.signed)
        for reply in replies:
            if not reply or len(reply) < 5:
                continue
            _length, status = decode_header(reply[:5])
            if status == STATUS_OK:
                # An OK reply to a fuzzed frame is only legitimate for the
                # ops that carry no message: pubkey and ping.
                self.assertNotEqual(64, len(reply) - 5,
                                    "a fuzzed frame produced something "
                                    "signature-shaped")

    def test_a_transaction_sized_frame_that_starts_with_a_brace_still_works(self):
        """The exact case the old detection got wrong, end to end.

        378 payload bytes make a header of `{`, `\\x01`, `\\x00`, `\\x00`,
        op -- so the server read `{`, called `json.loads` on the rest of the
        "line", and answered with a JSON error on a stream the client was
        reading as length-prefixed frames. The client then decoded `{"ok`
        as a header and everything after it was garbage that decoded as
        something.

        A payload of arbitrary bytes is not a message the policy can parse,
        so the correct answer is a REFUSAL. What this asserts is that it is a
        refusal -- a well-formed frame carrying a reason -- and that the held
        connection is still usable afterwards, which is exactly what the
        misdetection destroyed.
        """
        from src.execution.signer import SignerClient
        from src.execution.signer_protocol import encode_request, OP_SIGN

        self.assertEqual(b"{", encode_request(OP_SIGN, b"\x00" * 378)[0:1])

        async def body():
            client = SignerClient(self.path)
            try:
                with self.assertRaises(PermissionError):
                    await client.sign_message(b"\x00" * 378)
                # The connection survived, and the next signature is real.
                signature = await client.sign_message(self._message())
                self.assertEqual(64, len(signature))
                self.assertEqual(0, client.reconnects)
            finally:
                await client.close()

        self._run(body)
