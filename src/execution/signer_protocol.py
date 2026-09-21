"""A binary frame between the desk and its signer, on one persistent socket.

The signer must stay in its own process -- the key does not belong in the
process that talks to the internet, and no latency argument changes that.
What was never necessary is the cost of TALKING to it. Every signature used
to pay:

    open a new Unix connection
    JSON-encode a dict
    base64-encode the message
    write, read a line
    JSON-decode
    base64-decode
    close the connection

for a request that is, in substance, "here are 300 bytes, sign them". The
connection setup alone is two syscalls and a round trip through the kernel's
accept path, repeated for every transaction, on the one path where
microseconds are the product.

So: one connection held for the life of the process, and frames that are
already the shape of the data.

    handshake MCS[u8 version]          once, when the connection opens
    request   [u32 len][u8 op][payload]
    response  [u32 len][u8 status][payload]

Length-prefixed rather than newline-delimited because the payload is
arbitrary bytes -- a signature contains 0x0a about one time in thirty, and a
line-delimited protocol carrying binary is a bug waiting for the right
signature.

The security properties are unchanged, which is the point. The signer still
runs as its own process, still holds the only copy of the key, still parses
and independently validates every message against its policy, and still
refuses with a reason. This changes how the bytes get there, not who is
allowed to sign what.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

PROTOCOL_VERSION = 1

#: The four bytes a binary connection opens with, before any frame.
#:
#: The server used to tell the two protocols apart by looking at the first
#: byte and calling anything that was not ``{`` binary. That byte, on a
#: binary connection, is the LOW BYTE of a little-endian u32 length -- and
#: 0x7B is ``{``. So a frame whose body is 123 bytes long, or 379, or 635, or
#: 891, opens with ``{`` and was parsed as JSON. Those are not exotic
#: lengths: a Solana message of 378 or 634 bytes is an ordinary transaction,
#: which means the detection was wrong roughly one signature in 256 on the
#: one path where being wrong means an unsigned transaction or a hang.
#:
#: An explicit handshake removes the guess. ``MCS`` cannot begin a JSON
#: object, and the version byte turns a protocol mismatch between separately
#: deployed units into a stated error instead of a desynchronised stream
#: that decodes as something.
MAGIC = b"MCS"
HANDSHAKE = MAGIC + bytes([PROTOCOL_VERSION])
HANDSHAKE_SIZE = len(HANDSHAKE)

#: Operations. Deliberately few: the surface a compromised desk could reach
#: is the surface the signer has to defend, so it stays minimal.
OP_PUBKEY = 1
OP_SIGN = 2
OP_PING = 3

#: Statuses.
STATUS_OK = 0
STATUS_REFUSED = 1
STATUS_ERROR = 2

#: A Solana transaction is at most 1232 bytes; anything far larger is not a
#: message this signer should be looking at, and an unbounded length prefix
#: is how a framing bug becomes a memory exhaustion.
MAX_FRAME = 65_536

_HEADER = struct.Struct("<IB")


def encode_request(op: int, payload: bytes = b"") -> bytes:
    """One request frame."""
    if len(payload) + 1 > MAX_FRAME:
        raise ValueError(f"payload of {len(payload)} exceeds the frame limit")
    return _HEADER.pack(len(payload) + 1, op) + payload


def encode_response(status: int, payload: bytes = b"") -> bytes:
    """One response frame."""
    if len(payload) + 1 > MAX_FRAME:
        raise ValueError(f"payload of {len(payload)} exceeds the frame limit")
    return _HEADER.pack(len(payload) + 1, status) + payload


def decode_header(header: bytes) -> Tuple[int, int]:
    """(body length, op-or-status). Body length EXCLUDES the op byte."""
    if len(header) != _HEADER.size:
        raise ValueError(f"header must be {_HEADER.size} bytes")
    length, code = _HEADER.unpack(header)
    if length < 1 or length > MAX_FRAME:
        raise ValueError(f"frame length {length} outside 1..{MAX_FRAME}")
    return length - 1, code


HEADER_SIZE = _HEADER.size


async def read_frame(reader) -> Optional[Tuple[int, bytes]]:
    """(code, payload) from a stream, or None at a clean end.

    `readexactly` rather than `read`: a short read on a length-prefixed
    protocol silently desynchronises the stream, and every subsequent frame
    is then garbage that decodes as something.
    """
    try:
        header = await reader.readexactly(HEADER_SIZE)
    except Exception:
        return None
    length, code = decode_header(header)
    payload = await reader.readexactly(length) if length else b""
    return code, payload


def parse_handshake(handshake: bytes) -> int:
    """The peer's protocol version, or ValueError with what was wrong.

    Raising rather than returning a sentinel because every caller of this is
    about to either serve or refuse a connection, and a version mismatch that
    reads as "version 0" is exactly the silent desynchronisation the
    handshake exists to prevent.
    """
    if len(handshake) != HANDSHAKE_SIZE:
        raise ValueError(f"handshake must be {HANDSHAKE_SIZE} bytes")
    if handshake[:len(MAGIC)] != MAGIC:
        raise ValueError(f"not a signer connection: {handshake!r}")
    return handshake[len(MAGIC)]
