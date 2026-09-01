"""An isolated signer: the key never enters the process that decides trades.

Today the private key is read from the environment into the trading process
and held in memory beside every model, every HTTP client and every parser of
untrusted third-party JSON. Any defect in any of that -- a deserialisation
bug, a dependency compromise, a log line that stringifies too much -- reaches
the key. And the key is the whole account.

So the key moves to a separate process that does exactly one thing, and the
trading desk talks to it over a unix socket. The desk builds transactions and
never holds a secret; the signer holds a secret and never decides anything.

The point is NOT merely that the key lives elsewhere. It is that the signer is
an independent authority that can refuse. It re-derives what a transaction
actually does -- by decoding the message itself, not by reading a description
the caller supplied -- and applies policy to that. A caller that has been
compromised, or is simply wrong, cannot talk its way past a check by
mislabelling its request, because the label is never consulted.

Five refusals, each cheap and each covering a way an account dies:

  program allowlist     an instruction against a program we never intended to
                        touch is the shape of both an exploit and a bug. Only
                        the handful this desk actually uses are permitted.
  fee payer            must be our own account. A transaction that makes us
                        pay for someone else's business is not ours.
  lamport ceiling      a cap on SOL moved by System transfers in one
                        transaction. The single most direct drain.
  rate limit           a bounded number of signatures per minute. A loop bug
                        that submits ten thousand times is stopped by the
                        signer regardless of what the desk believes.
  kill file            the presence of one path on disk stops all signing.
                        An operator with shell access can halt the account in
                        one command without finding a process or a config.

Live signing additionally requires an explicit environment acknowledgement,
checked here rather than only in the desk, so a dry-run desk that is
misconfigured into live mode still cannot get a signature.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from src.execution.signer_protocol import (
    HANDSHAKE, HANDSHAKE_SIZE, HEADER_SIZE, MAGIC, OP_PING, OP_PUBKEY,
    OP_SIGN, PROTOCOL_VERSION, STATUS_ERROR, STATUS_OK, STATUS_REFUSED,
    decode_header, encode_request, encode_response, parse_handshake,
    read_frame)
from src.runtime.loop_local import loop_local_lock
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SIGNER_SCHEMA_VERSION = "v1"

#: The programs this desk legitimately touches. Anything else is refused --
#: including programs that are perfectly reputable, because the question is
#: not whether a program is safe but whether we meant to call it.
DEFAULT_ALLOWED_PROGRAMS: Tuple[str, ...] = (
    "11111111111111111111111111111111",             # System
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",   # SPL Token
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # Associated Token
    "ComputeBudget111111111111111111111111111111",   # Compute budget
    "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",   # Pump.fun
    "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",   # PumpSwap (pump_amm)
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter v6
)

#: Maximum SOL a single transaction may move via System transfers.
DEFAULT_MAX_TRANSFER_LAMPORTS = 2_000_000_000  # 2 SOL

#: Signatures per minute.
DEFAULT_RATE_LIMIT = 30

#: Presence of this file stops all signing, immediately, for any caller.
DEFAULT_KILL_FILE = Path("data/state/HALT_SIGNING")

#: The acknowledgement the operator must set for the signer to sign at all.
LIVE_ACK_ENV = "ALLOW_LIVE_TRADING"
LIVE_ACK_VALUE = "yes-i-understand"


@dataclass
class SignerPolicy:
    """What the signer will and will not put its name to."""

    allowed_programs: Tuple[str, ...] = DEFAULT_ALLOWED_PROGRAMS
    max_transfer_lamports: int = DEFAULT_MAX_TRANSFER_LAMPORTS
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT
    kill_file: Path = DEFAULT_KILL_FILE
    require_live_ack: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed_programs": list(self.allowed_programs),
                "max_transfer_lamports": self.max_transfer_lamports,
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "kill_file": str(self.kill_file),
                "require_live_ack": self.require_live_ack}


@dataclass
class Decision:
    """One allow-or-refuse, with the reason recorded either way."""

    allowed: bool
    reason: str = ""
    programs: List[str] = field(default_factory=list)
    transfer_lamports: int = 0
    fee_payer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason,
                "programs": self.programs,
                "transfer_lamports": self.transfer_lamports,
                "fee_payer": self.fee_payer}


def _system_transfer_lamports(program_id: str, data: bytes) -> int:
    """Lamports moved by a System transfer, or zero for anything else.

    The System program's instruction data is a little-endian u32 discriminant
    followed by the arguments; transfer is 2, with a u64 of lamports. Decoded
    here rather than trusted, because the whole point of this process is that
    it does not take the caller's word for what an instruction does.
    """
    if program_id != "11111111111111111111111111111111":
        return 0
    if len(data) < 12:
        return 0
    if int.from_bytes(data[:4], "little") != 2:
        return 0
    return int.from_bytes(data[4:12], "little")


class TransactionInspector:
    """Decodes a serialized message and says what it actually does."""

    def __init__(self, policy: SignerPolicy):
        self.policy = policy

    def inspect(self, message_bytes: bytes, *, expected_payer: str) -> Decision:
        """Independently derive the transaction's effects, then judge them."""
        try:
            from solders.message import MessageV0
        except ImportError as exc:  # pragma: no cover - solders is a hard dep
            return Decision(allowed=False, reason=f"cannot decode: {exc}")
        raw = bytes(message_bytes)
        # What gets SIGNED on Solana is the versioned payload -- the message
        # with a 0x80 version prefix -- but the decoder wants it without.
        # Strip for reading, sign what we were handed. Getting this backwards
        # is invisible to a local signer, which never decodes, and refuses
        # every single transaction once the signer is isolated.
        decodable = raw[1:] if raw and raw[0] & 0x80 else raw
        try:
            message = MessageV0.from_bytes(decodable)
        except Exception as exc:
            # An undecodable message is refused. Signing bytes we cannot read
            # is signing a blank cheque.
            return Decision(allowed=False,
                            reason=f"message did not decode: {type(exc).__name__}")

        keys = [str(key) for key in message.account_keys]
        if not keys:
            return Decision(allowed=False, reason="message carries no account keys")
        fee_payer = keys[0]
        programs: List[str] = []
        transfer_total = 0
        for instruction in message.instructions:
            index = int(instruction.program_id_index)
            if index >= len(keys):
                # An address-table lookup we cannot resolve locally. Refused:
                # a program we cannot name is a program we cannot allowlist.
                return Decision(
                    allowed=False, fee_payer=fee_payer,
                    reason=("instruction references a program outside the "
                            "static key set; it cannot be identified here"))
            program_id = keys[index]
            programs.append(program_id)
            transfer_total += _system_transfer_lamports(
                program_id, bytes(instruction.data))

        decision = Decision(allowed=True, programs=programs,
                            transfer_lamports=transfer_total,
                            fee_payer=fee_payer)

        if fee_payer != expected_payer:
            decision.allowed = False
            decision.reason = (f"fee payer {fee_payer[:8]}... is not this "
                               f"signer's account")
            return decision
        unknown = [item for item in programs
                   if item not in self.policy.allowed_programs]
        if unknown:
            decision.allowed = False
            decision.reason = (f"instruction against unlisted program "
                               f"{unknown[0]}")
            return decision
        if transfer_total > self.policy.max_transfer_lamports:
            decision.allowed = False
            decision.reason = (f"moves {transfer_total} lamports, above the "
                               f"{self.policy.max_transfer_lamports} ceiling")
            return decision
        return decision


class SignerService:
    """Holds the key. Decides nothing except whether to sign.

    Deliberately small and dependency-light: everything in this class is
    auditable in one sitting, which is the property that makes isolating the
    key worth doing at all.
    """

    def __init__(self, keypair: Any, policy: Optional[SignerPolicy] = None):
        self.keypair = keypair
        self.policy = policy or SignerPolicy()
        self.inspector = TransactionInspector(self.policy)
        self.public_key = str(keypair.pubkey())
        self._recent: Deque[float] = deque()
        self.signed = 0
        self.refused = 0
        self.refusals: Deque[Dict[str, Any]] = deque(maxlen=100)

    # --- gates -----------------------------------------------------------

    def _halted(self) -> str:
        try:
            if self.policy.kill_file.exists():
                return f"halt file present at {self.policy.kill_file}"
        except OSError:
            pass
        if self.policy.require_live_ack:
            if os.getenv(LIVE_ACK_ENV, "").strip().lower() != LIVE_ACK_VALUE:
                return (f"{LIVE_ACK_ENV} is not set to the acknowledgement; the "
                        "signer will not sign for an unacknowledged desk")
        return ""

    def _rate_limited(self, now: float) -> str:
        while self._recent and now - self._recent[0] > 60.0:
            self._recent.popleft()
        if len(self._recent) >= self.policy.rate_limit_per_minute:
            return (f"{len(self._recent)} signatures in the last minute is at "
                    f"the limit of {self.policy.rate_limit_per_minute}")
        return ""

    # --- the one operation -----------------------------------------------

    def sign(self, message_bytes: bytes, now: Optional[float] = None
             ) -> Tuple[Optional[bytes], Decision]:
        """Sign, or refuse with a reason. Never raises on a refusal."""
        moment = time.time() if now is None else now
        halted = self._halted()
        if halted:
            return None, self._refuse(halted)
        limited = self._rate_limited(moment)
        if limited:
            return None, self._refuse(limited)
        decision = self.inspector.inspect(message_bytes,
                                          expected_payer=self.public_key)
        if not decision.allowed:
            self.refused += 1
            self.refusals.append({"at": moment, **decision.to_dict()})
            logger.warning("signer refused: %s", decision.reason)
            return None, decision
        signature = self.keypair.sign_message(bytes(message_bytes))
        self._recent.append(moment)
        self.signed += 1
        return bytes(signature), decision

    def _refuse(self, reason: str) -> Decision:
        self.refused += 1
        decision = Decision(allowed=False, reason=reason)
        self.refusals.append({"at": time.time(), **decision.to_dict()})
        logger.warning("signer refused: %s", reason)
        return decision

    def report(self) -> Dict[str, Any]:
        """Never includes the key, in any encoding, under any key name."""
        return {
            "schema": SIGNER_SCHEMA_VERSION,
            "public_key": self.public_key,
            "signed": self.signed,
            "refused": self.refused,
            "halted": bool(self._halted()),
            "halt_reason": self._halted(),
            "policy": self.policy.to_dict(),
            "recent_refusals": list(self.refusals)[-10:],
        }


class SignerServer:
    """Unix-socket front end for the service. One request, one response."""

    def __init__(self, service: SignerService, socket_path: Path):
        self.service = service
        self.socket_path = Path(socket_path)
        self._server: Optional[asyncio.AbstractServer] = None
        # Binary connections are HELD, so a handler outlives its request and
        # is still awaiting the next frame at shutdown. Tracked so stop()
        # can cancel them; without this they are garbage-collected mid-await
        # and asyncio reports "coroutine ignored GeneratorExit" -- noise
        # beside a real fault, in the log an operator reads during one.
        self._connections: "set" = set()
        # Counted rather than merely logged. A client that cannot complete a
        # handshake is a deployment fault -- a version skew, or something
        # else entirely on the socket -- and it is invisible in a log nobody
        # is reading at 3am but obvious in /status.
        self.bad_handshakes = 0
        self.version_mismatches = 0

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path))
        # Owner-only. The socket IS the key as far as authority goes, so it
        # gets the same permissions a key file would.
        os.chmod(self.socket_path, 0o600)
        logger.info("signer listening on %s for %s",
                    self.socket_path, self.service.public_key)

    async def stop(self) -> None:
        for task in list(self._connections):
            task.cancel()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
            self._connections.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    def report(self) -> Dict[str, Any]:
        """What the socket has been asked to serve, including what it refused.

        A client that cannot complete a handshake is a deployment fault -- a
        version skew between two separately deployed units, or something else
        entirely on the socket -- and it is invisible in a log nobody reads at
        3am and obvious in a status endpoint.
        """
        return {
            "schema": SIGNER_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "held_connections": len(self._connections),
            "bad_handshakes": self.bad_handshakes,
            "version_mismatches": self.version_mismatches,
            "service": self.service.report(),
        }

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        """Serve one connection, in whichever protocol it opens with.

        Both protocols, chosen by an EXPLICIT handshake rather than by
        guessing from the first byte, and the connection is HELD for the
        binary one -- that is the whole point of it. A JSON caller still gets
        one request and a close, exactly as before.

        The guess was wrong about one signature in 256. On a binary
        connection the first byte is the low byte of a little-endian u32
        length, and 0x7B is `{`: a frame whose body is 123, 379, 635 or 891
        bytes long opened with `{` and was parsed as JSON. A Solana message
        of 378 or 634 bytes is an ordinary transaction, so this was reachable
        on the one path where being wrong means an unsigned transaction or a
        stalled connection. `MCS` cannot begin a JSON object.

        Keeping JSON is not politeness. The signer and the desk are separate
        units that get deployed separately, and a protocol change that only
        works when both restart in the right order is a protocol change that
        strands a running desk on the wrong afternoon. The version byte is
        what makes that survivable in the other direction too: a mismatch is
        stated, not silently desynchronised.
        """
        current = asyncio.current_task()
        if current is not None:
            self._connections.add(current)
            current.add_done_callback(self._connections.discard)
        try:
            first = await reader.read(1)
            if not first:
                return
            if first == MAGIC[:1]:
                rest = await reader.readexactly(HANDSHAKE_SIZE - 1)
                try:
                    version = parse_handshake(first + rest)
                except ValueError as exc:
                    writer.write(encode_response(
                        STATUS_ERROR, str(exc).encode("utf-8")))
                    await writer.drain()
                    writer.close()
                    return
                if version != PROTOCOL_VERSION:
                    # Stated, not tolerated. Two units that disagree about
                    # the frame layout produce garbage that decodes as
                    # something, and the something is a signing request.
                    self.version_mismatches += 1
                    writer.write(encode_response(
                        STATUS_ERROR,
                        f"protocol version {version} but this signer speaks "
                        f"{PROTOCOL_VERSION}".encode("utf-8")))
                    await writer.drain()
                    writer.close()
                    return
                await self._serve_binary(reader, writer)
                return
            if first != b"{":
                # Neither protocol. Refused with a reason rather than
                # silently dropped, because a client that has the socket
                # path and the wrong idea should be told which it is.
                self.bad_handshakes += 1
                writer.write(encode_response(
                    STATUS_ERROR, b"expected a JSON object or the binary handshake"))
                await writer.drain()
                writer.close()
                return
            raw = first + await reader.readline()
            request = json.loads(raw)
            if request.get("op") == "pubkey":
                response = {"ok": True, "public_key": self.service.public_key}
            elif request.get("op") == "sign":
                import base64
                message = base64.b64decode(request.get("message", ""))
                signature, decision = self.service.sign(message)
                response = {"ok": signature is not None,
                            "signature": (base64.b64encode(signature).decode()
                                          if signature else ""),
                            "decision": decision.to_dict()}
            else:
                response = {"ok": False, "error": "unknown op"}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
        finally:
            writer.close()

    async def _serve_binary(self, reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter) -> None:
        """Length-prefixed frames on a connection held until the peer leaves.

        Every frame is authorised independently. Holding the connection open
        buys the desk a connect and two encodings per signature; it buys it
        no standing permission, because the policy check happens per message
        and always has.
        """
        try:
            header = await reader.readexactly(HEADER_SIZE)
            while True:
                length, op = decode_header(header)
                payload = await reader.readexactly(length) if length else b""
                if op == OP_PUBKEY:
                    frame = encode_response(
                        STATUS_OK, self.service.public_key.encode("ascii"))
                elif op == OP_PING:
                    frame = encode_response(STATUS_OK)
                elif op == OP_SIGN:
                    signature, decision = self.service.sign(payload)
                    frame = (encode_response(STATUS_OK, signature)
                             if signature is not None else
                             encode_response(
                                 STATUS_REFUSED,
                                 str(decision.reason).encode("utf-8")[:4096]))
                else:
                    frame = encode_response(
                        STATUS_ERROR, f"unknown op {op}".encode("utf-8"))
                writer.write(frame)
                await writer.drain()
                header = await reader.readexactly(HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionError,
                asyncio.CancelledError):
            pass  # the peer went away, or we are shutting down
        except Exception as exc:  # pragma: no cover - defensive
            try:
                writer.write(encode_response(
                    STATUS_ERROR, f"{type(exc).__name__}: {exc}".encode("utf-8")))
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()


class SignerClient:
    """What the desk holds instead of a key.

    ``isolated`` is True: the key is in another process.

    Shaped like the part of a Keypair the desk actually uses, so the execution
    path does not need to know whether it is talking to a local key or to an
    isolated signer -- which is what makes adopting this a configuration
    change rather than a rewrite of the hot path.
    """

    isolated = True

    def __init__(self, socket_path: Path, *, timeout_s: float = 2.0):
        self.socket_path = Path(socket_path)
        self.timeout_s = float(timeout_s)
        self._public_key: Optional[str] = None
        self.signed = 0
        self.refusals = 0
        self.last_refusal = ""
        # One connection for the life of the process. Every signature used to
        # open a new Unix socket, JSON-encode a dict and base64 the message
        # -- a connect, a round trip through the kernel's accept path and two
        # encodings, on the one path where microseconds are the product.
        self._reader = None
        self._writer = None
        self._lock = loop_local_lock("signer")
        self.reconnects = 0
        self.binary = True
        self.frames = 0

    async def _connect(self):
        """Open the persistent connection, on the loop that will use it."""
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(self.socket_path)),
            timeout=self.timeout_s)
        # Declared, not inferred. Written before any frame so the server
        # never has to guess from a length byte which protocol this is --
        # and carried on the same write as the first frame by the kernel, so
        # the handshake costs no round trip.
        self._writer.write(HANDSHAKE)
        return self._reader, self._writer

    async def _frame(self, op: int, payload: bytes = b"") -> Tuple[int, bytes]:
        """One request/response over the held connection, reconnecting once.

        Exactly one retry, and only when the connection itself failed. A held
        socket can be closed by a signer restart, a deploy, or an idle
        timeout, and the first signature after that must not be the one that
        fails -- but retrying a REFUSAL would ask a signer that already said
        no to say it again, and retrying repeatedly on a dead signer would
        turn a clear failure into a slow one.
        """
        async with self._lock:
            for attempt in (0, 1):
                try:
                    if self._writer is None or self._writer.is_closing():
                        await self._connect()
                        if attempt:
                            self.reconnects += 1
                    self._writer.write(encode_request(op, payload))
                    await self._writer.drain()
                    frame = await asyncio.wait_for(
                        read_frame(self._reader), timeout=self.timeout_s)
                    if frame is None:
                        raise ConnectionError("signer closed the connection")
                    self.frames += 1
                    return frame
                except (ConnectionError, asyncio.IncompleteReadError, OSError):
                    self._drop()
                    if attempt:
                        raise
            raise ConnectionError("signer unreachable")

    def _drop(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # pragma: no cover - teardown only
                pass
        self._reader = None
        self._writer = None

    async def close(self) -> None:
        self._drop()

    async def _call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """The old JSON path, kept for anything that still speaks it."""
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(self.socket_path)),
            timeout=self.timeout_s)
        try:
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=self.timeout_s)
            return json.loads(raw) if raw else {"ok": False, "error": "no response"}
        finally:
            writer.close()

    async def pubkey(self) -> str:
        if self._public_key is None:
            status, payload = await self._frame(OP_PUBKEY)
            if status != STATUS_OK:
                raise RuntimeError(
                    f"signer unreachable: {payload.decode('utf-8', 'replace')}")
            self._public_key = payload.decode("ascii")
        return self._public_key

    async def sign_message(self, message_bytes: bytes) -> bytes:
        """Signature, or an exception carrying the signer's stated reason.

        A refusal raises rather than returning empty: an unsigned transaction
        that flows onward as if it were signed is the failure this whole
        design exists to prevent, and a caller that ignores a return value is
        far more likely than one that ignores an exception.
        """
        status, payload = await self._frame(OP_SIGN, bytes(message_bytes))
        if status != STATUS_OK:
            reason = payload.decode("utf-8", "replace") or "unknown"
            self.refusals += 1
            self.last_refusal = reason
            raise PermissionError(f"signer refused: {reason}")
        if len(payload) != 64:
            # A 64-byte signature is the only correct answer. Anything else
            # is a framing fault, and letting it through would put arbitrary
            # bytes where a signature goes.
            raise RuntimeError(
                f"signer returned {len(payload)} bytes where a signature belongs")
        self.signed += 1
        return payload

    def report(self) -> Dict[str, Any]:
        return {
            "schema": SIGNER_SCHEMA_VERSION,
            "mode": "isolated",
            "isolated": True,
            "public_key": self._public_key or "not yet resolved",
            "socket": str(self.socket_path),
            "signed": self.signed,
            "refused": self.refusals,
            "last_refusal": self.last_refusal,
            "transport": "binary/persistent" if self.binary else "json",
            "frames": self.frames,
            "reconnects": self.reconnects,
            "detail": ("one held connection and length-prefixed frames; the "
                       "key is still in its own process"),
        }


class LocalSigner:
    """The key in this process. What the desk used to do everywhere.

    Kept as an explicit, named class rather than as a bare Keypair so that
    "the key is in the trading process" is a visible configuration state that
    reports itself, instead of the silent default it used to be. A desk
    running this in live mode is a desk taking a risk it should have chosen
    deliberately.
    """

    isolated = False

    def __init__(self, keypair: Any):
        self._keypair = keypair
        self.public_key = str(keypair.pubkey())
        self.signed = 0
        self.refused = 0
        self.last_refusal = ""

    async def sign_message(self, message_bytes: bytes) -> bytes:
        self.signed += 1
        return bytes(self._keypair.sign_message(bytes(message_bytes)))

    def report(self) -> Dict[str, Any]:
        return {
            "schema": SIGNER_SCHEMA_VERSION,
            "mode": "local",
            "isolated": False,
            "public_key": self.public_key,
            "signed": self.signed,
            "refused": 0,
            "detail": ("the private key is held in the trading process; an "
                       "isolated signer is not configured"),
        }


def signer_from_env(keypair: Any, socket_path: Optional[Path] = None
                    ) -> Any:
    """A SignerClient when a socket is configured, otherwise a LocalSigner.

    The choice is made once, at startup, and reported. There is deliberately
    no runtime fallback from the isolated signer to the local one: a fallback
    would mean a compromised or misconfigured signer silently returns the
    desk to holding its own key, which is precisely the state isolation
    exists to leave.
    """
    configured = socket_path or os.getenv("MEMECOIN_SIGNER_SOCKET", "").strip()
    if configured:
        return SignerClient(Path(configured))
    return LocalSigner(keypair)
