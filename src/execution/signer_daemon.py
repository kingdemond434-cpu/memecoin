"""The signer, as its own process. Run this and the desk stops holding a key.

    python -m src.execution.signer_daemon

Reads SOLANA_PRIVATE_KEY, serves a unix socket, and signs only what its
policy permits. Nothing else lives here: no models, no HTTP client, no parser
of third-party JSON. That narrowness is the entire point -- the desk's attack
surface is large because a trading desk's attack surface is inherently large,
and this process exists so the key does not sit inside it.

Run it as a separate systemd unit from the desk, ideally as a separate user
with no read access to the desk's working directory. Then point the desk at
it:

    MEMECOIN_SIGNER_SOCKET=/run/memecoin/signer.sock

The desk has no fallback. If this process is not running, the desk cannot
sign -- which is the correct failure, because the alternative is a desk that
quietly reverts to holding its own key the moment the signer has a problem.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from src.execution.signer import (
    DEFAULT_ALLOWED_PROGRAMS, DEFAULT_MAX_TRANSFER_LAMPORTS,
    DEFAULT_RATE_LIMIT, SignerPolicy, SignerServer, SignerService,
)

logger = logging.getLogger("signer")

DEFAULT_SOCKET = Path("/run/memecoin/signer.sock")


def load_keypair() -> Any:
    """The key, from the environment, in any of the encodings people use.

    Refuses rather than guessing. A key that decodes to the wrong length is a
    key that will produce valid-looking signatures for an account nobody owns.
    """
    from solders.keypair import Keypair

    encoded = os.getenv("SOLANA_PRIVATE_KEY", "").strip()
    if not encoded:
        raise SystemExit(
            "SOLANA_PRIVATE_KEY is required.\n"
            "The signer exists to hold this key so the trading desk does not; "
            "without it there is nothing for this process to do.")
    try:
        if encoded.startswith("["):
            return Keypair.from_bytes(bytes(json.loads(encoded)))
        try:
            return Keypair.from_base58_string(encoded)
        except ValueError:
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) == 64:
                return Keypair.from_bytes(raw)
            if len(raw) == 32:
                return Keypair.from_seed(raw)
            raise ValueError(f"decoded to {len(raw)} bytes, expected 32 or 64")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"SOLANA_PRIVATE_KEY could not be decoded: {exc}")


def build_policy(args: argparse.Namespace) -> SignerPolicy:
    return SignerPolicy(
        allowed_programs=tuple(args.allow_program or DEFAULT_ALLOWED_PROGRAMS),
        max_transfer_lamports=int(args.max_transfer_lamports),
        rate_limit_per_minute=int(args.rate_limit),
        kill_file=Path(args.kill_file),
        require_live_ack=not args.no_live_ack,
    )


async def serve(args: argparse.Namespace) -> None:
    keypair = load_keypair()
    service = SignerService(keypair, build_policy(args))
    server = SignerServer(service, Path(args.socket))
    await server.start()
    # The PUBLIC key, so an operator can confirm which account this signs for.
    # No private material is ever logged, printed or reported.
    logger.info("signer ready for %s on %s", service.public_key, args.socket)
    logger.info("policy: %d programs, max %d lamports/tx, %d sigs/min, halt file %s",
                len(service.policy.allowed_programs),
                service.policy.max_transfer_lamports,
                service.policy.rate_limit_per_minute,
                service.policy.kill_file)

    stopping = asyncio.Event()

    def request_stop(*_args):
        stopping.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:  # pragma: no cover - not on this platform
            signal.signal(sig, request_stop)
    await stopping.wait()
    logger.info("signer stopping; signed %d, refused %d",
                service.signed, service.refused)
    await server.stop()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=str(DEFAULT_SOCKET))
    parser.add_argument("--max-transfer-lamports", type=int,
                        default=DEFAULT_MAX_TRANSFER_LAMPORTS,
                        help="ceiling on SOL moved by System transfers per tx")
    parser.add_argument("--rate-limit", type=int, default=DEFAULT_RATE_LIMIT,
                        help="signatures per minute")
    parser.add_argument("--kill-file", default="data/state/HALT_SIGNING",
                        help="presence of this file stops all signing")
    parser.add_argument("--allow-program", action="append",
                        help="repeatable; replaces the default allowlist")
    parser.add_argument("--no-live-ack", action="store_true",
                        help="do not require ALLOW_LIVE_TRADING. Only for a "
                             "signer serving a desk that cannot submit.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s signer %(message)s")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
