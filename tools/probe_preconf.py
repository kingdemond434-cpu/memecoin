"""Capture real preconfirmation frames so a decoder can be written against bytes.

The preconfirmation stream is documented as compact BINARY frames behind a
fixed-size header -- which is exactly why it is fast, and exactly why nobody
should write a decoder for it from a description. A binary layout guessed from
prose is a decoder that parses plausible garbage: the fields line up, the
values look like slots and pubkeys, and it is wrong in a way that only shows
up as trades against tokens that do not exist.

So this tool does not decode anything. It connects, subscribes with whatever
payload the docs give you, and DUMPS what arrives: frame sizes, the first
bytes as hex, and the raw frames to a file. Hand that output over and the
decoder gets written against real bytes.

    export HELIUS_API_KEY=...
    .venv/bin/python tools/probe_preconf.py \
        --url 'wss://<the endpoint from your dashboard>?api-key=$HELIUS_API_KEY' \
        --subscribe subscribe.json \
        --frames 40 --out data/state/preconf_frames.bin

`--subscribe` is a JSON file containing the subscribe message exactly as the
documentation shows it. Copy it verbatim rather than reconstructing it: the
method name and parameter shape are the two things most likely to be
mis-remembered, and a wrong subscribe returns an error frame that looks like
data.

What to send back: everything this prints. The hex prefixes, the size
histogram, and the offsets report. That is enough to derive the header layout
without either of us guessing at it.

Two things it deliberately does NOT do. It never prints your API key -- the
URL is redacted in every line of output. And it never writes a decoder or
feeds anything into the desk; this is a capture tool, and a capture tool that
started interpreting its own output would be the guess it exists to prevent.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SECRET = re.compile(r"(api[-_]?key=)[^&\s]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Never the key. Names and shapes only, in every line this prints."""
    return _SECRET.sub(r"\1<redacted>", text or "")


def describe(frames: List[bytes]) -> None:
    """Everything derivable from the bytes without assuming a layout."""
    if not frames:
        print("no frames arrived")
        return
    sizes = collections.Counter(len(frame) for frame in frames)
    print(f"\n{len(frames)} frame(s) captured")
    print("\nsize histogram (a fixed header shows up as a constant prefix, and")
    print("a variable body shows up as a spread of totals)")
    for size, count in sorted(sizes.items()):
        print(f"  {size:6d} bytes  x{count}")

    shortest = min(len(frame) for frame in frames)
    print(f"\nfirst {min(shortest, 48)} bytes of the first five frames")
    for frame in frames[:5]:
        print(f"  {frame[:48].hex()}")

    # Which byte offsets never change across frames. A fixed header's constant
    # fields (version, kind) sit here; its varying fields (slot, index) do not.
    # This is the single most useful thing for deriving a layout, and it is
    # arithmetic rather than interpretation.
    common = min(shortest, 64)
    constant = [offset for offset in range(common)
                if len({frame[offset] for frame in frames}) == 1]
    varying = [offset for offset in range(common) if offset not in constant]
    print(f"\nbyte offsets 0..{common - 1} that are IDENTICAL in every frame:")
    print(f"  {constant}")
    print("byte offsets that vary between frames:")
    print(f"  {varying}")
    print("\n(constant runs are usually version/kind bytes; varying runs are")
    print(" usually slot, timestamp or index fields, and their width tells you")
    print(" the integer size)")

    printable = sum(1 for frame in frames
                    if frame[:1] in (b"{", b"[") or frame[:1].isdigit())
    if printable:
        print(f"\n{printable} frame(s) start like JSON rather than binary --")
        print("that is usually the handshake acknowledgement or an ERROR, and")
        print("its text is worth reading before anything else:")
        for frame in frames[:20]:
            if frame[:1] in (b"{", b"["):
                print("  " + redact(frame[:400].decode("utf-8", "replace")))
                break


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True,
                        help="the websocket endpoint, key included")
    parser.add_argument("--subscribe", default="",
                        help="path to a JSON file holding the subscribe message, "
                             "copied verbatim from the documentation")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="data/state/preconf_frames.bin")
    args = parser.parse_args()

    url = os.path.expandvars(args.url)
    if "$" in url:
        print("the URL still contains an unexpanded variable; export it first",
              file=sys.stderr)
        return 2
    print(f"connecting to {redact(url)}")

    try:
        import websockets
    except ImportError:
        print("websockets is not installed: .venv/bin/pip install websockets",
              file=sys.stderr)
        return 2

    subscribe: Optional[str] = None
    if args.subscribe:
        with open(args.subscribe, "r", encoding="utf-8") as handle:
            subscribe = json.dumps(json.load(handle))
        print(f"subscribe payload: {redact(subscribe)[:400]}")

    frames: List[bytes] = []
    text_frames = 0
    started = time.time()
    try:
        async with websockets.connect(url, max_size=None,
                                      open_timeout=15) as socket:
            print("connected")
            if subscribe:
                await socket.send(subscribe)
                print("subscribe sent")
            while len(frames) < args.frames and time.time() - started < args.seconds:
                remaining = args.seconds - (time.time() - started)
                try:
                    message = await asyncio.wait_for(socket.recv(),
                                                     timeout=max(1.0, remaining))
                except asyncio.TimeoutError:
                    break
                if isinstance(message, str):
                    text_frames += 1
                    message = message.encode("utf-8")
                frames.append(message)
                if len(frames) <= 3:
                    print(f"frame {len(frames)}: {len(message)} bytes")
    except Exception as exc:
        # Redacted, because a connection error commonly echoes the URL back.
        print(f"connection failed: {redact(f'{type(exc).__name__}: {exc}')}",
              file=sys.stderr)
        if frames:
            describe(frames)
        return 1

    if text_frames:
        print(f"\n{text_frames} frame(s) arrived as TEXT rather than binary")

    describe(frames)

    if frames and args.out:
        directory = os.path.dirname(args.out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "wb") as handle:
            # Length-prefixed so frame boundaries survive the file. A flat
            # concatenation loses exactly the boundaries a decoder needs.
            for frame in frames:
                handle.write(len(frame).to_bytes(4, "big"))
                handle.write(frame)
        print(f"\n{len(frames)} frame(s) written to {args.out}")
        print("send the output above; the file stays on this node unless you "
              "choose to share it")
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
