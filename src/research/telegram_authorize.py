"""Create the persistent read-only Telegram session used by the shadow collector.

Run once, interactively, before starting the desk. Telegram sends a login code
to the account's own device and there is no way around that -- which is why the
session has to be created by a person at a terminal and cannot be created by a
systemd unit, where there is no stdin to type the code on.

The credentials live in the operator's environment file, which systemd reads
through `EnvironmentFile=` and an interactive shell does not. So this reads
that file itself rather than failing at the exact moment the runbook tells
someone to run it: a tool whose documented invocation does not work is a tool
that will be run wrong every time.

No credential is printed, and the session file is written 0600.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Optional

from telethon import TelegramClient

#: Where the installer puts the environment file, and where the shadow unit
#: reads it from. Checked in order; the first that exists wins.
ENV_CANDIDATES = (
    Path.home() / ".config" / "memecoin-shadow" / "env",
    Path("/etc/memecoin-shadow/env"),
    Path(".env"),
)

SESSION_PATH = Path("data/telegram/collector")


def parse_env_file(path: Path) -> Dict[str, str]:
    """KEY=VALUE pairs from an environment file.

    Deliberately not a shell source: this file is read by systemd, which does
    not run a shell either, so anything requiring one would work here and fail
    in the unit. Quotes are stripped because systemd strips them; `export` is
    tolerated because people write it.
    """
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            values[name] = value
    return values


def load_credentials() -> tuple:
    """(api_id, api_hash, where they came from). Never returns a value elsewhere."""
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    if api_id and api_hash:
        return api_id, api_hash, "the environment"
    for candidate in ENV_CANDIDATES:
        if not candidate.exists():
            continue
        values = parse_env_file(candidate)
        api_id = api_id or values.get("TELEGRAM_API_ID", "").strip()
        api_hash = api_hash or values.get("TELEGRAM_API_HASH", "").strip()
        if api_id and api_hash:
            return api_id, api_hash, str(candidate)
    return api_id, api_hash, ""


def _diagnosis() -> str:
    """Say which file was read and what it actually defined.

    "Required" on its own sends someone to check a key they have already set.
    The three failures that produce this message -- no file, the wrong file,
    a file missing these two names -- need different fixes and look identical
    from the outside, so each is named. Key NAMES only; no value is read into
    the message.
    """
    lines = ["TELEGRAM_API_ID and TELEGRAM_API_HASH are required.", ""]
    found_any = False
    for candidate in ENV_CANDIDATES:
        if not candidate.exists():
            lines.append(f"  {candidate}  (no such file)")
            continue
        found_any = True
        names = sorted(parse_env_file(candidate))
        if not names:
            lines.append(f"  {candidate}  (readable, defines nothing)")
        else:
            lines.append(f"  {candidate}  defines: {', '.join(names)}")
    lines.append("")
    if found_any:
        lines.append(
            "The file was found and read; it does not define those two names. "
            "Add them to it -- the same file the shadow unit loads through "
            "EnvironmentFile= -- then run this again.")
    else:
        lines.append(
            "No environment file was found at all. The shadow unit reads one "
            "through EnvironmentFile=, which an interactive shell does not, so "
            "a variable exported in your shell is invisible to the service and "
            "a variable in that file is invisible here until it exists.")
    return "\n".join(lines)


async def authorize() -> None:
    api_id, api_hash, origin = load_credentials()
    if not api_id or not api_hash:
        raise SystemExit(_diagnosis())
    print(f"Using credentials from {origin}; no value is displayed.")
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(str(SESSION_PATH), int(api_id), api_hash,
                            receive_updates=False)
    # This is the interactive part: Telegram asks for the phone number and the
    # code it sends to the account's own device.
    await client.start()
    me = await client.get_me()
    await client.disconnect()
    session_file = SESSION_PATH.with_suffix(".session")
    if session_file.exists():
        session_file.chmod(0o600)
    print(f"Telegram session authorized for account id {me.id}; "
          "no credential was displayed")
    print(f"Session written to {session_file.resolve()}")
    print("Now restart the desk: systemctl --user restart memecoin-shadow.service")


if __name__ == "__main__":
    asyncio.run(authorize())
