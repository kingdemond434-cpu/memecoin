"""Catch a corrupted credential before a data source silently 401s forever.

GITHUB_TOKEN held a Windows PowerShell SSH command on 2026-08-29 -- 226
characters, spaces, quotes, an embedded `$env:` variable -- not a token by
any definition, and every GitHub call failed with "Bad credentials" from the
moment it was pasted in until a human happened to read the raw value. The
desk's own status endpoint could not have caught this: it reports credential
PRESENCE, deliberately never values, so a present-but-garbage value and a
present-and-valid one look identical to everything that only asks "is it
set".

This is the one check in the fleet that has to read actual credential
values rather than their names, so it runs as its own narrow, standalone
tool instead of folding into the general status-based watchdog: reading
secrets is a wider responsibility than polling a JSON endpoint, and keeping
it in one small, auditable place is better than spreading value-reading
across the codebase for one rare failure mode.

What it checks, per credential, is SHAPE, not authenticity: plausible
length, no embedded whitespace or shell metacharacters, and a matching
prefix where the provider publishes one. It cannot tell a well-formed but
revoked key from a good one -- that only shows up as a runtime 401, which
the existing per-miner status already surfaces. It exists for the
categorically different failure where the value was never a credential at
all.

Names, never values: every message this tool can produce says which
credential looks wrong and why, and never repeats the value itself, in logs,
alerts, or its JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("tools.credential_doctor")

CREDENTIAL_DOCTOR_SCHEMA_VERSION = "v1"

#: A value containing any of these almost certainly is not a credential: no
#: provider issues a key with embedded whitespace, quotes, or shell
#: metacharacters, and every one of these appears in the exact corrupted
#: GITHUB_TOKEN value that prompted this tool.
SUSPICIOUS_SUBSTRINGS: Sequence[str] = (
    " ", "\t", "\n", '"', "'", "$", "`", ";", "&&", "||", "\\",
)


@dataclass
class Verdict:
    name: str
    ok: bool
    reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "ok": self.ok, "reason": self.reason}


def _looks_like_a_command(value: str) -> Optional[str]:
    """The specific shape of the incident this tool exists for."""
    lowered = value.lower()
    for marker in ("ssh ", "powershell", "$env:", "systemctl", "cmd.exe /c"):
        if marker in lowered:
            return f"contains {marker.strip()!r}, which reads as a command, not a credential"
    return None


def _shape_check(name: str, value: str, *, min_len: int, max_len: int,
                 prefix: str = "", pattern: str = "") -> Verdict:
    if not value:
        return Verdict(name, True, "unset")
    command_hit = _looks_like_a_command(value)
    if command_hit:
        return Verdict(name, False, command_hit)
    hit = next((s for s in SUSPICIOUS_SUBSTRINGS if s in value), None)
    if hit:
        return Verdict(name, False,
                       f"contains {hit!r}, which no provider issues inside a credential")
    if not (min_len <= len(value) <= max_len):
        return Verdict(name, False,
                       f"length {len(value)} is outside the plausible {min_len}-{max_len} "
                       f"range for {name}")
    if prefix and not value.startswith(prefix):
        return Verdict(name, False, f"does not start with the expected {prefix!r} prefix")
    if pattern and not re.fullmatch(pattern, value):
        return Verdict(name, False, f"does not match {name}'s expected shape")
    return Verdict(name, True)


#: One shape rule per credential this desk declares. Bounds are generous on
#: purpose -- providers rotate formats -- and exist to catch a value that is
#: structurally impossible, not to enforce a provider's exact scheme.
#: `_shape_check` already returns ok=True/"unset" for an empty value, so
#: every entry below is written the same way regardless of whether that
#: credential happens to be configured right now.
CHECKS: Dict[str, Callable[[str], Verdict]] = {
    "HELIUS_API_KEY": lambda v: _shape_check("HELIUS_API_KEY", v, min_len=20, max_len=60),
    "ALCHEMY_KEY": lambda v: _shape_check("ALCHEMY_KEY", v, min_len=20, max_len=60),
    "GITHUB_TOKEN": lambda v: _shape_check(
        "GITHUB_TOKEN", v, min_len=20, max_len=255,
        pattern=r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "JUPITER_API_KEY": lambda v: _shape_check(
        "JUPITER_API_KEY", v, min_len=20, max_len=80, prefix="jup_"),
    "YOUTUBE_API_KEY": lambda v: _shape_check(
        "YOUTUBE_API_KEY", v, min_len=30, max_len=50, prefix="AIza"),
    "YELLOWSTONE_GRPC_TOKEN": lambda v: _shape_check(
        "YELLOWSTONE_GRPC_TOKEN", v, min_len=10, max_len=200),
    "YELLOWSTONE_GRPC_URL": lambda v: _shape_check(
        "YELLOWSTONE_GRPC_URL", v, min_len=10, max_len=300, pattern=r"(https?|grpc)://.+"),
    "TELEGRAM_API_ID": lambda v: _shape_check(
        "TELEGRAM_API_ID", v, min_len=5, max_len=12, pattern=r"\d+"),
    "TELEGRAM_API_HASH": lambda v: _shape_check(
        "TELEGRAM_API_HASH", v, min_len=32, max_len=32, pattern=r"[a-f0-9]{32}"),
    "X_BEARER_TOKEN": lambda v: _shape_check("X_BEARER_TOKEN", v, min_len=20, max_len=200),
    "REDDIT_CLIENT_ID": lambda v: _shape_check("REDDIT_CLIENT_ID", v, min_len=10, max_len=40),
    "REDDIT_CLIENT_SECRET": lambda v: _shape_check(
        "REDDIT_CLIENT_SECRET", v, min_len=10, max_len=60),
}


def parse_env_file(path: Path) -> Dict[str, str]:
    """A tolerant KEY=VALUE reader, matching what systemd's EnvironmentFile
    accepts closely enough for this purpose: quotes stripped, blank and
    comment lines skipped."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        name = name.strip()
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def check_all(env: Dict[str, str]) -> List[Verdict]:
    return [check(env.get(name, "")) for name, check in CHECKS.items()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="check credential SHAPE (never authenticity) for obvious corruption")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--alert", action="store_true",
                        help="send a Telegram alert (via ops.watchdog's channel) "
                             "when a credential looks corrupted")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    env_path = Path(args.env_file) if args.env_file else (
        Path.home() / ".config/memecoin-shadow/env")
    env = parse_env_file(env_path)
    verdicts = check_all(env)
    broken = [v for v in verdicts if not v.ok]

    report = {
        "schema": CREDENTIAL_DOCTOR_SCHEMA_VERSION, "checked_at": time.time(),
        "env_file": str(env_path), "checked": len(verdicts),
        "broken": [v.as_dict() for v in broken],
    }

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        for v in broken:
            print(f"[CORRUPTED] {v.name}: {v.reason}")
        print(f"{len(verdicts) - len(broken)}/{len(verdicts)} credentials look shape-valid")

    if broken and args.alert:
        # Reuse the desk's one alert channel rather than inventing a second.
        # Names only, exactly as this module's docstring promises.
        from ops.watchdog import _send_telegram_alert
        names = ", ".join(v.name for v in broken)
        _send_telegram_alert(
            f"memecoin credential doctor: corrupted-looking value(s): {names} "
            "-- check the env file, never logged here")

    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
