"""Does the systemd on this box match the systemd in this repository?

On 2026-09-03 the running `memecoin-shadow-trainer` reported
`MemoryMax=infinity` while the unit file in this repository said `640M`. The
unit had been edited here and never reinstalled there, so the box had been
running an unbounded trainer for an unknown period and collected four OOM
kills. Nothing reported it, because every tool that looks at units either
reads the repository (and sees the cap) or reads the box (and has nothing to
compare against).

This compares them, for the properties where a difference is dangerous rather
than merely untidy:

    MemoryMax / MemoryHigh   an absent cap is not a smaller cap, it is no cap
    OOMScoreAdjust           decides WHICH process the kernel takes
    CPUQuota                 a trainer at 400% starves the collector
    ExecStart                the box running a command this repo no longer has
    SuccessExitStatus        a skip showing up as a failed unit, or worse, a
                             failure showing up as a skip

`show` is injected, so this is testable without systemd and can be pointed at a
recorded snapshot. It reads only; installing is a human's decision, and the
output ends with the exact command.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Properties compared, and whether a MISSING value on the box is dangerous.
#: `MemoryMax=infinity` is systemd's way of saying "unset", which is exactly
#: the state that produced the incident, so absence is never treated as equal.
WATCHED: Tuple[str, ...] = (
    "MemoryMax", "MemoryHigh", "OOMScoreAdjust", "CPUQuota",
    "ExecStart", "SuccessExitStatus", "Nice",
)

#: Fetched but never compared. A unit with a drop-in is a unit somebody
#: deliberately overrode on this box, and the difference that creates is a
#: DECISION rather than drift -- the two must not read the same, or the tool
#: reports a considered local override as an emergency on every run and stops
#: being read at all.
DROP_IN_PROPERTY = "DropInPaths"

#: What systemd reports for "no limit". Compared as a value, not as absence,
#: because the whole point is that infinity looks like a setting.
_UNSET = {"infinity", "[not set]", "", "0"}

#: A box cap below this fraction of the repository's is CRITICAL rather than a
#: difference. Ten per cent of slack is rounding and unit-suffix arithmetic;
#: half is a process that will be killed.
TIGHTER_CAP_RATIO = 0.9

#: Properties systemd does not report under the name the unit file uses.
#: `CPUQuota=` is read back as `CPUQuotaPerSecUSec`, so asking for `CPUQuota`
#: returns empty for every unit and reports drift on all of them -- noise that
#: buries the one line that matters. Compared through the reported name, or
#: not at all.
_REPORTED_AS = {"CPUQuota": "CPUQuotaPerSecUSec"}

_SUFFIX = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}


def parse_bytes(value: str) -> Optional[int]:
    """`1200M` and `1258291200` are the same limit; `infinity` is not a limit."""
    text = (value or "").strip().lower()
    if not text or text in _UNSET:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)b?", text)
    if not match:
        return None
    return int(float(match.group(1)) * _SUFFIX.get(match.group(2), 1))


def parse_unit_file(path: Path) -> Dict[str, List[str]]:
    """The [Service] directives, with continuations joined and comments cut."""
    directives: Dict[str, List[str]] = {}
    section = ""
    buffer = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "Service":
            continue
        if line.endswith("\\"):
            buffer += line[:-1].strip() + " "
            continue
        line = (buffer + line).strip()
        buffer = ""
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        directives.setdefault(key.strip(), []).append(value.strip())
    return directives


def _exec_target(command: str) -> str:
    """What a command actually runs: the module after -m, or the script.

    The interpreter path is shared by every unit here and therefore carries no
    information about which of them is installed.
    """
    tokens = (command or "").split()
    for index, token in enumerate(tokens):
        if token == "-m" and index + 1 < len(tokens):
            return tokens[index + 1]
    for token in tokens:
        if token.endswith(".py") or token.endswith(".sh"):
            return token.split("/")[-1]
    return tokens[0].split("/")[-1] if tokens else ""


@dataclass
class Drift:
    unit: str
    prop: str
    repo: str
    box: str
    severity: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"unit": self.unit, "property": self.prop, "repo": self.repo,
                "box": self.box, "severity": self.severity,
                "detail": self.detail}


def compare(unit: str, repo: Dict[str, List[str]], box: Dict[str, str]
            ) -> List[Drift]:
    drifts: List[Drift] = []
    drop_ins = [path for path in (box.get(DROP_IN_PROPERTY) or "").split()
                if path.strip()]
    for prop in WATCHED:
        wanted_values = repo.get(prop) or []
        wanted = wanted_values[-1] if wanted_values else ""
        reported = _REPORTED_AS.get(prop, prop)
        actual = (box.get(prop) or box.get(reported) or "").strip()
        if not wanted and not actual:
            continue
        if prop in _REPORTED_AS and reported not in box:
            # Asked for under a name this systemd does not answer to. Silence
            # is not agreement and it is not drift; it is an unanswered
            # question, and reporting it as a difference on every unit is how
            # a real finding gets lost in a wall of noise.
            continue
        if not wanted and actual in _UNSET:
            # The repository does not set it and the box reports its default.
            # Nothing has drifted.
            continue
        if prop in ("MemoryMax", "MemoryHigh"):
            want_bytes = parse_bytes(wanted)
            have_bytes = parse_bytes(actual)
            if want_bytes == have_bytes:
                continue
            if want_bytes is not None and have_bytes is None:
                drifts.append(Drift(
                    unit, prop, wanted, actual or "infinity", "CRITICAL",
                    "the repository sets a cap and the box has none; an "
                    "unbounded process on a 4 GB host is what the kernel OOM "
                    "killer resolves, and it does not have to pick this one"))
                continue
            if (want_bytes is not None and have_bytes is not None
                    and have_bytes < want_bytes * TIGHTER_CAP_RATIO
                    and drop_ins):
                # Tighter, and a drop-in explains why. That is somebody's
                # decision on this box -- often a better-measured one than the
                # repository's, since the repository cannot see this host's
                # real working set. Reported so it is visible, never as an
                # alarm.
                drifts.append(Drift(
                    unit, prop, wanted, actual, "OVERRIDE",
                    f"a drop-in sets this: {', '.join(drop_ins)}. The box "
                    "enforces "
                    f"{have_bytes / want_bytes:.0%} of the repository's value "
                    "deliberately; change the drop-in, not the unit file"))
                continue
            if (want_bytes is not None and have_bytes is not None
                    and have_bytes < want_bytes * TIGHTER_CAP_RATIO):
                # The direction that kills a process rather than merely
                # differing from a file. Seen 2026-09-03: the desk's unit
                # asks for 2560M after measurement, and the box was enforcing
                # 1341M -- about half. A cgroup cap below what a process needs
                # does not slow it down, it terminates it, and reporting that
                # as "cap differs" alongside a Nice value is how a fatal
                # setting reads as tidy-up.
                drifts.append(Drift(
                    unit, prop, wanted, actual, "CRITICAL",
                    f"the box enforces {have_bytes / want_bytes:.0%} of what "
                    "the repository asks for; a cap below what the process "
                    "needs terminates it rather than slowing it, and a "
                    "value the unit file does not contain means a drop-in "
                    "or slice is overriding it -- check "
                    "`systemctl --user cat` for the unit"))
                continue
            drifts.append(Drift(unit, prop, wanted, actual, "WARN",
                                "cap differs"))
            continue
        if prop == "OOMScoreAdjust":
            # `0` is systemd's default, so the box reporting "0" means the
            # priority was never applied. Comparing truthiness would read that
            # string as a setting and pass -- which is the same class of
            # mistake as reading `infinity` as a cap.
            if wanted and actual in _UNSET:
                drifts.append(Drift(
                    unit, prop, wanted, actual or "0", "CRITICAL",
                    "the repository decides which process the kernel takes "
                    "under pressure; the box has not been told"))
                continue
            if wanted and actual and wanted != actual:
                drifts.append(Drift(unit, prop, wanted, actual, "WARN",
                                    "priority differs"))
                continue
            continue
        if prop == "ExecStart":
            # systemd renders ExecStart as a struct, so this compares the part
            # that identifies WHAT runs rather than the whole rendering.
            #
            # Not the interpreter's basename: every one of these units invokes
            # `python`, so `python in actual` is true whichever module the box
            # is actually running, and the check would never fire on the one
            # difference it exists to catch.
            target = _exec_target(wanted)
            if target and target not in actual:
                drifts.append(Drift(
                    unit, prop, wanted[:120], actual[:120], "CRITICAL",
                    f"the box is not running {target}"))
            continue
        if wanted != actual:
            drifts.append(Drift(unit, prop, wanted, actual, "WARN",
                                "value differs"))
    return drifts


def systemctl_show(unit: str) -> Dict[str, str]:  # pragma: no cover - needs systemd
    result = subprocess.run(
        ["systemctl", "--user", "show", unit,
         "--property=" + ",".join(WATCHED + (DROP_IN_PROPERTY,))],
        capture_output=True, text=True, check=False, timeout=30)
    values: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            values[key.strip()] = value.strip()
    return values


def audit(unit_dir: Path,
          show: Callable[[str], Dict[str, str]] = systemctl_show,
          units: Sequence[str] = ()) -> Dict[str, Any]:
    directory = Path(unit_dir)
    paths = ([directory / name for name in units] if units
             else sorted(directory.glob("*.service")))
    drifts: List[Drift] = []
    checked: List[str] = []
    unreadable: List[Dict[str, str]] = []
    for path in paths:
        if not path.exists():
            unreadable.append({"unit": path.name, "reason": "not in the repo"})
            continue
        try:
            repo = parse_unit_file(path)
        except OSError as exc:
            unreadable.append({"unit": path.name, "reason": str(exc)})
            continue
        try:
            box = show(path.name)
        except Exception as exc:
            unreadable.append({"unit": path.name,
                               "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not box:
            unreadable.append({"unit": path.name,
                               "reason": "not installed on this box"})
            continue
        checked.append(path.name)
        drifts.extend(compare(path.name, repo, box))
    critical = [item for item in drifts if item.severity == "CRITICAL"]
    overrides = [item for item in drifts if item.severity == "OVERRIDE"]
    genuine = [item for item in drifts if item.severity != "OVERRIDE"]
    return {
        "checked": checked,
        "unreadable": unreadable,
        "drift": [item.to_dict() for item in genuine],
        "overrides": [item.to_dict() for item in overrides],
        "critical": [item.to_dict() for item in critical],
        "status": ("CRITICAL" if critical else
                   "WARN" if genuine else
                   "DATA_BLOCKED" if not checked else "OK"),
        "remedy": ("cp deploy/systemd/*.service ~/.config/systemd/user/ && "
                   "systemctl --user daemon-reload"),
    }


def main(argv: Any = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir", default="deploy/systemd")
    parser.add_argument("--unit", action="append", default=[])
    args = parser.parse_args(argv)
    report = audit(Path(args.unit_dir), units=args.unit)
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "CRITICAL" else 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
