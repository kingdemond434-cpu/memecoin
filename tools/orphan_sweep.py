"""Public methods that nothing in production ever calls.

Three separate bugs on 2026-09-19 were the same shape: a method written,
tested, and never wired.

  * four `LaunchCensus` funnel transitions, dropped by the commit that split
    `main.py`, so hard safety rejects were filed as ordinary screens;
  * `PromotionLedger.demote`, the only control that can lower the desk's
    trading authority, which meant a catastrophic loss at LIVE changed
    nothing;
  * `PriorityFeeOptimizer.record_attempt`, which left `landing_rates` empty
    forever and made the fee "optimiser" a lookup table of magic numbers.

The suite was green through all three, because tests covered the METHODS and
nothing covered the WIRING. This sweeps for the shape.

    python -m tools.orphan_sweep            # report
    python -m tools.orphan_sweep --baseline # rewrite the accepted set

It understands `getattr`-style dispatch (a quoted method name counts as a
call) and skips overrides of framework base classes, whose methods are
called by the framework rather than by us. Both were false positives in the
first version of this sweep.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

#: Base classes whose methods the framework calls. An override of one is not
#: an orphan however little our own code mentions it.
FRAMEWORK_BASES = {"HTMLParser", "Protocol", "ABC", "Thread", "Enum",
                   "BaseHTTPRequestHandler", "Exception"}

PRODUCTION_ROOTS = ("src", "tools")
BASELINE = Path("tests/orphan_baseline.txt")


def _definitions(roots: Sequence[str]) -> Tuple[Dict[str, List[str]], Set[str]]:
    defined: Dict[str, List[str]] = defaultdict(list)
    framework: Set[str] = set()
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = ({base.id for base in node.bases
                          if isinstance(base, ast.Name)}
                         | {base.attr for base in node.bases
                            if isinstance(base, ast.Attribute)})
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef,
                                             ast.AsyncFunctionDef)):
                        continue
                    if item.name.startswith("_"):
                        continue
                    defined[item.name].append(
                        f"{path}:{item.lineno} ({node.name})")
                    if bases & FRAMEWORK_BASES:
                        framework.add(item.name)
    return defined, framework


def _referenced(name: str, blob: str) -> bool:
    """An attribute access, or the name quoted for getattr/hasattr."""
    escaped = re.escape(name)
    return bool(re.search(rf"\.{escaped}\b", blob)
                or re.search(rf"[\"']{escaped}[\"']", blob))


def sweep(roots: Sequence[str] = PRODUCTION_ROOTS) -> Dict[str, List[str]]:
    """{method name: [definition sites]} for methods nothing calls."""
    defined, framework = _definitions(roots)
    blob = "\n".join(path.read_text(encoding="utf-8")
                     for root in roots for path in Path(root).rglob("*.py"))
    return {name: sites for name, sites in sorted(defined.items())
            if name not in framework and not _referenced(name, blob)}


#: What an accepted orphan is, and whether it is allowed to stay one.
#:   A  production capability that should be running and is not -- debt.
#:   B  superseded by something already wired -- a deletion waiting for a hand.
#:   C  offline research or operator API, correctly absent from the hot path.
#:   D  deliberately public and unused by design.
ORPHAN_CLASSES = ("A", "B", "C", "D")


def load_classified_baseline() -> Dict[str, Tuple[str, str]]:
    """name -> (class, reason). The class is mandatory.

    An unclassified entry is how a parking lot forms: everything lands in it,
    nothing is ever asked to leave, and the list stops meaning anything. A
    line without a class fails the guard rather than being read as accepted.
    """
    if not BASELINE.exists():
        return {}
    rows: Dict[str, Tuple[str, str]] = {}
    for number, raw in enumerate(
            BASELINE.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) < 2 or parts[1] not in ORPHAN_CLASSES:
            raise ValueError(
                f"{BASELINE.name}:{number}: every entry needs a class "
                f"({'/'.join(ORPHAN_CLASSES)}) and a reason, got {line!r}")
        rows[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return rows


def load_baseline() -> Set[str]:
    return set(load_classified_baseline())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true",
                        help="rewrite tests/orphan_baseline.txt from the "
                             "current sweep")
    args = parser.parse_args(argv)

    orphans = sweep()
    if args.baseline:
        BASELINE.write_text(
            "# Public methods with no production caller, accepted for now.\n"
            "# Every line here is a method that was written and never wired.\n"
            "# Wire one and DELETE its line -- tests/test_no_new_orphans.py\n"
            "# fails on a stale entry, so this set can only shrink.\n"
            + "".join(f"{name}\n" for name in sorted(orphans)),
            encoding="utf-8")
        print(f"baseline written: {len(orphans)} entries")
        return 0

    baseline = load_baseline()
    new = sorted(set(orphans) - baseline)
    fixed = sorted(baseline - set(orphans))
    for name in sorted(orphans):
        mark = "NEW " if name in new else "    "
        print(f"{mark}{name:42s} {orphans[name][0]}")
    print(f"\n{len(orphans)} orphans, {len(new)} new, {len(fixed)} since wired")
    return 1 if new else 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
