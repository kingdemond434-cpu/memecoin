"""A repeated key in a dict literal silently discards the earlier value.

Found the hard way: `readiness()` had "event_loop" twice, Python kept the
later one, and the field naming the running event-loop implementation had
never once reached /status -- the exact field that says whether uvloop is in
use. No test failed, because the document was still valid; it was just
missing something nobody could see it was missing.

Cheap to check for the whole tree, so it is checked for the whole tree.
"""

from __future__ import annotations

import ast
import collections
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCHED = ("src", "ops", "tools")


def _duplicate_keys(tree: ast.AST):
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        literals = [key.value for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, (str, int))]
        counts = collections.Counter(literals)
        for key, count in counts.items():
            if count > 1:
                found.append((key, getattr(node, "lineno", 0)))
    return found


class NoDictLiteralRepeatsAKey(unittest.TestCase):

    def test_every_module_is_free_of_shadowed_keys(self):
        offenders = []
        for package in SEARCHED:
            for path in sorted((ROOT / package).rglob("*.py")):
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except (OSError, SyntaxError):  # pragma: no cover
                    continue
                for key, line in _duplicate_keys(tree):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{line} repeats {key!r}")
        self.assertEqual(
            [], offenders,
            "a repeated key silently discards the earlier value:\n  "
            + "\n  ".join(offenders))

    def test_the_check_actually_catches_one(self):
        tree = ast.parse('{"a": 1, "b": 2, "a": 3}')
        self.assertEqual([("a", 1)], _duplicate_keys(tree))

    def test_and_does_not_fire_on_a_clean_literal(self):
        self.assertEqual([], _duplicate_keys(ast.parse('{"a": 1, "b": 2}')))


if __name__ == "__main__":
    unittest.main()
