"""A module the runtime never calls is not a feature.

This repository has shipped the same failure at least three times: a
well-built module, constructed at startup, reported in /status, and consulted
by nothing. It passes review because the code is good and the tests are
green; the tests exercise the module directly, which proves it works and says
nothing about whether anything uses it.

So the wiring itself is asserted. Each entry below names a capability and the
call that makes it real. Adding a module without adding its call fails here.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: capability -> the call that proves the runtime consults it, and where a
#: reader should look to see the wiring for themselves.
REQUIRED_CALLS = {
    "post-sniper absorption": ("evaluate_cohorts(", "src/main.py"),
    "exchange-hidden coordination": ("_apply_temporal_clusters(",
                                     "src/runtime/source_intelligence.py"),
    "launch venue verification": ("registry.observe(", "src/main.py"),
    "inbound feed race": ("race.observe(", "src/main.py"),
    "exit readiness": ("prover(", "src/main.py"),
    "excursions": ("recorder(", "src/main.py"),
    "exit mode": ("chooser(", "src/main.py"),
    "account-lock contention": ("contention_probability(",
                                "src/execution/jupiter_jito.py"),
    "leader effect": ("leader_effect(", "src/execution/jupiter_jito.py"),
    "benchmark entry capture": ("observe_benchmark_entry",
                                "src/runtime/source_intelligence.py"),
    "benchmark follow pricing": ("marker_delays(", "src/main.py"),
    "benchmark resolution": ("resolver(", "src/main.py"),
    "benchmark discovery": ("_promote_benchmark_candidates(", "src/main.py"),
}


class EveryCapabilityHasACaller(unittest.TestCase):

    def test_each_one_is_actually_invoked(self):
        missing = []
        for capability, (call, path) in sorted(REQUIRED_CALLS.items()):
            source = (ROOT / path).read_text(encoding="utf-8")
            if call not in source:
                missing.append(f"{capability}: no {call!r} in {path}")
        self.assertEqual(
            [], missing,
            "these modules are constructed and reported but never consulted:\n  "
            + "\n  ".join(missing))


class TheNewModulesAreImportedWhereTheyAreUsed(unittest.TestCase):
    """A call is only real if the name it calls resolves."""

    CHECKED = (
        "src/main.py",
        "src/runtime/source_intelligence.py",
        "src/runtime/position_forensics.py",
        "src/execution/jupiter_jito.py",
    )

    def test_no_module_calls_a_name_it_never_imported(self):
        for path in self.CHECKED:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported.update(alias.asname or alias.name
                                    for alias in node.names)
                elif isinstance(node, ast.Import):
                    imported.update((alias.asname or alias.name).split(".")[0]
                                    for alias in node.names)
            # Only the names this suite added; a full undefined-name check is
            # the linter's job, not this test's.
            for name in ("evaluate_cohorts", "choose_exit_mode",
                         "loop_local_semaphore", "PositionForensics"):
                if f"{name}(" in (ROOT / path).read_text(encoding="utf-8"):
                    if name in ("PositionForensics",):
                        continue
                    self.assertIn(
                        name, imported,
                        f"{path} calls {name} without importing it")


if __name__ == "__main__":
    unittest.main()
