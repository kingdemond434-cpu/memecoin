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
    "process-isolated miners": ("ProcessOffloadedPool(", "src/runtime/wiring.py"),
    "binary signer transport": ("self._frame(", "src/execution/signer.py"),
    # Constructed AND started AND fed. Requiring only construction is what
    # let this one ship reporting OFF with no reason: it existed, appeared in
    # /status, and nothing ever called start().
    "native chain ingress": ("NativeIngress(", "src/runtime/wiring.py"),
    "native ingress started": ("ingress.start()", "src/runtime/wiring.py"),
    "native ingress parity fed": ("note_python_event(", "src/main.py"),
    # Constructed, started AND consumed. The first two were true while the
    # sink filled and evicted its own events, because nothing drained it --
    # the orphan one level in from the orphan this file was written to catch.
    "native ingress drained": ("ingress.drain(", "src/runtime/ingestion.py"),
    "native ingress drain scheduled": ("self._native_ingress_loop()", "src/main.py"),
    "native ingress stopped": ("self.native_ingress.stop()", "src/main.py"),
    # The fee tiers are published only as an image, so the engine refuses to
    # transcribe them and blocks costing until the on-chain account is read.
    # `adopt_chain_config` had zero callers, so the block was permanent and
    # nothing could ever be priced net of cost.
    "pump fee config read from chain": ("adopt_chain_config(",
                                        "src/runtime/maintenance.py"),
    "pump fee config refreshed": ("self._refresh_pump_fee_config()", "src/main.py"),
    # T0 used to await three to five sequential RPC round trips before it
    # could decide. The local view has to be the one the decision reads, and
    # the full audit has to be scheduled beside it -- both, or neither works.
    "t0 local risk view used": ("self.t0_risk.assess(", "src/runtime/ingestion.py"),
    "t0 risk enrichment scheduled": ("self._schedule_risk_enrichment(",
                                     "src/runtime/ingestion.py"),
    "t0 risk view is what the decision reads": ("self._risk_for_decision(",
                                                "src/main.py"),
    "launch invariants learned from full reports": (
        "self.invariant_ledger.observe_report(", "src/runtime/ingestion.py"),
    "launch invariants persisted": ("self.invariant_ledger.save()", "src/main.py"),
    "portfolio refresh is off the decision path": ("self._ensure_portfolio_fresh()",
                                                   "src/main.py"),
    # Redundancy conditional on total failure is redundancy that never runs.
    "secondary feed races the primary": ("self._start_secondary_feed()",
                                         "src/runtime/wiring.py"),
    "secondary feed stopped": ("self.secondary_stream.stop()", "src/main.py"),
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
