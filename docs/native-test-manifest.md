# Scoped native parity receipts

`bind_parity_manifest` validates consistency between stable candidate fingerprints,
exact source revision/tree identifiers and a bounded private JUnit receipt containing
all six existing native/Python event-decoder parity cases. Missing, duplicate,
skipped, failed or wrong-scope cases reject. XML declarations with entities or
other `<!` constructs are rejected, and input is limited to64KiB.

The caller must obtain these inputs from an actual serialized isolated test run,
retain the raw receipt and prove the checkout and loaded artifact identity. This
binder does not independently attest execution and cannot make untrusted supplied
XML authoritative. A digest alone never demonstrates tests were run.

The accompanying operational evidence can say these specific bytes passed these
specific tests. It cannot establish reproducible build/source provenance, full ABI
or dependency coverage, live venue execution, costs, profitability, backup recovery
or deployment authorization. Those flags remain false. The receipt may supply
expected bytes to the scoped digest gate, with its limitations retained; release
validation still must verify current native source and intended test coverage.
