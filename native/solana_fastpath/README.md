# solana_fastpath

Hot-path primitives for the Solana launch stream.

## Layout

The pricing, decoding, construction and telemetry logic is pure Rust with no
Python in it. The pyo3 bindings sit behind the `python` feature, which is on by
default so `maturin build` works with no extra flags:

    cargo test --no-default-features     # logic tests, no libpython linked
    cargo clippy --no-default-features   # clean
    maturin build --release              # wheel, with the python feature

`extension-module` tells pyo3 not to link libpython, which is correct for a
wheel and fatal for a test binary. Gating it is what lets the logic be tested
at all.

## Parity

`cargo run --no-default-features --example parity` regenerates
`tests/fixtures/rust_curve_parity.txt`, which
`TestRustPythonQuoteParity` checks against the Python implementation. Two
implementations of one curve is two chances to be wrong, and a divergence is
the worst kind: the fast path would fill at one price while every label and
counterfactual in the research lake was computed against another. Regenerate
the fixture whenever either side's arithmetic changes, and expect the test to
fail first if the change was not intended.

## Scope

Instruction *construction* lives here. Signing and submission deliberately do
not: every safety control -- the `ALLOW_LIVE_TRADING` lock, `dry_run`, the
daily-loss kill switch -- lives on the Python path, and a process that owned
signing would route around all of them. Construction is where the CPU actually
goes; signing costs tens of microseconds and submission is network-bound, so
this split captures nearly all of the latency with the gate intact.
