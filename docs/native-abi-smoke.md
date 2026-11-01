# Offline native ABI smoke surface

`check_native_abi_smoke` accepts an already trusted loaded native module. The caller
must establish location and artifact trust before loading it. Five fixed vectors
exercise base58 encoding, base58 decoding, invalid-input ValueError propagation,
Anchor discriminator bytes and a neutral hazard prediction across the Python/native
boundary. No receiver is instantiated, no accounts are opened and no transaction
is built/submitted. Errors are suppressed to a fixed diagnostic.

Success is a small ABI smoke test only. It does not certify the entire ABI,
dependency binaries, execution curves, venue semantics, cost provenance, inference
models or profitability. It cannot authorize deployment. The smoke checker itself
does not impose a timeout on a native call: run it only in a bounded direct child
with the existing resource and environment checks. The actual deployment
orchestrator still must compose trusted digest provenance, tested-tree gates,
coherent backup recovery and fresh service receipts.
