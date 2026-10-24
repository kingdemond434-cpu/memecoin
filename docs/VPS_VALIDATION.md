# VPS validation without production state

Run tests in a fresh checkout, never inside the running bot's checkout. Some
integration tests construct a desk with default relative data paths. Running
them alongside the paper service can read or append its evidence even though
transaction submission is disabled.

Use the stage virtual environment and an exact release already fetched into
the repository:

```powershell
& C:\stage\.venv\Scripts\python.exe -m tools.validate_checkout `
  --revision <40-character-release-SHA> --output C:\stage\validation-new-run
```

The output must be new and outside the production checkout. The launcher
clones committed files and history, configures the repository hooks, pins the
release and runs pytest with a minimal environment. It removes credentials and
clears trading and signing settings. It retains the isolated checkout, test
log, JUnit report and summary. A passing software suite does not count as an
earned forward experiment or profitable-trading evidence.

On 24 October, an initial full VPS run from the service checkout appended two
`research_consumers_configured` test records from stale ledger instances. The
paper service was stopped. Both conflicting records were unreferenced by the
frontier and framework. The original ledger and checkpoint were retained in
`evidence-audit-20261024-003119` at the deployment root. The reviewed correction
replaced those two 900-byte slots with whitespace, preserving every production
record, hash and byte offset. All 350,250 production records passed hash-chain
verification. The recovery manifest retains excluded hashes, offsets and both
file seals. Subsequent full runs use a separate checkout.
