# VPS verification on 24 October 2026

Runtime release `e356621` has the exact tested runtime tree of `c126fea`.
The dedicated Windows paper task and its managed Tor child were restarted
after a complete state backup. Its Administrator Interactive/Limited identity
and Normal priority were preserved. The task still depends on the existing
Windows login session; an unattended elevated task was not installed.

## Runtime observations

- `/health`: healthy, dry-run true, live submission locked; 1.60 seconds in the
  retained post-deployment observation.
- `/status`: returned in 6.41 seconds. Its prior retained observation took
  17.80 seconds, with earlier requests exceeding a 20-second deadline.
- Source health in that snapshot: 38 OK, five DEAD. This does not establish
  coverage of every launch, chain or public site.
- A standalone real-Tor VPS probe read Tor Project's published onion page and
  produced one normalized observation. This certifies the network path only.
- Actual configured onion research still had zero successfully observed
  sources and zero automatically admitted hosts. Dread identity and collection
  remain unverified. An enabled discovery option is not successful coverage.

The legacy framework migration retained all observations, controls, receipt
children and visibility timestamps, and every other research component matched
exactly. The framework file shrank from 80,952,584 to 5,452,870 bytes.

On a copied real queue, all 284,633 jobs were retained. The derived index had
265,450 active tag rows, with no missing or extra rows. Its cold migration took
25.41 seconds. Four tag auctions took 10–85 ms; three previous queries had
exceeded a two-second budget. Compact report indexes were built in 4.19 seconds
and a complete consistent queue report took 1.51 seconds. These are isolated
database measurements, not trading-throughput or landing benchmarks.

## Software verification

Native Linux CI for `c126fea`: 3,494 passed, nine skipped, 14 subtests passed.
Rust tests, clippy, native parity, hotpath regression, vendored IDL and dry-run
safety checks passed. The merged runtime release also passed its main CI.

VPS checkpoints included a complete isolated suite on the earlier persistence
release (3,388 passed, 106 skipped, 14 subtests passed), 87 focused checks on
`550fd90`, and 65 replay/report/readiness checks on `c126fea`. An architecture
checkpoint retained 202 passing receipts and all 15 acceptance scenarios.

The first complete VPS run on `c126fea` passed 3,396 tests but failed one
source-cadence test: a busy Windows host delivered only two polls during its
fixed 100 ms window. The regression now waits for the third observed poll,
with guaranteed cleanup and a deadline shorter than the slow source's own
cadence. It still checks that the slow source is not repeatedly polled. It
does not relax or alter production timing. Final rerun counts and source seals
are retained in the isolated validation reports, rather than inferred from
the earlier run.

## Actual qualification blockers

The post-mapping corpus snapshot contained 990 episodes. None had complete
native route/depth/fill/cost evidence; 317 outcomes were unresolved and 22
episodes used protocols outside the existing replay adapter. Recognizing the
968 supported Pump identities does not supply the missing measurements.

The runtime reported zero historical validation receipts, zero qualifying
forward experiments, zero forward cohorts, zero paper experiment outcomes and
zero real-fill outcomes. Nine integration receipts describe observation
capability. The promotion stage remained historical, with unmeasured net log
growth, enrichment and rug-loss statistics. Calendar collection time is not an
earned qualifying shadow experiment.

Broader review work and unproven capabilities remain listed in
`REVIEW_REQUIREMENTS_20261024.md`, including dependent/time-to-target portfolio
comparisons, executable missed-winner reconstruction, realized marginal source
value and separately validated prediction-market/exchange execution.

## Retained evidence and test isolation

The final deployment backup verified 356,511 canonical records and frontier
integrity before restart. Subsequent observations continue to append; that
verification is a dated snapshot, not a claim that every later byte was audited.

An initial full run from the service checkout appended two unreferenced test
configuration records through stale ledger instances. The paper service was
stopped and the original ledger and checkpoint retained. A reviewed correction
preserved every production record, hash and byte offset and retained a recovery
manifest with both file seals. See `../VPS_VALIDATION.md` for the incident and
the isolated launcher; subsequent test runs copy no production state and remove
credential environment settings.

The VPS deployment root retains the full/focused JUnit reports, source-sealed
architecture manifests, acceptance evidence, Tor probe, queue/migration
receipts, corpus/promotion audits and recovery manifest. These reports grant no
signing, promotion or capital authority.
