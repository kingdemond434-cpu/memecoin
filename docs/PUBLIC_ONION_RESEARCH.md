# Public onion research

The source mesh supports public onion collection and continuous discovery.
This is separate from public Telegram previews. Every onion request uses a dedicated Tor SOCKS5 connector with
remote hostname resolution; the ordinary HTTP client rejects onion requests
and redirects. The connector never falls back to a direct TCP request.

The scope follows the original unknown-opportunity blueprint: public research,
public actor claims and public threat/scam information. There is no login,
access-challenge bypass, marketplace transaction, credential collection or
private-group intrusion capability. A forum post claiming to be an insider
remains a claim. An address mention is neither a verified wallet owner nor a
verified token mint.

## What is collected

The continuous `onion:continuous-frontier` transport persists discovered hosts,
pages and parent provenance in SQLite. It has no source-count, page-count or
crawl-depth admission ceiling. It follows valid v3 onion links, revisits pages
for content/link changes and retains inaccessible work with retry deadlines.
One paced worker services the queue; a growing inventory does not establish
that all pages have been visited or that the queue can keep up with arrivals.
`all_onion_coverage` remains false.

The separately configured `onion` transport is a bounded single-source crawler.
Its `max_pages` (up to 256), `max_depth` (up to eight) and `pages_per_poll` (up to
two) options still apply to that adapter. They do not limit the continuous
frontier's inventory. Dread can use the public-page transport, but a discovered
address does not establish that a service is the genuine forum.

Set `global.onion_auto_discovery: true` and append
`config/sources.onion-discovery.yaml` to the source registry to discover public
links from the selected announcement page. Links found in other collected
public messages, Telegram content and onion pages enter the same discovery
path. Their original canonical research work item is retained as provenance.
Those discoveries join the same disk-backed frontier rather than creating a
new polling task for each host. Historical admitted/pending source state,
unacknowledged snapshots and legacy capacity debt remain readable for migration;
the old `onion_max_sources`/512-candidate admission limits do not govern the
continuous frontier. New hosts and old pages share its paced service. No source
gains authenticity from its domain spelling or discovery origin. Current Dread
identity still needs independent verification.

Records retain their source URL, provenance, content hash, original observation
time, launch/wallet/claimed-insider/claimed-coordination topics and address
mentions. Script/style content is discarded. The HTTP document envelope is
512 KiB, and oversized or unsupported responses fail explicitly. The continuous
frontier retains the extracted text and links within that envelope without the
bounded adapter's 16,000-character/128-link truncation. The bounded adapter
retains its text-truncation flag. Document and request envelopes are resource
controls, not proof of unlimited content coverage.
Publication time is unknown unless separately established; an initial snapshot
does not establish when its claims were first published.

Both EVM and validated 32-byte Solana address mentions are retained in the
canonical claim metadata. Solana mentions are not automatically assigned token
roles: a public key can identify a wallet, mint, pool or program. Topic matching
includes English, Chinese, Russian and Spanish phrases; these are labels over
unverified text, not verified identities or independent signals.

The adapter emits `SourceClass.WEB`, not `OFFICIAL`. The ordinary source-mesh
consumer writes the original claim provenance to the canonical narrative work
item. It does not grant actor independence, authenticity, a profitable strategy
or execution authority.

The continuous page frontier, hashes and delivery outbox use SQLite WAL and
FULL synchronous transactions. The bounded adapter uses its own atomic JSON
checkpoint with flush/fsync. A pending snapshot is replayed with its original ID and timestamp
after restart until the existing research engine durably accepts it. Intake is
idempotent; a checkpoint acknowledgement failure can cause redelivery, not
manufacture a later observation. The transport does not claim exactly-once
market execution.

HTTP failures, unsupported documents, access challenges, missing proxy and
invalid checkpoints are explicit failures. A name in a configuration file does
not count as an observed source. `/status` includes `wallet_follow.research.onion_research`
and per-transport coverage, candidate links and pending delivery counts.

## Verify the deployed collector

`config/research-production.example.yaml` keeps automatic discovery disabled as
an explicit opt-in default. It does not describe a running deployment's actual
configuration. Check that deployment's `global.onion_auto_discovery`, configured
proxy and attached frontier; a Tor process alone is insufficient evidence.

Inspect `wallet_follow.research.onion_research` in `/status`, especially
`automatic_discovery_enabled` and `continuous_frontier.observed_pages`,
`last_observed_at`, `pending_pages`, `deferred_pages` and
`pending_canonical_delivery`. Distinguish recent successful reads, old successful
reads, discovered-but-unvisited URLs and pending research delivery. Unchanged
pages may advance the successful-read clock without producing another content
change. Refusals, inaccessible hosts and access challenges remain deferred;
they are not silently counted as successful coverage.

For delivery evidence, resolve an `origins.source_ref` of `frontier_job:<id>`
from the frontier database to its actual `NARRATIVE_OBSERVATION` job in the
research frontier. Verify the referenced canonical row hash and the parent's
URL/observation identity internally. The corresponding neural input can be
checked by its canonical source reference. Do not print raw documents, addresses
or proxy/credential values merely to prove counters. Deduplicate sampled job
references: multiple discovered child links can share one parent observation.

Successful collection and canonical delivery establish a working research
path. They do not establish authenticated insiders, complete coverage, NLP
completion, a predictive advantage or profitable trading.

## Configuration

Install the normal requirements, including `aiohttp-socks`. Run a Tor client
bound to localhost on a native deployment, then set
`MEMECOIN_TOR_PROXY_URL=socks5://127.0.0.1:9050` in that deployment's environment.
This value is a proxy endpoint, not an API credential. Tor client state and
executables should live outside the source checkout.

The Windows paper launcher accepts optional `-TorExe`, `-TorConfig`,
`-TorExeSha256` and `-TorProxyUrl` arguments. Verify the official release
signature and executable hash before supplying them. The launcher checks that
hash, starts a hidden Tor child bound to a localhost proxy, and owns it in the
same Windows process job as the paper desk. Normal exit and task termination
clean up that child. It does not change the scheduled task's account/logon
mode, install an unattended task, or unlock trading.

For Docker, use the explicit overlay:

```sh
docker compose -f docker-compose.yml -f docker-compose.tor.yml up -d --build
```

The overlay adds a client-only Tor service, a bounded temporary data directory,
and the desk's proxy endpoint. It exposes no SOCKS port on the host and retains
the desk's paper-only command and live lock. Service start does not establish
Tor bootstrap or successful source observations.

Append a selected source registry to `global.source_registry`, keeping the
existing registries. For example:

```yaml
sources:
  - id: onion:operator-selected-forum
    kind: onion
    poll_interval_seconds: 120
    poll_timeout_seconds: 45
    options:
      url: <verified public v3 onion URL>
      source_ref: <verification/provenance reference>
      max_pages: 128
      max_depth: 6
      pages_per_poll: 1
```

`config/sources.onion.example.yaml` tracks Dread as `UNAVAILABLE` until an
endpoint is actually verified. No historical address, lookalike, or mirror is
silently substituted. The code does not claim coverage of every onion service.

## Independent connectivity check

```sh
python -m tools.probe_onion --url <verified-public-onion-url> \
  --source-ref <provenance> --output <report.json>
```

The probe exercises the same connector, crawler and normalizer. It creates a
standalone report rather than writing to a running desk's canonical ledger.
Success proves that one public page was read, not Dread coverage, predictive
value or trade profitability.

## Primary technical references

- [Tor v3 address checksum and version](https://spec.torproject.org/rend-spec/encoding-onion-addresses.html)
- [Tor SOCKS/DNS leak checks](https://support.torproject.org/little-t-tor/troubleshooting/check-for-leaks/)
- [aiohttp-socks connector and remote DNS](https://github.com/romis2012/aiohttp-socks)
- [Tor Project's published onion connectivity example](https://arti.torproject.org/guides/connecting-to-onion/)
- [Tor release signature verification](https://support.torproject.org/tor-browser/getting-started/verifying-tor-browser/)
