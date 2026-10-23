# Public onion research

The source mesh supports a real `onion` transport. This is separate from public
Telegram previews. Every request uses a dedicated Tor SOCKS5 connector with
remote hostname resolution; the ordinary HTTP client rejects onion requests
and redirects. The connector never falls back to a direct TCP request.

The scope follows the original unknown-opportunity blueprint: public research,
public actor claims and public threat/scam information. There is no login,
access-challenge bypass, marketplace transaction, credential collection or
private-group intrusion capability. A forum post claiming to be an insider
remains a claim. An address mention is neither a verified wallet owner nor a
verified token mint.

## What is collected

Declared forum pages and same-host discussion links are polled at a bounded
cadence. A source can explore up to 256 pages and eight link levels, with at
most two requests per poll. Other onion links are retained as source candidates
without automatically authorizing an unrelated host. Dread can use this same
transport when its current public endpoint and provenance are verified.

Records retain their source URL, provenance, content hash, original observation
time, launch/wallet/claimed-insider/claimed-coordination topics and address
mentions. Script/style content is discarded; documents are capped at 512 KiB,
and emitted text at 16,000 characters with an explicit truncation flag.
Publication time is unknown unless separately established; an initial snapshot
does not establish when its claims were first published.

The adapter emits `SourceClass.WEB`, not `OFFICIAL`. The ordinary source-mesh
consumer writes the original claim provenance to the canonical narrative work
item. It does not grant actor independence, authenticity, a profitable strategy
or execution authority.

The page frontier, hashes and delivery outbox are atomically checkpointed with
flush/fsync. A pending snapshot is replayed with its original ID and timestamp
after restart until the existing research engine durably accepts it. Intake is
idempotent; a checkpoint acknowledgement failure can cause redelivery, not
manufacture a later observation. The transport does not claim exactly-once
market execution.

HTTP failures, unsupported documents, access challenges, missing proxy and
invalid checkpoints are explicit failures. A name in a configuration file does
not count as an observed source. `/status` includes `research.onion_research`
and per-transport coverage, candidate links and pending delivery counts.

## Configuration

Install the normal requirements, including `aiohttp-socks`. Run a Tor client
bound to localhost on a native deployment, then set
`MEMECOIN_TOR_PROXY_URL=socks5://127.0.0.1:9050` in that deployment's environment.
This value is a proxy endpoint, not an API credential. Tor client state and
executables should live outside the source checkout.

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
