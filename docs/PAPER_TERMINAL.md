# Private animated paper terminal

Run `python -m src.terminal.server` from a checked-out release with the existing
aiohttp dependency. It listens only on **127.0.0.1:18083**, reading the existing
memecoin desk at 18081 and Phantom World controller at 18082. The optional Windows
launcher accepts absolute Python, release and pre-created log-directory paths;
it imports no connection file and removes credential/signing-related environment
variables before starting Python. Keep deployment private; do not open a public
firewall port. Use an authorized SSH loopback tunnel for laptop/remote viewing.

This is a separate monitor. Starting/stopping it requires no collector restart,
database migration or change to either trading controller. It has no order,
signing, configuration-write or promotion route. Provider access limits remain
the collectors' responsibility. It makes serial health/status polls per source
every five seconds, bounded to eight seconds/eight MiB, with no redirects or
environment proxies. Its own API returns memory immediately while a source is
slow. Host checks reject non-loopback DNS-rebinding requests; no CORS is enabled.

The three tabs distinguish memecoin model capital, **unverified Phantom World**
execution data and public underlying-perps observations. Other-venue prediction
fills are never substituted for World. Catalog membership does not establish
Phantom eligibility. Model equity is not wallet balance or verified net fills.
Source cache age, upstream snapshot age, transport failure types, backlog and
unknown local-bank outcomes remain visible. Failures preserve the dated last
good snapshot; they do not create zeros or fabricated trades.

The operations-room robots are decorative CSS animation. Their speech and desk
conversation are timestamped counter changes observed from the real services,
explicitly labelled **derived service telemetry**, not invented AI deliberation.
Motion can be paused and respects reduced-motion preferences. Gauges use known
retained-work denominators, never claimed global coverage or win probabilities.
The local-bank grid shows aggregate outcome shares, including sell failures;
it is not a per-launch chronology or proof of future sellability.

Recent events (256) and observed status history (720 snapshots) are display
windows in monitor memory, reset on restart. The source evidence remains in the
existing canonical journals. There is no invented sample equity curve or persistent
fill history in the terminal. Raw view exposes the exact private monitor snapshot.
All assets are local, with no remote scripts, analytics, fonts or images.

Validate with `python -m pytest tests/test_paper_terminal.py -q` and
`node --check src/terminal/assets/terminal.js`. Tests cover gated upstream I/O,
truthful failure/stale handling, counter provenance, read-only routes, request
Host restrictions, asset allowlist, bounded/nonfinite input and redirect refusal.
