# Shared VPS CPU budgets

The optional Windows governor wraps the five existing memecoin task actions.
It preserves their executable, arguments, working directory and principal in a
private manifest. Quant tasks, global environment variables, trading rules,
model weights, source admission, live authority and data are not modified.

Budgets are percentages of the entire machine, shared by the process tree of
each role: main 5%, prediction 1%, capture 1.5%, terminal 0.5%, developer 2%.
Their configured sum is 10%. Development stays disabled after installation.
Windows ENABLE|HARD_CAP job rate control and a BelowNormal priority ceiling cover the
launcher, children and nested jobs. No breakaway flags are enabled. Failure to
set or verify the kernel limit prevents launching the original action. Native
math/tokenizer thread pools are set to one in the task process environment only.

Original task XML and state are backed up before changes. Manifests and compiled
governor code permit administrator/SYSTEM writes and LOCAL SERVICE reads; only
the status folder permits service writes. Every launch verifies the assembly
hash and rejects an invalid, absent or over-budget role. Stopping the owner
terminates its descendants. A clean owner exit waits for descendants and avoids
turning a successful worker turn into a spurious failure.

First run `test-cpu-budgets.ps1` against a fresh disposable directory. Its real
Windows fixture verifies descendant job membership, queried kernel rate flags,
breakaway rejection, unchanged deterministic outputs and fail-closed guards.
It uses no bot credentials or production data. Then run
`install-cpu-budgets.ps1`; `-StartPaper` restores the four paper/read-only tasks
after installing the wrapper. Verify per-role status and native kernel jobs,
service health, live locks, state continuity and actual progress before calling
the installation accepted. Never resume an uncapped fallback after an error.

**Deployment fence:** older deployment scripts expect original task actions and
must not overwrite these wrappers or re-enable unwrapped tasks. Development
remains paused until its deployment workflow preserves the resource manifest
and governor checks. Auxiliary test/maintenance tasks must also be budgeted;
starting them through Task Scheduler does not inherit a caller's job budget.

A CPU cap is not a RAM, disk-space or disk-throughput cap. Lower priority and
thread count reduce contention but cannot guarantee the original throughput or
latency under less CPU. Backlog, capture gaps and health must remain visible.
Retain original records and costs; never reduce checks or drop observations to
claim unchanged effectiveness. Review any budget adjustment with quant's
resource priority as the governing constraint.


Sandboxed development descendants can occupy a nested immediate job with no
CPU-rate setting. The worker guard first checks the immediate job. If that does
not match, it opens only the existing named development job with QUERY access,
uses IsProcessInJob on the current process and that exact handle, then verifies
ENABLE|HARD_CAP, the manifest development rate, BelowNormal priority limit and
kill-on-close. A named rate or status receipt without current-process membership
is insufficient. No assignment, limit change, new job or uncapped fallback occurs.
All native-thread environment checks still apply. Access denial or unavailable
kernel evidence blocks work. This verification does not establish RAM/IO limits.
