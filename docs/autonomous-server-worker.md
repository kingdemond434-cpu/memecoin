# Bounded server development continuation

`ops/autonomous_worker.py` is the supervisor installed as `worker.py` beside
`worker_resource_guard.py` in the existing private VPS worker directory. It uses
the same authorized Codex account and resumable session; no API key, new model,
account, signing authority or parallel worker is provisioned.

The existing `MemecoinAutonomousDevelopment` task remains Administrator **S4U,
Limited**, wrapped by the development CPU job. Its boot/periodic triggers and
`IgnoreNew` instance policy remain intact. One Windows file lock covers all
sequential turns and waits. The supervisor also resumes after a successful turn
without needing another desktop prompt; API/account errors retain exponential
backoff. `PAUSE` prevents further cycles, and `READY` must remain present.

Before work, the supervisor validates the shared manifest (total ≤10%, developer
≤2%), its **actual current process job** (`ENABLE|HARD_CAP`, matching rate), and
single-thread native environment. No uncapped fallback is possible. The check
uses Windows `QueryInformationJobObject(NULL, ...)`, avoiding named-object
queries across accounts/sessions. It records only nonsecret resource fields.

Installation is separate from task registration. With the developer disabled
and `PAUSE` retained, back up the existing supervisor, install both reviewed
Python sources and the current continuation instructions. Create the
`RESOURCE-CHECK-ONLY` marker and start the existing bounded task. A fresh
`logs/resource-guard.json` must prove its actual job/account path before removing
that marker and the user-authorized pause. Do not bypass an unsuccessful guard.

Model children inherit the CPU job. Their environments clear unrelated GitHub
tokens and live wallet/signer acknowledgement; GitHub uses the existing protected
wrapper. CLI requests use workspace-write automatic approval review. No sandbox
or review bypass, new task principal or persistent privilege elevation is added.

Current instructions explicitly forbid uncapped scheduled helper tasks, quant
changes, resource-limit removal and legacy deployments that replace CPU wrappers.
Source changes and deployments still need review, meaningful tests, verified
recoverable state and actual advancing paper data flow. Resource containment
alone does not establish those properties or a profitable strategy.

`logs/worker-status.json` exposes lifecycle, current child PID, event timestamps,
thread ID, resource receipt and retry deadline. `logs/resource-guard.json` records
native CPU verification. Never print raw event logs, credentials, connection
files or private URLs in a status bridge. The supervisor stores completion for
independent review; a local desktop app that is closed cannot deliver a chat
notification. Live trading remains separately authorized by the user.

The worker operates independently of the laptop and an interactive RDP window,
subject to server/network/account availability. Neither an infinite uptime nor
completion of externally unavailable requirements is promised. Genuine blockers
stay explicit while unaffected work continues, including a usable dashboard.
