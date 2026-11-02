The loopback health probe diagnoses one request to an existing paper-service port.
It returns cumulative completed-stage times, a failed stage and exception type,
HTTP status, and a strictly typed projection of approved health fields. It never
returns response headers, arbitrary status strings, body content or error text.

Only IPv4 127.0.0.1 ports18081..18084 and /health are supported. It ignores proxy
environment settings, requests connection close, refuses redirects and transfer
encoding, limits headers to16KiB/body to64KiB, and sets each socket timeout from
one monotonic deadline (maximum10seconds). Socket syscalls and scheduling are not
preemptible: elapsed time may exceed the requested budget. JSON decode is bounded
by body size and checked against the deadline afterward. There are no retries,
service/task controls, provider calls, authentication or credential loads.

Times include scheduling delay; a failed stage is not proof of server root cause,
CPU starvation, event-loop blocking or provider failure. A decoded response is
only a point observation: use actual lock values and advancing persisted counters
separately. This tool does not establish sustainable latency or capacity. It runs
as an ordinary child of the existing capped developer and does not change limits.
