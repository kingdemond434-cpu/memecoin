# Offline multilingual sentiment research

All normalized public/authorized source observations share a durable sentiment
queue, including Telegram, other connected social platforms and public onion
observations. Source availability still depends on each transport. Nothing in
this module unlocks private accounts or paid APIs. Text is retained in full;
nonoverlapping windows cover every tokenizer token rather than dropping the
end of a post after the model's 512-token context limit.

Raw source evidence precedes inference. A resumable hash-chain verifier also
replays historical `NARRATIVE_OBSERVATION` records and repairs missed immediate
intakes. Its byte quantum and four-window inference quantum bound each pass,
without limiting the source/history universe. Replay lag, input backlog,
oldest pending age, failures and unacknowledged windows appear on `/status`.
Partial last lines are retried; changed/truncated canonical prefixes stop
replay visibly. Historical replay has no retroactive trading availability.

Each window binds the source reference, full-text hash, model revision, weight
hash, worker-source hash, token interval and actual inference time. Results are
retained canonically before ACK advances the token cursor. Restart replays
unacknowledged output. Episode observations use actual canonical availability,
never an earlier social post time. Reposts remain separate observations with
source lineage; the module does not claim independent opinions or infer wallet
ownership from text. A window is document sentiment, not token-specific belief.

The local worker uses the Apache-2.0
[multilingual sentiment student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)
and a pinned
[ONNX conversion](https://huggingface.co/Xenova/distilbert-base-multilingual-cased-sentiments-student/tree/9d9ac661fd7b0b48535a1fc99b20ae6947629e65).
The three allowed data files have exact SHA-256 hashes in the module. No remote
model code, pickle, model download or network inference runs in the worker.
Its environment excludes RPC keys and wallet settings. ONNX telemetry is off,
CPU inference uses one thread, and timeout/cancellation kills and reaps the
worker. Shutdown runs on the miner's owning loop before databases close.

Install `requirements-neural-research.txt` into a **separate** interpreter and
provision the pinned model cache. Set these owner-reviewed paths in the
deployment's `global` section:

```yaml
neural_sentiment:
  python: /absolute/path/to/isolated/python
  model_root: /absolute/path/to/pinned/model/cache
```

Windows also needs the supported Microsoft C++ runtime, as specified by the
[ONNX Runtime prerequisites](https://onnxruntime.ai/docs/install/#requirements).
Use the [official Microsoft redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
Validate its Microsoft publisher signature before installation. A package
install alone is insufficient: run the offline actual-model smoke in the
clean worker environment. Record installer restart-pending status separately;
do not reboot an unattended trading host as part of this setup.

Inputs receive durable arrival and service tickets, independent of wall-clock
corrections. Each served input moves to the back of the age-fair order. Each result records measured
worker-request wall time (including startup/tokenization when applicable),
which is an observation of processing latency, not a billed inference cost.

Live intake and canonical historical replay have separate durable scheduling
lanes. Of every four available slots, two favor the newest live arrivals, one
serves the oldest live service ticket and one serves historical replay. Fresh
posts therefore bypass old live backlog; long-document tails and older live
posts retain their own service share. Empty lanes share their unused capacity.
Existing rows migrate by original intake provenance and durable insertion order.
Lane backlog and oldest age are reported separately. This bounds competition
from a large backfill; it does not guarantee processing deadlines under overload.

Replay verifies its source rows before indexing them. All inputs from one byte
quantum and its replay cursor commit together in the derived SQLite database,
with a writer reservation and cursor comparison to prevent concurrent rewind.
An intake failure rolls back that complete batch; retained canonical sources
remain available for retry. EOF does not commit an unchanged checkpoint. This
removes per-post commit/fsync overhead without weakening durability, source
hash checks or canonical annotation ACKs. `replay_commits` and measured
`last_replay_seconds` expose the work. These are processing observations, not
provider billing, token prediction accuracy or a guaranteed throughput rate.

The miner declares `LOCAL_CPU` resource use and has a separate serial execution
lane. Slow sockets can occupy all network slots without occupying this offline
slot. ONNX defaults to one CPU thread; `global.neural_sentiment.intra_op_threads`
may explicitly select a positive integer within host logical CPU capacity after
benchmarking on that host. Inter-op concurrency remains one. The profile is
passed as a subprocess argument, reported on every annotation and included in
the inference cache key. Each inference pass has bounded work, and
neither lane bypasses canonical retention or delivery acknowledgements. The
resource declaration is scheduling metadata for trusted callables, not a network
sandbox. Per-miner queue-wait and fetch wall times expose actual contention.

Identical text windows reuse actual inference keyed by full-text hash, offset,
model/tokenizer/configuration hashes and the inference-function source hash.
Each source observation still has its own retained input, source reference,
annotation and canonical ACK time. Reuse is labeled with the original inference
time/worker hash and never counts as an independent opinion or a new model run.
Cache corruption triggers real recomputation. Worker calls, cache hits and
corruption are measured separately. A time-bounded pass can drain cached work
faster while keeping expensive new model calls bounded; no input is discarded.
The two-second cooperative dispatch quantum and 256-output batch bound the
active work; there is no additional four-inference limit. Failed attempts also
consume that time quantum, so a failing worker cannot multiply its timeout by
the entire row batch. One in-flight inference may exceed the dispatch quantum
and retains its own timeout. Cache, annotation and service-ticket writes commit
together per result; canonical retention/ACK remains a separate required step.

The documented training languages are en/ar/de/es/fr/ja/zh/id/hi/it/ms/pt.
Other languages remain outside that stated training scope. Raw softmax outputs
are explicitly **uncalibrated**, with trading skill **UNMEASURED**. They do not
prove factual credibility, insider access, sentiment accuracy in crypto,
return probability or profitable entries. They have no order authority and do
not change the trained feature schema. Trading-domain outcome calibration and
forward evaluation remain separate work. Synthetic smoke checks validate the
offline pipeline and full-text windows, not trading accuracy.
