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

Inputs receive durable scheduling tickets; each served input moves to the
back of that order. Tickets survive restarts and wall-clock corrections.
New arrivals cannot repeatedly jump ahead of the
remaining windows of an older long document. Each result records measured
worker-request wall time (including startup/tokenization when applicable),
which is an observation of processing latency, not a billed inference cost.

The documented training languages are en/ar/de/es/fr/ja/zh/id/hi/it/ms/pt.
Other languages remain outside that stated training scope. Raw softmax outputs
are explicitly **uncalibrated**, with trading skill **UNMEASURED**. They do not
prove factual credibility, insider access, sentiment accuracy in crypto,
return probability or profitable entries. They have no order authority and do
not change the trained feature schema. Trading-domain outcome calibration and
forward evaluation remain separate work. Synthetic smoke checks validate the
offline pipeline and full-text windows, not trading accuracy.
