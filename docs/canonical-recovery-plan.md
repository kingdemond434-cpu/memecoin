# Canonical recovery input plan

`compose_recovery_plan` composes at most 32 historical segment receipts from an
explicit caller-supplied baseline cursor. It rejects gaps, overlaps, predecessor
hash conflicts, invalid sizes and duplicate Windows path references. It performs
no filesystem operations, copies, sweeps or production changes.

This is an input plan only. The baseline and receipt provenance must be trusted
separately; referenced segment contents and current existence are not checked.
A plan cannot authorize restoration or deployment. Actual bounded readback,
full chain reconstruction, large-store integrity and cross-store coherence
remain separate unfinished acceptance requirements. No live authority exists.
