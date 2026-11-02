# Canonical recovery progress

verify_next_recovery_segment records at most one actual staged-segment chain
verification per invocation. Progress is bound to the normalized bounded recovery
plan, stored as at most 64 KiB, fsynced and atomically replaced under the shared
exclusive canonical-segment.lock. Invalid history, changed manifests, corruption
and competing writers block progress. Failed publication preserves prior state.

Callers must provide trusted private ancestors and authenticate the baseline and
receipt provenance separately. Previous results are historical and not freshly
revalidated. Even all segment receipts recorded is not complete recovery,
full-prefix authentication, cross-store coherence or a deployment certificate.
No automatic scan/copy/loop or production mutations are performed. CPU proof is
not an IO or RAM cap; host admission is required.
