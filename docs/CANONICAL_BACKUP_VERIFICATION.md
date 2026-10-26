# Verifying canonical backup extensions

Full hash-chain verification must happen before a snapshot can serve as a
prefix. Run `python -m tools.verify_evidence_backup --ledger SEALED_BACKUP
--output NEW_CERTIFICATE.json` on an owner-controlled, immutable backup while
the paper service keeps collecting. The tool never modifies the ledger.

A later sealed backup can use `--prefix EARLIER_BACKUP --certificate
EARLIER_CERTIFICATE.json`. The tool checks the certificate's own checksum,
extent and metadata, hashes the entire earlier snapshot, and compares every
prefix byte with the newer snapshot. It then verifies every added row's hash,
sequence and predecessor. Size/mtime changes, partial final rows, mismatched
bytes and invalid certificates abort. The resulting certificate covers the
entire newer backup and records whether verification reused a certified prefix.

This avoids repeatedly parsing millions of unchanged JSON records while the
main service is stopped. Binary comparison/hashing still reads the whole
prefix, so disk throughput remains a real limit. SQLite backups and their
integrity checks remain separate. Startup recovery and restarts still cause
capture gaps; this does not establish uninterrupted collection.

Keep certificates with their protected owner-controlled backups. They are
local corruption-detection receipts, **not** a trust mechanism for certificates
downloaded from third parties and **not** notarization against an administrator
who can rewrite both data and certificates. Do not accept an arbitrary external
certificate as proof that an uninspected prefix is valid.
