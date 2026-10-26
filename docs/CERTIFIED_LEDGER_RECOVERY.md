# Recover a canonical ledger from a protected verification checkpoint

The legacy constructor decodes every historical JSON line to recover its last
sequence and hash. Large journals can delay startup even after the deployment
has already verified a complete sealed backup. Optional certified recovery
reuses that verification without trusting an unchecked file size or record count.

Set `global.evidence_recovery_certificate` to a protected owner-local certificate
created by `tools.verify_evidence_backup`. The certificate must be outside the
service account's writable state, with owner/system write access and service
read access. Its checksum detects corruption; it is not a signature or protection
against a malicious administrator. An arbitrary untrusted certificate is not
acceptable evidence. No certificate is enabled automatically by this source change.

At startup the ledger:

1. Validates certificate schema, checksum and extent/digest types.
2. Streams and SHA-256 hashes every byte of the live certified prefix.
3. Checks that the certified boundary ends at a complete row and that its last
   actual row agrees with the certificate's sequence and hash.
4. Verifies every subsequent row's sequence, predecessor and content hash.
5. Refuses changed/truncated files, incomplete tails, certificate mismatch or
   a concurrent file replacement/change before publishing recovered state.

Any failure blocks appends; it never falls back to unverified recovery. No history
is removed, rewritten or resubmitted, and actual observation times stay unchanged.
Without this setting the legacy constructor remains available. `/status` ledger
reports include recovery mode, prefix count, measured time and error type.

This saves JSON decoding of the certified history, not the mandatory prefix
byte read. Other startup work, disk throughput and unavailable providers remain
separate causes of delay. A local recovery benchmark is not an uptime guarantee.
Deployment still needs an integrity-verified backup and actual service checks.
