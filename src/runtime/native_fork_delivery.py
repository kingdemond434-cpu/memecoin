"""Owner-local native fork delivery outcomes; never durable acceptance receipts."""
from __future__ import annotations


class NativeForkOwnerDelivery:
    """Fixed counters around the existing exact/PIT lookup and episode append."""

    def __init__(self):
        self.counts = dict.fromkeys(('records_seen', 'invalid_identity', 'proof_unavailable',
            'identity_mismatch', 'lookup_errors', 'application_errors', 'interrupted',
            'episode_unavailable', 'in_memory_applied', 'application_unconfirmed'), 0)

    def deliver(self, record, *, lookup, builder, as_of):
        self.counts['records_seen'] += 1
        token, identity = record.get('token'), record.get('probe_id')
        if not isinstance(token, str) or not token or not isinstance(identity, str) or not identity:
            self.counts['invalid_identity'] += 1
            return
        try:
            proof = lookup(token, as_of=as_of, probe_id=identity)
        except Exception:
            self.counts['lookup_errors'] += 1
            raise
        except BaseException:
            self.counts['interrupted'] += 1
            raise
        if not proof:
            self.counts['proof_unavailable'] += 1
            return
        if proof.get('probe_id') != identity:
            self.counts['identity_mismatch'] += 1
            return
        try:
            applied = builder.record_market_observation(token,
                {**proof, 'type': 'native_fork_roundtrip', 'timestamp': proof['available_at']})
        except Exception:
            self.counts['application_errors'] += 1
            raise
        except BaseException:
            self.counts['interrupted'] += 1
            raise
        if applied is True:
            self.counts['in_memory_applied'] += 1
        elif applied is False:
            self.counts['episode_unavailable'] += 1
        else:
            # A normal return from an alternate/older sink proves no append.
            self.counts['application_unconfirmed'] += 1

    def report(self):
        return {'status': 'PROCESS_LOCAL', 'counts': dict(self.counts),
                'scope': 'owner callback and episode memory only; counters reset on restart',
                'durable_acceptance': False, 'full_replay_verified': False,
                'real_fill': False, 'promotion_eligible': False}
