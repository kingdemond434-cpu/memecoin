"""Short-lived CPU evidence for one unchanged, already kernel-capped operation."""
import math
import os
import time
from ops.worker_resource_guard import verify


class CpuAdmission:
    """At most one second of cached evidence; never a CPU limit or permission.

    Use only when the operation cannot change jobs, tasks, limits or identity.
    The kernel must enforce the existing development job throughout. Independent
    byte/VM/time/disk/IO gates remain mandatory. Every denial latches this object
    closed; create a new explicitly admitted operation rather than retrying it.
    """
    def __init__(self, *, max_age_seconds=1, resource_check=verify,
                 clock=time.monotonic, pid=os.getpid):
        if (type(max_age_seconds) not in (int, float)
                or not math.isfinite(max_age_seconds) or not 0 < max_age_seconds <= 1):
            raise ValueError('CPU evidence lifetime must be bounded to one second')
        self._age=max_age_seconds;self._check=resource_check;self._clock=clock;self._pid=pid
        self._owner=pid()
        if type(self._owner) is not int or self._owner<=0:
            raise ValueError('Current process identity required')
        self._last=None;self._checked=None;self._proof=None;self._denied=None
        self._checks=0;self._hits=0;self._allowed=None

    @staticmethod
    def _time(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value<0:
            raise ValueError('Finite monotonic clock required')
        return value

    def _deny(self, reason):
        self._denied=reason;self._proof=None;self._checked=None;self._allowed=False
        return False

    def _same_process(self):
        value=self._pid()
        return type(value) is int and value==self._owner

    def __getstate__(self):
        raise TypeError('CPU admission evidence cannot be serialized')

    def __call__(self):
        if self._denied is not None:return False
        try:
            if not self._same_process():return self._deny('process_changed')
            now=self._time(self._clock())
            if self._last is not None and now<self._last:return self._deny('clock_regressed')
            self._last=now
            if self._checked is not None and now-self._checked<self._age:
                self._hits+=1;self._allowed=True;return True
            self._checks+=1
            try:proof=self._check()
            except Exception:return self._deny('resource_check_unavailable')
            end=self._time(self._clock())
            if end<now:return self._deny('clock_regressed')
            self._last=end
            if not self._same_process():return self._deny('process_changed')
            if end-now>=self._age:return self._deny('proof_expired_during_check')
            if (not isinstance(proof,dict) or proof.get('status')!='VERIFIED'
                    or type(proof.get('cpu_rate')) is not int or proof['cpu_rate']!=200
                    or type(proof.get('kernel_flags')) is not int or proof['kernel_flags']!=5
                    or type(proof.get('native_threads')) is not int or proof['native_threads']!=1
                    or type(proof.get('total_configured_percent')) not in (int,float)
                    or proof['total_configured_percent']!=10):
                return self._deny('resource_proof_rejected')
            self._proof={k:proof[k] for k in ('status','cpu_rate','kernel_flags','native_threads','total_configured_percent')}
            self._checked=now;self._allowed=True;return True
        except Exception:return self._deny('clock_or_identity_unavailable')

    def snapshot(self):
        """Historical diagnostics only; callers must invoke the guard to admit work."""
        return dict(max_age_seconds=self._age,proof_checks=self._checks,cache_hits=self._hits,
                    last_admission_allowed=self._allowed,denial_reason=self._denied,
                    proof_started_monotonic=self._checked,
                    last_proof=dict(self._proof) if self._proof is not None else None,
                    kernel_enforcement_required=True,limits_changed=False,
                    permissions_granted=False,memory_or_io_cap_claimed=False)
