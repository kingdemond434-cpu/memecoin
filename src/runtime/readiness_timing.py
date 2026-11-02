"""Process-local wait states around the single existing readiness refresh."""
import threading,time

PHASES=('idle','scheduled','frontier_snapshot','projection_dispatch','projection_worker','projection_owner_resume')


class ReadinessRefreshTiming:
    def __init__(self,clock=time.monotonic,wall=time.time):
        self.clock,self.wall=clock,wall;self._lock=threading.Lock()
        self.started=clock();self.phase='scheduled';self.mark=self.started
        self.worker_started=None;self.worker_finished=None;self.finished=None

    def frontier(self):
        with self._lock:self.phase='frontier_snapshot';self.mark=self.clock()

    def worker(self,call,*args):
        with self._lock:self.phase='projection';self.mark=self.clock()
        def run():
            with self._lock:self.worker_started=self.clock()
            try:return call(*args)
            finally:
                with self._lock:self.worker_finished=self.clock()
        return run

    def finish(self):
        with self._lock:self.finished=self.clock();self.phase='idle'

    def report(self):
        with self._lock:
            now=self.clock();phase=self.phase;mark=self.mark
            if phase=='projection':
                if self.worker_finished is not None:phase='projection_owner_resume';mark=self.worker_finished
                elif self.worker_started is not None:phase='projection_worker';mark=self.worker_started
                else:phase='projection_dispatch'
            return dict(refresh_phase=phase,
                refresh_pending_seconds=max(0,now-self.started) if self.finished is None else 0.0,
                refresh_phase_seconds=max(0,now-mark) if self.finished is None else 0.0,
                refresh_wait_observed_at=self.wall(),
                refresh_wait_scope='process_local_wait_not_pure_loop_lag')
