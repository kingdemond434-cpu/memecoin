import unittest

from ops.watchdog import Policy, decide, observed_faults


class WatchdogUnitTests(unittest.TestCase):
    def healthy(self):
        return {
            "runtime_tasks": {"status": "OK", "failed": []},
            "source_mesh": {"sources": 2, "producers": 2,
                            "streaming": True},
            "yellowstone": {"status": "STREAMING"},
            "rpc_program_stream": {"status": "RPC_WS"},
            "memory": {"band": "calm"},
            "stream_events": {"total": 10, "token_created": 1},
            "pump_decoder": {"status": "OK"},
            "event_loop": {}, "data_miners": {"status": "OK"},
            "prediction": "OK", "rug_hazard": {"model_trained": True},
            "credentials": {"absent": []},
        }

    def call(self, readiness=None, state=None, now=10_000.0, **changes):
        values = dict(
            service_active=True, service_enabled=True,
            readiness=readiness or self.healthy(), readiness_age=10.0,
            state=state if state is not None else {}, now=now,
            policy=Policy(), trainer_active=False, training_age=10.0)
        values.update(changes)
        return decide(**values)

    def test_inactive_service_repairs_immediately(self):
        self.assertTrue(self.call(service_active=False).restart_desk)

    def test_internal_fault_requires_two_observations(self):
        readiness = self.healthy()
        readiness["yellowstone"] = {"status": "DEAD"}
        readiness["rpc_program_stream"] = {"status": "DEAD"}
        state = {}
        self.assertFalse(self.call(readiness, state).restart_desk)
        self.assertTrue(self.call(readiness, state, now=10_061.0).restart_desk)

    def test_restart_budget_cannot_be_bypassed(self):
        state = {"restart_times": [9_100.0, 9_400.0, 9_700.0]}
        plan = self.call(service_active=False, state=state)
        self.assertFalse(plan.restart_desk)
        self.assertIn("restart_budget_exhausted", plan.alerts)

    def test_external_blocker_is_alert_only(self):
        readiness = self.healthy()
        readiness["credentials"] = {"absent": [{"name": "X_BEARER_TOKEN"}]}
        repairs, alerts = observed_faults(readiness)
        self.assertEqual(repairs, [])
        self.assertIn("optional_credentials_absent", alerts)


if __name__ == "__main__":
    unittest.main()
