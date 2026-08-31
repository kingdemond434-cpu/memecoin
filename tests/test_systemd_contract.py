"""The unit file makes promises; these assert the process keeps them.

`memecoin-shadow.service` was `Type=notify` with `WatchdogSec=240s` while the
process implemented neither side of that contract. The desk ran fine and
systemd reported `activating` for ever, `systemctl start` blocked until its
start timeout, and the watchdog SIGABRTed a perfectly healthy process every
four minutes -- 62 kills in 7 days, recorded in the unit's own comment and
blamed there on CPU contention.

Nothing catches that. The unit is valid, the Python is valid, the tests pass,
and the two are wrong only about each other. So the contract is asserted here
directly: if a future unit adds Type=notify or WatchdogSec, or a refactor
drops the notifier, one of these fails.
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

from src.runtime.sd_notify import (
    MIN_PING_INTERVAL_S, SystemdNotifier, watchdog_interval_s)

UNITS = Path(__file__).resolve().parents[1] / "deploy" / "systemd"


def _directives(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values.setdefault(key.strip(), []).append(value.strip())
    return values


class NotifierSpeaksTheProtocol(unittest.TestCase):
    """Against a real AF_UNIX socket, because the protocol is the point."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "notify")
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.server.bind(self.path)
        self.server.settimeout(2.0)
        self.addCleanup(self.server.close)

    def _received(self) -> str:
        return self.server.recv(4096).decode("utf-8")

    def test_ready_is_the_datagram_systemd_waits_for(self):
        notifier = SystemdNotifier(self.path)
        self.assertTrue(notifier.available)
        self.assertTrue(notifier.ready("desk running"))
        message = self._received()
        self.assertIn("READY=1", message.splitlines())
        self.assertIn("STATUS=desk running", message)

    def test_watchdog_ping_is_sent_verbatim(self):
        notifier = SystemdNotifier(self.path)
        self.assertTrue(notifier.watchdog())
        self.assertEqual("WATCHDOG=1", self._received())

    def test_status_can_never_break_the_line_protocol(self):
        # STATUS= is newline-delimited. A phase name containing a newline
        # would inject a directive of the caller's choosing.
        notifier = SystemdNotifier(self.path)
        notifier.status("starting\nREADY=1\nphase")
        message = self._received()
        # One line is the whole defence: READY=1 survives only as text
        # inside the status string, where systemd reads it as a status and
        # not as the readiness announcement it is spelled like.
        self.assertEqual(1, len(message.splitlines()))
        self.assertTrue(message.startswith("STATUS="))
        self.assertNotIn("\n", message)

    def test_extend_timeout_is_expressed_in_microseconds(self):
        notifier = SystemdNotifier(self.path)
        notifier.extend_timeout(90.0)
        self.assertEqual("EXTEND_TIMEOUT_USEC=90000000", self._received())

    def test_stopping_precedes_a_deliberate_exit(self):
        notifier = SystemdNotifier(self.path)
        notifier.stopping("shutting down")
        self.assertIn("STOPPING=1", self._received().splitlines())


class NotifierIsInertOffSystemd(unittest.TestCase):
    """A desk started from a shell must not care that no manager is listening."""

    def test_no_socket_means_no_op_not_exception(self):
        notifier = SystemdNotifier("")
        self.assertFalse(notifier.available)
        self.assertFalse(notifier.ready())
        self.assertFalse(notifier.watchdog())
        self.assertEqual(0, notifier.sent)

    def test_a_socket_nobody_is_listening_on_is_also_survivable(self):
        notifier = SystemdNotifier(os.path.join(tempfile.mkdtemp(), "absent"))
        self.assertFalse(notifier.ready())


class WatchdogIntervalReadsTheEnvironment(unittest.TestCase):

    def setUp(self):
        self._saved = {key: os.environ.get(key)
                       for key in ("WATCHDOG_USEC", "WATCHDOG_PID")}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_interval_is_a_fraction_of_the_armed_deadline(self):
        os.environ["WATCHDOG_USEC"] = str(240 * 1_000_000)
        os.environ.pop("WATCHDOG_PID", None)
        interval = watchdog_interval_s()
        self.assertIsNotNone(interval)
        # Comfortably inside the deadline, so one missed tick is survivable.
        self.assertLess(interval, 240 / 2.0)

    def test_no_watchdog_armed_means_no_pinging(self):
        os.environ.pop("WATCHDOG_USEC", None)
        self.assertIsNone(watchdog_interval_s())

    def test_a_child_process_must_not_ping_for_its_parent(self):
        # Pinging from the wrong process would keep a hung main process
        # alive for ever, which is precisely what the watchdog prevents.
        os.environ["WATCHDOG_USEC"] = str(240 * 1_000_000)
        os.environ["WATCHDOG_PID"] = str(os.getpid() + 1)
        self.assertIsNone(watchdog_interval_s())

    def test_an_absurd_deadline_does_not_become_a_busy_loop(self):
        os.environ["WATCHDOG_USEC"] = "1000"
        os.environ.pop("WATCHDOG_PID", None)
        self.assertGreaterEqual(watchdog_interval_s(), MIN_PING_INTERVAL_S)


class UnitPromisesMatchTheProcess(unittest.TestCase):

    def test_every_notify_unit_running_the_desk_sends_readiness(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        for unit in sorted(UNITS.glob("*.service")):
            directives = _directives(unit)
            if "notify" not in directives.get("Type", []):
                continue
            if not any("src.main" in line for line in directives.get("ExecStart", [])):
                continue
            self.assertIn("systemd.ready(", source,
                          f"{unit.name} is Type=notify but src.main never "
                          "sends READY=1, so the unit stays `activating`")

    def test_every_watchdog_unit_running_the_desk_pings(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "main.py").read_text()
        for unit in sorted(UNITS.glob("*.service")):
            directives = _directives(unit)
            if not directives.get("WatchdogSec"):
                continue
            if not any("src.main" in line for line in directives.get("ExecStart", [])):
                continue
            self.assertIn("systemd.watchdog()", source,
                          f"{unit.name} arms WatchdogSec but src.main never "
                          "sends WATCHDOG=1, so systemd kills it on schedule")

    def test_the_memory_ceiling_leaves_the_host_room_to_survive(self):
        # A MemoryMax equal to host RAM is not a ceiling: the kernel OOM
        # killer reaches the box before the cgroup limit reaches the unit,
        # and the in-process governor's bands -- fractions of this number --
        # become unreachable, so no relief valve can ever fire.
        unit = UNITS / "memecoin-shadow.service"
        directives = _directives(unit)
        limit = directives.get("MemoryMax", [])
        self.assertTrue(limit, "the shadow unit must cap its memory")
        self.assertTrue(
            directives.get("MemoryHigh"),
            "MemoryHigh turns pressure into reclaim instead of a kill and "
            "must be set alongside MemoryMax")
        self.assertLess(_bytes(directives["MemoryHigh"][0]), _bytes(limit[0]),
                        "MemoryHigh must sit below MemoryMax or it never acts")
        # 4 GB host. Anything at or above that is the unreachable ceiling.
        self.assertLess(_bytes(limit[0]), 4 * 1024 ** 3)

    def test_the_governor_sheds_before_the_cgroup_kills(self):
        from src.runtime.memory_governor import DEFAULT_HARD_FRACTION
        directives = _directives(UNITS / "memecoin-shadow.service")
        ceiling = _bytes(directives["MemoryMax"][0])
        shed_at = ceiling * DEFAULT_HARD_FRACTION
        self.assertLess(shed_at, _bytes(directives["MemoryHigh"][0]),
                        "the desk must give something up before the kernel "
                        "starts reclaiming on its behalf")


def _bytes(value: str) -> int:
    units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    text = value.strip()
    if text and text[-1].upper() in units:
        return int(float(text[:-1]) * units[text[-1].upper()])
    return int(text)


if __name__ == "__main__":
    unittest.main()


class HttpClientSurvivesASecondEventLoop(unittest.TestCase):
    """The failure that took down four substitution ladders every run.

    `OffloadedPool` runs the miners on their own loop in their own thread and
    hands them the desk's shared HttpClient. An aiohttp session belongs to
    the loop that made it, so every miner request raised `RuntimeError:
    Timeout context manager should be used inside a task` -- and the ladder,
    reading that as the endpoint's fault, quarantined healthy operators for
    300s each. 1866 tests passed throughout, because none of them ever used
    the client from two loops.
    """

    def test_each_loop_gets_its_own_session(self):
        import asyncio as _asyncio

        from src.collectors.transports import HttpClient

        client = HttpClient()
        sessions = []

        async def take_one():
            sessions.append(await client.session())

        for _ in range(2):
            loop = _asyncio.new_event_loop()
            try:
                loop.run_until_complete(take_one())
            finally:
                loop.run_until_complete(client.close())
                loop.close()

        self.assertEqual(2, len(sessions))
        self.assertIsNot(sessions[0], sessions[1],
                         "a session shared across loops raises on first use")

    def test_one_loop_reuses_its_session(self):
        import asyncio as _asyncio

        from src.collectors.transports import HttpClient

        client = HttpClient()

        async def twice():
            first = await client.session()
            second = await client.session()
            # The bounded connection pool is the reason this class exists;
            # a session per CALL would be a socket per source.
            self.assertIs(first, second)
            await client.close()

        loop = _asyncio.new_event_loop()
        try:
            loop.run_until_complete(twice())
        finally:
            loop.close()


class LocalFaultsAreNotBlamedOnEndpoints(unittest.TestCase):

    def test_a_cross_loop_error_is_recognised_as_ours(self):
        from src.research.regional_miners import _is_local_fault

        self.assertTrue(_is_local_fault(RuntimeError(
            "Timeout context manager should be used inside a task")))
        self.assertTrue(_is_local_fault(RuntimeError("Event loop is closed")))

    def test_a_real_outage_is_still_the_endpoint_s(self):
        from src.research.regional_miners import _is_local_fault

        self.assertFalse(_is_local_fault(RuntimeError("HTTP 503 from api.example")))
        self.assertFalse(_is_local_fault(OSError("Connection refused")))


class SupervisorDoesNotRestartABootingDesk(unittest.TestCase):
    """The desk now answers while it boots; the supervisor must read that.

    Binding the port before the subsystems exist is what makes a slow start
    observable. It also means /status answers 503 with a phase name until
    readiness, and a supervisor that reads any non-200 as silence would
    restart a desk whose only fault was starting -- sending it back to the
    beginning of the very startup it was being punished for taking.
    """

    def test_a_starting_desk_is_a_warning_and_never_escalates(self):
        from ops.supervisor import build_health

        health = build_health({"status": "starting", "phase": "prediction",
                               "uptime_seconds": 12.0},
                              None, 0.0, Path("."))
        checks = health["checks"]
        self.assertEqual(1, len(checks))
        self.assertEqual("WARN", checks[0]["state"])
        self.assertFalse(checks[0]["escalate"])
        self.assertIn("prediction", checks[0]["detail"])

    def test_an_unreachable_desk_is_still_critical(self):
        from ops.supervisor import build_health

        checks = build_health(None, None, 0.0, Path("."))["checks"]
        self.assertEqual("CRITICAL", checks[0]["state"])
        self.assertTrue(checks[0]["escalate"])


class OneTelegramClientForTheWholeDesk(unittest.TestCase):
    """38 clients on one SQLite session is a lock fight nobody wins.

    Telethon's session is single-writer. The social collector opens it four
    startup phases before the transports do and keeps it, so a client built
    later does not contend with the other transports -- it contends with the
    collector, and loses: `database is locked`, from the very client added
    to stop that happening. The live client is the one to reuse.
    """

    def setUp(self):
        from src.collectors.transports import TelegramChannelTransport

        self.transports = {
            f"telegram:c{index}": TelegramChannelTransport(f"telegram:c{index}",
                                                           f"c{index}")
            for index in range(4)}

    def test_a_live_collector_client_is_reused_by_every_channel(self):
        import asyncio as _asyncio

        from src.collectors.transports import share_telegram_client

        class LiveClient:
            def is_connected(self):
                return True

        client = LiveClient()
        reason = _asyncio.new_event_loop().run_until_complete(
            share_telegram_client(self.transports, client))
        self.assertIsNone(reason)
        for transport in self.transports.values():
            self.assertIs(client, transport.client)
            # Ownership stays with the collector: a transport shutting down
            # must not disconnect the client the whole desk is using.
            self.assertFalse(transport._owns_client)

    def test_a_disconnected_client_is_not_handed_out(self):
        import asyncio as _asyncio

        from src.collectors.transports import share_telegram_client

        class DeadClient:
            def is_connected(self):
                return False

        _asyncio.new_event_loop().run_until_complete(
            share_telegram_client(self.transports, DeadClient()))
        for transport in self.transports.values():
            self.assertIsNone(transport.client)

    def test_a_shared_client_is_never_disconnected_by_a_transport(self):
        import asyncio as _asyncio

        class Client:
            def __init__(self):
                self.disconnected = False

            def is_connected(self):
                return True

            def remove_event_handler(self, handler):
                pass

            async def disconnect(self):
                self.disconnected = True

        client = Client()
        transport = self.transports["telegram:c0"]
        transport.attach_client(client)
        _asyncio.new_event_loop().run_until_complete(transport.stop())
        self.assertFalse(client.disconnected,
                         "stopping one channel must not take the desk's "
                         "Telegram connection down with it")
