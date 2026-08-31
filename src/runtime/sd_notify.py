"""Tell systemd this process is alive, because the unit file asked.

`deploy/systemd/memecoin-shadow.service` is `Type=notify` with
`NotifyAccess=main` and `WatchdogSec=240s`. Both of those are PROMISES the
process has to keep, and until this module existed it kept neither:

  * `Type=notify` means systemd holds the unit in `activating` until the
    main process sends `READY=1` over $NOTIFY_SOCKET. Nothing ever sent it,
    so the desk ran perfectly while `systemctl is-active` said `activating`
    for ever, `systemctl start` blocked until `TimeoutStartSec=120s`, and
    every dependency ordered `After=` this unit waited on a readiness that
    was never going to arrive.

  * `WatchdogSec=240s` means systemd SIGABRTs the process unless it sends
    `WATCHDOG=1` inside every interval. Nothing ever sent that either, so
    the desk was killed roughly every four minutes from the moment the
    directive was added. That is the "62 process kills in 7 days ... all
    'Failed with result watchdog', not an in-process hang" recorded in the
    unit's own comment and attributed there to CPU starvation on a shared
    box. It was not contention. It was an unimplemented ping.

There is no dependency here on purpose. `systemd-python` is a compiled
package that has to build against libsystemd headers, and the protocol is a
newline-separated datagram to an AF_UNIX socket -- roughly twenty lines of
stdlib. A desk that cannot report readiness because a wheel failed to build
is a worse outcome than any this file prevents.

Everything degrades to a no-op off systemd: no $NOTIFY_SOCKET means running
under a shell, a test, or a container without notify, and none of those want
an exception. A send that fails is logged once at debug and never retried in
a way that could stall the caller -- the watchdog ping runs on the same event
loop as the decision path, and blocking it to talk to the service manager
would be exactly the hang the watchdog exists to catch.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Optional

logger = logging.getLogger(__name__)

#: Fraction of WatchdogSec at which to ping. systemd's own recommendation is
#: half; a third leaves room for one entirely missed tick on a loaded box
#: without the process being killed for it.
WATCHDOG_PING_FRACTION = 1.0 / 3.0

#: Floor on the ping interval. A misconfigured WatchdogSec of a second or two
#: should not turn into a busy loop against the service manager.
MIN_PING_INTERVAL_S = 1.0


class SystemdNotifier:
    """A one-way channel to the service manager, or nothing at all.

    Holds the socket open rather than reconnecting per message: readiness is
    sent once but the watchdog ping is sent for the life of the process, and
    a datagram socket that is reopened every few seconds is a file descriptor
    churn nobody needs.
    """

    def __init__(self, address: Optional[str] = None):
        self.address = address if address is not None else os.getenv("NOTIFY_SOCKET", "")
        self._socket: Optional[socket.socket] = None
        self._failed = False
        self.sent = 0
        if self.address:
            self._connect()

    @property
    def available(self) -> bool:
        """True only when there is a real socket to a real service manager."""
        return self._socket is not None and not self._failed

    def _connect(self) -> None:
        address = self.address
        # A leading '@' is systemd's spelling of the abstract namespace, which
        # Python spells with a leading NUL.
        if address.startswith("@"):
            address = "\0" + address[1:]
        try:
            handle = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
            # Never block the event loop on the service manager.
            handle.setblocking(False)
            handle.connect(address)
        except OSError as exc:
            logger.debug("systemd notify socket unavailable: %s", exc)
            self._failed = True
            return
        self._socket = handle

    def _send(self, message: str) -> bool:
        if not self.available:
            return False
        try:
            self._socket.send(message.encode("utf-8"))  # type: ignore[union-attr]
        except OSError as exc:
            # Not fatal and not retried. The desk's job is not to keep the
            # service manager informed at the cost of its own liveness.
            logger.debug("systemd notify send failed: %s", exc)
            return False
        self.sent += 1
        return True

    def ready(self, status: str = "") -> bool:
        """Announce that startup is complete. Sent exactly once, by main."""
        payload = "READY=1"
        if status:
            payload += f"\nSTATUS={_one_line(status)}"
        return self._send(payload)

    def watchdog(self) -> bool:
        """Keep-alive. Must arrive inside every WatchdogSec interval."""
        return self._send("WATCHDOG=1")

    def status(self, text: str) -> bool:
        """A line for `systemctl status`, so the unit says what it is doing."""
        return self._send(f"STATUS={_one_line(text)}")

    def extend_timeout(self, seconds: float) -> bool:
        """Push TimeoutStartSec out by `seconds` from now.

        Startup restores thousands of wallets and resolves dozens of
        channels, and on a cold cache that can outrun TimeoutStartSec on a
        loaded box. The wrong fix is a huge static timeout, which makes a
        genuinely wedged start undetectable for just as long. This is
        systemd's own answer: each phase says "I am still making progress,
        give me another N seconds", so a phase that stops making progress
        still hits its deadline promptly.
        """
        usec = int(max(0.0, float(seconds)) * 1_000_000)
        return self._send(f"EXTEND_TIMEOUT_USEC={usec}")

    def stopping(self, status: str = "") -> bool:
        """Announce a deliberate shutdown, so systemd does not call it a fault."""
        payload = "STOPPING=1"
        if status:
            payload += f"\nSTATUS={_one_line(status)}"
        return self._send(payload)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:  # pragma: no cover - shutdown only
                pass
            self._socket = None


def _one_line(text: str) -> str:
    """STATUS= is newline-delimited in the protocol, so it cannot contain one."""
    return " ".join(str(text).split())


def watchdog_interval_s() -> Optional[float]:
    """How often WATCHDOG=1 must be sent, from $WATCHDOG_USEC.

    None when the unit did not arm a watchdog, which is the common case off
    systemd. $WATCHDOG_PID guards the case where the variable was inherited
    by a child that is not the process systemd is watching -- pinging from
    the wrong process would keep a hung main process alive indefinitely,
    which is the exact failure the watchdog exists to prevent.
    """
    raw = os.getenv("WATCHDOG_USEC", "")
    if not raw:
        return None
    owner = os.getenv("WATCHDOG_PID", "")
    if owner and owner.strip() != str(os.getpid()):
        return None
    try:
        usec = int(raw)
    except ValueError:
        return None
    if usec <= 0:
        return None
    return max(MIN_PING_INTERVAL_S, (usec / 1_000_000.0) * WATCHDOG_PING_FRACTION)
