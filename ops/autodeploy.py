"""Pull, verify, restart -- or roll back. No laptop required.

Every deploy so far has been a person SSHing in to run three commands. That is
fine while debugging and pointless afterwards: the node can watch its own
branch, and the only reason not to let it is the risk of pulling something
broken.

So the risk is removed rather than avoided. A deploy is only applied if the
test suite passes ON THIS NODE, against the new code, before the service is
restarted. If it does not, the checkout returns to the commit that was running
and nothing changes. A node that deploys unverified code will eventually
deploy code that stops it collecting evidence, and it will do so at 3am.

Four rules:

**Verify before restart, never after.** A restart into a broken build costs
the time to notice plus the time to roll back. Running the suite first costs a
minute and costs it only on deploy.

**Roll back to the exact commit that was running**, recorded before the pull
rather than inferred after it. "The previous commit" is ambiguous when a pull
brings several.

**Never deploy over a dirty tree.** Local modifications on a trading node are
either an operator mid-repair or a fault; either way, overwriting them
silently is wrong.

**Say what happened, always.** A deploy that rolled back and told nobody is
worse than one that failed loudly, because the node then runs old code while
its branch says otherwise.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("autodeploy")

AUTODEPLOY_SCHEMA_VERSION = "v1"

#: How long the verification suite may take before it is abandoned. A suite
#: that hangs must not leave the node mid-deploy for ever.
DEFAULT_TEST_TIMEOUT_S = 900.0


@dataclass
class DeployResult:
    status: str
    detail: str = ""
    was: str = ""
    now: str = ""
    tests_ran: bool = False
    rolled_back: bool = False
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": AUTODEPLOY_SCHEMA_VERSION, "status": self.status,
                "detail": self.detail, "was": self.was, "now": self.now,
                "tests_ran": self.tests_ran, "rolled_back": self.rolled_back,
                "at": self.at}


def _git(root: Path, *args: str, timeout: float = 120.0) -> Tuple[int, str]:
    try:
        result = subprocess.run(["git", "-C", str(root), *args],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return result.returncode, (result.stdout + result.stderr).strip()


class AutoDeployer:
    """Watches one branch and applies it only if it passes here."""

    def __init__(self, root: Path, *, branch: str = "main",
                 service: str = "memecoin-shadow.service",
                 python: Optional[Path] = None,
                 test_timeout_s: float = DEFAULT_TEST_TIMEOUT_S,
                 state_path: Optional[Path] = None):
        self.root = Path(root)
        self.branch = branch
        self.service = service
        self.python = Path(python) if python else self.root / ".venv" / "bin" / "python"
        self.test_timeout_s = float(test_timeout_s)
        self.state_path = Path(state_path) if state_path else None
        self.history: List[DeployResult] = []

    # --- the steps -------------------------------------------------------

    def head(self) -> str:
        code, out = _git(self.root, "rev-parse", "HEAD")
        return out.strip() if code == 0 else ""

    def dirty(self) -> List[str]:
        code, out = _git(self.root, "status", "--porcelain")
        if code != 0:
            return ["status unreadable"]
        # Untracked files are not a modification of anything we would
        # overwrite, so they do not block a fast-forward.
        return [line for line in out.splitlines() if line and not line.startswith("??")]

    def behind(self) -> int:
        if _git(self.root, "fetch", "origin", self.branch)[0] != 0:
            return -1
        code, out = _git(self.root, "rev-list", "--count",
                         f"HEAD..origin/{self.branch}")
        try:
            return int(out.strip()) if code == 0 else -1
        except ValueError:
            return -1

    def verify(self) -> Tuple[bool, str]:
        """Run the suite against the new code, on this node."""
        try:
            result = subprocess.run(
                [str(self.python), "-m", "unittest", "discover", "-s", "tests", "-q"],
                cwd=str(self.root), capture_output=True, text=True,
                timeout=self.test_timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"suite exceeded {self.test_timeout_s:.0f}s"
        except OSError as exc:
            return False, f"suite could not run: {exc}"
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        return result.returncode == 0, " | ".join(tail)

    # --- the whole thing -------------------------------------------------

    def run(self) -> DeployResult:
        was = self.head()
        dirty = self.dirty()
        if dirty:
            return self._record(DeployResult(
                "SKIPPED", was=was,
                detail=(f"{len(dirty)} local modification(s); a dirty tree on "
                        "a trading node is an operator mid-repair or a fault, "
                        "and overwriting either silently is wrong")))

        behind = self.behind()
        if behind < 0:
            return self._record(DeployResult(
                "SKIPPED", was=was, detail="could not reach origin"))
        if behind == 0:
            return self._record(DeployResult(
                "CURRENT", was=was, now=was, detail=""))

        code, out = _git(self.root, "merge", "--ff-only", f"origin/{self.branch}")
        if code != 0:
            return self._record(DeployResult(
                "SKIPPED", was=was,
                detail=f"not a fast-forward, leaving it alone: {out[:200]}"))
        now = self.head()

        passed, summary = self.verify()
        if not passed:
            # Back to exactly what was running, recorded before the pull.
            _git(self.root, "reset", "--hard", was)
            return self._record(DeployResult(
                "ROLLED_BACK", was=was, now=was, tests_ran=True, rolled_back=True,
                detail=(f"{behind} commit(s) failed verification here and were "
                        f"reverted: {summary}")))

        restarted = subprocess.run(
            ["systemctl", "--user", "restart", self.service],
            capture_output=True, text=True).returncode == 0
        return self._record(DeployResult(
            "DEPLOYED" if restarted else "DEPLOYED_NOT_RESTARTED",
            was=was, now=now, tests_ran=True,
            detail=(f"{behind} commit(s) verified and applied" if restarted else
                    f"{behind} commit(s) verified but the service did not "
                    "restart; it is running the old code")))

    def _record(self, result: DeployResult) -> DeployResult:
        self.history.append(result)
        self.history = self.history[-50:]
        if self.state_path is not None:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                self.state_path.write_text(json.dumps(
                    [item.to_dict() for item in self.history]))
            except OSError as exc:
                logger.warning("deploy history unwritable: %s", exc)
        level = (logger.error if result.status in ("ROLLED_BACK",
                                                   "DEPLOYED_NOT_RESTARTED")
                 else logger.info)
        level("autodeploy %s: %s", result.status, result.detail or result.now[:8])
        return result
