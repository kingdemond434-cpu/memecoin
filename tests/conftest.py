"""Suite-wide test environment."""
import pytest


@pytest.fixture(autouse=True)
def _no_host_disk_floor(monkeypatch):
    """The production free-space floor (20 GiB) reads the host volume. A CI
    runner with a small disk would make unrelated ledger and journal tests
    refuse writes, so the suite disables it; tests/test_disk_floor.py injects
    its own floors and covers the behaviour directly."""
    monkeypatch.setenv("MEMECOIN_DISK_FLOOR_BYTES", "0")
