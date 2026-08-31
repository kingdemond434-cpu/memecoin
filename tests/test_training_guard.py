import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime.training_guard import main, mem_available_mib


class TestTrainingGuard(unittest.TestCase):
    def test_reads_linux_memavailable_in_mib(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text("MemTotal: 4000000 kB\nMemAvailable: 1536000 kB\n")
            self.assertEqual(mem_available_mib(path), 1500.0)

    def test_missing_memavailable_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text("MemTotal: 4000000 kB\n")
            self.assertIsNone(mem_available_mib(path))

    def test_condition_skips_below_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text("MemAvailable: 1024000 kB\n")
            with patch("sys.argv", ["training_guard", "--min-available-mib", "1400",
                                    "--meminfo", str(path)]):
                self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
