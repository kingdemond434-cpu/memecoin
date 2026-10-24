import subprocess

import pytest

from tools.validate_checkout import validate


def test_validation_cannot_copy_production_state_or_inherit_credentials(tmp_path, monkeypatch):
    source = tmp_path / "production"
    (source / "tests").mkdir(parents=True)
    (source / "tests/test_isolation.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def test_isolated():\n"
        " assert not Path('data/state/trade_evidence.jsonl').exists()\n"
        " assert 'OPENAI_API_KEY' not in os.environ\n"
        " assert os.environ['ALLOW_LIVE_TRADING'] == ''\n"
        " assert os.environ['SOLANA_PRIVATE_KEY'] == ''\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "tests"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    state = source / "data/state/trade_evidence.jsonl"
    state.parent.mkdir(parents=True)
    state.write_text("retained production sentinel", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-never-pass-to-child")
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "fixture-never-pass-to-child")
    report = validate(source, revision, tmp_path / "validation")
    assert report["status"] == "PASS"
    assert report["counts"]["passed"] == 1
    assert report["production_state_copied"] is False
    assert state.read_text(encoding="utf-8") == "retained production sentinel"


@pytest.mark.parametrize("revision,relative", [("main", False), ("a" * 40, True)])
def test_unsafe_validation_destinations_or_mutable_revisions_are_rejected(tmp_path, revision, relative):
    source = tmp_path / "production"
    source.mkdir()
    with pytest.raises(ValueError):
        validate(source, revision, source / "validation" if relative else tmp_path / "validation")
