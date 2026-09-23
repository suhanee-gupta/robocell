import json
from pathlib import Path

import pytest

from robocell.cli import main

DEFAULT = str(Path(__file__).parent.parent / "configs" / "default.toml")


def test_run_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", DEFAULT, "--tasks", "5", "--seed", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tasks_generated"] == 5
    assert out["seed"] == 3


def test_missing_config_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", "does/not/exist.toml"]) == 2
    assert "error" in capsys.readouterr().err
