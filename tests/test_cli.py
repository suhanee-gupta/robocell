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


def test_summary_has_all_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", DEFAULT, "--tasks", "20", "--seed", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    for key in (
        "finished",
        "tasks_completed",
        "tasks_rejected",
        "tasks_requeued",
        "wait_time_mean_s",
        "wait_time_p95_s",
        "arm_utilization",
        "zone_wait_time_s",
        "zone_violations",
        "faults",
        "estops",
    ):
        assert key in out, key
    assert out["finished"] is True
    assert out["tasks_completed"] + out["tasks_rejected"] == 20


def test_unfinished_run_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = Path(DEFAULT).read_text().replace("max_time_s = 3600.0", "max_time_s = 1.0")
    cfg = tmp_path / "short.toml"
    cfg.write_text(text)
    assert main(["run", "--config", str(cfg), "--tasks", "50", "--seed", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["finished"] is False


def test_negative_task_count_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", DEFAULT, "--tasks", "-1"]) == 2


def test_frames_flag_writes_pngs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "frames"
    args = ["run", "--config", DEFAULT, "--tasks", "10", "--seed", "2"]
    assert main([*args, "--frames", str(out), "--max-frames", "4"]) == 0
    with_frames = json.loads(capsys.readouterr().out)
    assert len(list(out.glob("frame_*.png"))) == 4
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == with_frames


@pytest.mark.parametrize(
    "extra", [["--max-frames", "0"], ["--speed", "0"], ["--frame-every", "-1"]]
)
def test_bad_viewer_flags_rejected(extra: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", DEFAULT, "--tasks", "1", *extra]) == 2
    assert "error" in capsys.readouterr().err


def test_home_in_zone_config_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression (review 2 #4): this used to crash with a traceback and exit code 1."""
    text = Path(DEFAULT).read_text() + (
        '\n[[zones]]\nname = "bad"\nmin = [-1.0, -1.0, 0.0]\nmax = [-0.2, -0.2, 1.5]\n'
    )
    cfg = tmp_path / "bad.toml"
    cfg.write_text(text)
    assert main(["run", "--config", str(cfg), "--tasks", "1"]) == 2
    assert "home pose is inside" in capsys.readouterr().err
