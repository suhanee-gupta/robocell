from pathlib import Path

import pytest

from robocell.arm import ArmState
from robocell.cell import build_cell
from robocell.config import CellConfig
from robocell.viz import Recorder, animate, save_frames

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_recorder_snapshots_every_stride(default_config: CellConfig) -> None:
    cell = build_cell(default_config, 20, seed=1)
    rec = Recorder(cell, stride=25)
    result = cell.run()
    assert [f.tick for f in rec.frames] == list(range(25, result.metrics.ticks + 1, 25))
    frame = rec.frames[5]
    assert len(frame.arms) == 4
    assert len(frame.zone_holders) == 5
    assert all(a.state in {s.value for s in ArmState} for a in frame.arms)
    assert any(a.target is not None for f in rec.frames for a in f.arms)


def test_recording_does_not_change_the_run(default_config: CellConfig) -> None:
    plain = build_cell(default_config, 40, seed=2).run().metrics.summary()
    cell = build_cell(default_config, 40, seed=2)
    Recorder(cell, stride=1)
    assert cell.run().metrics.summary() == plain


def test_save_frames_headless(default_config: CellConfig, tmp_path: Path) -> None:
    cell = build_cell(default_config, 15, seed=3)
    rec = Recorder(cell, stride=50)
    cell.run()
    # Force a faulted and an E-stopped arm into a frame to exercise every marker.
    arms = list(rec.frames[-1].arms)
    arms[0] = arms[0].__class__(**{**arms[0].__dict__, "state": ArmState.FAULT.value})
    arms[1] = arms[1].__class__(**{**arms[1].__dict__, "state": ArmState.ESTOPPED.value})
    frames = [
        *rec.frames,
        rec.frames[-1].__class__(**{**rec.frames[-1].__dict__, "arms": tuple(arms)}),
    ]
    paths = save_frames(rec.scene, frames, tmp_path / "out", max_frames=6)
    assert len(paths) == 6
    for path in paths:
        assert path.read_bytes()[:8] == PNG_MAGIC
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [p.name for p in paths]


def test_animate_under_agg_explains_instead_of_hanging(
    default_config: CellConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    cell = build_cell(default_config, 3, seed=1)
    rec = Recorder(cell, stride=100)
    cell.run()
    animate(rec.scene, rec.frames[:3], speed=100.0)
    assert "--frames" in capsys.readouterr().err
    animate(rec.scene, [])


def test_bad_stride_rejected(default_config: CellConfig) -> None:
    with pytest.raises(ValueError, match="stride"):
        Recorder(build_cell(default_config, 1, seed=1), stride=0)
