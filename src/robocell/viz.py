"""matplotlib 3D viewer: record snapshots during a run, then animate or save frames.

Recording happens inside the simulation (a Cell observer, every `stride` ticks) and
never affects it. Rendering is a replay of the recording, so the same run can be shown
live (paced in real time) or written headless to PNG files with the Agg backend.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robocell.arm import ArmState

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from robocell.cell import Cell

NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
ARM_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:cyan"]
STATE_MARKERS = {
    ArmState.IDLE.value: ("o", "black"),
    ArmState.MOVING.value: ("o", "black"),
    ArmState.FAULT.value: ("X", "red"),
    ArmState.ESTOPPED.value: ("s", "red"),
}


@dataclass(frozen=True)
class ArmFrame:
    name: str
    base: tuple[float, float]
    shoulder: tuple[float, float, float]
    elbow: tuple[float, float, float]
    tip: tuple[float, float, float]
    state: str
    target: tuple[float, float, float] | None


@dataclass(frozen=True)
class Frame:
    tick: int
    time: float
    arms: tuple[ArmFrame, ...]
    zone_holders: tuple[str | None, ...]
    completed: int
    queued: int


@dataclass(frozen=True)
class Scene:
    """Static parts of the picture: zone boxes and plot bounds."""

    zone_names: tuple[str, ...]
    zone_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _xyz(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


class Recorder:
    """Cell observer that snapshots the cell every `stride` ticks."""

    def __init__(self, cell: Cell, stride: int = 10) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride
        self.frames: list[Frame] = []
        self.scene = scene_for(cell)
        cell.observers.append(self.observe)

    def observe(self, cell: Cell, tick: int) -> None:
        if tick % self.stride:
            return
        arms = []
        for arm in cell.arms:
            pose = arm.pose
            task_id = arm.task_id
            target = cell.scheduler.tasks[task_id].target if task_id is not None else None
            arms.append(
                ArmFrame(
                    name=arm.name,
                    base=arm.cfg.base,
                    shoulder=_xyz(pose.shoulder),
                    elbow=_xyz(pose.elbow),
                    tip=_xyz(pose.tip),
                    state=arm.state.value,
                    target=target,
                )
            )
        self.frames.append(
            Frame(
                tick=tick,
                time=tick * cell.clock.dt,
                arms=tuple(arms),
                zone_holders=tuple(cell.zones.holder(i) for i in range(len(cell.zones))),
                completed=cell.metrics.completed,
                queued=len(cell.scheduler.queued),
            )
        )


def scene_for(cell: Cell) -> Scene:
    reach = max(a.geom.reach for a in cell.arms)
    xs = [a.cfg.base[0] for a in cell.arms]
    ys = [a.cfg.base[1] for a in cell.arms]
    zmax = max(a.cfg.base_height for a in cell.arms) + reach
    return Scene(
        zone_names=tuple(z.name for z in cell.zones.zones),
        zone_boxes=tuple((_xyz(z.lo), _xyz(z.hi)) for z in cell.zones.zones),
        bounds=((min(xs) - reach, max(xs) + reach), (min(ys) - reach, max(ys) + reach), (0, zmax)),
    )


def _box_edges(lo: Sequence[float], hi: Sequence[float]) -> list[np.ndarray]:
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Corners differing in exactly one coordinate share an edge.
            if np.sum(corners[i] != corners[j]) == 1:
                edges.append(corners[[i, j]])
    return edges


def draw_frame(ax: Any, scene: Scene, frame: Frame) -> None:
    """Draw one frame onto a 3D axes (cleared first)."""
    ax.cla()
    (x0, x1), (y0, y1), (z0, z1) = scene.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    for (lo, hi), name, holder in zip(
        scene.zone_boxes, scene.zone_names, frame.zone_holders, strict=True
    ):
        color = "tab:red" if holder else "0.6"
        for edge in _box_edges(lo, hi):
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=color, lw=1.5 if holder else 0.8)
        label = f"{name}: {holder}" if holder else name
        ax.text(hi[0], hi[1], hi[2], label, color=color, fontsize=7)

    for i, arm in enumerate(frame.arms):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        pts = np.array([(*arm.base, 0.0), arm.shoulder, arm.elbow, arm.tip])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=3, solid_capstyle="round")
        marker, mcolor = STATE_MARKERS[arm.state]
        ax.scatter(*arm.tip, marker=marker, color=mcolor, s=30, depthshade=False)
        ax.text(arm.base[0], arm.base[1], 0.0, f"{arm.name} ({arm.state})", color=color, fontsize=8)
        if arm.target is not None:
            ax.scatter(*arm.target, marker="*", color=color, s=90, depthshade=False)

    ax.set_title(f"t = {frame.time:6.2f} s   done {frame.completed}   queued {frame.queued}")


def _figure() -> tuple[Figure, Any]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    return fig, ax


def save_frames(
    scene: Scene, frames: Sequence[Frame], out_dir: str | Path, max_frames: int | None = None
) -> list[Path]:
    """Render frames to PNG files (works headless with the Agg backend)."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chosen = list(frames)
    if max_frames is not None and len(chosen) > max_frames:
        # Evenly spread over the whole run, always including the last frame.
        idx = np.linspace(0, len(chosen) - 1, max_frames).round().astype(int)
        chosen = [chosen[i] for i in idx]
    fig, ax = _figure()
    paths = []
    try:
        for n, frame in enumerate(chosen):
            draw_frame(ax, scene, frame)
            path = out / f"frame_{n:05d}.png"
            fig.savefig(path, dpi=80)
            paths.append(path)
    finally:
        plt.close(fig)
    return paths


def animate(scene: Scene, frames: Sequence[Frame], speed: float = 1.0) -> None:
    """Replay in a window, paced so that sim time runs at `speed` x real time."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not frames:
        return
    backend = plt.get_backend().lower()
    if backend in NON_INTERACTIVE_BACKENDS:
        print(f"viz: backend {backend!r} cannot open a window; use --frames DIR", file=sys.stderr)
        return
    fig, ax = _figure()
    step_s = frames[1].time - frames[0].time if len(frames) > 1 else 0.1
    anim = FuncAnimation(
        fig,
        lambda i: draw_frame(ax, scene, frames[i]),
        frames=len(frames),
        interval=max(1.0, 1000.0 * step_s / speed),
        repeat=False,
    )
    plt.show()
    del anim
