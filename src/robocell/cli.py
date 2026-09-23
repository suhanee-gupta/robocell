"""Command line entry point: `python -m robocell run --config ... --tasks N --seed S`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from robocell.cell import build_cell
from robocell.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robocell", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a simulation and print a JSON summary")
    run.add_argument("--config", default="configs/default.toml")
    run.add_argument("--tasks", type=int, default=200)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--viz", action="store_true", help="replay the run in a 3D window")
    run.add_argument("--frames", metavar="DIR", help="save 3D frames as PNG files to DIR")
    run.add_argument("--frame-every", type=float, default=0.5, metavar="S", help="sim seconds")
    run.add_argument("--max-frames", type=int, default=120, help="cap for --frames")
    run.add_argument("--speed", type=float, default=1.0, help="--viz playback speed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    problems = [
        msg
        for bad, msg in (
            (args.tasks < 0, "--tasks must be >= 0"),
            (args.frame_every <= 0, "--frame-every must be > 0"),
            (args.max_frames < 1, "--max-frames must be >= 1"),
            (args.speed <= 0, "--speed must be > 0"),
        )
        if bad
    ]
    if problems:
        print(f"error: {'; '.join(problems)}", file=sys.stderr)
        return 2
    try:
        cell = build_cell(config, args.tasks, args.seed)
    except ConfigError as exc:  # e.g. a home pose inside a shared zone
        print(f"error: {exc}", file=sys.stderr)
        return 2
    recorder = None
    if args.viz or args.frames:
        from robocell.viz import Recorder

        recorder = Recorder(cell, stride=max(1, cell.clock.ticks(args.frame_every)))
    result = cell.run()
    summary = {"seed": args.seed, "finished": result.finished, **result.metrics.summary()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if recorder is not None:
        from robocell import viz

        if args.frames:
            paths = viz.save_frames(recorder.scene, recorder.frames, args.frames, args.max_frames)
            print(f"wrote {len(paths)} frames to {args.frames}", file=sys.stderr)
        if args.viz:
            viz.animate(recorder.scene, recorder.frames, speed=args.speed)
    return 0 if result.finished else 1
