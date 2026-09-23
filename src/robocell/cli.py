"""Command line entry point: `python -m robocell run --config ... --tasks N --seed S`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from robocell.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robocell", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a simulation and print a JSON summary")
    run.add_argument("--config", default="configs/default.toml")
    run.add_argument("--tasks", type=int, default=200)
    run.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "seed": args.seed,
        "tasks_generated": args.tasks,
        "arms": [a.name for a in config.arms],
        "zones": [z.name for z in config.zones],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
