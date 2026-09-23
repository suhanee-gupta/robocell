import math
from pathlib import Path
from typing import Any

import pytest

from robocell.config import ConfigError, load_config, parse_config

DEFAULT = Path(__file__).parent.parent / "configs" / "default.toml"


def minimal() -> dict[str, Any]:
    return {
        "arms": [
            {
                "name": "a",
                "base": [0.0, 0.0],
                "base_height": 0.4,
                "links": [0.6, 0.5],
                "joint_min_deg": [-170, -20, -150],
                "joint_max_deg": [170, 110, 150],
                "max_vel_deg": [90, 70, 100],
                "max_acc_deg": [180, 140, 220],
                "home_deg": [0, 60, -100],
            }
        ],
        "zones": [{"name": "z", "min": [0, 0, 0], "max": [1, 1, 1]}],
    }


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT)
    assert len(cfg.arms) == 4
    assert {z.name for z in cfg.zones} == {"center", "south", "north", "west", "east"}
    assert cfg.sim.tick_s == pytest.approx(0.01)
    assert cfg.arms[0].joint_max[0] == pytest.approx(math.radians(170))


def test_minimal_uses_defaults() -> None:
    cfg = parse_config(minimal())
    assert cfg.faults.probability == 0.0
    assert cfg.estop.global_at_s == ()
    assert cfg.arms[0].home[1] == pytest.approx(math.radians(60))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("arms", 0, "links"), [0.6, -0.5], "must be > 0"),
        (("arms", 0, "joint_min_deg"), [0, 200, 0], "min must be < max"),
        (("arms", 0, "home_deg"), [0, 150, 0], "outside its limits"),
        (("arms", 0, "max_vel_deg"), [1, 0, 1], "must be > 0"),
        (("arms", 0, "base"), [0.0], "list of 2"),
        (("zones", 0, "max"), [1, -1, 1], "min must be < max"),
    ],
)
def test_invalid_values_rejected(path: tuple[Any, ...], value: Any, match: str) -> None:
    data = minimal()
    target: Any = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=match):
        parse_config(data)


def test_duplicate_arm_names_rejected() -> None:
    data = minimal()
    data["arms"].append(dict(data["arms"][0]))
    with pytest.raises(ConfigError, match="unique"):
        parse_config(data)


def test_no_arms_rejected() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        parse_config({"arms": []})


def test_bad_toml_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[sim\n")
    with pytest.raises(ConfigError):
        load_config(bad)
