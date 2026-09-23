"""Shared test configurations."""

import math
from pathlib import Path

from robocell.config import (
    ArmConfig,
    CellConfig,
    EStopConfig,
    FaultConfig,
    SimConfig,
    TaskGenConfig,
    ZoneConfig,
)

D = math.radians
DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.toml"


def arm_cfg(name: str, base: tuple[float, float], home_yaw_deg: float) -> ArmConfig:
    return ArmConfig(
        name=name,
        base=base,
        base_height=0.4,
        links=(0.6, 0.5),
        joint_min=(D(-170), D(-20), D(-150)),
        joint_max=(D(170), D(110), D(150)),
        max_vel=(D(90), D(70), D(100)),
        max_acc=(D(180), D(140), D(220)),
        home=(D(home_yaw_deg), D(60), D(-100)),
    )


def two_arm_config(zones: tuple[ZoneConfig, ...] = ()) -> CellConfig:
    """Arm 'a' at the origin and 'b' 1.6 m along +x, each facing away from the other at home."""
    return CellConfig(
        sim=SimConfig(),
        tasks=TaskGenConfig(),
        faults=FaultConfig(),
        estop=EStopConfig(),
        arms=(arm_cfg("a", (0.0, 0.0), 150.0), arm_cfg("b", (1.6, 0.0), -30.0)),
        zones=zones,
    )
