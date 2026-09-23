"""Cell configuration: TOML file -> validated, immutable dataclasses (angles in radians)."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Vec3 = tuple[float, float, float]


class ConfigError(ValueError):
    """Raised when a config file is malformed or physically inconsistent."""


@dataclass(frozen=True)
class SimConfig:
    tick_s: float = 0.01
    max_time_s: float = 3600.0


@dataclass(frozen=True)
class TaskGenConfig:
    arrival_rate_hz: float = 2.0
    x: tuple[float, float] = (-1.0, 1.0)
    y: tuple[float, float] = (-1.0, 1.0)
    z: tuple[float, float] = (0.0, 1.0)
    priority_weights: tuple[float, ...] = (1.0,)


@dataclass(frozen=True)
class FaultConfig:
    probability: float = 0.0
    recovery_s: float = 1.0


@dataclass(frozen=True)
class EStopConfig:
    global_at_s: tuple[float, ...] = ()
    reset_after_s: float = 1.0


@dataclass(frozen=True)
class ArmConfig:
    name: str
    base: tuple[float, float]
    base_height: float
    links: tuple[float, float]
    joint_min: Vec3
    joint_max: Vec3
    max_vel: Vec3
    max_acc: Vec3
    home: Vec3


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    min: Vec3
    max: Vec3


@dataclass(frozen=True)
class CellConfig:
    sim: SimConfig
    tasks: TaskGenConfig
    faults: FaultConfig
    estop: EStopConfig
    arms: tuple[ArmConfig, ...]
    zones: tuple[ZoneConfig, ...]


def _floats(value: Any, n: int, what: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != n:
        raise ConfigError(f"{what}: expected a list of {n} numbers, got {value!r}")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        raise ConfigError(f"{what}: expected numbers, got {value!r}")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ConfigError(f"{what}: values must be finite, got {value!r}")
    return out


def _vec3(value: Any, what: str, *, degrees: bool = False) -> Vec3:
    a, b, c = _floats(value, 3, what)
    if degrees:
        return (math.radians(a), math.radians(b), math.radians(c))
    return (a, b, c)


def _pair(value: Any, what: str) -> tuple[float, float]:
    a, b = _floats(value, 2, what)
    return (a, b)


def _number(table: dict[str, Any], key: str, default: float, what: str) -> float:
    value = table.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ConfigError(f"{what}.{key}: expected a finite number, got {value!r}")
    return float(value)


def _positive(value: float, what: str) -> float:
    if value <= 0:
        raise ConfigError(f"{what}: must be > 0, got {value}")
    return value


def _parse_arm(raw: dict[str, Any], i: int) -> ArmConfig:
    what = f"arms[{i}]"
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{what}.name: expected a non-empty string")
    what = f"arm {name!r}"
    links = _pair(raw.get("links"), f"{what}.links")
    arm = ArmConfig(
        name=name,
        base=_pair(raw.get("base"), f"{what}.base"),
        base_height=_number(raw, "base_height", 0.0, what),
        links=(_positive(links[0], f"{what}.links[0]"), _positive(links[1], f"{what}.links[1]")),
        joint_min=_vec3(raw.get("joint_min_deg"), f"{what}.joint_min_deg", degrees=True),
        joint_max=_vec3(raw.get("joint_max_deg"), f"{what}.joint_max_deg", degrees=True),
        max_vel=_vec3(raw.get("max_vel_deg"), f"{what}.max_vel_deg", degrees=True),
        max_acc=_vec3(raw.get("max_acc_deg"), f"{what}.max_acc_deg", degrees=True),
        home=_vec3(raw.get("home_deg"), f"{what}.home_deg", degrees=True),
    )
    for j in range(3):
        if arm.joint_min[j] >= arm.joint_max[j]:
            raise ConfigError(f"{what}: joint {j} min must be < max")
        _positive(arm.max_vel[j], f"{what}.max_vel_deg[{j}]")
        _positive(arm.max_acc[j], f"{what}.max_acc_deg[{j}]")
        if not arm.joint_min[j] <= arm.home[j] <= arm.joint_max[j]:
            raise ConfigError(f"{what}: home joint {j} is outside its limits")
    return arm


def _parse_zone(raw: dict[str, Any], i: int) -> ZoneConfig:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"zones[{i}].name: expected a non-empty string")
    zone = ZoneConfig(
        name=name,
        min=_vec3(raw.get("min"), f"zone {name!r}.min"),
        max=_vec3(raw.get("max"), f"zone {name!r}.max"),
    )
    if not all(lo < hi for lo, hi in zip(zone.min, zone.max, strict=True)):
        raise ConfigError(f"zone {name!r}: min must be < max on every axis")
    return zone


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _tables(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ConfigError(f"[[{key}]] must be an array of tables")
    return value


def parse_config(data: dict[str, Any]) -> CellConfig:
    sim_raw = _table(data, "sim")
    sim = SimConfig(
        tick_s=_positive(_number(sim_raw, "tick_s", 0.01, "sim"), "sim.tick_s"),
        max_time_s=_positive(_number(sim_raw, "max_time_s", 3600.0, "sim"), "sim.max_time_s"),
    )

    t_raw = _table(data, "tasks")
    weights = t_raw.get("priority_weights", [1.0])
    if not isinstance(weights, list) or not weights:
        raise ConfigError("tasks.priority_weights: expected a non-empty list")
    weight_values = _floats(weights, len(weights), "tasks.priority_weights")
    if any(w < 0 for w in weight_values) or sum(weight_values) <= 0:
        raise ConfigError("tasks.priority_weights: must be >= 0 with a positive sum")
    tasks = TaskGenConfig(
        arrival_rate_hz=_positive(
            _number(t_raw, "arrival_rate_hz", 2.0, "tasks"), "tasks.arrival_rate_hz"
        ),
        x=_pair(t_raw.get("x", [-1.0, 1.0]), "tasks.x"),
        y=_pair(t_raw.get("y", [-1.0, 1.0]), "tasks.y"),
        z=_pair(t_raw.get("z", [0.0, 1.0]), "tasks.z"),
        priority_weights=weight_values,
    )
    for axis, (lo, hi) in (("x", tasks.x), ("y", tasks.y), ("z", tasks.z)):
        if lo > hi:
            raise ConfigError(f"tasks.{axis}: min must be <= max")

    f_raw = _table(data, "faults")
    faults = FaultConfig(
        probability=_number(f_raw, "probability", 0.0, "faults"),
        recovery_s=_positive(_number(f_raw, "recovery_s", 1.0, "faults"), "faults.recovery_s"),
    )
    if not 0.0 <= faults.probability <= 1.0:
        raise ConfigError("faults.probability must be in [0, 1]")

    e_raw = _table(data, "estop")
    at = e_raw.get("global_at_s", [])
    if not isinstance(at, list):
        raise ConfigError("estop.global_at_s: expected a list")
    estop = EStopConfig(
        global_at_s=tuple(sorted(_floats(at, len(at), "estop.global_at_s"))),
        reset_after_s=_positive(
            _number(e_raw, "reset_after_s", 1.0, "estop"), "estop.reset_after_s"
        ),
    )

    arms = tuple(_parse_arm(raw, i) for i, raw in enumerate(_tables(data, "arms")))
    if not arms:
        raise ConfigError("config must define at least one [[arms]] entry")
    if len({a.name for a in arms}) != len(arms):
        raise ConfigError("arm names must be unique")
    zones = tuple(_parse_zone(raw, i) for i, raw in enumerate(_tables(data, "zones")))
    if len({z.name for z in zones}) != len(zones):
        raise ConfigError("zone names must be unique")

    return CellConfig(sim=sim, tasks=tasks, faults=faults, estop=estop, arms=arms, zones=zones)


def load_config(path: str | Path) -> CellConfig:
    with Path(path).open("rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    return parse_config(data)
