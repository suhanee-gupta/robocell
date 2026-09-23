import matplotlib

matplotlib.use("Agg")  # headless: tests must never open a window

import pytest
from helpers import DEFAULT_CONFIG

from robocell.config import CellConfig, load_config


@pytest.fixture
def default_config() -> CellConfig:
    return load_config(DEFAULT_CONFIG)
