"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ktp_scenario as S  # noqa: E402


@pytest.fixture
def xor():
    """XOR mit vier Punkten a = (1, 1), b = (−1, −1) mit y = +1, c = (1, −1), d = (−1, 1) mit y = −1; von Hand: Polynom 2 gibt 4 Korrekturen, α = (1, 1, 1, 1)."""
    return S.xor_instance()


@pytest.fixture
def two_points():
    """Zwei Punkte x = 0 (y = −1) und x = 1 (y = +1) auf der Zahlengeraden (als 2D-Punkte mit zweiter Koordinate 0): Marge 1 mit freiem b, 2/√5 in (φ, 1)."""
    import numpy as np

    return np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([-1.0, 1.0])
