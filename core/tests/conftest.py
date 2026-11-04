"""Shared fixtures. Phantoms are expensive to build, so they are cached per test session."""

import pytest

from aortaplan_core.phantoms import elliptical_tube, helix, straight_tube, torus_arc


@pytest.fixture(scope="session")
def straight_phantom():
    return straight_tube(length_mm=40.0)


@pytest.fixture(scope="session")
def oblique_phantom():
    return straight_tube(
        name="straight_oblique", length_mm=40.0, direction=(0.3, -0.4, 1.0), spacing=(0.7, 0.7, 1.25)
    )


@pytest.fixture(scope="session")
def elliptical_phantom():
    return elliptical_tube(length_mm=40.0)


@pytest.fixture(scope="session")
def torus_phantom():
    return torus_arc(leg_length_mm=10.0)


@pytest.fixture(scope="session")
def helix_phantom():
    return helix(turns=1.0, helix_radius_mm=20.0, pitch_mm=50.0, diameter_mm=8.0)
