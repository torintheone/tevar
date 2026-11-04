import math

import numpy as np
import pytest

from aortaplan_core.geometry.curves import (
    cumulative_arc_length,
    interpolate_at_arc_length,
    polyline_length,
    resample_polyline,
    unit_tangents,
)


def circle(radius, n):
    t = np.linspace(0.0, 2.0 * math.pi, n + 1)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)


def test_length_of_straight_polyline():
    pts = np.array([[0, 0, 0], [3, 4, 0], [3, 4, 12]], dtype=float)
    assert polyline_length(pts) == pytest.approx(17.0)
    np.testing.assert_allclose(cumulative_arc_length(pts), [0.0, 5.0, 17.0])


def test_circle_length_converges():
    assert polyline_length(circle(20.0, 4000)) == pytest.approx(2 * math.pi * 20.0, rel=1e-6)


def test_resample_keeps_ends_and_step():
    pts = np.array([[0, 0, 0], [10, 0, 0], [10, 7, 0]], dtype=float)
    res = resample_polyline(pts, 0.5)
    np.testing.assert_allclose(res[0], pts[0])
    np.testing.assert_allclose(res[-1], pts[-1])
    assert len(res) == 35  # 17 mm / 0.5 mm + 1
    assert np.linalg.norm(np.diff(res, axis=0), axis=1).max() <= 0.5 + 1e-9
    # every resampled point lies on one of the two segments
    on_first = (np.abs(res[:, 1]) < 1e-12) & (res[:, 0] <= 10 + 1e-12)
    on_second = np.abs(res[:, 0] - 10) < 1e-12
    assert np.all(on_first | on_second)


def test_interpolate_clamps_to_curve():
    pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    np.testing.assert_allclose(interpolate_at_arc_length(pts, [-1.0, 2.5, 99.0]), [[0, 0, 0], [2.5, 0, 0], [10, 0, 0]])


def test_tangents_of_circle_are_perpendicular_to_radius():
    pts = circle(15.0, 2000)[:-1]
    t = unit_tangents(pts)
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert np.max(np.abs(np.einsum("ij,ij->i", t, radial))) < 1e-5
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0)


def test_repeated_points_rejected():
    with pytest.raises(ValueError):
        unit_tangents(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float))
