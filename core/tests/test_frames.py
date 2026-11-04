import math

import numpy as np

from aortaplan_core.geometry.curves import cumulative_arc_length
from aortaplan_core.geometry.frames import initial_normal, rotation_minimizing_frames


def check_orthonormal(t, n, b):
    for v in (t, n, b):
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-9)
    assert np.max(np.abs(np.einsum("ij,ij->i", t, n))) < 1e-9
    np.testing.assert_allclose(np.cross(t, n), b, atol=1e-12)


def test_initial_normal_is_projection_of_reference():
    n = initial_normal([0.0, 0.0, 1.0], [1.0, 1.0, 5.0])
    np.testing.assert_allclose(n, [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0])


def test_initial_normal_falls_back_when_reference_is_parallel():
    n = initial_normal([0.0, 0.0, 1.0], [0.0, 0.0, 2.0])
    assert abs(np.dot(n, [0, 0, 1])) < 1e-12
    np.testing.assert_allclose(np.linalg.norm(n), 1.0)


def test_straight_line_frame_is_constant():
    pts = np.stack([np.zeros(50), np.zeros(50), np.linspace(0, 49, 50)], axis=1)
    t, n, b = rotation_minimizing_frames(pts, reference_normal=[0.0, 1.0, 0.0])
    check_orthonormal(t, n, b)
    np.testing.assert_allclose(n, np.tile([0.0, 1.0, 0.0], (50, 1)), atol=1e-12)


def test_planar_circle_keeps_out_of_plane_normal():
    th = np.linspace(0.0, 1.5 * math.pi, 2000)
    pts = np.stack([30 * np.cos(th), 30 * np.sin(th), np.zeros_like(th)], axis=1)
    t, n, b = rotation_minimizing_frames(pts, reference_normal=[0.0, 0.0, 1.0])
    check_orthonormal(t, n, b)
    np.testing.assert_allclose(n, np.tile([0.0, 0.0, 1.0], (len(th), 1)), atol=1e-9)


def test_helix_frame_rotates_against_frenet_by_torsion():
    """For a rotation-minimizing frame the angle to the Frenet normal changes as -torsion * s."""
    R, c = 20.0, 8.0
    tt = np.linspace(0.0, 4.0 * math.pi, 20001)
    pts = np.stack([R * np.cos(tt), R * np.sin(tt), c * tt], axis=1)
    t, n, b = rotation_minimizing_frames(pts)
    check_orthonormal(t, n, b)
    frenet_n = np.stack([-np.cos(tt), -np.sin(tt), np.zeros_like(tt)], axis=1)
    frenet_b = np.stack([c * np.sin(tt), -c * np.cos(tt), np.full_like(tt, R)], axis=1) / math.hypot(R, c)
    phi = np.unwrap(np.arctan2(np.einsum("ij,ij->i", n, frenet_b), np.einsum("ij,ij->i", n, frenet_n)))
    s = cumulative_arc_length(pts)
    torsion = c / (R**2 + c**2)
    np.testing.assert_allclose(phi - phi[0], -torsion * s, atol=2e-3)
