import math

import numpy as np
import pytest

from aortaplan_core.centerline.navigation import ANTERIOR, SUPERIOR, CurvePath

LEFT = np.array([-1.0, 0.0, 0.0])


def line(start, end, n=200):
    return np.linspace(np.asarray(start, float), np.asarray(end, float), n)


def arch(R=40.0, leg=60.0, n=2000):
    """Sagittal arch: up the anterior leg, over the top, down the posterior leg."""
    theta = np.linspace(0.0, math.pi, n)
    arc = np.stack([np.zeros_like(theta), R * np.cos(theta), R * np.sin(theta)], axis=1)
    up_leg = line((0, R, -leg), (0, R, 0), 200)[:-1]
    down_leg = line((0, -R, 0), (0, -R, -leg), 200)[1:]
    return np.concatenate([up_leg, arc, down_leg]), leg, R


def check_frame(frame):
    for v in (frame.x_axis, frame.y_axis, frame.normal):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)
    assert abs(np.dot(frame.x_axis, frame.y_axis)) < 1e-9
    np.testing.assert_allclose(np.cross(frame.x_axis, frame.y_axis), frame.normal, atol=1e-12)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_vertical_vessel_looks_like_axial_from_feet(direction):
    path = CurvePath(line((0, 0, 100 * -direction), (0, 0, 100 * direction)))
    for s in (0.0, 50.0, path.length):
        f = path.cross_section_frame(s)
        check_frame(f)
        np.testing.assert_allclose(f.y_axis, ANTERIOR, atol=1e-9)  # anterior up
        np.testing.assert_allclose(f.x_axis, LEFT, atol=1e-9)  # patient's left on the right
        np.testing.assert_allclose(f.normal, -SUPERIOR, atol=1e-9)  # viewer below, looking up


def test_oblique_vessel_keeps_anatomical_orientation():
    # Iliac-like: downwards, to the patient's left and slightly anterior.
    path = CurvePath(line((0, 0, 0), (-40, 15, -100)))
    f = path.cross_section_frame(30.0)
    check_frame(f)
    t = path.tangent_at(30.0)
    assert abs(np.dot(f.y_axis, t)) < 1e-9 and abs(np.dot(f.x_axis, t)) < 1e-9
    assert np.dot(f.y_axis, ANTERIOR) > 0.9
    assert np.dot(f.x_axis, LEFT) > 0.9


def test_point_and_tangent_on_circle():
    th = np.linspace(0, math.pi, 3000)
    path = CurvePath(np.stack([30 * np.cos(th), 30 * np.sin(th), np.zeros_like(th)], axis=1), step_mm=0.25)
    assert path.length == pytest.approx(math.pi * 30, rel=1e-4)
    p = path.point_at(path.length / 2)
    np.testing.assert_allclose(p, [0, 30, 0], atol=1e-2)
    np.testing.assert_allclose(path.tangent_at(path.length / 2), [-1, 0, 0], atol=1e-3)
    np.testing.assert_allclose(path.point_at(-5.0), path.points[0])
    np.testing.assert_allclose(path.point_at(1e6), path.points[-1])


def test_nearest_arc_length():
    path = CurvePath(line((0, 0, 0), (0, 0, -100)))
    s, d = path.nearest_arc_length([3.0, 4.0, -37.0])
    assert s == pytest.approx(37.0, abs=1e-9)
    assert d == pytest.approx(5.0, abs=1e-9)
    s, d = path.nearest_arc_length([0.0, 0.0, 20.0])  # beyond the proximal end
    assert s == 0.0 and d == pytest.approx(20.0)


def test_arch_orientation_is_continuous_inside_the_arch():
    pts, leg, R = arch()
    path = CurvePath(pts, step_mm=0.5)
    jumps = path.frames_continuity(0.5)
    big = np.flatnonzero(jumps > 20.0)
    # Exactly one jump, at the proximal (ascending) end of the degenerate zone (theta = 60 deg).
    assert len(big) == 1
    s_jump = big[0] * 0.5
    assert s_jump == pytest.approx(leg + R * math.radians(60.0), abs=2.0)
    others = np.delete(jumps, big)
    assert others.max() < 2.0


def test_arch_descending_and_top():
    pts, leg, R = arch()
    path = CurvePath(pts)
    descending = path.cross_section_frame(path.length - 10.0)
    np.testing.assert_allclose(descending.y_axis, ANTERIOR, atol=1e-6)
    np.testing.assert_allclose(descending.x_axis, LEFT, atol=1e-6)
    top = path.cross_section_frame(leg + math.pi * R / 2)
    check_frame(top)
    # Carried from the descending aorta: up points to the arch centre (lesser curvature),
    # and the patient's left stays on the right of the screen.
    np.testing.assert_allclose(top.y_axis, -SUPERIOR, atol=1e-3)
    np.testing.assert_allclose(top.x_axis, LEFT, atol=1e-3)
    ascending = path.cross_section_frame(10.0)
    np.testing.assert_allclose(ascending.y_axis, ANTERIOR, atol=1e-6)
    np.testing.assert_allclose(ascending.x_axis, LEFT, atol=1e-6)


def test_vessel_along_anterior_axis_uses_transport():
    path = CurvePath(line((0, -50, 0), (0, 50, 0)))
    f = path.cross_section_frame(20.0)
    check_frame(f)
    np.testing.assert_allclose(np.abs(f.y_axis), [0, 0, 1], atol=1e-9)


def test_horizontal_vessel_has_deterministic_view():
    path = CurvePath(line((0, 0, 0), (-60, 0, 0)))  # renal-like, towards patient's left
    f1 = path.cross_section_frame(10.0)
    f2 = path.cross_section_frame(50.0)
    check_frame(f1)
    np.testing.assert_allclose(f1.y_axis, ANTERIOR, atol=1e-9)
    np.testing.assert_allclose(f1.x_axis, f2.x_axis, atol=1e-9)


def test_longitudinal_frame():
    path = CurvePath(line((0, 0, 100), (0, 0, -100)))  # abdominal aorta, downwards
    f0 = path.longitudinal_frame(50.0, 0.0)
    check_frame(f0)
    np.testing.assert_allclose(f0.y_axis, SUPERIOR, atol=1e-9)  # head up
    np.testing.assert_allclose(f0.x_axis, LEFT, atol=1e-9)  # coronal-like, radiological
    f90 = path.longitudinal_frame(50.0, 90.0)
    check_frame(f90)
    np.testing.assert_allclose(f90.x_axis, ANTERIOR, atol=1e-9)
    assert abs(np.dot(f90.normal, path.tangent_at(50.0))) < 1e-9  # plane contains the tangent
    np.testing.assert_allclose(f90.origin, path.point_at(50.0))


def test_slice_to_ras_matrix():
    path = CurvePath(line((0, 0, 100), (0, 0, -100)))
    f = path.cross_section_frame(25.0)
    m = f.slice_to_ras()
    np.testing.assert_allclose(m[:3, 3], [0, 0, 75], atol=1e-9)
    np.testing.assert_allclose(m @ [10, 0, 0, 1], [*(f.origin + 10 * f.x_axis), 1])
    np.testing.assert_allclose(f.screen_to_ras(0, 5), f.origin + 5 * ANTERIOR)


def test_rejects_degenerate_input():
    with pytest.raises(ValueError):
        CurvePath(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        CurvePath(np.zeros((5, 2)))


def test_cross_sections_of_torus_phantom_are_centred(torus_phantom):
    truth = torus_phantom.truth
    path = CurvePath(truth.centerline_points)
    r = truth.layer("lumen").semi_axis_n[0]
    g = torus_phantom.volume.geometry
    for s in np.linspace(20.0, path.length - 20.0, 25):
        f = path.cross_section_frame(s)
        pts = np.array([f.origin, *(f.screen_to_ras(dx, dy) for dx, dy in [(r - 1, 0), (0, r - 1), (-(r - 1), 0)])])
        outside = np.array([f.screen_to_ras(r + 1.5, 0), f.screen_to_ras(0, -(r + 1.5))])
        ijk = g.ras_to_ijk(np.vstack([pts, outside]))
        if np.any(ijk < 1) or np.any(ijk > np.array(g.dimensions) - 2):
            continue  # near the volume boundary
        assert np.all(torus_phantom.volume.sample_linear(pts) > 195)
        assert np.all(torus_phantom.volume.sample_linear(outside) < 195)


def test_closest_point_to_ray():
    path = CurvePath(line((0, 0, 100), (0, 0, -100)))
    hit = path.closest_arc_length_to_ray([0, -500, 30.2], [0, 1, 0], max_distance_mm=3.0)
    assert hit is not None
    assert hit[0] == pytest.approx(69.8, abs=0.3) and hit[1] < 0.3
    assert path.closest_arc_length_to_ray([10, -500, 0], [0, 1, 0], max_distance_mm=3.0) is None  # misses
    assert path.closest_arc_length_to_ray([0, 500, 0], [0, 1, 0], max_distance_mm=3.0) is None  # behind
