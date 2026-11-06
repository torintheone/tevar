import math

import numpy as np
import pytest

from aortaplan_core.crosssection.metrics import (
    ContourMetrics,
    contour_metrics,
    contour_metrics_3d,
    convex_hull,
    feret_diameters,
    fit_plane,
    plane_coordinates,
    signed_area,
)
from aortaplan_core.phantoms import ellipse_reference_metrics


def ellipse(a, b, n=1440, rotation_deg=0.0, center=(0.0, 0.0)):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    p = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
    r = math.radians(rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return p @ rot.T + np.asarray(center)


@pytest.mark.parametrize("a,b,rotation", [(10, 10, 0), (15, 10, 0), (15, 10, 37), (30, 27, 110)])
def test_ellipse_metrics(a, b, rotation):
    m = contour_metrics(ellipse(a, b, rotation_deg=rotation, center=(3.0, -7.0)))
    ref = ellipse_reference_metrics(a, b)
    for key, value in m.values().items():
        assert value == pytest.approx(ref[key], rel=2e-4, abs=2e-3), key
    assert m.centroid == pytest.approx([3.0, -7.0], abs=1e-6)
    # The drawn segments have the reported lengths.
    assert np.linalg.norm(np.diff(m.max_feret_segment, axis=0)) == pytest.approx(m.max_diameter_mm)
    assert np.linalg.norm(np.diff(m.min_feret_segment, axis=0)) == pytest.approx(m.min_diameter_mm)


def test_square_feret():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    min_w, min_seg, max_d, max_seg = feret_diameters(square)
    assert min_w == pytest.approx(10.0)
    assert max_d == pytest.approx(10.0 * math.sqrt(2))
    d = min_seg[1] - min_seg[0]
    assert abs(d[0]) < 1e-9 or abs(d[1]) < 1e-9  # perpendicular to a side
    m = contour_metrics(square)
    assert m.area_mm2 == pytest.approx(100.0)
    assert m.perimeter_mm == pytest.approx(40.0)
    assert m.mean_diameter_mm == pytest.approx(0.5 * (10 + 10 * math.sqrt(2)))
    assert m.area_equivalent_diameter_mm == pytest.approx(2 * math.sqrt(100 / math.pi))
    assert m.perimeter_equivalent_diameter_mm == pytest.approx(40 / math.pi)


def test_orientation_and_closing_point_do_not_matter():
    p = ellipse(12, 8, n=200)
    m1 = contour_metrics(p)
    m2 = contour_metrics(p[::-1])  # clockwise
    m3 = contour_metrics(np.vstack([p, p[:1]]))  # explicitly closed
    for key in m1.values():
        assert m1.values()[key] == pytest.approx(m2.values()[key])
        assert m1.values()[key] == pytest.approx(m3.values()[key])
    assert signed_area(p) > 0 > signed_area(p[::-1])


def test_non_convex_contour_uses_hull_for_feret_only():
    # A "C"-shaped contour: area of the polygon itself, calipers of its convex hull.
    t = np.linspace(0.3, 2 * math.pi - 0.3, 300)
    outer = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
    inner = np.stack([6 * np.cos(t[::-1]), 6 * np.sin(t[::-1])], axis=1)
    c = np.vstack([outer, inner])
    m = contour_metrics(c)
    hull = convex_hull(c)
    assert m.area_mm2 < abs(signed_area(hull))
    assert m.max_diameter_mm == pytest.approx(20.0, abs=0.05)
    assert m.min_diameter_mm <= 20.0


def test_metrics_in_an_oblique_plane():
    origin = np.array([10.0, -20.0, 5.0])
    normal = np.array([0.3, -0.5, 0.8])
    normal /= np.linalg.norm(normal)
    u = np.cross(normal, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p2 = ellipse(14, 9, rotation_deg=20)
    p3 = origin + p2[:, :1] * u + p2[:, 1:] * v
    m = contour_metrics_3d(p3, origin, u, v)
    ref = ellipse_reference_metrics(14, 9)
    assert m.max_diameter_mm == pytest.approx(ref["max_diameter_mm"], rel=1e-4)
    assert m.min_diameter_mm == pytest.approx(ref["min_diameter_mm"], rel=1e-4)
    for point in np.vstack([m.max_feret_segment, m.min_feret_segment, m.centroid[np.newaxis]]):
        assert abs(np.dot(point - origin, normal)) < 1e-9  # drawn in the contour's plane
    assert plane_coordinates(p3, origin, u, v) == pytest.approx(p2)
    centre, fu, fv, fn = fit_plane(p3, normal)
    assert centre == pytest.approx(origin, abs=1e-9)
    assert np.dot(fn, normal) == pytest.approx(1.0)
    assert abs(np.dot(fu, fn)) < 1e-12 and abs(np.dot(fv, fn)) < 1e-12


def test_metrics_round_trip():
    m = contour_metrics(ellipse(11, 7))
    d = m.to_dict()
    back = ContourMetrics.from_dict(d)
    for key, value in m.values().items():
        assert back.values()[key] == pytest.approx(value, abs=1e-4)
    assert back.max_feret_segment == pytest.approx(m.max_feret_segment, abs=1e-4)


def test_degenerate_contours_are_rejected():
    with pytest.raises(ValueError):
        contour_metrics(np.array([[0.0, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError):
        contour_metrics(np.zeros((5, 3)))
