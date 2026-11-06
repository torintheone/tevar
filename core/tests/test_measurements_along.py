"""Measurements along the centerline: diameter profile, maximum, markers, lengths, angles.

SPEC stage 4 acceptance for lengths along the axis: error <= 1 mm.
"""

import math

import numpy as np
import pytest

from aortaplan_core.centerline.angles import angle_between, axis, neck_angles
from aortaplan_core.centerline.extraction import extract_centerlines
from aortaplan_core.centerline.markers import (
    MARKER_PRESETS,
    Marker,
    distance,
    distance_table,
    divergence,
    marker_distance,
    marker_on_path,
    sort_along,
    tortuosity_index,
)
from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.crosssection import LUMEN, OUTER, DiameterProfile, diameter_profile, find_maximum_diameter
from aortaplan_core.crosssection.lengths import curvature_lengths
from aortaplan_core.i18n import tr
from aortaplan_core.lumen import Cancelled, segment_lumen

LENGTH_LIMIT_MM = 1.0




@pytest.fixture(scope="module")
def aneurysm_profile(aneurysm):
    path = CurvePath(aneurysm.truth.centerline_points)
    e = aneurysm.truth.expected
    return path, diameter_profile(aneurysm.volume, path, 2.0, e["volume_start_s_mm"] + 2, e["volume_end_s_mm"] - 2)


def test_profile_follows_the_aneurysm(aneurysm, aneurysm_profile):
    path, profile = aneurysm_profile
    e = aneurysm.truth.expected
    neck = profile.arc_length_mm < e["sac_start_s_mm"] - 2
    assert profile.values(LUMEN, "max_diameter_mm")[neck] == pytest.approx(22.0, abs=0.35)
    assert profile.values(OUTER, "max_diameter_mm")[neck] == pytest.approx(25.0, abs=0.35)
    s, value = profile.maximum(OUTER)
    assert s == pytest.approx(e["max_outer_s_mm"], abs=4.0)
    assert value == pytest.approx(60.0, abs=0.35)
    # Restricted range: the maximum lies at its end.
    s2, _ = profile.maximum(OUTER, s_min=20.0, s_max=70.0)
    assert s2 == pytest.approx(70.0, abs=2.0)
    back = DiameterProfile.from_dict(profile.to_dict())
    assert back.values(OUTER, "max_diameter_mm") == pytest.approx(profile.values(OUTER, "max_diameter_mm"), abs=1e-3)


def test_find_maximum_between_markers(aneurysm):
    path = CurvePath(aneurysm.truth.centerline_points)
    e = aneurysm.truth.expected
    result = find_maximum_diameter(aneurysm.volume, path, e["sac_start_s_mm"], e["sac_end_s_mm"])
    assert result.arc_length_mm == pytest.approx(e["max_outer_s_mm"], abs=3.0)
    assert result.value_mm == pytest.approx(60.0, abs=0.3)
    assert result.measurement.outer.metrics().min_diameter_mm == pytest.approx(54.0, abs=0.3)
    lumen = find_maximum_diameter(aneurysm.volume, path, e["sac_start_s_mm"], e["sac_end_s_mm"], layer=LUMEN)
    assert lumen.value_mm == pytest.approx(30.0, abs=0.3)
    assert lumen.measurement.outer is None


def test_profile_can_be_cancelled(aneurysm):
    path = CurvePath(aneurysm.truth.centerline_points)
    with pytest.raises(Cancelled):
        diameter_profile(aneurysm.volume, path, 1.0, progress=lambda fraction, step: fraction < 0.2)


# ------------------------------------------------------------------ markers
def straight(length=200.0, direction=(0.0, 0.0, -1.0)):
    d = np.asarray(direction, dtype=float)
    return CurvePath(np.array([[0.0, 0.0, 0.0], length * d]))


def test_marker_round_trip_and_presets():
    m = Marker("Нижняя почечная", np.array([1.0, 2.0, 3.0]), "right", 42.5, key="lowest_renal")
    back = Marker.from_dict(m.to_dict())
    assert back.name == m.name and back.key == "lowest_renal" and back.path == "right"
    assert back.point == pytest.approx(m.point) and back.arc_length_mm == pytest.approx(42.5)
    assert len(set(MARKER_PRESETS)) == len(MARKER_PRESETS)
    for key in MARKER_PRESETS:
        assert tr(f"marker.{key}") != f"marker.{key}"


def test_distance_on_one_path():
    path = straight()
    paths = {"main": path}
    a = marker_on_path("A", "main", path, 30.0)
    b = marker_on_path("B", "main", path, 110.0)
    d = marker_distance(a, b, paths)
    assert d.along_mm == pytest.approx(80.0) and d.straight_mm == pytest.approx(80.0) and d.path == "main"
    assert [m.name for m in sort_along([b, a], path)] == ["A", "B"]


def y_network(angle_deg=25.0, trunk=100.0, leg=80.0):
    """Two paths sharing a straight trunk, then diverging symmetrically in the x-z plane."""
    h = math.radians(angle_deg / 2)
    trunk_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -trunk]])

    def leg_points(sign):
        end = trunk_pts[-1] + leg * np.array([sign * math.sin(h), 0.0, -math.cos(h)])
        return np.vstack([trunk_pts, end])

    return {"right": CurvePath(leg_points(-1.0)), "left": CurvePath(leg_points(1.0))}


def test_distance_between_branches_goes_through_the_split():
    paths = y_network()
    right, left = paths["right"], paths["left"]
    assert divergence(right, left) == pytest.approx(100.0, abs=0.5)
    a = marker_on_path("R", "right", right, 100.0 + 30.0)
    b = marker_on_path("L", "left", left, 100.0 + 50.0)
    d = marker_distance(a, b, paths)
    assert d.path is None
    assert d.along_mm == pytest.approx(80.0, abs=LENGTH_LIMIT_MM)
    assert d.straight_mm < d.along_mm
    # A marker on the shared trunk lies on both paths.
    c = marker_on_path("T", "right", right, 40.0)
    assert marker_distance(c, b, paths).along_mm == pytest.approx(110.0, abs=1e-6)
    table = distance_table([a, b, c], paths)
    assert table[0][0] is None and table[0][1].along_mm == pytest.approx(d.along_mm)
    assert table[2][0].along_mm == pytest.approx(90.0, abs=1e-6)


def test_unconnected_points_have_no_along_distance():
    d = distance(np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, -50.0]), {"main": straight()})
    assert d.along_mm is None and d.straight_mm == pytest.approx(math.hypot(100, 50))


def _extracted_path(phantom, margin_mm=6.0):
    """Core minimal-path centerline between seeds near the ends of the truth centerline."""
    g = phantom.volume.geometry
    pts = phantom.truth.centerline_points
    ijk = g.ras_to_ijk(pts)
    inside = np.flatnonzero(np.all((ijk >= 3) & (ijk <= np.array(g.dimensions) - 4), axis=1))
    step = int(round(margin_mm / 0.5))
    start, end = pts[inside[0] + step], pts[inside[-1] - step]
    network = extract_centerlines(phantom.volume, segment_lumen(phantom.volume, start, [end]))
    return CurvePath(network.paths[0].points)


@pytest.mark.parametrize("fixture", ["realistic_torus", "realistic_helix", "aneurysm"])
def test_lengths_between_markers_on_extracted_centerline(request, fixture):
    phantom = request.getfixturevalue(fixture)
    truth_path = CurvePath(phantom.truth.centerline_points)
    extracted = _extracted_path(phantom)
    paths = {"extracted": extracted}
    s_first = truth_path.nearest_arc_length(extracted.points[0])[0] + 5.0
    s_last = truth_path.nearest_arc_length(extracted.points[-1])[0] - 5.0
    for s_a, s_b in [(s_first, s_last), (s_first, 0.5 * (s_first + s_last)), (s_first + 13.0, s_first + 47.5)]:
        # Markers placed on the extracted line at the true anatomical positions.
        a = marker_on_path("a", "extracted", extracted, extracted.nearest_arc_length(truth_path.point_at(s_a))[0])
        b = marker_on_path("b", "extracted", extracted, extracted.nearest_arc_length(truth_path.point_at(s_b))[0])
        d = marker_distance(a, b, paths)
        assert d.along_mm == pytest.approx(s_b - s_a, abs=LENGTH_LIMIT_MM), (s_a, s_b)


def test_curvature_lengths_of_the_arch(realistic_torus):
    e = realistic_torus.truth.expected
    path = CurvePath(realistic_torus.truth.centerline_points)
    r = curvature_lengths(realistic_torus.volume, path, e["arc_start_s_mm"], e["arc_end_s_mm"])
    R, radius = e["curvature_radius_mm"], 12.5
    assert r.centerline_mm == pytest.approx(math.pi * R, abs=0.1)
    assert r.greater_mm == pytest.approx(math.pi * (R + radius), abs=LENGTH_LIMIT_MM)
    assert r.lesser_mm == pytest.approx(math.pi * (R - radius), abs=LENGTH_LIMIT_MM)
    assert r.straight_mm == pytest.approx(2 * R, abs=0.1)
    assert set(r.to_dict()) == {"centerline_mm", "greater_curvature_mm", "lesser_curvature_mm", "straight_mm"}


def test_curvature_lengths_of_a_helix(realistic_helix):
    p = realistic_helix.truth.parameters
    path = CurvePath(realistic_helix.truth.centerline_points)
    rh, c, radius = p["helix_radius_mm"], p["pitch_mm"] / (2 * math.pi), 0.5 * p["diameter_mm"]
    turn = 2 * math.pi * math.hypot(rh, c)
    mid = path.length / 2
    r = curvature_lengths(realistic_helix.volume, path, mid - turn / 2, mid + turn / 2)
    assert r.greater_mm == pytest.approx(2 * math.pi * math.hypot(rh + radius, c), abs=LENGTH_LIMIT_MM)
    assert r.lesser_mm == pytest.approx(2 * math.pi * math.hypot(rh - radius, c), abs=LENGTH_LIMIT_MM)
    assert tortuosity_index(path, mid - turn / 2, mid + turn / 2) == pytest.approx(turn / p["pitch_mm"], rel=1e-3)


def test_straight_vessel_curvature_lengths_equal_centerline(aneurysm):
    path = CurvePath(aneurysm.truth.centerline_points)
    r = curvature_lengths(aneurysm.volume, path, 20.0, 60.0)
    assert r.greater_mm == pytest.approx(40.0) and r.lesser_mm == pytest.approx(40.0)


# ------------------------------------------------------------------ EVAR angles
def bent_path(alpha_deg, beta_deg):
    """Suprarenal 40 mm straight down, neck 30 mm bent by alpha, sac 80 mm bent by beta (x-z plane)."""
    def direction(theta):
        t = math.radians(theta)
        return np.array([math.sin(t), 0.0, -math.cos(t)])

    p0 = np.zeros(3)
    p1 = p0 + 40.0 * direction(0.0)
    p2 = p1 + 30.0 * direction(alpha_deg)
    p3 = p2 + 80.0 * direction(alpha_deg - beta_deg)  # beta bends back the other way
    return CurvePath(np.array([p0, p1, p2, p3]))


@pytest.mark.parametrize("alpha,beta", [(0.0, 0.0), (30.0, 45.0), (60.0, 20.0)])
def test_neck_angles(alpha, beta):
    path = bent_path(alpha, beta)
    angles = neck_angles(path, lowest_renal_mm=40.0, neck_end_mm=70.0, bifurcation_mm=150.0)
    assert angles.alpha_deg == pytest.approx(alpha, abs=0.5)
    assert angles.beta_deg == pytest.approx(beta, abs=0.5)
    assert angles.suprarenal_start_mm == pytest.approx(20.0)
    assert set(angles.to_dict()) >= {"alpha_deg", "beta_deg"}


def test_neck_angles_need_ordered_positions():
    path = bent_path(10.0, 10.0)
    with pytest.raises(ValueError):
        neck_angles(path, 70.0, 40.0, 150.0)
    assert angle_between(axis(path, 0.0, 40.0), axis(path, 0.0, 40.0)) == pytest.approx(0.0, abs=1e-6)
