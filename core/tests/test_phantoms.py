import math

import numpy as np
import pytest

from aortaplan_core.geometry.curves import cumulative_arc_length
from aortaplan_core.phantoms import (
    STANDARD_PHANTOMS,
    ellipse_reference_metrics,
    load_phantom,
    save_phantom,
    straight_tube,
)
from aortaplan_core.phantoms.__main__ import main as phantoms_cli
from aortaplan_core.phantoms.library import BACKGROUND_HU, LUMEN_HU
from aortaplan_core.volume import CT_LIKE_DIRECTIONS

LUMEN_THRESHOLD = 0.5 * (LUMEN_HU + BACKGROUND_HU)


def lumen_fraction(volume):
    return (volume.data.astype(float) - BACKGROUND_HU) / (LUMEN_HU - BACKGROUND_HU)


def inside_volume(phantom, points, margin_voxels=1.0):
    g = phantom.volume.geometry
    ijk = g.ras_to_ijk(points)
    upper = np.array(g.dimensions) - 1 - margin_voxels
    return np.all((ijk >= margin_voxels) & (ijk <= upper), axis=1)


def check_wall_position(phantom, offset_inside=1.0, offset_outside=1.5):
    """Points just inside/outside the lumen boundary (along N and B) have lumen/background HU."""
    truth = phantom.truth
    pts, n, b = truth.centerline_points, truth.centerline_normals, truth.centerline_binormals
    lumen = truth.layer("lumen")
    keep = inside_volume(phantom, pts, margin_voxels=2.0)
    assert keep.sum() > 20
    for axis, semi in ((n, lumen.semi_axis_n), (b, lumen.semi_axis_b)):
        for sign in (1.0, -1.0):
            inner = pts + sign * axis * (semi - offset_inside)[:, None]
            outer = pts + sign * axis * (semi + offset_outside)[:, None]
            k_in = keep & inside_volume(phantom, inner)
            k_out = keep & inside_volume(phantom, outer)
            assert np.all(phantom.volume.sample_linear(inner[k_in]) > LUMEN_THRESHOLD)
            assert np.all(phantom.volume.sample_linear(outer[k_out]) < LUMEN_THRESHOLD)
    on_centerline = phantom.volume.sample_linear(pts[keep])
    np.testing.assert_allclose(on_centerline, LUMEN_HU, atol=1.0)


def test_ellipse_reference_metrics():
    circle = ellipse_reference_metrics(10.0, 10.0)
    assert circle["perimeter_mm"] == pytest.approx(2 * math.pi * 10.0, rel=1e-12)
    assert circle["area_equivalent_diameter_mm"] == pytest.approx(20.0)
    assert circle["perimeter_equivalent_diameter_mm"] == pytest.approx(20.0)
    ell = ellipse_reference_metrics(10.0, 15.0)
    assert ell["min_diameter_mm"] == 20.0 and ell["max_diameter_mm"] == 30.0
    assert ell["perimeter_mm"] == pytest.approx(79.3274, abs=1e-3)  # reference value for a=15, b=10


def test_straight_tube_cross_section_area(straight_phantom):
    g = straight_phantom.volume.geometry
    assert straight_phantom.volume.data.dtype == np.int16
    np.testing.assert_allclose(g.directions, CT_LIKE_DIRECTIONS)
    area = lumen_fraction(straight_phantom.volume).sum(axis=(1, 2)) * g.spacing[0] * g.spacing[1]
    expected = straight_phantom.truth.expected["lumen"]["area_mm2"]
    assert expected == pytest.approx(math.pi * 12.5**2)
    np.testing.assert_allclose(area, expected, rtol=5e-3)


def test_straight_tube_wall_position(straight_phantom):
    check_wall_position(straight_phantom)


def test_oblique_anisotropic_tube(oblique_phantom):
    assert oblique_phantom.volume.geometry.spacing == (0.7, 0.7, 1.25)
    check_wall_position(oblique_phantom)
    # Lumen volume per unit length is pi r^2 regardless of obliquity.
    g = oblique_phantom.volume.geometry
    total = lumen_fraction(oblique_phantom.volume).sum() * g.voxel_volume
    assert total > 0


def test_elliptical_tube(elliptical_phantom):
    g = elliptical_phantom.volume.geometry
    area = lumen_fraction(elliptical_phantom.volume).sum(axis=(1, 2)) * g.spacing[0] * g.spacing[1]
    np.testing.assert_allclose(area, math.pi * 15.0 * 10.0, rtol=5e-3)
    # Major axis follows the reference direction (patient left-right).
    np.testing.assert_allclose(np.abs(elliptical_phantom.truth.centerline_normals[0]), [1.0, 0.0, 0.0], atol=1e-9)
    check_wall_position(elliptical_phantom)


def test_torus_arc(torus_phantom):
    truth = torus_phantom.truth
    e = truth.expected
    R = truth.parameters["major_radius_mm"]
    assert e["arc_length_mm"] == pytest.approx(math.pi * R)
    # Dense centerline: two straight legs plus the semicircle.
    assert e["centerline_length_mm"] == pytest.approx(2 * e["arc_start_s_mm"] + math.pi * R, abs=1e-3)
    s = cumulative_arc_length(truth.centerline_points)
    on_arc = (s > e["arc_start_s_mm"] + 0.5) & (s < e["arc_end_s_mm"] - 0.5)
    np.testing.assert_allclose(np.linalg.norm(truth.centerline_points[on_arc, 1:], axis=1), R, atol=1e-3)
    np.testing.assert_allclose(truth.centerline_points[:, 0], 0.0, atol=1e-9)  # planar (A-S plane)
    check_wall_position(torus_phantom)


def test_helix(helix_phantom):
    truth = helix_phantom.truth
    e = truth.expected
    span = e["parameter_range_rad"][1] - e["parameter_range_rad"][0]
    analytic = e["length_per_turn_mm"] * span / (2 * math.pi)
    assert e["centerline_length_mm"] == pytest.approx(analytic, rel=1e-6)  # dense curve
    # The truth polyline (0.5 mm chords) is shorter by about curvature^2 * h^2 / 24.
    total = cumulative_arc_length(truth.centerline_points)[-1]
    assert total == pytest.approx(analytic, rel=1e-4)
    check_wall_position(helix_phantom)


def test_frames_in_truth_are_orthonormal(torus_phantom, helix_phantom):
    for ph in (torus_phantom, helix_phantom):
        t, n = ph.truth.centerline_tangents, ph.truth.centerline_normals
        np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
        assert np.max(np.abs(np.einsum("ij,ij->i", t, n))) < 1e-6


def test_save_and_load_round_trip(tmp_path, straight_phantom):
    paths = save_phantom(straight_phantom, tmp_path)
    assert paths["volume"].endswith("straight_tube.nrrd")
    back = load_phantom(tmp_path, "straight_tube")
    np.testing.assert_array_equal(back.volume.data, straight_phantom.volume.data)
    assert back.volume.geometry.is_close(straight_phantom.volume.geometry)
    assert back.truth.to_dict() == straight_phantom.truth.to_dict()


def test_noise_is_deterministic():
    kwargs = dict(diameter_mm=10.0, length_mm=10.0, noise_sigma_hu=20.0, blur_sigma_mm=0.5, supersampling=2)
    a = straight_tube(seed=7, **kwargs).volume.data
    b = straight_tube(seed=7, **kwargs).volume.data
    c = straight_tube(seed=8, **kwargs).volume.data
    np.testing.assert_array_equal(a, b)
    assert np.any(a != c)
    background = a[:, 0, 0].astype(float)
    assert 5.0 < background.std() < 40.0


def test_cli_lists_standard_phantoms(capsys):
    assert phantoms_cli(["--list"]) == 0
    assert capsys.readouterr().out.split() == list(STANDARD_PHANTOMS)


def test_cli_rejects_unknown_name():
    with pytest.raises(SystemExit):
        phantoms_cli(["--only", "no_such_phantom"])


def test_aneurysm_phantom_truth_and_round_trip(tmp_path, aneurysm):
    truth = aneurysm.truth
    e = truth.expected
    outer, lumen = truth.layer("outer"), truth.layer("lumen")
    s = cumulative_arc_length(truth.centerline_points)
    i = int(np.argmin(np.abs(s - e["max_outer_s_mm"])))
    assert (outer.semi_axis_n[i], outer.semi_axis_b[i]) == pytest.approx((30.0, 27.0))
    assert lumen.semi_axis_n[i] == pytest.approx(15.0)
    assert outer.offset_b[i] == pytest.approx(0.6 * (27.0 - 15.0 - 1.5))  # thrombus thick anteriorly
    assert lumen.is_centered and not outer.is_centered
    neck = int(np.argmin(np.abs(s - (e["sac_start_s_mm"] - 10.0))))
    assert outer.semi_axis_n[neck] == pytest.approx(12.5) and outer.offset_b[neck] == pytest.approx(0.0)
    # Intensities: lumen 350, thrombus 40 and fat -100 HU at the widest section.
    g = aneurysm.volume.geometry
    centre = truth.centerline_points[i]
    k = int(round(g.ras_to_ijk(centre)[2]))
    section = aneurysm.volume.data[k].astype(float)
    assert np.median(section[section > 250]) == pytest.approx(350, abs=10)
    assert np.median(section[:5, :5]) == pytest.approx(-100, abs=10)
    anterior = g.ras_to_ijk(centre + np.array([0.0, 25.0, 0.0])).round().astype(int)
    assert aneurysm.volume.data[anterior[2], anterior[1], anterior[0]] == pytest.approx(40, abs=60)
    save_phantom(aneurysm, tmp_path)
    back = load_phantom(tmp_path, truth.name)
    assert back.truth.layer("outer").offset_b == pytest.approx(outer.offset_b, abs=1e-6)
    assert back.truth.to_dict() == truth.to_dict()
