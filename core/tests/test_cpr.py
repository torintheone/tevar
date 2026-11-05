import numpy as np
import pytest

from aortaplan_core.centerline.cpr import StraightenedCpr
from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.phantoms.branching import CALCIUM_HU


def lumen_columns(image, row, threshold=195.0):
    return int(np.sum(image[row] > threshold))


def test_straight_tube_is_a_vertical_band(straight_phantom):
    path = CurvePath(straight_phantom.truth.centerline_points[::-1])  # head -> feet
    cpr = StraightenedCpr(path, half_width_mm=20.0, column_spacing_mm=0.5)
    assert cpr.columns == 81
    np.testing.assert_allclose(cpr.lateral_directions(0.0)[0], [-1, 0, 0], atol=1e-9)  # patient left
    inside = [r for r in range(cpr.rows) if np.isfinite(straight_phantom.volume.sample_linear(cpr.points[r]))]
    for angle in (0.0, 37.0, 90.0):
        image = cpr.sample(straight_phantom.volume, angle)
        rows = [r for r in inside[5:-5:10]]
        widths = [lumen_columns(image, r) * cpr.column_spacing_mm for r in rows]
        np.testing.assert_allclose(widths, 25.0, atol=1.0)  # diameter 25 mm
        centre = (cpr.columns - 1) // 2
        assert np.all(image[rows, centre] > 300)


def test_rotation_by_180_mirrors_the_image(realistic_torus):
    cpr = StraightenedCpr(CurvePath(realistic_torus.truth.centerline_points), half_width_mm=15.0)
    a = cpr.sample(realistic_torus.volume, 20.0)
    b = cpr.sample(realistic_torus.volume, 200.0)
    np.testing.assert_allclose(a, b[:, ::-1], atol=1e-3)


def test_mapping_round_trip_on_a_curved_path(realistic_torus):
    cpr = StraightenedCpr(CurvePath(realistic_torus.truth.centerline_points), half_width_mm=15.0)
    rng = np.random.default_rng(3)
    for _ in range(50):
        offset, s, angle = rng.uniform(-12, 12), rng.uniform(5, cpr.path.length - 5), rng.uniform(0, 360)
        ras = cpr.to_ras(offset, s, angle)
        back = cpr.from_ras(ras, angle)
        assert back == pytest.approx((offset, s, 0.0), abs=1e-5)
        column, row = cpr.cpr_to_pixel(offset, s)
        assert cpr.pixel_to_cpr(column, row) == pytest.approx((offset, s), abs=1e-9)


def test_image_pixel_matches_volume_value(realistic_helix):
    cpr = StraightenedCpr(CurvePath(realistic_helix.truth.centerline_points), half_width_mm=10.0)
    image = cpr.sample(realistic_helix.volume, 45.0)
    for row, column in ((40, 3), (200, 25), (300, 37)):
        offset, s = cpr.pixel_to_cpr(column, row)
        expected = realistic_helix.volume.sample_linear(cpr.to_ras(offset, s, 45.0), outside_value=-1024.0)
        assert image[row, column] == pytest.approx(expected, abs=1e-3)


def test_calcification_is_found_where_mapped(aortoiliac_bone_phantom):
    truth = aortoiliac_bone_phantom.truth
    cpr = StraightenedCpr(CurvePath(truth.path_points("right")), half_width_mm=20.0)
    calcification = np.array([o for o in truth.objects if o["kind"] == "calcification"][0]["center"])
    angle = cpr.angle_through(calcification)
    offset, s, out_of_plane = cpr.from_ras(calcification, angle)
    assert abs(out_of_plane) < 1e-6 and abs(offset) == pytest.approx(10.0, abs=0.5)  # on the aortic wall
    image = cpr.sample(aortoiliac_bone_phantom.volume, angle)
    column, row = cpr.cpr_to_pixel(offset, s)
    assert image[int(round(row)), int(round(column))] > 0.6 * CALCIUM_HU


def test_lateral_directions_stay_perpendicular(realistic_torus):
    cpr = StraightenedCpr(CurvePath(realistic_torus.truth.centerline_points))
    lateral = cpr.lateral_directions(123.0)
    assert np.max(np.abs(np.einsum("ij,ij->i", lateral, cpr.tangents))) < 1e-9
    np.testing.assert_allclose(np.linalg.norm(lateral, axis=1), 1.0)
