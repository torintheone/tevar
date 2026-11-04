import numpy as np
import pytest

from aortaplan_core.volume import CT_LIKE_DIRECTIONS, ImageGeometry, Volume


def make_geometry():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about S
    return ImageGeometry(dimensions=(4, 5, 6), spacing=(0.5, 0.7, 1.2), origin=(10.0, -5.0, 3.0), directions=rot)


def test_ijk_ras_round_trip():
    g = make_geometry()
    ijk = np.array([[0, 0, 0], [1, 2, 3], [3.5, 4.2, 5.9]], dtype=float)
    np.testing.assert_allclose(g.ras_to_ijk(g.ijk_to_ras(ijk)), ijk, atol=1e-12)
    np.testing.assert_allclose(g.ijk_to_ras([0, 0, 0]), g.origin)


def test_matrix_round_trip():
    g = make_geometry()
    g2 = ImageGeometry.from_ijk_to_ras_matrix(g.ijk_to_ras_matrix(), g.dimensions)
    assert g.is_close(g2)


def test_ct_like_directions_point_left_posterior_superior():
    g = ImageGeometry(dimensions=(2, 2, 2), spacing=(1, 1, 1), origin=(0, 0, 0), directions=CT_LIKE_DIRECTIONS)
    np.testing.assert_allclose(g.ijk_to_ras([1, 0, 0]), [-1, 0, 0])  # i -> patient left
    np.testing.assert_allclose(g.ijk_to_ras([0, 1, 0]), [0, -1, 0])  # j -> posterior
    np.testing.assert_allclose(g.ijk_to_ras([0, 0, 1]), [0, 0, 1])  # k -> superior


def test_covering_box_covers_the_box():
    g = ImageGeometry.covering_box((-10, -20, -5), (10, 20, 5), (0.7, 0.7, 1.0), CT_LIKE_DIRECTIONS)
    centers = g.voxel_centers_ras().reshape(-1, 3)
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    np.testing.assert_allclose(lo[2], -5.0)
    np.testing.assert_allclose(hi[0], 10.0)
    assert np.all(hi <= np.array([10, 20, 5]) + 1e-9)
    assert np.all(lo >= np.array([-10, -20, -5]) - 1e-9)
    assert np.all(np.array([10, 20, 5]) - hi < np.array(g.spacing))  # covered to within one voxel


def test_voxel_centers_are_in_array_order():
    g = make_geometry()
    centers = g.voxel_centers_ras()
    assert centers.shape == g.array_shape + (3,)
    np.testing.assert_allclose(centers[3, 2, 1], g.ijk_to_ras([1, 2, 3]))


def test_dict_round_trip():
    g = make_geometry()
    assert g.is_close(ImageGeometry.from_dict(g.to_dict()))


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        ImageGeometry(dimensions=(2, 2, 2), spacing=(1, 1, 1), origin=(0, 0, 0), directions=np.ones((3, 3)))
    with pytest.raises(ValueError):
        ImageGeometry(dimensions=(2, 2, 2), spacing=(1, 0, 1), origin=(0, 0, 0), directions=np.eye(3))


def test_volume_shape_must_match_geometry():
    g = make_geometry()
    Volume(data=np.zeros(g.array_shape), geometry=g)
    with pytest.raises(ValueError):
        Volume(data=np.zeros(g.dimensions), geometry=g)


def test_sample_linear_is_exact_for_linear_field():
    g = make_geometry()
    centers = g.voxel_centers_ras()
    field = 2.0 * centers[..., 0] - 3.0 * centers[..., 1] + 0.5 * centers[..., 2]
    vol = Volume(data=field, geometry=g)
    pts = g.ijk_to_ras(np.array([[1.3, 2.7, 3.1], [0.2, 0.5, 4.9]]))
    expected = 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 0.5 * pts[:, 2]
    np.testing.assert_allclose(vol.sample_linear(pts), expected, atol=1e-9)
    assert np.isnan(vol.sample_linear(g.ijk_to_ras([-5.0, 0.0, 0.0])))
