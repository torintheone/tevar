import numpy as np

from aortaplan_core.geometry.coordinates import (
    PATIENT_DIRECTIONS_RAS,
    RAS_TO_LPS_4x4,
    lps_to_ras,
    lps_to_ras_matrix,
    ras_to_lps,
    ras_to_lps_matrix,
)


def test_point_conversion_flips_first_two_axes():
    np.testing.assert_allclose(ras_to_lps([1.0, 2.0, 3.0]), [-1.0, -2.0, 3.0])
    np.testing.assert_allclose(lps_to_ras([1.0, 2.0, 3.0]), [-1.0, -2.0, 3.0])


def test_conversion_is_an_involution_for_point_arrays():
    pts = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(lps_to_ras(ras_to_lps(pts)), pts)


def test_patient_left_is_positive_x_in_lps():
    np.testing.assert_allclose(ras_to_lps(PATIENT_DIRECTIONS_RAS["L"]), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ras_to_lps(PATIENT_DIRECTIONS_RAS["P"]), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(ras_to_lps(PATIENT_DIRECTIONS_RAS["S"]), [0.0, 0.0, 1.0])


def test_matrix_conversion_matches_point_conversion():
    m = np.eye(4)
    m[:3, :3] = np.diag([0.7, 0.8, 1.5])
    m[:3, 3] = [10.0, -20.0, 30.0]
    m_lps = ras_to_lps_matrix(m)
    ijk = np.array([3.0, 4.0, 5.0, 1.0])
    np.testing.assert_allclose((m_lps @ ijk)[:3], ras_to_lps((m @ ijk)[:3]))
    np.testing.assert_allclose(lps_to_ras_matrix(m_lps), m)
    np.testing.assert_allclose(RAS_TO_LPS_4x4 @ RAS_TO_LPS_4x4, np.eye(4))
