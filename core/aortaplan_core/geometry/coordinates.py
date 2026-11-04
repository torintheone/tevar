"""Coordinate-system conversions.

The core works in patient RAS (Right-Anterior-Superior) millimetres, like Slicer.
DICOM and most NRRD files written by other tools use LPS (Left-Posterior-Superior).
RAS and LPS differ by the sign of the first two axes. Conversions happen only at
I/O boundaries (file readers/writers, the Slicer bridge) and nowhere else.
"""

from __future__ import annotations

import numpy as np

#: 3x3 matrix converting RAS vectors to LPS vectors (it is its own inverse).
RAS_TO_LPS_3x3 = np.diag([-1.0, -1.0, 1.0])
#: 4x4 homogeneous version of :data:`RAS_TO_LPS_3x3`.
RAS_TO_LPS_4x4 = np.diag([-1.0, -1.0, 1.0, 1.0])

#: Unit direction vectors of the patient axes, expressed in RAS.
PATIENT_DIRECTIONS_RAS = {
    "R": np.array([1.0, 0.0, 0.0]),
    "L": np.array([-1.0, 0.0, 0.0]),
    "A": np.array([0.0, 1.0, 0.0]),
    "P": np.array([0.0, -1.0, 0.0]),
    "S": np.array([0.0, 0.0, 1.0]),
    "I": np.array([0.0, 0.0, -1.0]),
}


def ras_to_lps(points: np.ndarray) -> np.ndarray:
    """Convert points or vectors (shape ``(3,)`` or ``(N, 3)``) from RAS to LPS."""
    pts = np.asarray(points, dtype=float)
    return pts * np.array([-1.0, -1.0, 1.0])


def lps_to_ras(points: np.ndarray) -> np.ndarray:
    """Convert points or vectors (shape ``(3,)`` or ``(N, 3)``) from LPS to RAS."""
    # The conversion is an involution: applying it twice gives the identity.
    return ras_to_lps(points)


def ras_to_lps_matrix(matrix_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 matrix that maps *into* RAS (e.g. IJK->RAS) into one mapping into LPS."""
    m = np.asarray(matrix_4x4, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got shape {m.shape}")
    return RAS_TO_LPS_4x4 @ m


def lps_to_ras_matrix(matrix_4x4: np.ndarray) -> np.ndarray:
    """Convert a 4x4 matrix that maps *into* LPS (e.g. IJK->LPS) into one mapping into RAS."""
    return ras_to_lps_matrix(matrix_4x4)
