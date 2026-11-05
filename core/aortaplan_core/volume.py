"""Image volumes with explicit RAS geometry.

Conventions (shared with Slicer):

- ``Volume.data`` is a numpy array indexed ``[k, j, i]`` (slowest to fastest),
  exactly like ``slicer.util.arrayFromVolume``. ``i`` is the fastest axis on disk.
- ``ImageGeometry`` maps voxel indices ``(i, j, k)`` to RAS millimetres:
  ``ras = origin + directions @ (spacing * ijk)``, where the columns of
  ``directions`` are the unit RAS directions of the i, j and k axes.
- ``origin`` is the RAS position of the *centre* of voxel ``(0, 0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

#: Direction matrix of a typical axial CT series: IJK->LPS is identity, so in RAS
#: the i axis points to patient left, j to posterior and k to superior.
CT_LIKE_DIRECTIONS = np.diag([-1.0, -1.0, 1.0])


@dataclass
class ImageGeometry:
    """Voxel grid geometry in RAS millimetres."""

    dimensions: Tuple[int, int, int]  # (ni, nj, nk)
    spacing: Tuple[float, float, float]  # mm along i, j, k
    origin: Tuple[float, float, float]  # RAS of voxel (0, 0, 0) centre
    directions: np.ndarray  # 3x3, columns = unit RAS directions of i, j, k

    def __post_init__(self) -> None:
        self.dimensions = tuple(int(n) for n in self.dimensions)  # type: ignore[assignment]
        self.spacing = tuple(float(s) for s in self.spacing)  # type: ignore[assignment]
        self.origin = tuple(float(o) for o in self.origin)  # type: ignore[assignment]
        self.directions = np.array(self.directions, dtype=float).reshape(3, 3)
        if len(self.dimensions) != 3 or min(self.dimensions) < 1:
            raise ValueError(f"invalid dimensions {self.dimensions}")
        if min(self.spacing) <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        gram = self.directions.T @ self.directions
        if not np.allclose(gram, np.eye(3), atol=1e-6):
            raise ValueError("direction matrix columns must be orthonormal")

    # ------------------------------------------------------------------ matrices
    def ijk_to_ras_matrix(self) -> np.ndarray:
        """4x4 homogeneous IJK->RAS matrix (same meaning as Slicer's IJKToRASMatrix)."""
        m = np.eye(4)
        m[:3, :3] = self.directions * np.asarray(self.spacing)[np.newaxis, :]
        m[:3, 3] = self.origin
        return m

    def ras_to_ijk_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.ijk_to_ras_matrix())

    @classmethod
    def from_ijk_to_ras_matrix(cls, matrix_4x4: np.ndarray, dimensions: Sequence[int]) -> ImageGeometry:
        """Build a geometry from a 4x4 IJK->RAS matrix (e.g. read from a Slicer volume node)."""
        m = np.asarray(matrix_4x4, dtype=float)
        if m.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got shape {m.shape}")
        columns = m[:3, :3]
        spacing = np.linalg.norm(columns, axis=0)
        directions = columns / spacing[np.newaxis, :]
        return cls(
            dimensions=tuple(int(n) for n in dimensions),  # type: ignore[arg-type]
            spacing=tuple(spacing),  # type: ignore[arg-type]
            origin=tuple(m[:3, 3]),  # type: ignore[arg-type]
            directions=directions,
        )

    @classmethod
    def covering_box(
        cls,
        ras_min: Sequence[float],
        ras_max: Sequence[float],
        spacing: Sequence[float],
        directions: np.ndarray = CT_LIKE_DIRECTIONS,
    ) -> ImageGeometry:
        """Smallest grid with the given spacing/directions whose voxel centres cover an RAS box."""
        lo = np.asarray(ras_min, dtype=float)
        hi = np.asarray(ras_max, dtype=float)
        d = np.array(directions, dtype=float).reshape(3, 3)
        sp = np.asarray(spacing, dtype=float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        # Coordinates of the corners along the grid axes (in mm).
        along = corners @ d
        start = along.min(axis=0)
        extent = along.max(axis=0) - start
        dims = np.floor(extent / sp + 1e-9).astype(int) + 1
        origin = d @ start
        return cls(dimensions=tuple(dims), spacing=tuple(sp), origin=tuple(origin), directions=d)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ transforms
    def ijk_to_ras(self, ijk: np.ndarray) -> np.ndarray:
        """Map continuous voxel indices ``(i, j, k)`` (shape ``(3,)`` or ``(N, 3)``) to RAS."""
        p = np.asarray(ijk, dtype=float)
        m = self.ijk_to_ras_matrix()
        return p @ m[:3, :3].T + m[:3, 3]

    def ras_to_ijk(self, ras: np.ndarray) -> np.ndarray:
        """Map RAS points (shape ``(3,)`` or ``(N, 3)``) to continuous voxel indices ``(i, j, k)``."""
        p = np.asarray(ras, dtype=float)
        m = self.ras_to_ijk_matrix()
        return p @ m[:3, :3].T + m[:3, 3]

    def voxel_centers_ras(self) -> np.ndarray:
        """RAS coordinates of all voxel centres, shape ``(nk, nj, ni, 3)`` (array index order)."""
        ni, nj, nk = self.dimensions
        k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
        ijk = np.stack([i, j, k], axis=-1).astype(float)
        return self.ijk_to_ras(ijk.reshape(-1, 3)).reshape(nk, nj, ni, 3)

    def subgrid(self, start_ijk: Sequence[int], dimensions: Sequence[int]) -> ImageGeometry:
        """Geometry of a box of voxels starting at voxel ``start_ijk`` (i, j, k) of this grid."""
        return ImageGeometry(
            dimensions=tuple(int(d) for d in dimensions),  # type: ignore[arg-type]
            spacing=self.spacing,
            origin=tuple(self.ijk_to_ras(np.asarray(start_ijk, dtype=float))),  # type: ignore[arg-type]
            directions=self.directions,
        )

    @property
    def array_shape(self) -> Tuple[int, int, int]:
        """Shape of the numpy array holding this grid: ``(nk, nj, ni)``."""
        ni, nj, nk = self.dimensions
        return (nk, nj, ni)

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "directions": self.directions.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ImageGeometry:
        return cls(
            dimensions=tuple(d["dimensions"]),  # type: ignore[arg-type]
            spacing=tuple(d["spacing"]),  # type: ignore[arg-type]
            origin=tuple(d["origin"]),  # type: ignore[arg-type]
            directions=np.array(d["directions"], dtype=float),
        )

    def is_close(self, other: ImageGeometry, atol: float = 1e-6) -> bool:
        return (
            self.dimensions == other.dimensions
            and np.allclose(self.spacing, other.spacing, atol=atol)
            and np.allclose(self.origin, other.origin, atol=atol)
            and np.allclose(self.directions, other.directions, atol=atol)
        )


@dataclass
class Volume:
    """Scalar volume: voxel array ``[k, j, i]`` plus its RAS geometry."""

    data: np.ndarray
    geometry: ImageGeometry

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"volume data must be 3-D, got {self.data.ndim}-D")
        if tuple(self.data.shape) != self.geometry.array_shape:
            raise ValueError(
                f"data shape {self.data.shape} does not match geometry array shape {self.geometry.array_shape}"
            )

    def sample_linear(self, ras_points: np.ndarray, outside_value: float = np.nan) -> np.ndarray:
        """Trilinear interpolation of the volume at RAS points (shape ``(3,)`` or ``(N, 3)``)."""
        from scipy.ndimage import map_coordinates

        pts = np.atleast_2d(np.asarray(ras_points, dtype=float))
        ijk = self.geometry.ras_to_ijk(pts)
        # map_coordinates expects coordinates in array index order (k, j, i).
        coords = ijk[:, ::-1].T
        values = map_coordinates(self.data, coords, order=1, mode="constant", cval=np.nan, output=np.float64)
        values = np.where(np.isnan(values), outside_value, values)
        return values if np.ndim(ras_points) > 1 else values[0]
