"""Straightened curved planar reformation (CPR) along a centerline (SPEC 5, stage 3).

The CPR image has one row per centerline sample (row 0 = proximal end) and one column per
lateral offset. Each row samples the volume along a line through the centerline point in
the direction

    lateral(s, angle) = cos(angle) * N(s) + sin(angle) * B(s)

where (T, N, B) is a rotation-minimizing frame, so the image does not twist along the
vessel. At angle 0 the lateral direction at the proximal end is the patient's left
(coronal-like, radiological: patient's left on the right of the image); at 90 deg it is
B = T x N (for a vessel running towards the feet: anterior).

Coordinates of a CPR point: ``(offset_mm, s_mm)`` = lateral offset (positive to the image
right) and arc length along the centerline. ``to_ras`` and ``from_ras`` convert between
CPR and patient RAS coordinates; ``pixel_to_cpr`` / ``cpr_to_pixel`` handle image indices.

Known limitation of every straightened CPR: on the inner side of a bend with a radius of
curvature smaller than the half-width the sampling lines cross ("folding").
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..geometry.frames import initial_normal, rotation_minimizing_frames
from ..volume import Volume
from .navigation import CurvePath

LEFT = np.array([-1.0, 0.0, 0.0])
ANTERIOR = np.array([0.0, 1.0, 0.0])


class StraightenedCpr:
    """Geometry and sampling of a straightened CPR along a :class:`CurvePath`."""

    def __init__(self, path: CurvePath, half_width_mm: float = 35.0, column_spacing_mm: float = 0.5):
        self.path = path
        self.half_width_mm = float(half_width_mm)
        self.column_spacing_mm = float(column_spacing_mm)
        self.points = path.points
        self.arc_length = path.arc_length
        tangents = path.tangents
        reference = LEFT if abs(float(np.dot(tangents[0], LEFT))) < 0.9 else ANTERIOR
        start_normal = initial_normal(tangents[0], reference)
        self.tangents, self.normals, self.binormals = rotation_minimizing_frames(self.points, start_normal, tangents)
        n_half = int(round(self.half_width_mm / self.column_spacing_mm))
        self.offsets = np.arange(-n_half, n_half + 1) * self.column_spacing_mm

    # ------------------------------------------------------------------ sizes
    @property
    def rows(self) -> int:
        return len(self.points)

    @property
    def columns(self) -> int:
        return len(self.offsets)

    @property
    def row_spacing_mm(self) -> float:
        return float(np.median(np.diff(self.arc_length))) if self.rows > 1 else 1.0

    # ------------------------------------------------------------------ directions
    def lateral_directions(self, angle_deg: float) -> np.ndarray:
        """Unit lateral direction (image right) for every row, shape (rows, 3)."""
        a = math.radians(angle_deg)
        return math.cos(a) * self.normals + math.sin(a) * self.binormals

    def _frame_at(self, s: float, angle_deg: float):
        s = self.path.clamp(s)
        i = int(np.clip(np.searchsorted(self.arc_length, s, side="right") - 1, 0, self.rows - 2))
        seg = self.arc_length[i + 1] - self.arc_length[i]
        f = 0.0 if seg <= 0 else (s - self.arc_length[i]) / seg
        point = (1 - f) * self.points[i] + f * self.points[i + 1]
        tangent = (1 - f) * self.tangents[i] + f * self.tangents[i + 1]
        tangent /= np.linalg.norm(tangent)
        a = math.radians(angle_deg)
        lateral = math.cos(a) * ((1 - f) * self.normals[i] + f * self.normals[i + 1]) + math.sin(a) * (
            (1 - f) * self.binormals[i] + f * self.binormals[i + 1]
        )
        lateral -= np.dot(lateral, tangent) * tangent
        lateral /= np.linalg.norm(lateral)
        return point, tangent, lateral

    # ------------------------------------------------------------------ mapping
    def to_ras(self, offset_mm: float, s_mm: float, angle_deg: float) -> np.ndarray:
        """Patient RAS position of a CPR point."""
        point, _, lateral = self._frame_at(s_mm, angle_deg)
        return point + float(offset_mm) * lateral

    def from_ras(self, ras: np.ndarray, angle_deg: float) -> Tuple[float, float, float]:
        """CPR coordinates ``(offset_mm, s_mm, out_of_plane_mm)`` of a patient RAS point.

        ``s`` is the arc length of the nearest centerline point; ``out_of_plane`` is the
        distance from the CPR surface (0 for points shown in the image).
        """
        p = np.asarray(ras, dtype=float)
        s = self._foot_arc_length(p)
        point, tangent, lateral = self._frame_at(s, angle_deg)
        rel = p - point
        normal = np.cross(tangent, lateral)
        return float(np.dot(rel, lateral)), float(s), float(np.dot(rel, normal))

    def _foot_arc_length(self, p: np.ndarray) -> float:
        """Arc length whose (interpolated) normal plane contains ``p`` — the inverse of ``to_ras``."""
        s, _ = self.path.nearest_arc_length(p)

        def f(x: float) -> float:
            point, tangent, _ = self._frame_at(x, 0.0)
            return float(np.dot(p - point, tangent))

        h = 0.05
        for _ in range(8):  # Newton iterations with a numerical derivative
            value = f(s)
            if abs(value) < 1e-10:
                break
            derivative = (f(s + h) - f(s - h)) / (2 * h)
            if abs(derivative) < 1e-6:
                break
            s = self.path.clamp(s - value / derivative)
        return s

    def pixel_to_cpr(self, column: float, row: float) -> Tuple[float, float]:
        """Continuous image indices -> (offset_mm, s_mm)."""
        offset = (column - (self.columns - 1) / 2.0) * self.column_spacing_mm
        s = float(np.interp(row, np.arange(self.rows), self.arc_length))
        return offset, s

    def cpr_to_pixel(self, offset_mm: float, s_mm: float) -> Tuple[float, float]:
        """(offset_mm, s_mm) -> continuous image indices (column, row)."""
        column = offset_mm / self.column_spacing_mm + (self.columns - 1) / 2.0
        row = float(np.interp(s_mm, self.arc_length, np.arange(self.rows)))
        return column, row

    # ------------------------------------------------------------------ image
    def sample(self, volume: Volume, angle_deg: float, outside_value: float = -1024.0) -> np.ndarray:
        """CPR image (rows, columns), float32, trilinear interpolation."""
        lateral = self.lateral_directions(angle_deg)
        positions = self.points[:, np.newaxis, :] + self.offsets[np.newaxis, :, np.newaxis] * lateral[:, np.newaxis, :]
        values = volume.sample_linear(positions.reshape(-1, 3), outside_value=outside_value)
        return values.reshape(self.rows, self.columns).astype(np.float32)

    def angle_through(self, ras: np.ndarray) -> Optional[float]:
        """CPR angle (deg) whose image contains the given point (None on the centerline itself)."""
        p = np.asarray(ras, dtype=float)
        s, _ = self.path.nearest_arc_length(p)
        point, tangent, _ = self._frame_at(s, 0.0)
        rel = p - point
        rel -= np.dot(rel, tangent) * tangent
        if np.linalg.norm(rel) < 1e-9:
            return None
        n0 = self._frame_at(s, 0.0)[2]
        b0 = np.cross(tangent, n0)
        return float(math.degrees(math.atan2(np.dot(rel, b0), np.dot(rel, n0))) % 360.0)


def nearest_line_angle(direction_deg: float, current_deg: float) -> float:
    """CPR angle for a line through the centre that points to ``direction_deg``.

    The line also covers ``direction + 180``; of the two the one closer to ``current_deg`` is
    returned, so grabbing either end of the line never mirrors the image.
    """
    a = direction_deg % 360.0
    b = (a + 180.0) % 360.0

    def distance(x: float) -> float:
        return abs((x - current_deg + 180.0) % 360.0 - 180.0)

    return a if distance(a) <= distance(b) else b

