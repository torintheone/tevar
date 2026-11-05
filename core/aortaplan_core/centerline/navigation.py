"""Navigation along a centerline: positions by arc length and view orientations.

The curve runs from its first point (proximal, towards the heart) to its last point.
Positions are arc lengths in mm from the first point.

Cross-section (perpendicular) view orientation
-----------------------------------------------
Like an axial slice in radiological convention, seen from the feet:

- screen up = anterior (projection of A on the slice plane);
- screen right = patient's left;
- the viewer looks along ``view_dir = sign * tangent`` chosen so that ``view_dir`` points
  towards the head (``view_dir . S > 0``).

Both rules degenerate somewhere: "up" when the vessel runs along the anterior-posterior
axis (top of the aortic arch), the viewing side when it runs horizontally. In such a run
of samples the missing quantity is carried along the curve with a rotation-minimizing
frame (or simply kept, for the viewing side), starting from the nearest well-defined
sample on the *distal* side (for the arch: from the descending aorta, SPEC section 4).
Inside a degenerate run the view therefore turns continuously; a jump can occur only at
the proximal end of the run (for the arch: between the ascending aorta and the arch).

Longitudinal view
-----------------
A plane containing the tangent, rotated by ``angle_deg`` about it. At 0 deg it is
coronal-like (screen right = patient left), at 90 deg sagittal-like; screen up = towards
the head (``view_dir``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..geometry.curves import cumulative_arc_length, resample_polyline, unit_tangents
from ..geometry.frames import initial_normal, rotation_minimizing_frames

ANTERIOR = np.array([0.0, 1.0, 0.0])
SUPERIOR = np.array([0.0, 0.0, 1.0])

#: A direction is "well defined" when the tangent is at least this far (deg) from degenerating.
DEFAULT_DEGENERATE_ANGLE_DEG = 30.0


@dataclass
class ViewFrame:
    """Orientation of a 2-D view in RAS: origin at the screen centre, unit screen axes."""

    origin: np.ndarray
    x_axis: np.ndarray  # screen right
    y_axis: np.ndarray  # screen up
    normal: np.ndarray  # x_axis cross y_axis (towards the viewer)

    def slice_to_ras(self) -> np.ndarray:
        """4x4 matrix with columns x, y, normal, origin (Slicer's SliceToRAS layout)."""
        m = np.eye(4)
        m[:3, 0] = self.x_axis
        m[:3, 1] = self.y_axis
        m[:3, 2] = self.normal
        m[:3, 3] = self.origin
        return m

    def screen_to_ras(self, x_mm: float, y_mm: float) -> np.ndarray:
        """RAS point at a screen offset (mm) from the view centre."""
        return self.origin + x_mm * self.x_axis + y_mm * self.y_axis


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive index ranges where ``mask`` is True."""
    runs = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _transport(points: np.ndarray, tangents: np.ndarray, start_normal: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Normals at ``indices`` (first index carries ``start_normal``) by rotation-minimizing transport."""
    _, normals, _ = rotation_minimizing_frames(points[indices], start_normal, tangents[indices])
    return normals


def cross_section_orientation(
    points: np.ndarray, tangents: np.ndarray, degenerate_angle_deg: float = DEFAULT_DEGENERATE_ANGLE_DEG
) -> Tuple[np.ndarray, np.ndarray]:
    """Screen-up vectors and viewing signs for every sample (see the module docstring).

    Returns:
        ``(ups, signs)``: (N, 3) unit vectors perpendicular to the tangents and (N,) values
        in {+1, -1}; the viewer looks along ``signs[i] * tangents[i]``.
    """
    n = len(points)
    min_sin = math.sin(math.radians(degenerate_angle_deg))

    # --- screen up: anterior where defined, rotation-minimizing transport elsewhere
    along_a = tangents @ ANTERIOR
    projected = ANTERIOR[np.newaxis, :] - along_a[:, np.newaxis] * tangents
    strength = np.linalg.norm(projected, axis=1)
    defined = strength >= min_sin
    ups = np.zeros_like(points)
    ups[defined] = projected[defined] / strength[defined, np.newaxis]
    for i0, i1 in _runs(~defined):
        if i1 + 1 < n:  # transport backwards from the distal well-defined sample
            idx = np.arange(i1 + 1, i0 - 1, -1)
            normals = _transport(points, -tangents, ups[i1 + 1], idx)
            ups[idx[1:]] = normals[1:]
        elif i0 - 1 >= 0:  # run reaches the distal end: transport forwards
            idx = np.arange(i0 - 1, i1 + 1)
            normals = _transport(points, tangents, ups[i0 - 1], idx)
            ups[idx[1:]] = normals[1:]
        else:  # the whole curve runs along the A-P axis
            start = initial_normal(tangents[0], SUPERIOR)
            ups[:] = _transport(points, tangents, start, np.arange(n))

    # --- viewing side: from the feet (towards the head) where defined, inherited elsewhere
    along_s = tangents @ SUPERIOR
    signs = np.where(along_s >= 0.0, 1.0, -1.0)
    sign_defined = np.abs(along_s) >= min_sin
    for i0, i1 in _runs(~sign_defined):
        if i1 + 1 < n:
            signs[i0 : i1 + 1] = signs[i1 + 1]
        elif i0 - 1 >= 0:
            signs[i0 : i1 + 1] = signs[i0 - 1]
        else:
            signs[:] = 1.0
    return ups, signs


class CurvePath:
    """A centerline curve with arc-length parameterization and view frames."""

    def __init__(
        self,
        points: np.ndarray,
        step_mm: float = 0.5,
        degenerate_angle_deg: float = DEFAULT_DEGENERATE_ANGLE_DEG,
    ) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected points of shape (N, 3), got {pts.shape}")
        keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("a curve needs at least two distinct points")
        self.points = resample_polyline(pts, step_mm)
        if len(self.points) < 3:  # second-order tangents need three samples
            self.points = resample_polyline(pts, float(np.linalg.norm(pts[-1] - pts[0])) / 2.0)
        self.arc_length = cumulative_arc_length(self.points)
        self.tangents = unit_tangents(self.points)
        self.ups, self.view_signs = cross_section_orientation(self.points, self.tangents, degenerate_angle_deg)

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    def clamp(self, s: float) -> float:
        return float(min(max(s, 0.0), self.length))

    def _locate(self, s: float) -> Tuple[int, float]:
        s = self.clamp(s)
        i = int(np.searchsorted(self.arc_length, s, side="right") - 1)
        i = min(max(i, 0), len(self.arc_length) - 2)
        seg = self.arc_length[i + 1] - self.arc_length[i]
        f = 0.0 if seg <= 0 else (s - self.arc_length[i]) / seg
        return i, float(min(max(f, 0.0), 1.0))

    def point_at(self, s: float) -> np.ndarray:
        i, f = self._locate(s)
        return (1.0 - f) * self.points[i] + f * self.points[i + 1]

    def tangent_at(self, s: float) -> np.ndarray:
        i, f = self._locate(s)
        t = (1.0 - f) * self.tangents[i] + f * self.tangents[i + 1]
        return t / np.linalg.norm(t)

    def cross_section_frame(self, s: float) -> ViewFrame:
        """Perpendicular view at arc length ``s``: anterior up, patient's left on the right."""
        i, f = self._locate(s)
        t = self.tangent_at(s)
        up = (1.0 - f) * self.ups[i] + f * self.ups[i + 1]
        up = up - np.dot(up, t) * t
        norm = np.linalg.norm(up)
        if norm < 1e-9:  # interpolation between opposite vectors at a jump: use the nearer sample
            up = self.ups[i if f < 0.5 else i + 1]
            up = up - np.dot(up, t) * t
            norm = np.linalg.norm(up)
        up = up / norm
        sign = self.view_signs[i] if f < 0.5 else self.view_signs[i + 1]
        view_dir = sign * t
        x_axis = np.cross(view_dir, up)
        return ViewFrame(origin=self.point_at(s), x_axis=x_axis, y_axis=up, normal=np.cross(x_axis, up))

    def longitudinal_frame(self, s: float, angle_deg: float = 0.0) -> ViewFrame:
        """Plane containing the tangent at ``s``, rotated by ``angle_deg`` about it; head up."""
        cs = self.cross_section_frame(s)
        view_dir = np.cross(cs.y_axis, cs.x_axis)  # = sign * tangent (towards the head)
        a = math.radians(angle_deg)
        x_axis = math.cos(a) * cs.x_axis + math.sin(a) * cs.y_axis
        y_axis = view_dir
        return ViewFrame(origin=cs.origin, x_axis=x_axis, y_axis=y_axis, normal=np.cross(x_axis, y_axis))

    def nearest_arc_length(self, point: np.ndarray) -> Tuple[float, float]:
        """Arc length of the curve point closest to ``point`` and the distance to it (mm)."""
        p = np.asarray(point, dtype=float)
        a = self.points[:-1]
        ab = self.points[1:] - a
        seg_len2 = np.einsum("ij,ij->i", ab, ab)
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(seg_len2, 1e-18), 0.0, 1.0)
        proj = a + u[:, np.newaxis] * ab
        dist = np.linalg.norm(p - proj, axis=1)
        i = int(np.argmin(dist))
        s = self.arc_length[i] + u[i] * math.sqrt(seg_len2[i])
        return float(s), float(dist[i])

    def closest_arc_length_to_ray(
        self, origin: np.ndarray, direction: np.ndarray, max_distance_mm: float
    ) -> Optional[Tuple[float, float]]:
        """Curve sample closest to a viewing ray (a click in a 3-D view).

        Only samples in front of ``origin`` and within ``max_distance_mm`` of the ray count.
        Returns ``(arc length, distance to the ray)`` or ``None``.
        """
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        rel = self.points - o
        along = rel @ d
        dist = np.linalg.norm(rel - along[:, np.newaxis] * d, axis=1)
        dist = np.where((along > 0.0) & (dist <= max_distance_mm), dist, np.inf)
        i = int(np.argmin(dist))
        if not np.isfinite(dist[i]):
            return None
        return float(self.arc_length[i]), float(dist[i])

    def frames_continuity(self, step_mm: Optional[float] = None) -> np.ndarray:
        """Angles (deg) between screen-up vectors of consecutive cross-sections (diagnostics/tests)."""
        step = step_mm or float(np.median(np.diff(self.arc_length)))
        s_values = np.arange(0.0, self.length, step)
        ups = np.array([self.cross_section_frame(s).y_axis for s in s_values])
        dots = np.clip(np.einsum("ij,ij->i", ups[:-1], ups[1:]), -1.0, 1.0)
        return np.degrees(np.arccos(dots))
