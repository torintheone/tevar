"""Lengths along the greater and lesser curvature (SPEC section 4, "Lengths").

On every cross-section between two positions the direction towards the centre of
curvature of the centerline is found; the lumen wall point in that direction lies on the
lesser curvature, the opposite one on the greater curvature. The two lines through these
points are measured like the centerline.

The curvature direction comes from the smoothed second derivative of the centerline.
Where the vessel is almost straight it is undefined; there it is carried over from the
nearest curved part, so the wall lines do not zigzag across the vessel (on a straight
part any wall line has the same length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy import ndimage

from ..centerline.navigation import CurvePath
from ..geometry.curves import polyline_length
from ..volume import Volume
from .contours import ContourParameters, SectionPlane, lumen_contour

MIN_CURVATURE_PER_MM = 1.0 / 1000.0  # radius of curvature above 1 m counts as straight


@dataclass
class CurvatureLengths:
    centerline_mm: float
    greater_mm: Optional[float]
    lesser_mm: Optional[float]
    straight_mm: float
    greater_points: np.ndarray
    lesser_points: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        def r(x):
            return None if x is None else round(float(x), 3)

        return {
            "centerline_mm": r(self.centerline_mm),
            "greater_curvature_mm": r(self.greater_mm),
            "lesser_curvature_mm": r(self.lesser_mm),
            "straight_mm": r(self.straight_mm),
        }


def curvature_directions(path: CurvePath, s_values: np.ndarray, smoothing_mm: float = 10.0) -> np.ndarray:
    """Unit in-plane directions towards the centre of curvature at ``s_values`` (NaN if straight)."""
    step = float(np.median(np.diff(path.arc_length)))
    tangents = path.tangents
    if smoothing_mm > 0:
        tangents = ndimage.gaussian_filter1d(tangents, smoothing_mm / step, axis=0, mode="nearest")
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    k = np.gradient(tangents, path.arc_length, axis=0)  # curvature vector dT/ds
    out = np.full((len(s_values), 3), np.nan)
    for i, s in enumerate(s_values):
        kv = np.array([np.interp(s, path.arc_length, k[:, c]) for c in range(3)])
        t = path.tangent_at(s)
        kv -= np.dot(kv, t) * t
        norm = float(np.linalg.norm(kv))
        if norm >= MIN_CURVATURE_PER_MM:
            out[i] = kv / norm
    # Straight parts: carry the nearest defined direction, projected on the local plane.
    defined = np.flatnonzero(np.isfinite(out[:, 0]))
    if len(defined) == 0:
        return out
    for i in np.flatnonzero(~np.isfinite(out[:, 0])):
        j = defined[int(np.argmin(np.abs(defined - i)))]
        t = path.tangent_at(s_values[i])
        d = out[j] - np.dot(out[j], t) * t
        if np.linalg.norm(d) > 1e-6:
            out[i] = d / np.linalg.norm(d)
    return out


def _ray_polygon_distance(direction: np.ndarray, polygon: np.ndarray) -> Optional[float]:
    """Distance from the origin along ``direction`` (2-D) to a closed polygon."""
    p = polygon
    q = np.roll(polygon, -1, axis=0)
    e = q - p
    # Solve origin + t * d = p + u * e for every edge.
    det = direction[0] * (-e[:, 1]) - direction[1] * (-e[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (p[:, 0] * (-e[:, 1]) - p[:, 1] * (-e[:, 0])) / det
        u = (direction[0] * p[:, 1] - direction[1] * p[:, 0]) / det
    ok = np.isfinite(t) & (t > 0) & (u >= 0) & (u <= 1)
    if not ok.any():
        return None
    return float(np.min(t[ok]))


def curvature_lengths(
    volume: Volume,
    path: CurvePath,
    s_start: float,
    s_end: float,
    step_mm: float = 1.0,
    smoothing_mm: float = 10.0,
    params: Optional[ContourParameters] = None,
) -> CurvatureLengths:
    """Lengths along the centerline, the greater and the lesser curvature between two positions."""
    s0, s1 = sorted((path.clamp(s_start), path.clamp(s_end)))
    n = max(2, int(math.ceil((s1 - s0) / step_mm)) + 1)
    s_values = np.linspace(s0, s1, n)
    params = params or ContourParameters(n_rays=120)
    directions = curvature_directions(path, s_values, smoothing_mm)
    greater = np.full((n, 3), np.nan)
    lesser = np.full((n, 3), np.nan)
    for i, s in enumerate(s_values):
        plane = SectionPlane.at(path, float(s))
        if not np.isfinite(directions[i, 0]):
            continue
        contour = lumen_contour(volume, plane, params)
        if contour is None:
            continue
        d2 = np.array([np.dot(directions[i], plane.u), np.dot(directions[i], plane.v)])
        d2 /= np.linalg.norm(d2)
        r_in = _ray_polygon_distance(d2, contour.points_2d)
        r_out = _ray_polygon_distance(-d2, contour.points_2d)
        if r_in is not None:
            lesser[i] = plane.to_ras(r_in * d2)[0]
        if r_out is not None:
            greater[i] = plane.to_ras(-r_out * d2)[0]

    def length(points: np.ndarray) -> Optional[float]:
        ok = np.isfinite(points[:, 0])
        if ok.sum() < max(2, 0.8 * n):
            return None
        # Missing sections are bridged by the straight segment between their neighbours.
        return polyline_length(points[ok])

    straight = float(np.linalg.norm(path.point_at(s1) - path.point_at(s0)))
    if not np.isfinite(directions[:, 0]).any():
        # Straight vessel: every wall line is as long as the centerline.
        return CurvatureLengths(s1 - s0, s1 - s0, s1 - s0, straight, greater, lesser)
    return CurvatureLengths(s1 - s0, length(greater), length(lesser), straight, greater, lesser)
