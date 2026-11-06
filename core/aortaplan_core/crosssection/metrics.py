"""Metrics of a closed cross-section contour (SPEC section 4, "diameters on a section").

For a closed contour (polygon) in the section plane:

- area (shoelace formula) and perimeter;
- minimum and maximum Feret diameters (rotating calipers on the convex hull): the maximum
  is the largest distance between two contour points, the minimum is the smallest width
  between two parallel support lines;
- mean diameter = (min + max) / 2 (the convention of EVAR planning software; for an
  ellipse it equals the sum of the semi-axes);
- area-equivalent diameter ``2 * sqrt(A / pi)`` and perimeter-equivalent diameter ``P / pi``.

The Feret diameters are returned together with segments that can be drawn on the image.
Metrics of a 3-D contour are computed after projection on its plane (``plane_coordinates``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

METRIC_NAMES = (
    "min_diameter_mm",
    "max_diameter_mm",
    "mean_diameter_mm",
    "area_equivalent_diameter_mm",
    "perimeter_equivalent_diameter_mm",
    "area_mm2",
    "perimeter_mm",
)


@dataclass
class ContourMetrics:
    """Cross-section metrics; lengths in mm, area in mm^2.

    ``centroid``, ``max_feret_segment`` (2 points) and ``min_feret_segment`` (2 points) are
    in the coordinates of the input contour (2-D plane coordinates or RAS).
    """

    area_mm2: float
    perimeter_mm: float
    min_diameter_mm: float
    max_diameter_mm: float
    mean_diameter_mm: float
    area_equivalent_diameter_mm: float
    perimeter_equivalent_diameter_mm: float
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    max_feret_segment: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    min_feret_segment: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    def values(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in METRIC_NAMES}

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {name: round(float(getattr(self, name)), 4) for name in METRIC_NAMES}
        d["centroid"] = np.round(self.centroid, 4).tolist()
        d["max_feret_segment"] = np.round(self.max_feret_segment, 4).tolist()
        d["min_feret_segment"] = np.round(self.min_feret_segment, 4).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ContourMetrics:
        kwargs = {name: float(d[name]) for name in METRIC_NAMES}
        return cls(
            centroid=np.asarray(d.get("centroid", [0.0, 0.0]), dtype=float),
            max_feret_segment=np.asarray(d.get("max_feret_segment", np.zeros((2, 2))), dtype=float),
            min_feret_segment=np.asarray(d.get("min_feret_segment", np.zeros((2, 2))), dtype=float),
            **kwargs,
        )

    def transformed(self, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> ContourMetrics:
        """Copy with 2-D plane coordinates mapped to 3-D: ``origin + x * u + y * v``."""

        def to3(p: np.ndarray) -> np.ndarray:
            p = np.atleast_2d(p)
            return np.asarray(origin)[np.newaxis, :] + p[:, :1] * np.asarray(u) + p[:, 1:2] * np.asarray(v)

        return ContourMetrics(
            **self.values(),
            centroid=to3(self.centroid)[0],
            max_feret_segment=to3(self.max_feret_segment),
            min_feret_segment=to3(self.min_feret_segment),
        )


def _as_closed_polygon(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"expected 2-D points of shape (N, 2), got {p.shape}")
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]  # the closing point is implicit
    if len(p) < 3:
        raise ValueError("a contour needs at least three points")
    return p


def signed_area(points: np.ndarray) -> float:
    """Shoelace area, positive for counter-clockwise polygons."""
    p = _as_closed_polygon(points)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def perimeter(points: np.ndarray) -> float:
    p = _as_closed_polygon(points)
    return float(np.sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)))


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    """Area centroid of a simple polygon (vertex mean for degenerate polygons)."""
    p = _as_closed_polygon(points)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * float(np.sum(cross))
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    return np.array([np.sum((x + xn) * cross), np.sum((y + yn) * cross)]) / (6.0 * a)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull vertices in counter-clockwise order (Andrew's monotone chain)."""
    p = np.unique(np.asarray(points, dtype=float), axis=0)  # sorted by x, then y
    if len(p) < 3:
        return p

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper = []
    for point in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def feret_diameters(points: np.ndarray) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """Minimum and maximum Feret diameters with drawable segments.

    Returns ``(min_width, min_segment, max_diameter, max_segment)``; each segment is a
    (2, 2) array. The minimum-width segment goes from the hull vertex farthest from a hull
    edge perpendicularly to that edge's support line (rotating calipers: the minimum width
    of a convex polygon is attained with one caliper flush with an edge).
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        raise ValueError("degenerate contour")
    # Maximum Feret: farthest pair of hull vertices.
    diff = hull[:, np.newaxis, :] - hull[np.newaxis, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    max_d = math.sqrt(float(d2[i, j]))
    max_segment = np.array([hull[i], hull[j]])
    # Minimum Feret: for each edge, the largest distance of a vertex from its line.
    a = hull
    e = np.roll(hull, -1, axis=0) - hull
    e /= np.linalg.norm(e, axis=1)[:, np.newaxis]
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)  # inward normal of a counter-clockwise hull
    dist = np.einsum("ijk,ik->ij", hull[np.newaxis, :, :] - a[:, np.newaxis, :], n)  # (edge, vertex)
    widths = dist.max(axis=1)
    k = int(np.argmin(widths))
    far = hull[int(np.argmax(dist[k]))]
    min_w = float(widths[k])
    min_segment = np.array([far, far - min_w * n[k]])
    return min_w, min_segment, max_d, max_segment


def contour_metrics(points_2d: np.ndarray) -> ContourMetrics:
    """All section metrics of a closed 2-D contour (mm)."""
    p = _as_closed_polygon(points_2d)
    area = abs(signed_area(p))
    per = perimeter(p)
    min_w, min_seg, max_d, max_seg = feret_diameters(p)
    return ContourMetrics(
        area_mm2=area,
        perimeter_mm=per,
        min_diameter_mm=min_w,
        max_diameter_mm=max_d,
        mean_diameter_mm=0.5 * (min_w + max_d),
        area_equivalent_diameter_mm=2.0 * math.sqrt(area / math.pi),
        perimeter_equivalent_diameter_mm=per / math.pi,
        centroid=polygon_centroid(p),
        max_feret_segment=max_seg,
        min_feret_segment=min_seg,
    )


def plane_coordinates(points_ras: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """2-D coordinates of 3-D points projected on the plane ``origin + x u + y v``."""
    rel = np.asarray(points_ras, dtype=float) - np.asarray(origin, dtype=float)[np.newaxis, :]
    return np.stack([rel @ np.asarray(u, dtype=float), rel @ np.asarray(v, dtype=float)], axis=1)


def contour_metrics_3d(
    points_ras: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray
) -> ContourMetrics:
    """Metrics of a 3-D contour in the plane (origin, u, v); geometry returned in RAS."""
    return contour_metrics(plane_coordinates(points_ras, origin, u, v)).transformed(origin, u, v)


def fit_plane(points: np.ndarray, normal_hint: Optional[np.ndarray] = None):
    """Least-squares plane of 3-D points: ``(centre, u, v, normal)`` (orthonormal u, v)."""
    p = np.asarray(points, dtype=float)
    centre = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centre)
    normal = vt[2]
    if normal_hint is not None and np.dot(normal, normal_hint) < 0:
        normal = -normal
    u = vt[0]
    v = np.cross(normal, u)
    return centre, u, v, normal
