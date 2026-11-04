"""Polyline utilities: arc length, resampling, tangents. All points are RAS mm, shape (N, 3)."""

from __future__ import annotations

import numpy as np


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected points of shape (N, 3), got {pts.shape}")
    return pts


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    """Arc length from the first point to each point, shape (N,), starting at 0."""
    pts = _as_points(points)
    if len(pts) == 0:
        return np.zeros(0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_length(points: np.ndarray) -> float:
    """Total length of a polyline in mm."""
    s = cumulative_arc_length(points)
    return float(s[-1]) if len(s) else 0.0


def interpolate_at_arc_length(points: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Linearly interpolated points at the given arc-length positions (clamped to the curve)."""
    pts = _as_points(points)
    s = cumulative_arc_length(pts)
    q = np.clip(np.asarray(s_query, dtype=float), 0.0, s[-1])
    return np.stack([np.interp(q, s, pts[:, d]) for d in range(3)], axis=-1)


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at (nearly) uniform arc-length spacing ``<= step``; keeps both end points."""
    if step <= 0:
        raise ValueError("step must be positive")
    pts = _as_points(points)
    total = polyline_length(pts)
    if total == 0.0:
        return pts[:1].copy()
    n_seg = max(1, int(np.ceil(total / step - 1e-9)))
    return interpolate_at_arc_length(pts, np.linspace(0.0, total, n_seg + 1))


def unit_tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangent vectors (second-order finite differences with respect to arc length)."""
    pts = _as_points(points)
    if len(pts) < 2:
        raise ValueError("need at least two points to compute tangents")
    s = cumulative_arc_length(pts)
    if np.any(np.diff(s) <= 0):
        raise ValueError("polyline has repeated consecutive points")
    t = np.gradient(pts, s, axis=0, edge_order=2 if len(pts) > 2 else 1)
    return t / np.linalg.norm(t, axis=1, keepdims=True)
