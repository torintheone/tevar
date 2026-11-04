"""Rotation-minimizing (parallel-transport) frames along a polyline.

A rotation-minimizing frame (RMF) carries a normal vector along the curve with no
rotation about the tangent. Unlike the Frenet frame it is defined on straight
segments and does not flip at inflection points, which makes it the reference for
clock positions in the aortic arch (SPEC section 4).

Method: double reflection, W. Wang, B. Juettler, D. Zheng, Y. Liu,
"Computation of rotation minimizing frames", ACM TOG 27(1), 2008.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .curves import unit_tangents


def initial_normal(tangent: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit vector perpendicular to ``tangent``, as close as possible to ``reference``.

    If ``reference`` is ``None`` or (nearly) parallel to the tangent, the RAS axis
    that is most perpendicular to the tangent is used instead.
    """
    t = np.asarray(tangent, dtype=float)
    t = t / np.linalg.norm(t)
    candidates = []
    if reference is not None:
        candidates.append(np.asarray(reference, dtype=float))
    candidates.extend(np.eye(3)[np.argsort(np.abs(t))])  # most perpendicular axis first
    for ref in candidates:
        n = ref - np.dot(ref, t) * t
        norm = np.linalg.norm(n)
        if norm > 1e-3 * max(1.0, np.linalg.norm(ref)):
            return n / norm
    raise ValueError("could not build a normal vector")  # unreachable for a unit tangent


def rotation_minimizing_frames(
    points: np.ndarray,
    reference_normal: Optional[np.ndarray] = None,
    tangents: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimizing frames at each polyline point.

    Args:
        points: (N, 3) RAS points.
        reference_normal: vector whose projection on the first normal plane gives the
            first normal (e.g. the Anterior direction for clock positions).
        tangents: optional (N, 3) unit tangents; estimated from the points if omitted.

    Returns:
        ``(T, N, B)``: three (N, 3) arrays of unit tangents, normals and binormals,
        with ``B = T x N`` (right-handed).
    """
    pts = np.asarray(points, dtype=float)
    t = unit_tangents(pts) if tangents is None else np.asarray(tangents, dtype=float)
    if t.shape != pts.shape:
        raise ValueError("tangents must have the same shape as points")
    n_pts = len(pts)
    normals = np.empty_like(pts)
    normals[0] = initial_normal(t[0], reference_normal)
    for i in range(n_pts - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-24:
            normals[i + 1] = normals[i]
            continue
        r_l = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        t_l = t[i] - (2.0 / c1) * np.dot(v1, t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = float(np.dot(v2, v2))
        r_next = r_l if c2 < 1e-24 else r_l - (2.0 / c2) * np.dot(v2, r_l) * v2
        # Remove numerical drift: keep the normal exactly perpendicular and unit length.
        r_next = r_next - np.dot(r_next, t[i + 1]) * t[i + 1]
        normals[i + 1] = r_next / np.linalg.norm(r_next)
    binormals = np.cross(t, normals)
    return t, normals, binormals
