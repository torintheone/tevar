"""Comparison of a computed centerline with a reference (phantom truth, VMTK, 3mensio)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Exclusion = Tuple[Sequence[float], float]  # (centre RAS, radius mm)


def distances_to_polyline(points: np.ndarray, polyline: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance (mm) from every point to the nearest segment of a polyline."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    line = np.asarray(polyline, dtype=float)
    if len(line) == 1:
        return np.linalg.norm(pts - line[0], axis=1)
    a = line[:-1]
    ab = line[1:] - a
    len2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
    out = np.empty(len(pts))
    for start in range(0, len(pts), chunk):
        p = pts[start : start + chunk, np.newaxis, :]
        u = np.clip(np.einsum("pij,ij->pi", p - a, ab) / len2, 0.0, 1.0)
        proj = a + u[..., np.newaxis] * ab
        out[start : start + chunk] = np.min(np.linalg.norm(p - proj, axis=2), axis=1)
    return out


def _outside(points: np.ndarray, exclusions: Sequence[Exclusion]) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for center, radius in exclusions:
        keep &= np.linalg.norm(points - np.asarray(center, dtype=float), axis=1) > radius
    return keep


def centerline_deviation(
    points: np.ndarray,
    reference: np.ndarray,
    exclusions: Sequence[Exclusion] = (),
) -> Dict[str, float]:
    """Deviation of computed points from a reference polyline, outside the exclusion spheres."""
    pts = np.asarray(points, dtype=float)
    pts = pts[_outside(pts, exclusions)]
    d = distances_to_polyline(pts, reference)
    if len(d) == 0:
        return {"mean_mm": float("nan"), "max_mm": float("nan"), "p95_mm": float("nan"), "count": 0}
    return {
        "mean_mm": float(d.mean()),
        "max_mm": float(d.max()),
        "p95_mm": float(np.percentile(d, 95)),
        "count": int(len(d)),
    }


def coverage(
    reference: np.ndarray,
    points: np.ndarray,
    tolerance_mm: float,
    exclusions: Sequence[Exclusion] = (),
    region: Optional[np.ndarray] = None,
) -> float:
    """Fraction of reference samples (outside exclusions, inside ``region`` mask) near the computed line."""
    ref = np.asarray(reference, dtype=float)
    keep = _outside(ref, exclusions)
    if region is not None:
        keep &= region
    if not keep.any():
        return float("nan")
    return float(np.mean(distances_to_polyline(ref[keep], points) <= tolerance_mm))
