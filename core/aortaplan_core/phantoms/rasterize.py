"""Rasterization of tubular phantoms with partial-volume effect.

A tube is a dense centerline polyline with a rotation-minimizing frame and one or
more nested elliptical layers (e.g. thrombus/wall outside, lumen inside). For every
voxel the fraction of its volume inside each layer is computed:

- voxels far from any layer boundary are classified by their centre;
- voxels within half a voxel diagonal of a boundary are supersampled
  (``supersampling**3`` sub-points), which models the box-shaped partial-volume
  effect of CT voxels.

Tube ends are flat caps perpendicular to the centerline at its first and last points.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..volume import ImageGeometry
from .model import TubeLayer

_CHUNK = 20000  # boundary voxels processed per batch (bounds memory use)


def _normalized_radius(
    rel: np.ndarray, idx: np.ndarray, normals: np.ndarray, binormals: np.ndarray, layer: TubeLayer
) -> np.ndarray:
    u = np.einsum("ij,ij->i", rel, normals[idx])
    v = np.einsum("ij,ij->i", rel, binormals[idx])
    return np.hypot(u / layer.semi_axis_n[idx], v / layer.semi_axis_b[idx])


def _within_caps(rel: np.ndarray, idx: np.ndarray, tangents: np.ndarray, n_points: int) -> np.ndarray:
    w = np.einsum("ij,ij->i", rel, tangents[idx])
    beyond_start = (idx == 0) & (w < 0.0)
    beyond_end = (idx == n_points - 1) & (w > 0.0)
    return ~(beyond_start | beyond_end)


def rasterize_tube_fractions(
    geometry: ImageGeometry,
    points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    layers: Sequence[TubeLayer],
    supersampling: int = 4,
) -> np.ndarray:
    """Fraction of each voxel inside each layer.

    Args:
        geometry: output grid.
        points, tangents, normals: dense centerline samples (N, 3) and their frame.
            The sample spacing should be much smaller than the voxel size.
        layers: nested layers, outermost first.
        supersampling: sub-points per axis for boundary voxels.

    Returns:
        float32 array of shape ``(len(layers), nk, nj, ni)`` with values in [0, 1].
    """
    pts = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    normals = np.asarray(normals, dtype=float)
    binormals = np.cross(tangents, normals)
    n_points = len(pts)
    layers = list(layers)
    for layer in layers:
        if layer.semi_axis_n.shape != (n_points,):
            raise ValueError(f"layer {layer.name!r}: semi-axes must have one value per centerline point")

    centers = geometry.voxel_centers_ras().reshape(-1, 3)
    n_vox = len(centers)
    half_diag = 0.5 * float(np.linalg.norm(geometry.spacing))
    max_radius = max(float(max(layer.semi_axis_n.max(), layer.semi_axis_b.max())) for layer in layers)
    max_step = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if n_points > 1 else 0.0

    tree = cKDTree(pts)
    dist, idx = tree.query(centers, distance_upper_bound=max_radius + 2.0 * half_diag + max_step)
    candidates = np.flatnonzero(np.isfinite(dist))
    fractions = np.zeros((len(layers), n_vox), dtype=np.float32)
    if len(candidates) == 0:
        return fractions.reshape((len(layers),) + geometry.array_shape)

    c_idx = idx[candidates]
    rel = centers[candidates] - pts[c_idx]
    caps = _within_caps(rel, c_idx, tangents, n_points)
    near_end = (c_idx == 0) | (c_idx == n_points - 1)

    # Sub-voxel offsets (in RAS) of a regular s x s x s lattice inside a voxel.
    s = int(supersampling)
    if s < 1:
        raise ValueError("supersampling must be >= 1")
    lattice = (np.arange(s) + 0.5) / s - 0.5
    sub = np.stack(np.meshgrid(lattice, lattice, lattice, indexing="ij"), axis=-1).reshape(-1, 3)
    step_matrix = geometry.directions * np.asarray(geometry.spacing)[np.newaxis, :]
    sub_offsets = sub @ step_matrix.T  # (s^3, 3)

    for li, layer in enumerate(layers):
        g = _normalized_radius(rel, c_idx, normals, binormals, layer)
        min_axis = np.minimum(layer.semi_axis_n[c_idx], layer.semi_axis_b[c_idx])
        # Distance between the level set g and the boundary g = 1 is >= |g - 1| * min_axis
        # (nested homothetic convex curves), so this band contains every boundary voxel.
        band = (np.abs(g - 1.0) * min_axis <= half_diag * 1.05 + 1e-9) | near_end
        inside_center = (g <= 1.0) & caps
        layer_fraction = np.where(inside_center, 1.0, 0.0)

        band_ids = np.flatnonzero(band)
        for start in range(0, len(band_ids), _CHUNK):
            chunk = band_ids[start : start + _CHUNK]
            sub_points = (centers[candidates[chunk]][:, np.newaxis, :] + sub_offsets[np.newaxis, :, :]).reshape(-1, 3)
            _, sub_idx = tree.query(sub_points)
            sub_rel = sub_points - pts[sub_idx]
            sub_g = _normalized_radius(sub_rel, sub_idx, normals, binormals, layer)
            sub_inside = (sub_g <= 1.0) & _within_caps(sub_rel, sub_idx, tangents, n_points)
            layer_fraction[chunk] = sub_inside.reshape(len(chunk), -1).mean(axis=1)

        fractions[li, candidates] = layer_fraction
    return fractions.reshape((len(layers),) + geometry.array_shape)


def compose_intensity(fractions: np.ndarray, layer_hu: Sequence[float], background_hu: float) -> np.ndarray:
    """Combine nested layer fractions (outermost first) into an intensity image (float64)."""
    image = np.full(fractions.shape[1:], float(background_hu))
    previous = float(background_hu)
    for frac, hu in zip(fractions, layer_hu):
        image += (float(hu) - previous) * frac
        previous = float(hu)
    return image


def layer_hus(layers: List[TubeLayer]) -> List[float]:
    return [float(layer.hu) for layer in layers]
