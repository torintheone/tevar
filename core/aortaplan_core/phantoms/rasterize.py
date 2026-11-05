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

    n_vox = int(np.prod(geometry.array_shape))
    half_diag = 0.5 * float(np.linalg.norm(geometry.spacing))
    max_radius = max(float(max(layer.semi_axis_n.max(), layer.semi_axis_b.max())) for layer in layers)
    max_step = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if n_points > 1 else 0.0
    reach = max_radius + 2.0 * half_diag + max_step
    fractions = np.zeros((len(layers), n_vox), dtype=np.float32)

    # Only voxels inside the tube's bounding box can be touched by it.
    flat_ids = voxels_in_box(geometry, pts.min(axis=0) - reach, pts.max(axis=0) + reach)
    if len(flat_ids) == 0:
        return fractions.reshape((len(layers),) + geometry.array_shape)
    centers = flat_voxel_centers(geometry, flat_ids)

    tree = cKDTree(pts)
    dist, idx = tree.query(centers, distance_upper_bound=reach)
    local = np.flatnonzero(np.isfinite(dist))
    if len(local) == 0:
        return fractions.reshape((len(layers),) + geometry.array_shape)
    centers = centers[local]
    candidates = flat_ids[local]
    idx = idx[local]

    c_idx = idx
    rel = centers - pts[c_idx]
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
            sub_points = (centers[chunk][:, np.newaxis, :] + sub_offsets[np.newaxis, :, :]).reshape(-1, 3)
            _, sub_idx = tree.query(sub_points)
            sub_rel = sub_points - pts[sub_idx]
            sub_g = _normalized_radius(sub_rel, sub_idx, normals, binormals, layer)
            sub_inside = (sub_g <= 1.0) & _within_caps(sub_rel, sub_idx, tangents, n_points)
            layer_fraction[chunk] = sub_inside.reshape(len(chunk), -1).mean(axis=1)

        fractions[li, candidates] = layer_fraction
    return fractions.reshape((len(layers),) + geometry.array_shape)


def voxels_in_box(geometry: ImageGeometry, ras_min: np.ndarray, ras_max: np.ndarray) -> np.ndarray:
    """Flat (C-order) indices of the voxels whose centres may lie in an RAS box."""
    lo, hi = np.asarray(ras_min, dtype=float), np.asarray(ras_max, dtype=float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ijk = geometry.ras_to_ijk(corners)
    dims = np.array(geometry.dimensions)
    start = np.clip(np.floor(ijk.min(axis=0)).astype(int), 0, dims - 1)
    stop = np.clip(np.ceil(ijk.max(axis=0)).astype(int), 0, dims - 1)
    if np.any(ijk.max(axis=0) < 0) or np.any(ijk.min(axis=0) > dims - 1):
        return np.zeros(0, dtype=np.int64)
    i = np.arange(start[0], stop[0] + 1)
    j = np.arange(start[1], stop[1] + 1)
    k = np.arange(start[2], stop[2] + 1)
    kk, jj, ii = np.meshgrid(k, j, i, indexing="ij")
    ni, nj, _ = geometry.dimensions
    return ((kk * nj + jj) * ni + ii).ravel().astype(np.int64)


def flat_voxel_centers(geometry: ImageGeometry, flat_ids: np.ndarray) -> np.ndarray:
    """RAS centres of voxels given by flat (C-order) indices."""
    ni, nj, _ = geometry.dimensions
    i = flat_ids % ni
    j = (flat_ids // ni) % nj
    k = flat_ids // (ni * nj)
    return geometry.ijk_to_ras(np.stack([i, j, k], axis=1).astype(float))


def solid_fraction(
    geometry: ImageGeometry, inside, ras_min: np.ndarray, ras_max: np.ndarray, supersampling: int = 3
) -> np.ndarray:
    """Partial-volume fraction of an arbitrary solid given by ``inside(points) -> bool``.

    Every voxel in the solid's bounding box is supersampled (fine for small objects such as
    a vertebra or calcifications). Returns a float32 array in array order ``[k, j, i]``.
    """
    fraction = np.zeros(int(np.prod(geometry.array_shape)), dtype=np.float32)
    half_diag = 0.5 * float(np.linalg.norm(geometry.spacing))
    flat_ids = voxels_in_box(geometry, np.asarray(ras_min) - half_diag, np.asarray(ras_max) + half_diag)
    if len(flat_ids) == 0:
        return fraction.reshape(geometry.array_shape)
    s = int(supersampling)
    lattice = (np.arange(s) + 0.5) / s - 0.5
    sub = np.stack(np.meshgrid(lattice, lattice, lattice, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets = sub @ (geometry.directions * np.asarray(geometry.spacing)[np.newaxis, :]).T
    for start in range(0, len(flat_ids), _CHUNK):
        chunk = flat_ids[start : start + _CHUNK]
        centers = flat_voxel_centers(geometry, chunk)
        points = (centers[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 3)
        fraction[chunk] = inside(points).reshape(len(chunk), -1).mean(axis=1)
    return fraction.reshape(geometry.array_shape)


def paint(image: np.ndarray, fraction: np.ndarray, hu: float) -> np.ndarray:
    """Painter's algorithm: the object covers ``fraction`` of each voxel with intensity ``hu``."""
    return image * (1.0 - fraction) + float(hu) * fraction


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
