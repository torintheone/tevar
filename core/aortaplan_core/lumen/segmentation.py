"""Contrast-filled lumen from a few seed points (SPEC 6.2, items 2-3).

Algorithm:

1. Lumen intensity is estimated around every seed; the lower threshold is the half-way
   level between soft tissue and lumen (the 50 % partial-volume level, which puts the
   boundary at the true wall); the upper threshold excludes calcium and cortical bone.
2. Seeds are snapped to the lumen centre (a click on a 3-D rendering lands on the wall).
3. Region growing: voxels in the HU range inside a box around the seeds, minus stop
   spheres and an optional exclusion mask (e.g. bones from TotalSegmentator).
4. Leak protection: a morphological opening cuts bridges one or two voxels thick (between
   the aorta and a vertebra or a rib); only components that contain a seed are kept; one
   step of geodesic reconstruction restores the boundary voxels removed by the opening.

Everything works on the ROI box only, so memory stays proportional to the vessels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from ..volume import ImageGeometry, Volume

ProgressCallback = Callable[[float, str], bool]  # (fraction 0..1, step key) -> continue?

_FACE_CONNECTIVITY = ndimage.generate_binary_structure(3, 1)


class Cancelled(Exception):
    """Raised when a progress callback asks to stop."""


@dataclass
class LumenParameters:
    roi_margin_mm: float = 30.0
    seed_radius_mm: float = 6.0  # neighbourhood used to estimate the lumen intensity
    snap_radius_mm: float = 8.0  # how far a seed may be moved into the lumen
    soft_tissue_hu: float = 40.0
    lower_fraction: float = 0.5  # lower threshold between soft tissue (0) and lumen (1)
    min_lower_hu: float = 120.0
    upper_margin_hu: float = 200.0  # upper threshold = lumen level + margin
    max_considered_hu: float = 700.0  # brighter voxels (calcium, cortex) ignored when estimating
    opening_iterations: int = 1
    reconstruction_steps: int = 1
    stop_radius_mm: float = 5.0
    fill_holes: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> LumenParameters:
        return cls(**d)


@dataclass
class LumenResult:
    """Lumen mask of the ROI box ``[k, j, i]`` starting at voxel ``roi_start_ijk`` of the volume."""

    mask: np.ndarray
    roi_start_ijk: Tuple[int, int, int]
    roi_geometry: ImageGeometry
    hu_range: Tuple[float, float]
    lumen_hu: float
    start: np.ndarray  # snapped start seed, RAS
    ends: List[np.ndarray]  # snapped end seeds, RAS
    connected: List[bool]  # every end seed connected to the start seed
    warnings: List[str] = field(default_factory=list)

    @property
    def voxel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def volume_ml(self) -> float:
        return self.voxel_count * self.roi_geometry.voxel_volume / 1000.0

    def full_mask(self, geometry: ImageGeometry) -> np.ndarray:
        full = np.zeros(geometry.array_shape, dtype=bool)
        i0, j0, k0 = self.roi_start_ijk
        nk, nj, ni = self.mask.shape
        full[k0 : k0 + nk, j0 : j0 + nj, i0 : i0 + ni] = self.mask
        return full

    def summary(self) -> Dict[str, object]:
        return {
            "hu_range": [float(x) for x in self.hu_range],
            "lumen_hu": float(self.lumen_hu),
            "voxel_count": self.voxel_count,
            "volume_ml": self.volume_ml,
            "start": self.start.tolist(),
            "ends": [e.tolist() for e in self.ends],
            "connected": list(self.connected),
            "warnings": list(self.warnings),
        }


def _check(progress: Optional[ProgressCallback], fraction: float, step: str) -> None:
    if progress is not None and progress(fraction, step) is False:
        raise Cancelled()


def _ball_ids(geometry: ImageGeometry, center_ras: np.ndarray, radius_mm: float, shape) -> Tuple[np.ndarray, ...]:
    """Array indices (k, j, i) of the voxels within ``radius_mm`` of a point (clipped to ``shape``)."""
    c = geometry.ras_to_ijk(center_ras)
    r_vox = radius_mm / np.asarray(geometry.spacing)
    lo = np.maximum(np.floor(c - r_vox).astype(int), 0)
    hi = np.minimum(np.ceil(c + r_vox).astype(int), np.array(shape)[::-1] - 1)
    if np.any(hi < lo):
        return (np.zeros(0, int),) * 3
    i, j, k = np.meshgrid(*(np.arange(lo[a], hi[a] + 1) for a in range(3)), indexing="ij")
    pts = geometry.ijk_to_ras(np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(float))
    inside = np.linalg.norm(pts - center_ras, axis=1) <= radius_mm
    return k.ravel()[inside], j.ravel()[inside], i.ravel()[inside]


def estimate_hu_range(
    volume: Volume, seeds: Sequence[np.ndarray], params: Optional[LumenParameters] = None
) -> Tuple[float, float, float]:
    """(lower, upper, lumen level) from the neighbourhoods of the seeds."""
    params = params or LumenParameters()
    levels = []
    for seed in seeds:
        values = volume.data[_ball_ids(volume.geometry, np.asarray(seed, float), params.seed_radius_mm, volume.data.shape)]
        values = values[values < params.max_considered_hu].astype(float)
        if values.size:
            levels.append(float(np.percentile(values, 90)))
    if not levels:
        raise ValueError("no usable voxels around the seed points")
    level = float(np.median(levels))
    lower = max(params.min_lower_hu, params.soft_tissue_hu + params.lower_fraction * (level - params.soft_tissue_hu))
    upper = level + params.upper_margin_hu
    return lower, upper, level


def snap_seed(
    volume: Volume, seed_ras: np.ndarray, hu_range: Tuple[float, float], radius_mm: float
) -> Optional[np.ndarray]:
    """Move a seed to the deepest lumen voxel (largest distance to the boundary) nearby.

    Returns ``None`` if no voxel in the HU range is within ``radius_mm``.
    """
    g = volume.geometry
    seed = np.asarray(seed_ras, dtype=float)
    box = radius_mm + 25.0  # room for the distance transform of a large lumen
    c = g.ras_to_ijk(seed)
    half = np.ceil(box / np.asarray(g.spacing)).astype(int)
    dims = np.array(g.dimensions)
    lo = np.maximum(np.floor(c).astype(int) - half, 0)
    hi = np.minimum(np.floor(c).astype(int) + half, dims - 1)
    if np.any(hi < lo):
        return None
    sub = volume.data[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
    in_range = (sub >= hu_range[0]) & (sub <= hu_range[1])
    if not in_range.any():
        return None
    labels, _ = ndimage.label(in_range, structure=_FACE_CONNECTIVITY)
    sub_geometry = g.subgrid(lo, hi - lo + 1)
    near = _ball_ids(sub_geometry, seed, radius_mm, sub.shape)
    candidates = labels[near]
    if not np.any(candidates):
        return None
    # The component nearest to the click, then its deepest voxel within the snap radius.
    pts = sub_geometry.ijk_to_ras(np.stack([near[2], near[1], near[0]], axis=1).astype(float))
    dist = np.linalg.norm(pts - seed, axis=1)
    dist[candidates == 0] = np.inf
    component = candidates[int(np.argmin(dist))]
    depth = ndimage.distance_transform_edt(labels == component, sampling=tuple(np.asarray(g.spacing)[::-1]))
    depth_near = np.where(candidates == component, depth[near], -1.0)
    # Among the (nearly) deepest voxels take the one closest to the click, so the seed moves
    # across the vessel to its centre but not along it.
    deep = depth_near >= 0.9 * depth_near.max()
    best = int(np.argmin(np.where(deep, dist, np.inf)))
    return pts[best]


def _grow(volume, pts, margin, lower, upper, stops, exclusion_mask, params, progress, fraction0):
    """Region growing in a box around the seeds (``margin`` None = whole volume)."""
    g = volume.geometry
    dims = np.array(g.dimensions)
    if margin is None:
        roi_lo, roi_hi = np.zeros(3, dtype=int), dims - 1
    else:
        lo_ras, hi_ras = pts.min(axis=0) - margin, pts.max(axis=0) + margin
        corners = np.array(
            [[x, y, z] for x in (lo_ras[0], hi_ras[0]) for y in (lo_ras[1], hi_ras[1]) for z in (lo_ras[2], hi_ras[2])]
        )
        ijk = g.ras_to_ijk(corners)
        roi_lo = np.clip(np.floor(ijk.min(axis=0)).astype(int), 0, dims - 1)
        roi_hi = np.clip(np.ceil(ijk.max(axis=0)).astype(int), 0, dims - 1)
    box = (slice(roi_lo[2], roi_hi[2] + 1), slice(roi_lo[1], roi_hi[1] + 1), slice(roi_lo[0], roi_hi[0] + 1))
    _check(progress, fraction0, "lumen.threshold")
    data = volume.data[box]
    roi_geometry = g.subgrid(roi_lo, roi_hi - roi_lo + 1)
    mask = (data >= lower) & (data <= upper)
    if exclusion_mask is not None:
        mask &= ~exclusion_mask[box]
    for stop in stops:
        mask[_ball_ids(roi_geometry, np.asarray(stop, float), params.stop_radius_mm, mask.shape)] = False

    _check(progress, fraction0 + 0.05, "lumen.opening")
    opened = (
        ndimage.binary_opening(mask, structure=_FACE_CONNECTIVITY, iterations=params.opening_iterations)
        if params.opening_iterations > 0
        else mask.copy()
    )

    _check(progress, fraction0 + 0.1, "lumen.connectivity")
    seed_ijk = [np.round(roi_geometry.ras_to_ijk(p)).astype(int) for p in pts]
    labels, _ = ndimage.label(opened, structure=_FACE_CONNECTIVITY)
    keep = set()
    for i, j, k in seed_ijk:
        k0, k1 = max(k - 1, 0), min(k + 2, labels.shape[0])
        j0, j1 = max(j - 1, 0), min(j + 2, labels.shape[1])
        i0, i1 = max(i - 1, 0), min(i + 2, labels.shape[2])
        values = labels[k0:k1, j0:j1, i0:i1]
        values = values[values > 0]
        if values.size:
            keep.add(int(np.bincount(values).argmax()))
    lumen = np.isin(labels, sorted(keep)) if keep else np.zeros_like(mask)
    del labels
    if params.reconstruction_steps > 0 and lumen.any():
        lumen = ndimage.binary_dilation(
            lumen, structure=_FACE_CONNECTIVITY, iterations=params.reconstruction_steps, mask=mask
        )
    if params.fill_holes and lumen.any():
        _check(progress, fraction0 + 0.15, "lumen.fill_holes")
        lumen = ndimage.binary_fill_holes(lumen)

    final_labels, _ = ndimage.label(lumen, structure=_FACE_CONNECTIVITY)

    def label_at(index):
        i, j, k = index
        if not (0 <= k < lumen.shape[0] and 0 <= j < lumen.shape[1] and 0 <= i < lumen.shape[2]):
            return 0
        return int(final_labels[k, j, i])

    start_label = label_at(seed_ijk[0])
    connected = [start_label > 0 and label_at(index) == start_label for index in seed_ijk[1:]]
    return lumen, roi_lo, roi_geometry, seed_ijk, start_label, connected


def segment_lumen(
    volume: Volume,
    start: np.ndarray,
    ends: Sequence[np.ndarray],
    stops: Sequence[np.ndarray] = (),
    exclusion_mask: Optional[np.ndarray] = None,
    params: Optional[LumenParameters] = None,
    progress: Optional[ProgressCallback] = None,
) -> LumenResult:
    """Segment the contrast-filled lumen connecting the start seed with the end seeds."""
    params = params or LumenParameters()
    seeds = [np.asarray(start, dtype=float)] + [np.asarray(e, dtype=float) for e in ends]
    if len(seeds) < 2:
        raise ValueError("a start point and at least one end point are needed")
    warnings: List[str] = []

    _check(progress, 0.05, "lumen.hu_range")
    lower, upper, level = estimate_hu_range(volume, seeds, params)
    snapped = []
    for seed in seeds:
        moved = snap_seed(volume, seed, (lower, upper), params.snap_radius_mm)
        if moved is None:
            warnings.append("seed_outside_lumen")
            moved = seed
        snapped.append(moved)

    # The region of interest grows until every end point is connected to the start point
    # (curved vessels such as the aortic arch leave the box around the seeds).
    pts = np.array(snapped)
    margins: List[Optional[float]] = [params.roi_margin_mm * f for f in (1, 2, 4)] + [None]
    for attempt, margin in enumerate(margins):
        fraction0 = 0.15 + 0.2 * attempt
        result = _grow(volume, pts, margin, lower, upper, stops, exclusion_mask, params, progress, fraction0)
        mask, roi_lo, roi_geometry, seed_ijk, start_label, connected = result
        if start_label > 0 and all(connected):
            break
    if start_label == 0:
        warnings.append("start_not_in_lumen")
    if not all(connected):
        warnings.append("end_not_connected")
    _check(progress, 1.0, "lumen.done")
    return LumenResult(
        mask=mask,
        roi_start_ijk=tuple(int(x) for x in roi_lo),  # type: ignore[arg-type]
        roi_geometry=roi_geometry,
        hu_range=(lower, upper),
        lumen_hu=level,
        start=snapped[0],
        ends=snapped[1:],
        connected=connected,
        warnings=warnings,
    )
