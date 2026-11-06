"""Lumen and outer-wall contours on a cross-section (SPEC section 6.3).

Both contours are found by rays cast in the section plane from a centre inside the lumen
(one ray per degree by default) and read off the CT intensity profile along each ray,
with linear interpolation between samples (sub-voxel boundary positions).

Lumen
    The boundary is the 50 % level between the soft tissue around the lumen and the lumen's
    own contrast level, measured in this section (same rule as the lumen segmentation of
    stage 2, but evaluated locally and with sub-voxel accuracy). A ray also stops where the
    intensity rises well above the contrast level (calcification at the lumen boundary).
    After a first pass the rays are cast again from the centroid of the contour.

Outer wall (wall + thrombus)
    Beyond the lumen boundary each ray looks for the first drop below the 50 % level between
    the wall/thrombus and periaortic fat. If a bright structure (calcified plaque) lines
    the wall just before that drop, the edge of that structure is used instead (its own
    50 % level). Rays that run into a long bright structure (bone) or never meet fat (an
    adjacent organ without a fat plane) give no boundary.

Regularisation (both)
    Rays without a boundary and rays far outside a robust low-order Fourier fit of the
    radius (excursions into branch ostia, adjacent organs) are replaced by interpolation
    between their neighbours, then the radius is smoothed along the angle. Every contour
    remains editable: :class:`SectionContour` can also hold a manually edited polygon.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from ..centerline.navigation import CurvePath, ViewFrame
from ..volume import Volume
from .metrics import ContourMetrics, contour_metrics

LUMEN = "lumen"
OUTER = "outer"


@dataclass
class SectionPlane:
    """Cross-section plane: ``ras = origin + x * u + y * v``.

    ``u`` and ``v`` are the screen axes of the perpendicular view: ``u`` = image right
    (patient's left), ``v`` = image up (anterior), see ``centerline.navigation``.
    """

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    arc_length_mm: Optional[float] = None

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.v = np.asarray(self.v, dtype=float)

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.u, self.v)

    @classmethod
    def from_frame(cls, frame: ViewFrame, arc_length_mm: Optional[float] = None) -> SectionPlane:
        return cls(frame.origin.copy(), frame.x_axis.copy(), frame.y_axis.copy(), arc_length_mm)

    @classmethod
    def at(cls, path: CurvePath, s: float) -> SectionPlane:
        s = path.clamp(s)
        return cls.from_frame(path.cross_section_frame(s), s)

    def to_ras(self, points_2d: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_2d, dtype=float))
        return self.origin[np.newaxis, :] + p[:, :1] * self.u + p[:, 1:2] * self.v

    def to_plane(self, points_ras: np.ndarray) -> np.ndarray:
        rel = np.atleast_2d(np.asarray(points_ras, dtype=float)) - self.origin[np.newaxis, :]
        return np.stack([rel @ self.u, rel @ self.v], axis=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": np.round(self.origin, 5).tolist(),
            "u": np.round(self.u, 8).tolist(),
            "v": np.round(self.v, 8).tolist(),
            "arc_length_mm": None if self.arc_length_mm is None else round(float(self.arc_length_mm), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SectionPlane:
        return cls(d["origin"], d["u"], d["v"], d.get("arc_length_mm"))


@dataclass
class ContourParameters:
    n_rays: int = 360
    step_mm: float = 0.2  # profile sampling; the boundary is interpolated between samples
    max_lumen_radius_mm: float = 40.0
    max_outer_radius_mm: float = 60.0
    # Lumen: boundary at soft + fraction * (contrast - soft), like the stage 2 segmentation.
    soft_tissue_hu: float = 40.0
    lower_fraction: float = 0.5
    min_lower_hu: float = 120.0
    max_considered_hu: float = 700.0  # brighter samples are ignored when estimating the contrast level
    calcium_margin_hu: float = 200.0  # a ray stops where it rises above contrast + margin
    # Outer wall: boundary at the 50 % level between wall/thrombus and fat.
    wall_hu: float = 40.0
    fat_hu: float = -100.0
    outer_search_start_mm: float = 0.5  # beyond the lumen boundary
    bright_edge_hu: float = 140.0  # a plaque brighter than this at the outer edge sets its own level
    bright_window_mm: float = 2.5
    bone_hu: float = 250.0  # a bright run longer than bone_run_mm before fat = bone, no boundary
    bone_run_mm: float = 4.0
    min_wall_mm: float = 0.5
    fallback_wall_mm: float = 2.0
    # Regularisation
    fourier_order: int = 4
    outlier_mm: float = 3.0
    outlier_fraction: float = 0.25
    smoothing_deg: float = 3.0
    guard_deg: float = 4.0  # rays rejected next to every rejected run
    min_valid_fraction: float = 0.3
    recenter_mm: float = 0.3  # cast the lumen rays again from the centroid if it moved more

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ContourParameters:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class SectionContour:
    """A closed contour in a section plane (2-D plane coordinates, mm).

    ``source`` is ``"auto"`` for a contour found by rays (then ``center``, ``angles``,
    ``radii`` and ``measured`` describe the rays) or ``"manual"`` after editing.
    """

    kind: str
    plane: SectionPlane
    points_2d: np.ndarray
    source: str = "auto"
    threshold_hu: Optional[float] = None
    contrast_hu: Optional[float] = None
    center: Optional[np.ndarray] = None
    angles: Optional[np.ndarray] = None
    radii: Optional[np.ndarray] = None
    measured: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points_2d = np.asarray(self.points_2d, dtype=float)

    @property
    def points_ras(self) -> np.ndarray:
        return self.plane.to_ras(self.points_2d)

    @property
    def measured_fraction(self) -> float:
        if self.measured is None or len(self.measured) == 0:
            return 1.0
        return float(np.mean(self.measured))

    def metrics(self) -> ContourMetrics:
        """Metrics with geometry in plane coordinates."""
        return contour_metrics(self.points_2d)

    def metrics_ras(self) -> ContourMetrics:
        """Metrics with geometry (centroid, Feret segments) in RAS."""
        return self.metrics().transformed(self.plane.origin, self.plane.u, self.plane.v)

    def resampled(self, n_points: int) -> np.ndarray:
        """``n_points`` points equally spaced along the contour (e.g. control points to edit)."""
        p = np.vstack([self.points_2d, self.points_2d[:1]])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(0.0, s[-1], int(n_points), endpoint=False)
        return np.stack([np.interp(targets, s, p[:, 0]), np.interp(targets, s, p[:, 1])], axis=1)

    def edited(self, points_ras: np.ndarray) -> SectionContour:
        """Manual version of this contour from edited 3-D points (projected on the plane)."""
        return SectionContour(
            kind=self.kind,
            plane=self.plane,
            points_2d=self.plane.to_plane(points_ras),
            source="manual",
            threshold_hu=self.threshold_hu,
            contrast_hu=self.contrast_hu,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": self.kind,
            "source": self.source,
            "plane": self.plane.to_dict(),
            "points_2d": np.round(self.points_2d, 4).tolist(),
            "threshold_hu": None if self.threshold_hu is None else round(float(self.threshold_hu), 2),
            "contrast_hu": None if self.contrast_hu is None else round(float(self.contrast_hu), 2),
            "measured_fraction": round(self.measured_fraction, 4),
            "warnings": list(self.warnings),
            "metrics": self.metrics().to_dict(),
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SectionContour:
        return cls(
            kind=d["kind"],
            plane=SectionPlane.from_dict(d["plane"]),
            points_2d=d["points_2d"],
            source=d.get("source", "manual"),
            threshold_hu=d.get("threshold_hu"),
            contrast_hu=d.get("contrast_hu"),
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class SectionMeasurement:
    """Lumen and outer-wall contours of one cross-section."""

    plane: SectionPlane
    lumen: Optional[SectionContour]
    outer: Optional[SectionContour]
    warnings: List[str] = field(default_factory=list)

    @property
    def arc_length_mm(self) -> Optional[float]:
        return self.plane.arc_length_mm

    def metrics(self) -> Dict[str, Optional[ContourMetrics]]:
        return {
            LUMEN: self.lumen.metrics() if self.lumen is not None else None,
            OUTER: self.outer.metrics() if self.outer is not None else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plane": self.plane.to_dict(),
            "lumen": self.lumen.to_dict() if self.lumen is not None else None,
            "outer": self.outer.to_dict() if self.outer is not None else None,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SectionMeasurement:
        return cls(
            plane=SectionPlane.from_dict(d["plane"]),
            lumen=SectionContour.from_dict(d["lumen"]) if d.get("lumen") else None,
            outer=SectionContour.from_dict(d["outer"]) if d.get("outer") else None,
            warnings=list(d.get("warnings", [])),
        )


# ---------------------------------------------------------------------------- rays
def _ray_directions(n_rays: int) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, int(n_rays), endpoint=False)
    return angles, np.stack([np.cos(angles), np.sin(angles)], axis=1)


def ray_profiles(
    volume: Volume,
    plane: SectionPlane,
    center: np.ndarray,
    directions: np.ndarray,
    max_radius_mm: float,
    step_mm: float,
    start_mm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intensity along rays: ``(t, values)`` with values (rays, samples); NaN outside the volume."""
    start = max(0.0, float(start_mm))
    t = start + np.arange(0.0, max_radius_mm - start + 0.5 * step_mm, step_mm)
    origin = plane.origin + center[0] * plane.u + center[1] * plane.v
    dirs3 = directions[:, :1] * plane.u + directions[:, 1:2] * plane.v  # (rays, 3)
    ras = origin + dirs3[:, np.newaxis, :] * t[np.newaxis, :, np.newaxis]
    values = volume.sample_linear(ras.reshape(-1, 3)).reshape(len(directions), len(t))
    return t, values


def _crossings(
    values: np.ndarray,
    t: np.ndarray,
    start_index: np.ndarray,
    lower: float,
    upper: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """First crossing below ``lower`` (or above ``upper``) after ``start_index`` on each ray.

    Returns ``(radius, index)``: the interpolated radius (NaN if none before the ray leaves
    the volume) and the index of the first sample beyond the boundary.
    """
    n_rays, n = values.shape
    idx = np.arange(n)[np.newaxis, :]
    after = idx > np.asarray(start_index)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        hit = values < lower
        if upper is not None:
            hit |= values > upper
    hit &= after
    outside = np.isnan(values) & after
    first = np.where(hit.any(axis=1), np.argmax(hit, axis=1), n)
    first_out = np.where(outside.any(axis=1), np.argmax(outside, axis=1), n)
    found = (first < n) & (first < first_out)
    i = np.clip(first, 1, n - 1)
    rows = np.arange(n_rays)
    p = values[rows, i - 1]
    q = values[rows, i]
    level = np.where(q < lower, lower, upper if upper is not None else lower)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(p) & np.isfinite(q) & (p != q), (p - level) / (p - q), 0.0)
    step = t[1] - t[0] if len(t) > 1 else 0.0
    radius = t[i - 1] + np.clip(frac, 0.0, 1.0) * step
    return np.where(found, radius, np.nan), np.where(found, first, -1)


def _fourier_design(angles: np.ndarray, order: int) -> np.ndarray:
    columns = [np.ones_like(angles)]
    for k in range(1, order + 1):
        columns += [np.cos(k * angles), np.sin(k * angles)]
    return np.stack(columns, axis=1)


def _largest_gap_deg(kept: np.ndarray) -> float:
    """Largest run of consecutive rays without a boundary (periodic), in degrees."""
    n = len(kept)
    if kept.all():
        return 0.0
    if not kept.any():
        return 360.0
    shift = int(np.argmax(kept))  # start the scan at a kept ray
    k = np.roll(kept, -shift)
    longest = run = 0
    for value in k:
        run = 0 if value else run + 1
        longest = max(longest, run)
    return longest * 360.0 / n


def _fit_order(kept: np.ndarray, order: int) -> int:
    """Fewer harmonics when a large arc is missing (a high-order fit would oscillate there)."""
    gap = _largest_gap_deg(kept)
    if gap <= 60.0:
        return order
    if gap <= 120.0:
        return min(order, 2)
    return min(order, 1)


def _dilate_wrap(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.copy()
    for shift in range(1, iterations + 1):
        out |= np.roll(mask, shift) | np.roll(mask, -shift)
    return out


def regularize_radii(
    angles: np.ndarray,
    radii: np.ndarray,
    params: ContourParameters,
    two_sided: bool,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Remove outlier rays, fill gaps by interpolation, smooth along the angle.

    Returns ``(radii, kept)``; radii is None when too few rays are usable.
    """
    finite = np.isfinite(radii)
    n = len(radii)
    if finite.sum() < max(8, params.min_valid_fraction * n):
        return None, np.zeros(n, dtype=bool)
    # First pass against the median radius with a lenient tolerance (drops wild excursions
    # only), then against a low-order Fourier fit with the configured tolerance.
    fit = np.full(n, float(np.median(radii[finite])))
    kept = finite.copy()
    for iteration in range(6):
        resid = np.where(finite, radii - fit, np.inf)
        if iteration == 0:
            tol = np.maximum(2.0 * params.outlier_mm, 2.0 * params.outlier_fraction * fit)
        else:
            tol = np.maximum(params.outlier_mm, params.outlier_fraction * fit)
        new_kept = finite & (resid <= tol)
        if two_sided:
            new_kept &= resid >= -tol
        if new_kept.sum() < max(8, params.min_valid_fraction * n):
            break
        kept = new_kept
        design = _fourier_design(angles, _fit_order(kept, params.fourier_order))
        coef, *_ = np.linalg.lstsq(design[kept], radii[kept], rcond=None)
        new_fit = design @ coef
        converged = iteration > 0 and np.allclose(new_fit, fit, atol=1e-3)
        fit = new_fit
        if converged:
            break
    resid = np.where(finite, radii - fit, np.inf)
    tol = np.maximum(params.outlier_mm, params.outlier_fraction * fit)
    kept = finite & (resid <= tol) & ((resid >= -tol) if two_sided else True)
    # Rays next to a rejected run are contaminated by it (partial volume at a branch ostium,
    # the fat wedge next to an adjacent organ): reject a few more on both sides.
    guard = int(round(params.guard_deg / (360.0 / n)))
    if guard > 0 and not kept.all():
        kept &= ~_dilate_wrap(~kept, guard)
    if kept.sum() < max(8, params.min_valid_fraction * n):
        return None, kept
    # Gaps are filled with the low-order fit plus the residual interpolated between their
    # neighbours (so a gap over the pointed end of an ellipse keeps its shape); then only
    # the residual is smoothed: noise is removed, the overall shape is not flattened.
    design = _fourier_design(angles, _fit_order(kept, params.fourier_order))
    coef, *_ = np.linalg.lstsq(design[kept], radii[kept], rcond=None)
    fit = design @ coef
    a = angles[kept]
    resid = radii[kept] - fit[kept]
    resid_filled = np.interp(
        angles, np.concatenate([a - 2 * math.pi, a, a + 2 * math.pi]), np.concatenate([resid, resid, resid])
    )
    sigma = params.smoothing_deg / (360.0 / n)
    if sigma > 0:
        resid_filled = ndimage.gaussian_filter1d(resid_filled, sigma, mode="wrap")
    filled = fit + resid_filled
    return filled, kept


def _polygon(center: np.ndarray, directions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    return np.asarray(center)[np.newaxis, :] + directions * radii[:, np.newaxis]


def _centroid(points: np.ndarray) -> np.ndarray:
    from .metrics import polygon_centroid

    return polygon_centroid(points)


# ---------------------------------------------------------------------------- lumen
def _lumen_pass(volume: Volume, plane: SectionPlane, center: np.ndarray, params: ContourParameters):
    angles, dirs = _ray_directions(params.n_rays)
    t, values = ray_profiles(volume, plane, center, dirs, params.max_lumen_radius_mm, params.step_mm)
    start = np.zeros(len(dirs), dtype=int)
    core = values[:, t <= 1.5]
    core = core[np.isfinite(core) & (core < params.max_considered_hu)]
    if core.size == 0:
        return None
    level = float(np.median(core))

    def threshold(level_hu: float) -> float:
        return max(params.min_lower_hu, params.soft_tissue_hu + params.lower_fraction * (level_hu - params.soft_tissue_hu))

    if level < threshold(level) or level <= params.min_lower_hu:
        return None  # the centre is not in contrast-filled lumen
    radius, _ = _crossings(values, t, start, threshold(level), level + params.calcium_margin_hu)
    finite = np.isfinite(radius)
    if finite.sum() >= 8:
        # Contrast level: plateau inside the lumen (inner 60 % of every ray).
        inner = (t[np.newaxis, :] <= 0.6 * np.where(finite, radius, 0.0)[:, np.newaxis]) & np.isfinite(values)
        inner &= values < params.max_considered_hu
        if inner.sum() > 20:
            level = float(np.median(values[inner]))
        radius, _ = _crossings(values, t, start, threshold(level), level + params.calcium_margin_hu)
    return angles, dirs, radius, threshold(level), level


def lumen_contour(
    volume: Volume,
    plane: SectionPlane,
    params: Optional[ContourParameters] = None,
    center: Optional[np.ndarray] = None,
) -> Optional[SectionContour]:
    """Automatic lumen contour; ``center`` (plane coordinates) defaults to the plane origin.

    Returns None if the centre is not inside contrast-filled lumen or too few rays find
    a boundary.
    """
    params = params or ContourParameters()
    c = np.zeros(2) if center is None else np.asarray(center, dtype=float)
    warnings: List[str] = []
    result = None
    for attempt in range(3):
        pass_result = _lumen_pass(volume, plane, c, params)
        if pass_result is None:
            break
        angles, dirs, radius, thr, level = pass_result
        filled, kept = regularize_radii(angles, radius, params, two_sided=False)
        if filled is None:
            break
        points = _polygon(c, dirs, filled)
        result = (c.copy(), angles, filled, kept, thr, level, points)
        centroid = _centroid(points)
        if attempt == 2 or np.linalg.norm(centroid - c) <= params.recenter_mm:
            break
        c = centroid
    if result is None:
        return None
    c, angles, filled, kept, thr, level, points = result
    if kept.mean() < 0.8:
        warnings.append("lumen_gaps")
    return SectionContour(
        kind=LUMEN,
        plane=plane,
        points_2d=points,
        source="auto",
        threshold_hu=thr,
        contrast_hu=level,
        center=c,
        angles=angles,
        radii=filled,
        measured=kept,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------- outer wall
def _bone_runs(values: np.ndarray, start: np.ndarray, end: np.ndarray, bone_hu: float, run_samples: int) -> np.ndarray:
    """True for rays with a run of >= run_samples samples above bone_hu between start and end."""
    n_rays, n = values.shape
    idx = np.arange(n)[np.newaxis, :]
    with np.errstate(invalid="ignore"):
        bright = (values > bone_hu) & (idx > start[:, np.newaxis]) & (idx < end[:, np.newaxis])
    if run_samples <= 1:
        return bright.any(axis=1)
    kernel = np.ones(run_samples)
    counts = ndimage.convolve1d(bright.astype(float), kernel, axis=1, mode="constant")
    # convolve1d centres the kernel; a full window anywhere means a run of run_samples.
    return (counts >= run_samples - 0.5).any(axis=1)


def outer_contour(
    volume: Volume,
    lumen: SectionContour,
    params: Optional[ContourParameters] = None,
) -> SectionContour:
    """Semi-automatic outer wall (wall + thrombus) from the rays of an automatic lumen contour."""
    params = params or ContourParameters()
    if lumen.center is None or lumen.radii is None or lumen.angles is None:
        raise ValueError("the outer wall needs an automatic lumen contour (rays)")
    plane = lumen.plane
    c = lumen.center
    angles = lumen.angles
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    r_lumen = lumen.radii
    t0 = max(0.0, float(np.min(r_lumen)) - 1.0)
    t, values = ray_profiles(volume, plane, c, dirs, params.max_outer_radius_mm, params.step_mm, start_mm=t0)
    step = params.step_mm
    start = np.clip(np.floor((r_lumen + params.outer_search_start_mm - t0) / step).astype(int), 0, len(t) - 2)
    fat_level = 0.5 * (params.wall_hu + params.fat_hu)
    radius, first = _crossings(values, t, start, fat_level)
    rows = np.arange(len(dirs))

    # Bright plaque lining the outer wall: use the 50 % level between the plaque and fat.
    window = int(round(params.bright_window_mm / step))
    for r in rows[np.isfinite(radius)]:
        f = first[r]
        lo = max(start[r], f - window)
        segment = values[r, lo:f]
        if segment.size == 0:
            continue
        peak = float(np.nanmax(segment))
        if peak <= params.bright_edge_hu:
            continue
        local = 0.5 * (min(peak, 1000.0) + params.fat_hu)
        above = np.flatnonzero(segment >= local)
        if above.size == 0:
            continue
        i = lo + int(above[-1])  # last sample above the local level before the drop
        p, q = values[r, i], values[r, i + 1]
        frac = (p - local) / (p - q) if p != q else 0.0
        radius[r] = t[i] + min(max(frac, 0.0), 1.0) * step

    # A long bright run before the fat (bone) means the wall edge is not visible on this ray.
    end = np.where(first >= 0, first, len(t))
    bone = _bone_runs(values, start, end, params.bone_hu, int(round(params.bone_run_mm / step)))
    radius = np.where(bone, np.nan, radius)

    warnings: List[str] = []
    filled, kept = regularize_radii(angles, radius, params, two_sided=True)
    if filled is None:
        filled = r_lumen + params.fallback_wall_mm
        kept = np.zeros(len(angles), dtype=bool)
        warnings.append("outer_not_found")
    elif kept.mean() < 0.7:
        warnings.append("outer_gaps")
    filled = np.maximum(filled, r_lumen + params.min_wall_mm)
    return SectionContour(
        kind=OUTER,
        plane=plane,
        points_2d=_polygon(c, dirs, filled),
        source="auto",
        threshold_hu=fat_level,
        contrast_hu=lumen.contrast_hu,
        center=c.copy(),
        angles=angles,
        radii=filled,
        measured=kept,
        warnings=warnings,
    )


def measure_section(
    volume: Volume,
    plane: SectionPlane,
    params: Optional[ContourParameters] = None,
    with_outer: bool = True,
    center: Optional[np.ndarray] = None,
) -> SectionMeasurement:
    """Automatic lumen (and outer wall) contours of one cross-section."""
    params = params or ContourParameters()
    lumen = lumen_contour(volume, plane, params, center)
    warnings: List[str] = []
    outer = None
    if lumen is None:
        warnings.append("lumen_not_found")
    elif with_outer:
        outer = outer_contour(volume, lumen, params)
    return SectionMeasurement(plane=plane, lumen=lumen, outer=outer, warnings=warnings)
