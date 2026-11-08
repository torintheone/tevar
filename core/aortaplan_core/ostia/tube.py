"""The aortic lumen surface without its branches, from the contours of stage 4.

Every ``step_mm`` along a path the automatic lumen contour is found (rays from the lumen
centre); its regularisation rejects the rays that escape into a branch ostium and bridges
the ostium with a smooth fit. Stacked, these contours describe the aortic wall as if the
branches were cut off flush: the surface through which a branch centerline leaves the
aorta (the ostium, SPEC section 6.4).

``signed_distance`` tells how far a point lies outside (> 0) or inside (< 0) that surface,
interpolating between the two nearest sections.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..centerline.navigation import CurvePath
from ..crosssection.contours import ContourParameters, SectionPlane, lumen_contour
from ..geometry.curves import cumulative_arc_length, polyline_length, resample_polyline
from ..lumen.segmentation import Cancelled, ProgressCallback
from ..volume import Volume

TUBE_RAYS = 180


def tube_contour_parameters() -> ContourParameters:
    """Stricter outlier rejection than for measuring: rays escaping even 1 mm into an ostium
    are bridged, so the surface does not bulge into oblique branches (SMA, IMA)."""
    return ContourParameters(n_rays=TUBE_RAYS, outlier_mm=1.0, outlier_fraction=0.08)


def _check(progress: Optional[ProgressCallback], fraction: float, step: str) -> None:
    if progress is not None and progress(fraction, step) is False:
        raise Cancelled()


class AorticTube:
    def __init__(
        self,
        volume: Volume,
        path: CurvePath,
        step_mm: float = 1.0,
        s_min: Optional[float] = None,
        s_max: Optional[float] = None,
        params: Optional[ContourParameters] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.path = path
        self.step_mm = float(step_mm)
        self.params = params or tube_contour_parameters()
        lo = path.clamp(0.0 if s_min is None else s_min)
        hi = path.clamp(path.length if s_max is None else s_max)
        self.s_values = np.arange(lo, hi + 1e-9, self.step_mm)
        n, rays = len(self.s_values), self.params.n_rays
        self.angles = np.linspace(0.0, 2.0 * math.pi, rays, endpoint=False)
        self.origins = np.zeros((n, 3))
        self.u = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.centers = np.zeros((n, 2))
        self.radii = np.full((n, rays), np.nan)
        self.thresholds = np.full(n, np.nan)
        self.levels = np.full(n, np.nan)
        self.valid = np.zeros(n, dtype=bool)
        for i, s in enumerate(self.s_values):
            if i % 20 == 0:
                _check(progress, i / max(n, 1), "ostia.tube")
            plane = SectionPlane.at(path, float(s))
            self.origins[i], self.u[i], self.v[i] = plane.origin, plane.u, plane.v
            contour = lumen_contour(volume, plane, self.params)
            if contour is None:
                continue
            self.centers[i] = contour.center
            self.radii[i] = contour.radii
            self.thresholds[i] = contour.threshold_hu
            self.levels[i] = contour.contrast_hu
            self.valid[i] = True
        # Points along the path every 0.25 mm (their arc lengths are exact: they lie on it).
        self._dense = resample_polyline(path.points, 0.25)
        self._dense_s = cumulative_arc_length(self._dense) * (path.length / max(polyline_length(self._dense), 1e-9))
        self._tree = cKDTree(self._dense)

    # ------------------------------------------------------------------ queries
    @property
    def s_range(self) -> Tuple[float, float]:
        valid = self.s_values[self.valid]
        if len(valid) == 0:
            return (0.0, 0.0)
        return float(valid[0]), float(valid[-1])

    def arc_lengths(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate arc length of the nearest centerline point and distance to it."""
        distance, index = self._tree.query(np.atleast_2d(points))
        return self._dense_s[index], distance

    def _section_index(self, s: np.ndarray):
        k = (np.asarray(s, dtype=float) - self.s_values[0]) / self.step_mm
        i0 = np.clip(np.floor(k).astype(int), 0, len(self.s_values) - 1)
        i1 = np.clip(i0 + 1, 0, len(self.s_values) - 1)
        f = np.clip(k - i0, 0.0, 1.0)
        return i0, i1, f

    def _section_distance(self, i: np.ndarray, points: np.ndarray) -> np.ndarray:
        rel = points - self.origins[i]
        x = np.einsum("ij,ij->i", rel, self.u[i]) - self.centers[i, 0]
        y = np.einsum("ij,ij->i", rel, self.v[i]) - self.centers[i, 1]
        theta = np.mod(np.arctan2(y, x), 2.0 * math.pi)
        step = 2.0 * math.pi / len(self.angles)
        k = theta / step
        k0 = np.floor(k).astype(int) % len(self.angles)
        k1 = (k0 + 1) % len(self.angles)
        f = k - np.floor(k)
        r = (1.0 - f) * self.radii[i, k0] + f * self.radii[i, k1]
        out = np.hypot(x, y) - r
        out[~self.valid[i]] = np.nan
        return out

    def signed_distance(self, points: np.ndarray, s: Optional[np.ndarray] = None) -> np.ndarray:
        """Distance (mm) outside (+) or inside (-) the branch-free lumen surface; NaN off the tube."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if s is None:
            s, _ = self.arc_lengths(p)
        i0, i1, f = self._section_index(s)
        d0 = self._section_distance(i0, p)
        d1 = self._section_distance(i1, p)
        out = np.where(np.isfinite(d0) & np.isfinite(d1), (1.0 - f) * d0 + f * d1, np.where(np.isfinite(d0), d0, d1))
        lo, hi = self.s_values[0], self.s_values[-1]
        out[(s < lo - 0.5 * self.step_mm) | (s > hi + 0.5 * self.step_mm)] = np.nan
        return out

    def section_values(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lumen threshold and contrast level of the nearest valid section."""
        i0, _, _ = self._section_index(s)
        valid = np.flatnonzero(self.valid)
        if len(valid) == 0:
            raise ValueError("no valid section in the tube")
        nearest = valid[np.argmin(np.abs(valid[np.newaxis, :] - np.atleast_1d(i0)[:, np.newaxis]), axis=1)]
        return self.thresholds[nearest], self.levels[nearest]

    def local_radius(self, s: float) -> float:
        i0, _, _ = self._section_index(np.array([s]))
        r = self.radii[i0[0]]
        return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")

    def crossing(self, inside: np.ndarray, outside: np.ndarray, iterations: int = 40) -> np.ndarray:
        """Point on the surface between a point inside and a point outside (bisection)."""
        a = np.asarray(inside, dtype=float)
        b = np.asarray(outside, dtype=float)
        for _ in range(iterations):
            m = 0.5 * (a + b)
            d = self.signed_distance(m)[0]
            if np.isfinite(d) and d < 0:
                a = m
            else:
                b = m
        return 0.5 * (a + b)
