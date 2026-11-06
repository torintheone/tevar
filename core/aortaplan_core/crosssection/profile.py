"""Diameters along the whole centerline and the search for the maximum (SPEC section 6.3).

The profile measures automatic contours at regular arc-length steps (fewer rays than a
single measurement, for speed). The maximum search scans a range coarsely, then measures
around the best position with full accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..centerline.navigation import CurvePath
from ..lumen.segmentation import Cancelled, ProgressCallback
from ..volume import Volume
from .contours import LUMEN, OUTER, ContourParameters, SectionMeasurement, SectionPlane, measure_section
from .metrics import METRIC_NAMES

PROFILE_RAYS = 120


def _check(progress: Optional[ProgressCallback], fraction: float, step: str) -> None:
    if progress is not None and progress(fraction, step) is False:
        raise Cancelled()


@dataclass
class DiameterProfile:
    """Section metrics at arc lengths ``arc_length_mm``; NaN where no contour was found."""

    arc_length_mm: np.ndarray
    lumen: Dict[str, np.ndarray]
    outer: Optional[Dict[str, np.ndarray]] = None

    def values(self, layer: str = LUMEN, metric: str = "max_diameter_mm") -> np.ndarray:
        table = self.lumen if layer == LUMEN else self.outer
        if table is None:
            return np.full(len(self.arc_length_mm), np.nan)
        return table[metric]

    def maximum(
        self,
        layer: str = OUTER,
        metric: str = "max_diameter_mm",
        s_min: Optional[float] = None,
        s_max: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """``(arc length, value)`` of the largest value in [s_min, s_max], or None."""
        s = self.arc_length_mm
        values = self.values(layer, metric)
        inside = np.ones(len(s), dtype=bool)
        if s_min is not None:
            inside &= s >= s_min - 1e-9
        if s_max is not None:
            inside &= s <= s_max + 1e-9
        candidates = np.where(inside & np.isfinite(values), values, -np.inf)
        if not np.isfinite(candidates).any():
            return None
        i = int(np.argmax(candidates))
        return float(s[i]), float(values[i])

    def to_dict(self) -> Dict[str, Any]:
        def table(t):
            if t is None:
                return None
            return {k: [None if not np.isfinite(x) else round(float(x), 3) for x in v] for k, v in t.items()}

        return {
            "arc_length_mm": np.round(self.arc_length_mm, 3).tolist(),
            "lumen": table(self.lumen),
            "outer": table(self.outer),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DiameterProfile:
        def table(t):
            if t is None:
                return None
            return {k: np.array([np.nan if x is None else x for x in v], dtype=float) for k, v in t.items()}

        return cls(np.asarray(d["arc_length_mm"], dtype=float), table(d["lumen"]), table(d.get("outer")))


def _empty(n: int) -> Dict[str, np.ndarray]:
    return {name: np.full(n, np.nan) for name in METRIC_NAMES}


def diameter_profile(
    volume: Volume,
    path: CurvePath,
    step_mm: float = 1.0,
    s_min: Optional[float] = None,
    s_max: Optional[float] = None,
    params: Optional[ContourParameters] = None,
    with_outer: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> DiameterProfile:
    """Lumen (and outer wall) metrics every ``step_mm`` along the path."""
    params = params or ContourParameters(n_rays=PROFILE_RAYS)
    lo = path.clamp(0.0 if s_min is None else s_min)
    hi = path.clamp(path.length if s_max is None else s_max)
    s_values = np.arange(lo, hi + 0.5 * step_mm, step_mm)
    s_values = s_values[s_values <= hi + 1e-9]
    lumen = _empty(len(s_values))
    outer = _empty(len(s_values)) if with_outer else None
    for i, s in enumerate(s_values):
        if i % 10 == 0:
            _check(progress, i / max(len(s_values), 1), "profile")
        m = measure_section(volume, SectionPlane.at(path, float(s)), params, with_outer=with_outer)
        if m.lumen is not None:
            for k, v in m.lumen.metrics().values().items():
                lumen[k][i] = v
        if outer is not None and m.outer is not None:
            for k, v in m.outer.metrics().values().items():
                outer[k][i] = v
    _check(progress, 1.0, "profile")
    return DiameterProfile(s_values, lumen, outer)


@dataclass
class MaximumResult:
    arc_length_mm: float
    value_mm: float
    layer: str
    metric: str
    measurement: SectionMeasurement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arc_length_mm": round(self.arc_length_mm, 3),
            "value_mm": round(self.value_mm, 3),
            "layer": self.layer,
            "metric": self.metric,
            "measurement": self.measurement.to_dict(),
        }


def _value(m: SectionMeasurement, layer: str, metric: str) -> float:
    contour = m.lumen if layer == LUMEN else m.outer
    if contour is None:
        return float("nan")
    return float(contour.metrics().values()[metric])


def find_maximum_diameter(
    volume: Volume,
    path: CurvePath,
    s_min: float,
    s_max: float,
    layer: str = OUTER,
    metric: str = "max_diameter_mm",
    coarse_step_mm: float = 2.0,
    fine_step_mm: float = 0.25,
    params: Optional[ContourParameters] = None,
    progress: Optional[ProgressCallback] = None,
) -> Optional[MaximumResult]:
    """Largest diameter between two arc lengths (e.g. two markers), e.g. of an aneurysm.

    Coarse scan with the profile settings, then full-accuracy contours every
    ``fine_step_mm`` within one coarse step around the best position.
    """
    lo, hi = sorted((path.clamp(s_min), path.clamp(s_max)))
    with_outer = layer == OUTER
    coarse = diameter_profile(
        volume, path, coarse_step_mm, lo, hi, with_outer=with_outer,
        progress=None if progress is None else (lambda f, step: progress(0.7 * f, "maximum")),
    )
    best = coarse.maximum(layer, metric, lo, hi)
    if best is None:
        return None
    params = params or ContourParameters()
    s_fine = np.arange(max(lo, best[0] - coarse_step_mm), min(hi, best[0] + coarse_step_mm) + 1e-9, fine_step_mm)
    result: Optional[MaximumResult] = None
    for i, s in enumerate(s_fine):
        _check(progress, 0.7 + 0.3 * i / max(len(s_fine), 1), "maximum")
        m = measure_section(volume, SectionPlane.at(path, float(s)), params, with_outer=with_outer)
        value = _value(m, layer, metric)
        if np.isfinite(value) and (result is None or value > result.value_mm):
            result = MaximumResult(float(s), value, layer, metric, m)
    _check(progress, 1.0, "maximum")
    return result
