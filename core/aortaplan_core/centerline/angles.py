"""Angles between vessel axes along the centerline (SPEC section 6.3, EVAR).

An axis of a centerline segment is the straight line through its end points (the chord),
oriented proximal -> distal, as in EVAR reporting standards:

- alpha: suprarenal aortic axis vs. infrarenal neck axis. The suprarenal axis runs from
  ``suprarenal_length_mm`` (default 20 mm) above the lowest renal artery to it; the neck
  axis from the lowest renal artery to the end of the neck;
- beta: neck axis vs. aneurysm axis (end of the neck -> aortic bifurcation).

An angle of 0 deg means the two axes are collinear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .navigation import CurvePath

DEFAULT_SUPRARENAL_LENGTH_MM = 20.0


def axis(path: CurvePath, s_start: float, s_end: float) -> np.ndarray:
    """Unit chord direction from ``s_start`` to ``s_end``."""
    d = path.point_at(s_end) - path.point_at(s_start)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        raise ValueError("an axis needs two distinct points")
    return d / norm


def angle_between(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle (deg, 0..180) between two directions."""
    c = float(np.clip(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)), -1.0, 1.0))
    return math.degrees(math.acos(c))


@dataclass
class NeckAngles:
    alpha_deg: float
    beta_deg: float
    suprarenal_start_mm: float
    lowest_renal_mm: float
    neck_end_mm: float
    bifurcation_mm: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(float(v), 3) for k, v in self.__dict__.items()}


def neck_angles(
    path: CurvePath,
    lowest_renal_mm: float,
    neck_end_mm: float,
    bifurcation_mm: float,
    suprarenal_length_mm: float = DEFAULT_SUPRARENAL_LENGTH_MM,
) -> NeckAngles:
    """EVAR neck angles alpha and beta from positions (arc lengths) on one path."""
    if not lowest_renal_mm < neck_end_mm < bifurcation_mm:
        raise ValueError("expected lowest renal < end of neck < bifurcation along the path")
    supra_start = max(0.0, lowest_renal_mm - suprarenal_length_mm)
    if lowest_renal_mm - supra_start < 1.0:
        raise ValueError("the path starts too close to the lowest renal artery")
    supra = axis(path, supra_start, lowest_renal_mm)
    neck = axis(path, lowest_renal_mm, neck_end_mm)
    sac = axis(path, neck_end_mm, bifurcation_mm)
    return NeckAngles(
        alpha_deg=angle_between(supra, neck),
        beta_deg=angle_between(neck, sac),
        suprarenal_start_mm=supra_start,
        lowest_renal_mm=lowest_renal_mm,
        neck_end_mm=neck_end_mm,
        bifurcation_mm=bifurcation_mm,
    )
