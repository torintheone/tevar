"""Clock position of an ostium and take-off angles of a branch (SPEC section 4).

Convention (radiological axial view seen from the feet, in the plane perpendicular to the
aortic centerline at the ostium):

- 12:00 = reference direction "up": the anterior wall (``MODE_ANTERIOR``; in the arch,
  where the anterior direction degenerates, the direction carried along the centerline from
  the descending aorta, exactly as the perpendicular view is oriented, see
  ``centerline.navigation``) or the greater curvature (``MODE_GREATER_CURVATURE``: away from
  the centre of curvature; falls back to the anterior reference where the vessel is straight);
- 3:00 = the patient's left, 9:00 = the patient's right (with the greater-curvature
  reference 3:00 is the side of the section towards the patient's left);
- angle in degrees 0-360, clockwise from 12:00 (3:00 = 90 deg);
- arc length on a graft of diameter D: ``L = angle / 360 * pi * D``.

Take-off angles of a branch with unit direction ``d`` leaving the ostium (``T`` = distal
direction of the aortic centerline, ``r`` = radial direction of the ostium):

- craniocaudal: the angle between the branch and the distal aortic axis in the plane
  spanned by T and r (90 deg = perpendicular, below 90 = pointing distally, e.g. caudally
  in the abdominal aorta);
- axial: the angle of the branch projected on the cross-section plane relative to the
  radial direction, positive clockwise (in the clock sense).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..centerline.navigation import CurvePath

MODE_ANTERIOR = "anterior"
MODE_GREATER_CURVATURE = "greater_curvature"
CLOCK_MODES = (MODE_ANTERIOR, MODE_GREATER_CURVATURE)
CLOCK_STEP_MINUTES = 5
DEFAULT_GRAFT_DIAMETER_MM = 28.0
MIN_CURVATURE_PER_MM = 1.0 / 300.0  # greater curvature undefined on straighter vessels
PATIENT_LEFT = np.array([-1.0, 0.0, 0.0])  # RAS


@dataclass
class ClockFrame:
    """Section at an arc length: centre, 12:00 direction (``up``) and 3:00 direction (``right``)."""

    center: np.ndarray
    up: np.ndarray
    right: np.ndarray
    tangent: np.ndarray  # distal direction of the centerline
    mode: str
    fallback: bool = False  # greater curvature undefined here: the anterior reference was used


def _curvature_vector(path: CurvePath, s: float, smoothing_mm: float = 10.0) -> np.ndarray:
    from scipy import ndimage

    step = float(np.median(np.diff(path.arc_length)))
    t = ndimage.gaussian_filter1d(path.tangents, max(smoothing_mm / step, 1e-6), axis=0, mode="nearest")
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    k = np.gradient(t, path.arc_length, axis=0)
    return np.array([np.interp(s, path.arc_length, k[:, c]) for c in range(3)])


def clock_frame(path: CurvePath, s: float, mode: str = MODE_ANTERIOR) -> ClockFrame:
    """Reference directions of the clock at arc length ``s``."""
    s = path.clamp(s)
    view = path.cross_section_frame(s)  # x = patient left, y = anterior (or carried), normal to the viewer
    tangent = path.tangent_at(s)
    view_dir = np.cross(view.y_axis, view.x_axis)  # towards the head; screen x = view_dir x up
    if mode == MODE_GREATER_CURVATURE:
        k = _curvature_vector(path, s)
        k -= np.dot(k, tangent) * tangent
        norm = float(np.linalg.norm(k))
        if norm >= MIN_CURVATURE_PER_MM:
            up = -k / norm  # away from the centre of curvature
            right = np.cross(view_dir, up)
            if np.dot(right, PATIENT_LEFT) < -1e-3:
                # 3:00 towards the patient's left (e.g. arch top: the section seen from the front,
                # as a radiological coronal image).
                right = -right
            return ClockFrame(view.origin, up, right, tangent, mode)
        return ClockFrame(view.origin, view.y_axis, view.x_axis, tangent, mode, fallback=True)
    if mode != MODE_ANTERIOR:
        raise ValueError(f"unknown clock mode {mode!r}")
    return ClockFrame(view.origin, view.y_axis, view.x_axis, tangent, mode)


def clock_angle(frame: ClockFrame, point: np.ndarray) -> float:
    """Clockwise angle (deg, 0-360) of a point around the section centre, 0 = 12:00."""
    rel = np.asarray(point, dtype=float) - frame.center
    rel = rel - np.dot(rel, frame.tangent) * frame.tangent
    return math.degrees(math.atan2(float(np.dot(rel, frame.right)), float(np.dot(rel, frame.up)))) % 360.0


def direction_angle(frame: ClockFrame, direction: np.ndarray) -> float:
    """Clockwise angle (deg) of an in-plane direction (its projection), 0 = 12:00."""
    d = np.asarray(direction, dtype=float)
    d = d - np.dot(d, frame.tangent) * frame.tangent
    return math.degrees(math.atan2(float(np.dot(d, frame.right)), float(np.dot(d, frame.up)))) % 360.0


def clock_text(angle_deg: float, step_minutes: int = CLOCK_STEP_MINUTES) -> str:
    """'h:mm' for a clockwise angle from 12:00, rounded to ``step_minutes``."""
    total = (angle_deg % 360.0) / 30.0 * 60.0  # minutes after 12:00
    total = int(round(total / step_minutes)) * step_minutes % 720
    hours, minutes = divmod(total, 60)
    return f"{hours if hours else 12}:{minutes:02d}"


def arc_length_on_graft(angle_deg: float, graft_diameter_mm: float) -> float:
    """Arc (mm) from 12:00 clockwise to the angle on a graft of the given diameter."""
    return (angle_deg % 360.0) / 360.0 * math.pi * float(graft_diameter_mm)


@dataclass
class TakeoffAngles:
    craniocaudal_deg: float
    axial_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {"craniocaudal_deg": round(self.craniocaudal_deg, 2), "axial_deg": round(self.axial_deg, 2)}


def takeoff_angles(frame: ClockFrame, ostium: np.ndarray, direction: np.ndarray) -> Optional[TakeoffAngles]:
    """Craniocaudal and axial take-off angles of a branch direction at an ostium."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    radial = np.asarray(ostium, dtype=float) - frame.center
    radial -= np.dot(radial, frame.tangent) * frame.tangent
    norm = float(np.linalg.norm(radial))
    if norm < 1e-6:
        return None
    radial /= norm
    along = float(np.dot(d, frame.tangent))
    outward = float(np.dot(d, radial))
    craniocaudal = math.degrees(math.acos(max(-1.0, min(1.0, along / max(math.hypot(along, outward), 1e-12)))))
    axial = (direction_angle(frame, d) - clock_angle(frame, ostium) + 180.0) % 360.0 - 180.0
    return TakeoffAngles(craniocaudal, axial)
