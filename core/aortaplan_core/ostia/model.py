"""Ostium data and its description relative to the aortic centerline (SPEC section 6.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..centerline.navigation import CurvePath
from ..crosssection.metrics import ContourMetrics
from .clock import (
    DEFAULT_GRAFT_DIAMETER_MM,
    MODE_ANTERIOR,
    TakeoffAngles,
    arc_length_on_graft,
    clock_angle,
    clock_frame,
    clock_text,
    takeoff_angles,
)

SOURCE_AUTO = "auto"  # found in the image
SOURCE_PATH = "path"  # a branch of the centerline network (stage 2 "add branch")
SOURCE_MANUAL = "manual"  # clicked by the user
DEFAULT_DIAMETER_DISTANCE_MM = 5.0
AXIS_SMOOTHING_MM = 10.0  # local smoothing of the aortic axis for positions and clock frames


@dataclass
class Ostium:
    """A branch ostium. ``center`` is where the branch centerline leaves the aortic lumen
    surface; ``branch_points`` is the branch centerline from the ostium outwards."""

    center: np.ndarray
    direction: Optional[np.ndarray] = None  # unit, leaving the aorta
    branch_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    diameter: Optional[ContourMetrics] = None  # branch section at ``diameter_distance_mm``
    diameter_distance_mm: float = DEFAULT_DIAMETER_DISTANCE_MM
    name: str = ""
    key: Optional[str] = None
    confirmed: bool = False
    source: str = SOURCE_AUTO
    stenosis: bool = False
    note: str = ""
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        if self.direction is not None:
            self.direction = np.asarray(self.direction, dtype=float)
        self.branch_points = np.asarray(self.branch_points, dtype=float).reshape(-1, 3)

    @property
    def branch_diameter_mm(self) -> Optional[float]:
        return None if self.diameter is None else float(self.diameter.mean_diameter_mm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "confirmed": self.confirmed,
            "source": self.source,
            "center": np.round(self.center, 4).tolist(),
            "direction": None if self.direction is None else np.round(self.direction, 6).tolist(),
            "branch_points": np.round(self.branch_points, 3).tolist(),
            "diameter": None if self.diameter is None else self.diameter.to_dict(),
            "diameter_distance_mm": self.diameter_distance_mm,
            "stenosis": self.stenosis,
            "note": self.note,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Ostium:
        return cls(
            center=d["center"],
            direction=d.get("direction"),
            branch_points=d.get("branch_points") or np.zeros((0, 3)),
            diameter=ContourMetrics.from_dict(d["diameter"]) if d.get("diameter") else None,
            diameter_distance_mm=float(d.get("diameter_distance_mm", DEFAULT_DIAMETER_DISTANCE_MM)),
            name=d.get("name", ""),
            key=d.get("key"),
            confirmed=bool(d.get("confirmed", False)),
            source=d.get("source", SOURCE_AUTO),
            stenosis=bool(d.get("stenosis", False)),
            note=d.get("note", ""),
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class OstiumPosition:
    """An ostium described on a path: position, clock and take-off angles."""

    arc_length_mm: float
    distance_to_centerline_mm: float
    clock_deg: float
    clock: str
    arc_on_graft_mm: float
    graft_diameter_mm: float
    mode: str
    mode_fallback: bool
    takeoff: Optional[TakeoffAngles]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arc_length_mm": round(self.arc_length_mm, 3),
            "clock": self.clock,
            "clock_deg": round(self.clock_deg, 2),
            "arc_on_graft_mm": round(self.arc_on_graft_mm, 2),
            "graft_diameter_mm": self.graft_diameter_mm,
            "clock_mode": self.mode,
            "clock_mode_fallback": self.mode_fallback,
            "takeoff": None if self.takeoff is None else self.takeoff.to_dict(),
        }


def describe(
    ostium: Ostium,
    path: CurvePath,
    mode: str = MODE_ANTERIOR,
    graft_diameter_mm: float = DEFAULT_GRAFT_DIAMETER_MM,
) -> OstiumPosition:
    # Position and clock on the locally smoothed axis: a centerline bending slightly towards
    # the branch (as VMTK and minimal-path lines do at ostia) does not move the ostium.
    s_smooth, s = path.foot_arc_length(ostium.center, AXIS_SMOOTHING_MM)
    smooth = path.smoothed(AXIS_SMOOTHING_MM)
    distance = float(np.linalg.norm(ostium.center - smooth.point_at(s_smooth)))
    frame = clock_frame(smooth, s_smooth, mode)
    angle = clock_angle(frame, ostium.center)
    takeoff = takeoff_angles(frame, ostium.center, ostium.direction) if ostium.direction is not None else None
    return OstiumPosition(
        arc_length_mm=float(s),
        distance_to_centerline_mm=float(distance),
        clock_deg=angle,
        clock=clock_text(angle),
        arc_on_graft_mm=arc_length_on_graft(angle, graft_diameter_mm),
        graft_diameter_mm=float(graft_diameter_mm),
        mode=mode,
        mode_fallback=frame.fallback,
        takeoff=takeoff,
    )


def distal_edge_arc_length(ostium: Ostium, position: OstiumPosition) -> float:
    """Arc length of the distal edge of the ostium (centre + half its extent along the axis)."""
    if ostium.diameter is None:
        return position.arc_length_mm
    half = 0.5 * ostium.diameter.mean_diameter_mm
    if position.takeoff is not None:
        half /= max(math.sin(math.radians(position.takeoff.craniocaudal_deg)), math.sin(math.radians(20.0)))
    return position.arc_length_mm + half


def sort_along(ostia: List[Ostium], path: CurvePath) -> List[Ostium]:
    return sorted(ostia, key=lambda o: path.nearest_arc_length(o.center)[0])
