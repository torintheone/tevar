"""Named markers on the centerline and distances between them (SPEC sections 4 and 6.3).

A marker is a named point on a centerline path. Several paths of one network share their
proximal part (all start at the start point), so a marker on the aorta lies on every
path, while a marker on an iliac artery lies on one path only.

Distance between two markers:

- **along the centerline** (primary): if one path contains both markers, the difference of
  their arc lengths on it; otherwise (different branches) the way through the point where
  their paths divide: ``(s_a - s_div) + (s_b - s_div)``;
- **straight** (secondary): the Euclidean distance between the two points.

Positions are re-derived from the 3-D points, so markers survive a rebuilt centerline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import distances_to_polyline
from .navigation import CurvePath

#: Preset marker names (i18n keys ``marker.<key>``), in the order offered to the user.
MARKER_PRESETS: Tuple[str, ...] = (
    "proximal_landing",
    "distal_landing",
    "stj",
    "bca",
    "lcca",
    "lsa",
    "celiac",
    "sma",
    "right_renal",
    "left_renal",
    "lowest_renal",
    "neck_end",
    "aortic_bifurcation",
    "right_iliac_bifurcation",
    "left_iliac_bifurcation",
    "right_femoral",
    "left_femoral",
    "max_diameter",
)

#: A point counts as lying on a path if it is closer than this (mm).
ON_PATH_TOLERANCE_MM = 2.0


@dataclass
class Marker:
    """A named point on the centerline. ``key`` is the preset key, None for a custom name.

    ``arc_length_mm`` is the position on ``path`` (None when the marker is not on the path
    being looked at).
    """

    name: str
    point: np.ndarray
    path: str = ""
    arc_length_mm: Optional[float] = 0.0
    key: Optional[str] = None

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "point": np.round(self.point, 4).tolist(),
            "path": self.path,
            "arc_length_mm": None if self.arc_length_mm is None else round(float(self.arc_length_mm), 3),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Marker:
        return cls(
            name=d["name"],
            point=d["point"],
            path=d.get("path", ""),
            arc_length_mm=None if d.get("arc_length_mm") is None else float(d["arc_length_mm"]),
            key=d.get("key"),
        )


def marker_on_path(name: str, path_name: str, path: CurvePath, s: float, key: Optional[str] = None) -> Marker:
    s = path.clamp(s)
    return Marker(name=name, point=path.point_at(s), path=path_name, arc_length_mm=s, key=key)


def locate(point: np.ndarray, paths: Dict[str, CurvePath], tolerance_mm: float = ON_PATH_TOLERANCE_MM):
    """Paths that pass through a point: list of ``(name, arc length, distance)``."""
    found = []
    for name, path in paths.items():
        s, d = path.nearest_arc_length(point)
        if d <= tolerance_mm:
            found.append((name, s, d))
    return found


def divergence(path_a: CurvePath, path_b: CurvePath, tolerance_mm: float = ON_PATH_TOLERANCE_MM) -> float:
    """Arc length on ``path_a`` where it splits from ``path_b`` (its length if it never does).

    The paths are taken as separated where their distance first exceeds the tolerance;
    the split point is then traced back to the last sample that still lies on ``path_b``
    as closely as the shared part does (within 0.05 mm of its median offset).
    """
    d = distances_to_polyline(path_a.points, path_b.points, chunk=256)
    beyond = np.flatnonzero(d > tolerance_mm)
    if len(beyond) == 0:
        return path_a.length
    first = int(beyond[0])
    if first == 0:
        return 0.0
    base = float(np.median(d[:first]))
    close = np.flatnonzero(d[:first] <= base + 0.05)
    i = int(close[-1]) if len(close) else first - 1
    return float(path_a.arc_length[i])


@dataclass
class Distance:
    """Distances between two points; ``along_mm`` is None if they are on unconnected paths."""

    along_mm: Optional[float]
    straight_mm: float
    path: Optional[str] = None  # path used for the along-centerline distance (None: two branches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "along_centerline_mm": None if self.along_mm is None else round(self.along_mm, 3),
            "straight_mm": round(self.straight_mm, 3),
            "path": self.path,
        }


def distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
    paths: Dict[str, CurvePath],
    preferred: Sequence[str] = (),
    tolerance_mm: float = ON_PATH_TOLERANCE_MM,
    cache: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
) -> Distance:
    """Distance along the centerline network and straight distance between two points.

    ``cache`` (optional dict) keeps the split points of path pairs between calls.
    """
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    straight = float(np.linalg.norm(b - a))
    on_a = {name: s for name, s, _ in locate(a, paths, tolerance_mm)}
    on_b = {name: s for name, s, _ in locate(b, paths, tolerance_mm)}
    common = [name for name in list(preferred) + sorted(on_a) if name in on_a and name in on_b]
    if common:
        name = common[0]
        return Distance(abs(on_b[name] - on_a[name]), straight, name)
    best: Optional[float] = None
    for name_a, s_a in on_a.items():
        for name_b, s_b in on_b.items():
            key = (name_a, name_b)
            if cache is not None and key in cache:
                s_div, s_div_b = cache[key]
            else:
                s_div = divergence(paths[name_a], paths[name_b], tolerance_mm)
                # The divergence point is the same physical point on both paths (shared part).
                s_div_b = paths[name_b].nearest_arc_length(paths[name_a].point_at(s_div))[0]
                if cache is not None:
                    cache[key] = (s_div, s_div_b)
            if s_a < s_div or s_b < s_div_b:
                continue
            value = (s_a - s_div) + (s_b - s_div_b)
            if best is None or value < best:
                best = value
    return Distance(best, straight, None)


def marker_distance(a: Marker, b: Marker, paths: Dict[str, CurvePath], cache=None) -> Distance:
    preferred = [name for name in (a.path, b.path) if name]
    return distance(a.point, b.point, paths, preferred, cache=cache)


def distance_table(markers: List[Marker], paths: Dict[str, CurvePath]) -> List[List[Optional[Distance]]]:
    """Matrix of distances between all markers (None on the diagonal)."""
    n = len(markers)
    table: List[List[Optional[Distance]]] = [[None] * n for _ in range(n)]
    cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = marker_distance(markers[i], markers[j], paths, cache)
            table[i][j] = table[j][i] = d
    return table


def sort_along(markers: List[Marker], path: CurvePath) -> List[Marker]:
    """Markers ordered proximal -> distal by their projection on a path."""
    return sorted(markers, key=lambda m: path.nearest_arc_length(m.point)[0])


def tortuosity_index(path: CurvePath, s_start: float, s_end: float) -> float:
    """Centerline length / straight distance between two arc lengths (>= 1)."""
    s0, s1 = sorted((path.clamp(s_start), path.clamp(s_end)))
    chord = float(np.linalg.norm(path.point_at(s1) - path.point_at(s0)))
    if chord < 1e-6:
        return math.inf
    return (s1 - s0) / chord
