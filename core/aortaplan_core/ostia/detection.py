"""Finding branch ostia along an aortic path (SPEC section 6.4, stage 5).

1. ``AorticTube``: the aortic lumen surface without branches (stage 4 contours).
2. Candidates: voxels in the contrast range of the nearest aortic section (its 50 % lumen
   threshold, calcium excluded) that lie in a shell up to ``shell_mm`` outside that surface;
   26-connected components that touch the surface are branch stumps. Components too thick
   for a branch, or spread far along the aorta (bone lying against it), are rejected.
3. Each stump is traced: points are recentred on the branch lumen contour (perpendicular
   sections) outwards; the line through the first centred points, extended back to the
   aortic surface, gives the **ostium centre** (where the branch centerline leaves the aorta).
4. The branch diameter is measured on the section ``diameter_distance_mm`` from the ostium.

Branches that exist as centerline paths (stage 2 "add branch") and ostia clicked by the
user go through the same tracing and measurement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from ..centerline.navigation import CurvePath
from ..crosssection.contours import ContourParameters, SectionPlane, lumen_contour
from ..geometry.curves import cumulative_arc_length
from ..lumen.segmentation import Cancelled, ProgressCallback
from ..volume import Volume
from .model import SOURCE_AUTO, SOURCE_MANUAL, SOURCE_PATH, Ostium
from .tube import AorticTube

_FULL = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity


@dataclass
class OstiumParameters:
    min_branch_diameter_mm: float = 2.5
    shell_mm: float = 15.0  # how far outside the aortic surface stumps are looked for
    touch_mm: float = 1.5  # a stump must come this close to the surface
    surface_margin_mm: float = 0.3
    upper_margin_hu: float = 300.0  # brighter than contrast + margin = calcium/bone
    max_branch_to_aorta_ratio: float = 0.8  # stump radius / aortic radius
    max_contact_length_mm: float = 40.0  # along the aorta (longer: bone against the wall)
    max_stump_length_mm: float = 80.0
    end_margin_mm: float = 4.0  # no stumps at the open ends of the path
    diameter_distance_mm: float = 5.0
    direction_length_mm: float = 10.0
    trace_length_mm: float = 25.0
    merge_distance_mm: float = 5.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> OstiumParameters:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


BRANCH_CONTOUR = ContourParameters(n_rays=180, max_lumen_radius_mm=15.0, min_valid_fraction=0.3)


def _check(progress: Optional[ProgressCallback], fraction: float, step: str) -> None:
    if progress is not None and progress(fraction, step) is False:
        raise Cancelled()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def branch_plane(point: np.ndarray, direction: np.ndarray) -> SectionPlane:
    """A section plane perpendicular to a branch direction."""
    d = _unit(np.asarray(direction, dtype=float))
    helper = np.eye(3)[int(np.argmin(np.abs(d)))]
    u = _unit(np.cross(d, helper))
    return SectionPlane(np.asarray(point, dtype=float), u, np.cross(d, u))


def branch_section(volume: Volume, point: np.ndarray, direction: np.ndarray):
    """Lumen contour of a branch perpendicular to ``direction`` near ``point`` (or None)."""
    return lumen_contour(volume, branch_plane(point, direction), BRANCH_CONTOUR)


def trace_branch(
    volume: Volume, start: np.ndarray, direction: np.ndarray, length_mm: float, step_mm: float = 1.0
) -> np.ndarray:
    """Centred points along a branch from ``start`` outwards (recentred on its contours)."""
    points: List[np.ndarray] = []
    p = np.asarray(start, dtype=float)
    d = _unit(np.asarray(direction, dtype=float))
    for _ in range(int(length_mm / step_mm) + 1):
        plane = branch_plane(p, d)
        contour = lumen_contour(volume, plane, BRANCH_CONTOUR)
        if contour is None:
            break
        q = plane.to_ras(contour.metrics().centroid)[0]
        if np.linalg.norm(q - p) > 3.0:
            break  # jumped to another vessel
        points.append(q)
        if len(points) >= 4:
            new = _unit(points[-1] - points[-4])
            if np.dot(new, d) > 0.5:
                d = _unit(0.5 * d + 0.5 * new)
        p = q + step_mm * d
    return np.array(points).reshape(-1, 3)


def _fit_line(points: np.ndarray, outward_hint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    d = vt[0]
    if np.dot(d, outward_hint) < 0:
        d = -d
    return centre, d


def _ostium_from_trace(tube: AorticTube, trace: np.ndarray, hint: np.ndarray, params: OstiumParameters):
    """Ostium centre and branch direction from centred branch points (extend back to the surface)."""
    delta = tube.signed_distance(trace)
    near = trace[np.isfinite(delta) & (delta <= params.direction_length_mm + 3.0)]
    if len(near) < 3:
        near = trace[: min(len(trace), 8)]
    if len(near) < 2:
        return None, None
    anchor, d = _fit_line(near, hint)
    anchor = near[0] + np.dot(anchor - near[0], d) * d  # a point of the line near the stump start
    inside = None
    for t in np.arange(0.5, 30.0, 0.5):
        q = anchor - t * d
        value = tube.signed_distance(q)[0]
        if np.isfinite(value) and value < 0:
            inside = q
            break
    if inside is None:
        return None, d
    outside = anchor if tube.signed_distance(anchor)[0] > 0 else anchor + 3.0 * d
    return tube.crossing(inside, outside), d


def measure_branch(volume: Volume, ostium: Ostium, distance_mm: float) -> None:
    """Branch section ``distance_mm`` from the ostium along its centerline."""
    ostium.diameter_distance_mm = float(distance_mm)
    points = np.vstack([ostium.center[np.newaxis, :], ostium.branch_points]) if len(ostium.branch_points) else None
    if points is None or len(points) < 2:
        if ostium.direction is None:
            ostium.diameter = None
            return
        point, tangent = ostium.center + distance_mm * ostium.direction, ostium.direction
    else:
        s = cumulative_arc_length(points)
        if s[-1] >= distance_mm:
            point = np.array([np.interp(distance_mm, s, points[:, c]) for c in range(3)])
            i = int(np.clip(np.searchsorted(s, distance_mm), 1, len(points) - 1))
            j = min(i + 2, len(points) - 1)
            tangent = _unit(points[j] - points[max(i - 2, 0)])
        else:
            tangent = _unit(points[-1] - points[max(len(points) - 4, 0)])
            point = points[-1] + (distance_mm - s[-1]) * tangent
    contour = branch_section(volume, point, tangent)
    ostium.diameter = contour.metrics() if contour is not None else None
    if contour is None:
        ostium.warnings.append("branch_diameter_not_found")


def _finish(volume: Volume, tube: AorticTube, start: np.ndarray, direction: np.ndarray, params: OstiumParameters,
            source: str) -> Optional[Ostium]:
    trace = trace_branch(volume, start, direction, params.trace_length_mm)
    if len(trace) < 3:
        return None
    center, d = _ostium_from_trace(tube, trace, direction, params)
    if center is None:
        return None
    beyond = trace[np.einsum("ij,j->i", trace - center, d) > 0.5]
    ostium = Ostium(center=center, direction=d, branch_points=beyond, source=source)
    measure_branch(volume, ostium, params.diameter_distance_mm)
    return ostium


# ---------------------------------------------------------------------------- stumps
@dataclass
class Stump:
    """A connected candidate region outside the aortic surface."""

    points: np.ndarray  # voxel centres (RAS)
    delta: np.ndarray  # signed distances to the surface
    s: np.ndarray  # arc lengths on the path
    radius_mm: float  # largest inscribed radius (distance transform)
    rejected: Optional[str] = None


def find_stumps(volume: Volume, tube: AorticTube, params: OstiumParameters) -> List[Stump]:
    g = volume.geometry
    s_lo, s_hi = tube.s_range
    s_lo, s_hi = s_lo + params.end_margin_mm, s_hi - params.end_margin_mm
    if s_hi <= s_lo:
        return []
    path = tube.path
    sel = (path.arc_length >= s_lo) & (path.arc_length <= s_hi)
    reach = float(np.nanmax(tube.radii)) + params.shell_mm + 2.0
    lo_ras = path.points[sel].min(axis=0) - reach
    hi_ras = path.points[sel].max(axis=0) + reach
    corners = np.array([[x, y, z] for x in (lo_ras[0], hi_ras[0]) for y in (lo_ras[1], hi_ras[1]) for z in (lo_ras[2], hi_ras[2])])
    ijk = g.ras_to_ijk(corners)
    dims = np.array(g.dimensions)
    start = np.clip(np.floor(ijk.min(axis=0)).astype(int), 0, dims - 1)
    stop = np.clip(np.ceil(ijk.max(axis=0)).astype(int), 0, dims - 1)
    sub = volume.data[start[2] : stop[2] + 1, start[1] : stop[1] + 1, start[0] : stop[0] + 1]
    thresholds, levels = tube.thresholds[tube.valid], tube.levels[tube.valid]
    pre = (sub >= np.min(thresholds)) & (sub <= np.max(levels) + params.upper_margin_hu)
    kji = np.argwhere(pre)
    if len(kji) == 0:
        return []
    points = g.ijk_to_ras((kji[:, ::-1] + start).astype(float))
    s, dist = tube.arc_lengths(points)
    keep = (s >= s_lo) & (s <= s_hi) & (dist <= reach)
    kji, points, s = kji[keep], points[keep], s[keep]
    delta = tube.signed_distance(points, s)
    thr, lev = tube.section_values(s)
    values = sub[kji[:, 0], kji[:, 1], kji[:, 2]].astype(float)
    keep = (
        np.isfinite(delta)
        & (delta > params.surface_margin_mm)
        & (delta <= params.shell_mm)
        & (values >= thr)
        & (values <= lev + params.upper_margin_hu)
    )
    kji, points, s, delta = kji[keep], points[keep], s[keep], delta[keep]
    if len(kji) == 0:
        return []
    mask = np.zeros(sub.shape, dtype=bool)
    mask[kji[:, 0], kji[:, 1], kji[:, 2]] = True
    labels, count = ndimage.label(mask, structure=_FULL)
    spacing_kji = tuple(np.asarray(g.spacing, dtype=float)[::-1])
    edt = ndimage.distance_transform_edt(mask, sampling=spacing_kji)
    voxel_labels = labels[kji[:, 0], kji[:, 1], kji[:, 2]]
    order = np.argsort(voxel_labels)
    bounds = np.searchsorted(voxel_labels[order], np.arange(1, count + 2))
    stumps: List[Stump] = []
    for label in range(1, count + 1):
        idx = order[bounds[label - 1] : bounds[label]]
        if len(idx) == 0 or delta[idx].min() > params.touch_mm:
            continue  # not attached to the aorta
        core = idx[(delta[idx] >= 1.0) & (delta[idx] <= 8.0)]
        radius = float(edt[kji[core, 0], kji[core, 1], kji[core, 2]].max()) if len(core) else 0.0
        stump = Stump(points[idx], delta[idx], s[idx], radius)
        local_r = tube.local_radius(float(np.median(s[idx])))
        contact = idx[delta[idx] <= params.touch_mm]
        voxel = float(np.mean(g.spacing))
        if 2.0 * radius + voxel < params.min_branch_diameter_mm - 1.0:
            stump.rejected = "too_small"
        elif delta[idx].max() < 3.0:
            stump.rejected = "no_exit"
        elif np.isfinite(local_r) and radius > params.max_branch_to_aorta_ratio * local_r:
            stump.rejected = "too_thick"
        elif np.ptp(s[contact]) > params.max_contact_length_mm or np.ptp(s[idx]) > params.max_stump_length_mm:
            stump.rejected = "too_long"
        stumps.append(stump)
    return stumps


def _stump_start(stump: Stump, tube: AorticTube):
    """Starting point and outward direction for tracing a stump."""
    sel = (stump.delta >= 3.0) & (stump.delta <= 6.0)
    if sel.sum() < 3:
        sel = (stump.delta >= 2.0) & (stump.delta <= 10.0)
    if sel.sum() < 3:
        sel = np.ones(len(stump.delta), dtype=bool)
    start = stump.points[sel].mean(axis=0)
    outer = stump.delta >= 2.0
    if outer.sum() < 3:
        outer = np.ones(len(stump.delta), dtype=bool)
    region, depth = stump.points[outer], stump.delta[outer]
    _, _, vt = np.linalg.svd(region - region.mean(axis=0))
    d = vt[0]
    projection = region @ d
    if np.ptp(projection) > 0 and np.ptp(depth) > 0 and np.corrcoef(projection, depth)[0, 1] < 0:
        d = -d  # pointing away from the aorta
    return start, _unit(d)


def _merge(ostia: List[Ostium], distance_mm: float) -> List[Ostium]:
    """Drop duplicates (the earlier entry wins: network paths are listed first)."""
    out: List[Ostium] = []
    for o in ostia:
        if all(np.linalg.norm(o.center - p.center) > distance_mm for p in out):
            out.append(o)
    return out


def ostia_from_paths(volume: Volume, tube: AorticTube, other_paths: Sequence[CurvePath], params: OstiumParameters) -> List[Ostium]:
    """Ostia of branches that are centerline paths of their own (diverging from the aortic path)."""
    out = []
    for other in other_paths:
        delta = tube.signed_distance(other.points)
        outside = np.isfinite(delta) & (delta > 0.5)
        inside = np.isfinite(delta) & (delta < 0.0)
        if not inside.any() or not outside.any():
            continue
        last_inside = int(np.flatnonzero(inside).max())
        after = np.flatnonzero(outside[last_inside:])
        if len(after) == 0:
            continue
        first_out = last_inside + int(after[0])
        start_index = min(len(other.points) - 1, first_out + int(round(3.0 / 0.5)))
        direction = _unit(other.points[min(start_index + 4, len(other.points) - 1)] - other.points[first_out])
        ostium = _finish(volume, tube, other.points[start_index], direction, params, SOURCE_PATH)
        if ostium is not None:
            out.append(ostium)
    return out


def find_ostia(
    volume: Volume,
    path: CurvePath,
    params: Optional[OstiumParameters] = None,
    tube: Optional[AorticTube] = None,
    other_paths: Sequence[CurvePath] = (),
    progress: Optional[ProgressCallback] = None,
) -> Tuple[List[Ostium], AorticTube, List[Stump]]:
    """All ostia along ``path``: (ostia proximal -> distal, the aortic tube, the stumps examined)."""
    params = params or OstiumParameters()
    if tube is None:
        tube = AorticTube(volume, path, progress=None if progress is None else (lambda f, s: progress(0.5 * f, s)))
    _check(progress, 0.5, "ostia.stumps")
    stumps = find_stumps(volume, tube, params)
    found = ostia_from_paths(volume, tube, other_paths, params)
    accepted = [st for st in stumps if st.rejected is None]
    for i, stump in enumerate(accepted):
        _check(progress, 0.55 + 0.45 * i / max(len(accepted), 1), "ostia.trace")
        start, direction = _stump_start(stump, tube)
        ostium = _finish(volume, tube, start, direction, params, SOURCE_AUTO)
        if ostium is None:
            stump.rejected = "no_trace"
            continue
        if ostium.diameter is not None and ostium.diameter.mean_diameter_mm < params.min_branch_diameter_mm - 0.5:
            stump.rejected = "too_small"
            continue
        found.append(ostium)
    _check(progress, 1.0, "ostia.trace")
    ostia = _merge(found, params.merge_distance_mm)
    ostia.sort(key=lambda o: path.nearest_arc_length(o.center)[0])
    return ostia, tube, stumps


def manual_ostium(
    volume: Volume,
    tube: AorticTube,
    wall_point: np.ndarray,
    branch_point: Optional[np.ndarray] = None,
    params: Optional[OstiumParameters] = None,
) -> Ostium:
    """Ostium clicked by the user on the aortic wall (optionally a second click inside the branch).

    The click is moved radially onto the aortic surface. With a branch point the branch is
    traced towards it (diameter and take-off angles); otherwise the radial direction is used.
    """
    params = params or OstiumParameters()
    path = tube.path
    p = np.asarray(wall_point, dtype=float)
    s, _ = path.nearest_arc_length(p)
    centre = path.point_at(s)
    t = path.tangent_at(s)
    radial = p - centre
    radial -= np.dot(radial, t) * t
    if np.linalg.norm(radial) < 1e-6:
        raise ValueError("the click is on the centerline, not on the wall")
    radial = _unit(radial)
    on_surface = tube.crossing(centre, centre + (tube.local_radius(s) * 3.0 + 10.0) * radial)
    direction = radial if branch_point is None else _unit(np.asarray(branch_point, dtype=float) - on_surface)
    trace = trace_branch(volume, on_surface + 3.0 * direction, direction, params.trace_length_mm)
    ostium = Ostium(center=on_surface, direction=direction, branch_points=trace, source=SOURCE_MANUAL, confirmed=True)
    if branch_point is not None and len(trace) >= 3:
        _, d = _fit_line(trace[: min(len(trace), 10)], direction)
        ostium.direction = d
    measure_branch(volume, ostium, params.diameter_distance_mm)
    return ostium


def nearest_stump_distance(ostium: Ostium, stumps: Sequence[Stump]) -> float:
    """Distance from an ostium to the closest stump voxel (diagnostics)."""
    best = math.inf
    for stump in stumps:
        best = min(best, float(np.min(np.linalg.norm(stump.points - ostium.center, axis=1))))
    return best
