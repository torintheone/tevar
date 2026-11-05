"""Centerline extraction through a segmented lumen (SPEC 6.2, item 5 and its alternative).

This is the Slicer-independent method ("minimal path"); in Slicer the same lumen can also
be passed to VMTK Extract Centerline. Steps:

1. Distance to the lumen boundary ``D`` (mm) by an exact Euclidean distance transform.
2. Minimal-cost path from the start voxel to every end voxel over the 26-connected voxel
   graph of the lumen with cost ``1 / D^p``: the path keeps to the middle of the vessel
   (the discrete analogue of VMTK's maximal-inscribed-sphere path). One Dijkstra run gives
   all paths.
3. Smoothing along the arc length.
4. Sub-voxel recentering: in the plane perpendicular to the path, rays find the lumen
   boundary at the half-maximum level (linear interpolation of the CT values) and a circle
   fitted to the boundary points gives the centre. Rays that escape into a side branch are
   rejected as outliers. Repeated a few times with smoothing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from ..geometry.curves import cumulative_arc_length, resample_polyline, unit_tangents
from ..geometry.frames import rotation_minimizing_frames
from ..lumen.segmentation import Cancelled, LumenResult
from ..volume import Volume

ProgressCallback = Callable[[float, str], bool]

METHOD_MINIMAL_PATH = "minimal_path"
METHOD_VMTK = "vmtk"


@dataclass
class CenterlineParameters:
    cost_exponent: float = 2.0
    max_graph_nodes: int = 3_000_000
    smoothing_sigma_mm: float = 2.0
    refine_iterations: int = 3
    refine_rays: int = 32
    ray_step_mm: float = 0.2
    max_shift_per_iteration_mm: float = 2.0
    min_valid_ray_fraction: float = 0.6
    outlier_radius_factor: float = 1.6
    output_step_mm: float = 0.5
    side_tolerance_mm: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CenterlineParameters:
        return cls(**d)


@dataclass
class CenterlinePath:
    """One path from the start point to one end point (proximal first)."""

    name: str
    points: np.ndarray  # (N, 3) RAS
    radii_mm: np.ndarray  # (N,) lumen radius estimate
    side: str = "center"  # patient side of the end point relative to the start: right | left | center
    end_index: int = 0

    @property
    def length_mm(self) -> float:
        return float(cumulative_arc_length(self.points)[-1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "points": np.round(self.points, 4).tolist(),
            "radii_mm": np.round(self.radii_mm, 4).tolist(),
            "side": self.side,
            "end_index": self.end_index,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CenterlinePath:
        return cls(
            name=d["name"],
            points=np.asarray(d["points"], dtype=float),
            radii_mm=np.asarray(d["radii_mm"], dtype=float),
            side=d.get("side", "center"),
            end_index=d.get("end_index", 0),
        )


@dataclass
class CenterlineNetwork:
    method: str
    start: np.ndarray
    paths: List[CenterlinePath] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "start": np.asarray(self.start, dtype=float).tolist(),
            "paths": [p.to_dict() for p in self.paths],
            "warnings": list(self.warnings),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CenterlineNetwork:
        return cls(
            method=d["method"],
            start=np.asarray(d["start"], dtype=float),
            paths=[CenterlinePath.from_dict(p) for p in d.get("paths", [])],
            warnings=list(d.get("warnings", [])),
            parameters=dict(d.get("parameters", {})),
        )


def _check(progress: Optional[ProgressCallback], fraction: float, step: str) -> None:
    if progress is not None and progress(fraction, step) is False:
        raise Cancelled()


def side_of(end: np.ndarray, start: np.ndarray, tolerance_mm: float) -> str:
    """Patient side of ``end`` relative to ``start`` (RAS: +x is the patient's right)."""
    dx = float(end[0] - start[0])
    if dx > tolerance_mm:
        return "right"
    if dx < -tolerance_mm:
        return "left"
    return "center"


def _block_reduce(mask: np.ndarray, factor: int) -> np.ndarray:
    """Majority-vote downsampling of a boolean mask by an integer factor."""
    if factor <= 1:
        return mask
    pad = [(0, (-n) % factor) for n in mask.shape]
    m = np.pad(mask, pad).astype(np.float32)
    k, j, i = (n // factor for n in m.shape)
    m = m.reshape(k, factor, j, factor, i, factor).mean(axis=(1, 3, 5))
    return m >= 0.5


def _graph(mask: np.ndarray, cost: np.ndarray, spacing_kji: np.ndarray):
    index = -np.ones(mask.shape, dtype=np.int64)
    nodes = np.flatnonzero(mask)
    index.flat[nodes] = np.arange(len(nodes))
    rows, cols, weights = [], [], []
    shape = mask.shape
    offsets = [(dk, dj, di) for dk in (-1, 0, 1) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dk, dj, di) > (0, 0, 0)]
    for offset in offsets:
        sa = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(offset, shape))
        sb = tuple(slice(max(0, o), n - max(0, -o)) for o, n in zip(offset, shape))
        both = mask[sa] & mask[sb]
        if not both.any():
            continue
        length = float(np.sqrt(np.sum((np.asarray(offset) * spacing_kji) ** 2)))
        rows.append(index[sa][both])
        cols.append(index[sb][both])
        weights.append(length * 0.5 * (cost[sa][both] + cost[sb][both]))
    n = len(nodes)
    if not rows:
        return sparse.csr_matrix((n, n)), nodes
    graph = sparse.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    return graph, nodes


def _nearest_mask_voxel(mask: np.ndarray, kji: np.ndarray) -> Optional[np.ndarray]:
    k, j, i = np.clip(np.round(kji).astype(int), 0, np.array(mask.shape) - 1)
    if mask[k, j, i]:
        return np.array([k, j, i])
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    return coords[int(np.argmin(np.sum((coords - np.array([k, j, i])) ** 2, axis=1)))]


def _smooth(points: np.ndarray, sigma_mm: float, step_mm: float) -> np.ndarray:
    pts = resample_polyline(points, step_mm)
    if sigma_mm <= 0 or len(pts) < 5:
        return pts
    smoothed = ndimage.gaussian_filter1d(pts, sigma_mm / step_mm, axis=0, mode="nearest")
    # Keep the end points where the user put them.
    weight = np.clip(np.minimum(np.arange(len(pts)), np.arange(len(pts))[::-1]) * step_mm / (2 * sigma_mm), 0, 1)
    return weight[:, np.newaxis] * smoothed + (1 - weight[:, np.newaxis]) * pts


def boundary_radii(
    volume: Volume,
    points: np.ndarray,
    tangents: np.ndarray,
    threshold_hu: float,
    max_radius_mm: np.ndarray,
    n_rays: int = 32,
    step_mm: float = 0.2,
):
    """Distances along in-plane rays to the first crossing below ``threshold_hu``.

    Returns ``(radii, directions_u, directions_v, angles)``: radii (P, R) with NaN where no
    boundary was found, and the in-plane basis vectors (P, 3) used for the angles.
    """
    _, u, _ = rotation_minimizing_frames(points, None, tangents)
    v = np.cross(tangents, u)
    angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    t = np.arange(0.0, float(np.max(max_radius_mm)) + step_mm, step_mm)
    radii = np.full((len(points), n_rays), np.nan)
    chunk = max(1, 400000 // (n_rays * len(t)))
    for start in range(0, len(points), chunk):
        sl = slice(start, start + chunk)
        dirs = np.cos(angles)[np.newaxis, :, np.newaxis] * u[sl, np.newaxis, :] + np.sin(angles)[
            np.newaxis, :, np.newaxis
        ] * v[sl, np.newaxis, :]
        pos = points[sl, np.newaxis, np.newaxis, :] + dirs[:, :, np.newaxis, :] * t[np.newaxis, np.newaxis, :, np.newaxis]
        values = volume.sample_linear(pos.reshape(-1, 3)).reshape(pos.shape[:3])
        below = values < threshold_hu  # NaN (outside the volume) is not a boundary
        below[..., 0] = False
        outside = np.isnan(values)
        n = len(t)
        first = np.where(below.any(axis=2), np.argmax(below, axis=2), n)
        first_outside = np.where(outside.any(axis=2), np.argmax(outside, axis=2), n)
        # A ray is valid if it starts in the lumen and crosses the boundary before leaving the volume.
        found = (first < n) & (first < first_outside) & (values[..., 0] >= threshold_hu)
        idx = np.clip(first, 1, len(t) - 1)
        p = np.take_along_axis(values, (idx - 1)[..., np.newaxis], axis=2)[..., 0]
        q = np.take_along_axis(values, idx[..., np.newaxis], axis=2)[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(np.isfinite(q) & (p != q), (p - threshold_hu) / (p - q), 0.0)
        r = t[idx - 1] + np.clip(frac, 0.0, 1.0) * step_mm
        r = np.where(found & (r <= max_radius_mm[sl, np.newaxis]), r, np.nan)
        radii[sl] = r
    return radii, u, v, angles


def _fit_circle(x: np.ndarray, y: np.ndarray):
    """Algebraic (Kasa) circle fit; returns centre (a, b) and radius."""
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    rhs = x**2 + y**2
    (a, b, c), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return a, b, math.sqrt(max(c + a * a + b * b, 0.0))


def recenter(
    volume: Volume,
    points: np.ndarray,
    threshold_hu: float,
    depth_mm: np.ndarray,
    params: CenterlineParameters,
):
    """One recentering pass. Returns (new points, radius estimates, number of moved points)."""
    tangents = unit_tangents(points)
    max_radius = np.clip(2.5 * depth_mm + 4.0, 6.0, 45.0)
    radii, u, v, angles = boundary_radii(
        volume, points, tangents, threshold_hu, max_radius, params.refine_rays, params.ray_step_mm
    )
    new_points = points.copy()
    estimates = depth_mm.copy()
    moved = 0
    for i in range(len(points)):
        r = radii[i]
        valid = np.isfinite(r)
        if valid.sum() < params.min_valid_ray_fraction * len(r):
            continue
        median = float(np.median(r[valid]))
        valid &= r <= params.outlier_radius_factor * median
        if valid.sum() < params.min_valid_ray_fraction * len(r):
            continue
        x, y = r[valid] * np.cos(angles[valid]), r[valid] * np.sin(angles[valid])
        a, b, radius = _fit_circle(x, y)
        shift = math.hypot(a, b)
        if shift > params.max_shift_per_iteration_mm:
            a, b = a * params.max_shift_per_iteration_mm / shift, b * params.max_shift_per_iteration_mm / shift
        new_points[i] = points[i] + a * u[i] + b * v[i]
        estimates[i] = radius
        moved += 1
    return new_points, estimates, moved


def extract_centerlines(
    volume: Volume,
    lumen: LumenResult,
    params: Optional[CenterlineParameters] = None,
    progress: Optional[ProgressCallback] = None,
    end_names: Optional[Sequence[str]] = None,
) -> CenterlineNetwork:
    """Paths from ``lumen.start`` to every ``lumen.ends`` point through the lumen mask."""
    params = params or CenterlineParameters()
    warnings: List[str] = []
    geometry = lumen.roi_geometry
    mask = lumen.mask
    spacing_kji = np.asarray(geometry.spacing, dtype=float)[::-1]

    _check(progress, 0.05, "centerline.distance")
    depth = ndimage.distance_transform_edt(mask, sampling=tuple(spacing_kji))

    count = int(mask.sum())
    factor = max(1, math.ceil((count / params.max_graph_nodes) ** (1.0 / 3.0))) if count else 1
    graph_mask = _block_reduce(mask, factor)
    graph_depth = depth[::factor, ::factor, ::factor][tuple(slice(0, n) for n in graph_mask.shape)]
    if graph_depth.shape != graph_mask.shape:  # padding of the reduced mask
        graph_depth = np.pad(graph_depth, [(0, a - b) for a, b in zip(graph_mask.shape, graph_depth.shape)])
    graph_mask &= graph_depth > 0
    cost = 1.0 / (np.maximum(graph_depth, 1e-3) ** params.cost_exponent)

    _check(progress, 0.25, "centerline.graph")
    graph, nodes = _graph(graph_mask, cost, spacing_kji * factor)
    index = -np.ones(graph_mask.shape, dtype=np.int64)
    index.flat[nodes] = np.arange(len(nodes))

    def node_of(ras: np.ndarray) -> int:
        kji = geometry.ras_to_ijk(ras)[::-1] / factor
        voxel = _nearest_mask_voxel(graph_mask, kji)
        return -1 if voxel is None else int(index[tuple(voxel)])

    start_node = node_of(lumen.start)
    if start_node < 0:
        raise ValueError("the lumen mask is empty")
    _check(progress, 0.45, "centerline.paths")
    _, predecessors = dijkstra(graph, directed=False, indices=start_node, return_predecessors=True)

    network = CenterlineNetwork(method=METHOD_MINIMAL_PATH, start=np.asarray(lumen.start, float), parameters=params.to_dict())
    flat_shape = graph_mask.shape
    for n_end, end in enumerate(lumen.ends):
        _check(progress, 0.5 + 0.45 * n_end / max(1, len(lumen.ends)), "centerline.refine")
        end_node = node_of(end)
        if end_node < 0 or (end_node != start_node and predecessors[end_node] < 0):
            warnings.append(f"end_unreachable:{n_end}")
            continue
        chain = [end_node]
        while chain[-1] != start_node:
            chain.append(int(predecessors[chain[-1]]))
        chain.reverse()
        kji = np.array(np.unravel_index(nodes[chain], flat_shape)).T * factor
        points = geometry.ijk_to_ras(kji[:, ::-1].astype(float))
        points[0], points[-1] = lumen.start, end  # exact seed positions at both ends
        points = _smooth(points, params.smoothing_sigma_mm, params.output_step_mm)
        radii = _sample_depth(depth, geometry, points)
        for _ in range(params.refine_iterations):
            points, radii, _ = recenter(volume, points, lumen.hu_range[0], radii, params)
            points = _smooth(points, 0.5 * params.smoothing_sigma_mm, params.output_step_mm)
            radii = _resample_like(radii, len(points))
        name = end_names[n_end] if end_names and n_end < len(end_names) else f"end_{n_end + 1}"
        network.paths.append(
            CenterlinePath(
                name=name,
                points=points,
                radii_mm=radii,
                side=side_of(np.asarray(end, float), lumen.start, params.side_tolerance_mm),
                end_index=n_end,
            )
        )
    network.warnings = warnings
    _check(progress, 1.0, "centerline.done")
    return network


def _sample_depth(depth: np.ndarray, geometry, points: np.ndarray) -> np.ndarray:
    kji = geometry.ras_to_ijk(points)[:, ::-1].T
    return ndimage.map_coordinates(depth, kji, order=1, mode="nearest")


def _resample_like(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)


def divergence_arc_length(path_a: np.ndarray, path_b: np.ndarray, tolerance_mm: float) -> Optional[float]:
    """Arc length along ``path_a`` where it leaves ``path_b`` (first point farther than the tolerance)."""
    from .metrics import distances_to_polyline

    d = distances_to_polyline(path_a, path_b)
    beyond = np.flatnonzero(d > tolerance_mm)
    if len(beyond) == 0:
        return None
    return float(cumulative_arc_length(path_a)[beyond[0]])
