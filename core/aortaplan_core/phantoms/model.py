"""Phantom data model: ground truth that is saved next to the volume as JSON."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .. import __version__
from ..fileio.nrrd import read_nrrd, write_nrrd
from ..volume import ImageGeometry, Volume

PathLike = Union[str, "os.PathLike[str]"]

TRUTH_SCHEMA_VERSION = 2  # 2: branches, paths, junctions, objects


def ellipse_reference_metrics(semi_axis_a: float, semi_axis_b: float) -> Dict[str, float]:
    """Exact (analytic) cross-section metrics of an ellipse, used as ground truth.

    The perimeter uses Ramanujan's second approximation (relative error < 1e-9 for
    the eccentricities found in vessels).
    """
    a, b = float(max(semi_axis_a, semi_axis_b)), float(min(semi_axis_a, semi_axis_b))
    h = ((a - b) / (a + b)) ** 2
    perimeter = math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))
    area = math.pi * a * b
    return {
        "min_diameter_mm": 2.0 * b,
        "max_diameter_mm": 2.0 * a,
        "mean_diameter_mm": a + b,
        "area_mm2": area,
        "perimeter_mm": perimeter,
        "area_equivalent_diameter_mm": 2.0 * math.sqrt(area / math.pi),
        "perimeter_equivalent_diameter_mm": perimeter / math.pi,
    }


@dataclass
class TubeLayer:
    """One nested layer of a tubular phantom (outermost layers are listed first).

    The cross-section is an ellipse in the local frame: semi-axis ``semi_axis_n``
    along the frame normal N and ``semi_axis_b`` along the binormal B, one value per
    centerline sample. The ellipse centre may be shifted from the centerline by
    ``offset_n`` (along N) and ``offset_b`` (along B), e.g. an eccentric thrombus.
    """

    name: str
    hu: float
    semi_axis_n: np.ndarray
    semi_axis_b: np.ndarray
    offset_n: Optional[np.ndarray] = None
    offset_b: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.semi_axis_n = np.asarray(self.semi_axis_n, dtype=float)
        self.semi_axis_b = np.asarray(self.semi_axis_b, dtype=float)
        if self.semi_axis_n.shape != self.semi_axis_b.shape:
            raise ValueError("semi-axis arrays must have the same shape")
        if np.any(self.semi_axis_n <= 0) or np.any(self.semi_axis_b <= 0):
            raise ValueError("semi-axes must be positive")
        zeros = np.zeros_like(self.semi_axis_n)
        self.offset_n = zeros.copy() if self.offset_n is None else np.asarray(self.offset_n, dtype=float)
        self.offset_b = zeros.copy() if self.offset_b is None else np.asarray(self.offset_b, dtype=float)
        if self.offset_n.shape != self.semi_axis_n.shape or self.offset_b.shape != self.semi_axis_n.shape:
            raise ValueError("offset arrays must have the same shape as the semi-axes")

    @property
    def is_centered(self) -> bool:
        return not (np.any(self.offset_n) or np.any(self.offset_b))

    def subset(self, indices: np.ndarray) -> TubeLayer:
        return TubeLayer(
            self.name,
            self.hu,
            self.semi_axis_n[indices],
            self.semi_axis_b[indices],
            self.offset_n[indices],
            self.offset_b[indices],
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "hu": float(self.hu),
            "semi_axis_n_mm": np.round(self.semi_axis_n, 6).tolist(),
            "semi_axis_b_mm": np.round(self.semi_axis_b, 6).tolist(),
        }
        if not self.is_centered:
            d["offset_n_mm"] = np.round(self.offset_n, 6).tolist()
            d["offset_b_mm"] = np.round(self.offset_b, 6).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TubeLayer:
        return cls(
            name=d["name"],
            hu=d["hu"],
            semi_axis_n=d["semi_axis_n_mm"],
            semi_axis_b=d["semi_axis_b_mm"],
            offset_n=d.get("offset_n_mm"),
            offset_b=d.get("offset_b_mm"),
        )


@dataclass
class PhantomTruth:
    """Ground truth of a synthetic phantom. All coordinates are RAS mm."""

    name: str
    description: str
    geometry: ImageGeometry
    background_hu: float
    centerline_points: np.ndarray  # (N, 3)
    centerline_tangents: np.ndarray  # (N, 3)
    centerline_normals: np.ndarray  # (N, 3) rotation-minimizing normals (ellipse axis "n")
    layers: List[TubeLayer]
    expected: Dict[str, Any] = field(default_factory=dict)  # analytic reference values
    parameters: Dict[str, Any] = field(default_factory=dict)  # generator parameters
    blur_sigma_mm: float = 0.0
    noise_sigma_hu: float = 0.0
    seed: Optional[int] = None
    core_version: str = __version__
    #: Branching phantoms: named vessel centerlines {"name", "points", "radius_mm"}.
    branches: List[Dict[str, Any]] = field(default_factory=list)
    #: Named paths from the start point to an end point: list of branch names, proximal first.
    paths: Dict[str, List[str]] = field(default_factory=dict)
    #: Bifurcations: {"point": [x, y, z], "exclusion_radius_mm": r}; accuracy is not judged inside.
    junctions: List[Dict[str, Any]] = field(default_factory=list)
    #: Non-vessel objects (bone, calcifications) for leak tests: {"kind", ...parameters}.
    objects: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.centerline_points = np.asarray(self.centerline_points, dtype=float)
        self.centerline_tangents = np.asarray(self.centerline_tangents, dtype=float)
        self.centerline_normals = np.asarray(self.centerline_normals, dtype=float)

    @property
    def centerline_binormals(self) -> np.ndarray:
        return np.cross(self.centerline_tangents, self.centerline_normals)

    def branch(self, name: str) -> Dict[str, Any]:
        for branch in self.branches:
            if branch["name"] == name:
                return branch
        raise KeyError(name)

    def path_points(self, path_name: str) -> np.ndarray:
        """Concatenated centerline of a named path (duplicate junction points removed)."""
        parts = [np.asarray(self.branch(name)["points"], dtype=float) for name in self.paths[path_name]]
        points = parts[0]
        for part in parts[1:]:
            points = np.vstack([points, part[1:] if np.allclose(part[0], points[-1]) else part])
        return points

    def layer(self, name: str) -> TubeLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "coordinate_system": "RAS",
            "units": "mm",
            "name": self.name,
            "description": self.description,
            "geometry": self.geometry.to_dict(),
            "background_hu": float(self.background_hu),
            "centerline": {
                "points": np.round(self.centerline_points, 6).tolist(),
                "tangents": np.round(self.centerline_tangents, 9).tolist(),
                "normals": np.round(self.centerline_normals, 9).tolist(),
            },
            "layers": [layer.to_dict() for layer in self.layers],
            "expected": self.expected,
            "parameters": self.parameters,
            "blur_sigma_mm": float(self.blur_sigma_mm),
            "noise_sigma_hu": float(self.noise_sigma_hu),
            "seed": self.seed,
            "core_version": self.core_version,
            "branches": [
                {
                    "name": b["name"],
                    "points": np.round(np.asarray(b["points"], dtype=float), 6).tolist(),
                    "radius_mm": float(b["radius_mm"]),
                }
                for b in self.branches
            ],
            "paths": {k: list(v) for k, v in self.paths.items()},
            "junctions": self.junctions,
            "objects": self.objects,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhantomTruth:
        if d.get("schema_version") not in (1, TRUTH_SCHEMA_VERSION):
            raise ValueError(f"unsupported truth schema version {d.get('schema_version')}")
        cl = d["centerline"]
        return cls(
            name=d["name"],
            description=d["description"],
            geometry=ImageGeometry.from_dict(d["geometry"]),
            background_hu=d["background_hu"],
            centerline_points=cl["points"],
            centerline_tangents=cl["tangents"],
            centerline_normals=cl["normals"],
            layers=[TubeLayer.from_dict(x) for x in d["layers"]],
            expected=d.get("expected", {}),
            parameters=d.get("parameters", {}),
            blur_sigma_mm=d.get("blur_sigma_mm", 0.0),
            noise_sigma_hu=d.get("noise_sigma_hu", 0.0),
            seed=d.get("seed"),
            core_version=d.get("core_version", ""),
            branches=d.get("branches", []),
            paths=d.get("paths", {}),
            junctions=d.get("junctions", []),
            objects=d.get("objects", []),
        )


@dataclass
class Phantom:
    volume: Volume
    truth: PhantomTruth


def phantom_paths(directory: PathLike, name: str) -> Dict[str, str]:
    base = os.path.join(os.fspath(directory), name)
    return {"volume": base + ".nrrd", "truth": base + ".truth.json"}


def save_phantom(phantom: Phantom, directory: PathLike) -> Dict[str, str]:
    """Write ``<name>.nrrd`` and ``<name>.truth.json`` into ``directory``; returns the paths."""
    os.makedirs(directory, exist_ok=True)
    paths = phantom_paths(directory, phantom.truth.name)
    write_nrrd(paths["volume"], phantom.volume)
    with open(paths["truth"], "w", encoding="utf-8") as f:
        json.dump(phantom.truth.to_dict(), f, ensure_ascii=False, indent=1)
    return paths


def load_phantom(directory: PathLike, name: str) -> Phantom:
    paths = phantom_paths(directory, name)
    volume = read_nrrd(paths["volume"])
    with open(paths["truth"], encoding="utf-8") as f:
        truth = PhantomTruth.from_dict(json.load(f))
    return Phantom(volume=volume, truth=truth)
