"""Branching phantoms for lumen segmentation and centerline extraction (stage 2).

``aortoiliac``: abdominal aorta bifurcating into two common iliac arteries, optionally
with a vertebral body touching the posterior aortic wall (cortical and cancellous bone,
the classic cause of region-growing leaks) and calcifications at the aortic wall.

Ground truth: the centerline of every branch, the paths start -> each iliac end, the
bifurcation point with an exclusion radius (the "true" centerline of a union of tubes is
ill-defined inside the bifurcation), and suggested seed points.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry.curves import cumulative_arc_length, resample_polyline, unit_tangents
from ..geometry.frames import rotation_minimizing_frames
from ..volume import CT_LIKE_DIRECTIONS, ImageGeometry, Volume
from .artifacts import add_noise, apply_blur, to_ct_int16
from .library import BACKGROUND_HU, LUMEN_HU, TRUTH_STEP_MM
from .model import Phantom, PhantomTruth, TubeLayer, ellipse_reference_metrics
from .rasterize import paint, rasterize_tube_fractions, solid_fraction

CANCELLOUS_BONE_HU = 250.0  # overlaps the lumen range: a leak fills the vertebra
CORTICAL_BONE_HU = 900.0
CALCIUM_HU = 1000.0


def quadratic_bezier(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float], n: int = 600) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    a, b, c = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t**2 * c


def _dense_tube(points: np.ndarray, radius: float, spacing: Sequence[float]):
    dense = resample_polyline(points, min(0.1, min(spacing) / 7.0))
    tangents = unit_tangents(dense)
    tangents, normals, _ = rotation_minimizing_frames(dense, None, tangents)
    n = len(dense)
    layer = TubeLayer("lumen", LUMEN_HU, np.full(n, radius), np.full(n, radius))
    return dense, tangents, normals, layer


def aortoiliac(
    aorta_diameter_mm: float = 20.0,
    iliac_diameter_mm: float = 10.0,
    divergence_deg: float = 25.0,
    spacing: Sequence[float] = (0.75, 0.75, 1.25),
    with_bone: bool = False,
    bone_gap_mm: float = 0.75,
    with_calcifications: bool = False,
    blur_sigma_mm: float = 0.5,
    noise_sigma_hu: float = 15.0,
    seed: Optional[int] = 3,
    supersampling: int = 3,
    name: Optional[str] = None,
) -> Phantom:
    """Aorta (from above the volume) bifurcating at the origin into two iliac arteries."""
    ra, ri = 0.5 * aorta_diameter_mm, 0.5 * iliac_diameter_mm
    bifurcation = np.zeros(3)
    aorta = np.array([[0.0, 0.0, 130.0], bifurcation])
    a = math.radians(divergence_deg)
    branches: Dict[str, np.ndarray] = {"aorta": resample_polyline(aorta, TRUTH_STEP_MM)}
    for side, sign in (("right_iliac", 1.0), ("left_iliac", -1.0)):  # patient right = +R
        d0 = np.array([sign * math.sin(a), 0.0, -math.cos(a)])
        curve = quadratic_bezier(bifurcation, bifurcation + 40.0 * d0, [sign * 40.0, 10.0, -110.0])
        branches[side] = resample_polyline(curve, TRUTH_STEP_MM)

    y_min = -(ra + bone_gap_mm + 2 * 16.0 + 5.0) if with_bone else -25.0
    geometry = ImageGeometry.covering_box((-60.0, y_min, -95.0), (60.0, 30.0, 110.0), spacing, CT_LIKE_DIRECTIONS)

    lumen = np.zeros(geometry.array_shape, dtype=np.float32)
    for points, radius in ((branches["aorta"], ra), (branches["right_iliac"], ri), (branches["left_iliac"], ri)):
        dense, tangents, normals, layer = _dense_tube(points, radius, spacing)
        fraction = rasterize_tube_fractions(geometry, dense, tangents, normals, [layer], supersampling=supersampling)
        lumen = np.maximum(lumen, fraction[0])

    image = np.full(geometry.array_shape, BACKGROUND_HU)
    objects: List[Dict[str, Any]] = []
    if with_bone:
        rv, cortex = 16.0, 1.0
        # Separated from the lumen only by the aortic wall (``bone_gap_mm``): partial volume
        # bridges lumen and bone, as when the aorta lies on a vertebra.
        center = np.array([0.0, -(ra + bone_gap_mm + rv), 0.0])
        z_range = (-30.0, 100.0)

        def radial(points: np.ndarray) -> np.ndarray:
            return np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])

        def within_z(points: np.ndarray) -> np.ndarray:
            return (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])

        lo = np.array([center[0] - rv, center[1] - rv, z_range[0]])
        hi = np.array([center[0] + rv, center[1] + rv, z_range[1]])
        cancellous = solid_fraction(geometry, lambda p: (radial(p) <= rv - cortex) & within_z(p), lo, hi)
        cortical = solid_fraction(
            geometry, lambda p: (radial(p) <= rv) & (radial(p) > rv - cortex) & within_z(p), lo, hi
        )
        image = paint(image, cancellous, CANCELLOUS_BONE_HU)
        image = paint(image, cortical, CORTICAL_BONE_HU)
        objects.append(
            {
                "kind": "vertebra",
                "gap_to_lumen_mm": bone_gap_mm,
                "axis_point": center.tolist(),
                "radius_mm": rv,
                "cortex_mm": cortex,
                "z_range_mm": list(z_range),
                "cancellous_hu": CANCELLOUS_BONE_HU,
                "cortical_hu": CORTICAL_BONE_HU,
            }
        )
    image = paint(image, lumen, LUMEN_HU)
    if with_calcifications:
        for angle_deg, z in ((30.0, 70.0), (160.0, 40.0), (-60.0, 20.0)):
            t = math.radians(angle_deg)
            c = np.array([ra * math.cos(t), ra * math.sin(t), z])
            r = 1.5
            fraction = solid_fraction(geometry, lambda p, c=c, r=r: np.linalg.norm(p - c, axis=1) <= r, c - r, c + r)
            image = paint(image, fraction, CALCIUM_HU)
            objects.append({"kind": "calcification", "center": c.tolist(), "radius_mm": r, "hu": CALCIUM_HU})

    image = apply_blur(image, geometry, blur_sigma_mm)
    image = add_noise(image, noise_sigma_hu, seed)
    volume = Volume(data=to_ct_int16(image), geometry=geometry)

    # Main centerline fields of the truth = the right path (aorta + right iliac).
    right_path = np.vstack([branches["aorta"], branches["right_iliac"][1:]])
    radii = np.concatenate([np.full(len(branches["aorta"]), ra), np.full(len(branches["right_iliac"]) - 1, ri)])
    tangents, normals, _ = rotation_minimizing_frames(right_path, None)
    start = np.array([0.0, 0.0, 100.0])
    ends = [_point_at_z(branches[side], -85.0) for side in ("right_iliac", "left_iliac")]
    paths = {"right": ["aorta", "right_iliac"], "left": ["aorta", "left_iliac"]}
    truth = PhantomTruth(
        name=name or ("aortoiliac_bone" if with_bone else "aortoiliac"),
        description="Aorto-iliac bifurcation" + (" with vertebra and calcifications" if with_bone else ""),
        geometry=geometry,
        background_hu=BACKGROUND_HU,
        centerline_points=right_path,
        centerline_tangents=tangents,
        centerline_normals=normals,
        layers=[TubeLayer("lumen", LUMEN_HU, radii, radii)],
        expected={
            "aorta": ellipse_reference_metrics(ra, ra),
            "iliac": ellipse_reference_metrics(ri, ri),
            "path_lengths_mm": {
                k: float(cumulative_arc_length(np.vstack([branches[b] for b in v]))[-1]) for k, v in paths.items()
            },
            "seeds": {"start": start.tolist(), "ends": [e.tolist() for e in ends]},
        },
        parameters={
            "aorta_diameter_mm": aorta_diameter_mm,
            "iliac_diameter_mm": iliac_diameter_mm,
            "divergence_deg": divergence_deg,
            "spacing": list(map(float, spacing)),
            "with_bone": with_bone,
            "with_calcifications": with_calcifications,
        },
        blur_sigma_mm=blur_sigma_mm,
        noise_sigma_hu=noise_sigma_hu,
        seed=seed,
        branches=[
            {"name": k, "points": v, "radius_mm": ra if k == "aorta" else ri} for k, v in branches.items()
        ],
        paths=paths,
        junctions=[{"point": bifurcation.tolist(), "exclusion_radius_mm": 1.5 * ra}],
        objects=objects,
    )
    return Phantom(volume=volume, truth=truth)


def _point_at_z(points: np.ndarray, z: float) -> np.ndarray:
    i = int(np.argmin(np.abs(points[:, 2] - z)))
    return points[i].copy()


def suggested_seeds(truth: PhantomTruth) -> Tuple[np.ndarray, List[np.ndarray]]:
    seeds = truth.expected["seeds"]
    return np.asarray(seeds["start"], dtype=float), [np.asarray(e, dtype=float) for e in seeds["ends"]]
