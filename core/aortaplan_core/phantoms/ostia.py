"""Phantoms with branches at known positions, clock positions and angles (stage 5).

``visceral_aorta``: abdominal aorta (gently bowed anteriorly) with the celiac trunk, SMA,
both renal arteries, an accessory right renal artery (3 mm) and the IMA; optionally a
vertebral body against the posterior wall (it must not be taken for a branch).

``aortic_arch``: ascending aorta, arch and descending aorta in the sagittal plane with
the brachiocephalic trunk, left common carotid and left subclavian arteries.

Each branch is a straight tube starting inside the aorta. Visceral branches are given by
their ostium on the aortic cylinder (arc length ``s0`` and clock angle, ``ostia.clock``) and
the direction in which they leave it (craniocaudal angle to the distal aortic axis, axial
deviation from radial); the arch branches by a direction from the centerline point at ``s0``. The truth of an
ostium is the point where the branch centerline crosses the aortic lumen surface (the
aortic cylinder); its clock angles and take-off angles are evaluated there on the truth
centerline with the same conventions as the measurement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..centerline.metrics import distances_to_polyline
from ..centerline.navigation import CurvePath
from ..geometry.curves import resample_polyline
from ..geometry.frames import rotation_minimizing_frames
from ..ostia.clock import CLOCK_MODES, MODE_ANTERIOR, clock_angle, clock_frame, takeoff_angles
from ..volume import CT_LIKE_DIRECTIONS, ImageGeometry, Volume
from .artifacts import add_noise, apply_blur, to_ct_int16
from .branching import CANCELLOUS_BONE_HU, CORTICAL_BONE_HU, _dense_tube
from .library import BACKGROUND_HU, LUMEN_HU, TRUTH_STEP_MM
from .model import Phantom, PhantomTruth, TubeLayer, ellipse_reference_metrics
from .rasterize import paint, rasterize_tube_fractions, solid_fraction


def _branch_line(path: CurvePath, spec: Dict[str, Any], aorta_radius: float):
    """(start point inside the aorta, unit direction) of a branch.

    Either ``direction`` (RAS) from the centerline point at ``s0`` (a radial branch), or an
    ostium on the aortic cylinder at ``s0`` and ``clock_deg`` with the branch leaving it at
    ``craniocaudal_deg`` and ``axial_deg`` (deviation from radial, clockwise positive).
    """
    s0 = spec["s0"]
    if "direction" in spec:
        d = np.asarray(spec["direction"], dtype=float)
        return path.point_at(s0), d / np.linalg.norm(d)
    frame = clock_frame(path, s0, MODE_ANTERIOR)
    t = math.radians(spec["clock_deg"])
    radial = math.sin(t) * frame.right + math.cos(t) * frame.up
    clockwise = math.cos(t) * frame.right - math.sin(t) * frame.up
    a = math.radians(spec.get("axial_deg", 0.0))
    in_plane = math.cos(a) * radial + math.sin(a) * clockwise
    b = math.radians(spec["craniocaudal_deg"])
    d = math.cos(b) * frame.tangent + math.sin(b) * in_plane
    d /= np.linalg.norm(d)
    ostium = frame.center + aorta_radius * radial
    # Back along the branch to the point of the line closest to the aortic axis (inside the aorta).
    back = aorta_radius * math.cos(a) / max(math.sin(b), 1e-6)
    return ostium - back * d, d


def _crossing(origin: np.ndarray, direction: np.ndarray, aorta: np.ndarray, radius: float) -> np.ndarray:
    """Point of the branch line at distance ``radius`` from the aortic centerline (bisection)."""
    lo, hi = 0.0, 4.0 * radius + 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if distances_to_polyline(origin + mid * direction, aorta)[0] < radius:
            lo = mid
        else:
            hi = mid
    return origin + 0.5 * (lo + hi) * direction


def _build(
    *,
    name: str,
    description: str,
    aorta_points: np.ndarray,
    aorta_radius: float,
    branch_specs: Sequence[Dict[str, Any]],
    roi_min: Sequence[float],
    roi_max: Sequence[float],
    spacing: Sequence[float],
    blur_sigma_mm: float,
    noise_sigma_hu: float,
    seed: Optional[int],
    supersampling: int,
    with_bone: bool = False,
    bone_axis_xy: Optional[Sequence[float]] = None,
    bone_z_range: Sequence[float] = (0.0, 0.0),
    parameters: Optional[Dict[str, Any]] = None,
) -> Phantom:
    aorta = resample_polyline(np.asarray(aorta_points, dtype=float), TRUTH_STEP_MM)
    path = CurvePath(aorta)
    geometry = ImageGeometry.covering_box(roi_min, roi_max, spacing, CT_LIKE_DIRECTIONS)
    lumen = np.zeros(geometry.array_shape, dtype=np.float32)
    tubes = [(aorta, aorta_radius)]
    ostia: List[Dict[str, Any]] = []
    branches = [{"name": "aorta", "points": aorta, "radius_mm": aorta_radius}]
    for spec in branch_specs:
        origin, d = _branch_line(path, spec, aorta_radius)
        r = 0.5 * spec["diameter_mm"]
        points = resample_polyline(np.array([origin, origin + spec.get("length_mm", 60.0) * d]), TRUTH_STEP_MM)
        tubes.append((points, r))
        branches.append({"name": spec["key"], "points": points, "radius_mm": r})
        center = _crossing(origin, d, aorta, aorta_radius)
        s = path.nearest_arc_length(center)[0]
        entry: Dict[str, Any] = {
            "key": spec["key"],
            "center": center.tolist(),
            "arc_length_mm": s,
            "diameter_mm": spec["diameter_mm"],
            "direction": d.tolist(),
            "clock_deg": {},
            "takeoff": {},
        }
        for mode in CLOCK_MODES:
            frame = clock_frame(path, s, mode)
            entry["clock_deg"][mode] = clock_angle(frame, center)
            entry["takeoff"][mode] = takeoff_angles(frame, center, d).to_dict()
        ostia.append(entry)
    for points, radius in tubes:
        dense, tangents, normals, layer = _dense_tube(points, radius, spacing)
        fraction = rasterize_tube_fractions(geometry, dense, tangents, normals, [layer], supersampling=supersampling)
        lumen = np.maximum(lumen, fraction[0])

    image = np.full(geometry.array_shape, BACKGROUND_HU)
    objects: List[Dict[str, Any]] = []
    if with_bone:
        rv, cortex = 16.0, 1.0
        cx, cy = bone_axis_xy
        z0, z1 = bone_z_range

        def radial(p):
            return np.hypot(p[:, 0] - cx, p[:, 1] - cy)

        def within(p):
            return (p[:, 2] >= z0) & (p[:, 2] <= z1)

        lo, hi = np.array([cx - rv, cy - rv, z0]), np.array([cx + rv, cy + rv, z1])
        image = paint(image, solid_fraction(geometry, lambda p: (radial(p) <= rv - cortex) & within(p), lo, hi), CANCELLOUS_BONE_HU)
        image = paint(
            image, solid_fraction(geometry, lambda p: (radial(p) <= rv) & (radial(p) > rv - cortex) & within(p), lo, hi),
            CORTICAL_BONE_HU,
        )
        objects.append({"kind": "vertebra", "axis_point": [cx, cy, 0.0], "radius_mm": rv, "z_range_mm": [z0, z1]})
    image = paint(image, lumen, LUMEN_HU)
    image = apply_blur(image, geometry, blur_sigma_mm)
    image = add_noise(image, noise_sigma_hu, seed)
    volume = Volume(data=to_ct_int16(image), geometry=geometry)

    tangents, normals, _ = rotation_minimizing_frames(aorta, None)
    n = len(aorta)
    truth = PhantomTruth(
        name=name,
        description=description,
        geometry=geometry,
        background_hu=BACKGROUND_HU,
        centerline_points=aorta,
        centerline_tangents=tangents,
        centerline_normals=normals,
        layers=[TubeLayer("lumen", LUMEN_HU, np.full(n, aorta_radius), np.full(n, aorta_radius))],
        expected={"aorta": ellipse_reference_metrics(aorta_radius, aorta_radius), "ostia": ostia},
        parameters=dict(parameters or {}),
        blur_sigma_mm=blur_sigma_mm,
        noise_sigma_hu=noise_sigma_hu,
        seed=seed,
        branches=branches,
        paths={"aorta": ["aorta"]},
        objects=objects,
    )
    return Phantom(volume=volume, truth=truth)


#: Visceral branches: ostium position (arc length on the aortic centerline), clock angle
#: (deg, anterior reference), craniocaudal angle to the distal axis, axial deviation, diameter.
VISCERAL_BRANCHES = (
    {"key": "celiac", "s0": 48.0, "clock_deg": 0.0, "craniocaudal_deg": 75.0, "diameter_mm": 7.0},
    {"key": "sma", "s0": 70.0, "clock_deg": 5.0, "craniocaudal_deg": 40.0, "diameter_mm": 7.0},
    {"key": "right_renal", "s0": 86.0, "clock_deg": 280.0, "craniocaudal_deg": 80.0, "axial_deg": -20.0, "diameter_mm": 6.0},
    {"key": "left_renal", "s0": 90.0, "clock_deg": 95.0, "craniocaudal_deg": 85.0, "axial_deg": 15.0, "diameter_mm": 6.5},
    {"key": "accessory_renal", "s0": 108.0, "clock_deg": 255.0, "craniocaudal_deg": 70.0, "diameter_mm": 3.0},
    {"key": "ima", "s0": 145.0, "clock_deg": 25.0, "craniocaudal_deg": 50.0, "diameter_mm": 3.5},
)


def visceral_aorta(
    aorta_diameter_mm: float = 22.0,
    length_mm: float = 175.0,
    with_bone: bool = False,
    spacing: Sequence[float] = (0.7, 0.7, 0.7),
    blur_sigma_mm: float = 0.5,
    noise_sigma_hu: float = 15.0,
    seed: Optional[int] = 21,
    supersampling: int = 3,
    name: Optional[str] = None,
) -> Phantom:
    """Abdominal aorta from z = +15 mm (above the volume) down, bowed 5 mm anteriorly."""
    ra = 0.5 * aorta_diameter_mm
    z = np.linspace(15.0, -(length_mm + 15.0), 400)
    bow = 5.0 * np.sin(math.pi * np.clip(-z / length_mm, 0.0, 1.0))
    aorta = np.stack([np.zeros_like(z), bow, z], axis=1)
    y_min = -(ra + 0.75 + 32.0 + 6.0) if with_bone else -30.0
    return _build(
        name=name or ("visceral_aorta_bone" if with_bone else "visceral_aorta"),
        description="Abdominal aorta with celiac, SMA, renal, accessory renal and IMA branches"
        + (" and a vertebra" if with_bone else ""),
        aorta_points=aorta,
        aorta_radius=ra,
        branch_specs=VISCERAL_BRANCHES,
        roi_min=(-48.0, y_min, -length_mm),
        roi_max=(48.0, 48.0, 0.0),
        spacing=spacing,
        blur_sigma_mm=blur_sigma_mm,
        noise_sigma_hu=noise_sigma_hu,
        seed=seed,
        supersampling=supersampling,
        with_bone=with_bone,
        bone_axis_xy=(0.0, -(ra + 0.75 + 16.0)),
        bone_z_range=(-length_mm, 0.0),
        parameters={"aorta_diameter_mm": aorta_diameter_mm, "length_mm": length_mm, "with_bone": with_bone},
    )


ARCH_BRANCHES = (  # arch angle phi from the ascending end of the semicircle; superior directions
    {"key": "bca", "phi_deg": 68.0, "direction": (0.35, 0.10, 1.0), "diameter_mm": 12.0},
    {"key": "lcca", "phi_deg": 95.0, "direction": (-0.10, 0.0, 1.0), "diameter_mm": 8.0},
    {"key": "lsa", "phi_deg": 122.0, "direction": (-0.40, -0.10, 1.0), "diameter_mm": 10.0},
)


def aortic_arch(
    arch_radius_mm: float = 35.0,
    aorta_diameter_mm: float = 28.0,
    leg_length_mm: float = 50.0,
    spacing: Sequence[float] = (0.8, 0.8, 0.8),
    blur_sigma_mm: float = 0.5,
    noise_sigma_hu: float = 15.0,
    seed: Optional[int] = 22,
    supersampling: int = 3,
    name: str = "aortic_arch",
) -> Phantom:
    """Arch in the sagittal plane (ascending anterior, descending posterior) with three branches."""
    R, ra = float(arch_radius_mm), 0.5 * aorta_diameter_mm
    leg = leg_length_mm + 15.0
    phi = np.linspace(0.0, math.pi, 400)
    arc = np.stack([np.zeros_like(phi), R * np.cos(phi), R * np.sin(phi)], axis=1)
    aorta = np.vstack([[[0.0, R, -leg]], arc, [[0.0, -R, -leg]]])
    path = CurvePath(resample_polyline(aorta, TRUTH_STEP_MM))
    specs = []
    for b in ARCH_BRANCHES:
        p = np.array([0.0, R * math.cos(math.radians(b["phi_deg"])), R * math.sin(math.radians(b["phi_deg"]))])
        specs.append({"key": b["key"], "s0": path.nearest_arc_length(p)[0], "direction": b["direction"],
                      "diameter_mm": b["diameter_mm"]})
    pad = ra + 6.0
    return _build(
        name=name,
        description="Aortic arch in the sagittal plane with BCT, LCCA and LSA",
        aorta_points=aorta,
        aorta_radius=ra,
        branch_specs=specs,
        roi_min=(-36.0, -R - pad, -leg_length_mm),
        roi_max=(36.0, R + pad, R + ra + 40.0),
        spacing=spacing,
        blur_sigma_mm=blur_sigma_mm,
        noise_sigma_hu=noise_sigma_hu,
        seed=seed,
        supersampling=supersampling,
        parameters={"arch_radius_mm": R, "aorta_diameter_mm": aorta_diameter_mm, "leg_length_mm": leg_length_mm},
    )


def _register_standard_phantoms() -> None:
    from .library import STANDARD_PHANTOMS

    STANDARD_PHANTOMS["visceral_aorta"] = lambda: visceral_aorta()
    STANDARD_PHANTOMS["visceral_aorta_bone"] = lambda: visceral_aorta(with_bone=True)
    STANDARD_PHANTOMS["aortic_arch"] = lambda: aortic_arch()


_register_standard_phantoms()
