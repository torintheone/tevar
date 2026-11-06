"""Library of synthetic phantoms with analytic ground truth.

Stage 0 provides simple tubes. Aneurysm with thrombus, branches, the aortic arch with
three branches, calcifications and bone are added in the stages that need them.

All phantoms use a CT-like voxel grid by default (IJK->LPS = identity, i.e. i points to
patient left and j to posterior in RAS), so that sign errors in coordinate handling
show up immediately.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..geometry.curves import cumulative_arc_length, resample_polyline, unit_tangents
from ..geometry.frames import rotation_minimizing_frames
from ..volume import CT_LIKE_DIRECTIONS, ImageGeometry, Volume
from .artifacts import add_noise, apply_blur, to_ct_int16
from .model import Phantom, PhantomTruth, TubeLayer, ellipse_reference_metrics
from .rasterize import compose_intensity, rasterize_tube_fractions

DEFAULT_SPACING: Tuple[float, float, float] = (0.7, 0.7, 0.7)
BACKGROUND_HU = 40.0  # soft tissue
LUMEN_HU = 350.0  # contrast-enhanced blood
ROI_MARGIN_MM = 5.0  # tissue around the tube inside the volume
TRUTH_STEP_MM = 0.5  # centerline sampling stored in the truth JSON

# (name, hu, semi_axis_n(s), semi_axis_b(s)[, offset_n(s), offset_b(s)]);
# s = arc length along the centerline, mm
LayerSpec = Tuple[Any, ...]


def _constant(value: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda s: np.full(np.shape(s), float(value))


def build_tube_phantom(
    *,
    name: str,
    description: str,
    centerline: np.ndarray,
    layer_specs: Sequence[LayerSpec],
    roi_min: Sequence[float],
    roi_max: Sequence[float],
    spacing: Sequence[float] = DEFAULT_SPACING,
    directions: np.ndarray = CT_LIKE_DIRECTIONS,
    reference_normal: Optional[Sequence[float]] = None,
    background_hu: float = BACKGROUND_HU,
    blur_sigma_mm: float = 0.0,
    noise_sigma_hu: float = 0.0,
    seed: Optional[int] = 0,
    supersampling: int = 4,
    expected: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    paint_extra: Optional[Callable[[np.ndarray, ImageGeometry], np.ndarray]] = None,
) -> Phantom:
    """Generic tubular phantom builder; the specific phantoms below are thin wrappers.

    ``paint_extra(image, geometry)`` may add objects (calcifications, bone) before blur and noise.
    """
    dense_step = min(0.1, min(spacing) / 7.0)
    dense = resample_polyline(np.asarray(centerline, dtype=float), dense_step)
    tangents = unit_tangents(dense)
    tangents, normals, _ = rotation_minimizing_frames(
        dense, None if reference_normal is None else np.asarray(reference_normal, dtype=float), tangents
    )
    s = cumulative_arc_length(dense)
    layers = []
    for spec in layer_specs:
        lname, hu, fn_n, fn_b = spec[:4]
        offset_n = spec[4](s) if len(spec) > 4 and spec[4] is not None else None
        offset_b = spec[5](s) if len(spec) > 5 and spec[5] is not None else None
        layers.append(TubeLayer(lname, hu, fn_n(s), fn_b(s), offset_n, offset_b))

    geometry = ImageGeometry.covering_box(roi_min, roi_max, spacing, directions)
    fractions = rasterize_tube_fractions(geometry, dense, tangents, normals, layers, supersampling=supersampling)
    image = compose_intensity(fractions, [layer.hu for layer in layers], background_hu)
    if paint_extra is not None:
        image = paint_extra(image, geometry)
    image = apply_blur(image, geometry, blur_sigma_mm)
    image = add_noise(image, noise_sigma_hu, seed)
    volume = Volume(data=to_ct_int16(image), geometry=geometry)

    stride = max(1, int(round(TRUTH_STEP_MM / dense_step)))
    keep = np.unique(np.concatenate([np.arange(0, len(dense), stride), [len(dense) - 1]]))
    truth = PhantomTruth(
        name=name,
        description=description,
        geometry=geometry,
        background_hu=background_hu,
        centerline_points=dense[keep],
        centerline_tangents=tangents[keep],
        centerline_normals=normals[keep],
        layers=[layer.subset(keep) for layer in layers],
        expected=dict(expected or {}),
        parameters=dict(parameters or {}),
        blur_sigma_mm=blur_sigma_mm,
        noise_sigma_hu=noise_sigma_hu,
        seed=seed,
    )
    truth.expected.setdefault("centerline_length_mm", float(s[-1]))
    return Phantom(volume=volume, truth=truth)


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _straight(
    *,
    name: str,
    description: str,
    semi_axis_n: float,
    semi_axis_b: float,
    length_mm: float,
    direction: Sequence[float],
    reference_normal: Optional[Sequence[float]],
    spacing: Sequence[float],
    **kwargs: Any,
) -> Phantom:
    d = _unit(direction)
    r_max = max(semi_axis_n, semi_axis_b)
    half = 0.5 * length_mm
    # The centerline runs beyond the volume so that the tube is open at the volume faces.
    extension = half + r_max + ROI_MARGIN_MM + 10.0
    centerline = np.array([-extension * d, extension * d])
    ends = np.array([-half * d, half * d])
    pad = r_max + ROI_MARGIN_MM
    return build_tube_phantom(
        name=name,
        description=description,
        centerline=centerline,
        layer_specs=[("lumen", LUMEN_HU, _constant(semi_axis_n), _constant(semi_axis_b))],
        roi_min=ends.min(axis=0) - pad,
        roi_max=ends.max(axis=0) + pad,
        spacing=spacing,
        reference_normal=reference_normal,
        expected={"lumen": ellipse_reference_metrics(semi_axis_n, semi_axis_b)},
        **kwargs,
    )


def straight_tube(
    diameter_mm: float = 25.0,
    length_mm: float = 100.0,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    spacing: Sequence[float] = DEFAULT_SPACING,
    name: str = "straight_tube",
    **kwargs: Any,
) -> Phantom:
    """Straight tube of circular cross-section; by default along the S axis."""
    r = 0.5 * diameter_mm
    params = {"diameter_mm": diameter_mm, "length_mm": length_mm, "direction": list(map(float, direction))}
    return _straight(
        name=name,
        description="Straight tube, circular cross-section",
        semi_axis_n=r,
        semi_axis_b=r,
        length_mm=length_mm,
        direction=direction,
        reference_normal=(0.0, 1.0, 0.0),
        spacing=spacing,
        parameters=params,
        **kwargs,
    )


def elliptical_tube(
    major_diameter_mm: float = 30.0,
    minor_diameter_mm: float = 20.0,
    length_mm: float = 100.0,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    major_axis_reference: Sequence[float] = (1.0, 0.0, 0.0),
    spacing: Sequence[float] = DEFAULT_SPACING,
    name: str = "elliptical_tube",
    **kwargs: Any,
) -> Phantom:
    """Straight tube with an elliptical cross-section.

    The major axis is the frame normal: the projection of ``major_axis_reference``
    on the cross-section plane (patient left-right by default).
    """
    params = {
        "major_diameter_mm": major_diameter_mm,
        "minor_diameter_mm": minor_diameter_mm,
        "length_mm": length_mm,
        "direction": list(map(float, direction)),
        "major_axis_reference": list(map(float, major_axis_reference)),
    }
    return _straight(
        name=name,
        description="Straight tube, elliptical cross-section (major axis along the frame normal)",
        semi_axis_n=0.5 * major_diameter_mm,
        semi_axis_b=0.5 * minor_diameter_mm,
        length_mm=length_mm,
        direction=direction,
        reference_normal=major_axis_reference,
        spacing=spacing,
        parameters=params,
        **kwargs,
    )


def torus_arc(
    major_radius_mm: float = 40.0,
    diameter_mm: float = 25.0,
    leg_length_mm: float = 30.0,
    spacing: Sequence[float] = DEFAULT_SPACING,
    name: str = "torus_arc",
    **kwargs: Any,
) -> Phantom:
    """Arch-like tube: a half torus in the sagittal (A-S) plane with straight vertical legs.

    The centerline goes up the anterior leg, over a semicircle of radius
    ``major_radius_mm`` centred at the origin, and down the posterior leg.
    Legs are cut by the inferior face of the volume ``leg_length_mm`` below the arc.
    """
    R = float(major_radius_mm)
    r = 0.5 * diameter_mm
    leg_total = leg_length_mm + r + ROI_MARGIN_MM + 10.0
    n_arc = max(64, int(math.ceil(math.pi * R / 0.05)))
    theta = np.linspace(0.0, math.pi, n_arc + 1)
    arc = np.stack([np.zeros_like(theta), R * np.cos(theta), R * np.sin(theta)], axis=1)
    up = np.array([[0.0, R, -leg_total]])
    down = np.array([[0.0, -R, -leg_total]])
    centerline = np.concatenate([up, arc, down])
    pad = r + ROI_MARGIN_MM
    arc_length = math.pi * R
    expected = {
        "lumen": ellipse_reference_metrics(r, r),
        "arc_length_mm": arc_length,
        "arc_start_s_mm": leg_total,
        "arc_end_s_mm": leg_total + arc_length,
        "curvature_radius_mm": R,
    }
    params = {"major_radius_mm": R, "diameter_mm": diameter_mm, "leg_length_mm": leg_length_mm}
    return build_tube_phantom(
        name=name,
        description="Half torus in the A-S plane with straight legs (arch-like)",
        centerline=centerline,
        layer_specs=[("lumen", LUMEN_HU, _constant(r), _constant(r))],
        roi_min=(-pad, -R - pad, -leg_length_mm),
        roi_max=(pad, R + pad, R + pad),
        spacing=spacing,
        reference_normal=(0.0, 1.0, 0.0),
        expected=expected,
        parameters=params,
        **kwargs,
    )


def helix(
    helix_radius_mm: float = 25.0,
    pitch_mm: float = 60.0,
    turns: float = 1.5,
    diameter_mm: float = 10.0,
    spacing: Sequence[float] = DEFAULT_SPACING,
    name: str = "helix",
    **kwargs: Any,
) -> Phantom:
    """Helical tube around the S axis (tortuous-iliac-like); checks lengths and frames.

    ``turns`` full turns lie inside the volume (centred at z = 0); the centerline
    continues half a turn beyond each end.
    """
    Rh = float(helix_radius_mm)
    r = 0.5 * diameter_mm
    c = pitch_mm / (2.0 * math.pi)
    t_half = math.pi * (turns + 1.0)  # parameter range: [-t_half, t_half]
    length_per_radian = math.hypot(Rh, c)
    n = max(256, int(math.ceil(2 * t_half * length_per_radian / 0.05)))
    t = np.linspace(-t_half, t_half, n + 1)
    centerline = np.stack([Rh * np.cos(t), Rh * np.sin(t), c * t], axis=1)
    pad = r + ROI_MARGIN_MM
    z_half = 0.5 * turns * pitch_mm
    expected = {
        "lumen": ellipse_reference_metrics(r, r),
        "length_per_turn_mm": 2.0 * math.pi * length_per_radian,
        "curvature_per_mm": Rh / (Rh**2 + c**2),
        "torsion_per_mm": c / (Rh**2 + c**2),
        "parameter_range_rad": [-t_half, t_half],
    }
    params = {"helix_radius_mm": Rh, "pitch_mm": pitch_mm, "turns": turns, "diameter_mm": diameter_mm}
    return build_tube_phantom(
        name=name,
        description="Helical tube around the S axis",
        centerline=centerline,
        layer_specs=[("lumen", LUMEN_HU, _constant(r), _constant(r))],
        roi_min=(-Rh - pad, -Rh - pad, -z_half),
        roi_max=(Rh + pad, Rh + pad, z_half),
        spacing=spacing,
        reference_normal=(0.0, 0.0, 1.0),
        expected=expected,
        parameters=params,
        **kwargs,
    )


#: Phantoms written by ``python -m aortaplan_core.phantoms`` (name -> builder).
STANDARD_PHANTOMS: Dict[str, Callable[[], Phantom]] = {
    "straight_tube": lambda: straight_tube(),
    "straight_tube_realistic": lambda: straight_tube(
        name="straight_tube_realistic", blur_sigma_mm=0.5, noise_sigma_hu=20.0, seed=1
    ),
    "straight_tube_oblique_aniso": lambda: straight_tube(
        name="straight_tube_oblique_aniso", direction=(0.3, -0.4, 1.0), spacing=(0.7, 0.7, 1.25)
    ),
    "elliptical_tube": lambda: elliptical_tube(),
    "torus_arc": lambda: torus_arc(),
    "helix": lambda: helix(),
}
