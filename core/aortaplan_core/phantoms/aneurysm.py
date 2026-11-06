"""Fusiform abdominal aortic aneurysm with an eccentric "thrombus" (SPEC section 9).

The aorta runs straight from the top of the volume (proximal) towards the feet. Layers:

- background: periaortic fat (-100 HU);
- outer layer: aortic wall + thrombus (40 HU), an ellipse whose centre is shifted
  anteriorly in the sac, so the thrombus is thick anteriorly and thin posteriorly;
- lumen: contrast (350 HU), a circle centred on the centerline.

Profile along the centerline (arc length ``s`` from the top): neck, sac with a smooth
``sin^2`` bulge, distal aorta. The wall thickness outside the sac is ``wall_thickness_mm``.
Optionally a calcified plaque (900 HU) lines the outer wall at the patient's left in the
middle of the sac; the outer boundary there is the outer edge of the calcium. Optionally an
unenhanced inferior vena cava (40 HU, like the thrombus) touches the aorta on the patient's
right along its whole length without a fat plane: the outer wall is not visible there.

Ground truth per centerline sample: the layers' semi-axes and offsets (truth JSON), and
``expected``: positions of the sac, the maximum outer diameters and their position. All
positions are arc lengths along the truth centerline, which starts 15 mm above the volume.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from ..volume import ImageGeometry
from .library import LUMEN_HU, build_tube_phantom
from .model import Phantom, ellipse_reference_metrics
from .rasterize import paint, solid_fraction

FAT_HU = -100.0
THROMBUS_HU = 40.0
CALCIUM_HU = 900.0


def _bump(s: np.ndarray, s0: float, s1: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    t = np.clip((s - s0) / (s1 - s0), 0.0, 1.0)
    return np.sin(math.pi * t) ** 2


def fusiform_aneurysm(
    neck_lumen_diameter_mm: float = 22.0,
    wall_thickness_mm: float = 1.5,
    sac_outer_diameters_mm: Sequence[float] = (60.0, 54.0),
    sac_lumen_diameter_mm: float = 30.0,
    eccentricity: float = 0.6,
    neck_length_mm: float = 40.0,
    sac_length_mm: float = 90.0,
    distal_length_mm: float = 30.0,
    with_calcification: bool = False,
    with_ivc: bool = False,
    spacing: Sequence[float] = (0.7, 0.7, 0.7),
    blur_sigma_mm: float = 0.5,
    noise_sigma_hu: float = 15.0,
    seed: Optional[int] = 7,
    supersampling: int = 3,
    name: str = "aneurysm",
    **kwargs: Any,
) -> Phantom:
    """Straight infrarenal-like aorta with a fusiform aneurysm and eccentric thrombus.

    ``sac_outer_diameters_mm`` = (along the frame normal = patient left-right,
    along the binormal = anterior-posterior). ``eccentricity`` in [0, 1): the outer
    ellipse centre moves anteriorly by this fraction of the room left between the lumen
    and the outer wall.
    """
    r_lumen0 = 0.5 * neck_lumen_diameter_mm
    r_outer0 = r_lumen0 + wall_thickness_mm
    r_lumen_max = 0.5 * sac_lumen_diameter_mm
    a_max, b_max = 0.5 * float(sac_outer_diameters_mm[0]), 0.5 * float(sac_outer_diameters_mm[1])
    if min(a_max, b_max) < r_lumen_max + wall_thickness_mm:
        raise ValueError("the sac must contain the lumen and the wall")
    # The centerline starts ``extension`` mm above the volume, so arc lengths along it (the
    # truth) are depths below the top face (z = 0) plus ``extension``.
    extension = 15.0
    s0 = extension + float(neck_length_mm)
    s1 = s0 + float(sac_length_mm)
    total = float(neck_length_mm) + float(sac_length_mm) + float(distal_length_mm)

    def lumen_r(s):
        return r_lumen0 + (r_lumen_max - r_lumen0) * _bump(s, s0, s1)

    def outer_n(s):
        return r_outer0 + (a_max - r_outer0) * _bump(s, s0, s1)

    def outer_b(s):
        return r_outer0 + (b_max - r_outer0) * _bump(s, s0, s1)

    def offset_b(s):
        room = np.maximum(outer_b(s) - lumen_r(s) - wall_thickness_mm, 0.0)
        return float(eccentricity) * room

    # Frame: T = -S (towards the feet), N = patient's left (-x in RAS), B = T x N = anterior.
    tangent = np.array([0.0, 0.0, -1.0])
    normal = np.array([-1.0, 0.0, 0.0])
    binormal = np.cross(tangent, normal)
    margin = max(a_max, b_max) + abs(eccentricity) * b_max + 8.0
    centerline = np.array([[0.0, 0.0, extension], [0.0, 0.0, -(total + extension)]])
    sac_mid = 0.5 * (s0 + s1)
    calcium_half_length = 8.0
    calcium_thickness = 1.5
    ivc_radius = 10.0

    def extras(image: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
        if with_ivc:
            image = ivc(image, geometry)
        if with_calcification:
            image = calcium(image, geometry)
        return image

    def ivc(image: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
        def inside(points: np.ndarray) -> np.ndarray:
            s = extension - points[:, 2]
            x = points @ normal  # towards the patient's left
            y = points @ binormal - offset_b(s)
            a, b = outer_n(s), outer_b(s)
            in_aorta = np.hypot(x / a, y / b) <= 1.0
            # Centre on the patient's right, the tube overlaps the aortic outer wall by 0.5 mm.
            cx = -(a + ivc_radius - 0.5)
            in_ivc = np.hypot(x - cx, y) <= ivc_radius
            return in_ivc & ~in_aorta

        reach = a_max + 2 * ivc_radius + 2.0
        box_min = np.array([0.0, -ivc_radius - b_max, -total])  # patient's right half (+x in RAS)
        box_max = np.array([reach, ivc_radius + b_max, 0.0])
        return paint(image, solid_fraction(geometry, inside, box_min, box_max, supersampling=2), THROMBUS_HU)

    def calcium(image: np.ndarray, geometry: ImageGeometry) -> np.ndarray:
        half_angle = math.radians(25.0)

        def inside(points: np.ndarray) -> np.ndarray:
            s = extension - points[:, 2]
            x = points @ normal
            y = points @ binormal - offset_b(s)
            a, b = outer_n(s), outer_b(s)
            g = np.hypot(x / a, y / b)
            angle = np.arctan2(y, x)
            return (
                (np.abs(s - sac_mid) <= calcium_half_length)
                & (g <= 1.0)
                & (g >= 1.0 - calcium_thickness / b)
                & (np.abs(angle) <= half_angle)
            )

        reach_y = b_max + abs(eccentricity) * b_max + 3.0
        z_mid = extension - sac_mid
        box_min = np.array([-(a_max + 3.0), -reach_y, z_mid - calcium_half_length])  # patient's left half
        box_max = np.array([0.0, reach_y, z_mid + calcium_half_length])
        fraction = solid_fraction(geometry, inside, box_min, box_max, supersampling=3)
        return paint(image, fraction, CALCIUM_HU)

    s_dense = np.linspace(s0, s1, 2001)
    i_max = int(np.argmax(outer_n(s_dense)))
    expected = {
        "neck_lumen": ellipse_reference_metrics(r_lumen0, r_lumen0),
        "neck_outer": ellipse_reference_metrics(r_outer0, r_outer0),
        "volume_start_s_mm": extension,
        "volume_end_s_mm": extension + total,
        "sac_start_s_mm": s0,
        "sac_end_s_mm": s1,
        "max_outer_s_mm": float(s_dense[i_max]),
        "max_outer": ellipse_reference_metrics(a_max, b_max),
        "max_lumen": ellipse_reference_metrics(r_lumen_max, r_lumen_max),
        "calcification": (
            {"s_mm": sac_mid, "half_length_mm": calcium_half_length, "side": "left", "thickness_mm": calcium_thickness}
            if with_calcification
            else None
        ),
        "ivc": {"side": "right", "radius_mm": ivc_radius} if with_ivc else None,
    }
    params = {
        "neck_lumen_diameter_mm": neck_lumen_diameter_mm,
        "wall_thickness_mm": wall_thickness_mm,
        "sac_outer_diameters_mm": [a_max * 2, b_max * 2],
        "sac_lumen_diameter_mm": sac_lumen_diameter_mm,
        "eccentricity": eccentricity,
        "neck_length_mm": neck_length_mm,
        "sac_length_mm": sac_length_mm,
        "distal_length_mm": distal_length_mm,
        "with_calcification": with_calcification,
        "with_ivc": with_ivc,
    }
    return build_tube_phantom(
        name=name,
        description="Fusiform aneurysm with eccentric thrombus in fat (straight, proximal at the top)",
        centerline=centerline,
        layer_specs=[
            ("outer", THROMBUS_HU, outer_n, outer_b, None, offset_b),
            ("lumen", LUMEN_HU, lumen_r, lumen_r),
        ],
        roi_min=(-margin, -margin, -total),
        roi_max=(margin, margin, 0.0),
        spacing=spacing,
        reference_normal=normal,
        background_hu=FAT_HU,
        blur_sigma_mm=blur_sigma_mm,
        noise_sigma_hu=noise_sigma_hu,
        seed=seed,
        supersampling=supersampling,
        expected=expected,
        parameters=params,
        paint_extra=extras,
        **kwargs,
    )


def _register_standard_phantoms() -> None:
    from .library import STANDARD_PHANTOMS

    STANDARD_PHANTOMS["aneurysm"] = lambda: fusiform_aneurysm()
    STANDARD_PHANTOMS["aneurysm_calcified"] = lambda: fusiform_aneurysm(name="aneurysm_calcified", with_calcification=True)


_register_standard_phantoms()
