"""Study / series checks before planning (SPEC 6.1).

The Slicer shell reads DICOM attributes from its database and fills a
:class:`SeriesInfo`; :func:`assess_series` decides whether the series is a usable CT
volume and which warnings to show. A loaded volume without DICOM metadata (for example
Sample Data) is checked with :func:`assess_volume_geometry`.

Warnings are returned as codes with parameters; the user-facing text comes from
``i18n`` (keys ``study.warning.<code>``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .volume import ImageGeometry


@dataclass
class StudyThresholds:
    max_slice_thickness_mm: float = 1.5
    max_anisotropy_ratio: float = 2.0  # largest / smallest voxel spacing
    max_gantry_tilt_deg: float = 0.1
    max_spacing_irregularity_mm: float = 0.05  # deviation of a slice gap from the median gap
    min_slices: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StudyThresholds:
        return cls(**d)


DEFAULT_THRESHOLDS = StudyThresholds()


@dataclass
class SeriesInfo:
    """DICOM attributes of one series (values already parsed into numbers)."""

    series_uid: str
    modality: str = ""
    description: str = ""
    number_of_images: int = 0
    image_types: List[str] = field(default_factory=list)  # (0008,0008), e.g. ORIGINAL, PRIMARY, AXIAL
    slice_thickness_mm: Optional[float] = None  # (0018,0050)
    pixel_spacing_mm: Optional[Tuple[float, float]] = None  # (0028,0030): row spacing, column spacing
    gantry_tilt_deg: Optional[float] = None  # (0018,1120)
    image_orientation: Optional[Tuple[float, ...]] = None  # (0020,0037): row cosines + column cosines
    image_positions: Optional[List[Tuple[float, float, float]]] = None  # (0020,0032) of every image

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SeriesInfo:
        return cls(**d)


@dataclass
class SeriesWarning:
    code: str  # thick_slices | anisotropic | gantry_tilt | irregular_spacing | few_slices | no_geometry
    params: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "params": dict(self.params)}


@dataclass
class SeriesAssessment:
    is_ct: bool
    is_localizer: bool
    candidate: bool  # shown in the AortaPlan series list
    slice_spacing_mm: Optional[float]
    voxel_spacing_mm: Optional[Tuple[float, float, float]]
    warnings: List[SeriesWarning] = field(default_factory=list)

    @property
    def warning_codes(self) -> List[str]:
        return [w.code for w in self.warnings]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["warnings"] = [w.to_dict() for w in self.warnings]
        return d


def parse_dicom_numbers(value: Optional[str]) -> List[float]:
    """Parse a DICOM DS/IS multi-value string such as ``"0.7\\\\0.7"``; empty or invalid -> []."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return [float(part) for part in text.split("\\") if part.strip() != ""]
    except ValueError:
        return []


def parse_dicom_strings(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [part.strip().upper() for part in str(value).split("\\") if part.strip()]


def _scan_geometry(info: SeriesInfo) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Sorted slice positions along the slice normal, and the mean slice-to-slice vector."""
    if not info.image_orientation or len(info.image_orientation) != 6 or not info.image_positions:
        return None, None
    row = np.asarray(info.image_orientation[:3], dtype=float)
    col = np.asarray(info.image_orientation[3:], dtype=float)
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None, None
    normal /= norm
    positions = np.asarray(info.image_positions, dtype=float)
    distances = positions @ normal
    order = np.argsort(distances)
    sorted_positions = positions[order]
    step_vector = None
    if len(sorted_positions) > 1:
        step_vector = (sorted_positions[-1] - sorted_positions[0]) / (len(sorted_positions) - 1)
    return distances[order], step_vector


def assess_series(info: SeriesInfo, thresholds: StudyThresholds = DEFAULT_THRESHOLDS) -> SeriesAssessment:
    """Classify a series and list the warnings of SPEC 6.1."""
    is_ct = info.modality.strip().upper() == "CT"
    is_localizer = "LOCALIZER" in [t.upper() for t in info.image_types]
    warnings: List[SeriesWarning] = []

    distances, step_vector = _scan_geometry(info)
    slice_spacing = None
    if distances is not None and len(distances) > 1:
        gaps = np.diff(distances)
        slice_spacing = float(np.median(gaps))
        irregularity = float(np.max(np.abs(gaps - slice_spacing)))
        if irregularity > thresholds.max_spacing_irregularity_mm:
            warnings.append(SeriesWarning("irregular_spacing", {"deviation_mm": irregularity}))
    elif info.number_of_images > 1:
        warnings.append(SeriesWarning("no_geometry"))

    if info.number_of_images < thresholds.min_slices:
        warnings.append(SeriesWarning("few_slices", {"count": float(info.number_of_images)}))

    thickness = info.slice_thickness_mm
    if thickness is None:
        thickness = slice_spacing
    if thickness is not None and thickness > thresholds.max_slice_thickness_mm:
        warnings.append(
            SeriesWarning("thick_slices", {"thickness_mm": thickness, "limit_mm": thresholds.max_slice_thickness_mm})
        )

    voxel_spacing = None
    if info.pixel_spacing_mm is not None and slice_spacing is not None:
        row_spacing, col_spacing = info.pixel_spacing_mm
        voxel_spacing = (float(col_spacing), float(row_spacing), abs(float(slice_spacing)))
        ratio = max(voxel_spacing) / max(min(voxel_spacing), 1e-9)
        if ratio > thresholds.max_anisotropy_ratio:
            warnings.append(SeriesWarning("anisotropic", {"ratio": ratio}))

    tilt = abs(info.gantry_tilt_deg) if info.gantry_tilt_deg is not None else 0.0
    if step_vector is not None and info.image_orientation:
        # Geometric tilt: angle between the slice-to-slice direction and the slice normal.
        normal = np.cross(np.asarray(info.image_orientation[:3]), np.asarray(info.image_orientation[3:]))
        normal = normal / np.linalg.norm(normal)
        step_norm = np.linalg.norm(step_vector)
        if step_norm > 1e-9:
            cos_angle = abs(float(np.dot(step_vector / step_norm, normal)))
            tilt = max(tilt, float(np.degrees(np.arccos(min(1.0, cos_angle)))))
    if tilt > thresholds.max_gantry_tilt_deg:
        warnings.append(SeriesWarning("gantry_tilt", {"angle_deg": tilt}))

    candidate = is_ct and not is_localizer and info.number_of_images >= thresholds.min_slices
    return SeriesAssessment(
        is_ct=is_ct,
        is_localizer=is_localizer,
        candidate=candidate,
        slice_spacing_mm=abs(slice_spacing) if slice_spacing is not None else None,
        voxel_spacing_mm=voxel_spacing,
        warnings=warnings,
    )


def assess_volume_geometry(
    geometry: ImageGeometry, thresholds: StudyThresholds = DEFAULT_THRESHOLDS
) -> List[SeriesWarning]:
    """Warnings that can be derived from a loaded volume alone (no DICOM attributes)."""
    warnings: List[SeriesWarning] = []
    spacing = np.asarray(geometry.spacing, dtype=float)
    # The slice axis is the one closest to S (axial acquisitions); fall back to the largest spacing.
    axis = int(np.argmax(np.abs(geometry.directions[2, :])))
    if spacing[axis] > thresholds.max_slice_thickness_mm:
        warnings.append(
            SeriesWarning(
                "thick_slices", {"thickness_mm": float(spacing[axis]), "limit_mm": thresholds.max_slice_thickness_mm}
            )
        )
    ratio = float(spacing.max() / spacing.min())
    if ratio > thresholds.max_anisotropy_ratio:
        warnings.append(SeriesWarning("anisotropic", {"ratio": ratio}))
    if min(geometry.dimensions) < 2 or geometry.dimensions[axis] < thresholds.min_slices:
        warnings.append(SeriesWarning("few_slices", {"count": float(geometry.dimensions[axis])}))
    return warnings


def sort_candidates(items: Sequence[Tuple[SeriesInfo, SeriesAssessment]]) -> List[Tuple[SeriesInfo, SeriesAssessment]]:
    """Usable CT series first: fewer warnings, then more images (thin reconstructions first)."""
    return sorted(items, key=lambda pair: (len(pair[1].warnings), -pair[0].number_of_images))
