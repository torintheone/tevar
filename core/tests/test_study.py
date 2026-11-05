import math

import numpy as np
import pytest

from aortaplan_core.study import (
    SeriesInfo,
    StudyThresholds,
    assess_series,
    assess_volume_geometry,
    parse_dicom_numbers,
    parse_dicom_strings,
    sort_candidates,
)
from aortaplan_core.volume import CT_LIKE_DIRECTIONS, ImageGeometry

AXIAL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def series(n=300, spacing=0.625, thickness=0.625, pixel=(0.7, 0.7), tilt=None, shift_per_slice=(0.0, 0.0), **kw):
    positions = [(-200.0 + i * shift_per_slice[0], -180.0 + i * shift_per_slice[1], 100.0 - i * spacing) for i in range(n)]
    return SeriesInfo(
        series_uid=kw.pop("uid", "1.2.3"),
        modality=kw.pop("modality", "CT"),
        description="CTA",
        number_of_images=n,
        image_types=kw.pop("image_types", ["ORIGINAL", "PRIMARY", "AXIAL"]),
        slice_thickness_mm=thickness,
        pixel_spacing_mm=pixel,
        gantry_tilt_deg=tilt,
        image_orientation=AXIAL,
        image_positions=positions,
        **kw,
    )


def test_parse_dicom_values():
    assert parse_dicom_numbers("0.7\\0.7") == [0.7, 0.7]
    assert parse_dicom_numbers(" 1.25 ") == [1.25]
    assert parse_dicom_numbers("") == [] and parse_dicom_numbers(None) == [] and parse_dicom_numbers("abc") == []
    assert parse_dicom_strings("ORIGINAL\\primary\\AXIAL") == ["ORIGINAL", "PRIMARY", "AXIAL"]


def test_good_cta_series():
    a = assess_series(series())
    assert a.candidate and a.is_ct and not a.is_localizer
    assert a.warnings == []
    assert a.slice_spacing_mm == pytest.approx(0.625)
    assert a.voxel_spacing_mm == pytest.approx((0.7, 0.7, 0.625))


def test_thick_slices():
    a = assess_series(series(spacing=2.0, thickness=2.0))
    assert "thick_slices" in a.warning_codes
    assert a.candidate  # still listed, with a warning


def test_thickness_falls_back_to_spacing():
    a = assess_series(series(spacing=3.0, thickness=None))
    assert "thick_slices" in a.warning_codes


def test_anisotropic_voxels():
    a = assess_series(series(spacing=1.5, thickness=1.5, pixel=(0.5, 0.5)))
    assert a.warning_codes == ["anisotropic"]
    assert a.warnings[0].params["ratio"] == pytest.approx(3.0)


def test_gantry_tilt_from_tag_and_geometry():
    assert "gantry_tilt" in assess_series(series(tilt=12.0)).warning_codes
    shift = 0.625 * math.tan(math.radians(15.0))
    a = assess_series(series(shift_per_slice=(0.0, shift)))
    codes = a.warning_codes
    assert "gantry_tilt" in codes
    angle = [w for w in a.warnings if w.code == "gantry_tilt"][0].params["angle_deg"]
    assert angle == pytest.approx(15.0, abs=0.01)


def test_irregular_spacing_and_unsorted_positions():
    info = series(n=100)
    del info.image_positions[50]  # missing slice
    info.number_of_images = 99
    info.image_positions = list(reversed(info.image_positions))
    a = assess_series(info)
    assert "irregular_spacing" in a.warning_codes
    assert a.slice_spacing_mm == pytest.approx(0.625)


def test_non_candidates():
    assert not assess_series(series(modality="MR")).candidate
    loc = assess_series(series(image_types=["ORIGINAL", "PRIMARY", "LOCALIZER"]))
    assert loc.is_localizer and not loc.candidate
    few = assess_series(series(n=3))
    assert not few.candidate and "few_slices" in few.warning_codes


def test_missing_geometry():
    info = series()
    info.image_positions = None
    assert "no_geometry" in assess_series(info).warning_codes


def test_thresholds_are_configurable():
    strict = StudyThresholds(max_slice_thickness_mm=0.5)
    assert "thick_slices" in assess_series(series(), strict).warning_codes
    assert StudyThresholds.from_dict(strict.to_dict()) == strict


def test_series_info_round_trip():
    info = series(n=40)
    assert SeriesInfo.from_dict(info.to_dict()) == info


def test_volume_geometry_warnings():
    good = ImageGeometry((512, 512, 400), (0.7, 0.7, 0.8), (0, 0, 0), CT_LIKE_DIRECTIONS)
    assert assess_volume_geometry(good) == []
    thick = ImageGeometry((512, 512, 100), (0.7, 0.7, 2.5), (0, 0, 0), CT_LIKE_DIRECTIONS)
    assert [w.code for w in assess_volume_geometry(thick)] == ["thick_slices", "anisotropic"]
    tiny = ImageGeometry((64, 64, 5), (0.7, 0.7, 0.7), (0, 0, 0), np.eye(3))
    assert [w.code for w in assess_volume_geometry(tiny)] == ["few_slices"]


def test_sort_candidates():
    good = series(uid="good")
    thick = series(uid="thick", spacing=2.0, thickness=2.0, n=150)
    items = [(thick, assess_series(thick)), (good, assess_series(good))]
    assert [info.series_uid for info, _ in sort_candidates(items)] == ["good", "thick"]
