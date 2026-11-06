"""Automatic contours on phantoms. SPEC stage 4 acceptance: error of the minimum, maximum
and mean diameter <= 0.3 mm at 0.7 mm voxels (circle, ellipse, aneurysm with thrombus)."""

import math

import numpy as np
import pytest

from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.crosssection import (
    LUMEN,
    OUTER,
    ContourParameters,
    SectionContour,
    SectionMeasurement,
    SectionPlane,
    lumen_contour,
    measure_section,
)
from aortaplan_core.crosssection.contours import regularize_radii
from aortaplan_core.phantoms import ellipse_reference_metrics, elliptical_tube, fusiform_aneurysm, straight_tube

LIMIT_MM = 0.3
ACCEPTANCE = ("min_diameter_mm", "max_diameter_mm", "mean_diameter_mm")
EQUIVALENT = ("area_equivalent_diameter_mm", "perimeter_equivalent_diameter_mm")


def errors(contour, reference):
    values = contour.metrics().values()
    return {k: values[k] - reference[k] for k in ACCEPTANCE + EQUIVALENT}


def truth_arc_length(truth):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(truth.centerline_points, axis=0), axis=1))])


@pytest.fixture(scope="module")
def realistic_circle():
    return straight_tube(length_mm=40.0, blur_sigma_mm=0.5, noise_sigma_hu=15.0, seed=11, supersampling=3)


@pytest.fixture(scope="module")
def realistic_ellipse():
    return elliptical_tube(length_mm=40.0, blur_sigma_mm=0.5, noise_sigma_hu=15.0, seed=12, supersampling=3)


@pytest.fixture(scope="module")
def realistic_oblique_circle():
    return straight_tube(
        name="oblique_circle", length_mm=40.0, direction=(0.3, -0.4, 1.0), blur_sigma_mm=0.5,
        noise_sigma_hu=15.0, seed=13, supersampling=3,
    )




@pytest.fixture(scope="module")
def aneurysm_calcified():
    return fusiform_aneurysm(name="aneurysm_calcified", with_calcification=True)


@pytest.fixture(scope="module")
def aneurysm_ivc():
    return fusiform_aneurysm(name="aneurysm_ivc", with_ivc=True)


@pytest.mark.parametrize("fixture", ["realistic_circle", "realistic_ellipse", "realistic_oblique_circle"])
def test_lumen_diameters_on_tubes(request, fixture):
    phantom = request.getfixturevalue(fixture)
    layer = phantom.truth.layers[-1]
    reference = ellipse_reference_metrics(layer.semi_axis_n[0], layer.semi_axis_b[0])
    path = CurvePath(phantom.truth.centerline_points)
    for s in np.linspace(0.35, 0.65, 7) * path.length:
        contour = lumen_contour(phantom.volume, SectionPlane.at(path, s))
        assert contour is not None and contour.warnings == []
        err = errors(contour, reference)
        assert max(abs(v) for v in err.values()) <= LIMIT_MM, (s, err)


@pytest.mark.parametrize("fixture", ["aneurysm", "aneurysm_calcified", "aneurysm_ivc"])
def test_aneurysm_lumen_and_outer_wall(request, fixture):
    phantom = request.getfixturevalue(fixture)
    truth = phantom.truth
    path = CurvePath(truth.centerline_points)
    s_truth = truth_arc_length(truth)
    e = truth.expected
    worst = {LUMEN: 0.0, OUTER: 0.0}
    for s in np.arange(e["volume_start_s_mm"] + 5.0, e["volume_end_s_mm"] - 4.0, 5.0):
        i = int(np.argmin(np.abs(s_truth - s)))
        m = measure_section(phantom.volume, SectionPlane.at(path, s))
        assert m.lumen is not None and m.outer is not None, s
        for kind, contour in ((LUMEN, m.lumen), (OUTER, m.outer)):
            layer = truth.layer(kind)
            err = errors(contour, ellipse_reference_metrics(layer.semi_axis_n[i], layer.semi_axis_b[i]))
            worst[kind] = max(worst[kind], max(abs(err[k]) for k in ACCEPTANCE + EQUIVALENT))
            assert max(abs(err[k]) for k in ACCEPTANCE) <= LIMIT_MM, (kind, s, err)
    print(fixture, "worst diameter error (mm):", worst)


def test_eccentric_thrombus_outer_contour_is_offset(aneurysm):
    truth = aneurysm.truth
    path = CurvePath(truth.centerline_points)
    s = truth.expected["max_outer_s_mm"]
    m = measure_section(aneurysm.volume, SectionPlane.at(path, s))
    outer = m.outer.metrics()
    lumen = m.lumen.metrics()
    # Thrombus is thick anteriorly: the outer centroid lies ~6.3 mm anterior (screen up = +v).
    offset = outer.centroid - lumen.centroid
    assert offset[1] == pytest.approx(6.3, abs=0.3)
    assert abs(offset[0]) < 0.3
    assert outer.max_diameter_mm == pytest.approx(60.0, abs=LIMIT_MM)
    assert outer.min_diameter_mm == pytest.approx(54.0, abs=LIMIT_MM)
    # The major axis runs left-right (screen x).
    d = np.diff(outer.max_feret_segment, axis=0)[0]
    assert abs(d[0]) / np.linalg.norm(d) > 0.99


def test_ivc_rays_are_rejected_and_filled(aneurysm_ivc):
    path = CurvePath(aneurysm_ivc.truth.centerline_points)
    m = measure_section(aneurysm_ivc.volume, SectionPlane.at(path, 35.0))
    # Rays towards the patient's right (screen left, angle 180 deg) cross the IVC.
    angles = np.degrees(m.outer.angles)
    right = np.abs(angles - 180.0) < 10.0
    assert not m.outer.measured[right].any()
    assert m.outer.measured_fraction > 0.75
    assert np.max(m.outer.radii[right]) == pytest.approx(12.5, abs=0.3)


def test_no_fat_means_outer_fallback(realistic_circle):
    # Tube in soft tissue: no fat plane anywhere, the outer wall cannot be found.
    path = CurvePath(realistic_circle.truth.centerline_points)
    m = measure_section(realistic_circle.volume, SectionPlane.at(path, path.length / 2))
    assert "outer_not_found" in m.outer.warnings
    assert m.outer.radii == pytest.approx(m.lumen.radii + ContourParameters().fallback_wall_mm)


def test_centre_outside_lumen_gives_no_contour(realistic_circle):
    path = CurvePath(realistic_circle.truth.centerline_points)
    plane = SectionPlane.at(path, path.length / 2)
    assert lumen_contour(realistic_circle.volume, plane, center=np.array([20.0, 0.0])) is None
    m = measure_section(realistic_circle.volume, plane, center=np.array([20.0, 0.0]))
    assert m.lumen is None and m.outer is None and m.warnings == ["lumen_not_found"]


def test_off_centre_start_is_recentred(realistic_circle):
    path = CurvePath(realistic_circle.truth.centerline_points)
    plane = SectionPlane.at(path, path.length / 2)
    contour = lumen_contour(realistic_circle.volume, plane, center=np.array([5.0, -4.0]))
    assert np.linalg.norm(contour.center) < 0.5
    assert contour.metrics().mean_diameter_mm == pytest.approx(25.0, abs=LIMIT_MM)


def test_branch_excursion_is_filled():
    params = ContourParameters()
    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    radii = np.full(360, 12.0) + 0.05 * np.sin(7 * angles)
    radii[40:75] = 30.0  # rays escaping into a branch
    radii[200:210] = np.nan  # no boundary found
    filled, kept = regularize_radii(angles, radii, params, two_sided=False)
    assert not kept[40:75].any() and not kept[200:210].any()
    assert not kept[36:40].any() and not kept[75:79].any()  # guard rays next to the excursion
    assert filled == pytest.approx(12.0, abs=0.1)


def test_too_few_rays_give_up():
    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    radii = np.full(360, np.nan)
    radii[:50] = 10.0
    filled, _ = regularize_radii(angles, radii, ContourParameters(), two_sided=True)
    assert filled is None


def test_plane_matches_cross_section_view():
    path = CurvePath(np.array([[0.0, 0.0, 0.0], [0.0, 10.0, -50.0], [5.0, 20.0, -100.0]]))
    plane = SectionPlane.at(path, 40.0)
    frame = path.cross_section_frame(40.0)
    assert plane.u == pytest.approx(frame.x_axis)
    assert plane.v == pytest.approx(frame.y_axis)
    assert plane.arc_length_mm == pytest.approx(40.0)
    p2 = np.array([[1.0, 2.0], [-3.0, 0.5]])
    assert plane.to_plane(plane.to_ras(p2)) == pytest.approx(p2)
    assert abs(np.dot(plane.normal, path.tangent_at(40.0))) == pytest.approx(1.0)


def test_contour_editing_and_round_trip(aneurysm):
    path = CurvePath(aneurysm.truth.centerline_points)
    m = measure_section(aneurysm.volume, SectionPlane.at(path, 100.0))
    control = m.outer.resampled(32)
    assert len(control) == 32
    steps = np.linalg.norm(np.diff(np.vstack([control, control[:1]]), axis=0), axis=1)
    assert steps.max() / steps.min() < 1.05  # equally spaced along the contour
    # An edit: pull one point outwards by 3 mm.
    ras = m.outer.plane.to_ras(control)
    ras[0] += 3.0 * m.outer.plane.u * np.sign(control[0, 0] or 1.0)
    edited = m.outer.edited(ras)
    assert edited.source == "manual" and edited.kind == OUTER
    assert edited.metrics().area_mm2 > contour_area(control)
    back = SectionMeasurement.from_dict(SectionMeasurement(m.plane, m.lumen, edited).to_dict())
    assert back.outer.source == "manual"
    assert back.outer.metrics().max_diameter_mm == pytest.approx(edited.metrics().max_diameter_mm, abs=1e-3)
    assert back.lumen.metrics().area_mm2 == pytest.approx(m.lumen.metrics().area_mm2, rel=1e-4)
    assert back.plane.arc_length_mm == pytest.approx(100.0)
    c = SectionContour.from_dict(m.lumen.to_dict())
    assert c.kind == LUMEN and c.points_ras == pytest.approx(m.lumen.points_ras, abs=1e-3)


def contour_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def test_measurement_store_round_trip(aneurysm):
    from aortaplan_core.centerline.markers import Marker
    from aortaplan_core.crosssection import MeasurementStore, SavedMeasurement

    path = CurvePath(aneurysm.truth.centerline_points)
    store = MeasurementStore()
    a = SavedMeasurement("Шейка", "main", measure_section(aneurysm.volume, SectionPlane.at(path, 40.0)))
    b = SavedMeasurement("Мешок", "main", measure_section(aneurysm.volume, SectionPlane.at(path, 100.0)))
    assert store.save(b) == 0 and store.save(a) == 0  # kept in order along the path
    store.markers.append(Marker("Нижняя почечная", path.point_at(30.0), "main", 30.0, "lowest_renal"))
    assert store.find("main", 100.1) == 1 and store.find("main", 101.0) is None and store.find("other", 100.0) is None
    # Saving at the same position replaces the measurement.
    c = SavedMeasurement("Мешок (правка)", "main", measure_section(aneurysm.volume, SectionPlane.at(path, 100.05)))
    assert store.save(c) == 1 and len(store.measurements) == 2
    back = MeasurementStore.from_json(store.to_json())
    assert [m.name for m in back.measurements] == ["Шейка", "Мешок (правка)"]
    assert back.markers[0].key == "lowest_renal"
    assert back.measurements[1].measurement.outer.metrics().max_diameter_mm == pytest.approx(
        c.measurement.outer.metrics().max_diameter_mm, abs=1e-3
    )
    assert MeasurementStore.from_json("").measurements == []
