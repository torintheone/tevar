"""Ostia on phantoms. SPEC stage 5 acceptance: clock error <= 5 deg, distances <= 1 mm,
all branches >= 3 mm found. Measured on centerlines extracted by the core (not the truth)."""

import numpy as np
import pytest

from aortaplan_core.centerline.extraction import extract_centerlines
from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.crosssection import MeasurementStore
from aortaplan_core.lumen import segment_lumen
from aortaplan_core.ostia import (
    MODE_ANTERIOR,
    MODE_GREATER_CURVATURE,
    AorticTube,
    Ostium,
    OstiumParameters,
    describe,
    find_ostia,
    ishimaru_zones,
    manual_ostium,
    ostium_name,
    propose_labels,
    zone_diameters,
)
from aortaplan_core.ostia.detection import ostia_from_paths
from aortaplan_core.phantoms import aortic_arch, visceral_aorta

CLOCK_LIMIT_DEG = 5.0
DISTANCE_LIMIT_MM = 1.0
EXPECTED_VISCERAL = ["celiac", "sma", "right_renal", "left_renal", "accessory_renal_right", "ima"]


def _extracted(phantom, margin_points=8):
    pts = phantom.truth.centerline_points
    g = phantom.volume.geometry
    ijk = g.ras_to_ijk(pts)
    inside = np.flatnonzero(np.all((ijk >= 3) & (ijk <= np.array(g.dimensions) - 4), axis=1))
    start, end = pts[inside[0] + margin_points], pts[inside[-1] - margin_points]
    network = extract_centerlines(phantom.volume, segment_lumen(phantom.volume, start, [end]))
    return CurvePath(network.paths[0].points)


@pytest.fixture(scope="module")
def visceral():
    phantom = visceral_aorta()
    path = _extracted(phantom)
    ostia, tube, stumps = find_ostia(phantom.volume, path)
    propose_labels(ostia, path, aortic_diameter_mm=22.0)
    return phantom, path, ostia, tube, stumps


@pytest.fixture(scope="module")
def arch():
    phantom = aortic_arch()
    path = _extracted(phantom)
    ostia, tube, stumps = find_ostia(phantom.volume, path)
    propose_labels(ostia, path, aortic_diameter_mm=28.0)
    return phantom, path, ostia, tube


def _match(truth_ostia, ostia):
    pairs = []
    for t in truth_ostia:
        c = np.asarray(t["center"])
        best = min(ostia, key=lambda o: np.linalg.norm(o.center - c))
        assert np.linalg.norm(best.center - c) < 2.0, t["key"]
        pairs.append((t, best))
    return pairs


def _check_acceptance(phantom, path, ostia, mode):
    truth = phantom.truth.expected["ostia"]
    pairs = _match(truth, ostia)
    ref_truth, ref = pairs[0]
    ref_s = describe(ref, path, mode).arc_length_mm
    worst = {"clock": 0.0, "distance": 0.0, "craniocaudal": 0.0}
    for t, o in pairs:
        pos = describe(o, path, mode)
        clock_error = (pos.clock_deg - t["clock_deg"][mode] + 180.0) % 360.0 - 180.0
        distance_error = (pos.arc_length_mm - ref_s) - (t["arc_length_mm"] - ref_truth["arc_length_mm"])
        cc_error = pos.takeoff.craniocaudal_deg - t["takeoff"][mode]["craniocaudal_deg"]
        assert abs(clock_error) <= CLOCK_LIMIT_DEG, (t["key"], clock_error)
        assert abs(distance_error) <= DISTANCE_LIMIT_MM, (t["key"], distance_error)
        assert abs(cc_error) <= 5.0, (t["key"], cc_error)
        worst["clock"] = max(worst["clock"], abs(clock_error))
        worst["distance"] = max(worst["distance"], abs(distance_error))
        worst["craniocaudal"] = max(worst["craniocaudal"], abs(cc_error))
    print(phantom.truth.name, "worst errors:", worst)


def test_visceral_ostia_acceptance(visceral):
    phantom, path, ostia, tube, stumps = visceral
    assert len(ostia) == 6  # every branch >= 3 mm, nothing else
    _check_acceptance(phantom, path, ostia, MODE_ANTERIOR)
    for t, o in _match(phantom.truth.expected["ostia"], ostia):
        tolerance = 0.4 if t["diameter_mm"] >= 5 else 0.5
        assert o.branch_diameter_mm == pytest.approx(t["diameter_mm"], abs=tolerance), t["key"]
        assert o.source == "auto" and not o.confirmed
    # Axial deviation of the renal arteries (-20 and +15 deg).
    pairs = {t["key"]: o for t, o in _match(phantom.truth.expected["ostia"], ostia)}
    assert describe(pairs["right_renal"], path).takeoff.axial_deg == pytest.approx(-20.0, abs=3.0)
    assert describe(pairs["left_renal"], path).takeoff.axial_deg == pytest.approx(15.0, abs=3.0)


def test_visceral_labels(visceral):
    _, path, ostia, _, _ = visceral
    assert [o.key for o in ostia] == EXPECTED_VISCERAL
    assert ostia[0].name == ostium_name("celiac")
    # Confirmed names are kept; unconfirmed ones are proposed again.
    ostia[1].confirmed, ostia[1].key, ostia[1].name = True, "branch", "Моя ветвь"
    ostia[2].key = None
    propose_labels(ostia, path, 22.0)
    assert ostia[1].name == "Моя ветвь" and ostia[2].key == "right_renal"
    ostia[1].confirmed, ostia[1].key, ostia[1].name = False, "sma", ostium_name("sma")


def test_vertebra_is_not_a_branch():
    phantom = visceral_aorta(with_bone=True)
    path = CurvePath(phantom.truth.centerline_points)
    ostia, _, stumps = find_ostia(phantom.volume, path)
    assert len(ostia) == 6
    assert any(st.rejected == "too_long" for st in stumps)  # the vertebra against the wall
    for o in ostia:  # nothing posterior
        assert describe(o, path).clock_deg < 135.0 or describe(o, path).clock_deg > 225.0


def test_arch_ostia_acceptance_and_labels(arch):
    phantom, path, ostia, _ = arch
    assert [o.key for o in ostia] == ["bca", "lcca", "lsa"]
    _check_acceptance(phantom, path, ostia, MODE_GREATER_CURVATURE)
    _check_acceptance(phantom, path, ostia, MODE_ANTERIOR)
    # Greater-curvature reference: arch vessels near 12:00, BCT towards the patient's right.
    clocks = [describe(o, path, MODE_GREATER_CURVATURE).clock for o in ostia]
    assert clocks[0].startswith("11:") and clocks[2].startswith("12:")
    # No loops in the centerline at the wide arch ostia: its length matches the truth.
    truth = CurvePath(phantom.truth.centerline_points)
    s0 = truth.nearest_arc_length(path.points[0])[0]
    s1 = truth.nearest_arc_length(path.points[-1])[0]
    assert path.length == pytest.approx(s1 - s0, abs=1.0)


def test_ishimaru_zones(arch):
    phantom, path, ostia, _ = arch
    zones = ishimaru_zones(path, ostia)
    assert [z.name for z in zones] == ["0", "1", "2", "3", "4"]
    by_key = {o.key: o for o in ostia}
    lcca = describe(by_key["lcca"], path)
    # Zone 1 ends at the distal edge of the LCCA ostium.
    assert zones[1].end_mm > lcca.arc_length_mm
    assert zones[1].end_mm - lcca.arc_length_mm == pytest.approx(4.0, abs=1.0)  # LCCA radius 4 mm
    assert zones[3].length_mm == pytest.approx(20.0)
    assert zones[4].approximate  # no celiac trunk in this phantom
    zone_diameters(phantom.volume, path, zones[:2], with_outer=False)
    assert zones[1].diameters["lumen_min_mm"] == pytest.approx(28.0, abs=0.5)
    assert 28.0 <= zones[1].diameters["lumen_max_mm"] <= 31.0  # sections through the ostia bulge
    assert ishimaru_zones(path, ostia[:2]) == []


def test_manual_ostium_matches_the_truth(visceral):
    phantom, path, _, tube, _ = visceral
    truth = {t["key"]: t for t in phantom.truth.expected["ostia"]}["left_renal"]
    centre = np.asarray(truth["center"])
    click = centre + 0.8 * np.asarray([0.0, 0.0, 1.0])  # a slightly imprecise click on the wall
    branch_point = centre + 15.0 * np.asarray(truth["direction"])
    o = manual_ostium(phantom.volume, tube, click, branch_point)
    assert o.source == "manual" and o.confirmed
    pos = describe(o, path)
    assert pos.clock_deg == pytest.approx(truth["clock_deg"]["anterior"], abs=CLOCK_LIMIT_DEG)
    assert o.branch_diameter_mm == pytest.approx(6.5, abs=0.5)
    assert pos.takeoff.craniocaudal_deg == pytest.approx(truth["takeoff"]["anterior"]["craniocaudal_deg"], abs=5.0)
    # Without a branch point: radial direction, still on the wall at the right clock.
    o2 = manual_ostium(phantom.volume, tube, click)
    assert describe(o2, path).clock_deg == pytest.approx(truth["clock_deg"]["anterior"], abs=CLOCK_LIMIT_DEG)
    assert abs(tube.signed_distance(o2.center)[0]) < 0.1


def test_branch_from_a_centerline_path(visceral):
    phantom, path, _, tube, _ = visceral
    renal = next(b for b in phantom.truth.branches if b["name"] == "right_renal")
    aorta = np.asarray(phantom.truth.centerline_points)
    branch = np.asarray(renal["points"])
    # A path that follows the aorta and turns into the renal artery (as a stage 2 branch path).
    s_origin = int(np.argmin(np.linalg.norm(aorta - branch[0], axis=1)))
    other = CurvePath(np.vstack([aorta[: s_origin + 1], branch[1:]]))
    found = ostia_from_paths(phantom.volume, tube, [other], OstiumParameters())
    assert len(found) == 1 and found[0].source == "path"
    truth = {t["key"]: t for t in phantom.truth.expected["ostia"]}["right_renal"]
    assert np.linalg.norm(found[0].center - np.asarray(truth["center"])) < 1.0


def test_tube_surface(visceral):
    phantom, path, _, tube, _ = visceral
    s = 30.0
    centre = path.point_at(s)
    frame_x = np.cross(path.tangent_at(s), [0.0, 1.0, 0.0])
    frame_x /= np.linalg.norm(frame_x)
    d = tube.signed_distance(np.array([centre, centre + 11.0 * frame_x, centre + 15.0 * frame_x]))
    assert d[0] == pytest.approx(-11.0, abs=0.3) and abs(d[1]) < 0.3 and d[2] == pytest.approx(4.0, abs=0.3)
    assert isinstance(tube, AorticTube) and tube.local_radius(s) == pytest.approx(11.0, abs=0.2)


def test_ostium_round_trip(visceral):
    _, _, ostia, _, _ = visceral
    store = MeasurementStore(ostia=list(ostia), ostia_settings={"graft_diameter_mm": 28.0, "clock_mode": "anterior"})
    back = MeasurementStore.from_json(store.to_json())
    assert len(back.ostia) == len(ostia)
    for a, b in zip(ostia, back.ostia):
        assert isinstance(b, Ostium)
        assert b.key == a.key and b.name == a.name and b.source == a.source
        np.testing.assert_allclose(b.center, a.center, atol=1e-3)
        assert b.branch_diameter_mm == pytest.approx(a.branch_diameter_mm, abs=1e-3)
    assert back.ostia_settings["graft_diameter_mm"] == 28.0
    # A store written before stage 5 (schema 1) still loads.
    old = store.to_dict()
    old["schema_version"] = 1
    del old["ostia"], old["ostia_settings"]
    assert MeasurementStore.from_dict(old).ostia == []
