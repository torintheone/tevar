import numpy as np
import pytest

from aortaplan_core.centerline.extraction import (
    CenterlineNetwork,
    CenterlineParameters,
    divergence_arc_length,
    extract_centerlines,
)
from aortaplan_core.centerline.metrics import centerline_deviation, coverage, distances_to_polyline
from aortaplan_core.lumen import LumenParameters, segment_lumen
from aortaplan_core.phantoms.branching import suggested_seeds

# SPEC stage 2 acceptance: mean <= 0.5 mm, max <= 1 mm (outside bifurcation zones).
MEAN_LIMIT, MAX_LIMIT = 0.5, 1.0


def single_path(phantom):
    g = phantom.volume.geometry
    pts = phantom.truth.centerline_points
    ijk = g.ras_to_ijk(pts)
    inside = np.flatnonzero(np.all((ijk >= 3) & (ijk <= np.array(g.dimensions) - 4), axis=1))
    start, end = pts[inside[0] + 4], pts[inside[-1] - 4]
    lumen = segment_lumen(phantom.volume, start, [end])
    return extract_centerlines(phantom.volume, lumen), pts


@pytest.fixture(scope="module")
def bifurcation_networks(aortoiliac_phantom, aortoiliac_bone_phantom):
    out = {}
    for phantom in (aortoiliac_phantom, aortoiliac_bone_phantom):
        start, ends = suggested_seeds(phantom.truth)
        lumen = segment_lumen(phantom.volume, start + [6, 0, 0], [e + [2, 0, 0] for e in ends])
        out[phantom.truth.name] = (phantom, lumen, extract_centerlines(phantom.volume, lumen))
    return out


@pytest.mark.parametrize("name", ["aortoiliac", "aortoiliac_bone"])
def test_bifurcation_accuracy(bifurcation_networks, name):
    phantom, lumen, network = bifurcation_networks[name]
    truth = phantom.truth
    exclusions = [(j["point"], j["exclusion_radius_mm"]) for j in truth.junctions]
    assert [p.side for p in network.paths] == ["right", "left"]
    for path, truth_path in zip(network.paths, ("right", "left")):
        ref = truth.path_points(truth_path)
        dev = centerline_deviation(path.points, ref, exclusions)
        assert dev["mean_mm"] <= MEAN_LIMIT and dev["max_mm"] <= MAX_LIMIT, dev
        # The whole vessel between the seeds is covered.
        between = (ref[:, 2] < lumen.start[2] - 2) & (ref[:, 2] > lumen.ends[path.end_index][2] + 2)
        assert coverage(ref, path.points, 1.0, exclusions, region=between) > 0.98
    aortic = network.paths[0].radii_mm[: len(network.paths[0].radii_mm) // 4]
    assert np.median(aortic) == pytest.approx(10.0, abs=0.5)


def test_branch_divergence(bifurcation_networks):
    _, lumen, network = bifurcation_networks["aortoiliac"]
    right, left = network.paths
    s = divergence_arc_length(right.points, left.points, tolerance_mm=2.0)
    distance_to_bifurcation = lumen.start[2]  # bifurcation at z = 0, straight aorta above
    assert s == pytest.approx(distance_to_bifurcation, abs=15.0)


@pytest.mark.parametrize("fixture", ["realistic_torus", "realistic_helix", "realistic_oblique"])
def test_single_vessel_accuracy(request, fixture):
    phantom = request.getfixturevalue(fixture)
    network, truth_points = single_path(phantom)
    assert len(network.paths) == 1 and network.warnings == []
    dev = centerline_deviation(network.paths[0].points, truth_points)
    assert dev["mean_mm"] <= MEAN_LIMIT and dev["max_mm"] <= MAX_LIMIT, dev


def test_coarse_graph_is_refined(aortoiliac_phantom):
    start, ends = suggested_seeds(aortoiliac_phantom.truth)
    lumen = segment_lumen(aortoiliac_phantom.volume, start, ends)
    network = extract_centerlines(aortoiliac_phantom.volume, lumen, CenterlineParameters(max_graph_nodes=5000))
    exclusions = [(j["point"], j["exclusion_radius_mm"]) for j in aortoiliac_phantom.truth.junctions]
    dev = centerline_deviation(network.paths[0].points, aortoiliac_phantom.truth.path_points("right"), exclusions)
    assert dev["max_mm"] <= MAX_LIMIT, dev


def test_unreachable_end_is_reported():
    from test_lumen import two_tubes_with_bridge

    vol = two_tubes_with_bridge()
    lumen = segment_lumen(vol, [-10, 0, 10], [[-10, 0, 50], [10, 0, 50]], stops=[[0, 0, 30]], params=LumenParameters(opening_iterations=0))
    network = extract_centerlines(vol, lumen, CenterlineParameters(refine_iterations=1))
    assert len(network.paths) == 1 and network.warnings == ["end_unreachable:1"]


def test_network_round_trip(bifurcation_networks):
    _, _, network = bifurcation_networks["aortoiliac"]
    back = CenterlineNetwork.from_dict(network.to_dict())
    assert back.method == network.method and len(back.paths) == 2
    np.testing.assert_allclose(back.paths[1].points, network.paths[1].points, atol=1e-4)


def test_metrics():
    line = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
    d = distances_to_polyline(np.array([[5, 2, 0], [12, 5, 0], [-3, 0, 4]]), line)
    np.testing.assert_allclose(d, [2.0, 2.0, 5.0])
    dev = centerline_deviation(np.array([[5, 1, 0], [9, 5, 0]]), line, exclusions=[([10, 5, 0], 2.0)])
    assert dev["count"] == 1 and dev["max_mm"] == pytest.approx(1.0)
    assert coverage(line, line[:2], tolerance_mm=0.5) == pytest.approx(2 / 3)


def test_end_in_erased_lumen_is_unreachable(aortoiliac_phantom):
    from aortaplan_core.lumen import lumen_from_mask

    start, ends = suggested_seeds(aortoiliac_phantom.truth)
    g = aortoiliac_phantom.volume.geometry
    lumen = segment_lumen(aortoiliac_phantom.volume, start, ends)
    mask = lumen.full_mask(g)
    mask[:, :, mask.shape[2] // 2 + 8 :] = False  # erase the left iliac (high i = patient left)
    edited = lumen_from_mask(mask, g, lumen.start, lumen.ends, lumen.hu_range, lumen.lumen_hu)
    network = extract_centerlines(aortoiliac_phantom.volume, edited, CenterlineParameters(refine_iterations=1))
    assert [p.side for p in network.paths] == ["right"]
    assert network.warnings == ["end_unreachable:1"]
