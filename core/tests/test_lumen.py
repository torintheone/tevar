import numpy as np
import pytest

from aortaplan_core.lumen import Cancelled, LumenParameters, estimate_hu_range, segment_lumen, snap_seed
from aortaplan_core.phantoms.branching import suggested_seeds
from aortaplan_core.volume import ImageGeometry, Volume


def true_lumen_ml():
    """Lumen volume of the aorto-iliac phantom from its noise-free, unblurred version."""
    from aortaplan_core.phantoms.branching import aortoiliac

    clean = aortoiliac(noise_sigma_hu=0.0, blur_sigma_mm=0.0)
    fraction = np.clip((clean.volume.data.astype(float) - 40.0) / 310.0, 0.0, 1.0)
    return fraction.sum() * clean.volume.geometry.voxel_volume / 1000.0


def bone_voxels(phantom, mask):
    g = phantom.volume.geometry
    v = [o for o in phantom.truth.objects if o["kind"] == "vertebra"][0]
    pts = g.voxel_centers_ras()
    r = np.hypot(pts[..., 0] - v["axis_point"][0], pts[..., 1] - v["axis_point"][1])
    inside = (r < v["radius_mm"] - 1.5) & (pts[..., 2] > v["z_range_mm"][0] + 5) & (pts[..., 2] < v["z_range_mm"][1] - 5)
    return int((mask & inside).sum())


def test_hu_range(aortoiliac_phantom):
    start, ends = suggested_seeds(aortoiliac_phantom.truth)
    lower, upper, level = estimate_hu_range(aortoiliac_phantom.volume, [start, *ends])
    assert 330 < level < 400
    assert lower == pytest.approx(40 + 0.5 * (level - 40))
    assert upper > level + 150


def test_snap_moves_a_wall_click_to_the_centre(aortoiliac_phantom):
    vol = aortoiliac_phantom.volume
    snapped = snap_seed(vol, np.array([8.0, 0.0, 60.0]), (195.0, 560.0), radius_mm=8.0)
    assert np.hypot(snapped[0], snapped[1]) < 1.5  # aortic axis is x = y = 0
    assert abs(snapped[2] - 60.0) < 2.0  # not moved along the vessel
    assert snap_seed(vol, np.array([50.0, 25.0, 60.0]), (195.0, 560.0), radius_mm=5.0) is None


def test_segmentation_of_bifurcation(aortoiliac_phantom):
    start, ends = suggested_seeds(aortoiliac_phantom.truth)
    result = segment_lumen(aortoiliac_phantom.volume, start + [6, 0, 0], [e + [2, 0, 0] for e in ends])
    assert result.connected == [True, True] and result.warnings == []
    assert result.volume_ml == pytest.approx(true_lumen_ml(), rel=0.05)
    full = result.full_mask(aortoiliac_phantom.volume.geometry)
    assert full.shape == aortoiliac_phantom.volume.data.shape and full.sum() == result.voxel_count


def test_leak_into_vertebra_is_prevented(aortoiliac_bone_phantom):
    start, ends = suggested_seeds(aortoiliac_bone_phantom.truth)
    geometry = aortoiliac_bone_phantom.volume.geometry
    protected = segment_lumen(aortoiliac_bone_phantom.volume, start, ends)
    assert bone_voxels(aortoiliac_bone_phantom, protected.full_mask(geometry)) == 0
    assert protected.connected == [True, True]
    # Without the opening the region growing leaks: the phantom is a meaningful test.
    unprotected = segment_lumen(
        aortoiliac_bone_phantom.volume, start, ends, params=LumenParameters(opening_iterations=0)
    )
    assert bone_voxels(aortoiliac_bone_phantom, unprotected.full_mask(geometry)) > 10000


def two_tubes_with_bridge():
    g = ImageGeometry((60, 40, 60), (1.0, 1.0, 1.0), (-30.0, -20.0, 0.0), np.eye(3))
    pts = g.voxel_centers_ras()
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    data = np.full(g.array_shape, 40.0)
    data[np.hypot(x + 10, y) <= 5] = 350.0  # tube A
    data[np.hypot(x - 10, y) <= 5] = 350.0  # tube B
    data[(np.abs(x) <= 6) & (np.hypot(y, z - 30) <= 1.2)] = 350.0  # thin bridge
    return Volume(data=data.astype(np.int16), geometry=g)


def test_stop_point_cuts_a_bridge():
    vol = two_tubes_with_bridge()
    params = LumenParameters(opening_iterations=0)
    leaked = segment_lumen(vol, [-10, 0, 10], [[-10, 0, 50]], params=params)
    assert leaked.voxel_count > 2 * np.pi * 25 * 55  # both tubes
    cut = segment_lumen(vol, [-10, 0, 10], [[-10, 0, 50]], stops=[[0, 0, 30]], params=params)
    assert cut.voxel_count < 1.1 * np.pi * 25 * 60  # tube A only
    disconnected = segment_lumen(vol, [-10, 0, 10], [[10, 0, 50]], stops=[[0, 0, 30]], params=params)
    assert disconnected.connected == [False] and "end_not_connected" in disconnected.warnings


def test_exclusion_mask():
    vol = two_tubes_with_bridge()
    exclusion = np.zeros(vol.data.shape, dtype=bool)
    exclusion[:, :, 28:33] = True  # x = -2 .. 2: removes the bridge
    result = segment_lumen(vol, [-10, 0, 10], [[-10, 0, 50]], exclusion_mask=exclusion, params=LumenParameters(opening_iterations=0))
    assert result.voxel_count < 1.1 * np.pi * 25 * 60


def test_roi_grows_for_curved_vessels(realistic_torus):
    pts = realistic_torus.truth.centerline_points
    start, end = pts[70], pts[-70]  # both legs, below the arch
    result = segment_lumen(realistic_torus.volume, start, [end])
    assert result.connected == [True]


def test_cancel():
    vol = two_tubes_with_bridge()
    with pytest.raises(Cancelled):
        segment_lumen(vol, [-10, 0, 10], [[-10, 0, 50]], progress=lambda f, step: f < 0.3)


def test_needs_two_seeds():
    with pytest.raises(ValueError):
        segment_lumen(two_tubes_with_bridge(), [-10, 0, 10], [])
