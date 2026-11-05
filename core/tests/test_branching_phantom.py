import json

import numpy as np
import pytest

from aortaplan_core.phantoms import load_phantom, save_phantom
from aortaplan_core.phantoms.branching import CANCELLOUS_BONE_HU, suggested_seeds
from aortaplan_core.phantoms.model import PhantomTruth


def test_aortoiliac_truth(aortoiliac_phantom):
    truth = aortoiliac_phantom.truth
    assert [b["name"] for b in truth.branches] == ["aorta", "right_iliac", "left_iliac"]
    right, left = truth.path_points("right"), truth.path_points("left")
    assert np.all(np.linalg.norm(np.diff(right, axis=0), axis=1) < 0.6)  # continuous
    assert right[-1][0] > 30 and left[-1][0] < -30  # patient right is +x in RAS
    np.testing.assert_allclose(truth.junctions[0]["point"], [0, 0, 0])
    start, ends = suggested_seeds(truth)
    values = aortoiliac_phantom.volume.sample_linear(np.vstack([start, *ends]))
    assert np.all(values > 280)  # seeds are in the lumen


def test_bone_phantom(aortoiliac_bone_phantom):
    truth = aortoiliac_bone_phantom.truth
    vertebra = [o for o in truth.objects if o["kind"] == "vertebra"][0]
    centre = np.array(vertebra["axis_point"]) + [0, 0, 40]
    assert aortoiliac_bone_phantom.volume.sample_linear(centre) == pytest.approx(CANCELLOUS_BONE_HU, abs=60)
    assert len([o for o in truth.objects if o["kind"] == "calcification"]) == 3


def test_truth_v2_round_trip_and_v1_compatibility(tmp_path, aortoiliac_phantom):
    save_phantom(aortoiliac_phantom, tmp_path)
    back = load_phantom(tmp_path, aortoiliac_phantom.truth.name)
    np.testing.assert_allclose(back.truth.path_points("left"), aortoiliac_phantom.truth.path_points("left"), atol=1e-5)
    d = json.loads((tmp_path / f"{aortoiliac_phantom.truth.name}.truth.json").read_text())
    d["schema_version"] = 1
    for key in ("branches", "paths", "junctions", "objects"):
        d.pop(key)
    old = PhantomTruth.from_dict(d)
    assert old.branches == [] and old.paths == {}
