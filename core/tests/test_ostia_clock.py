import math

import numpy as np
import pytest

from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.ostia.clock import (
    MODE_ANTERIOR,
    MODE_GREATER_CURVATURE,
    arc_length_on_graft,
    clock_angle,
    clock_frame,
    clock_text,
    direction_angle,
    takeoff_angles,
)

LEFT, RIGHT = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])  # RAS
ANTERIOR, POSTERIOR = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])


def descending():
    return CurvePath(np.array([[0.0, 0.0, 100.0], [0.0, 0.0, -100.0]]))


@pytest.mark.parametrize(
    "angle,text",
    [(0.0, "12:00"), (90.0, "3:00"), (180.0, "6:00"), (270.0, "9:00"), (359.9, "12:00"), (7.5, "12:15"),
     (2.4, "12:05"), (97.5, "3:15"), (-30.0, "11:00"), (345.0, "11:30")],
)
def test_clock_text(angle, text):
    assert clock_text(angle) == text


def test_arc_length_on_graft():
    assert arc_length_on_graft(90.0, 28.0) == pytest.approx(math.pi * 28.0 / 4)
    assert arc_length_on_graft(360.0, 28.0) == pytest.approx(0.0)


def test_anterior_frame_of_the_abdominal_aorta():
    path = descending()
    frame = clock_frame(path, 100.0, MODE_ANTERIOR)
    centre = frame.center
    np.testing.assert_allclose(frame.up, ANTERIOR, atol=1e-9)
    np.testing.assert_allclose(frame.right, LEFT, atol=1e-9)  # 3:00 = patient left
    assert clock_angle(frame, centre + 10 * ANTERIOR) == pytest.approx(0.0)
    assert clock_angle(frame, centre + 10 * LEFT) == pytest.approx(90.0)
    assert clock_angle(frame, centre + 10 * POSTERIOR) == pytest.approx(180.0)
    assert clock_angle(frame, centre + 10 * RIGHT) == pytest.approx(270.0)
    # The out-of-plane component does not matter.
    assert clock_angle(frame, centre + 10 * LEFT + np.array([0.0, 0.0, 3.0])) == pytest.approx(90.0)
    # Straight vessel: no greater curvature, the anterior reference is used and reported.
    gc = clock_frame(path, 100.0, MODE_GREATER_CURVATURE)
    assert gc.fallback and np.allclose(gc.up, ANTERIOR)
    with pytest.raises(ValueError):
        clock_frame(path, 100.0, "screen")


def arch(radius=35.0, leg=60.0):
    phi = np.linspace(0.0, math.pi, 400)
    arc = np.stack([np.zeros_like(phi), radius * np.cos(phi), radius * np.sin(phi)], axis=1)
    return CurvePath(np.vstack([[[0.0, radius, -leg]], arc, [[0.0, -radius, -leg]]]))


def test_arch_top_frames():
    path = arch()
    s_top = path.nearest_arc_length(np.array([0.0, 0.0, 35.0]))[0]
    # Greater curvature: 12:00 superior (outer side of the arch), 3:00 towards the patient's left.
    gc = clock_frame(path, s_top, MODE_GREATER_CURVATURE)
    assert not gc.fallback
    np.testing.assert_allclose(gc.up, [0.0, 0.0, 1.0], atol=1e-3)
    assert np.dot(gc.right, LEFT) > 0.99
    # Anterior reference carried from the descending aorta: "up" points to the lesser curvature.
    an = clock_frame(path, s_top, MODE_ANTERIOR)
    np.testing.assert_allclose(an.up, [0.0, 0.0, -1.0], atol=1e-3)
    assert np.dot(an.right, LEFT) > 0.99


def test_takeoff_angles():
    path = descending()
    frame = clock_frame(path, 100.0, MODE_ANTERIOR)
    ostium = frame.center + 11.0 * LEFT
    # Perpendicular, radial: 90 deg and 0 deg.
    t = takeoff_angles(frame, ostium, LEFT)
    assert t.craniocaudal_deg == pytest.approx(90.0) and t.axial_deg == pytest.approx(0.0)
    # Pointing caudally at 45 deg (distal = towards the feet here).
    t = takeoff_angles(frame, ostium, LEFT + np.array([0.0, 0.0, -1.0]))
    assert t.craniocaudal_deg == pytest.approx(45.0)
    # Deviating towards posterior (clockwise from 3:00 towards 6:00): positive axial angle.
    d = math.cos(math.radians(20)) * LEFT + math.sin(math.radians(20)) * POSTERIOR
    t = takeoff_angles(frame, ostium, d)
    assert t.axial_deg == pytest.approx(20.0) and t.craniocaudal_deg == pytest.approx(90.0)
    assert direction_angle(frame, d) == pytest.approx(110.0)
    assert takeoff_angles(frame, frame.center, LEFT) is None  # ostium on the axis
