import numpy as np

from aortaplan_core.volume import ImageGeometry, Volume
from aortaplan_core.vtk_bridge import (
    points_to_vtk_polyline,
    volume_to_vtk_image,
    vtk_image_to_volume,
    vtk_polydata_points,
)


def test_volume_vtk_round_trip():
    rot = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    g = ImageGeometry(dimensions=(4, 3, 2), spacing=(0.5, 0.7, 1.1), origin=(1, 2, 3), directions=rot)
    vol = Volume(data=np.arange(24, dtype=np.int16).reshape(g.array_shape), geometry=g)
    image = volume_to_vtk_image(vol)
    assert image.GetDimensions() == (4, 3, 2)
    # VTK and the core agree on where voxel (i, j, k) is.
    ras = [0.0, 0.0, 0.0]
    image.TransformIndexToPhysicalPoint(3, 2, 1, ras)
    np.testing.assert_allclose(ras, g.ijk_to_ras([3, 2, 1]))
    assert image.GetScalarComponentAsDouble(3, 2, 1, 0) == vol.data[1, 2, 3]
    back = vtk_image_to_volume(image)
    np.testing.assert_array_equal(back.data, vol.data)
    assert back.geometry.is_close(g)


def test_polyline_round_trip():
    pts = np.random.default_rng(1).normal(size=(7, 3))
    poly = points_to_vtk_polyline(pts)
    assert poly.GetNumberOfLines() == 1
    assert poly.GetCell(0).GetNumberOfPoints() == 7
    np.testing.assert_allclose(vtk_polydata_points(poly), pts)
