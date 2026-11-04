"""Conversions between core types and VTK objects.

Only ``vtkmodules`` sub-modules that need no rendering libraries are imported, so
the core works on headless machines (CI) as well as inside Slicer.
"""

from __future__ import annotations

import numpy as np
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData, vtkPolyLine

from .volume import ImageGeometry, Volume


def volume_to_vtk_image(volume: Volume) -> vtkImageData:
    """Copy a volume into a ``vtkImageData`` carrying spacing, origin and direction (RAS)."""
    g = volume.geometry
    image = vtkImageData()
    image.SetDimensions(*g.dimensions)
    image.SetSpacing(*g.spacing)
    image.SetOrigin(*g.origin)
    image.SetDirectionMatrix(*g.directions.ravel())
    flat = np.ascontiguousarray(volume.data).ravel(order="C")  # i is the fastest axis, as in VTK
    scalars = numpy_support.numpy_to_vtk(flat, deep=True)
    scalars.SetName("ImageScalars")
    image.GetPointData().SetScalars(scalars)
    return image


def vtk_image_to_volume(image: vtkImageData) -> Volume:
    """Copy a ``vtkImageData`` (with its direction matrix) into a :class:`Volume`."""
    dims = image.GetDimensions()
    m = image.GetDirectionMatrix()
    directions = np.array([[m.GetElement(r, c) for c in range(3)] for r in range(3)])
    geometry = ImageGeometry(dimensions=dims, spacing=image.GetSpacing(), origin=image.GetOrigin(), directions=directions)
    scalars = image.GetPointData().GetScalars()
    if scalars is None or scalars.GetNumberOfComponents() != 1:
        raise ValueError("image must have single-component point scalars")
    data = numpy_support.vtk_to_numpy(scalars).reshape(geometry.array_shape).copy()
    return Volume(data=data, geometry=geometry)


def points_to_vtk_polyline(points: np.ndarray) -> vtkPolyData:
    """Make a ``vtkPolyData`` with a single polyline through the given (N, 3) points."""
    pts = np.asarray(points, dtype=float)
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    line = vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(pts))
    for i in range(len(pts)):
        line.GetPointIds().SetId(i, i)
    cells = vtkCellArray()
    cells.InsertNextCell(line)
    poly = vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(cells)
    return poly


def vtk_polydata_points(poly: vtkPolyData) -> np.ndarray:
    """All points of a ``vtkPolyData`` as an (N, 3) array."""
    if poly.GetPoints() is None:
        return np.zeros((0, 3))
    return numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float).copy()
