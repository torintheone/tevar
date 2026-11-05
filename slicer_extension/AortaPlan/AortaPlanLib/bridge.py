"""Boundary between MRML nodes and core data types.

Both sides use RAS millimetres and ``[k, j, i]`` voxel arrays, so the conversion is a
copy of the array and the IJK->RAS matrix. Parent transforms are not supported yet:
a transformed volume is rejected instead of being silently measured in the wrong place.
"""

import numpy as np
import slicer
import vtk

from aortaplan_core.volume import ImageGeometry, Volume


def volumeToNode(volume, name="AortaPlanVolume"):
    """Create a scalar volume node from a core :class:`Volume` (data are copied)."""
    return slicer.util.addVolumeFromArray(volume.data, volume.geometry.ijk_to_ras_matrix(), name=name)


def nodeGeometry(volumeNode):
    """IJK->RAS geometry of a volume node without copying its voxels."""
    imageData = volumeNode.GetImageData()
    if imageData is None:
        raise ValueError("volume node has no image data")
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    return ImageGeometry.from_ijk_to_ras_matrix(slicer.util.arrayFromVTKMatrix(matrix), imageData.GetDimensions())


def nodeToVolume(volumeNode):
    """Copy a scalar volume node into a core :class:`Volume`."""
    if volumeNode.GetParentTransformNode() is not None:
        raise ValueError("volumes under a transform are not supported; harden the transform first")
    imageData = volumeNode.GetImageData()
    if imageData is None:
        raise ValueError("volume node has no image data")
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    geometry = ImageGeometry.from_ijk_to_ras_matrix(slicer.util.arrayFromVTKMatrix(matrix), imageData.GetDimensions())
    data = np.array(slicer.util.arrayFromVolume(volumeNode), copy=True)
    return Volume(data=data, geometry=geometry)
