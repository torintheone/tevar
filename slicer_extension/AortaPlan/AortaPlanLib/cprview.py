"""Straightened CPR views (stage 3): image volumes, slice geometry, position line.

Each CPR image is stored in a scalar volume node placed in its own region of RAS space,
far above the patient (so nothing else is ever displayed in that plane and the CPR is
never mixed with patient data): column i -> +R (lateral offset), row j -> +A (arc length),
a single slice. The CPR slice view looks at that plane with row 0 (proximal) at the top.
"""

import numpy as np
import slicer

from . import layout

CPR_ORIGINS = {layout.LONGITUDINAL_VIEW: np.array([0.0, 0.0, 50000.0]), layout.CPR2_VIEW: np.array([0.0, 0.0, 60000.0])}
CPR_NODE_NAMES = {layout.LONGITUDINAL_VIEW: "AortaPlan_CPR_1", layout.CPR2_VIEW: "AortaPlan_CPR_2"}
LINE_COLORS = {layout.LONGITUDINAL_VIEW: (0.85, 0.63, 0.23), layout.CPR2_VIEW: (0.23, 0.65, 0.85)}
POSITION_LINE_COLOR = (1.0, 0.85, 0.2)


def _hidden(node):
    node.SetHideFromEditors(True)
    node.SetSaveWithScene(False)
    return node


def _lineNode(name, color, visible3D=False):
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    if node is None:
        node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name))
        node.AddControlPoint(0.0, 0.0, 0.0, "")
        node.AddControlPoint(1.0, 0.0, 0.0, "")
        node.SetLocked(True)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetColor(*color)
        display.SetSelectedColor(*color)
        display.SetPointLabelsVisibility(False)
        display.SetPropertiesLabelVisibility(False)
        display.SetGlyphScale(0.5)
        display.SetVisibility3D(visible3D)
    return node


def setLine(node, start, end, visible=True):
    node.SetNthControlPointPositionWorld(0, *np.asarray(start, float))
    node.SetNthControlPointPositionWorld(1, *np.asarray(end, float))
    if node.GetDisplayNode() is not None:
        node.GetDisplayNode().SetVisibility(bool(visible))


def compositeNode(layoutName):
    for node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
        if node.GetLayoutName() == layoutName:
            return node
    return None


class CprView:
    """One straightened CPR view at ``angle + angleOffset``."""

    def __init__(self, layoutName, angleOffset):
        self.layoutName = layoutName
        self.angleOffset = float(angleOffset)
        self.origin = CPR_ORIGINS[layoutName]
        self.cpr = None
        self.angle = 0.0

    @property
    def sliceNode(self):
        return layout.ensureSliceNode(self.layoutName)

    def volumeNode(self, create=True):
        name = CPR_NODE_NAMES[self.layoutName]
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name))
            node.CreateDefaultDisplayNodes()
        return node

    def positionLine(self):
        return _lineNode(CPR_NODE_NAMES[self.layoutName] + "_position", POSITION_LINE_COLOR)

    # ------------------------------------------------------------------ geometry
    def ijkToRas(self):
        m = np.eye(4)
        m[0, 0] = self.cpr.column_spacing_mm
        m[1, 1] = self.cpr.row_spacing_mm
        m[:3, 3] = self.origin
        return m

    def pixelToRas(self, column, row):
        return self.origin + np.array([column * self.cpr.column_spacing_mm, row * self.cpr.row_spacing_mm, 0.0])

    def rasToPixel(self, ras):
        rel = np.asarray(ras, float)[:3] - self.origin
        return rel[0] / self.cpr.column_spacing_mm, rel[1] / self.cpr.row_spacing_mm

    def cprFromXY(self, x, y):
        """(offset_mm, s_mm) under a slice-view pixel."""
        ras = self.sliceNode.GetXYToRAS().MultiplyPoint([float(x), float(y), 0.0, 1.0])
        column, row = self.rasToPixel(ras)
        return self.cpr.pixel_to_cpr(column, row)

    def xyFromCpr(self, offset, s):
        """Slice-view pixel of a CPR point (for tests and overlays)."""
        import vtk

        column, row = self.cpr.cpr_to_pixel(offset, s)
        ras = self.pixelToRas(column, row)
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(self.sliceNode.GetXYToRAS(), rasToXy)
        return rasToXy.MultiplyPoint([*ras, 1.0])[:2]

    # ------------------------------------------------------------------ updates
    def update(self, cpr, volume, angle, sourceVolumeNode, fit=False):
        """Recompute the CPR image for ``angle`` and show it in the view."""
        self.cpr = cpr
        self.angle = float(angle)
        image = cpr.sample(volume, self.angle + self.angleOffset)
        node = self.volumeNode()
        node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(self.ijkToRas()))
        slicer.util.updateVolumeFromArray(node, image[np.newaxis, :, :])
        self.syncWindowLevel(sourceVolumeNode)
        composite = compositeNode(self.layoutName)
        if composite is not None:
            composite.SetBackgroundVolumeID(node.GetID())
            composite.SetForegroundVolumeID(None)
            composite.SetLabelVolumeID(None)
        if fit:
            self.fit()

    def fit(self):
        """Look at the CPR plane, whole image height visible, row 0 at the top."""
        cpr = self.cpr
        width = (cpr.columns - 1) * cpr.column_spacing_mm
        height = (cpr.rows - 1) * cpr.row_spacing_mm
        m = np.eye(4)
        m[:3, 0] = [1.0, 0.0, 0.0]  # screen right = +R = increasing column
        m[:3, 1] = [0.0, -1.0, 0.0]  # screen up = row 0 (proximal)
        m[:3, 2] = [0.0, 0.0, -1.0]
        m[:3, 3] = self.origin + np.array([width / 2.0, height / 2.0, 0.0])
        node = self.sliceNode
        node.GetSliceToRAS().DeepCopy(slicer.util.vtkMatrixFromArray(m))
        node.UpdateMatrices()
        dims = node.GetDimensions()
        aspect = dims[1] / dims[0] if dims[0] > 0 and dims[1] > 0 else 1.0
        fovHeight = max(height * 1.04, width * 1.1 * aspect)
        node.SetFieldOfView(fovHeight / aspect, fovHeight, node.GetFieldOfView()[2])

    def setPosition(self, s):
        if self.cpr is None:
            return
        _, row = self.cpr.cpr_to_pixel(0.0, s)
        setLine(self.positionLine(), self.pixelToRas(-0.5, row), self.pixelToRas(self.cpr.columns - 0.5, row))

    def hide(self):
        line = slicer.mrmlScene.GetFirstNodeByName(CPR_NODE_NAMES[self.layoutName] + "_position")
        if line is not None and line.GetDisplayNode() is not None:
            line.GetDisplayNode().SetVisibility(False)

    def syncWindowLevel(self, sourceVolumeNode):
        node = self.volumeNode(create=False)
        if node is None or sourceVolumeNode is None:
            return
        source, target = sourceVolumeNode.GetDisplayNode(), node.GetDisplayNode()
        if source is None or target is None:
            return
        target.SetAutoWindowLevel(False)
        target.SetWindowLevel(source.GetWindow(), source.GetLevel())
