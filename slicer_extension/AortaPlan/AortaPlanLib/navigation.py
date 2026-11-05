"""Synchronised navigation along a centerline curve (SPEC 5 "Синхронизация").

``SliceNavigator`` keeps the perpendicular slice, the longitudinal (interim CPR) slice,
the cursor and the overlays at the current position. All geometry comes from
``aortaplan_core.centerline.navigation.CurvePath``.

``ViewInteraction`` connects mouse and keyboard to the navigator:
- mouse wheel / Up, Down in the AortaPlan slice views: +-1 mm (Shift: +-5 mm);
- click in the longitudinal view: jump to the nearest curve point;
- click on the curve in the 3D view: jump there;
- Ctrl+Up/Down: +-1 mm, Ctrl+Left/Right: rotate the longitudinal view by -+10 deg.
"""

import logging

import numpy as np
import qt
import slicer
import vtk

from aortaplan_core.centerline.navigation import CurvePath

from . import layout
from .display import ViewOverlay, setSliceVisibleIn3D

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


CURSOR_NODE_NAME = "AortaPlanCursor"
CLICK_TOLERANCE_PX = 4
PICK_TOLERANCE_3D_MM = 6.0
STEP_MM = 1.0
FAST_STEP_MM = 5.0
ROTATION_STEP_DEG = 10.0

AXIAL_ROTATION = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # columns: X, Y, Z
CORONAL_ROTATION = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def setSliceToRAS(sliceNode, matrix4x4):
    sliceNode.GetSliceToRAS().DeepCopy(slicer.util.vtkMatrixFromArray(matrix4x4))
    sliceNode.UpdateMatrices()


def setFieldOfViewWidth(sliceNode, width_mm):
    dims = sliceNode.GetDimensions()
    aspect = dims[1] / dims[0] if dims[0] > 0 and dims[1] > 0 else 1.0
    sliceNode.SetFieldOfView(width_mm, width_mm * aspect, sliceNode.GetFieldOfView()[2])


def volumeCenterRas(volumeNode):
    bounds = [0.0] * 6
    volumeNode.GetRASBounds(bounds)
    return np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])


class SliceNavigator:
    """Current position (mm along the curve) and CPR angle, applied to the views."""

    def __init__(self):
        self.path = None
        self.curveNode = None
        self.volumeNode = None
        self.position = 0.0
        self.angle = 0.0
        self.frozen = False
        self.crossFieldOfView = 80.0
        self.longitudinalFieldOfView = 220.0
        self.listeners = []
        self.overlays = {}
        self._curveObservations = []

    # ------------------------------------------------------------------ nodes
    @property
    def crossNode(self):
        return layout.ensureSliceNode(layout.CROSS_VIEW)

    @property
    def longitudinalNode(self):
        return layout.ensureSliceNode(layout.LONGITUDINAL_VIEW)

    def cursorNode(self, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(CURSOR_NODE_NAME)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", CURSOR_NODE_NAME)
            node.SetHideFromEditors(True)
            node.SetSaveWithScene(False)
            node.AddControlPoint(0.0, 0.0, 0.0, "")
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetGlyphTypeFromString("CrossDot2D")
            display.SetGlyphScale(4.0)
            display.SetColor(1.0, 0.85, 0.2)
            display.SetSelectedColor(1.0, 0.85, 0.2)
            display.SetPointLabelsVisibility(False)
            display.SetPropertiesLabelVisibility(False)
        return node

    # ------------------------------------------------------------------ inputs
    def setVolume(self, volumeNode):
        self.volumeNode = volumeNode
        if volumeNode is not None:
            slicer.util.setSliceViewerLayers(background=volumeNode, fit=False)
        if self.path is None:
            self.showStandardViews()

    def setCurve(self, curveNode):
        for node, tag in self._curveObservations:
            node.RemoveObserver(tag)
        self._curveObservations = []
        self.curveNode = curveNode
        if curveNode is not None:
            for event in (
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                slicer.vtkMRMLMarkupsNode.PointAddedEvent,
                slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
            ):
                self._curveObservations.append((curveNode, curveNode.AddObserver(event, self._onCurveModified)))
        self.rebuild()

    def _onCurveModified(self, caller, event):
        if not self.frozen:
            self.rebuild()

    def setFrozen(self, frozen):
        """While a curve is being drawn the views stay standard (axial/coronal) for placing points."""
        self.frozen = bool(frozen)
        if self.frozen:
            self.showStandardViews()
        else:
            self.rebuild()

    def rebuild(self):
        self.path = None
        node = self.curveNode
        if node is not None and slicer.mrmlScene.IsNodePresent(node) and node.GetNumberOfControlPoints() >= 2:
            points = slicer.util.arrayFromMarkupsCurvePoints(node, world=True)
            try:
                self.path = CurvePath(points)
            except ValueError as exc:
                logging.warning("AortaPlan: cannot navigate along the curve: %s", exc)
        if self.path is None:
            self.showStandardViews()
            self._notify()
            return
        self.position = self.path.clamp(self.position)
        self.apply()

    # ------------------------------------------------------------------ navigation
    def setPosition(self, position_mm):
        if self.path is None:
            return
        self.position = self.path.clamp(float(position_mm))
        self.apply()

    def step(self, delta_mm):
        self.setPosition(self.position + delta_mm)

    def setAngle(self, angle_deg):
        self.angle = float(angle_deg) % 360.0
        if self.path is not None:
            self.apply()
        else:
            self._notify()

    def rotate(self, delta_deg):
        self.setAngle(self.angle + delta_deg)

    def positionFromRas(self, ras):
        """Arc length of the curve point nearest to an RAS point (a click in the CPR view)."""
        if self.path is None:
            return None
        return self.path.nearest_arc_length(ras)[0]

    def jumpToRas(self, ras):
        s = self.positionFromRas(ras)
        if s is not None:
            self.setPosition(s)
        return s

    # ------------------------------------------------------------------ views
    def apply(self):
        if self.path is None or self.frozen:
            return
        cross = self.path.cross_section_frame(self.position)
        longitudinal = self.path.longitudinal_frame(self.position, self.angle)
        setSliceToRAS(self.crossNode, cross.slice_to_ras())
        setSliceToRAS(self.longitudinalNode, longitudinal.slice_to_ras())
        cursor = self.cursorNode()
        cursor.SetNthControlPointPositionWorld(0, *cross.origin)
        self._updateOverlays()
        self._notify()

    def showStandardViews(self):
        """Axial (perpendicular view) and coronal (large view) through the volume centre."""
        center = volumeCenterRas(self.volumeNode) if self.volumeNode is not None else np.zeros(3)
        for node, rotation in ((self.crossNode, AXIAL_ROTATION), (self.longitudinalNode, CORONAL_ROTATION)):
            m = np.eye(4)
            m[:3, :3] = rotation
            m[:3, 3] = center
            setSliceToRAS(node, m)
        cursor = self.cursorNode(create=False)
        if cursor is not None:
            slicer.mrmlScene.RemoveNode(cursor)
        for overlay in self.overlays.values():
            overlay.setText("")
        if self.volumeNode is not None:
            for name in (layout.CROSS_VIEW, layout.LONGITUDINAL_VIEW):
                widget = layout.sliceWidget(name)
                if widget is not None:
                    widget.sliceLogic().FitSliceToAll()

    def resetFieldsOfView(self):
        setFieldOfViewWidth(self.crossNode, self.crossFieldOfView)
        setFieldOfViewWidth(self.longitudinalNode, self.longitudinalFieldOfView)

    def attachOverlays(self):
        self.detachOverlays()
        for name in (layout.CROSS_VIEW, layout.LONGITUDINAL_VIEW):
            widget = layout.sliceWidget(name)
            if widget is not None:
                self.overlays[name] = ViewOverlay(widget.sliceView())
        self._updateOverlays()

    def detachOverlays(self):
        for overlay in self.overlays.values():
            overlay.remove()
        self.overlays = {}

    def _updateOverlays(self):
        if self.path is None:
            return
        if layout.CROSS_VIEW in self.overlays:
            self.overlays[layout.CROSS_VIEW].setText(tr("view.position", value=self.position))
        if layout.LONGITUDINAL_VIEW in self.overlays:
            self.overlays[layout.LONGITUDINAL_VIEW].setText(
                tr("view.position_angle", value=self.position, angle=self.angle)
            )

    def showCrossSectionIn3D(self, visible):
        setSliceVisibleIn3D(self.crossNode, visible)

    def _notify(self):
        for listener in list(self.listeners):
            try:
                listener()
            except Exception:
                logging.exception("AortaPlan: navigation listener failed")

    def reset(self):
        """Forget the curve and volume without touching the views (the scene is being closed)."""
        for node, tag in self._curveObservations:
            node.RemoveObserver(tag)
        self._curveObservations = []
        self.curveNode = None
        self.volumeNode = None
        self.path = None
        self.frozen = False
        self._notify()

    def cleanup(self):
        self.reset()
        self.detachOverlays()
        self.listeners = []


class ViewInteraction:
    """Mouse, keyboard and shortcut bindings for the navigator (only with a main window)."""

    def __init__(self, navigator):
        self.navigator = navigator
        self._observations = []  # (vtkObject, tag)
        self._styles = []
        self._intersectionDisplays = []
        self._shortcuts = []
        self._press = {}
        self.installed = False

    # ------------------------------------------------------------------ install
    def install(self):
        if self.installed or slicer.util.mainWindow() is None:
            return
        for name in (layout.CROSS_VIEW, layout.LONGITUDINAL_VIEW):
            widget = layout.sliceWidget(name)
            if widget is None:
                continue
            view = widget.sliceView()
            self._styles.append(view.sliceViewInteractorStyle())
            interactor = view.interactor()
            self._observe(interactor, vtk.vtkCommand.MouseWheelForwardEvent, self._onWheel)
            self._observe(interactor, vtk.vtkCommand.MouseWheelBackwardEvent, self._onWheel)
            self._observe(interactor, vtk.vtkCommand.KeyPressEvent, self._onKey)
            if name == layout.LONGITUDINAL_VIEW:
                self._observe(interactor, vtk.vtkCommand.LeftButtonPressEvent, self._onPress)
                self._observe(interactor, vtk.vtkCommand.LeftButtonReleaseEvent, self._onSliceRelease)
            displayNode = widget.sliceLogic().GetSliceDisplayNode()
            if displayNode is not None:
                self._intersectionDisplays.append((displayNode, displayNode.GetIntersectingSlicesVisibility()))
                displayNode.SetIntersectingSlicesVisibility(True)
        threeD = layout.threeDWidget()
        if threeD is not None:
            interactor = threeD.threeDView().interactor()
            self._observe(interactor, vtk.vtkCommand.LeftButtonPressEvent, self._onPress)
            self._observe(interactor, vtk.vtkCommand.LeftButtonReleaseEvent, self._on3DRelease)
        mainWindow = slicer.util.mainWindow()
        for sequence, callback in (
            ("Ctrl+Up", lambda: self.navigator.step(STEP_MM)),
            ("Ctrl+Down", lambda: self.navigator.step(-STEP_MM)),
            ("Ctrl+Right", lambda: self.navigator.rotate(ROTATION_STEP_DEG)),
            ("Ctrl+Left", lambda: self.navigator.rotate(-ROTATION_STEP_DEG)),
        ):
            shortcut = qt.QShortcut(qt.QKeySequence(sequence), mainWindow)
            shortcut.connect("activated()", callback)
            self._shortcuts.append(shortcut)
        self.installed = True
        self.updateBrowsing()

    def uninstall(self):
        for obj, tag in self._observations:
            obj.RemoveObserver(tag)
        self._observations = []
        for style in self._styles:
            style.SetActionEnabled(slicer.vtkMRMLSliceViewInteractorStyle.BrowseSlice, True)
        self._styles = []
        for displayNode, visible in self._intersectionDisplays:
            displayNode.SetIntersectingSlicesVisibility(visible)
        self._intersectionDisplays = []
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts = []
        self.installed = False

    def updateBrowsing(self):
        """Standard slice browsing while there is no curve; along-the-curve navigation otherwise."""
        navigating = self.navigator.path is not None and not self.navigator.frozen
        for style in self._styles:
            style.SetActionEnabled(slicer.vtkMRMLSliceViewInteractorStyle.BrowseSlice, not navigating)

    def _observe(self, obj, event, callback):
        self._observations.append((obj, obj.AddObserver(event, callback, 10.0)))

    # ------------------------------------------------------------------ handlers
    def _navigating(self):
        return self.navigator.path is not None and not self.navigator.frozen

    @staticmethod
    def _viewTransformMode():
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        return interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.ViewTransform

    def _onWheel(self, interactor, event):
        if not self._navigating() or interactor.GetControlKey():
            return  # Ctrl+wheel keeps Slicer's zoom
        step = FAST_STEP_MM if interactor.GetShiftKey() else STEP_MM
        self.navigator.step(step if event == "MouseWheelForwardEvent" else -step)

    def _onKey(self, interactor, event):
        if not self._navigating() or interactor.GetControlKey():
            return
        key = interactor.GetKeySym()
        step = FAST_STEP_MM if interactor.GetShiftKey() else STEP_MM
        if key == "Up":
            self.navigator.step(step)
        elif key == "Down":
            self.navigator.step(-step)

    def _onPress(self, interactor, event):
        self._press[interactor] = interactor.GetEventPosition()

    def _isClick(self, interactor):
        start = self._press.pop(interactor, None)
        if start is None or not self._navigating() or not self._viewTransformMode():
            return False
        x, y = interactor.GetEventPosition()
        return abs(x - start[0]) <= CLICK_TOLERANCE_PX and abs(y - start[1]) <= CLICK_TOLERANCE_PX

    def _onSliceRelease(self, interactor, event):
        if not self._isClick(interactor):
            return
        x, y = interactor.GetEventPosition()
        xyToRas = self.navigator.longitudinalNode.GetXYToRAS()
        ras = xyToRas.MultiplyPoint([float(x), float(y), 0.0, 1.0])[:3]
        self.navigator.jumpToRas(ras)

    def _on3DRelease(self, interactor, event):
        if not self._isClick(interactor):
            return
        x, y = interactor.GetEventPosition()
        renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
        points = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), depth)
            renderer.DisplayToWorld()
            world = renderer.GetWorldPoint()
            points.append(np.array(world[:3]) / (world[3] if world[3] else 1.0))
        hit = self.navigator.path.closest_arc_length_to_ray(points[0], points[1] - points[0], PICK_TOLERANCE_3D_MM)
        if hit is not None:
            self.navigator.setPosition(hit[0])
