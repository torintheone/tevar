"""Synchronised navigation along a centerline (SPEC 5 "Синхронизация", stage 3).

``SliceNavigator`` keeps the views at the current position (mm along the curve):
- two straightened CPR views (``cprview.CprView``) at ``angle`` and ``angle + 90``,
  each with a horizontal line at the current position;
- the perpendicular slice, with two short lines showing the planes of the CPR views;
- a cursor point (also visible in 3D) and large numbers in the view corners.
All geometry comes from the core (``CurvePath``, ``StraightenedCpr``).

``ViewInteraction`` connects mouse and keyboard to the navigator:
- CPR views: left button drags the position (a click jumps), wheel +-1 mm (Shift +-5 mm),
  Ctrl+wheel rotates the CPR by +-5 deg, Up/Down +-1 mm; right-button zoom as usual;
- perpendicular slice: wheel or Up/Down +-1 mm (Shift +-5 mm);
- 3D view: a click on the centerline jumps there;
- Ctrl+Up/Down: +-1 mm, Ctrl+Left/Right: rotate by -+10 deg.
While the module is active Shift+mouse move does not move the slices to the crosshair.
"""

import logging

import numpy as np
import qt
import slicer
import vtk

from aortaplan_core.centerline.cpr import StraightenedCpr
from aortaplan_core.centerline.navigation import CurvePath

from . import layout
from .cprview import LINE_COLORS, CprView, _lineNode, compositeNode, setLine
from .display import ViewOverlay, setSliceVisibleIn3D

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


CURSOR_NODE_NAME = "AortaPlanCursor"
AXIS_LINE_HALF_LENGTH_MM = 25.0
CLICK_TOLERANCE_PX = 4
PICK_TOLERANCE_3D_MM = 6.0
STEP_MM = 1.0
FAST_STEP_MM = 5.0
ROTATION_STEP_DEG = 10.0
WHEEL_ROTATION_STEP_DEG = 5.0

# Columns are the screen X, screen Y and normal directions (Slicer's standard orientations).
AXIAL_ROTATION = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
CORONAL_ROTATION = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
SAGITTAL_ROTATION = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
STANDARD_ROTATIONS = {
    layout.CROSS_VIEW: AXIAL_ROTATION,
    layout.LONGITUDINAL_VIEW: CORONAL_ROTATION,
    layout.CPR2_VIEW: SAGITTAL_ROTATION,
}


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
        self.cpr = None
        self.curveNode = None
        self.volumeNode = None
        self.position = 0.0
        self.angle = 0.0
        self.frozen = False
        self.crossFieldOfView = 80.0
        self.listeners = []
        self.overlays = {}
        self.planeIn3D = False
        self.cprViews = [CprView(layout.LONGITUDINAL_VIEW, 0.0), CprView(layout.CPR2_VIEW, 90.0)]
        self._curveObservations = []
        self._displayObservation = None

    # ------------------------------------------------------------------ nodes
    @property
    def crossNode(self):
        return layout.ensureSliceNode(layout.CROSS_VIEW)

    @property
    def longitudinalNode(self):
        """Slice node of the first CPR view."""
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

    def axisLine(self, view):
        return _lineNode(f"{view.layoutName}_axis", LINE_COLORS[view.layoutName])

    # ------------------------------------------------------------------ inputs
    def setVolume(self, volumeNode):
        if self._displayObservation is not None:
            node, tag = self._displayObservation
            node.RemoveObserver(tag)
            self._displayObservation = None
        self.volumeNode = volumeNode
        if volumeNode is not None:
            slicer.util.setSliceViewerLayers(background=volumeNode, fit=False)
            display = volumeNode.GetDisplayNode()
            if display is not None:
                tag = display.AddObserver(vtk.vtkCommand.ModifiedEvent, self._onWindowLevelChanged)
                self._displayObservation = (display, tag)
        self.rebuild()

    def _onWindowLevelChanged(self, caller, event):
        for view in self.cprViews:
            view.syncWindowLevel(self.volumeNode)

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
        """While points are placed the views stay standard (axial/coronal/sagittal)."""
        self.frozen = bool(frozen)
        if self.frozen:
            self.showStandardViews()
        else:
            self.rebuild()

    def rebuild(self):
        self.path = None
        self.cpr = None
        node = self.curveNode
        if node is not None and slicer.mrmlScene.IsNodePresent(node) and node.GetNumberOfControlPoints() >= 2:
            points = slicer.util.arrayFromMarkupsCurvePoints(node, world=True)
            try:
                self.path = CurvePath(points)
                self.cpr = StraightenedCpr(self.path)
            except ValueError as exc:
                logging.warning("AortaPlan: cannot navigate along the curve: %s", exc)
                self.path = self.cpr = None
        if self.path is None or self.frozen:
            self.showStandardViews()
            self._notify()
            return
        self.position = self.path.clamp(self.position)
        self._updateCprImages(fit=True)
        self.apply()

    def _updateCprImages(self, fit=False):
        if self.cpr is None or self.volumeNode is None:
            return
        from .centerline import volumeFromNode

        volume = volumeFromNode(self.volumeNode)
        for view in self.cprViews:
            view.update(self.cpr, volume, self.angle, self.volumeNode, fit=fit)

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
        if self.path is not None and not self.frozen:
            self._updateCprImages(fit=False)
            self.apply()
        else:
            self._notify()

    def rotate(self, delta_deg):
        self.setAngle(self.angle + delta_deg)

    def positionFromRas(self, ras):
        """Arc length of the curve point nearest to an RAS point."""
        if self.path is None:
            return None
        return self.path.nearest_arc_length(ras)[0]

    def jumpToRas(self, ras):
        s = self.positionFromRas(ras)
        if s is not None:
            self.setPosition(s)
        return s

    def jumpToCprPixel(self, viewIndex, x, y):
        """Move to the row under a pixel of a CPR view; returns (offset_mm, s_mm)."""
        if self.cpr is None:
            return None
        offset, s = self.cprViews[viewIndex].cprFromXY(x, y)
        self.setPosition(s)
        return offset, s

    # ------------------------------------------------------------------ views
    def apply(self):
        if self.path is None or self.frozen:
            return
        cross = self.path.cross_section_frame(self.position)
        setSliceToRAS(self.crossNode, cross.slice_to_ras())
        cursor = self.cursorNode()
        cursor.SetNthControlPointPositionWorld(0, *cross.origin)
        for view in self.cprViews:
            view.setPosition(self.position)
            if self.cpr is not None:
                lateral = self.cpr.to_ras(1.0, self.position, self.angle + view.angleOffset) - cross.origin
                half = AXIS_LINE_HALF_LENGTH_MM * lateral
                setLine(self.axisLine(view), cross.origin - half, cross.origin + half)
        setSliceVisibleIn3D(self.crossNode, self.planeIn3D)
        self._updateOverlays()
        self._notify()

    def showStandardViews(self):
        """Axial / coronal / sagittal through the volume centre (no path, or placing points)."""
        center = volumeCenterRas(self.volumeNode) if self.volumeNode is not None else np.zeros(3)
        for name, rotation in STANDARD_ROTATIONS.items():
            m = np.eye(4)
            m[:3, :3] = rotation
            m[:3, 3] = center
            setSliceToRAS(layout.ensureSliceNode(name), m)
            composite = compositeNode(name)
            if composite is not None and self.volumeNode is not None:
                composite.SetBackgroundVolumeID(self.volumeNode.GetID())
        cursor = self.cursorNode(create=False)
        if cursor is not None:
            slicer.mrmlScene.RemoveNode(cursor)
        for view in self.cprViews:
            view.hide()
            line = slicer.mrmlScene.GetFirstNodeByName(f"{view.layoutName}_axis")
            if line is not None and line.GetDisplayNode() is not None:
                line.GetDisplayNode().SetVisibility(False)
        setSliceVisibleIn3D(self.crossNode, False)  # no plane over the 3D rendering without a path
        for overlay in self.overlays.values():
            overlay.setText("")
        if self.volumeNode is not None:
            for name in STANDARD_ROTATIONS:
                widget = layout.sliceWidget(name)
                if widget is not None:
                    widget.sliceLogic().FitSliceToAll()

    def resetFieldsOfView(self):
        setFieldOfViewWidth(self.crossNode, self.crossFieldOfView)

    def attachOverlays(self):
        self.detachOverlays()
        for name in (layout.CROSS_VIEW, *layout.CPR_VIEWS):
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
        for view in self.cprViews:
            if view.layoutName in self.overlays:
                angle = (self.angle + view.angleOffset) % 360.0
                text = (
                    tr("view.position_angle", value=self.position, angle=angle)
                    if view.angleOffset == 0.0
                    else tr("view.angle", angle=angle)
                )
                self.overlays[view.layoutName].setText(text)

    def showCrossSectionIn3D(self, visible):
        """Show the perpendicular slice in the 3D view (only while navigating along a path)."""
        self.planeIn3D = bool(visible)
        setSliceVisibleIn3D(self.crossNode, self.planeIn3D and self.path is not None and not self.frozen)

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
        if self._displayObservation is not None:
            node, tag = self._displayObservation
            node.RemoveObserver(tag)
            self._displayObservation = None
        self.curveNode = None
        self.volumeNode = None
        self.path = None
        self.cpr = None
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
        self._abortTags = {}  # (view name, event) -> observer tag, for events we may consume
        self._styles = {}  # view name -> interactor style
        self._shortcuts = []
        self._press = {}
        self._scrubbing = None  # index of the CPR view being dragged
        self._crosshairBehavior = None
        self.installed = False

    # ------------------------------------------------------------------ install
    def install(self):
        if self.installed or slicer.util.mainWindow() is None:
            return
        for name in (layout.CROSS_VIEW, *layout.CPR_VIEWS):
            widget = layout.sliceWidget(name)
            if widget is None:
                continue
            view = widget.sliceView()
            style = view.sliceViewInteractorStyle()
            self._styles[name] = style
            # Shift is the fast-scroll modifier here: Shift+mouse move must not set the crosshair.
            style.SetActionEnabled(slicer.vtkMRMLSliceViewInteractorStyle.SetCrosshairPosition, False)
            interactor = view.interactor()
            for event in (vtk.vtkCommand.MouseWheelForwardEvent, vtk.vtkCommand.MouseWheelBackwardEvent):
                tag = self._observe(interactor, event, lambda caller, ev, name=name: self._onWheel(caller, ev, name))
                self._abortTags[(name, event)] = (interactor, tag)
            self._observe(interactor, vtk.vtkCommand.KeyPressEvent, self._onKey)
            if name in layout.CPR_VIEWS:
                index = layout.CPR_VIEWS.index(name)
                self._observe(
                    interactor, vtk.vtkCommand.LeftButtonPressEvent, lambda caller, ev, i=index: self._onCprPress(caller, i)
                )
                self._observe(interactor, vtk.vtkCommand.MouseMoveEvent, lambda caller, ev, i=index: self._onCprMove(caller, i))
                self._observe(interactor, vtk.vtkCommand.LeftButtonReleaseEvent, self._onCprRelease)
        threeD = layout.threeDWidget()
        if threeD is not None:
            interactor = threeD.threeDView().interactor()
            self._observe(interactor, vtk.vtkCommand.LeftButtonPressEvent, self._on3DPress)
            self._observe(interactor, vtk.vtkCommand.LeftButtonReleaseEvent, self._on3DRelease)
        # Slicer moves every slice view to the crosshair on Shift+mouse move ("jump slices"),
        # which would pull the perpendicular and CPR views off the centerline.
        crosshair = self._crosshairNode()
        if crosshair is not None:
            self._crosshairBehavior = crosshair.GetCrosshairBehavior()
            crosshair.SetCrosshairBehavior(slicer.vtkMRMLCrosshairNode.NoAction)
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
        self._abortTags = {}
        for style in self._styles.values():
            for action in (
                slicer.vtkMRMLSliceViewInteractorStyle.BrowseSlice,
                slicer.vtkMRMLSliceViewInteractorStyle.SetCrosshairPosition,
                slicer.vtkMRMLSliceViewInteractorStyle.AdjustWindowLevelBackground,
            ):
                style.SetActionEnabled(action, True)
        self._styles = {}
        crosshair = self._crosshairNode()
        if crosshair is not None and self._crosshairBehavior is not None:
            crosshair.SetCrosshairBehavior(self._crosshairBehavior)
        self._crosshairBehavior = None
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts = []
        self._scrubbing = None
        self.installed = False

    @staticmethod
    def _crosshairNode():
        return slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")

    def updateBrowsing(self):
        """Standard slice browsing while there is no path; along-the-path navigation otherwise."""
        navigating = self._navigating()
        for name, style in self._styles.items():
            style.SetActionEnabled(slicer.vtkMRMLSliceViewInteractorStyle.BrowseSlice, not navigating)
            if name in layout.CPR_VIEWS:  # left drag scrubs the position instead of window/level
                style.SetActionEnabled(slicer.vtkMRMLSliceViewInteractorStyle.AdjustWindowLevelBackground, not navigating)

    def _observe(self, obj, event, callback):
        tag = obj.AddObserver(event, callback, 10.0)
        self._observations.append((obj, tag))
        return tag

    def _abort(self, name, event):
        entry = self._abortTags.get((name, event))
        if entry is not None:
            interactor, tag = entry
            command = interactor.GetCommand(tag)
            if command is not None:
                command.SetAbortFlag(1)

    # ------------------------------------------------------------------ handlers
    def _navigating(self):
        return self.navigator.path is not None and not self.navigator.frozen

    @staticmethod
    def _viewTransformMode():
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        return interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.ViewTransform

    def _onWheel(self, interactor, event, name):
        if not self._navigating():
            return
        forward = event == "MouseWheelForwardEvent"
        if interactor.GetControlKey():
            if name in layout.CPR_VIEWS:  # rotate the CPR instead of Slicer's zoom
                self.navigator.rotate(WHEEL_ROTATION_STEP_DEG if forward else -WHEEL_ROTATION_STEP_DEG)
                self._abort(name, vtk.vtkCommand.MouseWheelForwardEvent if forward else vtk.vtkCommand.MouseWheelBackwardEvent)
            return
        step = FAST_STEP_MM if interactor.GetShiftKey() else STEP_MM
        self.navigator.step(step if forward else -step)

    def _onKey(self, interactor, event):
        if not self._navigating() or interactor.GetControlKey():
            return
        key = interactor.GetKeySym()
        step = FAST_STEP_MM if interactor.GetShiftKey() else STEP_MM
        if key == "Up":
            self.navigator.step(step)
        elif key == "Down":
            self.navigator.step(-step)

    def _onCprPress(self, interactor, index):
        if (
            not self._navigating()
            or not self._viewTransformMode()
            or interactor.GetShiftKey()
            or interactor.GetControlKey()
            or interactor.GetAltKey()
        ):
            return
        self._scrubbing = index
        self.navigator.jumpToCprPixel(index, *interactor.GetEventPosition())

    def _onCprMove(self, interactor, index):
        if self._scrubbing == index and self._navigating():
            self.navigator.jumpToCprPixel(index, *interactor.GetEventPosition())

    def _onCprRelease(self, interactor, event):
        self._scrubbing = None

    def _on3DPress(self, interactor, event):
        self._press[interactor] = interactor.GetEventPosition()

    def _on3DRelease(self, interactor, event):
        start = self._press.pop(interactor, None)
        if start is None or not self._navigating() or not self._viewTransformMode():
            return
        x, y = interactor.GetEventPosition()
        if abs(x - start[0]) > CLICK_TOLERANCE_PX or abs(y - start[1]) > CLICK_TOLERANCE_PX:
            return  # the view was rotated, not clicked
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
