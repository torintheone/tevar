"""Synchronised navigation along a centerline (SPEC 5 "Синхронизация", stage 3).

``SliceNavigator`` keeps the views at the current position (mm along the curve):
- two straightened CPR views (``cprview.CprView``) at ``angle`` and ``angle + 90``,
  each with a horizontal line at the current position;
- the perpendicular slice, with two short lines showing the planes of the CPR views;
- a cursor point (also visible in 3D) and large numbers in the view corners.
All geometry comes from the core (``CurvePath``, ``StraightenedCpr``).

Two viewing modes:
- ``MODE_PATH`` (along the centerline, above);
- ``MODE_MPR`` (as in RadiAnt): axial, coronal and sagittal slices through a common point with
  interactive reference lines (drag the centre to move, the lines to rotate); a click moves
  the other slices to the clicked point; Shift+mouse move does the same (Slicer crosshair).
Without a centerline (or while points are placed) the views are always MPR.

``ViewInteraction`` connects mouse and keyboard to the navigator in MODE_PATH:
- CPR views: left button drags the position (a click jumps), wheel +-1 mm (Shift +-5 mm),
  Ctrl+wheel rotates the CPR by +-5 deg, Up/Down +-1 mm; right-button zoom as usual;
- perpendicular slice: wheel or Up/Down +-1 mm (Shift +-5 mm); dragging one of the coloured
  CPR lines rotates the CPRs;
- 3D view: a click on the centerline jumps there;
- Ctrl+Up/Down: +-1 mm, Ctrl+Left/Right: rotate by -+10 deg.
In MODE_PATH Shift+mouse move does not move the slices to the crosshair.
"""

import logging

import numpy as np
import qt
import slicer
import vtk

from aortaplan_core.centerline.cpr import StraightenedCpr, nearest_line_angle
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
LINE_GRAB_TOLERANCE_PX = 10.0
MODE_PATH = "path"
MODE_MPR = "mpr"

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
        self.mode = MODE_PATH
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

    @property
    def navigating(self):
        """Views follow the centerline (otherwise they are a free MPR)."""
        return self.path is not None and not self.frozen and self.mode == MODE_PATH

    def setMode(self, mode):
        """MODE_PATH or MODE_MPR; entering MPR centres the slices on the current point."""
        self.mode = MODE_MPR if mode == MODE_MPR else MODE_PATH
        if self.mode == MODE_MPR:
            center = self.path.point_at(self.position) if self.path is not None else None
            self.showStandardViews(center)
            self._notify()
        else:
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
        if self.path is not None:
            self.position = self.path.clamp(self.position)
        if self.mode == MODE_MPR and self.path is not None and not self.frozen:
            self._notify()  # free MPR: the views are left where the user put them
            return
        if self.path is None or self.frozen:
            self.showStandardViews()
            self._notify()
            return
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
        if self.navigating:
            self.apply()
        else:
            self._notify()

    def step(self, delta_mm):
        self.setPosition(self.position + delta_mm)

    def setAngle(self, angle_deg):
        self.angle = float(angle_deg) % 360.0
        if self.navigating:
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
        if not self.navigating:
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
        axial = layout.findSliceNode(layout.AXIAL_VIEW)
        if axial is not None:  # the optional axial view follows the position
            axial.JumpSliceByOffsetting(*cross.origin)
        setSliceVisibleIn3D(self.crossNode, self.planeIn3D)
        self._updateOverlays()
        self._notify()

    def showStandardViews(self, center=None):
        """MPR: axial / coronal / sagittal through ``center`` (default: the volume centre)."""
        if center is None:
            center = volumeCenterRas(self.volumeNode) if self.volumeNode is not None else np.zeros(3)
        center = np.asarray(center, dtype=float)
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
                    layout.ensureSliceNode(name).JumpSliceByOffsetting(*center)  # FitSliceToAll recentres
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        if crosshair is not None:
            crosshair.SetCrosshairRAS(*center)

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
        setSliceVisibleIn3D(self.crossNode, self.planeIn3D and self.navigating)

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
        self._abortTags = {}  # (view name, event) -> (interactor, observer tag), for events we may consume
        self._styles = {}  # view name -> interactor style
        self._intersections = {}  # view name -> (slice display node, original settings)
        self._shortcuts = []
        self._press = {}
        self._scrubbing = None  # index of the CPR view being dragged
        self._rotating = None  # index of the CPR whose line is being dragged in the perpendicular slice
        self._crosshairBehavior = None
        self.installed = False

    # ------------------------------------------------------------------ install
    def install(self):
        if self.installed or slicer.util.mainWindow() is None:
            return
        for name in layout.SLICE_VIEWS:
            widget = layout.sliceWidget(name)
            if widget is None:
                continue
            view = widget.sliceView()
            self._styles[name] = view.sliceViewInteractorStyle()
            interactor = view.interactor()
            for event in (vtk.vtkCommand.MouseWheelForwardEvent, vtk.vtkCommand.MouseWheelBackwardEvent):
                tag = self._observe(interactor, event, lambda caller, ev, name=name: self._onWheel(caller, ev, name))
                self._abortTags[(name, event)] = (interactor, tag)
            tag = self._observe(
                interactor, vtk.vtkCommand.LeftButtonPressEvent, lambda caller, ev, name=name: self._onPress(caller, name)
            )
            self._abortTags[(name, vtk.vtkCommand.LeftButtonPressEvent)] = (interactor, tag)
            self._observe(interactor, vtk.vtkCommand.MouseMoveEvent, lambda caller, ev, name=name: self._onMove(caller, name))
            self._observe(
                interactor, vtk.vtkCommand.LeftButtonReleaseEvent, lambda caller, ev, name=name: self._onRelease(caller, name)
            )
            if name != layout.AXIAL_VIEW:
                self._observe(interactor, vtk.vtkCommand.KeyPressEvent, self._onKey)
            displayNode = widget.sliceLogic().GetSliceDisplayNode()
            if displayNode is not None:
                original = (
                    displayNode.GetIntersectingSlicesVisibility(),
                    displayNode.GetIntersectingSlicesInteractive(),
                    displayNode.GetIntersectingSlicesTranslationEnabled(),
                    displayNode.GetIntersectingSlicesRotationEnabled(),
                )
                self._intersections[name] = (displayNode, original)
        threeD = layout.threeDWidget()
        if threeD is not None:
            interactor = threeD.threeDView().interactor()
            self._observe(interactor, vtk.vtkCommand.LeftButtonPressEvent, self._on3DPress)
            self._observe(interactor, vtk.vtkCommand.LeftButtonReleaseEvent, self._on3DRelease)
        crosshair = self._crosshairNode()
        if crosshair is not None:
            self._crosshairBehavior = crosshair.GetCrosshairBehavior()
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
        for displayNode, (visible, interactive, translation, rotation) in self._intersections.values():
            displayNode.SetIntersectingSlicesVisibility(visible)
            displayNode.SetIntersectingSlicesInteractive(interactive)
            displayNode.SetIntersectingSlicesTranslationEnabled(translation)
            displayNode.SetIntersectingSlicesRotationEnabled(rotation)
        self._intersections = {}
        crosshair = self._crosshairNode()
        if crosshair is not None and self._crosshairBehavior is not None:
            crosshair.SetCrosshairBehavior(self._crosshairBehavior)
        self._crosshairBehavior = None
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts = []
        self._scrubbing = self._rotating = None
        self.installed = False

    @staticmethod
    def _crosshairNode():
        return slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")

    def updateBrowsing(self):
        """MODE_PATH: navigation along the centerline. MPR: standard Slicer browsing + RadiAnt-like lines."""
        navigating = self.navigator.navigating
        style = slicer.vtkMRMLSliceViewInteractorStyle
        for name, interactorStyle in self._styles.items():
            if name == layout.AXIAL_VIEW:
                continue  # the axial view keeps Slicer's behaviour in both modes
            interactorStyle.SetActionEnabled(style.BrowseSlice, not navigating)
            interactorStyle.SetActionEnabled(style.SetCrosshairPosition, not navigating)
            if name in layout.CPR_VIEWS:  # left drag scrubs the position instead of window/level
                interactorStyle.SetActionEnabled(style.AdjustWindowLevelBackground, not navigating)
        for displayNode, _ in self._intersections.values():
            displayNode.SetIntersectingSlicesVisibility(not navigating)
            displayNode.SetIntersectingSlicesInteractive(not navigating)
            displayNode.SetIntersectingSlicesTranslationEnabled(True)
            displayNode.SetIntersectingSlicesRotationEnabled(True)
        crosshair = self._crosshairNode()
        if crosshair is not None:
            if navigating:
                # Shift+mouse move would otherwise move every slice to the crosshair.
                crosshair.SetCrosshairBehavior(slicer.vtkMRMLCrosshairNode.NoAction)
            else:
                original = self._crosshairBehavior
                if original is None or original == slicer.vtkMRMLCrosshairNode.NoAction:
                    original = slicer.vtkMRMLCrosshairNode.OffsetJumpSlice
                crosshair.SetCrosshairBehavior(original)

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

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _viewTransformMode():
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        return interactionNode.GetCurrentInteractionMode() == slicer.vtkMRMLInteractionNode.ViewTransform

    @staticmethod
    def _plainClick(interactor):
        return not (interactor.GetShiftKey() or interactor.GetControlKey() or interactor.GetAltKey())

    @staticmethod
    def _xyToRas(name, x, y):
        node = layout.ensureSliceNode(name)
        return np.array(node.GetXYToRAS().MultiplyPoint([float(x), float(y), 0.0, 1.0])[:3])

    def _cprLineUnder(self, x, y):
        """Index of the CPR plane line (in the perpendicular slice) under a pixel, or None."""
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(self.navigator.crossNode.GetXYToRAS(), rasToXy)
        best, bestDistance = None, LINE_GRAB_TOLERANCE_PX
        for index, view in enumerate(self.navigator.cprViews):
            points = slicer.util.arrayFromMarkupsControlPoints(self.navigator.axisLine(view), world=True)
            a, b = (np.array(rasToXy.MultiplyPoint([*p, 1.0])[:2]) for p in points)
            ab = b - a
            t = np.clip(np.dot(np.array([x, y]) - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
            distance = float(np.linalg.norm(a + t * ab - np.array([x, y])))
            if distance < bestDistance:
                best, bestDistance = index, distance
        return best

    def _rotateTowards(self, x, y):
        ras = self._xyToRas(layout.CROSS_VIEW, x, y)
        direction = self.navigator.cpr.angle_through(ras)
        if direction is None:
            return
        offset = self.navigator.cprViews[self._rotating].angleOffset
        current = (self.navigator.angle + offset) % 360.0
        self.navigator.setAngle(nearest_line_angle(direction, current) - offset)

    # ------------------------------------------------------------------ handlers
    def _onWheel(self, interactor, event, name):
        if not self.navigator.navigating or name == layout.AXIAL_VIEW:
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
        if not self.navigator.navigating or interactor.GetControlKey():
            return
        key = interactor.GetKeySym()
        step = FAST_STEP_MM if interactor.GetShiftKey() else STEP_MM
        if key == "Up":
            self.navigator.step(step)
        elif key == "Down":
            self.navigator.step(-step)

    def _onPress(self, interactor, name):
        x, y = interactor.GetEventPosition()
        self._press[name] = (x, y)
        if not self.navigator.navigating or not self._viewTransformMode() or not self._plainClick(interactor):
            return
        if name in layout.CPR_VIEWS:
            self._scrubbing = layout.CPR_VIEWS.index(name)
            self.navigator.jumpToCprPixel(self._scrubbing, x, y)
        elif name == layout.CROSS_VIEW and self.navigator.cpr is not None:
            line = self._cprLineUnder(x, y)
            if line is not None:
                self._rotating = line
                self._abort(name, vtk.vtkCommand.LeftButtonPressEvent)  # no window/level drag

    def _onMove(self, interactor, name):
        x, y = interactor.GetEventPosition()
        if self._scrubbing is not None and name in layout.CPR_VIEWS and self.navigator.navigating:
            self.navigator.jumpToCprPixel(self._scrubbing, x, y)
        elif self._rotating is not None and name == layout.CROSS_VIEW and self.navigator.navigating:
            self._rotateTowards(x, y)

    def _onRelease(self, interactor, name):
        start = self._press.pop(name, None)
        self._scrubbing = self._rotating = None
        if start is None or self.navigator.navigating or not self._viewTransformMode() or not self._plainClick(interactor):
            return
        x, y = interactor.GetEventPosition()
        if abs(x - start[0]) > CLICK_TOLERANCE_PX or abs(y - start[1]) > CLICK_TOLERANCE_PX:
            return  # a window/level drag, not a click
        # MPR (as in RadiAnt): a click moves the other slices to the clicked point.
        ras = self._xyToRas(name, x, y)
        crosshair = self._crosshairNode()
        if crosshair is not None:
            crosshair.SetCrosshairRAS(*ras)
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            slicer.mrmlScene, *ras, slicer.vtkMRMLSliceNode.OffsetJumpSlice, -1, layout.ensureSliceNode(name)
        )

    def _on3DPress(self, interactor, event):
        self._press["3D"] = interactor.GetEventPosition()

    def _on3DRelease(self, interactor, event):
        start = self._press.pop("3D", None)
        if start is None or not self.navigator.navigating or not self._viewTransformMode():
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
