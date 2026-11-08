"""Measurements on the perpendicular section and along the centerline (stage 4).

``MeasurementController`` connects the core (``aortaplan_core.crosssection``,
``centerline.markers``, ``centerline.angles``) with the scene:

- automatic lumen and outer-wall contours at the current position, shown as editable closed
  curves in the perpendicular view, with the min/max Feret segments and large numbers;
  editing a control point turns the contour into a manual one; edited sections are saved
  automatically when the position changes;
- saved measurements (``MeasurementStore``, JSON in the parameter node);
- named markers on the centerline (a markups node saved with the scene), shown in 3D and
  as ticks with names on both CPRs; distances along the centerline, the curvatures and
  straight; EVAR neck angles, neck length and iliac tortuosity;
- the diameter profile and the maximum diameter between two markers;
- length along the centerline from two clicks on a CPR; rulers.
"""

import json
import logging

import numpy as np
import qt
import slicer

from aortaplan_core.centerline.angles import neck_angles
from aortaplan_core.centerline.markers import MARKER_PRESETS, ON_PATH_TOLERANCE_MM, Marker, marker_distance
from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.crosssection import (
    LUMEN,
    OUTER,
    ContourParameters,
    MeasurementStore,
    SavedMeasurement,
    SectionMeasurement,
    SectionPlane,
    diameter_profile,
    find_maximum_diameter,
    measure_section,
)
from aortaplan_core.crosssection.lengths import curvature_lengths

from . import layout
from .cprview import _hidden
from .display import BOTTOM_LEFT, ViewOverlay

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


CONTOUR_NAMES = {LUMEN: "AortaPlan_Contour_Lumen", OUTER: "AortaPlan_Contour_Outer"}
CONTOUR_COLORS = {LUMEN: (1.0, 0.35, 0.35), OUTER: (0.35, 0.75, 1.0)}
FERET_COLORS = {LUMEN: (1.0, 0.65, 0.65), OUTER: (0.65, 0.85, 1.0)}
MARKERS_NODE_NAME = "AortaPlan_Markers"
MARKER_COLOR = (0.3, 0.85, 0.4)
TICKS_NODE_PREFIX = "AortaPlan_MarkerTicks_"
LENGTH_NODE_NAME = "AortaPlan_CprLength"
LENGTH_ATTRIBUTE = "AortaPlan.CprLength"
RULER_ATTRIBUTE = "AortaPlan.Ruler"
PATH_ATTRIBUTE = "AortaPlan.Path"
CONTROL_POINT_SPACING_MM = 4.0
UPDATE_DELAY_MS = 60  # contours follow after a short pause, so scrolling itself stays fast
HIT_TOLERANCE_PX = 8.0


def formatNumber(value, digits=1):
    """Russian-style decimal comma for display."""
    if value is None or not np.isfinite(value):
        return "—"
    text = f"{value:.{digits}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]  # no "-0"
    return text.replace(".", ",")


def pathKey(curveNode):
    """Stable name of a centerline path: its side for built paths, else the node name."""
    if curveNode is None:
        return ""
    return curveNode.GetAttribute(PATH_ATTRIBUTE) or curveNode.GetName() or curveNode.GetID()


def restrictToViews(displayNode, viewNodes):
    """Show a display node only in the given views (None entries are skipped)."""
    displayNode.RemoveAllViewNodeIDs()
    for viewNode in viewNodes:
        if viewNode is not None:
            displayNode.AddViewNodeID(viewNode.GetID())


def presetKeyForName(name):
    for key in MARKER_PRESETS:
        if tr(f"marker.{key}") == name:
            return key
    return None


class MeasurementController:
    def __init__(self, navigator):
        self.navigator = navigator
        self.store = MeasurementStore()
        self.params = ContourParameters()
        self.current = None  # SectionMeasurement shown at the current position
        self.currentName = None  # name of the saved measurement being shown
        self.dirty = False  # manual edits not saved yet
        self.live = True
        self.withOuter = True
        self.active = False
        self.listeners = []  # called after every change (the module refreshes its tables)
        self.persist = None  # callback(json text): keeps the store in the parameter node
        self.profile = None
        self.profileKey = None
        self.lengthClicks = []
        self.lengthResult = None
        self.maximumResult = None
        self.overlay = None
        self._shownKey = None
        self._settingNodes = False
        self._lastPath = None
        self._contourObservations = []
        self._markerObservations = []
        self._timer = qt.QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(UPDATE_DELAY_MS)
        self._timer.connect("timeout()", lambda: self.update())

    # ------------------------------------------------------------------ helpers
    @property
    def path(self):
        return self.navigator.path

    @property
    def key(self):
        return pathKey(self.navigator.curveNode)

    def volume(self):
        if self.navigator.volumeNode is None:
            return None
        from .centerline import volumeFromNode

        return volumeFromNode(self.navigator.volumeNode)

    def paths(self):
        """All centerline paths of the scene by key (built paths and the current curve)."""
        from .centerline import CenterlineBuilder

        out = {}
        for curve in CenterlineBuilder.pathCurves():
            if curve.GetNumberOfControlPoints() >= 2:
                try:
                    out[pathKey(curve)] = CurvePath(slicer.util.arrayFromMarkupsCurvePoints(curve, world=True))
                except ValueError:
                    pass
        if self.path is not None:
            out[self.key] = self.path
        return out

    def _notify(self):
        for listener in list(self.listeners):
            try:
                listener()
            except Exception:
                logging.exception("AortaPlan: measurement listener failed")

    def _persist(self):
        if self.persist is not None:
            self.store.markers = self.markers()
            self.persist(self.store.to_json())

    # ------------------------------------------------------------------ scene nodes
    def contourNode(self, kind, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(CONTOUR_NAMES[kind])
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode", CONTOUR_NAMES[kind]))
            node.CreateDefaultDisplayNodes()
            for i in range(node.GetNumberOfMeasurements()):  # no area/length computed by Slicer
                node.GetNthMeasurement(i).SetEnabled(False)
            display = node.GetDisplayNode()
            display.SetColor(*CONTOUR_COLORS[kind])
            display.SetSelectedColor(*CONTOUR_COLORS[kind])
            display.SetActiveColor(1.0, 1.0, 0.3)
            display.SetFillVisibility(False)
            display.SetPointLabelsVisibility(False)
            display.SetPropertiesLabelVisibility(False)
            display.SetGlyphTypeFromString("Circle2D")
            display.SetGlyphScale(1.6)
            display.SetLineThickness(0.25)
            restrictToViews(display, [self.navigator.crossNode])
            for event in (slicer.vtkMRMLMarkupsNode.PointModifiedEvent, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent):
                tag = node.AddObserver(event, lambda caller, ev, kind=kind: self._onContourEdited(kind, ev))
                self._contourObservations.append((node, tag))
        return node

    def feretNode(self, kind, which, create=True):
        name = f"AortaPlan_Feret_{kind}_{which}"
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name))
            node.AddControlPoint(0.0, 0.0, 0.0, "")
            node.AddControlPoint(1.0, 0.0, 0.0, "")
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            for i in range(node.GetNumberOfMeasurements()):
                node.GetNthMeasurement(i).SetEnabled(False)
            display = node.GetDisplayNode()
            display.SetColor(*FERET_COLORS[kind])
            display.SetSelectedColor(*FERET_COLORS[kind])
            display.SetPointLabelsVisibility(False)
            display.SetPropertiesLabelVisibility(False)
            display.SetGlyphScale(0.8)
            display.SetLineThickness(0.15)
            restrictToViews(display, [self.navigator.crossNode])
        return node

    def _setVisible(self, node, visible):
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(bool(visible))

    def hideSection(self):
        for kind in (LUMEN, OUTER):
            self._setVisible(self.contourNode(kind, create=False), False)
            for which in ("max", "min"):
                self._setVisible(self.feretNode(kind, which, create=False), False)

    def attachOverlay(self):
        self.detachOverlay()
        widget = layout.sliceWidget(layout.CROSS_VIEW)
        if widget is not None:
            self.overlay = ViewOverlay(widget.sliceView(), fontSize=18, corner=BOTTOM_LEFT, color=(1.0, 1.0, 1.0))
        self._updateOverlay()

    def detachOverlay(self):
        if self.overlay is not None:
            self.overlay.remove()
            self.overlay = None

    # ------------------------------------------------------------------ activation
    def setActive(self, active):
        """The measurement step is shown: contours follow the position."""
        self.active = bool(active)
        if self.active:
            self.update(force=True)
        else:
            self._autosave()
            self.hideSection()
            self._updateOverlay()

    def onNavigationChanged(self):
        if self.path is not self._lastPath:
            self._lastPath = self.path
            self._onPathChanged()
        if not self.active:
            return
        if self._positionKey() == self._shownKey:
            return  # e.g. only the CPR angle changed
        self._timer.start()

    def _positionKey(self):
        if self.path is None:
            return None
        return (self.key, round(self.navigator.position, 3), round(self.path.length, 3))

    def _onPathChanged(self):
        if self.profile is not None and self.profileKey != self._profileKey():
            self.profile = None
        self.updateMarkerTicks()
        self._notify()

    def _profileKey(self):
        return (self.key, round(self.path.length, 1)) if self.path is not None else None

    # ------------------------------------------------------------------ section
    def update(self, force=False):
        """Show the saved or automatic contours at the current position."""
        self._timer.stop()
        if not self.active:
            return
        key = self._positionKey()
        if not force and key is not None and key == self._shownKey and self.current is not None:
            return  # already shown (e.g. the maximum search showed its own contours)
        self._autosave()
        self._shownKey = key
        volume = self.volume()
        if key is None or volume is None:
            self.current = None
            self.currentName = None
            self.hideSection()
            self._updateOverlay()
            self._notify()
            return
        s = self.navigator.position
        index = self.store.find(self.key, s)
        if index is not None:
            saved = self.store.measurements[index]
            self._show(saved.measurement, saved.name)
        elif self.live:
            self._show(self._measure(volume, s), None)
        else:
            self.current = None
            self.currentName = None
            self.hideSection()
        self._updateOverlay()
        self._notify()

    def _measure(self, volume, s):
        plane = SectionPlane.at(self.path, s)
        return measure_section(volume, plane, self.params, with_outer=self.withOuter)

    def recompute(self):
        """Automatic contours again at the current position (manual edits are discarded)."""
        volume = self.volume()
        if self.path is None or volume is None:
            return
        self.dirty = False
        index = self.store.find(self.key, self.navigator.position)
        if index is not None:
            del self.store.measurements[index]
            self._persist()
        self._show(self._measure(volume, self.navigator.position), None)
        self._shownKey = self._positionKey()
        self._updateOverlay()
        self._notify()

    def _show(self, measurement, name):
        self.current = measurement
        self.currentName = name
        self.dirty = False
        self._settingNodes = True
        try:
            for kind in (LUMEN, OUTER):
                contour = measurement.lumen if kind == LUMEN else measurement.outer
                node = self.contourNode(kind)
                if contour is None:
                    self._setVisible(node, False)
                    self._setFeret(kind, None)
                    continue
                count = int(np.clip(round(contour.metrics().perimeter_mm / CONTROL_POINT_SPACING_MM), 12, 48))
                points = contour.plane.to_ras(contour.resampled(count))
                slicer.util.updateMarkupsControlPointsFromArray(node, points, world=True)
                self._setVisible(node, True)
                self._setFeret(kind, contour.metrics_ras())
        finally:
            self._settingNodes = False

    def _setFeret(self, kind, metrics):
        for which in ("max", "min"):
            node = self.feretNode(kind, which, create=metrics is not None)
            if node is None:
                continue
            if metrics is None:
                self._setVisible(node, False)
                continue
            segment = metrics.max_feret_segment if which == "max" else metrics.min_feret_segment
            node.SetNthControlPointPositionWorld(0, *segment[0])
            node.SetNthControlPointPositionWorld(1, *segment[1])
            self._setVisible(node, True)

    def _onContourEdited(self, kind, event):
        if self._settingNodes or self.current is None:
            return
        contour = self.current.lumen if kind == LUMEN else self.current.outer
        node = self.contourNode(kind, create=False)
        if contour is None or node is None or node.GetNumberOfControlPoints() < 3:
            return
        points = slicer.util.arrayFromMarkupsCurvePoints(node, world=True)
        edited = contour.edited(points)
        if kind == LUMEN:
            self.current.lumen = edited
        else:
            self.current.outer = edited
        self.dirty = True
        self._settingNodes = True
        try:
            self._setFeret(kind, edited.metrics_ras())
        finally:
            self._settingNodes = False
        self._updateOverlay()
        self._notify()

    def _autosave(self):
        if self.dirty and self.current is not None:
            self.saveCurrent(self.currentName)

    def defaultName(self, s):
        for marker in self.markers():
            if marker.arc_length_mm is not None and abs(marker.arc_length_mm - s) <= 1.0:
                return marker.name
        return tr("measure.saved.default_name", position=formatNumber(s))

    def saveCurrent(self, name=None):
        if self.current is None:
            return None
        s = self.current.arc_length_mm if self.current.arc_length_mm is not None else self.navigator.position
        item = SavedMeasurement(name or self.defaultName(s), self.key, self.current)
        index = self.store.save(item)
        self.currentName = item.name
        self.dirty = False
        self._persist()
        self._notify()
        return index

    def deleteMeasurement(self, index):
        if 0 <= index < len(self.store.measurements):
            removed = self.store.measurements.pop(index)
            self._persist()
            if self.currentName == removed.name:
                self.update(force=True)
            self._notify()

    def renameMeasurement(self, index, name):
        if 0 <= index < len(self.store.measurements) and name:
            self.store.measurements[index].name = name
            self._persist()
            self._notify()

    def jumpToMeasurement(self, index):
        if 0 <= index < len(self.store.measurements):
            item = self.store.measurements[index]
            if item.path == self.key:
                self.navigator.setPosition(item.arc_length_mm)

    def sectionMetrics(self):
        """{"lumen": ContourMetrics or None, "outer": ...} of the shown section."""
        if self.current is None:
            return {LUMEN: None, OUTER: None}
        return self.current.metrics()

    def sectionWarnings(self):
        if self.current is None:
            return []
        warnings = list(self.current.warnings)
        for contour in (self.current.lumen, self.current.outer):
            if contour is not None and contour.source == "auto":
                warnings += contour.warnings
        return warnings

    def _updateOverlay(self):
        if self.overlay is None:
            return
        if not self.active or self.current is None:
            self.overlay.setText("")
            return
        lines = []
        metrics = self.sectionMetrics()
        if metrics[LUMEN] is not None:
            m = metrics[LUMEN]
            lines.append(tr("measure.overlay.lumen", min=formatNumber(m.min_diameter_mm), max=formatNumber(m.max_diameter_mm),
                            mean=formatNumber(m.mean_diameter_mm), area=formatNumber(m.area_mm2, 0)))
        if metrics[OUTER] is not None:
            m = metrics[OUTER]
            lines.append(tr("measure.overlay.outer", min=formatNumber(m.min_diameter_mm), max=formatNumber(m.max_diameter_mm),
                            mean=formatNumber(m.mean_diameter_mm)))
        self.overlay.setText("\n".join(lines))

    def isOverContourPoint(self, viewName, x, y):
        """A contour control point is under the mouse in the perpendicular view."""
        if viewName != layout.CROSS_VIEW or not self.active:
            return False
        import vtk

        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(self.navigator.crossNode.GetXYToRAS(), rasToXy)
        for kind in (LUMEN, OUTER):
            node = self.contourNode(kind, create=False)
            if node is None or node.GetDisplayNode() is None or not node.GetDisplayNode().GetVisibility():
                continue
            for p in slicer.util.arrayFromMarkupsControlPoints(node, world=True):
                xy = rasToXy.MultiplyPoint([*p, 1.0])
                if abs(xy[0] - x) <= HIT_TOLERANCE_PX and abs(xy[1] - y) <= HIT_TOLERANCE_PX:
                    return True
        return False

    # ------------------------------------------------------------------ markers
    def markersNode(self, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(MARKERS_NODE_NAME)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", MARKERS_NODE_NAME)
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*MARKER_COLOR)
            display.SetSelectedColor(*MARKER_COLOR)
            display.SetGlyphTypeFromString("Sphere3D")
            display.SetGlyphScale(2.0)
            display.SetTextScale(3.0)
            threeD = layout.threeDWidget()
            if threeD is not None:
                restrictToViews(display, [threeD.mrmlViewNode()])
        if node is not None and not self._markerObservations:
            for event in (slicer.vtkMRMLMarkupsNode.PointAddedEvent, slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                          slicer.vtkMRMLMarkupsNode.PointModifiedEvent):
                self._markerObservations.append((node, node.AddObserver(event, self._onMarkersModified)))
        return node

    def _onMarkersModified(self, caller, event):
        if self._settingNodes:
            return
        self.updateMarkerTicks()
        self._notify()

    def markers(self):
        """Markers with their position on the current path (``arc_length_mm`` None if off it)."""
        node = self.markersNode(create=False)
        out = []
        if node is None:
            return out
        path = self.path
        points = slicer.util.arrayFromMarkupsControlPoints(node, world=True) if node.GetNumberOfControlPoints() else []
        for i in range(node.GetNumberOfControlPoints()):
            point = np.asarray(points[i], dtype=float)
            try:
                info = json.loads(node.GetNthControlPointDescription(i) or "{}")
            except ValueError:
                info = {}
            s = None
            if path is not None:
                s_on, distance = path.nearest_arc_length(point)
                s = s_on if distance <= ON_PATH_TOLERANCE_MM else None
            out.append(Marker(node.GetNthControlPointLabel(i), point, info.get("path", ""), s, info.get("key")))
        return out

    def addMarker(self, name, s=None):
        """Marker ``name`` at arc length ``s`` (default: the current position); returns its index."""
        if self.path is None or not name:
            return None
        s = self.navigator.position if s is None else self.path.clamp(s)
        node = self.markersNode()
        point = self.path.point_at(s)
        self._settingNodes = True
        try:
            index = node.AddControlPoint(*point, name)
            node.SetNthControlPointDescription(index, json.dumps({"key": presetKeyForName(name), "path": self.key}))
        finally:
            self._settingNodes = False
        self._afterMarkerChange()
        return index

    def setMarker(self, key, s):
        """Place the preset marker ``key`` at ``s``, moving it if it is already on this path."""
        name = tr(f"marker.{key}")
        for i, marker in enumerate(self.markers()):
            if marker.key == key and marker.arc_length_mm is not None:
                self.moveMarker(i, s)
                return i
        return self.addMarker(name, s)

    def deleteMarker(self, index):
        node = self.markersNode(create=False)
        if node is not None and 0 <= index < node.GetNumberOfControlPoints():
            self._settingNodes = True
            try:
                node.RemoveNthControlPoint(index)
            finally:
                self._settingNodes = False
            self._afterMarkerChange()

    def renameMarker(self, index, name):
        node = self.markersNode(create=False)
        if node is None or not name or not (0 <= index < node.GetNumberOfControlPoints()):
            return
        self._settingNodes = True
        try:
            node.SetNthControlPointLabel(index, name)
            info = json.loads(node.GetNthControlPointDescription(index) or "{}")
            info["key"] = presetKeyForName(name) or info.get("key")
            node.SetNthControlPointDescription(index, json.dumps(info))
        finally:
            self._settingNodes = False
        self._afterMarkerChange()

    def moveMarker(self, index, s=None):
        node = self.markersNode(create=False)
        if node is None or self.path is None or not (0 <= index < node.GetNumberOfControlPoints()):
            return
        s = self.navigator.position if s is None else self.path.clamp(s)
        self._settingNodes = True
        try:
            node.SetNthControlPointPositionWorld(index, *self.path.point_at(s))
        finally:
            self._settingNodes = False
        self._afterMarkerChange()

    def jumpToMarker(self, index):
        markers = self.markers()
        if 0 <= index < len(markers) and markers[index].arc_length_mm is not None:
            self.navigator.setPosition(markers[index].arc_length_mm)

    def _afterMarkerChange(self):
        self.updateMarkerTicks()
        self._persist()
        self._notify()

    def tickNode(self, view, create=True):
        name = TICKS_NODE_PREFIX + view.layoutName
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name))
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*MARKER_COLOR)
            display.SetSelectedColor(*MARKER_COLOR)
            display.SetGlyphTypeFromString("Dash2D")
            display.SetGlyphScale(4.0)
            display.SetTextScale(2.5)
            restrictToViews(display, [view.sliceNode])
        return node

    def updateMarkerTicks(self):
        """Ticks with names at the left edge of both CPRs, at the markers on the current path."""
        markers = [m for m in self.markers() if m.arc_length_mm is not None] if self.path is not None else []
        for view in self.navigator.cprViews:
            node = self.tickNode(view, create=bool(markers))
            if node is None:
                continue
            node.RemoveAllControlPoints()
            if view.cpr is None:
                continue
            offset = -view.cpr.half_width_mm + 2.0
            for marker in markers:
                column, row = view.cpr.cpr_to_pixel(offset, marker.arc_length_mm)
                node.AddControlPoint(*view.pixelToRas(column, row), marker.name)

    def distance(self, i, j, withCurvature=True):
        """(core Distance, CurvatureLengths or None) between markers i and j."""
        markers = self.markers()
        a, b = markers[i], markers[j]
        result = marker_distance(a, b, self.paths())
        curvature = None
        volume = self.volume()
        if withCurvature and volume is not None and a.arc_length_mm is not None and b.arc_length_mm is not None:
            try:
                curvature = curvature_lengths(volume, self.path, a.arc_length_mm, b.arc_length_mm)
            except Exception:
                logging.exception("AortaPlan: curvature lengths failed")
        return result, curvature

    def distanceMatrix(self):
        from aortaplan_core.centerline.markers import distance_table

        return distance_table(self.markers(), self.paths())

    # ------------------------------------------------------------------ EVAR
    def evarSummary(self):
        """Lines of text for the EVAR section (neck angles, neck length, tortuosity)."""
        lines = []
        byKey = {}
        for marker in self.markers():
            if marker.key and marker.arc_length_mm is not None:
                byKey.setdefault(marker.key, marker)
        needed = ("lowest_renal", "neck_end", "aortic_bifurcation")
        missing = [tr(f"marker.{k}") for k in needed if k not in byKey]
        if self.path is None:
            return [tr("measure.status.no_path")]
        if missing:
            lines.append(tr("evar.need", names=", ".join(missing)))
        else:
            try:
                angles = neck_angles(self.path, *(byKey[k].arc_length_mm for k in needed))
                lines.append(tr("evar.alpha", value=formatNumber(angles.alpha_deg)))
                lines.append(tr("evar.beta", value=formatNumber(angles.beta_deg)))
                lines.append(tr("evar.neck_length",
                                value=formatNumber(byKey["neck_end"].arc_length_mm - byKey["lowest_renal"].arc_length_mm)))
            except ValueError as exc:
                lines.append(str(exc))
        lines += self._tortuosityLines()
        return lines

    def _tortuosityLines(self):
        from aortaplan_core.centerline.markers import locate

        lines = []
        allMarkers = self.markers()
        bifurcation = next((m for m in allMarkers if m.key == "aortic_bifurcation"), None)
        if bifurcation is None:
            return lines
        paths = self.paths()
        for side, femoralKey in (("right", "right_femoral"), ("left", "left_femoral")):
            path = paths.get(side)
            if path is None:
                continue
            on = {name: s for name, s, _ in locate(bifurcation.point, {side: path})}
            if side not in on:
                continue
            s0 = on[side]
            s1 = path.length
            femoral = next((m for m in allMarkers if m.key == femoralKey), None)
            if femoral is not None:
                s_f, d = path.nearest_arc_length(femoral.point)
                if d <= ON_PATH_TOLERANCE_MM:
                    s1 = s_f
            if s1 - s0 < 5.0:
                continue
            chord = float(np.linalg.norm(path.point_at(s1) - path.point_at(s0)))
            lines.append(tr("evar.tortuosity", side=tr(f"evar.side.{side}"), value=formatNumber((s1 - s0) / chord, 2),
                            along=formatNumber(s1 - s0), straight=formatNumber(chord)))
        return lines

    # ------------------------------------------------------------------ profile and maximum
    def computeProfile(self, progress=None):
        volume = self.volume()
        if self.path is None or volume is None:
            return None
        self.profile = diameter_profile(volume, self.path, 1.0, with_outer=self.withOuter, progress=progress)
        self.profileKey = self._profileKey()
        self._notify()
        return self.profile

    def findMaximum(self, s0, s1, progress=None):
        volume = self.volume()
        if self.path is None or volume is None:
            return None
        layer = OUTER if self.withOuter else LUMEN
        result = find_maximum_diameter(volume, self.path, s0, s1, layer=layer, params=self.params, progress=progress)
        self.maximumResult = result
        if result is not None:
            self.setMarker("max_diameter", result.arc_length_mm)
            self.navigator.setPosition(result.arc_length_mm)
            if self.active:
                self._show(result.measurement, None)
                self._shownKey = self._positionKey()
                self._updateOverlay()
        self._notify()
        return result

    # ------------------------------------------------------------------ CPR tools
    def lengthNode(self, view, create=True):
        """Line of the length tool in a CPR view (found by attribute: its name shows the length)."""
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetAttribute(LENGTH_ATTRIBUTE) == view.layoutName:
                return node
        node = None
        if create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", LENGTH_NODE_NAME))
            node.SetAttribute(LENGTH_ATTRIBUTE, view.layoutName)
            node.AddControlPoint(0.0, 0.0, 0.0, "")
            node.AddControlPoint(1.0, 0.0, 0.0, "")
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            for i in range(node.GetNumberOfMeasurements()):
                node.GetNthMeasurement(i).SetEnabled(False)
            display = node.GetDisplayNode()
            display.SetColor(1.0, 1.0, 0.4)
            display.SetSelectedColor(1.0, 1.0, 0.4)
            display.SetPointLabelsVisibility(False)
            display.SetPropertiesLabelVisibility(True)
            display.SetTextScale(3.0)
            restrictToViews(display, [view.sliceNode])
        return node

    def startLength(self):
        self.lengthClicks = []
        self.lengthResult = None
        for view in self.navigator.cprViews:
            self._setVisible(self.lengthNode(view, create=False), False)
        self._notify()

    def lengthClick(self, viewIndex, offset, s):
        """One click of the length tool; returns True when both points are set."""
        self.lengthClicks.append((viewIndex, float(offset), float(s)))
        if len(self.lengthClicks) < 2:
            self._notify()
            return False
        (index, offset0, s0), (_, _, s1) = self.lengthClicks[:2]
        a, b = sorted((s0, s1))
        self.lengthResult = (a, b, b - a)
        view = self.navigator.cprViews[index]
        node = self.lengthNode(view)
        if view.cpr is not None:
            start = view.pixelToRas(*view.cpr.cpr_to_pixel(offset0, a))
            end = view.pixelToRas(*view.cpr.cpr_to_pixel(offset0, b))
            node.SetNthControlPointPositionWorld(0, *start)
            node.SetNthControlPointPositionWorld(1, *end)
            node.SetName(f"{formatNumber(b - a)} {tr('units.mm')}")
            self._setVisible(node, True)
        self._notify()
        return True

    def addRuler(self):
        """A markups line placed by the user in any view (Slicer shows its length)."""
        count = len([n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode") if n.GetAttribute(RULER_ATTRIBUTE)])
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", f"{tr('tools.ruler')} {count + 1}")
        node.SetAttribute(RULER_ATTRIBUTE, "1")
        node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetColor(1.0, 0.9, 0.2)
        node.GetDisplayNode().SetSelectedColor(1.0, 0.9, 0.2)
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
        selectionNode.SetActivePlaceNodeID(node.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        return node

    @staticmethod
    def clearRulers():
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetAttribute(RULER_ATTRIBUTE):
                slicer.mrmlScene.RemoveNode(node)

    # ------------------------------------------------------------------ persistence
    def loadJson(self, text):
        try:
            self.store = MeasurementStore.from_json(text or "")
        except Exception:
            logging.exception("AortaPlan: cannot read the saved measurements")
            self.store = MeasurementStore()
        self._shownKey = None
        self.profile = None
        self.updateMarkerTicks()
        if self.active:
            self.update(force=True)
        self._notify()

    def reset(self):
        """The scene is closed: forget everything (the nodes go with the scene)."""
        self._timer.stop()
        for node, tag in self._contourObservations + self._markerObservations:
            node.RemoveObserver(tag)
        self._contourObservations = []
        self._markerObservations = []
        self.store = MeasurementStore()
        self.current = None
        self.currentName = None
        self.dirty = False
        self.profile = None
        self.lengthClicks = []
        self.lengthResult = None
        self.maximumResult = None
        self._shownKey = None
        self._lastPath = None

    def cleanup(self):
        self.reset()
        self.detachOverlay()
        self.listeners = []


def currentSectionMeasurement(controller):
    """The section shown (for tests)."""
    return controller.current if isinstance(controller.current, SectionMeasurement) else None
