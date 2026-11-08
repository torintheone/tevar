"""Ostia of the aortic branches in Slicer (stage 5).

``OstiaController`` keeps the ostia in the measurement store (``MeasurementStore.ostia``,
saved in the parameter node as JSON) and shows them:

- points with names in 3D and in the perpendicular view (an unconfirmed name ends with "?");
- branch centerlines in 3D (one model);
- ticks with names at the right edge of both CPRs;
- a "clock hand" from the section centre to the ostium in the perpendicular view when the
  current position is at an ostium.

Search: ``aortaplan_core.ostia.find_ostia`` along the current path, also using the other
centerline paths (stage 2 branches). Manual ostia: one click on the aortic wall and an
optional second click inside the branch (markups placement; clicks on a CPR are mapped
back to the patient).
"""

import json
import logging

import numpy as np
import slicer
import vtk

from aortaplan_core.centerline.navigation import CurvePath
from aortaplan_core.ostia import (
    CLOCK_MODES,
    DEFAULT_GRAFT_DIAMETER_MM,
    MODE_ANTERIOR,
    OSTIUM_KEYS,
    AorticTube,
    OstiumParameters,
    describe,
    distal_edge_arc_length,
    find_ostia,
    ishimaru_zones,
    manual_ostium,
    measure_branch,
    ostium_name,
    propose_labels,
    zone_diameters,
)
from aortaplan_core.ostia.model import SOURCE_MANUAL

from . import layout
from .cprview import CPR_ORIGINS, _hidden
from .measurements import formatNumber, pathKey, restrictToViews

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


OSTIA_NODE_NAME = "AortaPlan_Ostia"
BRANCHES_NODE_NAME = "AortaPlan_Branches"
CLICKS_NODE_NAME = "AortaPlan_OstiumClicks"
TICKS_PREFIX = "AortaPlan_OstiumTicks_"
CLOCK_HAND_NAME = "AortaPlan_ClockHand"
CLOCK_HAND_ATTRIBUTE = "AortaPlan.ClockHand"
OSTIUM_COLOR = (1.0, 0.6, 0.15)
CLOCK_HAND_RANGE_MM = 2.0
MAX_DISTANCE_TO_PATH_MM = 45.0  # an ostium farther from the current path belongs to another one
#: Ostium keys -> marker keys of stage 4 ("lowest_renal" is derived from the renal ostia).
MARKER_KEYS = {"celiac": "celiac", "sma": "sma", "right_renal": "right_renal", "left_renal": "left_renal",
               "bca": "bca", "lcca": "lcca", "lsa": "lsa"}


def keyForName(name):
    for key in OSTIUM_KEYS:
        if ostium_name(key) == name:
            return key
    return None


class OstiaController:
    def __init__(self, navigator, measurements):
        self.navigator = navigator
        self.measurements = measurements  # MeasurementController (store, markers, persistence)
        self.listeners = []
        self.tube = None
        self._tubeKey = None
        self.zones = []
        self.clicks = []
        self._clickObservation = None
        self._settingNodes = False
        self.lastStumps = []

    # ------------------------------------------------------------------ settings
    @property
    def store(self):
        return self.measurements.store

    @property
    def ostia(self):
        return self.store.ostia

    def setting(self, name, default):
        return self.store.ostia_settings.get(name, default)

    def setSetting(self, name, value):
        if self.store.ostia_settings.get(name) == value:
            return
        self.store.ostia_settings[name] = value
        if name == "diameter_distance_mm":
            volume = self.measurements.volume()
            if volume is not None:
                for o in self.ostia:
                    measure_branch(volume, o, float(value))
        self.measurements._persist()
        self.refresh()

    @property
    def clockMode(self):
        mode = self.setting("clock_mode", MODE_ANTERIOR)
        return mode if mode in CLOCK_MODES else MODE_ANTERIOR

    @property
    def graftDiameter(self):
        return float(self.setting("graft_diameter_mm", DEFAULT_GRAFT_DIAMETER_MM))

    def parameters(self):
        return OstiumParameters(
            min_branch_diameter_mm=float(self.setting("min_branch_diameter_mm", 2.5)),
            diameter_distance_mm=float(self.setting("diameter_distance_mm", 5.0)),
        )

    # ------------------------------------------------------------------ rows
    def referenceArcLength(self):
        name = self.setting("reference_marker", "")
        if name:
            for marker in self.measurements.markers():
                if marker.name == name and marker.arc_length_mm is not None:
                    return marker.arc_length_mm
        return 0.0

    def rows(self):
        """[(index in the store, Ostium, OstiumPosition)] proximal -> distal on the current path."""
        path = self.navigator.path
        if path is None:
            return []
        out = []
        for i, o in enumerate(self.ostia):
            try:
                position = describe(o, path, self.clockMode, self.graftDiameter)
            except Exception:
                logging.exception("AortaPlan: cannot describe an ostium")
                continue
            if position.distance_to_centerline_mm <= MAX_DISTANCE_TO_PATH_MM:  # ostia of other paths are skipped
                out.append((i, o, position))
        out.sort(key=lambda row: row[2].arc_length_mm)
        return out

    def displayName(self, ostium):
        name = ostium.name or ostium_name(ostium.key)
        return name if ostium.confirmed else tr("ostia.unconfirmed", name=name)

    # ------------------------------------------------------------------ search
    def _ensureTube(self, progress=None):
        path, volume = self.navigator.path, self.measurements.volume()
        if path is None or volume is None:
            return None
        key = (pathKey(self.navigator.curveNode), round(path.length, 2), id(self.navigator.volumeNode))
        if self.tube is None or self._tubeKey != key:
            self.tube = AorticTube(volume, path, progress=progress)
            self._tubeKey = key
        return self.tube

    def otherPaths(self):
        from .centerline import CenterlineBuilder

        current = self.navigator.curveNode
        out = []
        for curve in CenterlineBuilder.pathCurves():
            if current is not None and curve.GetID() == current.GetID():
                continue
            if curve.GetNumberOfControlPoints() >= 2:
                out.append(CurvePath(slicer.util.arrayFromMarkupsCurvePoints(curve, world=True)))
        return out

    def aorticDiameter(self):
        if self.tube is None or not self.tube.valid.any():
            return None
        return 2.0 * float(np.nanmedian(self.tube.radii[self.tube.valid]))

    def find(self, progress=None):
        """Search along the current path; manual and confirmed ostia are kept."""
        path, volume = self.navigator.path, self.measurements.volume()
        if path is None or volume is None:
            return []
        tube = self._ensureTube(None if progress is None else (lambda f, s: progress(0.4 * f, s)))
        params = self.parameters()
        found, _, stumps = find_ostia(
            volume, path, params, tube=tube, other_paths=self.otherPaths(),
            progress=None if progress is None else (lambda f, s: progress(0.4 + 0.6 * f, s)),
        )
        self.lastStumps = stumps
        kept = [o for o in self.ostia if o.confirmed or o.source == SOURCE_MANUAL]
        for o in found:
            if all(np.linalg.norm(o.center - k.center) > params.merge_distance_mm for k in kept):
                kept.append(o)
        self.store.ostia = kept
        propose_labels(self.store.ostia, path, self.aorticDiameter())
        self._changed()
        return self.store.ostia

    # ------------------------------------------------------------------ editing
    def rename(self, index, name, confirm=True):
        """New name (a preset name sets the anatomical key; a custom name has none)."""
        if not (0 <= index < len(self.ostia)) or not name:
            return
        o = self.ostia[index]
        o.name = name
        o.key = keyForName(name)
        if confirm:
            o.confirmed = True
        self._changed()

    def setConfirmed(self, index, confirmed):
        if 0 <= index < len(self.ostia):
            self.ostia[index].confirmed = bool(confirmed)
            self._changed()

    def confirmAll(self):
        for o in self.ostia:
            o.confirmed = True
        self._changed()

    def setStenosis(self, index, stenosis):
        if 0 <= index < len(self.ostia):
            self.ostia[index].stenosis = bool(stenosis)
            self._changed()

    def delete(self, index):
        if 0 <= index < len(self.ostia):
            del self.ostia[index]
            self._changed()

    def jump(self, index):
        """Go to the ostium; the first CPR turns so that the branch lies in its plane."""
        path = self.navigator.path
        if path is None or not (0 <= index < len(self.ostia)):
            return
        o = self.ostia[index]
        position = describe(o, path, self.clockMode, self.graftDiameter)
        self.navigator.setPosition(position.arc_length_mm)
        if self.navigator.cpr is not None:
            target = o.branch_points[min(len(o.branch_points) - 1, 6)] if len(o.branch_points) else o.center
            angle = self.navigator.cpr.angle_through(target)
            if angle is not None:
                self.navigator.setAngle(angle)

    def addManual(self, wallPoint, branchPoint=None):
        tube, volume = self._ensureTube(), self.measurements.volume()
        if tube is None:
            return None
        o = manual_ostium(volume, tube, wallPoint, branchPoint, self.parameters())
        # The name is proposed like for found ostia and still needs the user's confirmation.
        o.confirmed, o.key, o.name = False, None, ""
        self.store.ostia.append(o)
        propose_labels(self.store.ostia, self.navigator.path, self.aorticDiameter())
        self._changed()
        return o

    def markersFromOstia(self):
        """Stage 4 markers at the ostia (celiac, SMA, renals, lowest renal, arch vessels)."""
        path = self.navigator.path
        if path is None:
            return 0
        count = 0
        lowest = None
        for o in self.ostia:
            if o.key not in MARKER_KEYS:
                continue
            position = describe(o, path, self.clockMode, self.graftDiameter)
            self.measurements.setMarker(MARKER_KEYS[o.key], position.arc_length_mm)
            count += 1
            if o.key in ("right_renal", "left_renal"):
                edge = distal_edge_arc_length(o, position)
                lowest = edge if lowest is None else max(lowest, edge)
        if lowest is not None:
            self.measurements.setMarker("lowest_renal", lowest)
            count += 1
        return count

    # ------------------------------------------------------------------ zones
    def computeZones(self, withDiameters=False, progress=None):
        path = self.navigator.path
        if path is None:
            self.zones = []
            return self.zones
        self.zones = ishimaru_zones(path, self.ostia)
        if withDiameters and self.zones:
            volume = self.measurements.volume()
            if volume is not None:
                for i, zone in enumerate(self.zones):
                    if progress is not None and progress(i / len(self.zones), "ostia.zones") is False:
                        break
                    zone_diameters(volume, path, [zone], with_outer=self.measurements.withOuter)
        return self.zones

    # ------------------------------------------------------------------ manual clicks
    def clicksNode(self, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(CLICKS_NODE_NAME)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", CLICKS_NODE_NAME))
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetColor(*OSTIUM_COLOR)
            node.GetDisplayNode().SetSelectedColor(*OSTIUM_COLOR)
            node.GetDisplayNode().SetPointLabelsVisibility(False)
        return node

    def startClicks(self):
        node = self.clicksNode()
        node.RemoveAllControlPoints()
        self.clicks = []
        if self._clickObservation is None:
            tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self._onClick)
            self._clickObservation = (node, tag)
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selection.SetActivePlaceNodeID(node.GetID())
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        interaction.SetPlaceModePersistence(1)
        interaction.SetCurrentInteractionMode(interaction.Place)

    def patientPoint(self, ras):
        """A clicked point; points in a CPR plane are mapped back to the patient."""
        p = np.asarray(ras, dtype=float)
        for view in self.navigator.cprViews:
            origin = CPR_ORIGINS[view.layoutName]
            if view.cpr is not None and abs(p[2] - origin[2]) < 1.0:
                column, row = view.rasToPixel(p)
                offset, s = view.cpr.pixel_to_cpr(column, row)
                return view.cpr.to_ras(offset, s, self.navigator.angle + view.angleOffset)
        return p

    def _onClick(self, caller, event):
        node = self.clicksNode(create=False)
        if node is None or node.GetNumberOfControlPoints() == 0:
            return
        points = slicer.util.arrayFromMarkupsControlPoints(node, world=True)
        self.clicks = [self.patientPoint(p) for p in points]
        if len(self.clicks) >= 2:
            self.finishClicks()
        else:
            self._notify()

    def finishClicks(self):
        """Create the ostium from the clicks so far (1 or 2) and leave the placement mode."""
        clicks = list(self.clicks)
        self.stopClicks()
        if not clicks:
            return None
        try:
            return self.addManual(clicks[0], clicks[1] if len(clicks) > 1 else None)
        except ValueError as exc:
            logging.warning("AortaPlan: manual ostium: %s", exc)
            return None

    def stopClicks(self):
        self.clicks = []
        if self._clickObservation is not None:
            node, tag = self._clickObservation
            node.RemoveObserver(tag)
            self._clickObservation = None
        node = self.clicksNode(create=False)
        if node is not None:
            node.RemoveAllControlPoints()
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        if interaction.GetCurrentInteractionMode() == interaction.Place:
            interaction.SetCurrentInteractionMode(interaction.ViewTransform)
        self._notify()

    # ------------------------------------------------------------------ display
    def ostiaNode(self, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(OSTIA_NODE_NAME)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", OSTIA_NODE_NAME))
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*OSTIUM_COLOR)
            display.SetSelectedColor(*OSTIUM_COLOR)
            display.SetGlyphTypeFromString("Sphere3D")
            display.SetGlyphScale(2.0)
            display.SetTextScale(3.0)
            threeD = layout.threeDWidget()
            restrictToViews(display, [threeD.mrmlViewNode() if threeD is not None else None, self.navigator.crossNode])
        return node

    def branchesNode(self, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(BRANCHES_NODE_NAME)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", BRANCHES_NODE_NAME))
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*OSTIUM_COLOR)
            display.SetLineWidth(3.0)
            display.SetVisibility2D(False)
        return node

    def tickNode(self, view, create=True):
        name = TICKS_PREFIX + view.layoutName
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None and create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name))
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetColor(*OSTIUM_COLOR)
            display.SetSelectedColor(*OSTIUM_COLOR)
            display.SetGlyphTypeFromString("Dash2D")
            display.SetGlyphScale(4.0)
            display.SetTextScale(2.5)
            restrictToViews(display, [view.sliceNode])
        return node

    def clockHandNode(self, create=True):
        """The clock hand line (found by attribute: its name is the label shown)."""
        for node in slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode"):
            if node.GetAttribute(CLOCK_HAND_ATTRIBUTE):
                return node
        node = None
        if create:
            node = _hidden(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", CLOCK_HAND_NAME))
            node.SetAttribute(CLOCK_HAND_ATTRIBUTE, "1")
            node.AddControlPoint(0.0, 0.0, 0.0, "")
            node.AddControlPoint(1.0, 0.0, 0.0, "")
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            for i in range(node.GetNumberOfMeasurements()):
                node.GetNthMeasurement(i).SetEnabled(False)
            display = node.GetDisplayNode()
            display.SetColor(*OSTIUM_COLOR)
            display.SetSelectedColor(*OSTIUM_COLOR)
            display.SetPointLabelsVisibility(False)
            display.SetPropertiesLabelVisibility(True)
            display.SetTextScale(3.0)
            restrictToViews(display, [self.navigator.crossNode])
        return node

    def updateDisplay(self):
        rows = self.rows()
        self._settingNodes = True
        try:
            node = self.ostiaNode(create=bool(rows))
            if node is not None:
                node.RemoveAllControlPoints()
                for _, o, _ in rows:
                    index = node.AddControlPoint(*o.center, self.displayName(o))
                    node.SetNthControlPointDescription(index, json.dumps({"key": o.key}))
            self._updateBranches(rows)
            self._updateTicks(rows)
            self.updateClockHand(rows)
        finally:
            self._settingNodes = False

    def _updateBranches(self, rows):
        node = self.branchesNode(create=bool(rows))
        if node is None:
            return
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for _, o, _ in rows:
            chain = np.vstack([o.center[np.newaxis, :], o.branch_points]) if len(o.branch_points) else None
            if chain is None or len(chain) < 2:
                continue
            ids = []
            for p in chain:
                ids.append(points.InsertNextPoint(*p))
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(ids))
            for k, pid in enumerate(ids):
                line.GetPointIds().SetId(k, pid)
            lines.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        node.SetAndObservePolyData(poly)

    def _updateTicks(self, rows):
        for view in self.navigator.cprViews:
            node = self.tickNode(view, create=bool(rows))
            if node is None:
                continue
            node.RemoveAllControlPoints()
            if view.cpr is None:
                continue
            offset = view.cpr.half_width_mm - 2.0
            for _, o, position in rows:
                column, row = view.cpr.cpr_to_pixel(offset, position.arc_length_mm)
                node.AddControlPoint(*view.pixelToRas(column, row), self.displayName(o))

    def updateClockHand(self, rows=None):
        """Line from the section centre to the ostium when the position is at an ostium."""
        rows = self.rows() if rows is None else rows
        node = self.clockHandNode(create=bool(rows))
        if node is None:
            return
        path = self.navigator.path
        near = None
        if path is not None:
            for _, o, position in rows:
                if abs(position.arc_length_mm - self.navigator.position) <= CLOCK_HAND_RANGE_MM:
                    if near is None or abs(position.arc_length_mm - self.navigator.position) < abs(near[1].arc_length_mm - self.navigator.position):
                        near = (o, position)
        display = node.GetDisplayNode()
        if near is None or not self.navigator.navigating:
            display.SetVisibility(False)
            return
        o, position = near
        frame = path.cross_section_frame(self.navigator.position)
        centre = frame.origin
        target = o.center - np.dot(o.center - centre, frame.normal) * frame.normal  # into the shown plane
        node.SetNthControlPointPositionWorld(0, *centre)
        node.SetNthControlPointPositionWorld(1, *target)
        node.SetName(f"{self.displayName(o)} {position.clock}")
        display.SetVisibility(True)

    # ------------------------------------------------------------------ notifications
    def onNavigationChanged(self):
        if self.ostia:
            self.updateClockHand()

    def onPathChanged(self):
        self.tube = None
        self._tubeKey = None
        self.refresh()

    def refresh(self):
        self.updateDisplay()
        self.zones = ishimaru_zones(self.navigator.path, self.ostia) if self.navigator.path is not None else []
        self._notify()

    def _changed(self):
        self.measurements._persist()
        self.refresh()

    def _notify(self):
        for listener in list(self.listeners):
            try:
                listener()
            except Exception:
                logging.exception("AortaPlan: ostia listener failed")

    def reset(self):
        self.stopClicksQuietly()
        self.tube = None
        self._tubeKey = None
        self.zones = []
        self.lastStumps = []

    def stopClicksQuietly(self):
        self.clicks = []
        if self._clickObservation is not None:
            node, tag = self._clickObservation
            node.RemoveObserver(tag)
            self._clickObservation = None

    def cleanup(self):
        self.reset()
        self.listeners = []


def describeRow(controller, ostium, position, reference, previous):
    """Texts of one table row (used by the panel and the tests)."""
    diameter = ostium.branch_diameter_mm
    takeoff = position.takeoff
    return {
        "name": controller.displayName(ostium),
        "position": formatNumber(position.arc_length_mm - reference),
        "previous": "" if previous is None else formatNumber(position.arc_length_mm - previous),
        "clock": position.clock,
        "degrees": formatNumber(position.clock_deg, 0),
        "arc": formatNumber(position.arc_on_graft_mm),
        "diameter": formatNumber(diameter) if diameter is not None else "—",
        "craniocaudal": formatNumber(takeoff.craniocaudal_deg, 0) if takeoff is not None else "—",
        "axial": formatNumber(takeoff.axial_deg, 0) if takeoff is not None else "—",
    }
