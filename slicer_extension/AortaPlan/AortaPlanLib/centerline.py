"""Step 2: automatic lumen and centerline from a few clicks (SPEC 6.2).

The user places a start point and end points (in the 3D rendering or in slices), and
optionally stop points. ``CenterlineBuilder.build`` segments the lumen with the core,
stores it in a segmentation node (editable with brush/eraser/scissors), extracts the
centerlines with VMTK Extract Centerline or with the core minimal-path method, and creates
one editable markups curve per path.
"""

import logging
import time

import numpy as np
import slicer
import vtk
from vtk.util import numpy_support

from aortaplan_core.centerline.extraction import (
    METHOD_VMTK,
    CenterlineNetwork,
    CenterlinePath,
    extract_centerlines,
    side_of,
)
from aortaplan_core.geometry.curves import resample_polyline
from aortaplan_core.lumen import LumenParameters, lumen_from_mask, segment_lumen, snap_seed
from aortaplan_core.volume import Volume

from .bridge import nodeGeometry

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


START, ENDS, STOPS = "start", "ends", "stops"
POINT_NODE_NAMES = {START: "AortaPlan_Start", ENDS: "AortaPlan_Ends", STOPS: "AortaPlan_Stops"}
POINT_COLORS = {START: (0.2, 0.85, 0.3), ENDS: (0.25, 0.55, 1.0), STOPS: (1.0, 0.25, 0.2)}
POINT_LABELS = {START: "points.start", ENDS: "points.end", STOPS: "points.stop"}
LUMEN_NODE_NAME = "AortaPlan_Lumen"
LUMEN_SEGMENT_ID = "lumen"
PATH_ATTRIBUTE = "AortaPlan.Path"
PATH_COLORS = {"right": (0.95, 0.75, 0.1), "left": (0.1, 0.85, 0.85), "center": (0.9, 0.4, 0.9)}
CURVE_CONTROL_POINT_SPACING_MM = 8.0
VMTK_TARGET_POINTS = 15000
VMTK_SAMPLING_MM = 0.5
VOLUME_RENDERING_PRESET = "CT-AAA"


def vmtkAvailable():
    return hasattr(slicer.modules, "extractcenterline")


def volumeFromNode(volumeNode):
    """Core Volume sharing the voxel array of the node (no copy; the core does not modify it)."""
    if volumeNode.GetParentTransformNode() is not None:
        raise ValueError("volumes under a transform are not supported; harden the transform first")
    return Volume(data=slicer.util.arrayFromVolume(volumeNode), geometry=nodeGeometry(volumeNode))


def pointsArray(node):
    if node is None or node.GetNumberOfControlPoints() == 0:
        return np.zeros((0, 3))
    return slicer.util.arrayFromMarkupsControlPoints(node, world=True)


def setVolumeRendering(volumeNode, visible):
    """Show the CTA as a 3D rendering (contrast-filled vessels bright, bones dimmed)."""
    if volumeNode is None:
        return None
    logic = slicer.modules.volumerendering.logic()
    displayNode = logic.GetFirstVolumeRenderingDisplayNode(volumeNode)
    if displayNode is None:
        if not visible:
            return None
        displayNode = logic.CreateDefaultVolumeRenderingNodes(volumeNode)
        preset = logic.GetPresetByName(VOLUME_RENDERING_PRESET)
        if preset is not None:
            displayNode.GetVolumePropertyNode().Copy(preset)
    displayNode.SetVisibility(bool(visible))
    return displayNode


class CenterlineBuilder:
    """Points, lumen segmentation and centerline paths of the current study."""

    def __init__(self):
        self.lumenParameters = LumenParameters()
        self.lastLumen = None
        self.lastNetwork = None
        self.lastTimings = {}

    # ------------------------------------------------------------------ points
    def pointsNode(self, kind, create=True):
        name = POINT_NODE_NAMES[kind]
        node = slicer.mrmlScene.GetFirstNodeByName(name)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetSelectedColor(*POINT_COLORS[kind])
            display.SetColor(*POINT_COLORS[kind])
            display.SetGlyphScale(2.5)
            display.SetTextScale(2.5)
            node.SetControlPointLabelFormat(tr(POINT_LABELS[kind]) + (" %d" if kind != START else ""))
        return node

    def startPlacing(self, kind):
        """Enter place mode for one kind of point (the start point replaces the previous one)."""
        node = self.pointsNode(kind)
        if kind == START:
            node.RemoveAllControlPoints()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(node.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0 if kind == START else 1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    @staticmethod
    def stopPlacing():
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def clearPoints(self):
        for kind in (START, ENDS, STOPS):
            node = self.pointsNode(kind, create=False)
            if node is not None:
                node.RemoveAllControlPoints()

    def pointCounts(self):
        return {kind: len(pointsArray(self.pointsNode(kind, create=False))) for kind in (START, ENDS, STOPS)}

    # ------------------------------------------------------------------ build
    def build(self, volumeNode, method=METHOD_VMTK, exclusionSegmentationNode=None, progress=None):
        """Segment the lumen and extract the centerlines. Returns the list of path curve nodes."""
        start = pointsArray(self.pointsNode(START, create=False))
        ends = pointsArray(self.pointsNode(ENDS, create=False))
        if volumeNode is None:
            raise ValueError(tr("auto.need_volume"))
        if len(start) == 0 or len(ends) == 0:
            raise ValueError(tr("auto.need_points"))
        stops = pointsArray(self.pointsNode(STOPS, create=False))
        volume = volumeFromNode(volumeNode)
        exclusion = self._exclusionMask(exclusionSegmentationNode, volumeNode)

        t0 = time.perf_counter()
        lumen = segment_lumen(
            volume,
            start[0],
            list(ends),
            stops=list(stops),
            exclusion_mask=exclusion,
            params=self.lumenParameters,
            progress=_scaled(progress, 0.0, 0.4),
        )
        t1 = time.perf_counter()
        segmentationNode = self.updateLumenSegmentation(volumeNode, lumen.full_mask(volume.geometry))
        _report(progress, 0.45, "surface")
        network = self.extractCenterlines(volume, lumen, segmentationNode, method, _scaled(progress, 0.5, 1.0))
        t2 = time.perf_counter()
        self.lastLumen, self.lastNetwork = lumen, network
        self.lastTimings = {"lumen_s": t1 - t0, "centerline_s": t2 - t1, "total_s": t2 - t0}
        return self.updatePathCurves(network)

    def rebuildCenterline(self, volumeNode, method=METHOD_VMTK, progress=None):
        """Centerline from the (possibly hand-edited) lumen segmentation."""
        segmentationNode = self.lumenSegmentation(create=False)
        if segmentationNode is None or self.lastLumen is None:
            return self.build(volumeNode, method, progress=progress)
        volume = volumeFromNode(volumeNode)
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, LUMEN_SEGMENT_ID, volumeNode) > 0
        previous = self.lastLumen
        snapped = [snap_seed(volume, p, previous.hu_range, 10.0) for p in [previous.start, *previous.ends]]
        snapped = [p if p is not None else q for p, q in zip(snapped, [previous.start, *previous.ends])]
        lumen = lumen_from_mask(mask, volume.geometry, snapped[0], snapped[1:], previous.hu_range, previous.lumen_hu)
        t0 = time.perf_counter()
        network = self.extractCenterlines(volume, lumen, segmentationNode, method, progress)
        self.lastLumen, self.lastNetwork = lumen, network
        self.lastTimings = {"centerline_s": time.perf_counter() - t0}
        return self.updatePathCurves(network)

    def extractCenterlines(self, volume, lumen, segmentationNode, method, progress=None):
        if method == METHOD_VMTK and vmtkAvailable():
            _report(progress, 0.1, "vmtk")
            return vmtkCenterlines(segmentationNode, LUMEN_SEGMENT_ID, lumen.start, lumen.ends)
        network = extract_centerlines(volume, lumen, progress=progress)
        if method == METHOD_VMTK:
            network.warnings.append("vmtk_missing")
        return network

    def _exclusionMask(self, segmentationNode, volumeNode):
        lumenNode = self.lumenSegmentation(create=False)
        if segmentationNode is None or (lumenNode is not None and segmentationNode.GetID() == lumenNode.GetID()):
            return None
        mask = None
        segmentIds = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIds)
        for index in range(segmentIds.GetNumberOfValues()):
            array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentIds.GetValue(index), volumeNode)
            mask = array > 0 if mask is None else (mask | (array > 0))
        return mask

    # ------------------------------------------------------------------ outputs
    def lumenSegmentation(self, create=True):
        node = slicer.mrmlScene.GetFirstNodeByName(LUMEN_NODE_NAME)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", LUMEN_NODE_NAME)
            node.CreateDefaultDisplayNodes()
        return node

    def updateLumenSegmentation(self, volumeNode, mask):
        node = self.lumenSegmentation()
        node.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segmentation = node.GetSegmentation()
        if segmentation.GetSegment(LUMEN_SEGMENT_ID) is None:
            segmentation.AddEmptySegment(LUMEN_SEGMENT_ID, tr("segment.lumen"), [0.85, 0.2, 0.2])
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), node, LUMEN_SEGMENT_ID, volumeNode)
        node.CreateClosedSurfaceRepresentation()
        display = node.GetDisplayNode()
        display.SetOpacity3D(0.45)
        display.SetOpacity2DFill(0.15)
        display.SetOpacity2DOutline(0.8)
        return node

    def setLumenVisible(self, visible):
        node = self.lumenSegmentation(create=False)
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(bool(visible))

    @staticmethod
    def pathCurves():
        return [n for n in slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode") if n.GetAttribute(PATH_ATTRIBUTE)]

    def updatePathCurves(self, network):
        for node in self.pathCurves():
            slicer.mrmlScene.RemoveNode(node)
        curves = []
        for path in network.paths:
            name = tr("auto.path_curve", side=tr(f"auto.side.{path.side}"))
            curve = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
            curve.SetAttribute(PATH_ATTRIBUTE, path.side)
            curve.SetAttribute("AortaPlan.Method", network.method)
            controls = resample_polyline(path.points, CURVE_CONTROL_POINT_SPACING_MM)
            slicer.util.updateMarkupsControlPointsFromArray(curve, controls)
            curve.CreateDefaultDisplayNodes()
            display = curve.GetDisplayNode()
            color = PATH_COLORS.get(path.side, PATH_COLORS["center"])
            display.SetColor(*color)
            display.SetSelectedColor(*color)
            display.SetGlyphScale(1.5)
            display.SetPointLabelsVisibility(False)
            display.SetPropertiesLabelVisibility(False)
            curves.append(curve)
        return curves


def vmtkCenterlines(segmentationNode, segmentId, start, ends):
    """Centerlines from VMTK Extract Centerline, converted into a core CenterlineNetwork."""
    import ExtractCenterline

    logic = ExtractCenterline.ExtractCenterlineLogic()
    surface = logic.polyDataFromNode(segmentationNode, segmentId)
    surface = logic.preprocess(surface, VMTK_TARGET_POINTS, 4.0, False)
    endpoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "AortaPlan_VMTK_endpoints")
    try:
        endpoints.AddControlPoint(*np.asarray(start, float), "start")
        endpoints.SetNthControlPointSelected(0, False)  # VMTK: the unselected point is the source
        for i, end in enumerate(ends):
            endpoints.AddControlPoint(*np.asarray(end, float), f"end{i + 1}")
            endpoints.SetNthControlPointSelected(i + 1, True)
        polyData, _ = logic.extractCenterline(surface, endpoints, VMTK_SAMPLING_MM)
    finally:
        slicer.mrmlScene.RemoveNode(endpoints)

    network = CenterlineNetwork(method=METHOD_VMTK, start=np.asarray(start, float))
    points = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData()) if polyData.GetPoints() else np.zeros((0, 3))
    radiusArray = polyData.GetPointData().GetArray("Radius")
    radii = numpy_support.vtk_to_numpy(radiusArray) if radiusArray is not None else np.zeros(len(points))
    ends = [np.asarray(e, float) for e in ends]
    used = set()
    for cell in range(polyData.GetNumberOfCells()):
        ids = polyData.GetCell(cell).GetPointIds()
        index = np.array([ids.GetId(k) for k in range(ids.GetNumberOfIds())])
        if len(index) < 2:
            continue
        pts, rad = points[index].astype(float), radii[index].astype(float)
        if np.linalg.norm(pts[-1] - start) < np.linalg.norm(pts[0] - start):
            pts, rad = pts[::-1], rad[::-1]  # proximal first
        endIndex = int(np.argmin([np.linalg.norm(pts[-1] - e) for e in ends]))
        if endIndex in used:
            continue
        used.add(endIndex)
        network.paths.append(
            CenterlinePath(
                name=f"end_{endIndex + 1}",
                points=pts,
                radii_mm=rad,
                side=side_of(ends[endIndex], np.asarray(start, float), 10.0),
                end_index=endIndex,
            )
        )
    network.paths.sort(key=lambda p: p.end_index)
    missing = [i for i in range(len(ends)) if i not in used]
    network.warnings = [f"end_unreachable:{i}" for i in missing]
    return network


def _scaled(progress, start, stop):
    if progress is None:
        return None
    return lambda fraction, step: progress(start + (stop - start) * fraction, step)


def _report(progress, fraction, step):
    if progress is not None and progress(fraction, step) is False:
        from aortaplan_core.lumen import Cancelled

        raise Cancelled()


def describeWarnings(warnings):
    texts = []
    for code in warnings:
        if code.startswith("end_unreachable:"):
            texts.append(tr("auto.warning.end_unreachable", index=int(code.split(":")[1]) + 1))
        elif code == "vmtk_missing":
            texts.append(tr("auto.vmtk_missing"))
        else:
            texts.append(tr(f"auto.warning.{code}"))
    return texts


def logNetwork(network):
    for path in network.paths:
        logging.info("AortaPlan: path %s (%s): %.1f mm, %d points", path.name, path.side, path.length_mm, len(path.points))
