"""AortaPlan view layout (SPEC 5): large CPR view, perpendicular slice, 3D, optional axial.

The slice nodes are identified by their layout names. Without a main window
(tests with --no-main-window) the slice nodes are created directly, so the navigation
logic can still be exercised.
"""

import slicer

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


LONGITUDINAL_VIEW = "AortaPlanCPR"  # straightened CPR at the current angle
CPR2_VIEW = "AortaPlanCPR2"  # straightened CPR at angle + 90 deg
CPR_VIEWS = (LONGITUDINAL_VIEW, CPR2_VIEW)
CROSS_VIEW = "AortaPlanCross"
AXIAL_VIEW = "Red"

LAYOUT_ID = 7101
LAYOUT_WITH_AXIAL_ID = 7102

_SLICE_VIEW_XML = """
<view class="vtkMRMLSliceNode" singletontag="{tag}">
 <property name="orientation" action="default">{orientation}</property>
 <property name="viewlabel" action="default">{label}</property>
 <property name="viewcolor" action="default">{color}</property>
</view>"""

_THREED_VIEW_XML = """
<view class="vtkMRMLViewNode" singletontag="1">
 <property name="viewlabel" action="default">1</property>
</view>"""


def _sliceXml(tag, orientation, label, color):
    return _SLICE_VIEW_XML.format(tag=tag, orientation=orientation, label=label, color=color)


def layoutXml(withAxial=False):
    """Two CPR views side by side on the left; perpendicular slice, (axial) and 3D on the right."""
    cpr1 = _sliceXml(LONGITUDINAL_VIEW, "Reformat", tr("view.label.cpr1"), "#D9A13A")
    cpr2 = _sliceXml(CPR2_VIEW, "Reformat", tr("view.label.cpr2"), "#3AA6D9")
    cross = _sliceXml(CROSS_VIEW, "Reformat", tr("view.label.cross"), "#8C8C8C")
    rightItems = [cross]
    if withAxial:
        rightItems.append(_sliceXml(AXIAL_VIEW, "Axial", "R", "#F34A33"))
    rightItems.append(_THREED_VIEW_XML)
    right = "".join(f"<item>{x}</item>" for x in rightItems)
    return (
        '<layout type="horizontal" split="true">'
        f'<item splitSize="300">{cpr1}</item>'
        f'<item splitSize="300">{cpr2}</item>'
        f'<item splitSize="400"><layout type="vertical" split="true">{right}</layout></item>'
        "</layout>"
    )


def registerLayouts():
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return False
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    for layoutId, withAxial in ((LAYOUT_ID, False), (LAYOUT_WITH_AXIAL_ID, True)):
        xml = layoutXml(withAxial)
        if layoutNode.IsLayoutDescription(layoutId):
            layoutNode.SetLayoutDescription(layoutId, xml)
        else:
            layoutNode.AddLayoutDescription(layoutId, xml)
    return True


class LayoutController:
    """Switches to the AortaPlan layout and back to the user's previous layout."""

    def __init__(self):
        self.previousLayout = None
        self.withAxial = False

    def activate(self, withAxial=None):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return False
        if withAxial is not None:
            self.withAxial = bool(withAxial)
        registerLayouts()
        current = layoutManager.layout
        if self.previousLayout is None and current not in (LAYOUT_ID, LAYOUT_WITH_AXIAL_ID):
            self.previousLayout = current
        layoutManager.setLayout(LAYOUT_WITH_AXIAL_ID if self.withAxial else LAYOUT_ID)
        return True

    def deactivate(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or self.previousLayout is None:
            return
        if layoutManager.layout in (LAYOUT_ID, LAYOUT_WITH_AXIAL_ID):
            layoutManager.setLayout(self.previousLayout)
        self.previousLayout = None


def findSliceNode(layoutName):
    for node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
        if node.GetLayoutName() == layoutName:
            return node
    return None


def ensureSliceNode(layoutName):
    """The slice node of a view; created if no layout has created it yet (headless runs)."""
    node = findSliceNode(layoutName)
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSliceNode", layoutName)
        node.SetLayoutName(layoutName)
    return node


def sliceWidget(layoutName):
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return None
    return layoutManager.sliceWidget(layoutName)


def threeDWidget():
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None or layoutManager.threeDViewCount == 0:
        return None
    return layoutManager.threeDWidget(0)
