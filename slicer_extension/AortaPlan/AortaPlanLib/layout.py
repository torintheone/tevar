"""AortaPlan view layout (SPEC 5): two CPR views, perpendicular slice, 3D, optional axial.

The user can choose a layout template and swap views by dragging a view's title bar onto
another view (``viewdrag.py``); the choice is remembered in the Slicer settings.

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
THREE_D_VIEW = "1"  # layout name of Slicer's first 3D view
SLICE_VIEWS = (LONGITUDINAL_VIEW, CPR2_VIEW, CROSS_VIEW, AXIAL_VIEW)

LAYOUT_ID = 7101
DEFAULT_ORDER = [LONGITUDINAL_VIEW, CPR2_VIEW, CROSS_VIEW, AXIAL_VIEW, THREE_D_VIEW]
TEMPLATES = ("columns", "grid", "large_first")
DEFAULT_TEMPLATE = "columns"
SETTINGS_ORDER = "AortaPlan/LayoutOrder"
SETTINGS_TEMPLATE = "AortaPlan/LayoutTemplate"

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


def _viewXml(tag):
    if tag == THREE_D_VIEW:
        return _THREED_VIEW_XML
    orientation, label, color = {
        LONGITUDINAL_VIEW: ("Reformat", tr("view.label.cpr1"), "#D9A13A"),
        CPR2_VIEW: ("Reformat", tr("view.label.cpr2"), "#3AA6D9"),
        CROSS_VIEW: ("Reformat", tr("view.label.cross"), "#8C8C8C"),
        AXIAL_VIEW: ("Axial", "R", "#F34A33"),
    }[tag]
    return _SLICE_VIEW_XML.format(tag=tag, orientation=orientation, label=label, color=color)


def _items(tags, sizes=None):
    sizes = sizes or [None] * len(tags)
    out = []
    for tag, size in zip(tags, sizes):
        attribute = f' splitSize="{size}"' if size else ""
        content = tag if tag.startswith("<") else _viewXml(tag)
        out.append(f"<item{attribute}>{content}</item>")
    return "".join(out)


def _stack(kind, tags, sizes=None):
    return f'<layout type="{kind}" split="true">{_items(tags, sizes)}</layout>'


def layoutXml(withAxial=False, order=None, template=DEFAULT_TEMPLATE):
    """Layout description for the given view order and template.

    - ``columns``: view 1 | view 2 | the others stacked (default: two CPRs side by side);
    - ``grid``: two rows, the first two views on top;
    - ``large_first``: view 1 large on the left, the others stacked on the right.
    """
    tags = [t for t in (order or DEFAULT_ORDER) if withAxial or t != AXIAL_VIEW]
    if template == "grid":
        top, bottom = tags[:2], tags[2:]
        return _stack("vertical", [_stack("horizontal", top), _stack("horizontal", bottom)])
    if template == "large_first":
        return _stack("horizontal", [tags[0], _stack("vertical", tags[1:])], [650, 350])
    return _stack("horizontal", [tags[0], tags[1], _stack("vertical", tags[2:])], [300, 300, 400])


def _registerLayout(xml):
    layoutManager = slicer.app.layoutManager()
    if layoutManager is None:
        return False
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if layoutNode.IsLayoutDescription(LAYOUT_ID):
        layoutNode.SetLayoutDescription(LAYOUT_ID, xml)  # refreshes the views if it is the current layout
    else:
        layoutNode.AddLayoutDescription(LAYOUT_ID, xml)
    return True


def _loadSettings():
    settings = slicer.app.userSettings()
    template = str(settings.value(SETTINGS_TEMPLATE, DEFAULT_TEMPLATE) or DEFAULT_TEMPLATE)
    if template not in TEMPLATES:
        template = DEFAULT_TEMPLATE
    stored = settings.value(SETTINGS_ORDER, "")
    order = [t for t in str(stored or "").split(",") if t]
    if sorted(order) != sorted(DEFAULT_ORDER):
        order = list(DEFAULT_ORDER)
    return template, order


class LayoutController:
    """Switches to the AortaPlan layout and back; keeps the user's template and view order."""

    def __init__(self):
        self.previousLayout = None
        self.withAxial = False
        self.template, self.order = _loadSettings()
        self.listeners = []  # called after the views were re-arranged

    @property
    def active(self):
        layoutManager = slicer.app.layoutManager()
        return layoutManager is not None and layoutManager.layout == LAYOUT_ID

    def visibleOrder(self):
        return [t for t in self.order if self.withAxial or t != AXIAL_VIEW]

    def xml(self):
        return layoutXml(self.withAxial, self.order, self.template)

    def activate(self, withAxial=None):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return False
        if withAxial is not None:
            self.withAxial = bool(withAxial)
        _registerLayout(self.xml())
        current = layoutManager.layout
        if self.previousLayout is None and current != LAYOUT_ID:
            self.previousLayout = current
        layoutManager.setLayout(LAYOUT_ID)
        return True

    def deactivate(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None or self.previousLayout is None:
            return
        if layoutManager.layout == LAYOUT_ID:
            layoutManager.setLayout(self.previousLayout)
        self.previousLayout = None

    def apply(self):
        """Re-arrange the views in place after a change of template, order or axial view."""
        settings = slicer.app.userSettings()
        settings.setValue(SETTINGS_TEMPLATE, self.template)
        settings.setValue(SETTINGS_ORDER, ",".join(self.order))
        if self.active:
            _registerLayout(self.xml())
            slicer.app.processEvents()
        for listener in list(self.listeners):
            listener()

    def setTemplate(self, template):
        if template in TEMPLATES and template != self.template:
            self.template = template
            self.apply()

    def setWithAxial(self, withAxial):
        self.withAxial = bool(withAxial)
        self.apply()

    def swap(self, tagA, tagB):
        if tagA == tagB or tagA not in self.order or tagB not in self.order:
            return False
        i, j = self.order.index(tagA), self.order.index(tagB)
        self.order[i], self.order[j] = self.order[j], self.order[i]
        self.apply()
        return True

    def reset(self):
        self.template, self.order = DEFAULT_TEMPLATE, list(DEFAULT_ORDER)
        self.apply()


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


def viewWidget(tag):
    """Slice or 3D widget of a view in the layout (None if it is not shown)."""
    return threeDWidget() if tag == THREE_D_VIEW else sliceWidget(tag)


def widgetObjectName(tag):
    """Object name the layout manager gives to the widget of a view."""
    return "ThreeDWidget1" if tag == THREE_D_VIEW else "qMRMLSliceWidget" + tag
