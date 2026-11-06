"""Re-arranging views with the mouse: drag a view by its title bar onto another view.

An event filter on each view's controller bar (the coloured bar at the top of a view)
notices a press-drag-release; on release over another view the two views swap places
in the layout (``LayoutController.swap``). Clicks on the bar's own buttons and slider are
handled by those widgets and never reach the filter.
"""

import qt

from . import layout

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


DRAG_THRESHOLD_PX = 6


def _controllerBar(tag):
    widget = layout.viewWidget(tag)
    if widget is None:
        return None
    controller = widget.threeDController() if tag == layout.THREE_D_VIEW else widget.sliceController()
    return controller.barWidget() if controller is not None else None


class _BarEventFilter(qt.QObject):
    def __init__(self, swapper, tag):
        qt.QObject.__init__(self)
        self.swapper = swapper
        self.tag = tag
        self.pressPos = None
        self.dragging = False

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
            self.pressPos = event.globalPos()
            self.dragging = False
            return False
        if kind == qt.QEvent.MouseMove and self.pressPos is not None:
            pos = event.globalPos()
            if not self.dragging and abs(pos.x() - self.pressPos.x()) + abs(pos.y() - self.pressPos.y()) > DRAG_THRESHOLD_PX:
                self.dragging = True
                qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.ClosedHandCursor))
            return False
        if kind == qt.QEvent.MouseButtonRelease and self.pressPos is not None:
            dragging, self.pressPos, self.dragging = self.dragging, None, False
            if dragging:
                qt.QApplication.restoreOverrideCursor()
                target = self.swapper.tagAt(event.globalPos())
                # Re-arrange after this event has been handled (the layout re-parents the views).
                qt.QTimer.singleShot(0, lambda: self.swapper.swap(self.tag, target))
                return True
        return False


class ViewDragSwapper:
    """Installs the title-bar filters on the views of the AortaPlan layout."""

    def __init__(self, controller):
        self.controller = controller
        self._filters = []  # (bar widget, filter)

    def install(self):
        self.uninstall()
        for tag in self.controller.visibleOrder():
            bar = _controllerBar(tag)
            if bar is None:
                continue
            eventFilter = _BarEventFilter(self, tag)
            bar.installEventFilter(eventFilter)
            bar.setToolTip(tr("layout.drag_hint"))
            self._filters.append((bar, eventFilter))

    def uninstall(self):
        for bar, eventFilter in self._filters:
            try:
                bar.removeEventFilter(eventFilter)
            except Exception:
                pass  # the widget may already be gone
        self._filters = []

    def tagAt(self, globalPos):
        """View under a screen position (None outside the AortaPlan views)."""
        names = {layout.widgetObjectName(tag): tag for tag in self.controller.visibleOrder()}
        widget = qt.QApplication.widgetAt(globalPos)
        while widget is not None:
            if widget.objectName in names:
                return names[widget.objectName]
            widget = widget.parentWidget()
        return None

    def swap(self, source, target):
        if target is None or target == source:
            return False
        return self.controller.swap(source, target)
