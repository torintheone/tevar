"""AortaPlan application mode: hide the standard Slicer user interface ("chrome").

The controller records the visibility of every element it hides and restores exactly
that state afterwards, so the user's own Slicer layout is not lost.

Slicer writes the main-window state to its settings when it closes, but only after it
has exited the current module (``qSlicerMainWindow::closeEvent``). The module widget
calls :meth:`AppModeController.suspend` from ``exit()``, so the saved state never
contains hidden toolbars.
"""

import logging

import qt
import slicer


class _VisibilityElement:
    """A widget whose visibility is hidden and later restored."""

    def __init__(self, widget):
        self.widget = widget

    def capture(self):
        return not self.widget.isHidden()

    def hide(self):
        self.widget.setVisible(False)

    def restore(self, wasVisible):
        self.widget.setVisible(wasVisible)


class _PropertyElement:
    """A boolean Qt property (or Python attribute) that is set to a value while hidden."""

    def __init__(self, obj, name, hiddenValue=False):
        self.obj = obj
        self.name = name
        self.hiddenValue = hiddenValue

    def capture(self):
        value = getattr(self.obj, self.name)
        # PythonQt may return the getter (a Q_INVOKABLE with the same name) instead of the value.
        return value() if callable(value) else value

    def hide(self):
        setattr(self.obj, self.name, self.hiddenValue)

    def restore(self, value):
        setattr(self.obj, self.name, value)


class _DockTitleBarElement:
    """The title bar of the module panel dock (replaced by an empty widget while hidden)."""

    def __init__(self, dock):
        self.dock = dock

    def capture(self):
        return self.dock.titleBarWidget() is None

    def hide(self):
        if self.dock.titleBarWidget() is None:
            self.dock.setTitleBarWidget(qt.QWidget(self.dock))

    def restore(self, hadDefaultTitleBar):
        if hadDefaultTitleBar:
            self.dock.setTitleBarWidget(None)


class _WindowTitleElement:
    """Main window title, replaced by a title that carries the research-use disclaimer."""

    def __init__(self, mainWindow, title):
        self.mainWindow = mainWindow
        self.title = title

    def capture(self):
        return windowTitle(self.mainWindow)

    def hide(self):
        self.mainWindow.setWindowTitle(self.title)

    def restore(self, title):
        self.mainWindow.setWindowTitle(title)


def windowTitle(widget):
    """Window title as a string (PythonQt may expose the getter method instead of the property)."""
    title = widget.windowTitle
    return title() if callable(title) else title


def _optionalChild(mainWindow, name):
    try:
        return slicer.util.findChild(mainWindow, name)
    except RuntimeError:
        return None


def chromeElements(mainWindow, title=None):
    """All interface elements hidden by the application mode, in hiding order."""
    elements = []
    for widget in mainWindow.findChildren("QMenuBar"):
        elements.append(_VisibilityElement(widget))
    for widget in mainWindow.findChildren("QToolBar"):  # includes the module selector toolbar
        elements.append(_VisibilityElement(widget))
    if mainWindow.statusBar():
        elements.append(_VisibilityElement(mainWindow.statusBar()))
    try:
        console = mainWindow.pythonConsole()
        if console is not None and console.parent() is not None:
            elements.append(_VisibilityElement(console.parent()))
    except AttributeError:
        pass
    try:
        errorLog = mainWindow.errorLogDockWidget()
        if errorLog is not None:
            elements.append(_VisibilityElement(errorLog))
    except AttributeError:
        pass
    for name in ("DataProbeCollapsibleWidget", "LogoLabel"):
        widget = _optionalChild(mainWindow, name)
        if widget is not None:
            elements.append(_VisibilityElement(widget))
    modulePanel = _optionalChild(mainWindow, "ModulePanel")
    if modulePanel is not None:
        elements.append(_PropertyElement(modulePanel, "helpAndAcknowledgmentVisible"))
    docks = mainWindow.findChildren("QDockWidget", "PanelDockWidget")
    if docks:
        elements.append(_DockTitleBarElement(docks[0]))
    sequences = getattr(slicer.modules, "sequences", None)
    if sequences is not None and hasattr(sequences, "autoShowToolBar"):
        # Prevent the sequence browser toolbar from popping up when a sequence is loaded.
        elements.append(_PropertyElement(sequences, "autoShowToolBar"))
    if title:
        elements.append(_WindowTitleElement(mainWindow, title))
    return elements


class AppModeController:
    """Turns the AortaPlan application mode on and off.

    State:
      - ``enabled``: the user switched AortaPlan mode on;
      - ``slicerUiVisible``: debugging override that shows the Slicer interface anyway;
      - suspended: the module is not active (``exit()``), the interface must be normal.
    The chrome is hidden only if enabled, not overridden and not suspended.
    """

    def __init__(self, windowTitle=None):
        self.windowTitle = windowTitle
        self.enabled = False
        self.slicerUiVisible = False
        self._suspended = False
        self._saved = None  # list of (element, captured value) while the chrome is hidden

    @property
    def chromeHidden(self):
        return self._saved is not None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)
        self._update()

    def setSlicerUiVisible(self, visible):
        self.slicerUiVisible = bool(visible)
        self._update()

    def suspend(self):
        self._suspended = True
        self._update()

    def resume(self):
        self._suspended = False
        self._update()

    def _update(self):
        shouldHide = self.enabled and not self.slicerUiVisible and not self._suspended
        if shouldHide and self._saved is None:
            self._hide()
        elif not shouldHide and self._saved is not None:
            self._restore()

    def _hide(self):
        mainWindow = slicer.util.mainWindow()
        if mainWindow is None:
            return  # e.g. Slicer started with --no-main-window
        saved = []
        for element in chromeElements(mainWindow, self.windowTitle):
            try:
                saved.append((element, element.capture()))
                element.hide()
            except Exception:  # one broken element must not leave the UI half-hidden
                logging.exception("AortaPlan: could not hide %s", element)
        self._saved = saved

    def _restore(self):
        saved, self._saved = self._saved, None
        for element, value in reversed(saved):
            try:
                element.restore(value)
            except Exception:
                logging.exception("AortaPlan: could not restore %s", element)
