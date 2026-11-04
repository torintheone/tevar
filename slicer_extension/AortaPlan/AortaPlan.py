"""AortaPlan: TEVAR/EVAR/FEVAR planning on CT angiography.

Research use only. Not for clinical decision making.

This file is the Slicer shell only: loading, display and interaction. All measurements
are computed by the Slicer-independent library ``aortaplan_core`` (``<repo>/core``).
"""

import logging
import os
import sys

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from AortaPlanLib.corepath import ensureCoreOnPath

ensureCoreOnPath()

try:
    from aortaplan_core.i18n import tr
except ImportError:  # the environment check reports the missing core

    def tr(key, **kwargs):
        return key


#
# AortaPlan
#


class AortaPlan(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AortaPlan"
        self.parent.categories = ["AortaPlan"]
        # Extensions are checked at run time (see AortaPlanLib.environment) so that the
        # module still loads and can explain what is missing.
        self.parent.dependencies = []
        self.parent.contributors = ["AortaPlan project"]
        self.parent.helpText = tr("module.help")
        self.parent.acknowledgementText = tr("module.acknowledgement")


#
# AortaPlanWidget
#


class AortaPlanWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.appMode = None
        self.ui = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        from AortaPlanLib.appmode import AppModeController

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AortaPlan.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = AortaPlanLogic()
        self.appMode = AppModeController(windowTitle=tr("app.window_title"))
        self._applyStrings()

        self.ui.appModeCheckBox.connect("toggled(bool)", self.onAppModeToggled)
        self.ui.showSlicerUiCheckBox.connect("toggled(bool)", self.onShowSlicerUiToggled)
        self.ui.recheckButton.connect("clicked()", self.updateEnvironmentReport)
        self.ui.installPackagesButton.connect("clicked()", self.onInstallPackages)

        self.updateEnvironmentReport()

    def _applyStrings(self):
        """All visible texts come from aortaplan_core.i18n (one place for all UI strings)."""
        self.ui.disclaimerLabel.text = tr("disclaimer")
        self.ui.modeCollapsibleButton.text = tr("appmode.section")
        self.ui.appModeCheckBox.text = tr("appmode.enable")
        self.ui.appModeCheckBox.toolTip = tr("appmode.enable.tooltip")
        self.ui.showSlicerUiCheckBox.text = tr("appmode.show_slicer_ui")
        self.ui.showSlicerUiCheckBox.toolTip = tr("appmode.show_slicer_ui.tooltip")
        self.ui.environmentCollapsibleButton.text = tr("env.section")
        self.ui.recheckButton.text = tr("env.recheck")
        self.ui.installPackagesButton.text = tr("deps.install_button")

    def cleanup(self):
        """Called when the application closes or the module is reloaded."""
        if self.appMode is not None:
            self.appMode.setEnabled(False)
        self.removeObservers()

    def enter(self):
        if self.appMode is not None:
            self.appMode.resume()

    def exit(self):
        # Restore the Slicer interface whenever the module is left, in particular before
        # Slicer saves the main-window state on close.
        if self.appMode is not None:
            self.appMode.suspend()

    def onAppModeToggled(self, checked):
        self.appMode.setEnabled(checked)
        self.ui.showSlicerUiCheckBox.enabled = checked
        if not checked:
            self.ui.showSlicerUiCheckBox.checked = False

    def onShowSlicerUiToggled(self, checked):
        self.appMode.setSlicerUiVisible(checked)

    def updateEnvironmentReport(self):
        from AortaPlanLib import environment

        items = self.logic.checkEnvironment()
        self.ui.environmentLabel.text = environment.formatReportHtml(items)
        self.ui.installPackagesButton.visible = bool(environment.missingPythonPackages())

    def onInstallPackages(self):
        from AortaPlanLib import environment

        try:
            environment.installMissingPythonPackages(ask=True)
        except Exception:
            logging.exception("AortaPlan: package installation failed")
        self.updateEnvironmentReport()


#
# AortaPlanLogic
#


class AortaPlanLogic(ScriptedLoadableModuleLogic):
    """Thin Slicer-side logic; the computations are delegated to aortaplan_core."""

    def checkEnvironment(self):
        from AortaPlanLib import environment

        return environment.checkEnvironment()


#
# AortaPlanTest
#


class AortaPlanTest(ScriptedLoadableModuleTest):
    """Runs the integration tests in ``Testing/AortaPlanSlicerTests.py`` ("Reload and Test")."""

    def runTest(self, **kwargs):
        import importlib
        import unittest

        testsDir = os.path.join(os.path.dirname(slicer.modules.aortaplan.path), "Testing")
        if testsDir not in sys.path:
            sys.path.insert(0, testsDir)
        import AortaPlanSlicerTests

        importlib.reload(AortaPlanSlicerTests)
        suite = unittest.TestLoader().loadTestsFromModule(AortaPlanSlicerTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        self.assertTrue(result.wasSuccessful(), "AortaPlan integration tests failed, see the Python console")
