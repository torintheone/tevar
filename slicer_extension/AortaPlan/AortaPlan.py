"""AortaPlan: TEVAR/EVAR/FEVAR planning on CT angiography.

Research use only. Not for clinical decision making.

This file is the Slicer shell only: loading, display and interaction. All measurements
are computed by the Slicer-independent library ``aortaplan_core`` (``<repo>/core``).
"""

import logging
import os
import sys

import qt
import slicer
from slicer import vtkMRMLMarkupsCurveNode, vtkMRMLScalarVolumeNode
from slicer.parameterNodeWrapper import parameterNodeWrapper
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
# Parameter node: the module state (saved with the scene; used again in stage 9)
#


@parameterNodeWrapper
class AortaPlanParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    centerline: vtkMRMLMarkupsCurveNode
    positionMm: float = 0.0
    cprAngleDeg: float = 0.0
    currentStep: int = 0
    showAxial: bool = False
    windowLevelPreset: str = "cta"
    centerlineMethod: str = "vmtk"  # chosen by the user on Panoramix (stage 2)


STEP_KEYS = [
    "wizard.step.study",
    "wizard.step.centerline",
    "wizard.step.measurements",
    "wizard.step.ostia",
    "wizard.step.carm",
    "wizard.step.report",
    "wizard.step.printing",
]
#: Development stage (SPEC section 8) in which each placeholder step is implemented.
PLACEHOLDER_STAGES = {3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
PRESET_BUTTONS = {"cta": "wlCtaButton", "calcium": "wlCalciumButton", "soft_tissue": "wlSoftTissueButton", "bone": "wlBoneButton"}
STEP_STUDY, STEP_CENTERLINE = 0, 1
DEVELOPMENT_STAGE = 2  # last implemented stage of SPEC section 8, shown under the disclaimer


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
        self.navigator = None
        self.interaction = None
        self.layoutController = None
        self._parameterNode = None
        self._seriesRows = []
        self._updatingGui = False
        self._active = False
        self.builder = None
        self.segmentEditor = None
        self._rebuildAfterPlacement = False

    # ------------------------------------------------------------------ setup
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        from AortaPlanLib.appmode import AppModeController
        from AortaPlanLib.layout import LayoutController
        from AortaPlanLib.navigation import SliceNavigator, ViewInteraction

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AortaPlan.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.curveSelector.setMRMLScene(slicer.mrmlScene)

        self.logic = AortaPlanLogic()
        self.appMode = AppModeController(windowTitle=tr("app.window_title"))
        self.layoutController = LayoutController()
        self.navigator = SliceNavigator()
        self.navigator.listeners.append(self.onNavigationChanged)
        self.interaction = ViewInteraction(self.navigator)
        self._applyStrings()
        self._setupSeriesTable()

        # Wizard
        self._stepButtons = [getattr(self.ui, f"step{i + 1}Button") for i in range(len(STEP_KEYS))]
        for index, button in enumerate(self._stepButtons):
            button.connect("clicked()", lambda index=index: self.setCurrentStep(index))
        self.ui.backButton.connect("clicked()", lambda: self.setCurrentStep(self.ui.stepStack.currentIndex - 1))
        self.ui.nextButton.connect("clicked()", lambda: self.setCurrentStep(self.ui.stepStack.currentIndex + 1))

        # Step 1
        self.ui.importDicomButton.connect("clicked()", self.onImportDicom)
        self.ui.refreshSeriesButton.connect("clicked()", self.refreshSeriesList)
        self.ui.loadSeriesButton.connect("clicked()", self.onLoadSeries)
        self.ui.seriesTable.connect("itemSelectionChanged()", self._updateButtons)
        self.ui.sampleDataButton.connect("clicked()", self.onLoadSampleData)
        self.ui.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeSelected)

        # Step 2
        self.ui.curveSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onCurveSelected)
        self.ui.drawCurveButton.connect("clicked()", self.onDrawCurve)
        self.ui.finishCurveButton.connect("clicked()", self.onFinishCurve)
        self.ui.reverseCurveButton.connect("clicked()", self.onReverseCurve)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        self.addObserver(interactionNode, interactionNode.InteractionModeChangedEvent, self.onInteractionModeChanged)

        # Step 2: automatic lumen and centerline
        from AortaPlanLib.centerline import CenterlineBuilder

        self.builder = CenterlineBuilder()
        self._pointButtons = {
            "start": self.ui.startPointButton,
            "ends": self.ui.endPointsButton,
            "stops": self.ui.stopPointsButton,
        }
        for kind, button in self._pointButtons.items():
            button.connect("toggled(bool)", lambda checked, kind=kind: self.onPointButtonToggled(kind, checked))
        self.ui.clearPointsButton.connect("clicked()", self.onClearPoints)
        self.ui.methodComboBox.addItem(tr("auto.method.vmtk"), "vmtk")
        self.ui.methodComboBox.addItem(tr("auto.method.minimal_path"), "minimal_path")
        self.ui.methodComboBox.connect("currentIndexChanged(int)", self.onMethodChanged)
        self.ui.exclusionSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.buildButton.connect("clicked()", self.onBuild)
        self.ui.rebuildCenterlineButton.connect("clicked()", self.onRebuildCenterline)
        self.ui.addBranchButton.connect("clicked()", self.onAddBranch)
        self.ui.pathComboBox.connect("currentIndexChanged(int)", self.onPathSelected)
        self.ui.showRenderingCheckBox.connect("toggled(bool)", self.onShowRenderingToggled)
        self.ui.showLumenCheckBox.connect("toggled(bool)", lambda checked: self.builder.setLumenVisible(checked))
        self._setupSegmentEditor()

        # Navigation and display
        self.ui.positionSliderWidget.connect("valueChanged(double)", self.onPositionSliderChanged)
        self.ui.rotationSliderWidget.connect("valueChanged(double)", self.onRotationSliderChanged)
        for key, buttonName in PRESET_BUTTONS.items():
            getattr(self.ui, buttonName).connect("clicked()", lambda key=key: self.onWindowLevelPreset(key))
        self.ui.showAxialCheckBox.connect("toggled(bool)", self.onShowAxialToggled)

        # Mode and environment
        self.ui.appModeCheckBox.connect("toggled(bool)", self.onAppModeToggled)
        self.ui.showSlicerUiCheckBox.connect("toggled(bool)", self.onShowSlicerUiToggled)
        self.ui.recheckButton.connect("clicked()", self.updateEnvironmentReport)
        self.ui.installPackagesButton.connect("clicked()", self.onInstallPackages)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.updateEnvironmentReport()
        self.initializeParameterNode()
        self.refreshSeriesList()
        self.setCurrentStep(self._parameterNode.currentStep if self._parameterNode else STEP_STUDY)

    def _applyStrings(self):
        """All visible texts come from aortaplan_core.i18n (one place for all UI strings)."""
        ui = self.ui
        ui.disclaimerLabel.text = tr("disclaimer")
        import aortaplan_core

        ui.versionLabel.text = tr("app.version", version=aortaplan_core.__version__, stage=DEVELOPMENT_STAGE)
        for i, key in enumerate(STEP_KEYS):
            getattr(ui, f"step{i + 1}Button").text = f"{i + 1}. {tr(key)}"
        for page, stage in PLACEHOLDER_STAGES.items():
            getattr(ui, f"placeholderLabel{page}").text = tr("wizard.placeholder", stage=stage)
        ui.backButton.text = tr("wizard.back")
        ui.nextButton.text = tr("wizard.next")
        ui.importDicomButton.text = tr("study.import_dicom")
        ui.refreshSeriesButton.text = tr("study.refresh")
        ui.loadSeriesButton.text = tr("study.load_series")
        ui.volumeLabel.text = tr("study.volume")
        ui.sampleDataButton.text = tr("study.sample_data")
        ui.autoHintLabel.text = tr("auto.hint")
        ui.startPointButton.text = tr("auto.start")
        ui.endPointsButton.text = tr("auto.ends")
        ui.stopPointsButton.text = tr("auto.stops")
        ui.clearPointsButton.text = tr("auto.clear")
        ui.methodLabel.text = tr("auto.method")
        ui.exclusionLabel.text = tr("auto.exclusion")
        ui.buildButton.text = tr("auto.build")
        ui.pathLabel.text = tr("auto.path")
        ui.addBranchButton.text = tr("auto.add_branch")
        ui.rebuildCenterlineButton.text = tr("auto.rebuild_centerline")
        ui.showRenderingCheckBox.text = tr("auto.show_rendering")
        ui.showLumenCheckBox.text = tr("auto.show_lumen")
        ui.editCollapsibleButton.text = tr("auto.edit_section")
        ui.manualCollapsibleButton.text = tr("auto.manual_section")
        ui.curveLabel.text = tr("centerline.curve")
        ui.drawCurveButton.text = tr("centerline.draw")
        ui.finishCurveButton.text = tr("centerline.finish")
        ui.reverseCurveButton.text = tr("centerline.reverse")
        ui.curveHintLabel.text = tr("centerline.hint")
        ui.navigationCollapsibleButton.text = tr("nav.section")
        ui.positionLabel.text = tr("nav.position")
        ui.rotationLabel.text = tr("nav.rotation")
        ui.positionSliderWidget.suffix = " " + tr("units.mm")
        ui.rotationSliderWidget.suffix = tr("units.deg")
        ui.navigationHelpLabel.text = tr("nav.help")
        ui.displayCollapsibleButton.text = tr("wl.section")
        from aortaplan_core.display import window_level_preset

        for key, buttonName in PRESET_BUTTONS.items():
            preset = window_level_preset(key)
            button = getattr(ui, buttonName)
            button.text = tr(f"wl.{key}")
            button.toolTip = tr("wl.tooltip", window=preset.window, level=preset.level)
        ui.showAxialCheckBox.text = tr("layout.show_axial")
        ui.modeCollapsibleButton.text = tr("appmode.section")
        ui.appModeCheckBox.text = tr("appmode.enable")
        ui.appModeCheckBox.toolTip = tr("appmode.enable.tooltip")
        ui.showSlicerUiCheckBox.text = tr("appmode.show_slicer_ui")
        ui.showSlicerUiCheckBox.toolTip = tr("appmode.show_slicer_ui.tooltip")
        ui.environmentCollapsibleButton.text = tr("env.section")
        ui.recheckButton.text = tr("env.recheck")
        ui.installPackagesButton.text = tr("deps.install_button")

    def _setupSeriesTable(self):
        table = self.ui.seriesTable
        columns = ["study.col.patient", "study.col.study", "study.col.series", "study.col.images",
                   "study.col.thickness", "study.col.spacing", "study.col.pixel", "study.col.warnings"]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([tr(c) for c in columns])
        table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(140)

    # ------------------------------------------------------------------ lifecycle
    def cleanup(self):
        """Called when the application closes or the module is reloaded."""
        self._deactivateViews()
        if self.appMode is not None:
            self.appMode.setEnabled(False)
        if self.navigator is not None:
            self.navigator.cleanup()
        self.removeObservers()

    def enter(self):
        if self.appMode is not None:
            self.appMode.resume()
        self.initializeParameterNode()
        self._activateViews()

    def exit(self):
        # Restore the Slicer interface whenever the module is left, in particular before
        # Slicer saves the main-window state on close.
        self._deactivateViews()
        if self.appMode is not None:
            self.appMode.suspend()

    def _activateViews(self):
        if self._active:
            return
        self._active = True
        showAxial = self._parameterNode.showAxial if self._parameterNode else False
        self.layoutController.activate(withAxial=showAxial)
        slicer.app.processEvents()  # let the layout create its views
        self.navigator.resetFieldsOfView()
        self.navigator.setVolume(self._parameterNode.inputVolume if self._parameterNode else None)
        self.navigator.setCurve(self._parameterNode.centerline if self._parameterNode else None)
        self.navigator.showCrossSectionIn3D(True)
        self.navigator.attachOverlays()
        self.interaction.install()
        self.interaction.updateBrowsing()

    def _deactivateViews(self):
        if not self._active:
            return
        self._active = False
        if self.segmentEditor is not None:
            self.segmentEditor.setActiveEffect(None)
        self.interaction.uninstall()
        self.navigator.detachOverlays()
        self.navigator.showCrossSectionIn3D(False)
        self.layoutController.deactivate()

    def onSceneStartClose(self, caller, event):
        self.navigator.reset()
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        self._ensureSegmentEditorNode()
        if self.builder is not None:
            self.builder.lastLumen = self.builder.lastNetwork = None
        if self.parent.isEntered:
            self.initializeParameterNode()

    # ------------------------------------------------------------------ parameter node
    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, parameterNode):
        self._parameterNode = parameterNode
        if parameterNode is None:
            return
        self._updatingGui = True
        try:
            self.ui.volumeSelector.setCurrentNode(parameterNode.inputVolume)
            self.ui.curveSelector.setCurrentNode(parameterNode.centerline)
            self.ui.rotationSliderWidget.value = parameterNode.cprAngleDeg
            self.ui.showAxialCheckBox.checked = parameterNode.showAxial
            methodIndex = self.ui.methodComboBox.findData(parameterNode.centerlineMethod)
            self.ui.methodComboBox.currentIndex = max(0, methodIndex)
        finally:
            self._updatingGui = False
        self.navigator.angle = parameterNode.cprAngleDeg
        self.navigator.position = parameterNode.positionMm
        self._updateVolumeInfo()
        self._updateCurveInfo()
        self._updatePointsInfo()
        self._refreshPathList()
        self._updateButtons()

    # ------------------------------------------------------------------ wizard
    def setCurrentStep(self, index):
        index = max(0, min(len(STEP_KEYS) - 1, int(index)))
        self.ui.stepStack.setCurrentIndex(index)
        self._stepButtons[index].checked = True
        if self._parameterNode is not None:
            self._parameterNode.currentStep = index
        self._updateButtons()

    def _updateButtons(self):
        ui = self.ui
        index = ui.stepStack.currentIndex
        ui.backButton.enabled = index > 0
        ui.nextButton.enabled = index < len(STEP_KEYS) - 1
        ui.loadSeriesButton.enabled = len(ui.seriesTable.selectedItems()) > 0
        hasVolume = ui.volumeSelector.currentNode() is not None
        drawing = self._isDrawing()
        ui.drawCurveButton.enabled = hasVolume and not drawing
        ui.finishCurveButton.enabled = drawing
        ui.reverseCurveButton.enabled = ui.curveSelector.currentNode() is not None and not drawing
        counts = self.builder.pointCounts() if self.builder else {"start": 0, "ends": 0}
        ui.buildButton.enabled = hasVolume and counts["start"] > 0 and counts["ends"] > 0
        hasLumen = self.builder is not None and self.builder.lastLumen is not None
        ui.rebuildCenterlineButton.enabled = hasVolume and hasLumen
        ui.addBranchButton.enabled = hasVolume and hasLumen
        ui.pathComboBox.enabled = ui.pathComboBox.count > 0
        hasPath = self.navigator.path is not None
        ui.positionSliderWidget.enabled = hasPath
        ui.rotationSliderWidget.enabled = hasPath

    # ------------------------------------------------------------------ step 1: study
    def refreshSeriesList(self):
        from AortaPlanLib import study

        with slicer.util.tryWithErrorDisplay(tr("study.load_failed", error=""), waitCursor=True):
            self._seriesRows = study.listCtSeries()
        table = self.ui.seriesTable
        table.setRowCount(0)
        for row in self._seriesRows:
            info, assessment = row.info, row.assessment
            warnings = "; ".join(tr(f"study.warning.{w.code}", **w.params) for w in assessment.warnings)
            pixel = f"{info.pixel_spacing_mm[1]:.2f}" if info.pixel_spacing_mm else ""
            values = [
                row.patient_name,
                row.study,
                info.description,
                str(info.number_of_images),
                f"{info.slice_thickness_mm:.2f}" if info.slice_thickness_mm else "",
                f"{assessment.slice_spacing_mm:.2f}" if assessment.slice_spacing_mm else "",
                pixel,
                warnings,
            ]
            r = table.rowCount
            table.insertRow(r)
            for c, value in enumerate(values):
                item = qt.QTableWidgetItem(value)
                if warnings:
                    item.setForeground(qt.QBrush(qt.QColor("#b35900")))
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        count = len(self._seriesRows)
        self.ui.seriesStatusLabel.text = tr("study.series_count", count=count) if count else tr("study.no_series")
        self._updateButtons()

    def onImportDicom(self):
        from AortaPlanLib import study

        folder = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), tr("study.import_dialog_title"))
        if not folder:
            return
        with slicer.util.tryWithErrorDisplay(tr("study.load_failed", error=folder), waitCursor=True):
            study.importFolder(folder)
        self.refreshSeriesList()
        self.ui.seriesStatusLabel.text = tr("study.imported", count=len(self._seriesRows))

    def onLoadSeries(self):
        from AortaPlanLib import study

        rows = self.ui.seriesTable.selectionModel().selectedRows()
        if not rows:
            return
        seriesRow = self._seriesRows[rows[0].row()]
        try:
            with slicer.util.WaitCursor():
                volume, regularized = study.loadSeries(seriesRow.info.series_uid)
        except Exception as exc:
            logging.exception("AortaPlan: loading failed")
            slicer.util.errorDisplay(tr("study.load_failed", error=exc))
            return
        self.ui.volumeSelector.setCurrentNode(volume)
        if regularized:
            slicer.util.warningDisplay(tr("study.regularized"))

    def onLoadSampleData(self):
        from AortaPlanLib import study

        try:
            with slicer.util.WaitCursor():
                volume = study.loadSampleData()
        except Exception as exc:
            logging.exception("AortaPlan: sample data failed")
            slicer.util.errorDisplay(tr("study.load_failed", error=exc))
            return
        self.ui.volumeSelector.setCurrentNode(volume)

    def onVolumeSelected(self, node):
        if self._updatingGui:
            return
        if self._parameterNode is not None:
            self._parameterNode.inputVolume = node
        if node is not None:
            from AortaPlanLib.display import applyWindowLevel

            applyWindowLevel(node, self._parameterNode.windowLevelPreset if self._parameterNode else "cta")
        self.navigator.setVolume(node)
        self._showRendering(node)
        if self.segmentEditor is not None:
            self.segmentEditor.setSourceVolumeNode(node)
        self._updateVolumeInfo()
        self._updateButtons()

    def _updateVolumeInfo(self):
        from aortaplan_core.study import assess_volume_geometry
        from AortaPlanLib.bridge import nodeGeometry

        node = self.ui.volumeSelector.currentNode()
        if node is None or node.GetImageData() is None:
            self.ui.volumeInfoLabel.text = ""
            return
        geometry = nodeGeometry(node)
        dims = " × ".join(str(d) for d in geometry.dimensions)
        spacing = " × ".join(f"{s:.2f}" for s in geometry.spacing)
        lines = [tr("study.volume_info", dims=dims, spacing=spacing)]
        warnings = assess_volume_geometry(geometry)
        if warnings:
            lines += ["<span style='color:#b35900'>⚠ " + tr(f"study.warning.{w.code}", **w.params) + "</span>" for w in warnings]
        else:
            lines.append(tr("study.volume_ok"))
        self.ui.volumeInfoLabel.text = "<br>".join(lines)

    # ------------------------------------------------------------------ step 2: centerline
    def onCurveSelected(self, node):
        if self._updatingGui:
            return
        if self._parameterNode is not None:
            self._parameterNode.centerline = node
        self.navigator.setCurve(node)
        self.interaction.updateBrowsing()
        self._updateCurveInfo()
        self._updateButtons()

    def _isDrawing(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode.GetCurrentInteractionMode() != interactionNode.Place:
            return False
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        curve = self.ui.curveSelector.currentNode()
        return curve is not None and selectionNode.GetActivePlaceNodeID() == curve.GetID()

    def onDrawCurve(self):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", tr("wizard.step.centerline"))
        node.CreateDefaultDisplayNodes()
        self.ui.curveSelector.setCurrentNode(node)
        self.navigator.setFrozen(True)
        self.interaction.updateBrowsing()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsCurveNode")
        selectionNode.SetActivePlaceNodeID(node.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        self._updateButtons()

    def onFinishCurve(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def onInteractionModeChanged(self, caller, event):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode.GetCurrentInteractionMode() != interactionNode.Place:
            for button in self._pointButtons.values():
                button.blockSignals(True)
                button.checked = False
                button.blockSignals(False)
            self._updatePointsInfo()
            if self._rebuildAfterPlacement:
                self._rebuildAfterPlacement = False
                self.onBuild()
        if self.navigator.frozen and not self._isDrawing():
            self.navigator.setFrozen(False)
            self.interaction.updateBrowsing()
            self._updateCurveInfo()
        self._updateButtons()

    def onReverseCurve(self):
        node = self.ui.curveSelector.currentNode()
        if node is None or node.GetNumberOfControlPoints() < 2:
            return
        points = slicer.util.arrayFromMarkupsControlPoints(node)
        slicer.util.updateMarkupsControlPointsFromArray(node, points[::-1].copy())
        if self.navigator.path is not None:
            self.navigator.setPosition(self.navigator.path.length - self.navigator.position)

    def _updateCurveInfo(self):
        path = self.navigator.path
        if path is None:
            node = self.ui.curveSelector.currentNode()
            self.ui.curveLengthLabel.text = tr("centerline.too_short") if node is not None else ""
        else:
            self.ui.curveLengthLabel.text = tr("centerline.length", length=path.length)

    # ------------------------------------------------------------------ step 2: automatic centerline
    def _setupSegmentEditor(self):
        editor = slicer.qMRMLSegmentEditorWidget()
        editor.setMRMLScene(slicer.mrmlScene)
        editor.segmentationNodeSelectorVisible = False
        editor.sourceVolumeNodeSelectorVisible = False
        editor.switchToSegmentationsButtonVisible = False
        editor.setEffectNameOrder(["Paint", "Erase", "Scissors"])
        editor.unorderedEffectsVisible = False
        editor.setMaximumNumberOfUndoStates(20)
        self.ui.segmentEditorPlaceholder.layout().addWidget(editor)
        self.segmentEditor = editor
        self._ensureSegmentEditorNode()

    def _ensureSegmentEditorNode(self):
        if self.segmentEditor is None:
            return
        node = slicer.mrmlScene.GetFirstNodeByName("AortaPlanSegmentEditor")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "AortaPlanSegmentEditor")
            node.SetSaveWithScene(False)
        self.segmentEditor.setMRMLSegmentEditorNode(node)

    def _currentMethod(self):
        return self.ui.methodComboBox.itemData(self.ui.methodComboBox.currentIndex) or "vmtk"

    def onMethodChanged(self, index):
        if not self._updatingGui and self._parameterNode is not None:
            self._parameterNode.centerlineMethod = self._currentMethod()

    def onPointButtonToggled(self, kind, checked):
        if checked:
            for other, button in self._pointButtons.items():
                if other != kind:
                    button.blockSignals(True)
                    button.checked = False
                    button.blockSignals(False)
            self.navigator.setFrozen(True)  # standard views while points are placed in slices
            self.interaction.updateBrowsing()
            self.builder.startPlacing(kind)
            node = self.builder.pointsNode(kind)
            for event in (slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, slicer.vtkMRMLMarkupsNode.PointRemovedEvent):
                if not self.hasObserver(node, event, self.onPointsModified):
                    self.addObserver(node, event, self.onPointsModified)
        else:
            self.builder.stopPlacing()
        self._updatePointsInfo()

    def onPointsModified(self, caller, event):
        self._updatePointsInfo()

    def onClearPoints(self):
        self.builder.clearPoints()
        self._updatePointsInfo()
        self._updateButtons()

    def _updatePointsInfo(self):
        if self.builder is None:
            return
        counts = self.builder.pointCounts()
        self.ui.pointsInfoLabel.text = tr(
            "auto.points_info",
            start=tr("auto.yes") if counts["start"] else tr("auto.no"),
            ends=counts["ends"],
            stops=counts["stops"],
        )
        self._updateButtons()

    def _progressCallback(self, dialog):
        def callback(fraction, step):
            dialog.setValue(int(round(100 * fraction)))
            dialog.labelText = tr(f"progress.{step}")
            slicer.app.processEvents()
            return not dialog.wasCanceled

        return callback

    def _runBuild(self, function):
        from aortaplan_core.lumen import Cancelled
        from AortaPlanLib.centerline import describeWarnings

        volume = self.ui.volumeSelector.currentNode()
        if volume is None:
            slicer.util.errorDisplay(tr("auto.need_volume"))
            return
        self.builder.stopPlacing()
        dialog = slicer.util.createProgressDialog(windowTitle=tr("progress.title"), labelText=tr("progress.lumen.hu_range"))
        try:
            curves = function(volume, self._progressCallback(dialog))
        except Cancelled:
            self.ui.buildStatusLabel.text = tr("auto.cancelled")
            return
        except Exception as exc:
            logging.exception("AortaPlan: build failed")
            slicer.util.errorDisplay(tr("auto.failed", error=exc))
            return
        finally:
            dialog.close()
        lumen, network = self.builder.lastLumen, self.builder.lastNetwork
        lines = [
            tr(
                "auto.result",
                volume_ml=lumen.volume_ml,
                low=lumen.hu_range[0],
                high=lumen.hu_range[1],
                paths=len(network.paths),
                seconds=sum(v for k, v in self.builder.lastTimings.items() if k != "total_s"),
            )
        ]
        for text in describeWarnings(list(lumen.warnings) + list(network.warnings)):
            lines.append(f"<span style='color:#b35900'>⚠ {text}</span>")
        self.ui.buildStatusLabel.text = "<br>".join(lines)
        self.segmentEditor.setSegmentationNode(self.builder.lumenSegmentation(create=False))
        self.segmentEditor.setSourceVolumeNode(volume)
        self.builder.setLumenVisible(self.ui.showLumenCheckBox.checked)
        self._refreshPathList(select=curves[0] if curves else None)

    def onBuild(self):
        exclusion = self.ui.exclusionSelector.currentNode()
        method = self._currentMethod()
        self._runBuild(lambda volume, progress: self.builder.build(volume, method, exclusion, progress))

    def onRebuildCenterline(self):
        method = self._currentMethod()
        self._runBuild(lambda volume, progress: self.builder.rebuildCenterline(volume, method, progress))

    def onAddBranch(self):
        self._rebuildAfterPlacement = True
        self.builder.startPlacing("ends")
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)

    def _refreshPathList(self, select=None):
        if self.builder is None:
            return
        from AortaPlanLib.centerline import CenterlineBuilder

        combo = self.ui.pathComboBox
        current = select or (self._parameterNode.centerline if self._parameterNode else None)
        combo.blockSignals(True)
        combo.clear()
        selectedIndex = -1
        for curve in CenterlineBuilder.pathCurves():
            side = curve.GetAttribute("AortaPlan.Path") or "center"
            length = curve.GetCurveLengthWorld()
            combo.addItem(tr("auto.path_name", side=tr(f"auto.side.{side}"), length=length), curve.GetID())
            if current is not None and curve.GetID() == current.GetID():
                selectedIndex = combo.count - 1
        combo.blockSignals(False)
        if combo.count:
            combo.currentIndex = selectedIndex if selectedIndex >= 0 else 0
            self.onPathSelected(combo.currentIndex)
        self._updateButtons()

    def onPathSelected(self, index):
        if index < 0:
            return
        curve = slicer.mrmlScene.GetNodeByID(self.ui.pathComboBox.itemData(index))
        if curve is None:
            return
        self.navigator.setFrozen(False)
        self.ui.curveSelector.setCurrentNode(curve)  # -> onCurveSelected -> navigator

    def _showRendering(self, volumeNode):
        from AortaPlanLib.centerline import setVolumeRendering

        if volumeNode is None or not self.ui.showRenderingCheckBox.checked:
            return
        setVolumeRendering(volumeNode, True)
        from AortaPlanLib import layout

        threeD = layout.threeDWidget()
        if threeD is not None:
            import ctk

            threeD.threeDView().lookFromViewAxis(ctk.ctkAxesWidget.Anterior)
            threeD.threeDView().resetFocalPoint()

    def onShowRenderingToggled(self, checked):
        from AortaPlanLib.centerline import setVolumeRendering

        setVolumeRendering(self.ui.volumeSelector.currentNode(), checked)

    # ------------------------------------------------------------------ navigation
    def onNavigationChanged(self):
        path = self.navigator.path
        ui = self.ui
        self._updatingGui = True
        try:
            if path is None:
                ui.positionValueLabel.text = tr("nav.no_curve")
            else:
                ui.positionSliderWidget.maximum = path.length
                ui.positionSliderWidget.value = self.navigator.position
                ui.rotationSliderWidget.value = self.navigator.angle
                ui.positionValueLabel.text = tr("nav.position_of", value=self.navigator.position, length=path.length)
        finally:
            self._updatingGui = False
        if self._parameterNode is not None and path is not None:
            self._parameterNode.positionMm = self.navigator.position
            self._parameterNode.cprAngleDeg = self.navigator.angle
        self.interaction.updateBrowsing()
        self._updateCurveInfo()
        self._updateButtons()

    def onPositionSliderChanged(self, value):
        if not self._updatingGui:
            self.navigator.setPosition(value)

    def onRotationSliderChanged(self, value):
        if not self._updatingGui:
            self.navigator.setAngle(value)

    def onWindowLevelPreset(self, key):
        from AortaPlanLib.display import applyWindowLevel

        if self._parameterNode is not None:
            self._parameterNode.windowLevelPreset = key
        applyWindowLevel(self.ui.volumeSelector.currentNode(), key)

    def onShowAxialToggled(self, checked):
        if self._updatingGui:
            return
        if self._parameterNode is not None:
            self._parameterNode.showAxial = checked
        if self._active:
            self._deactivateViews()
            self.layoutController.withAxial = checked
            self._activateViews()

    # ------------------------------------------------------------------ mode and environment
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
        self.ui.environmentCollapsibleButton.collapsed = not any(item.blocking for item in items)

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

    def getParameterNode(self):
        return AortaPlanParameterNode(super().getParameterNode())

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
