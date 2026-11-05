"""Integration tests that need Slicer (run by Testing/run_slicer_tests.py or "Reload and Test").

They cover the Slicer <-> core boundary: module loading, NRRD files written by one side
and read by the other, IJK->RAS geometry, and the application mode (only when Slicer
runs with a main window).
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import slicer
import vtk


def _smallPhantom():
    from aortaplan_core.phantoms import straight_tube

    # Oblique tube on an anisotropic CT-like grid: exercises every geometry sign.
    return straight_tube(
        diameter_mm=10.0, length_mm=12.0, direction=(0.3, -0.4, 1.0), spacing=(0.7, 0.8, 1.25), supersampling=2
    )


def _ijkToRasArray(volumeNode):
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    return slicer.util.arrayFromVTKMatrix(matrix)


class AortaPlanModuleTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.tempDir = tempfile.mkdtemp(prefix="aortaplan_test_")

    def tearDown(self):
        slicer.mrmlScene.Clear(0)
        shutil.rmtree(self.tempDir, ignore_errors=True)

    def test_module_is_registered(self):
        self.assertTrue(hasattr(slicer.modules, "aortaplan"), "AortaPlan module not found: check module paths")
        widget = slicer.modules.aortaplan.widgetRepresentation()
        self.assertIsNotNone(widget)

    def test_core_is_importable(self):
        import aortaplan_core
        from aortaplan_core.i18n import tr

        self.assertTrue(aortaplan_core.__version__)
        self.assertIn("исследований", tr("disclaimer"))

    def test_environment_check_runs(self):
        from AortaPlanLib import environment

        items = environment.checkEnvironment()
        byKey = {item.key: item for item in items}
        self.assertEqual(byKey["core"].status, environment.OK)
        for requirement in ("numpy", "scipy", "vtk"):
            self.assertEqual(byKey[f"package:{requirement}"].status, environment.OK, requirement)
        self.assertIn("slicer_version", byKey)
        self.assertIn("extension:SlicerVMTK", byKey)
        html = environment.formatReportHtml(items)
        self.assertIn("<table", html)

    def test_core_nrrd_loads_in_slicer_with_same_geometry(self):
        from aortaplan_core.fileio import write_nrrd

        phantom = _smallPhantom()
        path = os.path.join(self.tempDir, "phantom.nrrd")
        write_nrrd(path, phantom.volume)
        node = slicer.util.loadVolume(path)
        np.testing.assert_allclose(_ijkToRasArray(node), phantom.volume.geometry.ijk_to_ras_matrix(), atol=1e-6)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), phantom.volume.data)

    def test_slicer_nrrd_is_read_by_core(self):
        from aortaplan_core.fileio import read_nrrd
        from AortaPlanLib.bridge import volumeToNode

        phantom = _smallPhantom()
        node = volumeToNode(phantom.volume, "phantom")
        path = os.path.join(self.tempDir, "saved_by_slicer.nrrd")
        self.assertTrue(slicer.util.saveNode(node, path))
        back = read_nrrd(path)
        self.assertTrue(back.geometry.is_close(phantom.volume.geometry, atol=1e-5))
        np.testing.assert_array_equal(back.data, phantom.volume.data)

    def test_bridge_round_trip_and_ras_lookup(self):
        from AortaPlanLib.bridge import nodeToVolume, volumeToNode

        phantom = _smallPhantom()
        node = volumeToNode(phantom.volume, "phantom")
        back = nodeToVolume(node)
        self.assertTrue(back.geometry.is_close(phantom.volume.geometry))
        np.testing.assert_array_equal(back.data, phantom.volume.data)
        # Slicer's own RAS->IJK mapping finds the lumen at the tube centre (RAS origin).
        rasToIjk = vtk.vtkMatrix4x4()
        node.GetRASToIJKMatrix(rasToIjk)
        ijk = [int(round(c)) for c in rasToIjk.MultiplyPoint([0.0, 0.0, 0.0, 1.0])[:3]]
        value = slicer.util.arrayFromVolume(node)[ijk[2], ijk[1], ijk[0]]
        self.assertGreater(value, 300)

    def test_transformed_volume_rejected(self):
        from AortaPlanLib.bridge import nodeToVolume, volumeToNode

        node = volumeToNode(_smallPhantom().volume, "phantom")
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        node.SetAndObserveTransformNodeID(transform.GetID())
        with self.assertRaises(ValueError):
            nodeToVolume(node)


def windowTitle(widget):
    from AortaPlanLib.appmode import windowTitle as _windowTitle

    return _windowTitle(widget)


@unittest.skipIf(slicer.util.mainWindow() is None, "application mode needs the main window (run with --slicer-gui)")
class AortaPlanAppModeTests(unittest.TestCase):
    def _state(self):
        mw = slicer.util.mainWindow()
        return {
            "menubars": [not w.isHidden() for w in mw.findChildren("QMenuBar")],
            "toolbars": [not w.isHidden() for w in mw.findChildren("QToolBar")],
            "statusbar": not mw.statusBar().isHidden(),
            "pythonConsole": not mw.pythonConsole().parent().isHidden(),
            "title": windowTitle(mw),
        }

    def test_enable_disable_restores_original_state(self):
        from AortaPlanLib.appmode import AppModeController

        mw = slicer.util.mainWindow()
        before = self._state()
        controller = AppModeController(windowTitle="AortaPlan test title")
        controller.setEnabled(True)
        self.assertTrue(controller.chromeHidden)
        hidden = self._state()
        self.assertFalse(any(hidden["menubars"]))
        self.assertFalse(any(hidden["toolbars"]))
        self.assertFalse(hidden["statusbar"])
        self.assertFalse(hidden["pythonConsole"])
        self.assertEqual(windowTitle(mw), "AortaPlan test title")
        # The module panel (which hosts the AortaPlan UI) stays visible.
        self.assertFalse(mw.findChildren("QDockWidget", "PanelDockWidget")[0].isHidden())

        controller.setSlicerUiVisible(True)
        self.assertEqual(self._state(), before)
        controller.setSlicerUiVisible(False)
        self.assertTrue(controller.chromeHidden)

        controller.suspend()
        self.assertEqual(self._state(), before)
        controller.resume()
        self.assertTrue(controller.chromeHidden)

        controller.setEnabled(False)
        self.assertFalse(controller.chromeHidden)
        self.assertEqual(self._state(), before)

    def test_module_checkbox_toggles_mode(self):
        slicer.util.selectModule("AortaPlan")
        widget = slicer.modules.aortaplan.widgetRepresentation().self()
        before = self._state()
        widget.ui.appModeCheckBox.checked = True
        self.assertTrue(widget.appMode.chromeHidden)
        self.assertTrue(widget.ui.showSlicerUiCheckBox.enabled)
        widget.ui.showSlicerUiCheckBox.checked = True
        self.assertFalse(widget.appMode.chromeHidden)
        widget.ui.appModeCheckBox.checked = False
        self.assertFalse(widget.appMode.chromeHidden)
        self.assertFalse(widget.ui.showSlicerUiCheckBox.checked)
        self.assertEqual(self._state(), before)


# ---------------------------------------------------------------------------- stage 1


def _curveFromPoints(points, name="TestCurve", every=20):
    """Markups curve through every ``every``-th point (and the last one)."""
    indices = list(range(0, len(points), every))
    if indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
    slicer.util.updateMarkupsControlPointsFromArray(node, np.asarray(points)[indices])
    return node


def _torusSetup():
    from aortaplan_core.phantoms import torus_arc
    from AortaPlanLib.bridge import volumeToNode
    from AortaPlanLib.navigation import SliceNavigator

    phantom = torus_arc(leg_length_mm=10.0, supersampling=2)
    volume = volumeToNode(phantom.volume, "torus")
    curve = _curveFromPoints(phantom.truth.centerline_points)
    navigator = SliceNavigator()
    navigator.setVolume(volume)
    navigator.setCurve(curve)
    return phantom, volume, curve, navigator


def _cursorPosition(navigator):
    position = [0.0, 0.0, 0.0]
    navigator.cursorNode().GetNthControlPointPositionWorld(0, position)
    return np.array(position)


class AortaPlanStudyTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.tempDir = tempfile.mkdtemp(prefix="aortaplan_dicom_")

    def tearDown(self):
        slicer.mrmlScene.Clear(0)
        shutil.rmtree(self.tempDir, ignore_errors=True)

    def test_dicom_series_listing_and_loading(self):
        from DICOMLib import DICOMUtils

        from aortaplan_core.phantoms import straight_tube
        from AortaPlanLib import bridge, study

        phantom = straight_tube(diameter_mm=10.0, length_mm=30.0, supersampling=2)
        node = bridge.volumeToNode(phantom.volume, "phantom")
        dicomDir = os.path.join(self.tempDir, "dicom")
        os.makedirs(dicomDir)
        parameters = {
            "inputVolume": node.GetID(),
            "dicomDirectory": dicomDir,
            "dicomPrefix": "IMG",
            "modality": "CT",
            "seriesDescription": "AortaPlan phantom",
            "patientName": "Phantom^Test",
        }
        cliNode = slicer.cli.run(slicer.modules.createdicomseries, None, parameters, wait_for_completion=True)
        self.assertEqual(cliNode.GetStatus(), cliNode.Completed)

        with DICOMUtils.TemporaryDICOMDatabase(os.path.join(self.tempDir, "db")) as db:
            study.importFolder(dicomDir)
            rows = study.listCtSeries(db)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.info.number_of_images, phantom.volume.geometry.dimensions[2])
            self.assertAlmostEqual(row.assessment.slice_spacing_mm, 0.7, places=3)
            self.assertNotIn("gantry_tilt", row.assessment.warning_codes)
            self.assertNotIn("irregular_spacing", row.assessment.warning_codes)
            volume, regularized = study.loadSeries(row.info.series_uid)
            self.assertFalse(regularized)
            loaded = bridge.nodeToVolume(volume)
            self.assertTrue(loaded.geometry.is_close(phantom.volume.geometry, atol=1e-3))
            np.testing.assert_array_equal(loaded.data, phantom.volume.data)

    def test_volume_geometry_check(self):
        from aortaplan_core.phantoms import straight_tube
        from aortaplan_core.study import assess_volume_geometry
        from AortaPlanLib.bridge import nodeGeometry, volumeToNode

        phantom = straight_tube(diameter_mm=10.0, length_mm=40.0, supersampling=2)
        geometry = nodeGeometry(volumeToNode(phantom.volume, "phantom"))
        self.assertTrue(geometry.is_close(phantom.volume.geometry))
        self.assertEqual(assess_volume_geometry(geometry), [])


class AortaPlanNavigationTests(unittest.TestCase):
    """Runs without a main window too: the navigator then creates the slice nodes itself."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        slicer.mrmlScene.Clear(0)

    def test_slices_and_cursor_follow_the_curve(self):
        from aortaplan_core.centerline.navigation import CurvePath

        phantom, volume, curve, navigator = _torusSetup()
        path = CurvePath(slicer.util.arrayFromMarkupsCurvePoints(curve, world=True))
        self.assertAlmostEqual(navigator.path.length, path.length, places=6)
        for s in (5.0, path.length / 2.0, path.length - 5.0):
            navigator.setPosition(s)
            cross = slicer.util.arrayFromVTKMatrix(navigator.crossNode.GetSliceToRAS())
            np.testing.assert_allclose(cross, path.cross_section_frame(s).slice_to_ras(), atol=1e-6)
            longitudinal = slicer.util.arrayFromVTKMatrix(navigator.longitudinalNode.GetSliceToRAS())
            np.testing.assert_allclose(longitudinal, path.longitudinal_frame(s, 0.0).slice_to_ras(), atol=1e-6)
            np.testing.assert_allclose(_cursorPosition(navigator), path.point_at(s), atol=1e-6)
            # The cursor (slice centre) is inside the phantom lumen.
            self.assertGreater(phantom.volume.sample_linear(path.point_at(s)), 250.0)
        navigator.cleanup()

    def test_anterior_up_and_patient_left_on_the_right(self):
        from AortaPlanLib.navigation import SliceNavigator

        curve = _curveFromPoints(np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 0.0], [0.0, 0.0, -50.0]]), every=1)
        navigator = SliceNavigator()
        navigator.setCurve(curve)
        navigator.setPosition(30.0)  # RAS (0, 0, 20)
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(navigator.crossNode.GetXYToRAS(), rasToXy)
        centre = rasToXy.MultiplyPoint([0.0, 0.0, 20.0, 1.0])
        anterior = rasToXy.MultiplyPoint([0.0, 10.0, 20.0, 1.0])
        left = rasToXy.MultiplyPoint([-10.0, 0.0, 20.0, 1.0])
        self.assertGreater(anterior[1], centre[1] + 1)  # anterior is up on screen
        self.assertAlmostEqual(anterior[0], centre[0], places=3)
        self.assertGreater(left[0], centre[0] + 1)  # patient's left is on the right
        navigator.cleanup()

    def test_angle_clamping_and_click(self):
        phantom, volume, curve, navigator = _torusSetup()
        navigator.setAngle(370.0)
        self.assertAlmostEqual(navigator.angle, 10.0)
        navigator.setPosition(-5.0)
        self.assertEqual(navigator.position, 0.0)
        navigator.setPosition(1e6)
        self.assertAlmostEqual(navigator.position, navigator.path.length)
        navigator.setPosition(40.0)
        navigator.step(-2.5)
        self.assertAlmostEqual(navigator.position, 37.5)
        target = navigator.path.point_at(60.0) + np.array([3.0, 0.0, 0.0])  # 3 mm off the curve
        self.assertAlmostEqual(navigator.jumpToRas(target), 60.0, delta=0.3)
        self.assertAlmostEqual(navigator.position, 60.0, delta=0.3)
        navigator.cleanup()

    def test_curve_edits_and_standard_views(self):
        phantom, volume, curve, navigator = _torusSetup()
        length = navigator.path.length
        position = [0.0, 0.0, 0.0]
        curve.GetNthControlPointPosition(0, position)
        curve.SetNthControlPointPosition(0, position[0], position[1], position[2] - 10.0)
        self.assertGreater(navigator.path.length, length + 5.0)
        navigator.setFrozen(True)  # drawing: standard views, no navigation
        rotation = slicer.util.arrayFromVTKMatrix(navigator.crossNode.GetSliceToRAS())[:3, :3]
        np.testing.assert_allclose(rotation, np.diag([-1.0, 1.0, 1.0]), atol=1e-9)  # axial
        navigator.setFrozen(False)
        self.assertIsNotNone(navigator.path)
        navigator.setCurve(None)
        self.assertIsNone(navigator.path)
        navigator.cleanup()


@unittest.skipIf(slicer.util.mainWindow() is None, "views and interaction need the main window (--slicer-gui)")
class AortaPlanViewTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.previousModule = slicer.util.moduleSelector().selectedModule

    def tearDown(self):
        slicer.util.selectModule(self.previousModule or "Welcome")
        slicer.mrmlScene.Clear(0)

    def _openModuleWithTorus(self):
        from aortaplan_core.phantoms import torus_arc
        from AortaPlanLib.bridge import volumeToNode

        slicer.util.selectModule("AortaPlan")
        widget = slicer.modules.aortaplan.widgetRepresentation().self()
        phantom = torus_arc(leg_length_mm=10.0, supersampling=2)
        volume = volumeToNode(phantom.volume, "torus")
        widget.ui.volumeSelector.setCurrentNode(volume)
        curve = _curveFromPoints(phantom.truth.centerline_points)
        widget.ui.curveSelector.setCurrentNode(curve)
        slicer.app.processEvents()
        slicer.util.forceRenderAllViews()
        return widget, phantom, volume, curve

    def test_layout_switch_and_restore(self):
        from AortaPlanLib import layout

        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        controller = layout.LayoutController()
        controller.activate()
        self.assertEqual(layoutManager.layout, layout.LAYOUT_ID)
        self.assertIsNotNone(layout.sliceWidget(layout.CROSS_VIEW))
        self.assertIsNotNone(layout.sliceWidget(layout.LONGITUDINAL_VIEW))
        controller.activate(withAxial=True)
        self.assertEqual(layoutManager.layout, layout.LAYOUT_WITH_AXIAL_ID)
        controller.deactivate()
        self.assertEqual(layoutManager.layout, slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    def test_module_gui_drives_navigation(self):
        widget, phantom, volume, curve = self._openModuleWithTorus()
        navigator = widget.navigator
        self.assertIsNotNone(navigator.path)
        self.assertIs(widget._parameterNode.inputVolume, volume)
        self.assertAlmostEqual(widget.ui.positionSliderWidget.maximum, navigator.path.length, places=3)
        widget.ui.positionSliderWidget.value = 50.0
        self.assertAlmostEqual(navigator.position, 50.0, places=3)
        widget.ui.rotationSliderWidget.value = 90.0
        self.assertAlmostEqual(navigator.angle, 90.0)
        self.assertAlmostEqual(widget._parameterNode.positionMm, 50.0, places=3)
        widget.ui.wlCalciumButton.click()
        self.assertEqual(volume.GetDisplayNode().GetWindow(), 1500.0)
        widget.setCurrentStep(3)
        self.assertEqual(widget.ui.stepStack.currentIndex, 3)
        widget.ui.backButton.click()
        self.assertEqual(widget.ui.stepStack.currentIndex, 2)

    def test_wheel_keys_and_clicks(self):
        from AortaPlanLib import layout

        widget, phantom, volume, curve = self._openModuleWithTorus()
        navigator = widget.navigator
        navigator.setPosition(50.0)
        cross = layout.sliceWidget(layout.CROSS_VIEW).sliceView().interactor()
        cross.SetEventInformation(10, 10, 0, 0)
        cross.InvokeEvent(vtk.vtkCommand.MouseWheelForwardEvent)
        self.assertAlmostEqual(navigator.position, 51.0, places=3)
        cross.SetEventInformation(10, 10, 0, 1)  # Shift
        cross.InvokeEvent(vtk.vtkCommand.MouseWheelBackwardEvent)
        self.assertAlmostEqual(navigator.position, 46.0, places=3)
        cross.SetEventInformation(10, 10, 0, 0, "\0", 0, "Down")
        cross.InvokeEvent(vtk.vtkCommand.KeyPressEvent)
        self.assertAlmostEqual(navigator.position, 45.0, places=3)

        # Click in the longitudinal view on the curve point at 30 mm.
        target = navigator.path.point_at(30.0)
        longitudinal = layout.sliceWidget(layout.LONGITUDINAL_VIEW).sliceView().interactor()
        navigator.setPosition(30.0)  # the target must lie in the displayed plane
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(navigator.longitudinalNode.GetXYToRAS(), rasToXy)
        navigator.position = 0.0  # move away without re-slicing, then click back
        x, y = rasToXy.MultiplyPoint([*target, 1.0])[:2]
        longitudinal.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        longitudinal.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
        longitudinal.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        self.assertAlmostEqual(navigator.position, 30.0, delta=1.5)

        # Click on the curve in the 3D view.
        threeDView = layout.threeDWidget().threeDView()
        threeDView.resetFocalPoint()
        slicer.util.forceRenderAllViews()
        renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
        target = navigator.path.point_at(80.0)
        renderer.SetWorldPoint(*target, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        interactor = threeDView.interactor()
        interactor.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        interactor.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
        interactor.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        self.assertAlmostEqual(navigator.position, 80.0, delta=3.0)

    def test_scrolling_speed(self):
        import time

        widget, phantom, volume, curve = self._openModuleWithTorus()
        navigator = widget.navigator
        positions = np.linspace(5.0, navigator.path.length - 5.0, 40)
        start = time.perf_counter()
        for s in positions:
            navigator.setPosition(s)
            slicer.util.forceRenderAllViews()
        perFrame = (time.perf_counter() - start) / len(positions)
        self.assertLess(perFrame, 1.0 / 15.0, f"{perFrame * 1000:.1f} ms per update")
