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
import qt
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
        from aortaplan_core.centerline.cpr import StraightenedCpr
        from aortaplan_core.centerline.navigation import CurvePath

        phantom, volume, curve, navigator = _torusSetup()
        path = CurvePath(slicer.util.arrayFromMarkupsCurvePoints(curve, world=True))
        cpr = StraightenedCpr(path)
        self.assertAlmostEqual(navigator.path.length, path.length, places=6)
        images = [slicer.util.arrayFromVolume(view.volumeNode(create=False))[0] for view in navigator.cprViews]
        np.testing.assert_allclose(images[0], cpr.sample(phantom.volume, 0.0), atol=1e-3)
        np.testing.assert_allclose(images[1], cpr.sample(phantom.volume, 90.0), atol=1e-3)
        for s in (5.0, path.length / 2.0, path.length - 5.0):
            navigator.setPosition(s)
            cross = slicer.util.arrayFromVTKMatrix(navigator.crossNode.GetSliceToRAS())
            np.testing.assert_allclose(cross, path.cross_section_frame(s).slice_to_ras(), atol=1e-6)
            np.testing.assert_allclose(_cursorPosition(navigator), path.point_at(s), atol=1e-6)
            for view in navigator.cprViews:  # horizontal position line at the row of s
                line = slicer.util.arrayFromMarkupsControlPoints(view.positionLine(), world=True)
                _, row = cpr.cpr_to_pixel(0.0, s)
                np.testing.assert_allclose(line[:, 1], view.origin[1] + row * cpr.row_spacing_mm, atol=1e-6)
                axis = slicer.util.arrayFromMarkupsControlPoints(navigator.axisLine(view), world=True)
                self.assertLess(abs(np.dot(axis[1] - axis[0], path.tangent_at(s))), 1e-6)  # in the slice plane
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

    def test_cpr_rotation_and_pixel_mapping(self):
        from aortaplan_core.centerline.cpr import StraightenedCpr

        phantom, volume, curve, navigator = _torusSetup()
        navigator.setAngle(30.0)
        cpr = StraightenedCpr(navigator.path)
        first, second = (slicer.util.arrayFromVolume(v.volumeNode(create=False))[0] for v in navigator.cprViews)
        np.testing.assert_allclose(first, cpr.sample(phantom.volume, 30.0), atol=1e-3)
        np.testing.assert_allclose(second, cpr.sample(phantom.volume, 120.0), atol=1e-3)
        view = navigator.cprViews[0]
        x, y = view.xyFromCpr(2.0, 40.0)
        offset, s = view.cprFromXY(x, y)
        self.assertAlmostEqual(offset, 2.0, places=3)
        self.assertAlmostEqual(s, 40.0, places=3)
        navigator.jumpToCprPixel(0, x, y)
        self.assertAlmostEqual(navigator.position, 40.0, places=3)
        # The CPR pixel and the volume agree: the RAS point of a CPR point has the image value.
        ras = cpr.to_ras(2.0, 40.0, 30.0)
        column, row = cpr.cpr_to_pixel(2.0, 40.0)
        self.assertAlmostEqual(
            float(first[int(round(row)), int(round(column))]), float(phantom.volume.sample_linear(ras)), delta=40.0
        )
        navigator.cleanup()

    def test_mpr_and_path_modes(self):
        phantom, volume, curve, navigator = _torusSetup()
        navigator.setPosition(50.0)
        navigator.setMode("mpr")
        self.assertFalse(navigator.navigating)
        m = slicer.util.arrayFromVTKMatrix(navigator.crossNode.GetSliceToRAS())
        np.testing.assert_allclose(m[:3, :3], np.diag([-1.0, 1.0, 1.0]), atol=1e-9)  # axial
        np.testing.assert_allclose(m[:3, 3], navigator.path.point_at(50.0), atol=1e-6)  # through the point
        navigator.setMode("path")
        self.assertTrue(navigator.navigating)
        np.testing.assert_allclose(_cursorPosition(navigator), navigator.path.point_at(50.0), atol=1e-6)
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
        self.assertIsNotNone(layout.sliceWidget(layout.CPR2_VIEW))
        controller.setWithAxial(True)
        self.assertEqual(layoutManager.layout, layout.LAYOUT_ID)
        self.assertIsNotNone(layout.sliceWidget(layout.AXIAL_VIEW))
        controller.deactivate()
        self.assertEqual(layoutManager.layout, slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

    def test_layout_templates_and_swapping(self):
        import re

        from AortaPlanLib import layout

        controller = layout.LayoutController()
        controller.reset()
        controller.activate(withAxial=False)
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()

        def order():
            return re.findall(r'singletontag="([^"]+)"', layoutNode.GetCurrentLayoutDescription())

        self.assertEqual(order(), [layout.LONGITUDINAL_VIEW, layout.CPR2_VIEW, layout.CROSS_VIEW, "1"])
        controller.swap(layout.CROSS_VIEW, layout.LONGITUDINAL_VIEW)
        self.assertEqual(order()[0], layout.CROSS_VIEW)
        self.assertIsNotNone(layout.sliceWidget(layout.CROSS_VIEW))
        for template in layout.TEMPLATES:
            controller.setTemplate(template)
            self.assertEqual(sorted(order()), sorted([layout.LONGITUDINAL_VIEW, layout.CPR2_VIEW, layout.CROSS_VIEW, "1"]))
        # The arrangement is remembered for the next session.
        self.assertEqual(layout.LayoutController().order[0], layout.CROSS_VIEW)
        controller.reset()
        self.assertEqual(order()[0], layout.LONGITUDINAL_VIEW)
        controller.deactivate()

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

    def test_drag_view_onto_another_swaps_them(self):
        from AortaPlanLib import layout

        widget, phantom, volume, curve = self._openModuleWithTorus()
        widget.layoutController.reset()
        target = layout.viewWidget(layout.THREE_D_VIEW)
        center = target.mapToGlobal(qt.QPoint(target.width // 2, target.height // 2))
        found = widget.dragSwapper.tagAt(center)
        if found is not None:  # None only if another window covers Slicer during the test
            self.assertEqual(found, layout.THREE_D_VIEW)
        widget.dragSwapper.swap(layout.CPR2_VIEW, layout.THREE_D_VIEW)
        slicer.app.processEvents()
        self.assertEqual(widget.layoutController.order.index(layout.THREE_D_VIEW), 1)
        self.assertIsNotNone(widget.navigator.path)  # navigation survives the re-arrangement
        widget.layoutController.reset()

    def test_mpr_mode_like_radiant(self):
        from AortaPlanLib import layout

        widget, phantom, volume, curve = self._openModuleWithTorus()
        widget.navigator.setPosition(60.0)
        point = widget.navigator.path.point_at(60.0)
        widget.setViewMode("mpr")
        self.assertFalse(widget.navigator.navigating)
        cross = widget.navigator.crossNode
        np.testing.assert_allclose(slicer.util.arrayFromVTKMatrix(cross.GetSliceToRAS())[:3, :3], np.diag([-1, 1, 1]), atol=1e-9)
        displayNode = layout.sliceWidget(layout.CROSS_VIEW).sliceLogic().GetSliceDisplayNode()
        self.assertTrue(displayNode.GetIntersectingSlicesVisibility())
        self.assertTrue(displayNode.GetIntersectingSlicesInteractive())
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        self.assertNotEqual(crosshair.GetCrosshairBehavior(), slicer.vtkMRMLCrosshairNode.NoAction)
        # A click in the axial (perpendicular) view moves the coronal and sagittal slices to the point.
        slicer.app.processEvents()
        target = point + np.array([5.0, 3.0, 0.0])
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(cross.GetXYToRAS(), rasToXy)
        x, y = rasToXy.MultiplyPoint([*target, 1.0])[:2]
        interactor = layout.sliceWidget(layout.CROSS_VIEW).sliceView().interactor()
        interactor.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        interactor.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
        interactor.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        clicked = np.array(cross.GetXYToRAS().MultiplyPoint([int(round(x)), int(round(y)), 0.0, 1.0])[:3])
        for name in layout.CPR_VIEWS:  # coronal and sagittal in MPR mode
            m = slicer.util.arrayFromVTKMatrix(layout.ensureSliceNode(name).GetSliceToRAS())
            self.assertLess(abs(np.dot(clicked - m[:3, 3], m[:3, 2])), 0.5)
        widget.setViewMode("path")
        self.assertTrue(widget.navigator.navigating)
        self.assertEqual(crosshair.GetCrosshairBehavior(), slicer.vtkMRMLCrosshairNode.NoAction)
        self.assertFalse(displayNode.GetIntersectingSlicesVisibility())

    def test_rotate_cpr_by_dragging_its_line(self):
        from AortaPlanLib import layout

        widget, phantom, volume, curve = self._openModuleWithTorus()
        navigator = widget.navigator
        navigator.setPosition(40.0)
        navigator.setAngle(0.0)
        slicer.app.processEvents()
        cross = navigator.crossNode
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(cross.GetXYToRAS(), rasToXy)
        line = slicer.util.arrayFromMarkupsControlPoints(navigator.axisLine(navigator.cprViews[0]), world=True)
        x, y = rasToXy.MultiplyPoint([*line[1], 1.0])[:2]
        interactor = layout.sliceWidget(layout.CROSS_VIEW).sliceView().interactor()
        interactor.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        interactor.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
        target = navigator.cpr.to_ras(20.0, 40.0, 40.0)  # a point on the 40 deg line
        x, y = rasToXy.MultiplyPoint([*target, 1.0])[:2]
        interactor.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        interactor.InvokeEvent(vtk.vtkCommand.MouseMoveEvent)
        interactor.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        self.assertAlmostEqual(navigator.angle, 40.0, delta=3.0)

    def test_shift_does_not_jump_slices(self):
        from AortaPlanLib import layout

        widget, phantom, volume, curve = self._openModuleWithTorus()
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        self.assertEqual(crosshair.GetCrosshairBehavior(), slicer.vtkMRMLCrosshairNode.NoAction)
        widget.navigator.setPosition(40.0)
        before = slicer.util.arrayFromVTKMatrix(widget.navigator.crossNode.GetSliceToRAS())
        longitudinal = layout.sliceWidget(layout.LONGITUDINAL_VIEW).sliceView().interactor()
        longitudinal.SetEventInformation(30, 40, 0, 1)  # Shift + mouse move
        longitudinal.InvokeEvent(vtk.vtkCommand.MouseMoveEvent)
        crosshair.SetCrosshairRAS(*widget.navigator.path.point_at(80.0))  # e.g. Shift+move in another view
        after = slicer.util.arrayFromVTKMatrix(widget.navigator.crossNode.GetSliceToRAS())
        np.testing.assert_allclose(after, before, atol=1e-9)

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

        # CPR view: press at 30 mm, drag to 60 mm, release.
        view = navigator.cprViews[0]
        cpr = layout.sliceWidget(layout.LONGITUDINAL_VIEW).sliceView().interactor()
        x, y = view.xyFromCpr(0.0, 30.0)
        cpr.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        cpr.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
        self.assertAlmostEqual(navigator.position, 30.0, delta=1.5)
        x, y = view.xyFromCpr(0.0, 60.0)
        cpr.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        cpr.InvokeEvent(vtk.vtkCommand.MouseMoveEvent)
        self.assertAlmostEqual(navigator.position, 60.0, delta=1.5)
        cpr.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        x, y = view.xyFromCpr(0.0, 90.0)
        cpr.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        cpr.InvokeEvent(vtk.vtkCommand.MouseMoveEvent)  # no button: no change
        self.assertAlmostEqual(navigator.position, 60.0, delta=1.5)
        # Wheel in the CPR view moves; Ctrl+wheel rotates.
        cpr.InvokeEvent(vtk.vtkCommand.MouseWheelForwardEvent)
        self.assertAlmostEqual(navigator.position, 61.0, delta=1.5)
        angle = navigator.angle
        cpr.SetEventInformation(int(round(x)), int(round(y)), 1, 0)  # Ctrl
        cpr.InvokeEvent(vtk.vtkCommand.MouseWheelForwardEvent)
        self.assertAlmostEqual(navigator.angle, (angle + 5.0) % 360.0)

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


# ---------------------------------------------------------------------------- stage 2

#: Measurements reported by run_slicer_tests.py (e.g. VMTK vs core centerline accuracy).
METRICS = {}


def _aortoiliacSetup():
    from aortaplan_core.phantoms.branching import aortoiliac, suggested_seeds
    from AortaPlanLib.bridge import volumeToNode
    from AortaPlanLib.centerline import ENDS, START, CenterlineBuilder

    phantom = aortoiliac(with_bone=True, with_calcifications=True)
    volume = volumeToNode(phantom.volume, "aortoiliac")
    builder = CenterlineBuilder()
    start, ends = suggested_seeds(phantom.truth)
    builder.pointsNode(START).AddControlPoint(*(start + np.array([6.0, 0.0, 0.0])), "")  # a click on the wall
    for end in ends:
        builder.pointsNode(ENDS).AddControlPoint(*end, "")
    return phantom, volume, builder


def _boneVoxels(phantom, mask):
    g = phantom.volume.geometry
    vertebra = [o for o in phantom.truth.objects if o["kind"] == "vertebra"][0]
    pts = g.voxel_centers_ras()
    r = np.hypot(pts[..., 0] - vertebra["axis_point"][0], pts[..., 1] - vertebra["axis_point"][1])
    zr = vertebra["z_range_mm"]
    inside = (r < vertebra["radius_mm"] - 1.5) & (pts[..., 2] > zr[0] + 5) & (pts[..., 2] < zr[1] - 5)
    return int((mask & inside).sum())


class AortaPlanCenterlineTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        slicer.mrmlScene.Clear(0)

    def _buildAndMeasure(self, method):
        import time

        from aortaplan_core.centerline.metrics import centerline_deviation
        from AortaPlanLib.centerline import LUMEN_SEGMENT_ID, PATH_ATTRIBUTE

        phantom, volume, builder = _aortoiliacSetup()
        start = time.perf_counter()
        curves = builder.build(volume, method)
        METRICS[f"{method}: build time, s"] = round(time.perf_counter() - start, 2)
        self.assertEqual(len(curves), 2)
        segmentation = builder.lumenSegmentation(create=False)
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, LUMEN_SEGMENT_ID, volume) > 0
        self.assertEqual(_boneVoxels(phantom, mask), 0)  # no leak into the vertebra
        truth = phantom.truth
        exclusions = [(j["point"], j["exclusion_radius_mm"]) for j in truth.junctions]
        results = {}
        for curve in curves:
            side = curve.GetAttribute(PATH_ATTRIBUTE)
            points = slicer.util.arrayFromMarkupsCurvePoints(curve, world=True)
            dev = centerline_deviation(points, truth.path_points(side), exclusions)
            METRICS[f"{method}: {side} path deviation mean/max, mm"] = f"{dev['mean_mm']:.3f} / {dev['max_mm']:.3f}"
            results[side] = dev
        self.assertEqual(sorted(results), ["left", "right"])
        return results, builder, volume

    def test_core_minimal_path_on_phantom(self):
        results, _, _ = self._buildAndMeasure("minimal_path")
        for dev in results.values():
            self.assertLessEqual(dev["mean_mm"], 0.5)
            self.assertLessEqual(dev["max_mm"], 1.0)

    @unittest.skipIf(not hasattr(slicer.modules, "extractcenterline"), "SlicerVMTK is not installed")
    def test_vmtk_on_phantom(self):
        results, builder, _ = self._buildAndMeasure("vmtk")
        self.assertEqual(builder.lastNetwork.method, "vmtk")
        for dev in results.values():
            self.assertLess(dev["mean_mm"], 1.0)  # comparison values are reported in METRICS

    def test_rebuild_after_editing_the_lumen(self):
        from AortaPlanLib.centerline import LUMEN_SEGMENT_ID

        phantom, volume, builder = _aortoiliacSetup()
        builder.build(volume, "minimal_path")
        segmentation = builder.lumenSegmentation(create=False)
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, LUMEN_SEGMENT_ID, volume)
        mask[:, :, mask.shape[2] // 2 + 8 :] = 0  # erase the left iliac (high i = patient left)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentation, LUMEN_SEGMENT_ID, volume)
        curves = builder.rebuildCenterline(volume, "minimal_path")
        self.assertEqual(len(curves), 1)
        self.assertIn("end_unreachable:1", builder.lastNetwork.warnings)

    def test_navigation_on_built_path(self):
        from AortaPlanLib.navigation import SliceNavigator

        phantom, volume, builder = _aortoiliacSetup()
        curves = builder.build(volume, "minimal_path")
        navigator = SliceNavigator()
        navigator.setVolume(volume)
        navigator.setCurve(curves[0])
        self.assertAlmostEqual(navigator.path.length, curves[0].GetCurveLengthWorld(), delta=2.0)
        navigator.setPosition(50.0)
        self.assertGreater(phantom.volume.sample_linear(navigator.path.point_at(50.0)), 250.0)
        navigator.cleanup()

    def test_point_placement_modes(self):
        from AortaPlanLib.centerline import ENDS, START, CenterlineBuilder

        builder = CenterlineBuilder()
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        selection = slicer.app.applicationLogic().GetSelectionNode()
        builder.startPlacing(START)
        self.assertEqual(interaction.GetCurrentInteractionMode(), interaction.Place)
        self.assertEqual(selection.GetActivePlaceNodeID(), builder.pointsNode(START).GetID())
        self.assertEqual(interaction.GetPlaceModePersistence(), 0)
        builder.startPlacing(ENDS)
        self.assertEqual(interaction.GetPlaceModePersistence(), 1)
        builder.stopPlacing()
        self.assertEqual(interaction.GetCurrentInteractionMode(), interaction.ViewTransform)


@unittest.skipIf(slicer.util.mainWindow() is None, "the module GUI needs the main window (--slicer-gui)")
class AortaPlanAutoCenterlineGuiTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        slicer.util.selectModule("Welcome")
        slicer.mrmlScene.Clear(0)

    def test_build_button_creates_paths_and_navigation(self):
        from aortaplan_core.phantoms.branching import aortoiliac, suggested_seeds
        from AortaPlanLib.bridge import volumeToNode
        from AortaPlanLib.centerline import ENDS, START

        slicer.util.selectModule("AortaPlan")
        widget = slicer.modules.aortaplan.widgetRepresentation().self()
        phantom = aortoiliac()
        volume = volumeToNode(phantom.volume, "aortoiliac")
        widget.ui.volumeSelector.setCurrentNode(volume)
        widget.setCurrentStep(1)
        start, ends = suggested_seeds(phantom.truth)
        widget.builder.pointsNode(START).AddControlPoint(*start, "")
        for end in ends:
            widget.builder.pointsNode(ENDS).AddControlPoint(*end, "")
        widget._updatePointsInfo()
        self.assertTrue(widget.ui.buildButton.enabled)
        widget.ui.methodComboBox.currentIndex = widget.ui.methodComboBox.findData("minimal_path")
        widget.ui.buildButton.click()
        self.assertEqual(widget.ui.pathComboBox.count, 2)
        self.assertIsNotNone(widget.navigator.path)
        self.assertIsNotNone(widget._parameterNode.centerline)
        widget.ui.pathComboBox.currentIndex = 1
        self.assertIs(widget.navigator.curveNode, slicer.mrmlScene.GetNodeByID(widget.ui.pathComboBox.itemData(1)))


# ---------------------------------------------------------------------------- stage 4

_ANEURYSM = {}


def _aneurysmPhantom():
    """Built once per test run (about 6 s)."""
    if "phantom" not in _ANEURYSM:
        from aortaplan_core.phantoms import fusiform_aneurysm

        _ANEURYSM["phantom"] = fusiform_aneurysm()
    return _ANEURYSM["phantom"]


def _aneurysmSetup():
    from AortaPlanLib.bridge import volumeToNode
    from AortaPlanLib.measurements import MeasurementController
    from AortaPlanLib.navigation import SliceNavigator

    phantom = _aneurysmPhantom()
    volume = volumeToNode(phantom.volume, "aneurysm")
    curve = _curveFromPoints(phantom.truth.centerline_points, name="AneurysmCurve", every=40)
    navigator = SliceNavigator()
    navigator.setVolume(volume)
    navigator.setCurve(curve)
    controller = MeasurementController(navigator)
    saved = []
    controller.persist = saved.append
    return phantom, volume, curve, navigator, controller, saved


def _lineLength(node):
    points = slicer.util.arrayFromMarkupsControlPoints(node, world=True)
    return float(np.linalg.norm(points[1] - points[0]))


class AortaPlanMeasurementTests(unittest.TestCase):
    """Stage 4 in Slicer: contours, editing, saved measurements, markers, tools (no main window needed)."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        slicer.mrmlScene.Clear(0)

    def test_section_contours_and_feret_lines(self):
        from aortaplan_core.crosssection import LUMEN, OUTER

        phantom, volume, curve, navigator, controller, saved = _aneurysmSetup()
        s = phantom.truth.expected["max_outer_s_mm"]
        navigator.setPosition(s)
        controller.setActive(True)
        metrics = controller.sectionMetrics()
        self.assertAlmostEqual(metrics[OUTER].max_diameter_mm, 60.0, delta=0.3)
        self.assertAlmostEqual(metrics[OUTER].min_diameter_mm, 54.0, delta=0.3)
        self.assertAlmostEqual(metrics[LUMEN].mean_diameter_mm, 30.0, delta=0.3)
        outer, lumen = metrics[OUTER], metrics[LUMEN]
        METRICS["aneurysm in Slicer: outer max/min, lumen mean, mm"] = (
            f"{outer.max_diameter_mm:.2f} / {outer.min_diameter_mm:.2f} / {lumen.mean_diameter_mm:.2f}"
        )
        normal = navigator.path.tangent_at(s)
        origin = navigator.path.point_at(s)
        for kind in (LUMEN, OUTER):
            node = controller.contourNode(kind, create=False)
            self.assertIsNotNone(node)
            self.assertTrue(node.GetDisplayNode().GetVisibility())
            self.assertGreaterEqual(node.GetNumberOfControlPoints(), 12)
            points = slicer.util.arrayFromMarkupsControlPoints(node, world=True)
            np.testing.assert_allclose((points - origin) @ normal, 0.0, atol=1e-6)  # in the section plane
            # Shown only in the perpendicular view.
            display = node.GetDisplayNode()
            self.assertEqual(display.GetNumberOfViewNodeIDs(), 1)
            self.assertEqual(display.GetNthViewNodeID(0), navigator.crossNode.GetID())
            m = metrics[kind]
            self.assertAlmostEqual(_lineLength(controller.feretNode(kind, "max")), m.max_diameter_mm, places=3)
            self.assertAlmostEqual(_lineLength(controller.feretNode(kind, "min")), m.min_diameter_mm, places=3)
        controller.setActive(False)
        self.assertFalse(controller.contourNode(OUTER, create=False).GetDisplayNode().GetVisibility())

    def test_edit_contour_autosave_and_restore(self):
        from aortaplan_core.crosssection import OUTER, MeasurementStore

        phantom, volume, curve, navigator, controller, saved = _aneurysmSetup()
        navigator.setPosition(100.0)
        controller.setActive(True)
        before = controller.sectionMetrics()[OUTER].area_mm2
        node = controller.contourNode(OUTER, create=False)
        points = slicer.util.arrayFromMarkupsControlPoints(node, world=True)
        centre = points.mean(axis=0)
        moved = points[0] + 5.0 * (points[0] - centre) / np.linalg.norm(points[0] - centre)
        node.SetNthControlPointPositionWorld(0, *moved)  # like dragging a point
        self.assertTrue(controller.dirty)
        self.assertEqual(controller.current.outer.source, "manual")
        edited = controller.sectionMetrics()[OUTER].area_mm2
        self.assertGreater(edited, before + 20.0)
        # Moving on saves the edit; coming back shows it again.
        navigator.setPosition(120.0)
        controller.update()
        self.assertEqual(len(controller.store.measurements), 1)
        self.assertFalse(controller.dirty)
        navigator.setPosition(100.0)
        controller.update()
        self.assertIsNotNone(controller.currentName)
        self.assertAlmostEqual(controller.sectionMetrics()[OUTER].area_mm2, edited, delta=0.5)
        store = MeasurementStore.from_json(saved[-1])
        self.assertEqual(store.measurements[0].measurement.outer.source, "manual")
        # "Recompute" discards the edit.
        controller.recompute()
        self.assertEqual(len(controller.store.measurements), 0)
        self.assertAlmostEqual(controller.sectionMetrics()[OUTER].area_mm2, before, delta=1.0)

    def test_markers_ticks_distances_and_evar(self):
        from aortaplan_core.crosssection import MeasurementStore
        from aortaplan_core.i18n import tr

        phantom, volume, curve, navigator, controller, saved = _aneurysmSetup()
        controller.addMarker(tr("marker.lowest_renal"), 30.0)
        controller.addMarker(tr("marker.neck_end"), 55.0)
        navigator.setPosition(180.0)
        controller.addMarker(tr("marker.aortic_bifurcation"))  # at the current position
        markers = controller.markers()
        self.assertEqual([m.key for m in markers], ["lowest_renal", "neck_end", "aortic_bifurcation"])
        self.assertAlmostEqual(markers[2].arc_length_mm, 180.0, delta=0.05)
        for view in navigator.cprViews:  # ticks with names on both CPRs, in the CPR plane
            ticks = controller.tickNode(view, create=False)
            self.assertEqual(ticks.GetNumberOfControlPoints(), 3)
            self.assertEqual(ticks.GetNthControlPointLabel(1), tr("marker.neck_end"))
            points = slicer.util.arrayFromMarkupsControlPoints(ticks, world=True)
            np.testing.assert_allclose(points[:, 2], view.origin[2], atol=1e-6)
        distance, curvature = controller.distance(0, 2)
        self.assertAlmostEqual(distance.along_mm, 150.0, delta=0.1)
        self.assertAlmostEqual(distance.straight_mm, 150.0, delta=0.1)
        self.assertAlmostEqual(curvature.greater_mm, 150.0, delta=0.5)  # straight aorta
        summary = "\n".join(controller.evarSummary())
        self.assertIn(tr("evar.alpha", value="0,0"), summary)
        self.assertIn(tr("evar.neck_length", value="25,0"), summary)
        controller.renameMarker(1, "Шейка")
        controller.moveMarker(1, 60.0)
        self.assertEqual(controller.markers()[1].name, "Шейка")
        self.assertAlmostEqual(controller.markers()[1].arc_length_mm, 60.0, delta=0.05)
        controller.deleteMarker(0)
        self.assertEqual(len(controller.markers()), 2)
        store = MeasurementStore.from_json(saved[-1])
        self.assertEqual([m.name for m in store.markers], ["Шейка", tr("marker.aortic_bifurcation")])

    def test_profile_and_maximum_between_markers(self):
        from aortaplan_core.crosssection import OUTER

        phantom, volume, curve, navigator, controller, saved = _aneurysmSetup()
        e = phantom.truth.expected
        profile = controller.computeProfile()
        s, value = profile.maximum(OUTER)
        self.assertAlmostEqual(s, e["max_outer_s_mm"], delta=4.0)
        result = controller.findMaximum(e["sac_start_s_mm"], e["sac_end_s_mm"])
        self.assertAlmostEqual(result.value_mm, 60.0, delta=0.3)
        self.assertAlmostEqual(navigator.position, result.arc_length_mm, places=3)
        marker = [m for m in controller.markers() if m.key == "max_diameter"]
        self.assertEqual(len(marker), 1)
        self.assertAlmostEqual(marker[0].arc_length_mm, result.arc_length_mm, delta=0.05)
        controller.findMaximum(e["sac_start_s_mm"], e["max_outer_s_mm"] - 20.0)  # moves the same marker
        self.assertEqual(len([m for m in controller.markers() if m.key == "max_diameter"]), 1)

    def test_length_tool_in_cpr(self):
        phantom, volume, curve, navigator, controller, saved = _aneurysmSetup()
        controller.startLength()
        self.assertFalse(controller.lengthClick(0, 5.0, 90.0))
        self.assertTrue(controller.lengthClick(0, -3.0, 40.0))
        a, b, length = controller.lengthResult
        self.assertEqual((a, b), (40.0, 90.0))
        self.assertAlmostEqual(length, 50.0)
        node = controller.lengthNode(navigator.cprViews[0], create=False)
        self.assertAlmostEqual(_lineLength(node), 50.0, delta=0.6)  # drawn in the CPR plane
        self.assertIn("50,0", node.GetName())


@unittest.skipIf(slicer.util.mainWindow() is None, "the module GUI needs the main window (--slicer-gui)")
class AortaPlanMeasurementGuiTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        slicer.util.selectModule("Welcome")
        slicer.mrmlScene.Clear(0)

    def _openWithAneurysm(self):
        from AortaPlanLib.bridge import volumeToNode

        slicer.util.selectModule("AortaPlan")
        widget = slicer.modules.aortaplan.widgetRepresentation().self()
        phantom = _aneurysmPhantom()
        volume = volumeToNode(phantom.volume, "aneurysm")
        widget.ui.volumeSelector.setCurrentNode(volume)
        curve = _curveFromPoints(phantom.truth.centerline_points, name="AneurysmCurve", every=40)
        widget.ui.curveSelector.setCurrentNode(curve)
        widget.setCurrentStep(2)
        widget.navigator.setPosition(100.0)
        widget.measurements.update(force=True)
        slicer.app.processEvents()
        slicer.util.forceRenderAllViews()
        return widget, phantom

    def test_measurement_step(self):
        import time

        from aortaplan_core.i18n import tr
        from AortaPlanLib import layout

        widget, phantom = self._openWithAneurysm()
        ui = widget.ui
        self.assertTrue(widget.measurementsPanel.visible)
        self.assertNotEqual(ui.metricsTable.item(1, 1).text(), "—")  # outer max diameter shown
        self.assertIn("60", ui.metricsTable.item(1, 1).text())
        # M places a marker with the selected name (shortcut and view key).
        ui.markerNameComboBox.currentIndex = ui.markerNameComboBox.findText(tr("marker.max_diameter"))
        widget.interaction._markerKey()
        self.assertEqual(ui.markersTable.rowCount, 1)
        widget.interaction._lastMarkerKey = 0.0
        cross = layout.sliceWidget(layout.CROSS_VIEW).sliceView().interactor()
        cross.SetEventInformation(10, 10, 0, 0, "m", 0, "m")
        cross.InvokeEvent(vtk.vtkCommand.KeyPressEvent)
        self.assertEqual(ui.markersTable.rowCount, 2)
        # Marker by a click on the CPR.
        ui.placeMarkerByClickButton.checked = True
        view = widget.navigator.cprViews[0]
        cpr = layout.sliceWidget(layout.LONGITUDINAL_VIEW).sliceView().interactor()
        x, y = view.xyFromCpr(0.0, 70.0)
        cpr.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
        cpr.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
        cpr.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        self.assertFalse(ui.placeMarkerByClickButton.checked)
        self.assertEqual(ui.markersTable.rowCount, 3)
        self.assertAlmostEqual(widget.navigator.position, 70.0, delta=1.5)
        # Length tool: two clicks on the CPR.
        ui.lengthToolButton.checked = True
        for s in (40.0, 90.0):
            x, y = view.xyFromCpr(0.0, s)
            cpr.SetEventInformation(int(round(x)), int(round(y)), 0, 0)
            cpr.InvokeEvent(vtk.vtkCommand.LeftButtonPressEvent)
            cpr.InvokeEvent(vtk.vtkCommand.LeftButtonReleaseEvent)
        self.assertFalse(ui.lengthToolButton.checked)
        self.assertAlmostEqual(widget.measurements.lengthResult[2], 50.0, delta=1.5)
        self.assertIn(tr("tools.length.result", length="", a="", b="")[:10], ui.lengthResultLabel.text)
        # A press on a contour control point is left to Slicer (no CPR rotation, no position jump).
        widget.navigator.setPosition(100.0)
        widget.measurements.update(force=True)
        node = widget.measurements.contourNode("outer", create=False)
        point = slicer.util.arrayFromMarkupsControlPoints(node, world=True)[0]
        rasToXy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(widget.navigator.crossNode.GetXYToRAS(), rasToXy)
        x, y = rasToXy.MultiplyPoint([*point, 1.0])[:2]
        self.assertTrue(widget.measurements.isOverContourPoint(layout.CROSS_VIEW, x, y))
        # The plot: compute, click jumps.
        ui.computeProfileButton.click()
        self.assertIsNotNone(widget.measurements.profile)
        widget.measurementsPanel.onPlotClicked(120.0)
        self.assertAlmostEqual(widget.navigator.position, 120.0, places=3)
        # Saved measurements survive in the parameter node.
        ui.saveMeasurementButton.click()
        self.assertEqual(ui.savedTable.rowCount, 1)
        self.assertIn('"measurements"', widget._parameterNode.measurementsJson)
        # Scrolling with live contours (informational number; the contour follows after a short pause).
        positions = np.linspace(20.0, 170.0, 20)
        start = time.perf_counter()
        for s in positions:
            widget.navigator.setPosition(s)
            widget.measurements.update()
            slicer.util.forceRenderAllViews()
        perFrame = (time.perf_counter() - start) / len(positions)
        METRICS["live contours: update + render per section, ms"] = round(perFrame * 1000, 1)
        self.assertLess(perFrame, 0.25)
