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
