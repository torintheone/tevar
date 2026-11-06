"""Display helpers: window/level presets and large text overlays in views."""

import os
import sys

import slicer
import vtk

from aortaplan_core.display import window_level_preset

_FONT_CANDIDATES = []
if sys.platform.startswith("win"):
    _fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    _FONT_CANDIDATES += [os.path.join(_fonts, name) for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "tahoma.ttf")]
elif sys.platform == "darwin":
    _FONT_CANDIDATES += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial.ttf"]
else:
    _FONT_CANDIDATES += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]


def unicodeFontFile():
    """A system TrueType font with Cyrillic glyphs (VTK's built-in fonts may lack them), or None."""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def applyWindowLevel(volumeNode, presetKey):
    """Set a window/level preset on a scalar volume."""
    if volumeNode is None:
        return
    preset = window_level_preset(presetKey)
    displayNode = volumeNode.GetDisplayNode()
    if displayNode is None:
        volumeNode.CreateDefaultDisplayNodes()
        displayNode = volumeNode.GetDisplayNode()
    displayNode.SetAutoWindowLevel(False)
    displayNode.SetWindowLevel(preset.window, preset.level)


TOP_RIGHT = "top_right"
BOTTOM_LEFT = "bottom_left"


class ViewOverlay:
    """Large text in a corner of a slice or 3D view (independent of Slicer's annotations)."""

    def __init__(self, view, fontSize=22, corner=TOP_RIGHT, color=(1.0, 0.85, 0.2)):
        self.view = view
        self.fontFile = unicodeFontFile()
        self.actor = vtk.vtkTextActor()
        prop = self.actor.GetTextProperty()
        prop.SetFontSize(fontSize)
        prop.SetBold(True)
        prop.SetShadow(True)
        prop.SetColor(*color)
        if self.fontFile:
            prop.SetFontFamily(vtk.VTK_FONT_FILE)
            prop.SetFontFile(self.fontFile)
        coordinate = self.actor.GetPositionCoordinate()
        coordinate.SetCoordinateSystemToNormalizedViewport()
        if corner == BOTTOM_LEFT:
            prop.SetJustificationToLeft()
            prop.SetVerticalJustificationToBottom()
            coordinate.SetValue(0.02, 0.04)
        else:
            prop.SetJustificationToRight()
            prop.SetVerticalJustificationToTop()
            coordinate.SetValue(0.98, 0.96)
        self.renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        self.renderer.AddActor2D(self.actor)

    def setText(self, text):
        if not self.fontFile:
            text = text.replace("мм", "mm")  # no font with Cyrillic glyphs found
        self.actor.SetInput(text)
        self.view.scheduleRender()

    def remove(self):
        if self.renderer is not None:
            self.renderer.RemoveActor2D(self.actor)
            self.renderer = None
            self.view.scheduleRender()


def setSliceVisibleIn3D(sliceNode, visible):
    if sliceNode is not None:
        sliceNode.SetSliceVisible(bool(visible))


def forceRender():
    layoutManager = slicer.app.layoutManager()
    if layoutManager is not None:
        slicer.util.forceRenderAllViews()
