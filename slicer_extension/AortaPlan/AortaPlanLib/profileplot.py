"""Diameter plot along the centerline, drawn with QPainter into a QLabel (stage 4).

x = arc length (mm), y = diameter (mm). Curves: lumen maximum and minimum diameter, outer
wall maximum diameter. A vertical line marks the current position, short ticks with names
mark the markers. A click on the plot reports the arc length under the mouse.
"""

import math

import numpy as np
import qt

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


CURVES = (  # (layer, metric, colour)
    ("lumen", "max_diameter_mm", "#d03030"),
    ("lumen", "min_diameter_mm", "#e08a20"),
    ("outer", "max_diameter_mm", "#2f74c0"),
)
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 42, 10, 22, 26


def niceStep(span, targetTicks=5):
    """A 1/2/5 x 10^n tick step giving about ``targetTicks`` ticks over ``span``."""
    if span <= 0:
        return 1.0
    raw = span / max(targetTicks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if factor * magnitude >= raw:
            return factor * magnitude
    return 10 * magnitude


class _PlotEventFilter(qt.QObject):
    def __init__(self, plot):
        qt.QObject.__init__(self)
        self.plot = plot

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
            self.plot._clicked(event.pos().x())
            return True
        if kind == qt.QEvent.Resize:
            qt.QTimer.singleShot(0, self.plot.render)
        return False


class ProfilePlot:
    def __init__(self, label, onClick=None):
        self.label = label
        self.onClick = onClick
        self.profile = None
        self.length = 0.0
        self.position = None
        self.markers = []  # (s, name)
        self.label.setMinimumHeight(200)
        self.label.setAlignment(qt.Qt.AlignCenter)
        self.label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self._filter = _PlotEventFilter(self)
        self.label.installEventFilter(self._filter)
        self.label.setCursor(qt.QCursor(qt.Qt.PointingHandCursor))

    # ------------------------------------------------------------------ data
    def setData(self, profile, length, position=None, markers=()):
        self.profile = profile
        self.length = float(length or 0.0)
        self.position = position
        self.markers = list(markers)
        self.render()

    def setPosition(self, position):
        self.position = position
        self.render()

    def setMarkers(self, markers):
        self.markers = list(markers)
        self.render()

    # ------------------------------------------------------------------ geometry
    def _size(self):
        return max(self.label.width, 200), max(self.label.height, 160)

    def _ranges(self):
        s = self.profile.arc_length_mm
        values = []
        for layer, metric, _ in CURVES:
            v = self.profile.values(layer, metric)
            values.append(v[np.isfinite(v)])
        values = np.concatenate(values) if values else np.zeros(0)
        top = float(values.max()) if values.size else 30.0
        step = niceStep(top)
        y_max = step * math.ceil(top * 1.05 / step)
        x_max = self.length if self.length > 0 else (float(s[-1]) if len(s) else 1.0)
        return 0.0, x_max, 0.0, max(y_max, step)

    def _toPixel(self, s, d, ranges, size):
        x0, x1, y0, y1 = ranges
        width, height = size
        px = MARGIN_LEFT + (s - x0) / max(x1 - x0, 1e-9) * (width - MARGIN_LEFT - MARGIN_RIGHT)
        py = height - MARGIN_BOTTOM - (d - y0) / max(y1 - y0, 1e-9) * (height - MARGIN_TOP - MARGIN_BOTTOM)
        return px, py

    def arcLengthAt(self, x):
        """Arc length (mm) under a horizontal pixel position (None without a plot)."""
        if self.profile is None:
            return None
        ranges = self._ranges()
        width, _ = self._size()
        fraction = (x - MARGIN_LEFT) / max(width - MARGIN_LEFT - MARGIN_RIGHT, 1)
        return float(min(max(ranges[0] + fraction * (ranges[1] - ranges[0]), ranges[0]), ranges[1]))

    def _clicked(self, x):
        s = self.arcLengthAt(x)
        if s is not None and self.onClick is not None:
            self.onClick(s)

    # ------------------------------------------------------------------ drawing
    def render(self):
        width, height = self._size()
        pixmap = qt.QPixmap(width, height)
        pixmap.fill(qt.QColor("#ffffff"))
        painter = qt.QPainter(pixmap)
        try:
            painter.setRenderHint(qt.QPainter.Antialiasing, True)
            if self.profile is None or len(self.profile.arc_length_mm) == 0:
                painter.setPen(qt.QColor("#808080"))
                painter.drawText(qt.QRect(0, 0, width, height), qt.Qt.AlignCenter, tr("profile.empty"))
            else:
                self._draw(painter, (width, height))
        finally:
            painter.end()
        self.label.setPixmap(pixmap)

    def _draw(self, painter, size):
        width, height = size
        ranges = self._ranges()
        x0, x1, y0, y1 = ranges
        grid = qt.QPen(qt.QColor("#e0e0e0"))
        text = qt.QPen(qt.QColor("#404040"))
        # Horizontal grid and diameter labels
        step = niceStep(y1 - y0, 4)
        d = y0
        while d <= y1 + 1e-9:
            _, py = self._toPixel(x0, d, ranges, size)
            painter.setPen(grid)
            painter.drawLine(MARGIN_LEFT, int(py), width - MARGIN_RIGHT, int(py))
            painter.setPen(text)
            painter.drawText(qt.QRect(0, int(py) - 8, MARGIN_LEFT - 4, 16), qt.Qt.AlignRight | qt.Qt.AlignVCenter, f"{d:g}")
            d += step
        # Vertical grid and position labels
        step = niceStep(x1 - x0, 6)
        s = x0
        while s <= x1 + 1e-9:
            px, _ = self._toPixel(s, y0, ranges, size)
            painter.setPen(grid)
            painter.drawLine(int(px), MARGIN_TOP, int(px), height - MARGIN_BOTTOM)
            painter.setPen(text)
            painter.drawText(qt.QRect(int(px) - 30, height - MARGIN_BOTTOM + 2, 60, 14), qt.Qt.AlignHCenter, f"{s:g}")
            s += step
        painter.drawText(
            qt.QRect(MARGIN_LEFT, height - 13, width - MARGIN_LEFT - MARGIN_RIGHT, 13),
            qt.Qt.AlignRight, tr("profile.axis.position"),
        )
        painter.drawText(qt.QRect(2, 2, 120, 16), qt.Qt.AlignLeft, tr("profile.axis.diameter"))
        # Curves (gaps where no contour was found)
        sValues = self.profile.arc_length_mm
        for layer, metric, color in CURVES:
            values = self.profile.values(layer, metric)
            pen = qt.QPen(qt.QColor(color))
            pen.setWidthF(1.6)
            painter.setPen(pen)
            previous = None
            for s, v in zip(sValues, values):
                if not np.isfinite(v):
                    previous = None
                    continue
                point = self._toPixel(float(s), float(v), ranges, size)
                if previous is not None:
                    painter.drawLine(int(previous[0]), int(previous[1]), int(point[0]), int(point[1]))
                previous = point
        # Markers
        painter.setPen(qt.QPen(qt.QColor("#208040")))
        for s, name in self.markers:
            if s is None or not (x0 <= s <= x1):
                continue
            px, _ = self._toPixel(s, y0, ranges, size)
            painter.drawLine(int(px), MARGIN_TOP, int(px), MARGIN_TOP + 8)
            painter.drawText(qt.QRect(int(px) + 2, MARGIN_TOP - 16, 160, 14), qt.Qt.AlignLeft, name)
        # Current position
        if self.position is not None:
            px, _ = self._toPixel(float(self.position), y0, ranges, size)
            pen = qt.QPen(qt.QColor("#c8a000"))
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.drawLine(int(px), MARGIN_TOP, int(px), height - MARGIN_BOTTOM)
