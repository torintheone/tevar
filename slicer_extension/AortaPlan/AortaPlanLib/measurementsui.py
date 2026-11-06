"""Step 3 "Measurements" of the module panel (stage 4): connects the widgets of
``measurementsPage`` (Resources/UI/AortaPlan.ui) with ``MeasurementController``."""

import logging

import qt
import slicer

from aortaplan_core.centerline.markers import MARKER_PRESETS
from aortaplan_core.crosssection import LUMEN, METRIC_NAMES, OUTER

from .measurements import formatNumber
from .profileplot import ProfilePlot

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


def _item(text, editable=False):
    item = qt.QTableWidgetItem(text)
    flags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled
    if editable:
        flags |= qt.Qt.ItemIsEditable
    item.setFlags(flags)
    return item


def _setupTable(table, headers, stretchLast=True):
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
    table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
    table.setEditTriggers(qt.QAbstractItemView.SelectedClicked | qt.QAbstractItemView.EditKeyPressed)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(stretchLast)


def _selectedRow(table):
    rows = table.selectionModel().selectedRows()
    return rows[0].row() if rows else -1


class MeasurementsPanel:
    def __init__(self, ui, controller, navigator, interaction, progressFactory):
        self.ui = ui
        self.controller = controller
        self.navigator = navigator
        self.interaction = interaction
        self.progressFactory = progressFactory  # () -> (dialog, callback)
        self.visible = False
        self._markersSignature = None
        self._savedSignature = None
        self._distanceSignature = None
        self._updating = False
        self.plot = ProfilePlot(ui.profilePlotLabel, onClick=self.onPlotClicked)
        self._applyStrings()
        self._setupWidgets()
        controller.listeners.append(self.refresh)

    # ------------------------------------------------------------------ setup
    def _applyStrings(self):
        ui = self.ui
        ui.sectionCollapsibleButton.text = tr("measure.section.title")
        ui.liveContourCheckBox.text = tr("measure.live")
        ui.liveContourCheckBox.toolTip = tr("measure.live.tooltip")
        ui.outerWallCheckBox.text = tr("measure.outer")
        ui.recomputeContourButton.text = tr("measure.recompute")
        ui.recomputeContourButton.toolTip = tr("measure.recompute.tooltip")
        ui.saveMeasurementButton.text = tr("measure.save")
        ui.savedTitleLabel.text = tr("measure.saved.title")
        ui.savedHintLabel.text = tr("measure.saved.hint")
        ui.deleteMeasurementButton.text = tr("measure.delete")
        ui.profileCollapsibleButton.text = tr("profile.title")
        ui.computeProfileButton.text = tr("profile.compute")
        ui.profileLegendLabel.text = tr("profile.legend")
        ui.maxTitleLabel.text = tr("profile.max.title")
        ui.maxFromLabel.text = tr("profile.max.from")
        ui.maxToLabel.text = tr("profile.max.to")
        ui.findMaxButton.text = tr("profile.max.find")
        ui.markersCollapsibleButton.text = tr("markers.title")
        ui.markerNameLabel.text = tr("markers.name")
        ui.addMarkerButton.text = tr("markers.add")
        ui.placeMarkerByClickButton.text = tr("markers.add_by_click")
        ui.jumpToMarkerButton.text = tr("markers.jump")
        ui.moveMarkerButton.text = tr("markers.move")
        ui.deleteMarkerButton.text = tr("markers.delete")
        ui.distanceTitleLabel.text = tr("markers.distance.title")
        ui.distanceFromLabel.text = tr("markers.distance.from")
        ui.distanceToLabel.text = tr("markers.distance.to")
        ui.matrixTitleLabel.text = tr("markers.matrix.title")
        ui.toolsCollapsibleButton.text = tr("tools.title")
        ui.lengthToolButton.text = tr("tools.length")
        ui.rulerButton.text = tr("tools.ruler")
        ui.clearRulersButton.text = tr("tools.ruler.clear")
        ui.rulerHintLabel.text = tr("tools.ruler.hint")
        ui.evarCollapsibleButton.text = tr("evar.title")
        ui.evarHintLabel.text = tr("evar.hint")

    def _setupWidgets(self):
        ui = self.ui
        table = ui.metricsTable
        table.setColumnCount(2)
        table.setRowCount(len(METRIC_NAMES))
        table.setHorizontalHeaderLabels([tr("measure.column.lumen"), tr("measure.column.outer")])
        table.setVerticalHeaderLabels([tr(f"measure.metric.{name}") for name in METRIC_NAMES])
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.Stretch)
        font = table.font
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        table.setFont(font)
        for row in range(len(METRIC_NAMES)):
            for column in range(2):
                item = _item("—")
                item.setTextAlignment(qt.Qt.AlignCenter)
                table.setItem(row, column, item)

        _setupTable(ui.savedTable, [tr(f"measure.saved.column.{c}") for c in ("name", "position", "lumen", "outer", "area")])
        _setupTable(ui.markersTable, [tr(f"markers.column.{c}") for c in ("name", "position", "previous")])
        ui.distanceMatrixTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        for key in MARKER_PRESETS:
            ui.markerNameComboBox.addItem(tr(f"marker.{key}"), key)
        ui.markerNameComboBox.setInsertPolicy(qt.QComboBox.NoInsert)

        ui.liveContourCheckBox.connect("toggled(bool)", self.onLiveToggled)
        ui.outerWallCheckBox.connect("toggled(bool)", self.onOuterToggled)
        ui.recomputeContourButton.connect("clicked()", self.controller.recompute)
        ui.saveMeasurementButton.connect("clicked()", self.onSave)
        ui.deleteMeasurementButton.connect("clicked()", self.onDeleteMeasurement)
        ui.savedTable.connect("cellDoubleClicked(int,int)", lambda row, column: self.controller.jumpToMeasurement(row))
        ui.savedTable.connect("itemChanged(QTableWidgetItem*)", self.onSavedItemChanged)
        ui.computeProfileButton.connect("clicked()", self.onComputeProfile)
        ui.findMaxButton.connect("clicked()", self.onFindMaximum)
        ui.addMarkerButton.connect("clicked()", self.onAddMarker)
        ui.placeMarkerByClickButton.connect("toggled(bool)", self.onPlaceMarkerByClickToggled)
        ui.markersTable.connect("cellDoubleClicked(int,int)", lambda row, column: self.controller.jumpToMarker(row))
        ui.markersTable.connect("itemChanged(QTableWidgetItem*)", self.onMarkerItemChanged)
        ui.jumpToMarkerButton.connect("clicked()", lambda: self.controller.jumpToMarker(_selectedRow(ui.markersTable)))
        ui.moveMarkerButton.connect("clicked()", lambda: self.controller.moveMarker(_selectedRow(ui.markersTable)))
        ui.deleteMarkerButton.connect("clicked()", lambda: self.controller.deleteMarker(_selectedRow(ui.markersTable)))
        ui.distanceFromComboBox.connect("currentIndexChanged(int)", lambda index: self.updateDistance())
        ui.distanceToComboBox.connect("currentIndexChanged(int)", lambda index: self.updateDistance())
        ui.lengthToolButton.connect("toggled(bool)", self.onLengthToolToggled)
        ui.rulerButton.connect("clicked()", self.controller.addRuler)
        ui.clearRulersButton.connect("clicked()", self.controller.clearRulers)

    # ------------------------------------------------------------------ visibility
    def setVisible(self, visible):
        """The measurement step is the current step (contours follow the position)."""
        self.visible = bool(visible)
        if not visible:
            self.ui.placeMarkerByClickButton.checked = False
            self.ui.lengthToolButton.checked = False
        self.controller.setActive(visible)
        self.refresh(force=True)

    # ------------------------------------------------------------------ refresh
    def refresh(self, force=False):
        if self._updating:
            return
        self._updating = True
        try:
            self._refreshSection()
            self._refreshSaved(force)
            self._refreshMarkers(force)
            self._refreshPlot()
            self._refreshTools()
        except Exception:
            logging.exception("AortaPlan: measurement panel refresh failed")
        finally:
            self._updating = False

    def _refreshSection(self):
        ui = self.ui
        metrics = self.controller.sectionMetrics()
        for column, kind in enumerate((LUMEN, OUTER)):
            m = metrics[kind]
            values = m.values() if m is not None else {}
            for row, name in enumerate(METRIC_NAMES):
                digits = 0 if name == "area_mm2" else 1
                ui.metricsTable.item(row, column).setText(formatNumber(values.get(name), digits) if m is not None else "—")
        if self.navigator.path is None:
            status = tr("measure.status.no_path")
        elif self.controller.current is None:
            status = "" if not self.controller.live else tr("measure.status.no_lumen")
        elif self.controller.current.lumen is None:
            status = tr("measure.status.no_lumen")
        elif self.controller.dirty:
            status = tr("measure.status.manual")
        elif self.controller.currentName:
            status = tr("measure.status.saved", name=self.controller.currentName)
        else:
            status = tr("measure.status.auto")
        warnings = [tr(f"measure.warning.{w}") for w in self.controller.sectionWarnings() if w != "lumen_not_found"]
        lines = [status] + [f"<span style='color:#b35900'>⚠ {w}</span>" for w in warnings]
        ui.contourStatusLabel.text = "<br>".join(line for line in lines if line)
        hasPath = self.navigator.path is not None
        ui.recomputeContourButton.enabled = hasPath
        ui.saveMeasurementButton.enabled = self.controller.current is not None and self.controller.current.lumen is not None

    def _refreshSaved(self, force):
        store = self.controller.store
        signature = tuple((m.name, m.path, round(m.arc_length_mm, 2)) for m in store.measurements)
        if not force and signature == self._savedSignature:
            return
        self._savedSignature = signature
        table = self.ui.savedTable
        table.blockSignals(True)
        try:
            table.setRowCount(len(store.measurements))
            for row, item in enumerate(store.measurements):
                metrics = item.measurement.metrics()
                lumen, outer = metrics[LUMEN], metrics[OUTER]
                cells = [
                    _item(item.name, editable=True),
                    _item(formatNumber(item.arc_length_mm)),
                    _item(f"{formatNumber(lumen.min_diameter_mm)}–{formatNumber(lumen.max_diameter_mm)}" if lumen else "—"),
                    _item(f"{formatNumber(outer.min_diameter_mm)}–{formatNumber(outer.max_diameter_mm)}" if outer else "—"),
                    _item(formatNumber(lumen.area_mm2, 0) if lumen else "—"),
                ]
                for column, cell in enumerate(cells):
                    table.setItem(row, column, cell)
            table.resizeColumnsToContents()
        finally:
            table.blockSignals(False)

    def _markerSignature(self):
        path = self.navigator.path
        markers = self.controller.markers()
        key = (self.controller.key, round(path.length, 2)) if path is not None else None
        return key, tuple((m.name, tuple(round(float(x), 2) for x in m.point)) for m in markers)

    def _refreshMarkers(self, force):
        signature = self._markerSignature()
        if not force and signature == self._markersSignature:
            return
        self._markersSignature = signature
        ui = self.ui
        markers = self.controller.markers()
        table = ui.markersTable
        table.blockSignals(True)
        try:
            table.setRowCount(len(markers))
            previous = None
            for row, marker in enumerate(markers):
                s = marker.arc_length_mm
                position = formatNumber(s) if s is not None else tr("markers.off_path")
                gap = formatNumber(s - previous) if s is not None and previous is not None else ""
                table.setItem(row, 0, _item(marker.name, editable=True))
                table.setItem(row, 1, _item(position))
                table.setItem(row, 2, _item(gap))
                if s is not None:
                    previous = s
            table.resizeColumnsToContents()
        finally:
            table.blockSignals(False)
        # Distance combos and the matrix
        for combo in (ui.distanceFromComboBox, ui.distanceToComboBox):
            current = combo.currentIndex
            combo.blockSignals(True)
            combo.clear()
            for marker in markers:
                combo.addItem(marker.name)
            if combo.count:
                combo.currentIndex = min(max(current, 0), combo.count - 1)
            combo.blockSignals(False)
        if ui.distanceToComboBox.count > 1 and ui.distanceToComboBox.currentIndex == ui.distanceFromComboBox.currentIndex:
            ui.distanceToComboBox.currentIndex = ui.distanceToComboBox.count - 1
        self._refreshMatrix(markers)
        self._refreshRangeCombos(markers)
        self.updateDistance()
        ui.evarLabel.text = "<br>".join(self.controller.evarSummary())
        hasPath = self.navigator.path is not None
        ui.addMarkerButton.enabled = hasPath
        ui.placeMarkerByClickButton.enabled = hasPath
        ui.findMaxButton.enabled = hasPath
        ui.computeProfileButton.enabled = hasPath

    def _refreshMatrix(self, markers):
        table = self.ui.distanceMatrixTable
        names = [m.name for m in markers]
        table.setColumnCount(len(names))
        table.setRowCount(len(names))
        table.setHorizontalHeaderLabels(names)
        table.setVerticalHeaderLabels(names)
        if len(markers) < 2:
            return
        try:
            matrix = self.controller.distanceMatrix()
        except Exception:
            logging.exception("AortaPlan: distance matrix failed")
            return
        for i in range(len(names)):
            for j in range(len(names)):
                d = matrix[i][j]
                text = "" if d is None else (formatNumber(d.along_mm) if d.along_mm is not None else "—")
                cell = _item(text)
                if d is not None:
                    cell.setToolTip(tr("markers.distance.result", along=formatNumber(d.along_mm), straight=formatNumber(d.straight_mm)))
                table.setItem(i, j, cell)
        table.resizeColumnsToContents()

    def _refreshRangeCombos(self, markers):
        path = self.navigator.path
        entries = [(tr("profile.max.start"), 0.0)]
        entries += [(m.name, m.arc_length_mm) for m in markers if m.arc_length_mm is not None]
        entries += [(tr("profile.max.end"), path.length if path is not None else 0.0)]
        for combo, default in ((self.ui.maxFromComboBox, 0), (self.ui.maxToComboBox, len(entries) - 1)):
            current = combo.currentText
            combo.blockSignals(True)
            combo.clear()
            for name, s in entries:
                combo.addItem(name, s)
            index = combo.findText(current)
            combo.currentIndex = index if index >= 0 else default
            combo.blockSignals(False)

    def updateDistance(self):
        ui = self.ui
        i, j = ui.distanceFromComboBox.currentIndex, ui.distanceToComboBox.currentIndex
        markers = self.controller.markers()
        if i < 0 or j < 0 or i == j or max(i, j) >= len(markers):
            ui.distanceResultLabel.text = ""
            self._distanceSignature = None
            return
        signature = (self._markersSignature, i, j)
        if signature == self._distanceSignature:
            return
        self._distanceSignature = signature
        with slicer.util.WaitCursor():
            distance, curvature = self.controller.distance(i, j)
        straight = formatNumber(distance.straight_mm)
        if distance.along_mm is None:
            lines = [tr("markers.distance.none", straight=straight)]
        elif distance.path is None:
            lines = [tr("markers.distance.branches", along=formatNumber(distance.along_mm), straight=straight)]
        else:
            lines = [tr("markers.distance.result", along=formatNumber(distance.along_mm), straight=straight)]
        if curvature is not None and curvature.greater_mm is not None and curvature.lesser_mm is not None:
            lines.append(tr("markers.distance.curvature", greater=formatNumber(curvature.greater_mm),
                            lesser=formatNumber(curvature.lesser_mm)))
        ui.distanceResultLabel.text = "<br>".join(lines)

    def _refreshPlot(self):
        if not self.visible:
            return
        path = self.navigator.path
        markers = [(m.arc_length_mm, m.name) for m in self.controller.markers() if m.arc_length_mm is not None]
        self.plot.setData(self.controller.profile, path.length if path is not None else 0.0,
                          self.navigator.position if path is not None else None, markers)

    def _refreshTools(self):
        ui = self.ui
        result = self.controller.lengthResult
        if ui.lengthToolButton.checked:
            ui.lengthResultLabel.text = tr("tools.length.second") if self.controller.lengthClicks else tr("tools.length.start")
        elif result is not None:
            a, b, length = result
            ui.lengthResultLabel.text = tr("tools.length.result", length=formatNumber(length), a=formatNumber(a), b=formatNumber(b))
        else:
            ui.lengthResultLabel.text = ""
        maximum = self.controller.maximumResult
        if maximum is not None:
            key = "profile.max.result.outer" if maximum.layer == OUTER else "profile.max.result.lumen"
            ui.maxResultLabel.text = tr(key, value=formatNumber(maximum.value_mm), position=formatNumber(maximum.arc_length_mm),
                                        marker=tr("marker.max_diameter"))

    # ------------------------------------------------------------------ handlers
    def onLiveToggled(self, checked):
        self.controller.live = bool(checked)
        self.controller.update(force=True)

    def onOuterToggled(self, checked):
        self.controller.withOuter = bool(checked)
        self.controller.profile = None
        self.controller.update(force=True)

    def onSave(self):
        self.controller.saveCurrent(self.controller.currentName)

    def onDeleteMeasurement(self):
        self.controller.deleteMeasurement(_selectedRow(self.ui.savedTable))

    def onSavedItemChanged(self, item):
        if item.column() == 0:
            self.controller.renameMeasurement(item.row(), item.text().strip())

    def onMarkerItemChanged(self, item):
        if item.column() == 0:
            self.controller.renameMarker(item.row(), item.text().strip())

    def markerName(self):
        return self.ui.markerNameComboBox.currentText.strip()

    def onAddMarker(self):
        self.controller.addMarker(self.markerName())

    def onMarkerKey(self):
        """Hot key M: marker with the selected name at the current position."""
        if self.navigator.path is not None:
            self.controller.addMarker(self.markerName())

    def onPlaceMarkerByClickToggled(self, checked):
        if checked:
            self.ui.lengthToolButton.checked = False
            self.interaction.cprClickHandler = self._onMarkerClick
            self.ui.lengthResultLabel.text = tr("markers.click_hint")
        elif self.interaction.cprClickHandler == self._onMarkerClick:
            self.interaction.cprClickHandler = None

    def _onMarkerClick(self, viewIndex, offset, s):
        self.controller.addMarker(self.markerName(), s)
        self.navigator.setPosition(s)
        self.ui.placeMarkerByClickButton.checked = False
        return True

    def onLengthToolToggled(self, checked):
        if checked:
            self.ui.placeMarkerByClickButton.checked = False
            self.controller.startLength()
            self.interaction.cprClickHandler = self._onLengthClick
        elif self.interaction.cprClickHandler == self._onLengthClick:
            self.interaction.cprClickHandler = None
        self._refreshTools()

    def _onLengthClick(self, viewIndex, offset, s):
        if self.controller.lengthClick(viewIndex, offset, s):
            self.ui.lengthToolButton.checked = False
        self._refreshTools()
        return True

    def onPlotClicked(self, s):
        if self.navigator.path is not None:
            self.navigator.setPosition(s)

    def onComputeProfile(self):
        from aortaplan_core.lumen import Cancelled

        dialog, callback = self.progressFactory("progress.profile")
        try:
            self.controller.computeProfile(callback)
        except Cancelled:
            pass
        finally:
            dialog.close()
        self.refresh(force=True)

    def onFindMaximum(self):
        from aortaplan_core.lumen import Cancelled

        ui = self.ui
        s0 = ui.maxFromComboBox.itemData(ui.maxFromComboBox.currentIndex)
        s1 = ui.maxToComboBox.itemData(ui.maxToComboBox.currentIndex)
        if s0 is None or s1 is None:
            return
        dialog, callback = self.progressFactory("progress.maximum")
        result = None
        try:
            result = self.controller.findMaximum(float(s0), float(s1), callback)
        except Cancelled:
            pass
        finally:
            dialog.close()
        if result is None:
            ui.maxResultLabel.text = tr("profile.max.none")
        self.refresh(force=True)
