"""Step 4 "Ostia" of the module panel (stage 5): widgets of ``ostiaPage`` <-> ``OstiaController``."""

import logging

import qt
import slicer

from aortaplan_core.ostia import CLOCK_MODES, OSTIUM_KEYS, ostium_name

from .measurements import formatNumber
from .ostia import describeRow

try:
    from aortaplan_core.i18n import tr
except ImportError:

    def tr(key, **kwargs):
        return key


COLUMNS = ("name", "confirmed", "position", "previous", "clock", "degrees", "arc", "diameter", "craniocaudal", "axial",
           "stenosis")
COL = {name: i for i, name in enumerate(COLUMNS)}
ZONE_COLUMNS = ("zone", "start", "end", "length", "lumen", "outer")


def _item(text, editable=False, checkable=False, checked=False):
    item = qt.QTableWidgetItem(text)
    flags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled
    if editable:
        flags |= qt.Qt.ItemIsEditable
    if checkable:
        flags |= qt.Qt.ItemIsUserCheckable
        item.setCheckState(qt.Qt.Checked if checked else qt.Qt.Unchecked)
    item.setFlags(flags)
    return item


class OstiaPanel:
    def __init__(self, ui, controller, navigator, progressFactory):
        self.ui = ui
        self.controller = controller
        self.navigator = navigator
        self.progressFactory = progressFactory
        self.visible = False
        self._rowIndex = []  # table row -> index in the store
        self._updating = False
        self._applyStrings()
        self._setupWidgets()
        controller.listeners.append(self.refresh)
        controller.measurements.listeners.append(self._onMeasurementsChanged)

    def _applyStrings(self):
        ui = self.ui
        ui.ostiaCollapsibleButton.text = tr("wizard.step.ostia")
        ui.findOstiaButton.text = tr("ostia.find")
        ui.findOstiaButton.toolTip = tr("ostia.find.tooltip")
        ui.addOstiumButton.text = tr("ostia.add")
        ui.addOstiumButton.toolTip = tr("ostia.add.hint")
        ui.ostiaReferenceLabel.text = tr("ostia.reference")
        ui.ostiaClockModeLabel.text = tr("ostia.clock_mode")
        ui.ostiaGraftLabel.text = tr("ostia.graft_diameter")
        ui.ostiaDistanceLabel.text = tr("ostia.branch_distance")
        ui.ostiaMinBranchLabel.text = tr("ostia.min_branch")
        for box in (ui.ostiaGraftSpinBox, ui.ostiaDistanceSpinBox, ui.ostiaMinBranchSpinBox):
            box.suffix = " " + tr("units.mm")
        ui.ostiumNameLabel.text = tr("ostia.name")
        ui.renameOstiumButton.text = tr("ostia.rename")
        ui.jumpToOstiumButton.text = tr("ostia.jump")
        ui.confirmAllOstiaButton.text = tr("ostia.confirm_all")
        ui.deleteOstiumButton.text = tr("ostia.delete")
        ui.markersFromOstiaButton.text = tr("ostia.markers")
        ui.markersFromOstiaButton.toolTip = tr("ostia.markers.tooltip")
        ui.zonesCollapsibleButton.text = tr("ostia.zones.title")
        ui.computeZonesButton.text = tr("ostia.zones.compute")
        ui.zonesHintLabel.text = tr("ostia.zones.hint")

    def _setupWidgets(self):
        ui = self.ui
        for mode in CLOCK_MODES:
            ui.ostiaClockModeComboBox.addItem(tr(f"ostia.clock_mode.{mode}"), mode)
        for key in OSTIUM_KEYS:
            ui.ostiumNameComboBox.addItem(ostium_name(key), key)
        ui.ostiumNameComboBox.setInsertPolicy(qt.QComboBox.NoInsert)
        table = ui.ostiaTable
        table.setColumnCount(len(COLUMNS))
        table.setHorizontalHeaderLabels([tr(f"ostia.column.{c}") for c in COLUMNS])
        table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        table.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        table.setEditTriggers(qt.QAbstractItemView.SelectedClicked | qt.QAbstractItemView.EditKeyPressed)
        table.verticalHeader().setVisible(False)
        zones = ui.zonesTable
        zones.setColumnCount(len(ZONE_COLUMNS))
        zones.setHorizontalHeaderLabels([tr(f"ostia.zones.column.{c}") for c in ZONE_COLUMNS])
        zones.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        zones.verticalHeader().setVisible(False)
        zones.horizontalHeader().setStretchLastSection(True)

        ui.findOstiaButton.connect("clicked()", self.onFind)
        ui.addOstiumButton.connect("toggled(bool)", self.onAddToggled)
        ui.ostiaReferenceComboBox.connect("currentIndexChanged(int)", self.onReferenceChanged)
        ui.ostiaClockModeComboBox.connect("currentIndexChanged(int)", self.onClockModeChanged)
        ui.ostiaGraftSpinBox.connect("valueChanged(double)", lambda v: self._setSetting("graft_diameter_mm", float(v)))
        ui.ostiaDistanceSpinBox.connect("valueChanged(double)", lambda v: self._setSetting("diameter_distance_mm", float(v)))
        ui.ostiaMinBranchSpinBox.connect("valueChanged(double)", lambda v: self._setSetting("min_branch_diameter_mm", float(v)))
        table.connect("itemChanged(QTableWidgetItem*)", self.onItemChanged)
        table.connect("cellDoubleClicked(int,int)", lambda row, column: self.controller.jump(self._storeIndex(row)))
        ui.renameOstiumButton.connect("clicked()", self.onRename)
        ui.jumpToOstiumButton.connect("clicked()", lambda: self.controller.jump(self._storeIndex(self._selectedRow())))
        ui.confirmAllOstiaButton.connect("clicked()", self.controller.confirmAll)
        ui.deleteOstiumButton.connect("clicked()", lambda: self.controller.delete(self._storeIndex(self._selectedRow())))
        ui.markersFromOstiaButton.connect("clicked()", self.onMarkersFromOstia)
        ui.computeZonesButton.connect("clicked()", self.onComputeZones)

    # ------------------------------------------------------------------ helpers
    def _selectedRow(self):
        rows = self.ui.ostiaTable.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _storeIndex(self, row):
        return self._rowIndex[row] if 0 <= row < len(self._rowIndex) else -1

    def _setSetting(self, name, value):
        if not self._updating:
            self.controller.setSetting(name, value)

    def setVisible(self, visible):
        self.visible = bool(visible)
        if not visible:
            self.ui.addOstiumButton.checked = False
        self.refresh()

    def _onMeasurementsChanged(self):
        if self.visible:
            self._refreshReference()

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        if self._updating:
            return
        self._updating = True
        try:
            self._refreshSettings()
            self._refreshReference()
            self._refreshTable()
            self._refreshZones()
            self._refreshStatus()
        except Exception:
            logging.exception("AortaPlan: ostia panel refresh failed")
        finally:
            self._updating = False

    def _refreshSettings(self):
        ui, c = self.ui, self.controller
        index = ui.ostiaClockModeComboBox.findData(c.clockMode)
        ui.ostiaClockModeComboBox.currentIndex = max(index, 0)
        ui.ostiaGraftSpinBox.value = c.graftDiameter
        ui.ostiaDistanceSpinBox.value = float(c.setting("diameter_distance_mm", 5.0))
        ui.ostiaMinBranchSpinBox.value = float(c.setting("min_branch_diameter_mm", 2.5))
        ui.ostiaHelpLabel.text = tr("ostia.help", mode=tr(f"ostia.clock_mode.{c.clockMode}"))
        hasPath = self.navigator.path is not None
        for widget in (ui.findOstiaButton, ui.addOstiumButton, ui.markersFromOstiaButton):
            widget.enabled = hasPath

    def _refreshReference(self):
        combo = self.ui.ostiaReferenceComboBox
        wanted = self.controller.setting("reference_marker", "")
        names = [m.name for m in self.controller.measurements.markers() if m.arc_length_mm is not None]
        was = self._updating
        self._updating = True
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(tr("ostia.reference.start"), "")
            for name in names:
                combo.addItem(name, name)
            index = combo.findData(wanted)
            combo.currentIndex = index if index >= 0 else 0
        finally:
            combo.blockSignals(False)
            self._updating = was

    def _refreshTable(self):
        table = self.ui.ostiaTable
        rows = self.controller.rows()
        reference = self.controller.referenceArcLength()
        table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            self._rowIndex = []
            previous = None
            for r, (index, ostium, position) in enumerate(rows):
                self._rowIndex.append(index)
                texts = describeRow(self.controller, ostium, position, reference, previous)
                previous = position.arc_length_mm
                for name in COLUMNS:
                    if name == "confirmed":
                        item = _item("", checkable=True, checked=ostium.confirmed)
                    elif name == "stenosis":
                        item = _item("", checkable=True, checked=ostium.stenosis)
                    elif name == "name":
                        item = _item(ostium.name or ostium_name(ostium.key), editable=True)
                        if not ostium.confirmed:
                            font = item.font()
                            font.setItalic(True)
                            item.setFont(font)
                        if ostium.warnings:
                            item.setToolTip("; ".join(tr(f"ostia.warning.{w}") for w in ostium.warnings))
                    else:
                        item = _item(texts[name])
                        item.setTextAlignment(qt.Qt.AlignCenter)
                    table.setItem(r, COL[name], item)
            table.resizeColumnsToContents()
        finally:
            table.blockSignals(False)

    def _refreshZones(self):
        table = self.ui.zonesTable
        zones = self.controller.zones
        table.setRowCount(len(zones))
        for r, zone in enumerate(zones):
            d = zone.diameters or {}
            lumen = (
                f"{formatNumber(d.get('lumen_min_mm'))}–{formatNumber(d.get('lumen_max_mm'))}"
                if d.get("lumen_min_mm") is not None else "—"
            )
            end = formatNumber(zone.end_mm)
            length = formatNumber(zone.length_mm)
            if zone.approximate:
                end = tr("ostia.zones.approximate", value=end)
                length = tr("ostia.zones.approximate", value=length)
            cells = [zone.name, formatNumber(zone.start_mm), end, length, lumen, formatNumber(d.get("outer_max_mm"))]
            for column, text in enumerate(cells):
                item = _item(text)
                item.setTextAlignment(qt.Qt.AlignCenter)
                table.setItem(r, column, item)
        table.resizeColumnsToContents()
        self.ui.computeZonesButton.enabled = bool(zones)
        if not zones:
            self.ui.zonesHintLabel.text = tr("ostia.zones.none") + " " + tr("ostia.zones.hint")
        else:
            self.ui.zonesHintLabel.text = tr("ostia.zones.hint")

    def _refreshStatus(self):
        ui = self.ui
        if self.controller.clicks:
            ui.ostiaStatusLabel.text = tr("ostia.add.second")
        elif ui.addOstiumButton.checked:
            ui.ostiaStatusLabel.text = tr("ostia.add.hint")
        elif self.controller.ostia:
            ui.ostiaStatusLabel.text = tr("ostia.status.found", count=len(self.controller.ostia))
        else:
            ui.ostiaStatusLabel.text = tr("ostia.status.none")

    # ------------------------------------------------------------------ handlers
    def onFind(self):
        from aortaplan_core.lumen import Cancelled

        dialog, callback = self.progressFactory("progress.ostia.tube")
        try:
            self.controller.find(callback)
        except Cancelled:
            pass
        finally:
            dialog.close()

    def onAddToggled(self, checked):
        if checked:
            self.controller.startClicks()
        elif self.controller.clicks or self.controller._clickObservation is not None:
            self.controller.finishClicks()
        self._refreshStatus()

    def onReferenceChanged(self, index):
        if not self._updating:
            self.controller.setSetting("reference_marker", self.ui.ostiaReferenceComboBox.itemData(index) or "")

    def onClockModeChanged(self, index):
        if not self._updating:
            self.controller.setSetting("clock_mode", self.ui.ostiaClockModeComboBox.itemData(index))

    def onItemChanged(self, item):
        index = self._storeIndex(item.row())
        if index < 0:
            return
        column = item.column()
        if column == COL["name"]:
            self.controller.rename(index, item.text().strip(), confirm=True)
        elif column == COL["confirmed"]:
            self.controller.setConfirmed(index, item.checkState() == qt.Qt.Checked)
        elif column == COL["stenosis"]:
            self.controller.setStenosis(index, item.checkState() == qt.Qt.Checked)

    def onRename(self):
        index = self._storeIndex(self._selectedRow())
        name = self.ui.ostiumNameComboBox.currentText.strip()
        if index >= 0 and name:
            self.controller.rename(index, name, confirm=True)

    def onMarkersFromOstia(self):
        count = self.controller.markersFromOstia()
        self.ui.ostiaStatusLabel.text = tr("ostia.markers.done", count=count)

    def onComputeZones(self):
        from aortaplan_core.lumen import Cancelled

        dialog, callback = self.progressFactory("progress.ostia.zones")
        try:
            with slicer.util.WaitCursor():
                self.controller.computeZones(withDiameters=True, progress=callback)
        except Cancelled:
            pass
        finally:
            dialog.close()
        self._refreshZones()

    def onInteractionModeChanged(self):
        """The user left the placement mode (e.g. Escape): finish the clicked ostium."""
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        if self.ui.addOstiumButton.checked and interaction.GetCurrentInteractionMode() != interaction.Place:
            self.ui.addOstiumButton.checked = False  # -> onAddToggled -> finishClicks
