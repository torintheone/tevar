"""Saved section measurements, markers and ostia of a study, serialisable to JSON.

The Slicer module keeps one :class:`MeasurementStore` per scene (stage 9 will add it to
the session file); the report (stage 7) reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..centerline.markers import Marker
from .contours import SectionMeasurement

STORE_SCHEMA_VERSION = 2  # 2: ostia and their settings
SAME_SECTION_MM = 0.25  # a saved measurement is shown again within this distance along the line


@dataclass
class SavedMeasurement:
    """A named cross-section measurement on a path (``path`` = path name, may be empty)."""

    name: str
    path: str
    measurement: SectionMeasurement

    @property
    def arc_length_mm(self) -> float:
        s = self.measurement.arc_length_mm
        return 0.0 if s is None else float(s)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "path": self.path, "measurement": self.measurement.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SavedMeasurement:
        return cls(d["name"], d.get("path", ""), SectionMeasurement.from_dict(d["measurement"]))


@dataclass
class MeasurementStore:
    measurements: List[SavedMeasurement] = field(default_factory=list)
    markers: List[Marker] = field(default_factory=list)
    ostia: List[Any] = field(default_factory=list)  # aortaplan_core.ostia.model.Ostium
    ostia_settings: Dict[str, Any] = field(default_factory=dict)

    def find(self, path: str, s: float, tolerance_mm: float = SAME_SECTION_MM) -> Optional[int]:
        """Index of the saved measurement at this position of this path, if any."""
        best, best_d = None, tolerance_mm
        for i, m in enumerate(self.measurements):
            d = abs(m.arc_length_mm - s)
            if m.path == path and d <= best_d:
                best, best_d = i, d
        return best

    def save(self, item: SavedMeasurement) -> int:
        """Add a measurement, replacing one at the same position; returns its index."""
        i = self.find(item.path, item.arc_length_mm)
        if i is None:
            self.measurements.append(item)
            self.measurements.sort(key=lambda m: (m.path, m.arc_length_mm))
            return self.measurements.index(item)
        item.name = self.measurements[i].name if item.name == "" else item.name
        self.measurements[i] = item
        return i

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "units": "mm",
            "coordinate_system": "RAS",
            "measurements": [m.to_dict() for m in self.measurements],
            "markers": [m.to_dict() for m in self.markers],
            "ostia": [o.to_dict() for o in self.ostia],
            "ostia_settings": dict(self.ostia_settings),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MeasurementStore:
        from ..ostia.model import Ostium

        if d.get("schema_version", STORE_SCHEMA_VERSION) not in (1, STORE_SCHEMA_VERSION):
            raise ValueError(f"unsupported measurement schema version {d.get('schema_version')}")
        return cls(
            measurements=[SavedMeasurement.from_dict(x) for x in d.get("measurements", [])],
            markers=[Marker.from_dict(x) for x in d.get("markers", [])],
            ostia=[Ostium.from_dict(x) for x in d.get("ostia", [])],
            ostia_settings=dict(d.get("ostia_settings", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> MeasurementStore:
        return cls.from_dict(json.loads(text)) if text else cls()
