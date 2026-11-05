"""Display settings shared by viewers: window/level presets (SPEC 5).

Values are defaults; they can be changed in the settings later. Names are translated
with ``i18n`` keys ``wl.<key>``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class WindowLevelPreset:
    key: str
    window: float  # HU
    level: float  # HU

    @property
    def range_hu(self):
        return (self.level - self.window / 2.0, self.level + self.window / 2.0)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


WINDOW_LEVEL_PRESETS: List[WindowLevelPreset] = [
    WindowLevelPreset("cta", window=700.0, level=200.0),
    WindowLevelPreset("calcium", window=1500.0, level=500.0),
    WindowLevelPreset("soft_tissue", window=400.0, level=40.0),
    WindowLevelPreset("bone", window=2000.0, level=500.0),
]

DEFAULT_PRESET_KEY = "cta"


def window_level_preset(key: str) -> WindowLevelPreset:
    for preset in WINDOW_LEVEL_PRESETS:
        if preset.key == key:
            return preset
    raise KeyError(key)
