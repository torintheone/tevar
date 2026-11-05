import pytest

from aortaplan_core.display import DEFAULT_PRESET_KEY, WINDOW_LEVEL_PRESETS, window_level_preset


def test_presets():
    keys = [p.key for p in WINDOW_LEVEL_PRESETS]
    assert keys == ["cta", "calcium", "soft_tissue", "bone"]
    cta = window_level_preset(DEFAULT_PRESET_KEY)
    assert (cta.window, cta.level) == (700.0, 200.0)
    assert cta.range_hu == (-150.0, 550.0)
    with pytest.raises(KeyError):
        window_level_preset("nope")
