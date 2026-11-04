"""All user-facing strings of AortaPlan (Slicer UI and reports) in one place.

Usage::

    from aortaplan_core.i18n import tr
    label.text = tr("appmode.enable")
    tr("env.extension_hint", name="SlicerVMTK")

The default language is Russian. To add a language, add a dictionary with the same
keys to ``_STRINGS``; ``tests/test_i18n.py`` checks that all languages have the same keys.
Missing keys fall back to Russian, then to the key itself.
"""

from __future__ import annotations

from typing import Dict, List

DEFAULT_LANGUAGE = "ru"

_STRINGS: Dict[str, Dict[str, str]] = {
    "ru": {
        "app.name": "AortaPlan",
        "app.window_title": "AortaPlan — только для исследований",
        "disclaimer": "Только для исследований. Не для принятия клинических решений.",
        "module.help": (
            "AortaPlan — планирование TEVAR, EVAR и FEVAR/BEVAR по КТ-ангиографии.<br>"
            "<b>Только для исследований. Не для принятия клинических решений.</b>"
        ),
        "module.acknowledgement": "Исследовательский прототип. Не является медицинским изделием.",
        # Application mode
        "appmode.section": "Режим работы",
        "appmode.enable": "Режим AortaPlan",
        "appmode.enable.tooltip": (
            "Скрыть меню, панели инструментов, выбор модулей, Data Probe и Python-консоль Slicer, "
            "оставив только интерфейс AortaPlan."
        ),
        "appmode.show_slicer_ui": "Показать интерфейс Slicer",
        "appmode.show_slicer_ui.tooltip": (
            "Временно вернуть стандартный интерфейс Slicer, не выходя из режима AortaPlan (для отладки)."
        ),
        # Environment check
        "env.section": "Проверка окружения",
        "env.recheck": "Проверить снова",
        "env.status.ok": "есть",
        "env.status.missing": "нет",
        "env.status.optional_missing": "нет (необязательно)",
        "env.status.warning": "внимание",
        "env.item.core": "Ядро AortaPlan",
        "env.item.python_package": "Python-пакет {name}",
        "env.item.slicer_version": "Версия Slicer",
        "env.item.extension": "Расширение {name}",
        "env.item.optional_extension": "Расширение {name} (необязательно)",
        "env.hint.core": "Не удалось загрузить ядро: {error}",
        "env.hint.python_package": "Нажмите «Установить недостающие пакеты» или выполните pip_install('{name}').",
        "env.hint.slicer_version": "Рекомендуется Slicer {recommended}; установлена версия {current}.",
        "env.hint.extension": (
            "Установите расширение «{name}» через Extensions Manager (меню View → Extensions Manager) "
            "и перезапустите Slicer."
        ),
        "env.all_ok": "Все обязательные компоненты на месте.",
        "env.some_missing": "Не хватает обязательных компонентов — см. список выше.",
        "deps.install_button": "Установить недостающие пакеты",
        "deps.confirm": "Не хватает Python-пакетов: {packages}. Установить их сейчас? Это может занять несколько минут.",
        "deps.installed": "Пакеты установлены. Перезапустите Slicer.",
        "deps.failed": "Не удалось установить пакеты: {error}",
    },
    "en": {
        "app.name": "AortaPlan",
        "app.window_title": "AortaPlan — research use only",
        "disclaimer": "Research use only. Not for clinical decision making.",
        "module.help": (
            "AortaPlan — TEVAR, EVAR and FEVAR/BEVAR planning on CT angiography.<br>"
            "<b>Research use only. Not for clinical decision making.</b>"
        ),
        "module.acknowledgement": "Research prototype. Not a medical device.",
        "appmode.section": "Mode",
        "appmode.enable": "AortaPlan mode",
        "appmode.enable.tooltip": (
            "Hide Slicer menus, toolbars, module selector, Data Probe and Python console, "
            "keeping only the AortaPlan interface."
        ),
        "appmode.show_slicer_ui": "Show Slicer interface",
        "appmode.show_slicer_ui.tooltip": (
            "Temporarily show the standard Slicer interface without leaving AortaPlan mode (debugging)."
        ),
        "env.section": "Environment check",
        "env.recheck": "Check again",
        "env.status.ok": "ok",
        "env.status.missing": "missing",
        "env.status.optional_missing": "missing (optional)",
        "env.status.warning": "warning",
        "env.item.core": "AortaPlan core",
        "env.item.python_package": "Python package {name}",
        "env.item.slicer_version": "Slicer version",
        "env.item.extension": "Extension {name}",
        "env.item.optional_extension": "Extension {name} (optional)",
        "env.hint.core": "Could not load the core: {error}",
        "env.hint.python_package": "Click 'Install missing packages' or run pip_install('{name}').",
        "env.hint.slicer_version": "Slicer {recommended} is recommended; installed version is {current}.",
        "env.hint.extension": (
            "Install the '{name}' extension using the Extensions Manager (View → Extensions Manager) "
            "and restart Slicer."
        ),
        "env.all_ok": "All required components are present.",
        "env.some_missing": "Some required components are missing — see the list above.",
        "deps.install_button": "Install missing packages",
        "deps.confirm": "Missing Python packages: {packages}. Install them now? This may take a few minutes.",
        "deps.installed": "Packages installed. Please restart Slicer.",
        "deps.failed": "Could not install packages: {error}",
    },
}

_current_language = DEFAULT_LANGUAGE


def available_languages() -> List[str]:
    return sorted(_STRINGS)


def get_language() -> str:
    return _current_language


def set_language(language: str) -> None:
    global _current_language
    if language not in _STRINGS:
        raise ValueError(f"unknown language {language!r}; available: {available_languages()}")
    _current_language = language


def keys(language: str = DEFAULT_LANGUAGE) -> List[str]:
    return sorted(_STRINGS[language])


def tr(key: str, **kwargs: object) -> str:
    """Translated string for ``key`` in the current language, formatted with ``kwargs``."""
    text = _STRINGS[_current_language].get(key)
    if text is None:
        text = _STRINGS[DEFAULT_LANGUAGE].get(key, key)
    return text.format(**kwargs) if kwargs else text
