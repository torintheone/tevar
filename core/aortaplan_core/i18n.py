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
        "app.version": "AortaPlan {version} · этап {stage}",
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
        # Wizard
        "wizard.step.study": "Исследование",
        "wizard.step.centerline": "Центральная линия",
        "wizard.step.measurements": "Измерения",
        "wizard.step.ostia": "Устья",
        "wizard.step.carm": "Углы С-дуги",
        "wizard.step.report": "Отчёт",
        "wizard.step.printing": "3D-печать",
        "wizard.back": "← Назад",
        "wizard.next": "Далее →",
        "wizard.placeholder": "Этот шаг появится на этапе {stage}.",
        # Step 1: study
        "study.import_dicom": "Импорт папки DICOM…",
        "study.import_dialog_title": "Папка с DICOM-файлами",
        "study.refresh": "Обновить список",
        "study.col.patient": "Пациент",
        "study.col.study": "Исследование",
        "study.col.series": "Серия",
        "study.col.images": "Срезов",
        "study.col.thickness": "Толщина, мм",
        "study.col.spacing": "Шаг, мм",
        "study.col.pixel": "Пиксель, мм",
        "study.col.warnings": "Замечания",
        "study.load_series": "Загрузить выбранную серию",
        "study.no_series": "В базе DICOM нет КТ-серий. Импортируйте папку или загрузите тестовые данные.",
        "study.series_count": "КТ-серий в базе: {count}.",
        "study.volume": "Объём:",
        "study.sample_data": "Тестовые данные: CTA abdomen (Panoramix)",
        "study.loading": "Загрузка…",
        "study.load_failed": "Не удалось загрузить: {error}",
        "study.imported": "Импорт завершён. КТ-серий в базе: {count}.",
        "study.regularized": (
            "Серия получена с наклоном гантри или неравномерным шагом: объём пересчитан на прямоугольную сетку."
        ),
        "study.volume_info": "{dims} вокселей, шаг {spacing} мм",
        "study.volume_ok": "Замечаний к геометрии нет.",
        "study.warning.thick_slices": "толщина среза {thickness_mm:.2f} мм > {limit_mm:.1f} мм",
        "study.warning.anisotropic": "анизотропные воксели (отношение {ratio:.1f})",
        "study.warning.gantry_tilt": "наклон гантри {angle_deg:.1f}°",
        "study.warning.irregular_spacing": "неравномерный шаг срезов (отклонение до {deviation_mm:.2f} мм)",
        "study.warning.few_slices": "мало срезов ({count:.0f})",
        "study.warning.no_geometry": "нет данных о положении срезов",
        # Window/level
        "wl.section": "Окно / уровень",
        "wl.cta": "КТА",
        "wl.calcium": "Кальций",
        "wl.soft_tissue": "Мягкие ткани",
        "wl.bone": "Кость",
        "wl.tooltip": "Окно {window:.0f} HU, уровень {level:.0f} HU",
        # Views and layout
        "layout.show_axial": "Показать аксиальный срез",
        "view.label.longitudinal": "CPR",
        "view.label.cross": "⊥",
        "view.position": "{value:.1f} мм",
        "view.position_angle": "{value:.1f} мм   {angle:.0f}°",
        # Step 2: centerline
        "centerline.curve": "Кривая:",
        "centerline.draw": "Нарисовать кривую",
        "centerline.finish": "Закончить рисование",
        "centerline.reverse": "Развернуть направление",
        "centerline.hint": (
            "Ставьте точки по центру просвета от проксимального конца (ближе к сердцу) к дистальному; "
            "точки можно перетаскивать. Автоматическая центральная линия появится на этапе 2."
        ),
        "centerline.length": "Длина кривой: {length:.1f} мм",
        "centerline.need_volume": "Сначала выберите объём на шаге «Исследование».",
        "centerline.too_short": "Нужно минимум две точки.",
        # Navigation
        "nav.section": "Навигация по кривой",
        "nav.position": "Положение:",
        "nav.rotation": "Поворот CPR:",
        "nav.position_of": "{value:.1f} мм из {length:.1f} мм",
        "nav.no_curve": "Нет кривой: нарисуйте её на шаге «Центральная линия».",
        "nav.help": (
            "Колесо мыши или ↑/↓ в срезе или CPR: ±1 мм (с Shift ±5 мм). "
            "Ctrl+↑/↓: ±1 мм, Ctrl+←/→: поворот CPR на ±10°. "
            "Клик в CPR или по кривой в 3D переносит срез в эту точку."
        ),
        "units.mm": "мм",
        "units.deg": "°",
        # Step 2: automatic lumen and centerline
        "auto.hint": (
            "1) «Начальная точка» — кликните по аорте на 3D-реконструкции или на срезе. "
            "2) «Конечные точки» — кликните по каждой подвздошной/бедренной (для TEVAR — и по ветвям дуги). "
            "3) «Построить». Стоп-точки ставятся там, где просвет «протёк» в кость."
        ),
        "auto.start": "Начальная точка",
        "auto.ends": "Конечные точки",
        "auto.stops": "Стоп-точки",
        "auto.clear": "Очистить точки",
        "auto.points_info": "Начало: {start} · концов: {ends} · стоп-точек: {stops}",
        "auto.yes": "есть",
        "auto.no": "нет",
        "auto.method": "Метод линии:",
        "auto.method.vmtk": "VMTK Extract Centerline",
        "auto.method.minimal_path": "Минимальный путь (ядро AortaPlan)",
        "auto.exclusion": "Исключить (кости):",
        "auto.build": "Построить просвет и центральную линию",
        "auto.rebuild_centerline": "Перестроить линию по просвету",
        "auto.need_points": "Нужны начальная точка и хотя бы одна конечная.",
        "auto.need_volume": "Сначала выберите объём на шаге «Исследование».",
        "auto.vmtk_missing": "Расширение SlicerVMTK не найдено — линия построена методом «Минимальный путь».",
        "auto.result": (
            "Просвет: {volume_ml:.1f} мл, HU {low:.0f}–{high:.0f}. Путей: {paths}. Время: {seconds:.1f} с."
        ),
        "auto.warning.seed_outside_lumen": "точка далеко от просвета — проверьте её положение",
        "auto.warning.start_not_in_lumen": "начальная точка не в просвете",
        "auto.warning.end_not_connected": "не все конечные точки соединены с начальной (стеноз/окклюзия или протекание)",
        "auto.warning.end_unreachable": "до конечной точки {index} путь не найден",
        "auto.path": "Путь для навигации:",
        "auto.path_name": "{side} ({length:.0f} мм)",
        "auto.side.right": "справа",
        "auto.side.left": "слева",
        "auto.side.center": "по центру",
        "auto.path_curve": "Центральная линия {side}",
        "auto.add_branch": "Добавить ветвь",
        "auto.show_rendering": "3D-реконструкция",
        "auto.show_lumen": "Модель просвета",
        "auto.edit_section": "Правка просвета (кисть, ластик, ножницы)",
        "auto.manual_section": "Ручная кривая",
        "auto.cancelled": "Построение отменено.",
        "auto.failed": "Не удалось построить: {error}",
        "points.start": "Начало",
        "points.end": "Конец",
        "points.stop": "Стоп",
        "segment.lumen": "Просвет",
        "progress.title": "AortaPlan",
        "progress.lumen.hu_range": "Оценка диапазона HU контраста…",
        "progress.lumen.threshold": "Выделение просвета…",
        "progress.lumen.opening": "Защита от протекания…",
        "progress.lumen.connectivity": "Связность…",
        "progress.lumen.fill_holes": "Заполнение пустот…",
        "progress.lumen.done": "Просвет готов.",
        "progress.surface": "Поверхность просвета…",
        "progress.centerline.distance": "Карта расстояний…",
        "progress.centerline.graph": "Граф просвета…",
        "progress.centerline.paths": "Поиск путей…",
        "progress.centerline.refine": "Уточнение центра просвета…",
        "progress.centerline.done": "Центральная линия готова.",
        "progress.vmtk": "VMTK Extract Centerline…",
    },
    "en": {
        "app.name": "AortaPlan",
        "app.window_title": "AortaPlan — research use only",
        "app.version": "AortaPlan {version} · stage {stage}",
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
        "wizard.step.study": "Study",
        "wizard.step.centerline": "Centerline",
        "wizard.step.measurements": "Measurements",
        "wizard.step.ostia": "Ostia",
        "wizard.step.carm": "C-arm angles",
        "wizard.step.report": "Report",
        "wizard.step.printing": "3D printing",
        "wizard.back": "← Back",
        "wizard.next": "Next →",
        "wizard.placeholder": "This step will be added in stage {stage}.",
        "study.import_dicom": "Import DICOM folder…",
        "study.import_dialog_title": "Folder with DICOM files",
        "study.refresh": "Refresh",
        "study.col.patient": "Patient",
        "study.col.study": "Study",
        "study.col.series": "Series",
        "study.col.images": "Images",
        "study.col.thickness": "Thickness, mm",
        "study.col.spacing": "Spacing, mm",
        "study.col.pixel": "Pixel, mm",
        "study.col.warnings": "Warnings",
        "study.load_series": "Load selected series",
        "study.no_series": "No CT series in the DICOM database. Import a folder or load the sample data.",
        "study.series_count": "CT series in the database: {count}.",
        "study.volume": "Volume:",
        "study.sample_data": "Sample data: CTA abdomen (Panoramix)",
        "study.loading": "Loading…",
        "study.load_failed": "Loading failed: {error}",
        "study.imported": "Import finished. CT series in the database: {count}.",
        "study.regularized": (
            "The series has gantry tilt or irregular spacing: the volume was resampled to a rectilinear grid."
        ),
        "study.volume_info": "{dims} voxels, spacing {spacing} mm",
        "study.volume_ok": "No geometry warnings.",
        "study.warning.thick_slices": "slice thickness {thickness_mm:.2f} mm > {limit_mm:.1f} mm",
        "study.warning.anisotropic": "anisotropic voxels (ratio {ratio:.1f})",
        "study.warning.gantry_tilt": "gantry tilt {angle_deg:.1f}°",
        "study.warning.irregular_spacing": "irregular slice spacing (up to {deviation_mm:.2f} mm)",
        "study.warning.few_slices": "few slices ({count:.0f})",
        "study.warning.no_geometry": "no slice position data",
        "wl.section": "Window / level",
        "wl.cta": "CTA",
        "wl.calcium": "Calcium",
        "wl.soft_tissue": "Soft tissue",
        "wl.bone": "Bone",
        "wl.tooltip": "Window {window:.0f} HU, level {level:.0f} HU",
        "layout.show_axial": "Show axial slice",
        "view.label.longitudinal": "CPR",
        "view.label.cross": "⊥",
        "view.position": "{value:.1f} mm",
        "view.position_angle": "{value:.1f} mm   {angle:.0f}°",
        "centerline.curve": "Curve:",
        "centerline.draw": "Draw curve",
        "centerline.finish": "Finish drawing",
        "centerline.reverse": "Reverse direction",
        "centerline.hint": (
            "Place points in the lumen centre from the proximal end (towards the heart) to the distal end; "
            "points can be dragged. Automatic centerline extraction comes in stage 2."
        ),
        "centerline.length": "Curve length: {length:.1f} mm",
        "centerline.need_volume": "Select a volume in the Study step first.",
        "centerline.too_short": "At least two points are needed.",
        "nav.section": "Navigation along the curve",
        "nav.position": "Position:",
        "nav.rotation": "CPR rotation:",
        "nav.position_of": "{value:.1f} mm of {length:.1f} mm",
        "nav.no_curve": "No curve: draw one in the Centerline step.",
        "nav.help": (
            "Mouse wheel or Up/Down in the slice or CPR view: ±1 mm (with Shift ±5 mm). "
            "Ctrl+Up/Down: ±1 mm, Ctrl+Left/Right: rotate CPR by ±10°. "
            "Click in the CPR view or on the curve in 3D to move the slice there."
        ),
        "units.mm": "mm",
        "units.deg": "°",
        "auto.hint": (
            "1) 'Start point' — click on the aorta in the 3D rendering or in a slice. "
            "2) 'End points' — click on each iliac/femoral artery (for TEVAR also the arch branches). "
            "3) 'Build'. Put stop points where the lumen leaked into bone."
        ),
        "auto.start": "Start point",
        "auto.ends": "End points",
        "auto.stops": "Stop points",
        "auto.clear": "Clear points",
        "auto.points_info": "Start: {start} · ends: {ends} · stops: {stops}",
        "auto.yes": "yes",
        "auto.no": "no",
        "auto.method": "Centerline method:",
        "auto.method.vmtk": "VMTK Extract Centerline",
        "auto.method.minimal_path": "Minimal path (AortaPlan core)",
        "auto.exclusion": "Exclude (bones):",
        "auto.build": "Build lumen and centerline",
        "auto.rebuild_centerline": "Rebuild centerline from lumen",
        "auto.need_points": "A start point and at least one end point are needed.",
        "auto.need_volume": "Select a volume in the Study step first.",
        "auto.vmtk_missing": "SlicerVMTK not found — the centerline was built with the minimal path method.",
        "auto.result": (
            "Lumen: {volume_ml:.1f} ml, HU {low:.0f}–{high:.0f}. Paths: {paths}. Time: {seconds:.1f} s."
        ),
        "auto.warning.seed_outside_lumen": "a point is far from the lumen — check its position",
        "auto.warning.start_not_in_lumen": "the start point is not in the lumen",
        "auto.warning.end_not_connected": "not all end points connect to the start (stenosis/occlusion or leak)",
        "auto.warning.end_unreachable": "no path found to end point {index}",
        "auto.path": "Path for navigation:",
        "auto.path_name": "{side} ({length:.0f} mm)",
        "auto.side.right": "right",
        "auto.side.left": "left",
        "auto.side.center": "centre",
        "auto.path_curve": "Centerline {side}",
        "auto.add_branch": "Add branch",
        "auto.show_rendering": "3D rendering",
        "auto.show_lumen": "Lumen model",
        "auto.edit_section": "Lumen editing (brush, eraser, scissors)",
        "auto.manual_section": "Manual curve",
        "auto.cancelled": "Build cancelled.",
        "auto.failed": "Build failed: {error}",
        "points.start": "Start",
        "points.end": "End",
        "points.stop": "Stop",
        "segment.lumen": "Lumen",
        "progress.title": "AortaPlan",
        "progress.lumen.hu_range": "Estimating contrast HU range…",
        "progress.lumen.threshold": "Extracting lumen…",
        "progress.lumen.opening": "Leak protection…",
        "progress.lumen.connectivity": "Connectivity…",
        "progress.lumen.fill_holes": "Filling holes…",
        "progress.lumen.done": "Lumen ready.",
        "progress.surface": "Lumen surface…",
        "progress.centerline.distance": "Distance map…",
        "progress.centerline.graph": "Lumen graph…",
        "progress.centerline.paths": "Path search…",
        "progress.centerline.refine": "Refining lumen centre…",
        "progress.centerline.done": "Centerline ready.",
        "progress.vmtk": "VMTK Extract Centerline…",
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
