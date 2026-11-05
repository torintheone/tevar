# AortaPlan

Планирование TEVAR, EVAR и FEVAR/BEVAR по КТ-ангиографии на базе 3D Slicer.

> **Только для исследований. Не для принятия клинических решений.**
> Программа не является медицинским изделием.

Техническое задание — [`SPEC.md`](SPEC.md), состояние работ — [`PROGRESS.md`](PROGRESS.md).

## Структура

| Папка | Что внутри |
|---|---|
| `core/aortaplan_core/` | ядро: вся математика, без зависимости от Slicer (numpy, scipy, vtk) |
| `core/tests/` | тесты ядра (pytest, обычный Python) |
| `slicer_extension/AortaPlan/` | модуль Slicer: интерфейс, вызов ядра |
| `slicer_extension/AortaPlan/Testing/` | тесты внутри Slicer |
| `scripts/` | запуск тестов |
| `data/phantoms/` | синтетические фантомы (генерируются, в git не хранятся) |
| `data/public/` | публичные обезличенные данные (кладёт пользователь, в git не хранятся) |
| `validation/` | сравнение с 3mensio (этап 10) |

## Установка (Windows)

1. **3D Slicer 5.12** — установить с [download.slicer.org](https://download.slicer.org)
   (по умолчанию ставится в `%LOCALAPPDATA%\slicer.org\Slicer 5.12.x\`).
2. **Расширения Slicer.** View → Extensions Manager → Install Extensions → установить
   **SlicerVMTK** и **SegmentEditorExtraEffects** → перезапустить Slicer.
   Если чего-то не хватает, модуль AortaPlan сам покажет это в разделе «Проверка окружения».
3. **Код.** Все команды в этом README вводятся в **командной строке Windows** (Win+R → `cmd` → Enter),
   а не в Python-консоли Slicer.
   - Вариант А, через git (удобно получать обновления). Один раз установить
     [Git for Windows](https://git-scm.com/download/win) с настройками по умолчанию, открыть
     новое окно командной строки и выполнить:
     ```
     cd /d %USERPROFILE%\Documents
     git clone https://github.com/torintheone/tevar.git
     cd tevar
     git checkout ccr-0312d8ba-qbg2ch
     ```
     Код окажется в `Документы\tevar`. Обновление после каждого этапа:
     `cd /d %USERPROFILE%\Documents\tevar` и `git pull`.
   - Вариант Б, без git: скачать архив ветки
     <https://github.com/torintheone/tevar/archive/refs/heads/ccr-0312d8ba-qbg2ch.zip>
     и распаковать в `Документы`; внутри архива папка `tevar-ccr-0312d8ba-qbg2ch` — переименуйте
     её в `tevar`. Для обновления — скачать архив заново.
4. **Подключение модуля (Additional module paths).**
   Edit → Application Settings → Modules → раздел Additional module paths
   (если список скрыт, раскройте его кнопкой «>>») → Add → выбрать папку
   `<путь к tevar>\slicer_extension\AortaPlan` → OK → перезапустить Slicer.
5. **Запуск.** В списке модулей: категория **AortaPlan** → **AortaPlan**.
   Галочка «Режим AortaPlan» скрывает стандартный интерфейс Slicer,
   «Показать интерфейс Slicer» временно возвращает его (для отладки).

Ядро модуль находит сам по относительному пути (`slicer_extension/AortaPlan` → `core`).
Если папка ядра лежит в другом месте, укажите её в переменной окружения `AORTAPLAN_CORE_PATH`.

### Режим разработчика (по желанию)
Edit → Application Settings → Developer → Enable developer mode. После перезапуска в модуле
появятся кнопки «Reload» (перезагрузить код модуля без перезапуска Slicer) и «Reload and Test».
Изменения в ядре (`core/`) подхватываются только после перезапуска Slicer.

## Тесты

Все тесты одной командой — в командной строке Windows, из папки репозитория
(например, `cd /d %USERPROFILE%\Documents\tevar`):

```
scripts\run_tests.bat                 :: ядро + тесты внутри Slicer без окна
scripts\run_tests.bat --slicer-gui    :: плюс тесты режима приложения (окно Slicer откроется на несколько секунд)
scripts\run_tests.bat --core-only     :: только ядро
```

`run_tests.bat` использует обычный Python (если установлен), а иначе — `PythonSlicer.exe`
из поставки Slicer. Отдельный Python **не обязателен**; один раз нужно поставить pytest:

```
"%LOCALAPPDATA%\slicer.org\Slicer 5.12.0\bin\PythonSlicer.exe" -m pip install pytest
```

(подставьте свою версию в имени папки). С обычным Python:
`python -m pip install -r core\requirements-dev.txt`, затем `python scripts\run_tests.py`.

Slicer ищется автоматически в стандартных папках установки. Другой путь задаётся так:
`scripts\run_tests.bat --slicer "D:\Slicer 5.12.0\Slicer.exe"` или переменной `SLICER_EXECUTABLE`.

Тесты внутри Slicer запускаются командой
`Slicer.exe --no-splash --no-main-window --additional-module-paths ... --python-script ...`;
результат пишется в JSON-файл, потому что на Windows `Slicer.exe` не выводит текст в консоль.

Тесты ядра автоматически запускаются в GitHub Actions (Windows и Linux с Python 3.12, Windows с Python 3.9)
при каждом push.

## Фантомы

```
cd core
python -m aortaplan_core.phantoms
```

Создаёт в `data/phantoms/` объёмы `*.nrrd` и файлы истинных значений `*.truth.json`
(подробнее — `data/phantoms/README.md`). Фантомы можно открыть в Slicer перетаскиванием файла.

## Для разработчиков

- Линтер: `ruff check .`
- Правила проекта для Claude Code: [`CLAUDE.md`](CLAUDE.md).
