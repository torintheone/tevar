@echo off
rem Run all AortaPlan tests on Windows.
rem Uses the Python launcher "py" if a regular Python is installed,
rem otherwise Slicer's own PythonSlicer.exe (needs: PythonSlicer.exe -m pip install pytest).
rem Extra arguments are passed to scripts\run_tests.py, for example:
rem   scripts\run_tests.bat --slicer-gui
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%run_tests.py" %*
  exit /b %ERRORLEVEL%
)

set "PYTHONSLICER="
for /d %%D in ("%LOCALAPPDATA%\slicer.org\Slicer *") do (
  if exist "%%D\bin\PythonSlicer.exe" set "PYTHONSLICER=%%D\bin\PythonSlicer.exe"
)
if defined PYTHONSLICER (
  "%PYTHONSLICER%" "%SCRIPT_DIR%run_tests.py" %*
  exit /b %ERRORLEVEL%
)

echo Python not found: install Python 3 or 3D Slicer, or run run_tests.py with PythonSlicer.exe manually.
exit /b 1
