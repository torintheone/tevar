@echo off
rem Run all AortaPlan tests on Windows (core tests + tests inside Slicer).
rem
rem Python used to run the test script, in this order:
rem   1. the Python launcher "py" (a regular Python installation);
rem   2. PythonSlicer.exe next to %SLICER_EXECUTABLE% (set it once with:
rem        setx SLICER_EXECUTABLE "E:\Program\3D Slicer 5.12.4\Slicer.exe" );
rem   3. PythonSlicer.exe of Slicer installed in the default folder (%LOCALAPPDATA%\slicer.org).
rem PythonSlicer needs pytest once:  "<Slicer folder>\bin\PythonSlicer.exe" -m pip install pytest
rem
rem Extra arguments are passed to scripts\run_tests.py, for example:
rem   scripts\run_tests.bat --slicer-gui
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 goto use_py

set "PYTHONSLICER="
if not defined SLICER_EXECUTABLE goto search_default
for %%F in ("%SLICER_EXECUTABLE%") do set "PYTHONSLICER=%%~dpFbin\PythonSlicer.exe"
if exist "%PYTHONSLICER%" goto use_pythonslicer
echo PythonSlicer.exe not found next to SLICER_EXECUTABLE=%SLICER_EXECUTABLE%

:search_default
set "PYTHONSLICER="
for /d %%D in ("%LOCALAPPDATA%\slicer.org\Slicer *") do if exist "%%D\bin\PythonSlicer.exe" set "PYTHONSLICER=%%D\bin\PythonSlicer.exe"
if defined PYTHONSLICER goto use_pythonslicer

echo Python not found.
echo Tell the script where Slicer is (once, then open a new command prompt), for example:
echo   setx SLICER_EXECUTABLE "E:\Program\3D Slicer 5.12.4\Slicer.exe"
exit /b 1

:use_py
py -3 "%SCRIPT_DIR%run_tests.py" %*
exit /b %ERRORLEVEL%

:use_pythonslicer
echo Using %PYTHONSLICER%
"%PYTHONSLICER%" "%SCRIPT_DIR%run_tests.py" %*
exit /b %ERRORLEVEL%
