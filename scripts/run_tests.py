#!/usr/bin/env python3
"""Run all AortaPlan tests with one command.

1. Core tests (pytest, plain Python, no Slicer).
2. Integration tests inside Slicer, if Slicer is found (``--slicer``, the
   ``SLICER_EXECUTABLE`` environment variable, or the default install locations).

Examples::

    python scripts/run_tests.py                  # core + Slicer (without main window)
    python scripts/run_tests.py --slicer-gui     # also application-mode tests (window opens briefly)
    python scripts/run_tests.py --core-only
    python scripts/run_tests.py --slicer "C:/Users/me/AppData/Local/slicer.org/Slicer 5.12.0/Slicer.exe"

Works with any Python >= 3.9 that has the core requirements (``pip install -r
core/requirements-dev.txt``), including Slicer's own ``PythonSlicer`` executable.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CORE_DIR = os.path.join(REPO, "core")
MODULE_DIR = os.path.join(REPO, "slicer_extension", "AortaPlan")
SLICER_RUNNER = os.path.join(MODULE_DIR, "Testing", "run_slicer_tests.py")


def _version_key(path):
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", path)
    return tuple(int(x or 0) for x in match.groups()) if match else (0, 0, 0)


def find_slicer(explicit=None):
    """Path of the Slicer launcher, or None."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    env = os.environ.get("SLICER_EXECUTABLE")
    if env and os.path.isfile(env):
        return env
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidates = []
    # Running under PythonSlicer: the launcher is in the parent folder of bin/.
    for name in ("Slicer.exe", "Slicer"):
        candidates.append(os.path.join(exe_dir, "..", name))
    patterns = []
    if sys.platform.startswith("win"):
        for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
            if root:
                patterns.append(os.path.join(root, "slicer.org", "Slicer *", "Slicer.exe"))
                patterns.append(os.path.join(root, "Slicer *", "Slicer.exe"))
    elif sys.platform == "darwin":
        patterns.append("/Applications/Slicer*.app/Contents/MacOS/Slicer")
    else:
        patterns += [os.path.expanduser("~/Slicer*/Slicer"), "/opt/Slicer*/Slicer", "/opt/slicer*/Slicer"]
    found = []
    for pattern in patterns:
        found += glob.glob(pattern)
    candidates += sorted(found, key=_version_key, reverse=True)
    for path in candidates:
        if os.path.isfile(path):
            return os.path.normpath(path)
    return None


def run_core_tests(pytest_args):
    print("=== Core tests (pytest) ===", flush=True)
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "pytest is not installed for this Python:\n"
            f"  {sys.executable} -m pip install -r core/requirements-dev.txt",
            file=sys.stderr,
        )
        return 2
    return subprocess.call([sys.executable, "-m", "pytest", *pytest_args], cwd=CORE_DIR)


def run_slicer_tests(slicer, gui, timeout):
    print(f"=== Slicer integration tests ({slicer}) ===", flush=True)
    handle, results_path = tempfile.mkstemp(prefix="aortaplan_slicer_results_", suffix=".json")
    os.close(handle)
    os.remove(results_path)
    env = dict(os.environ, AORTAPLAN_TEST_RESULTS=results_path)
    cmd = [slicer, "--no-splash"]
    if not gui:
        cmd.append("--no-main-window")
    cmd += ["--additional-module-paths", MODULE_DIR, "--python-script", SLICER_RUNNER]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        code, output = proc.returncode, proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        code, output = None, (exc.stdout or b"").decode("utf-8", errors="replace")
    elapsed = time.perf_counter() - start

    if not os.path.isfile(results_path):
        print(output)
        reason = f"timed out after {timeout} s" if code is None else f"exit code {code}"
        print(f"Slicer did not write a test report ({reason}).", file=sys.stderr)
        return 1
    with open(results_path, encoding="utf-8") as f:
        report = json.load(f)
    os.remove(results_path)

    print(f"Slicer {report.get('slicer_version', '?')}, main window: {report.get('main_window')}")
    counts = {}
    for test in report["tests"]:
        counts[test["status"]] = counts.get(test["status"], 0) + 1
        print(f"  {test['status'].upper():8s} {test['name']} ({test['seconds']} s)")
        if test["status"] in ("failed", "error") or (test["status"] == "skipped" and test["message"]):
            for line in test["message"].rstrip().splitlines():
                print(f"           {line}")
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "no tests ran"
    print(f"Slicer tests: {summary} in {elapsed:.1f} s")
    failed = report.get("error") or counts.get("failed") or counts.get("error") or not report["tests"]
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AortaPlan core and Slicer tests.")
    parser.add_argument("--core-only", action="store_true", help="run only the core (pytest) tests")
    parser.add_argument("--slicer-only", action="store_true", help="run only the tests inside Slicer")
    parser.add_argument("--slicer", help="path to the Slicer executable (Slicer.exe)")
    parser.add_argument("--slicer-gui", action="store_true", help="run Slicer with its main window (app-mode tests)")
    parser.add_argument("--timeout", type=int, default=600, help="Slicer timeout in seconds (default 600)")
    parser.add_argument("pytest_args", nargs="*", help="extra arguments passed to pytest (after --)")
    args = parser.parse_args(argv)

    results = {}
    if not args.slicer_only:
        results["core"] = run_core_tests(args.pytest_args or ["-q"])
    if not args.core_only:
        slicer = find_slicer(args.slicer)
        if slicer is None:
            if args.slicer:
                print(f"Slicer executable not found: {args.slicer}", file=sys.stderr)
                results["slicer"] = 1
            else:
                print("=== Slicer integration tests: SKIPPED (Slicer not found; use --slicer PATH) ===")
        else:
            results["slicer"] = run_slicer_tests(slicer, args.slicer_gui, args.timeout)

    print("=== Summary ===")
    for name, code in results.items():
        print(f"  {name}: {'OK' if code == 0 else 'FAILED'}")
    return 0 if results and all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
