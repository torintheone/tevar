"""Tests of scripts/run_tests.py (the one-command test runner), using a fake Slicer."""

import json
import os
import pathlib
import stat
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_tests  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="fake Slicer is a POSIX shell script")


def make_fake_slicer(tmp_path, tests, error=None, write_report=True):
    report = {"slicer_version": "5.12.0", "main_window": False, "tests": tests, "error": error, "log": ""}
    script = tmp_path / "Slicer"
    body = f"""\
        #!{sys.executable}
        import json, os, sys
        args = sys.argv[1:]
        assert "--python-script" in args and "--additional-module-paths" in args, args
        if {write_report!r}:
            with open(os.environ["AORTAPLAN_TEST_RESULTS"], "w") as f:
                json.dump(json.loads({json.dumps(report)!r}), f)
        sys.exit(0)
    """
    script.write_text(textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_passing_slicer_run(tmp_path):
    slicer = make_fake_slicer(tmp_path, [{"name": "t1", "status": "passed", "message": "", "seconds": 0.1}])
    assert run_tests.run_slicer_tests(slicer, gui=False, timeout=60) == 0


def test_failing_slicer_run(tmp_path):
    tests = [{"name": "t1", "status": "failed", "message": "boom", "seconds": 0.1}]
    assert run_tests.run_slicer_tests(make_fake_slicer(tmp_path, tests), gui=False, timeout=60) == 1


def test_missing_report_is_a_failure(tmp_path):
    slicer = make_fake_slicer(tmp_path, [], write_report=False)
    assert run_tests.run_slicer_tests(slicer, gui=False, timeout=60) == 1


def test_find_slicer_prefers_explicit_and_env(tmp_path, monkeypatch):
    exe = tmp_path / "Slicer"
    exe.write_text("")
    assert run_tests.find_slicer(str(exe)) == str(exe)
    assert run_tests.find_slicer(str(tmp_path / "missing")) is None
    monkeypatch.setenv("SLICER_EXECUTABLE", str(exe))
    assert run_tests.find_slicer() == str(exe)


def test_version_sorting():
    paths = ["Slicer 5.8.1/Slicer.exe", "Slicer 5.12.0/Slicer.exe", "Slicer 5.10.2/Slicer.exe"]
    assert sorted(paths, key=run_tests._version_key, reverse=True)[0] == "Slicer 5.12.0/Slicer.exe"


def test_runner_paths_exist():
    assert os.path.isfile(run_tests.SLICER_RUNNER)
    assert os.path.isfile(os.path.join(run_tests.MODULE_DIR, "AortaPlan.py"))
