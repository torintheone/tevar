"""Run the AortaPlan integration tests inside Slicer, write a JSON report and exit.

Started by ``scripts/run_tests.py``::

    Slicer --no-splash --no-main-window --additional-module-paths <...>/AortaPlan \
           --python-script <...>/Testing/run_slicer_tests.py

The report path comes from the ``AORTAPLAN_TEST_RESULTS`` environment variable,
because on Windows ``Slicer.exe`` does not reliably print to the console.
Slicer is always asked to exit, even if the tests crash, so that a run without a
main window never hangs.
"""

import io
import json
import os
import sys
import time
import traceback
import unittest


class _RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._started = {}

    def startTest(self, test):
        self._started[test.id()] = time.perf_counter()
        super().startTest(test)

    def _record(self, test, status, message=""):
        duration = time.perf_counter() - self._started.get(test.id(), time.perf_counter())
        self.records.append({"name": test.id(), "status": status, "message": message, "seconds": round(duration, 3)})

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "skipped", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, "passed", "expected failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, "failed", "unexpected success")


def main():
    report = {"tests": [], "error": None, "log": ""}
    exitCode = 1
    try:
        import slicer

        report["slicer_version"] = slicer.app.applicationVersion
        report["main_window"] = slicer.util.mainWindow() is not None
        testsDir = os.path.dirname(os.path.abspath(__file__))
        if testsDir not in sys.path:
            sys.path.insert(0, testsDir)
        if not hasattr(slicer.modules, "aortaplan"):
            raise RuntimeError("AortaPlan module is not loaded; check --additional-module-paths")
        import AortaPlanSlicerTests

        suite = unittest.TestLoader().loadTestsFromModule(AortaPlanSlicerTests)
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=_RecordingResult)
        result = runner.run(suite)
        report["tests"] = result.records
        report["log"] = stream.getvalue()
        report["metrics"] = dict(getattr(AortaPlanSlicerTests, "METRICS", {}))
        exitCode = 0 if result.wasSuccessful() else 1
    except Exception:
        report["error"] = traceback.format_exc()
        exitCode = 1
    finally:
        resultsPath = os.environ.get("AORTAPLAN_TEST_RESULTS")
        if resultsPath:
            try:
                with open(resultsPath, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=1)
            except Exception:
                traceback.print_exc()
        print(report["log"] or report["error"] or "")
        try:
            import slicer

            slicer.util.exit(exitCode)
        except Exception:
            sys.exit(exitCode)


main()
