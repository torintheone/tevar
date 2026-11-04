"""Start-up checks: core library, Python packages, Slicer version and extensions."""

import importlib
import logging
from dataclasses import dataclass

import slicer

try:
    from aortaplan_core.i18n import tr
except ImportError:  # the check below reports the missing core

    def tr(key, **kwargs):
        return key


RECOMMENDED_SLICER_VERSION = (5, 12)

#: (import name, pip requirement). numpy, scipy and vtk ship with Slicer 5.x.
PYTHON_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("vtkmodules", "vtk"),
]

#: (extension name, module names; any one of them proves the extension is installed)
REQUIRED_EXTENSIONS = [
    ("SlicerVMTK", ["ExtractCenterline"]),
    ("SegmentEditorExtraEffects", ["SegmentEditorLocalThreshold", "SegmentEditorFloodFilling"]),
]
OPTIONAL_EXTENSIONS = [
    ("TotalSegmentator", ["TotalSegmentator"]),
]

OK = "ok"
MISSING = "missing"
OPTIONAL_MISSING = "optional_missing"
WARNING = "warning"


@dataclass
class CheckItem:
    key: str
    label: str
    status: str
    required: bool
    hint: str = ""
    detail: str = ""

    @property
    def blocking(self):
        return self.required and self.status == MISSING


def moduleAvailable(moduleName):
    return hasattr(slicer.modules, moduleName.lower())


def missingPythonPackages():
    """pip requirements of the packages that cannot be imported."""
    missing = []
    for importName, requirement in PYTHON_PACKAGES:
        try:
            importlib.import_module(importName)
        except ImportError:
            missing.append(requirement)
    return missing


def checkEnvironment():
    """Run all checks and return a list of :class:`CheckItem`."""
    items = []

    try:
        import aortaplan_core

        items.append(CheckItem("core", tr("env.item.core"), OK, True, detail=aortaplan_core.__version__))
    except Exception as exc:  # report any failure, not only ImportError
        logging.exception("AortaPlan: could not import aortaplan_core")
        items.append(CheckItem("core", tr("env.item.core"), MISSING, True, hint=tr("env.hint.core", error=exc)))

    missing = set(missingPythonPackages())
    for _importName, requirement in PYTHON_PACKAGES:
        ok = requirement not in missing
        items.append(
            CheckItem(
                f"package:{requirement}",
                tr("env.item.python_package", name=requirement),
                OK if ok else MISSING,
                True,
                hint="" if ok else tr("env.hint.python_package", name=requirement),
            )
        )

    current = (int(slicer.app.majorVersion), int(slicer.app.minorVersion))
    versionText = f"{current[0]}.{current[1]}"
    recommendedText = "{}.{}".format(*RECOMMENDED_SLICER_VERSION)
    versionOk = current >= RECOMMENDED_SLICER_VERSION
    items.append(
        CheckItem(
            "slicer_version",
            tr("env.item.slicer_version"),
            OK if versionOk else WARNING,
            False,
            hint="" if versionOk else tr("env.hint.slicer_version", recommended=recommendedText, current=versionText),
            detail=versionText,
        )
    )

    for extensions, required in ((REQUIRED_EXTENSIONS, True), (OPTIONAL_EXTENSIONS, False)):
        for name, modules in extensions:
            ok = any(moduleAvailable(m) for m in modules)
            label = tr("env.item.extension" if required else "env.item.optional_extension", name=name)
            status = OK if ok else (MISSING if required else OPTIONAL_MISSING)
            hint = "" if ok or not required else tr("env.hint.extension", name=name)
            items.append(CheckItem(f"extension:{name}", label, status, required, hint=hint))
    return items


def installMissingPythonPackages(ask=True):
    """Install missing packages with ``slicer.util.pip_install`` after confirmation.

    Returns the list of installed requirements (empty if nothing was missing or the
    user declined).
    """
    missing = missingPythonPackages()
    if not missing:
        return []
    if ask and not slicer.util.confirmOkCancelDisplay(tr("deps.confirm", packages=", ".join(missing))):
        return []
    try:
        slicer.util.pip_install(missing)
    except Exception as exc:
        slicer.util.errorDisplay(tr("deps.failed", error=exc))
        raise
    slicer.util.infoDisplay(tr("deps.installed"))
    return missing


def formatReportHtml(items):
    """Small HTML table for the module panel."""
    colors = {OK: "#2e7d32", MISSING: "#b71c1c", OPTIONAL_MISSING: "#757575", WARNING: "#e65100"}
    statusText = {
        OK: tr("env.status.ok"),
        MISSING: tr("env.status.missing"),
        OPTIONAL_MISSING: tr("env.status.optional_missing"),
        WARNING: tr("env.status.warning"),
    }
    rows = []
    for item in items:
        detail = f" ({item.detail})" if item.detail else ""
        rows.append(
            f"<tr><td>{item.label}{detail}</td>"
            f"<td style='color:{colors[item.status]}'><b>{statusText[item.status]}</b></td></tr>"
        )
        if item.hint:
            rows.append(f"<tr><td colspan='2' style='color:#555555'>{item.hint}</td></tr>")
    summary = tr("env.some_missing") if any(i.blocking for i in items) else tr("env.all_ok")
    return "<table cellspacing='4'>" + "".join(rows) + f"</table><p>{summary}</p>"
