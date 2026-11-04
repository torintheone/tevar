"""Architecture rules from SPEC section 3, enforced as tests."""

import pathlib
import re

import aortaplan_core

CORE_DIR = pathlib.Path(aortaplan_core.__file__).parent
# Slicer, Qt and CTK must never be imported by the core. Plain "import vtk" is also banned:
# it loads rendering libraries; the core uses vtkmodules.* sub-modules instead.
FORBIDDEN = re.compile(r"^\s*(?:import|from)\s+(slicer|qt|ctk|vtk|PythonQt|mrml)\b", re.MULTILINE)


def test_core_does_not_import_slicer_qt_or_ctk():
    offenders = []
    for path in CORE_DIR.rglob("*.py"):
        for match in FORBIDDEN.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(CORE_DIR)}: {match.group(0).strip()}")
    assert not offenders, "forbidden imports in core:\n" + "\n".join(offenders)


def test_all_planned_subpackages_exist():
    for name in ("geometry", "centerline", "crosssection", "ostia", "carm", "sizing", "printing", "report", "phantoms"):
        assert (CORE_DIR / name / "__init__.py").is_file(), name
