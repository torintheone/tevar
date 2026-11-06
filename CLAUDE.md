# CLAUDE.md — AortaPlan rules (read at the start of every session)

Full specification: `SPEC.md` (Russian). Status: `PROGRESS.md`. Re-read both before every stage.

## Workflow (SPEC §0)
- Work stage by stage (SPEC §8), strictly in order. Do not start the next stage before the
  acceptance criteria of the current one are met.
- Before a stage: re-read SPEC.md and PROGRESS.md, show the user a short plan, wait for approval.
- End of a stage: all tests pass; acceptance criteria met; PROGRESS.md updated (done, not done,
  limitations); git commit + push; give the user a 5–10 item checklist to verify by hand in Slicer.
- Verify facts about the Slicer API, extensions and versions in the sources (Slicer tag `v5.12.0`,
  `vmtk/SlicerExtension-VMTK`, `lassoan/SlicerSegmentEditorExtraEffects`, discourse.slicer.org)
  instead of guessing. `raw.githubusercontent.com` is reachable from the cloud container.
- If something in the spec is impossible or there is a clearly better way, say so before implementing.
- Code, comments and identifiers in English. UI text in Russian, all strings in
  `core/aortaplan_core/i18n.py` (shared by the Slicer UI and reports).
- Never claim something works without running the tests. Report explicitly what was tested in the
  container and what the user must check in Slicer.

## Environment
- User: Windows 10/11 x64, 3D Slicer 5.12 (Python 3.12.10, VTK 9.6). No Slicer build from source.
- Claude usually runs in a Linux cloud container **without Slicer** (download.slicer.org is blocked):
  core tests run there; Slicer tests are run by the user (`scripts\run_tests.bat`); GitHub Actions
  runs the core tests on Windows and Linux.
- Core code must stay Python 3.9+ compatible (`from __future__ import annotations`, `typing.Tuple`
  etc. in runtime expressions, no `match`).

## Constraints (SPEC §2, §10)
- Research use only. "Только для исследований. Не для принятия клинических решений." in the
  application window and in every report.
- Patient data: never open, read or analyse real patient studies. Only `data/phantoms/`
  (synthetic) and `data/public/` (public anonymised sets placed there by the user).
- Automation proposes, the physician confirms: every automatic action has a manual correction.
- Every displayed number has units: mm, °, mm².
- No mandatory heavy dependencies (TotalSegmentator etc. optional only). Ask before adding any new
  heavy dependency (core deps: numpy, scipy, vtk).
- Performance targets: SPEC §7 (≥15 fps scrolling, segmentation + centerline ≤30–60 s, progress UI).

## Architecture (SPEC §3)
- `core/aortaplan_core/`: all surgical math. **No `import slicer` / `qt` / `ctk`**, and no plain
  `import vtk` (use `vtkmodules.*`; enforced by `core/tests/test_core_purity.py`).
- Core I/O: numpy arrays, `vtkPolyData` / `vtkImageData`; inputs/outputs are JSON-serialisable
  dataclasses with `to_dict` / `from_dict`.
- `slicer_extension/AortaPlan/`: Slicer shell only (load, display, interaction, call the core).
  `.ui` files in `Resources/UI/`, helpers in `AortaPlanLib/`, integration tests in `Testing/`.
  Only `AortaPlan.py` may live directly in the module folder (Slicer scans it for modules).
- Extensions: SlicerVMTK, SegmentEditorExtraEffects, checked at start-up with a clear message
  (`AortaPlanLib/environment.py`). Sandbox only if CPR is missing from core Slicer (stage 3).
- Python packages inside Slicer: `slicer.util.pip_install` after a start-up check and confirmation.

## Coordinates and measurements (SPEC §4)
- Core works in RAS millimetres. RAS↔LPS conversion only at boundaries (NRRD/DICOM I/O, Slicer
  bridge), covered by tests.
- Volume arrays are `[k, j, i]` (same as `slicer.util.arrayFromVolume`). `ImageGeometry`:
  `ras = origin + directions @ (spacing * ijk)`, origin = centre of voxel (0, 0, 0).
- Patient orientation for clock positions and angles comes from the image geometry, never from
  the screen.
- Clock position: plane perpendicular to the aortic centerline at the ostium; 12:00 = anterior
  (projection of A), 3:00 = patient left, 9:00 = patient right (radiological axial view from the
  feet). Output h:mm (step ≤ 15 min), degrees 0–360 clockwise, arc mm `L = angle/360·π·D`. In the
  arch the anterior reference degenerates: use a rotation-minimizing frame transported from the
  descending aorta and/or 12:00 at the greater curvature; mode is a setting and is printed in the report.
- Cross-section view orientation (stage 1, `centerline/navigation.py`): anterior up, patient's left on
  the right (axial seen from the feet); in degenerate runs (arch top, horizontal vessels) the missing
  direction is carried from the distal side by rotation-minimizing transport. Curves run proximal → distal.
- Lumen (stage 2, `lumen/segmentation.py`): lower threshold = 50 % level between soft tissue and contrast
  (boundary at the true wall), opening against leaks, ROI grows until seeds connect. Centerline
  (`centerline/extraction.py`): minimal path 1/D^2 + half-max ray recentering; VMTK is the alternative.
- CPR (stage 3, `centerline/cpr.py`): straightened, rows = arc length (0.5 mm), lateral lines from a
  rotation-minimizing frame (angle 0 = patient left at the proximal end); two views at angle and +90°.
  In Slicer each CPR image is a volume placed far above the patient (z = 50 000 / 60 000 mm).
- Lengths: along the centerline (primary); along greater/lesser curvature (from wall contour points);
  straight distance only as a secondary value.
- Diameters per cross-section: min/max Feret (rotating calipers), mean, area-equivalent `2√(A/π)`,
  perimeter-equivalent `P/π`, area, perimeter — each for lumen (inner) and outer wall.
  Mean = (min + max) / 2. Contours (stage 4, `crosssection/contours.py`): 360 rays from the lumen
  centre in `SectionPlane` (u = image right = patient left, v = anterior); lumen = local 50 % level
  soft tissue/contrast (+ calcium stop), outer wall = 50 % wall/thrombus (40 HU) vs fat (−100 HU);
  outliers vs a robust Fourier fit are rejected and filled. Markers re-project their 3-D point on the
  current path; distances between branches go through the split point.
- C-arm: LAO/RAO = rotation about the patient's long axis (LAO = detector towards patient's left);
  CRAN/CAUD = tilt (CRAN = detector towards the head). Document the math convention in code
  (source→detector ray, signs) and test known geometries (AP = 0°/0°, LAO 90° = detector on the left…).

## Commands
- All tests: `python scripts/run_tests.py` (core pytest, then Slicer if found);
  `--slicer-gui` adds application-mode tests; `--core-only`; `--slicer PATH`.
- Core tests only: `cd core && python -m pytest`.
- Lint: `ruff check .`
- Phantoms: `cd core && python -m aortaplan_core.phantoms` → `data/phantoms/` (generated, gitignored).
