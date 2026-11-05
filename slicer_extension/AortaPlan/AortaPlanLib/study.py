"""Step 1 "Study": DICOM database queries, series loading and sample data.

DICOM attributes are read from Slicer's DICOM database and converted into
``aortaplan_core.study.SeriesInfo``; the checks themselves are done by the core.
"""

import logging
from dataclasses import dataclass

import slicer

from aortaplan_core.study import (
    SeriesAssessment,
    SeriesInfo,
    assess_series,
    parse_dicom_numbers,
    parse_dicom_strings,
    sort_candidates,
)

TAGS = {
    "patient_name": "0010,0010",
    "study_date": "0008,0020",
    "study_description": "0008,1030",
    "modality": "0008,0060",
    "series_description": "0008,103E",
    "image_type": "0008,0008",
    "slice_thickness": "0018,0050",
    "pixel_spacing": "0028,0030",
    "gantry_tilt": "0018,1120",
    "orientation": "0020,0037",
    "position": "0020,0032",
    "number_of_frames": "0028,0008",
}

SAMPLE_DATA_NAME = "CTAAbdomenPanoramix"


@dataclass
class SeriesRow:
    """One line of the series table."""

    info: SeriesInfo
    assessment: SeriesAssessment
    patient_name: str
    study: str


def database():
    db = getattr(slicer, "dicomDatabase", None)
    if db is None or not db.isOpen:
        return None
    return db


def _value(db, filePath, tag):
    try:
        return db.fileValue(filePath, TAGS[tag]) or ""
    except Exception:
        return ""


def _first_number(text):
    values = parse_dicom_numbers(text)
    return values[0] if values else None


def seriesInfoFromDatabase(db, seriesUID, readPositions=True):
    """Build a SeriesInfo from the database (first file + positions of all files)."""
    files = list(db.filesForSeries(seriesUID))
    if not files:
        return None, "", ""
    first = files[0]
    frames = _first_number(_value(db, first, "number_of_frames"))
    numberOfImages = len(files) if len(files) > 1 or not frames else int(frames)
    pixel = parse_dicom_numbers(_value(db, first, "pixel_spacing"))
    orientation = parse_dicom_numbers(_value(db, first, "orientation"))
    positions = None
    if readPositions and len(files) > 1:
        positions = []
        for path in files:
            values = parse_dicom_numbers(_value(db, path, "position"))
            if len(values) == 3:
                positions.append(tuple(values))
    info = SeriesInfo(
        series_uid=seriesUID,
        modality=_value(db, first, "modality"),
        description=_value(db, first, "series_description"),
        number_of_images=numberOfImages,
        image_types=parse_dicom_strings(_value(db, first, "image_type")),
        slice_thickness_mm=_first_number(_value(db, first, "slice_thickness")),
        pixel_spacing_mm=(pixel[0], pixel[1]) if len(pixel) == 2 else None,
        gantry_tilt_deg=_first_number(_value(db, first, "gantry_tilt")),
        image_orientation=tuple(orientation) if len(orientation) == 6 else None,
        image_positions=positions,
    )
    patient = _value(db, first, "patient_name").replace("^", " ").strip()
    study = " ".join(x for x in (_value(db, first, "study_date"), _value(db, first, "study_description")) if x)
    return info, patient, study


def listCtSeries(db=None):
    """Candidate CT series in the database, best first."""
    db = db or database()
    if db is None:
        return []
    rows = []
    for patient in db.patients():
        for study in db.studiesForPatient(patient):
            for seriesUID in db.seriesForStudy(study):
                files = db.filesForSeries(seriesUID)
                if not files or _value(db, files[0], "modality").strip().upper() != "CT":
                    continue  # skip non-CT series without reading positions
                info, patientName, studyText = seriesInfoFromDatabase(db, seriesUID)
                if info is None:
                    continue
                assessment = assess_series(info)
                if assessment.candidate:
                    rows.append(SeriesRow(info, assessment, patientName, studyText))
    ordered = sort_candidates([(row.info, row.assessment) for row in rows])
    byUid = {row.info.series_uid: row for row in rows}
    return [byUid[info.series_uid] for info, _ in ordered]


def importFolder(folder):
    from DICOMLib import DICOMUtils

    if database() is None:
        raise RuntimeError("DICOM database is not open")
    if not DICOMUtils.importDicom(folder):
        raise RuntimeError(f"could not import {folder}")


def hardenRegularization(volumeNode):
    """Apply the acquisition-geometry regularization transform added by Slicer (gantry tilt).

    Returns True if the volume was resampled.
    """
    transformNode = volumeNode.GetParentTransformNode()
    if transformNode is None:
        return False
    slicer.vtkSlicerTransformLogic().hardenTransform(volumeNode)
    if transformNode.GetName().lower().find("regularization") >= 0:
        slicer.mrmlScene.RemoveNode(transformNode)
    return True


def loadSeries(seriesUID):
    """Load a CT series as a scalar volume. Returns ``(volumeNode, regularized)``."""
    from DICOMLib import DICOMUtils

    loadedIds = DICOMUtils.loadSeriesByUID([seriesUID], enabledPluginClassNames=["DICOMScalarVolumePlugin"])
    volumes = [slicer.mrmlScene.GetNodeByID(i) for i in loadedIds]
    volumes = [v for v in volumes if v is not None and v.IsA("vtkMRMLScalarVolumeNode")]
    if not volumes:
        raise RuntimeError("no scalar volume was loaded")
    volume = volumes[0]
    regularized = hardenRegularization(volume)
    return volume, regularized


def loadSampleData():
    import SampleData

    node = SampleData.SampleDataLogic().downloadSample(SAMPLE_DATA_NAME)
    if isinstance(node, (list, tuple)):
        node = node[0]
    if node is None:
        raise RuntimeError("sample data download failed")
    logging.info("AortaPlan: loaded sample data %s", node.GetName())
    return node
