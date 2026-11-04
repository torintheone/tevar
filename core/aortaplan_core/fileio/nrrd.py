"""Minimal NRRD reader/writer for 3-D scalar volumes (no extra dependencies).

Files are written in the convention used by Slicer and DICOM-derived NRRDs:
``space: left-posterior-superior``. The in-memory :class:`~aortaplan_core.volume.Volume`
is always RAS; the RAS<->LPS conversion happens only here.

Supported on read: NRRD0001-0005, 3-D scalar data, ``raw`` and ``gzip`` encodings,
attached or single detached data file, spaces LPS / RAS / LAS.
Format reference: http://teem.sourceforge.net/nrrd/format.html
"""

from __future__ import annotations

import gzip
import os
import re
from typing import Dict, Tuple, Union

import numpy as np

from ..volume import ImageGeometry, Volume

PathLike = Union[str, "os.PathLike[str]"]

_WRITE_TYPES = {
    np.dtype(np.int8): "signed char",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.int64): "longlong",
    np.dtype(np.uint64): "ulonglong",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}

_READ_TYPES = {}
for _names, _dtype in [
    (("signed char", "int8", "int8_t"), np.int8),
    (("uchar", "unsigned char", "uint8", "uint8_t"), np.uint8),
    (("short", "short int", "signed short", "signed short int", "int16", "int16_t"), np.int16),
    (("ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t"), np.uint16),
    (("int", "signed int", "int32", "int32_t"), np.int32),
    (("uint", "unsigned int", "uint32", "uint32_t"), np.uint32),
    (
        ("longlong", "long long", "long long int", "signed long long", "signed long long int", "int64", "int64_t"),
        np.int64,
    ),
    (("ulonglong", "unsigned long long", "unsigned long long int", "uint64", "uint64_t"), np.uint64),
    (("float",), np.float32),
    (("double",), np.float64),
]:
    for _name in _names:
        _READ_TYPES[_name] = np.dtype(_dtype)

# Sign flips that convert each supported NRRD space into RAS.
_SPACE_TO_RAS = {
    "left-posterior-superior": np.array([-1.0, -1.0, 1.0]),
    "lps": np.array([-1.0, -1.0, 1.0]),
    "right-anterior-superior": np.array([1.0, 1.0, 1.0]),
    "ras": np.array([1.0, 1.0, 1.0]),
    "left-anterior-superior": np.array([-1.0, 1.0, 1.0]),
    "las": np.array([-1.0, 1.0, 1.0]),
}


def _fmt(x: float) -> str:
    # Shortest representation that round-trips exactly; "+ 0.0" turns -0.0 into 0.0.
    return repr(float(x) + 0.0)


def _vec(v: np.ndarray) -> str:
    return "(" + ",".join(_fmt(x) for x in v) + ")"


def write_nrrd(path: PathLike, volume: Volume, compress: bool = True) -> None:
    """Write a volume as an attached-header NRRD in LPS space."""
    data = np.ascontiguousarray(volume.data)
    if data.dtype not in _WRITE_TYPES:
        raise TypeError(f"unsupported voxel type {data.dtype}")
    m_ras = volume.geometry.ijk_to_ras_matrix()
    flip = _SPACE_TO_RAS["lps"]  # RAS->LPS uses the same sign flips as LPS->RAS
    columns_lps = m_ras[:3, :3] * flip[:, np.newaxis]
    origin_lps = m_ras[:3, 3] * flip
    ni, nj, nk = volume.geometry.dimensions
    header = [
        "NRRD0004",
        "# Complete NRRD file format specification at:",
        "# http://teem.sourceforge.net/nrrd/format.html",
        f"type: {_WRITE_TYPES[data.dtype]}",
        "dimension: 3",
        "space: left-posterior-superior",
        f"sizes: {ni} {nj} {nk}",
        "space directions: " + " ".join(_vec(columns_lps[:, a]) for a in range(3)),
        "kinds: domain domain domain",
        "endian: little",
        f"encoding: {'gzip' if compress else 'raw'}",
        f"space origin: {_vec(origin_lps)}",
    ]
    payload = data.astype(data.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
    if compress:
        payload = gzip.compress(payload, compresslevel=6)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n\n").encode("ascii"))
        f.write(payload)


def _parse_vector(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip().strip("()").split(",")])


def _parse_header(raw: bytes) -> Tuple[Dict[str, str], int]:
    """Return header fields and the byte offset where attached data starts."""
    if not raw.startswith(b"NRRD000"):
        raise ValueError("not a NRRD file (missing NRRD000x magic)")
    fields: Dict[str, str] = {}
    pos = 0
    first = True
    while True:
        end = raw.find(b"\n", pos)
        if end < 0:
            # Header without data (detached) that does not end with a blank line.
            line = raw[pos:].decode("ascii", errors="replace").rstrip("\r")
            pos = len(raw)
        else:
            line = raw[pos:end].decode("ascii", errors="replace").rstrip("\r")
            pos = end + 1
        if first:
            first = False
            continue
        if line == "":
            break
        if line.startswith("#") or ":=" in line:
            continue  # comment or key/value pair
        if ":" not in line:
            raise ValueError(f"malformed NRRD header line: {line!r}")
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
        if pos >= len(raw):
            break
    return fields, pos


def read_nrrd(path: PathLike) -> Volume:
    """Read a 3-D scalar NRRD into an RAS :class:`Volume`."""
    with open(path, "rb") as f:
        raw = f.read()
    fields, data_offset = _parse_header(raw)

    if int(fields.get("dimension", "0")) != 3:
        raise ValueError("only 3-D scalar NRRD volumes are supported")
    type_name = re.sub(r"\s+", " ", fields["type"].lower())
    if type_name not in _READ_TYPES:
        raise ValueError(f"unsupported NRRD type {fields['type']!r}")
    dtype = _READ_TYPES[type_name]
    if dtype.itemsize > 1:
        endian = fields.get("endian", "little").lower()
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    sizes = [int(s) for s in fields["sizes"].split()]
    for key in ("byte skip", "byteskip", "line skip", "lineskip"):
        if int(fields.get(key, "0")) != 0:
            raise ValueError(f"NRRD field {key!r} is not supported")

    data_file = fields.get("data file", fields.get("datafile"))
    if data_file is not None:
        detached = os.path.join(os.path.dirname(os.fspath(path)), data_file)
        with open(detached, "rb") as f:
            payload = f.read()
    else:
        payload = raw[data_offset:]
    encoding = fields.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        payload = gzip.decompress(payload)
    elif encoding != "raw":
        raise ValueError(f"unsupported NRRD encoding {encoding!r}")

    ni, nj, nk = sizes
    count = ni * nj * nk
    data = np.frombuffer(payload, dtype=dtype, count=count).reshape(nk, nj, ni)
    data = data.astype(dtype.newbyteorder("="), copy=True)

    space = fields.get("space", "left-posterior-superior").lower()
    if space not in _SPACE_TO_RAS:
        raise ValueError(f"unsupported NRRD space {space!r}")
    flip = _SPACE_TO_RAS[space]
    if "space directions" in fields:
        vectors = re.findall(r"\([^)]*\)", fields["space directions"])
        if len(vectors) != 3:
            raise ValueError("expected three space direction vectors")
        columns = np.stack([_parse_vector(v) for v in vectors], axis=1)
    else:
        spacings = [float(s) for s in fields.get("spacings", "1 1 1").split()]
        columns = np.diag(spacings)
    origin = _parse_vector(fields["space origin"]) if "space origin" in fields else np.zeros(3)

    m = np.eye(4)
    m[:3, :3] = columns * flip[:, np.newaxis]
    m[:3, 3] = origin * flip
    geometry = ImageGeometry.from_ijk_to_ras_matrix(m, (ni, nj, nk))
    return Volume(data=data, geometry=geometry)
