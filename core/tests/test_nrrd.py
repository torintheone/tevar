import gzip

import numpy as np
import pytest

from aortaplan_core.fileio.nrrd import read_nrrd, write_nrrd
from aortaplan_core.volume import CT_LIKE_DIRECTIONS, ImageGeometry, Volume


def make_volume(dtype=np.int16):
    g = ImageGeometry(dimensions=(5, 4, 3), spacing=(0.5, 0.6, 0.7), origin=(10.0, 20.0, 30.0), directions=CT_LIKE_DIRECTIONS)
    data = (np.arange(np.prod(g.array_shape)).reshape(g.array_shape) - 20).astype(dtype)
    return Volume(data=data, geometry=g)


def header_of(path):
    raw = open(path, "rb").read()
    return raw[: raw.index(b"\n\n")].decode("ascii")


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.float32, np.float64])
def test_round_trip(tmp_path, compress, dtype):
    vol = make_volume(dtype)
    path = tmp_path / "v.nrrd"
    write_nrrd(path, vol, compress=compress)
    back = read_nrrd(path)
    assert back.data.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(back.data, vol.data)
    assert back.geometry.is_close(vol.geometry)


def test_header_is_lps(tmp_path):
    path = tmp_path / "v.nrrd"
    write_nrrd(path, make_volume())
    header = header_of(path)
    assert "space: left-posterior-superior" in header
    assert "sizes: 5 4 3" in header
    # RAS origin (10, 20, 30) is LPS (-10, -20, 30); CT-like RAS directions are identity in LPS.
    assert "space origin: (-10.0,-20.0,30.0)" in header
    assert "space directions: (0.5,0.0,0.0) (0.0,0.6,0.0) (0.0,0.0,0.7)" in header


def _write_manual(path, header_lines, payload):
    with open(path, "wb") as f:
        f.write(("\n".join(header_lines) + "\n\n").encode("ascii"))
        f.write(payload)


def test_read_ras_space(tmp_path):
    data = np.arange(8, dtype="<i2").reshape(2, 2, 2)
    path = tmp_path / "ras.nrrd"
    _write_manual(
        path,
        [
            "NRRD0004",
            "type: short",
            "dimension: 3",
            "space: right-anterior-superior",
            "sizes: 2 2 2",
            "space directions: (1,0,0) (0,2,0) (0,0,3)",
            "endian: little",
            "encoding: raw",
            "space origin: (1,2,3)",
        ],
        data.tobytes(),
    )
    vol = read_nrrd(path)
    np.testing.assert_allclose(vol.geometry.origin, [1, 2, 3])
    np.testing.assert_allclose(vol.geometry.ijk_to_ras([1, 1, 1]), [2, 4, 6])
    np.testing.assert_array_equal(vol.data, data)


def test_read_detached_big_endian_gzip(tmp_path):
    data = np.arange(24, dtype=">f4").reshape(2, 3, 4)
    (tmp_path / "d.raw.gz").write_bytes(gzip.compress(data.tobytes()))
    path = tmp_path / "d.nhdr"
    _write_manual(
        path,
        [
            "NRRD0005",
            "# comment",
            "type: float",
            "dimension: 3",
            "space: left-posterior-superior",
            "sizes: 4 3 2",
            "space directions: (1,0,0) (0,1,0) (0,0,1)",
            "kinds: domain domain domain",
            "endian: big",
            "encoding: gzip",
            "space origin: (5,6,7)",
            "data file: d.raw.gz",
            "Segment0_Name:=ignored",
        ],
        b"",
    )
    vol = read_nrrd(path)
    np.testing.assert_array_equal(vol.data, data.astype(np.float32))
    np.testing.assert_allclose(vol.geometry.origin, [-5, -6, 7])
    np.testing.assert_allclose(vol.geometry.directions, CT_LIKE_DIRECTIONS)


def test_rejects_non_nrrd(tmp_path):
    path = tmp_path / "x.nrrd"
    path.write_bytes(b"hello")
    with pytest.raises(ValueError):
        read_nrrd(path)
