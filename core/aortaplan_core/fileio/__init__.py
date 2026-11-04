"""File input/output for the core (volumes, ground-truth files)."""

from .nrrd import read_nrrd, write_nrrd

__all__ = ["read_nrrd", "write_nrrd"]
