"""AortaPlan core: Slicer-independent math for TEVAR/EVAR/FEVAR planning.

Rules (see CLAUDE.md):
- no ``import slicer`` / ``qt`` / ``ctk`` anywhere in this package;
- all geometry is in patient RAS coordinates, millimetres;
- volume arrays are indexed ``[k, j, i]`` (same as ``slicer.util.arrayFromVolume``).

For research use only. Not for clinical decision making.
"""

__version__ = "0.1.0"  # stage 1

__all__ = ["__version__"]
