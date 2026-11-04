"""Synthetic phantoms: NRRD volume + JSON ground truth (see SPEC section 9)."""

from .library import STANDARD_PHANTOMS, build_tube_phantom, elliptical_tube, helix, straight_tube, torus_arc
from .model import (
    Phantom,
    PhantomTruth,
    TubeLayer,
    ellipse_reference_metrics,
    load_phantom,
    phantom_paths,
    save_phantom,
)

__all__ = [
    "STANDARD_PHANTOMS",
    "Phantom",
    "PhantomTruth",
    "TubeLayer",
    "build_tube_phantom",
    "ellipse_reference_metrics",
    "elliptical_tube",
    "helix",
    "load_phantom",
    "phantom_paths",
    "save_phantom",
    "straight_tube",
    "torus_arc",
]
