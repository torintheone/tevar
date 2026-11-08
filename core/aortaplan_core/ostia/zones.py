"""Ishimaru landing zones of the aortic arch for TEVAR (SPEC section 6.3).

Boundaries along the centerline, from the found arch vessels (distal edges of the ostia,
``model.distal_edge_arc_length``):

- zone 0: ascending aorta up to the distal edge of the BCT ostium;
- zone 1: up to the distal edge of the LCCA ostium;
- zone 2: up to the distal edge of the LSA ostium;
- zone 3: the first 20 mm distal to the LSA;
- zone 4: the proximal half of the descending thoracic aorta: up to halfway between the
  end of zone 3 and the celiac trunk; without a celiac ostium 100 mm (marked approximate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..centerline.navigation import CurvePath
from ..crosssection import LUMEN, OUTER, diameter_profile
from ..volume import Volume
from .model import Ostium, describe, distal_edge_arc_length

ZONE3_LENGTH_MM = 20.0
ZONE4_DEFAULT_LENGTH_MM = 100.0


@dataclass
class Zone:
    name: str  # "0" .. "4"
    start_mm: float
    end_mm: float
    approximate: bool = False
    diameters: Dict[str, Optional[float]] = field(default_factory=dict)

    @property
    def length_mm(self) -> float:
        return self.end_mm - self.start_mm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone": self.name,
            "start_mm": round(self.start_mm, 2),
            "end_mm": round(self.end_mm, 2),
            "length_mm": round(self.length_mm, 2),
            "approximate": self.approximate,
            "diameters": {k: None if v is None else round(v, 2) for k, v in self.diameters.items()},
        }


def ishimaru_zones(path: CurvePath, ostia: List[Ostium]) -> List[Zone]:
    """Zones 0-4 on ``path``; empty without BCT, LCCA and LSA ostia (by ``key``)."""
    by_key = {o.key: o for o in ostia if o.key}
    if not all(k in by_key for k in ("bca", "lcca", "lsa")):
        return []
    edges = []
    for key in ("bca", "lcca", "lsa"):
        o = by_key[key]
        edges.append(distal_edge_arc_length(o, describe(o, path)))
    if not (edges[0] < edges[1] < edges[2]):
        return []
    z3_end = path.clamp(edges[2] + ZONE3_LENGTH_MM)
    approximate = "celiac" not in by_key
    if approximate:
        z4_end = path.clamp(z3_end + ZONE4_DEFAULT_LENGTH_MM)
    else:
        celiac_s = describe(by_key["celiac"], path).arc_length_mm
        z4_end = path.clamp(z3_end + 0.5 * max(celiac_s - z3_end, 0.0))
    bounds = [0.0, edges[0], edges[1], edges[2], z3_end, z4_end]
    return [Zone(str(i), bounds[i], bounds[i + 1], approximate=(i == 4 and approximate)) for i in range(5)]


def zone_diameters(volume: Volume, path: CurvePath, zones: List[Zone], step_mm: float = 2.0, with_outer: bool = True) -> None:
    """Fill each zone with the min/max lumen diameter and the max outer diameter (mm)."""
    for zone in zones:
        if zone.length_mm <= 0:
            continue
        profile = diameter_profile(volume, path, step_mm, zone.start_mm, zone.end_mm, with_outer=with_outer)
        lumen_min = profile.values(LUMEN, "min_diameter_mm")
        lumen_max = profile.values(LUMEN, "max_diameter_mm")
        outer_max = profile.values(OUTER, "max_diameter_mm")

        def pick(values, fn):
            v = values[np.isfinite(values)]
            return float(fn(v)) if v.size else None

        zone.diameters = {
            "lumen_min_mm": pick(lumen_min, np.min),
            "lumen_max_mm": pick(lumen_max, np.max),
            "outer_max_mm": pick(outer_max, np.max) if with_outer else None,
        }
