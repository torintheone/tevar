"""Automatic names of ostia by simple anatomical rules (SPEC section 6.4).

The user always confirms or renames them; confirmed ostia are never relabelled.

- Arch (the aorta runs roughly horizontally, the branch goes up): in order along the
  centerline BCT, LCCA, LSA; further ones "arch branch".
- Descending/abdominal aorta (running towards the feet), by the clock position with the
  anterior reference: anterior (10:30-1:30), left (1:30-5:00), posterior (5:00-7:00),
  right (7:00-10:30):
  - anterior ostia above the renal level: celiac trunk, then SMA; the first anterior ostium
    at least 20 mm below the renal arteries: IMA;
  - lateral ostia: the largest on each side near the SMA/renal level is the renal artery,
    the other lateral ostia nearby are accessory renal arteries;
  - posterior ostia: lumbar/intercostal arteries;
  - a large branch leaving at a small angle at the distal end: the contralateral common
    iliac artery; ostia beyond it (on the iliac artery): internal iliac artery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..centerline.navigation import CurvePath
from ..i18n import tr
from .clock import MODE_ANTERIOR
from .model import Ostium, OstiumPosition, describe

#: Ostium name keys (i18n ``ostium.<key>``), in the order offered to the user.
OSTIUM_KEYS = (
    "bca",
    "lcca",
    "lsa",
    "arch_branch",
    "celiac",
    "sma",
    "right_renal",
    "left_renal",
    "accessory_renal_right",
    "accessory_renal_left",
    "ima",
    "right_cia",
    "left_cia",
    "internal_iliac_right",
    "internal_iliac_left",
    "posterior",
    "branch",
)

ARCH_MAX_VERTICAL = 0.6  # |tangent . S| below this: the aorta runs roughly horizontally (arch)
DESCENDING_MIN = 0.5  # tangent . (-S) above this: descending/abdominal aorta
BRANCH_UP_MIN = 0.5  # branch direction . S above this: arch vessel


def ostium_name(key: Optional[str]) -> str:
    return tr(f"ostium.{key}") if key else tr("ostium.branch")


@dataclass
class _Item:
    ostium: Ostium
    position: OstiumPosition
    tangent: np.ndarray
    diameter: float


def _sector(clock_deg: float) -> str:
    a = clock_deg % 360.0
    if a >= 315.0 or a <= 45.0:
        return "anterior"
    if a <= 150.0:
        return "left"
    if a < 210.0:
        return "posterior"
    return "right"


def propose_labels(ostia: List[Ostium], path: CurvePath, aortic_diameter_mm: Optional[float] = None) -> Dict[int, str]:
    """Assign ``key``/``name`` to the unconfirmed ostia; returns {index: key} of the changes."""
    items = []
    for o in ostia:
        position = describe(o, path, MODE_ANTERIOR)
        diameter = o.branch_diameter_mm if o.branch_diameter_mm is not None else 0.0
        items.append(_Item(o, position, path.tangent_at(position.arc_length_mm), diameter))
    keys: Dict[int, str] = {}
    order = sorted(range(len(items)), key=lambda i: items[i].position.arc_length_mm)
    up = np.array([0.0, 0.0, 1.0])

    # Arch vessels
    arch = [i for i in order
            if abs(float(np.dot(items[i].tangent, up))) < ARCH_MAX_VERTICAL
            and items[i].ostium.direction is not None and float(np.dot(items[i].ostium.direction, up)) > BRANCH_UP_MIN]
    for n, i in enumerate(arch):
        keys[i] = ("bca", "lcca", "lsa")[n] if n < 3 else "arch_branch"

    # Descending and abdominal aorta
    descending = [i for i in order if i not in keys and -float(np.dot(items[i].tangent, up)) > DESCENDING_MIN]
    cia_index = None
    for i in reversed(descending):  # the contralateral common iliac: large, small angle, distal
        it = items[i]
        large = aortic_diameter_mm is not None and it.diameter >= 0.4 * aortic_diameter_mm
        if large and it.position.takeoff is not None and it.position.takeoff.craniocaudal_deg < 50.0:
            cia_index = i
            break
    if cia_index is not None:
        d = items[cia_index].ostium.direction
        keys[cia_index] = "right_cia" if d is not None and d[0] > 0 else "left_cia"  # RAS: +x = right
        s_cia = items[cia_index].position.arc_length_mm
        iliac_side = "left" if keys[cia_index] == "right_cia" else "right"
        beyond = [i for i in descending if i != cia_index and items[i].position.arc_length_mm > s_cia]
        for n, i in enumerate(beyond):
            keys[i] = f"internal_iliac_{iliac_side}" if n == 0 else "branch"
        descending = [i for i in descending if items[i].position.arc_length_mm < s_cia]

    sectors = {i: _sector(items[i].position.clock_deg) for i in descending}
    left = [i for i in descending if sectors[i] == "left"]
    right = [i for i in descending if sectors[i] == "right"]
    anterior = [i for i in descending if sectors[i] == "anterior"]
    for i in descending:
        if sectors[i] == "posterior":
            keys[i] = "posterior"

    renal_main = []
    for side, group in (("left", left), ("right", right)):
        if not group:
            continue
        main = max(group, key=lambda i: items[i].diameter)
        keys[main] = f"{side}_renal"
        renal_main.append(main)
        s_main = items[main].position.arc_length_mm
        for i in group:
            if i != main:
                near = abs(items[i].position.arc_length_mm - s_main) <= 50.0
                keys[i] = f"accessory_renal_{side}" if near else "branch"
    renal_s = min(items[i].position.arc_length_mm for i in renal_main) if renal_main else None
    lowest_renal = max(items[i].position.arc_length_mm for i in renal_main) if renal_main else None
    above = [i for i in anterior if renal_s is None or items[i].position.arc_length_mm < renal_s + 10.0]
    below = [i for i in anterior if i not in above]
    names_above = ("celiac", "sma") if renal_s is not None else ("celiac", "sma", "ima")
    for n, i in enumerate(above):
        keys[i] = names_above[n] if n < len(names_above) else "branch"
    first_ima = True
    for i in below:
        if first_ima and lowest_renal is not None and items[i].position.arc_length_mm >= lowest_renal + 20.0:
            keys[i] = "ima"
            first_ima = False
        else:
            keys[i] = "branch"

    changes = {}
    for i, it in enumerate(items):
        if it.ostium.confirmed:
            continue
        key = keys.get(i, "branch")
        if it.ostium.key != key or not it.ostium.name:
            it.ostium.key = key
            it.ostium.name = ostium_name(key)
            changes[i] = key
    return changes
