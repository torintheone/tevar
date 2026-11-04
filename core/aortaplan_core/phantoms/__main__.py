"""Generate the standard phantom set.

Usage::

    python -m aortaplan_core.phantoms [--out DIR] [--only NAME ...] [--list]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .library import STANDARD_PHANTOMS
from .model import save_phantom

_DEFAULT_OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "phantoms"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aortaplan_core.phantoms", description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=_DEFAULT_OUT, help=f"output directory (default: {_DEFAULT_OUT})")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="generate only these phantoms")
    parser.add_argument("--list", action="store_true", help="list phantom names and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(STANDARD_PHANTOMS))
        return 0
    names = args.only or list(STANDARD_PHANTOMS)
    unknown = [n for n in names if n not in STANDARD_PHANTOMS]
    if unknown:
        parser.error(f"unknown phantom(s): {', '.join(unknown)}")
    for name in names:
        start = time.perf_counter()
        phantom = STANDARD_PHANTOMS[name]()
        paths = save_phantom(phantom, args.out)
        dims = "x".join(str(n) for n in phantom.volume.geometry.dimensions)
        print(f"{name}: {dims} voxels, {time.perf_counter() - start:.1f} s -> {paths['volume']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
