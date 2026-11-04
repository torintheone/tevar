"""Make the Slicer-independent core library (``aortaplan_core``) importable inside Slicer.

Search order:
1. the ``AORTAPLAN_CORE_PATH`` environment variable (folder that contains ``aortaplan_core``);
2. ``<repository>/core`` when the module is used from a development checkout;
3. ``<module folder>/core`` (reserved for a packaged extension).
"""

import os
import sys

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../slicer_extension/AortaPlan


def candidateCorePaths():
    paths = []
    env = os.environ.get("AORTAPLAN_CORE_PATH")
    if env:
        paths.append(env)
    paths.append(os.path.normpath(os.path.join(_MODULE_DIR, "..", "..", "core")))
    paths.append(os.path.join(_MODULE_DIR, "core"))
    return paths


def ensureCoreOnPath():
    """Add the first folder containing ``aortaplan_core`` to ``sys.path``; return it (or None)."""
    for path in candidateCorePaths():
        if os.path.isfile(os.path.join(path, "aortaplan_core", "__init__.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    return None
