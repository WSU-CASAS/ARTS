"""Make `config` (repo root) and sibling `src` modules importable regardless of
how a script is launched. Import this first in every entry-point module."""
import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
