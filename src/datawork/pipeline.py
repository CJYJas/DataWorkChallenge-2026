"""Compatibility wrapper for the package-oriented layout.

This keeps the old repository-root CLI working while exposing the project as a
Python package under src/datawork.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import *  # noqa: F401,F403

if __name__ == "__main__":
    from pipeline import main
    main()
