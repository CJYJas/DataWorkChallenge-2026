"""Compatibility wrapper for the package-oriented layout."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from summarize_results import *  # noqa: F401,F403

if __name__ == "__main__":
    from summarize_results import summarize
    summarize()
