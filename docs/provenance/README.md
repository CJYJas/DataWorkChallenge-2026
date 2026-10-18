# Repository layout notes

This repository now includes a `src/` package layout for engineering-friendly imports
while keeping the historical root-level scripts available for backwards compatibility.

Structure:
- `src/datawork/` — package entry points and compatibility wrappers
- `tests/` — focused Python tests
- `docs/provenance/` — provenance and packaging metadata (if not already colocated)
- `notebooks/` — notebook artifacts and launch files
