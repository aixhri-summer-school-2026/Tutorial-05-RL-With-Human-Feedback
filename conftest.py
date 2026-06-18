"""Ensure the repo root is importable so `pytest` finds the `mile_franka` package even
when it is not pip-installed in the active environment (e.g. running the unit tests on the
host rather than inside the container). `python -m pytest` adds cwd automatically; the bare
`pytest` console script does not, so make it explicit here."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
