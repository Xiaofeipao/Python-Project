"""Compatibility command; use scripts/reporting/decision_timeline.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / 'reporting/decision_timeline.py'),run_name="__main__")
