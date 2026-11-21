"""Compatibility command; use scripts/validation/check_frozen_workflow.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / 'validation/check_frozen_workflow.py'),run_name="__main__")
