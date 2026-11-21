"""Compatibility command; use scripts/data/convert_numpy_task.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / 'data/convert_numpy_task.py'),run_name="__main__")
