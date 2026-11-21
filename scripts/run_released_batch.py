"""Compatibility command; use scripts/workflows/run_released_batch.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / 'workflows/run_released_batch.py'),run_name="__main__")
