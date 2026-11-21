"""Compatibility command; use scripts/api/probe_hkust_api.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / 'api/probe_hkust_api.py'),run_name="__main__")
