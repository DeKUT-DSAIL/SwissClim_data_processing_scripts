#!/usr/bin/env python3
"""
build_tahmo_dataset.py
----------------------
CLI entrypoint forwarding to scripts/TAHMO/build_tahmo_dataset.py.
"""
import sys
from pathlib import Path

tahmo_script_dir = Path(__file__).resolve().parent / "TAHMO"
sys.path.insert(0, str(tahmo_script_dir))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "build_tahmo_dataset", tahmo_script_dir / "build_tahmo_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main()

