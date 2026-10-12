#!/usr/bin/env python3
"""
Test/verify wrapper for TAHMO dataset.
Imports and runs verification from parent directory verify_tahmo_dataset.py.
"""
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from verify_tahmo_dataset import verify_dataset, main

if __name__ == "__main__":
    main()

