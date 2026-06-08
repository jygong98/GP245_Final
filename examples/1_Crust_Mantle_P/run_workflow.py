#!/usr/bin/env python3
"""Run the Crust_Mantle_P workflow."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "_common"))

from rf_example_workflow import ExampleConfig, run_example_workflow

if __name__ == "__main__":
    cfg = ExampleConfig(
        case_name="test1_P_out",
        title_label="test1 P (crust + mantle)",
        vs_depth_max_km=50.0,
        bwf_pad_factor=2.0,  # anti-wraparound: synthesise on ~2x time axis
        force_regen=True,
        force_snapshot_regen=True,
    )
    run_example_workflow(EXAMPLE_DIR, cfg)
