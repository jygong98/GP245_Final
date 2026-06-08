#!/usr/bin/env python3
"""Run the Antarctica_RF workflow."""

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
        case_name="AT_crust_ice",
        title_label="AT_crust_ice",
        vs_depth_max_km=12.0,
        force_regen=False,
        force_snapshot_regen=False,
    )
    run_example_workflow(EXAMPLE_DIR, cfg)
