#!/usr/bin/env python3
"""Build model files for AT_crust_ice."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MY_TESTS_DIR = THIS_DIR.parent.parent
if str(MY_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(MY_TESTS_DIR))

from at_variant_builder import CRUST_VP, CRUST_VS, LayerProps, VariantSpec, build_variant_case


def main() -> None:
    spec = VariantSpec(
        case_name="AT_crust_ice",
        double_ice=False,
        basement=LayerProps(vp=CRUST_VP, vs=CRUST_VS),
    )
    build_variant_case(output_dir=THIS_DIR, spec=spec)


if __name__ == "__main__":
    main()
