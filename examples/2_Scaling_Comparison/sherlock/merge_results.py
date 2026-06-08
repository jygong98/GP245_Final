#!/usr/bin/env python3
"""Merge per-job scaling JSON files into python_scaling_times.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_results(*, results_dir: Path, output_path: Path) -> dict[str, Any]:
    """Combine sherlock/results/*.json into one cache file."""
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        raise SystemExit(f"No result files found in {results_dir}")

    runs_by_label: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}

    for path in result_files:
        payload = _load_json(path)
        for run in payload.get("runs", []):
            label = run.get("label")
            if not label:
                raise SystemExit(f"{path} contains a run without label")
            runs_by_label[label] = run
        metadata.update(payload.get("metadata", {}))

    merged = {
        "runs": sorted(runs_by_label.values(), key=lambda r: r["label"]),
        "metadata": {
            **metadata,
            "merged_from": [p.name for p in result_files],
            "merged_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Sherlock per-job scaling results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory with per-job JSON files (default: ./results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "python_scaling_times.json",
        help="Merged cache path (default: ../python_scaling_times.json)",
    )
    args = parser.parse_args()

    merged = merge_results(results_dir=args.results_dir, output_path=args.output)
    print(f"Merged {len(merged['runs'])} runs -> {args.output}")
    for run in merged["runs"]:
        print(f"  {run['label']}: {run['time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
