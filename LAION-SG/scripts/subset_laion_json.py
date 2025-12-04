#!/usr/bin/env python3
"""
Create a smaller LAION-style JSON split by sampling a subset of entries.

Example:
    python scripts/subset_laion_json.py \
        --input_json LAION-SG/vg_train.json \
        --output_json LAION-SG/vg_train_1k.json \
        --num_samples 1000 \
        --seed 123
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a subset of LAION-style JSON entries.")
    parser.add_argument("--input_json", type=Path, required=True, help="Source JSON (LAION/SDXL-SG format).")
    parser.add_argument("--output_json", type=Path, required=True, help="Destination JSON for the subset.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of entries to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--ensure_images_exist",
        action="store_true",
        help="If set, only keep entries whose image file exists relative to --image_root.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=Path("."),
        help="Root directory containing image files referenced in JSON (used with --ensure_images_exist).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input_json.open("r") as f:
        data = json.load(f)

    random.seed(args.seed)
    candidates = data

    if args.ensure_images_exist:
        filtered = []
        for entry in data:
            img_path = args.image_root / entry["name"]
            if img_path.exists():
                filtered.append(entry)
        candidates = filtered
        if len(filtered) < args.num_samples:
            print(f"[subset] Warning: only {len(filtered)} entries have existing images; sampling from them.")

    subset = random.sample(candidates, min(args.num_samples, len(candidates)))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(subset, f, indent=2)

    print(f"[subset] Wrote {len(subset)} entries to {args.output_json}")


if __name__ == "__main__":
    main()


