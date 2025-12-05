#!/usr/bin/env python3
"""
Add auto-generated constraints to existing LAION-style JSON files.

This script reads a LAION-SG JSON file, generates constraints for each entry
based on its scene graph content, and outputs a new JSON with constraints added.

Usage:
    python add_constraints_to_json.py \
        --input_json vg_train.json \
        --output_json vg_train_with_constraints.json \
        --max_presence 2 \
        --max_attribute 2 \
        --max_relation 1 \
        --seed 42
"""

import argparse
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgEncoderTraining.constraints import generate_constraints_from_scene_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Add constraints to LAION-SG JSON")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to input LAION-style JSON file")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to output JSON file with constraints")
    parser.add_argument("--max_presence", type=int, default=2,
                        help="Max presence constraints per image (default: 2)")
    parser.add_argument("--max_attribute", type=int, default=2,
                        help="Max attribute constraints per image (default: 2)")
    parser.add_argument("--max_relation", type=int, default=1,
                        help="Max relation constraints per image (default: 1)")
    parser.add_argument("--include_count", action="store_true", default=True,
                        help="Include count constraints (default: True)")
    parser.add_argument("--no_count", action="store_true",
                        help="Disable count constraints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def add_constraints_to_entry(
    entry: dict,
    max_presence: int,
    max_attribute: int,
    max_relation: int,
    include_count: bool,
    seed: int
) -> dict:
    """Add constraints to a single JSON entry."""
    items = entry.get("items", [])
    relations = entry.get("relations", [])
    
    # Use entry-specific seed for reproducibility
    entry_seed = seed + hash(entry.get("img_id", "")) % 10000
    
    constraints = generate_constraints_from_scene_graph(
        items=items,
        relations=relations,
        max_presence=max_presence,
        max_attribute=max_attribute,
        max_relation=max_relation,
        include_count=include_count,
        seed=entry_seed
    )
    
    entry["constraints"] = constraints
    return entry


def main():
    args = parse_args()
    
    include_count = args.include_count and not args.no_count
    
    print(f"Loading {args.input_json}...")
    with open(args.input_json, "r") as f:
        data = json.load(f)
    
    print(f"Found {len(data)} entries")
    print(f"Generating constraints (presence={args.max_presence}, "
          f"attribute={args.max_attribute}, relation={args.max_relation}, "
          f"count={include_count})...")
    
    # Statistics
    total_constraints = 0
    type_counts = {"presence": 0, "attribute": 0, "relation": 0, "count": 0}
    
    for i, entry in enumerate(data):
        entry = add_constraints_to_entry(
            entry,
            max_presence=args.max_presence,
            max_attribute=args.max_attribute,
            max_relation=args.max_relation,
            include_count=include_count,
            seed=args.seed
        )
        
        # Track statistics
        for c in entry.get("constraints", []):
            total_constraints += 1
            ctype = c.get("type", "unknown")
            if ctype in type_counts:
                type_counts[ctype] += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} entries...")
    
    # Save output
    print(f"\nSaving to {args.output_json}...")
    with open(args.output_json, "w") as f:
        json.dump(data, f, indent=2)
    
    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Total entries: {len(data)}")
    print(f"Total constraints: {total_constraints}")
    print(f"Average constraints per image: {total_constraints / len(data):.1f}")
    print(f"\nConstraint type breakdown:")
    for ctype, count in type_counts.items():
        print(f"  {ctype}: {count}")
    
    # Show sample entry
    print(f"\n=== Sample Entry ===")
    sample = data[0] if data else {}
    print(f"Image: {sample.get('name', 'N/A')}")
    print(f"Caption: {sample.get('caption_ori', 'N/A')}")
    print(f"Constraints:")
    for c in sample.get("constraints", []):
        print(f"  - {c}")


if __name__ == "__main__":
    main()

