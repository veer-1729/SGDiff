#!/usr/bin/env python3
"""
Convert SGDiff Visual Genome HDF5 splits into LAION-SG style JSON.

Each entry in the output JSON looks like:
{
  "name": "VG_100K/10.jpg",
  "img_id": "vg_10",
  "caption_ori": "...",
  "items": [
    {
      "item_id": 0,
      "label": "boy",
      "attributes": ["young"],
      "global_item_id": 12345
    },
    ...
  ],
  "relations": [
    {"item1": 0, "item2": 1, "relation": "flying"},
    ...
  ],
  "constraints": [
    {"type": "presence", "object": "boy", "polarity": "positive"},
    {"type": "attribute", "object": "boy", "attribute": "young"},
    ...
  ]
}

This allows training or fine-tuning the SDXL-SG pipeline with VG data.
Constraints are auto-generated from the scene graph to enable constraint-conditioned generation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add LAION-SG to path for constraint imports BEFORE importing heuristic module
LAION_SG_PATH = Path(__file__).parent.parent / "LAION-SG"
if LAION_SG_PATH.exists():
    sys.path.insert(0, str(LAION_SG_PATH))

from sgEncoderTraining.constraints.heuristic_constraints import (
    generate_constraints_from_caption_and_sg,
)

import h5py


def _decode_path(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8")
    return str(value)


def build_caption_map(captions_json: Path) -> Dict[str, str]:
    with captions_json.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "captions" in data:
        entries = data["captions"]
    else:
        entries = data
    return {entry["image_path"]: entry["caption"] for entry in entries}


def build_attribute_lookup(
    attributes_json: Path | None,
    valid_object_ids: set[int],
    attr_names_list: List[str],
) -> Dict[int, List[str]]:
    if not attributes_json or not attributes_json.exists():
        return {}
    attr_set = set(attr_names_list)
    if not attr_set:
        return {}
    lookup: Dict[int, List[str]] = {}
    with attributes_json.open("r") as f:
        entries = json.load(f)
    for entry in entries:
        for obj in entry.get("attributes", []):
            object_id = obj.get("object_id")
            if object_id is None or object_id not in valid_object_ids:
                continue
            names = obj.get("attributes", [])
            filtered = [name for name in names if name in attr_set]
            if filtered:
                lookup[object_id] = filtered
    return lookup


def convert_split(
    h5_path: Path,
    vocab_path: Path,
    captions_json: Path,
    attributes_json: Path | None,
    output_path: Path,
    max_samples: int | None = None,
    # Constraint generation settings (for heuristic_caption+SG constraints)
    generate_constraints: bool = True,
    max_presence: int = 2,
    max_attribute: int = 2,
    max_relation: int = 1,
    max_count: int = 1,
) -> None:
    with vocab_path.open("r") as f:
        vocab = json.load(f)
    caption_map = build_caption_map(captions_json)

    obj_names_list = vocab["object_idx_to_name"]
    pred_names_list = vocab["pred_idx_to_name"]
    attr_names_list = vocab.get("attribute_idx_to_name", [])

    output: List[dict] = []

    with h5py.File(h5_path, "r") as f:
        image_paths = f["image_paths"]
        image_ids = f["image_ids"][:]
        object_ids = f["object_ids"][:]
        object_names = f["object_names"][:]
        objects_per_img = f["objects_per_image"][:]
        rel_subjects = f["relationship_subjects"][:]
        rel_objects = f["relationship_objects"][:]
        rel_predicates = f["relationship_predicates"][:]
        relationships_per_img = f["relationships_per_image"][:]
        attributes_per_obj = f["attributes_per_object"][:]
        object_attributes = f.get("object_attributes")

        valid_object_ids = set(int(x) for x in object_ids.ravel() if x >= 0)
        attr_lookup = build_attribute_lookup(attributes_json, valid_object_ids, attr_names_list)

        total = len(image_ids)
        filtered_indices = []
        for idx in range(total):
            rel_path = _decode_path(image_paths[idx])
            if rel_path in caption_map:
                filtered_indices.append(idx)

        indices = filtered_indices
        if max_samples is not None and max_samples < len(indices):
            random.shuffle(indices)
            indices = indices[:max_samples]
        attr_row = 0

        for idx in indices:
            if max_samples is not None and len(output) >= max_samples:
                break
            num_obj = int(objects_per_img[idx])
            num_rel = int(relationships_per_img[idx])
            if num_obj <= 0 or num_rel <= 0:
                continue

            rel_path = _decode_path(image_paths[idx])
            caption = caption_map.get(rel_path, "")

            items = []
            slot_to_item = {}
            for slot in range(num_obj):
                name_idx = int(object_names[idx, slot])
                if name_idx < 0:
                    continue
                label = obj_names_list[name_idx]
                object_id = int(object_ids[idx, slot])
                attributes = attr_lookup.get(object_id, [])
                if not attributes and attr_names_list:
                    attr_count = int(attributes_per_obj[idx, slot]) if attributes_per_obj.ndim == 2 else 0
                    if attr_count > 0 and object_attributes is not None:
                        if object_attributes.ndim == 3:
                            attr_slice = object_attributes[idx, slot, :attr_count]
                        else:
                            attr_slice = object_attributes[attr_row, :attr_count]
                        tmp = []
                        for attr_idx in attr_slice:
                            attr_idx = int(attr_idx)
                            if 0 <= attr_idx < len(attr_names_list):
                                tmp.append(attr_names_list[attr_idx])
                        attributes = tmp
                if object_attributes is not None and object_attributes.ndim == 2:
                    attr_row += 1
                item = {
                    "item_id": len(items),
                    "label": label,
                    "attributes": attributes,
                    "global_item_id": object_id,
                }
                slot_to_item[slot] = item["item_id"]
                items.append(item)

            if not items:
                continue

            relations = []
            global_ids = []
            for rel_idx in range(num_rel):
                sub_slot = int(rel_subjects[idx, rel_idx])
                obj_slot = int(rel_objects[idx, rel_idx])
                pred_idx = int(rel_predicates[idx, rel_idx])
                if (
                    sub_slot not in slot_to_item
                    or obj_slot not in slot_to_item
                    or pred_idx < 0
                    or pred_idx >= len(pred_names_list)
                ):
                    continue
                relations.append(
                    {
                        "item1": slot_to_item[sub_slot],
                        "item2": slot_to_item[obj_slot],
                        "relation": pred_names_list[pred_idx],
                    }
                )
                global_ids.append(
                    {
                        "item1": items[slot_to_item[sub_slot]]["global_item_id"],
                        "item2": items[slot_to_item[obj_slot]]["global_item_id"],
                    }
                )

            if not relations:
                continue

            entry = {
                "name": rel_path,
                "img_id": f"vg_{int(image_ids[idx])}",
                "caption_ori": caption,
                "items": items,
                "relations": relations,
                "global_ids": global_ids,
            }

            # Generate caption+SG-based constraints if enabled and heuristic is available
            if generate_constraints:
                entry["constraints"] = generate_constraints_from_caption_and_sg(
                    caption=caption,
                    items=items,
                    relations=relations,
                    max_presence=max_presence,
                    max_attribute=max_attribute,
                    max_relation=max_relation,
                    max_count=max_count,
                )

            output.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(output)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert VG HDF5 splits to LAION-SG JSON with constraints.")
    parser.add_argument("--h5_path", type=Path, required=True, help="Path to VG split HDF5 file.")
    parser.add_argument("--vocab_json", type=Path, default=Path("datasets/vg/vocab.json"))
    parser.add_argument("--captions_json", type=Path, default=Path("datasets/vg/captions.json"))
    parser.add_argument("--attributes_json", type=Path, default=Path("datasets/vg/attributes.json"))
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit for debugging.")
    
    # Constraint generation arguments (for heuristic caption+SG constraints)
    parser.add_argument("--no_constraints", action="store_true",
                        help="Disable constraint generation")
    parser.add_argument("--max_presence", type=int, default=2,
                        help="Max presence constraints per image (default: 2)")
    parser.add_argument("--max_attribute", type=int, default=2,
                        help="Max attribute constraints per image (default: 2)")
    parser.add_argument("--max_relation", type=int, default=1,
                        help="Max relation constraints per image (default: 1)")
    parser.add_argument("--no_count", action="store_true",
                        help="Disable count constraints")
    
    args = parser.parse_args()

    convert_split(
        h5_path=args.h5_path,
        vocab_path=args.vocab_json,
        captions_json=args.captions_json,
        attributes_json=args.attributes_json,
        output_path=args.output_json,
        max_samples=args.max_samples,
        generate_constraints=not args.no_constraints,
        max_presence=args.max_presence,
        max_attribute=args.max_attribute,
        max_relation=args.max_relation,
        max_count=None if args.no_count else 1,
    )
    
    # Print constraint statistics if enabled
    if not args.no_constraints:
        with args.output_json.open("r") as f:
            data = json.load(f)
        
        total_constraints = sum(len(e.get("constraints", [])) for e in data)
        type_counts = {"presence": 0, "attribute": 0, "relation": 0, "count": 0}
        for e in data:
            for c in e.get("constraints", []):
                ctype = c.get("type", "unknown")
                if ctype in type_counts:
                    type_counts[ctype] += 1
        
        print(f"\n=== Constraint Statistics ===")
        print(f"Total constraints: {total_constraints}")
        print(f"Average per image: {total_constraints / len(data):.1f}" if data else "N/A")
        for ctype, count in type_counts.items():
            print(f"  {ctype}: {count}")


if __name__ == "__main__":
    main()

