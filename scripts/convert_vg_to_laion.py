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
  ]
}

This allows training or fine-tuning the SDXL-SG pipeline with VG data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def convert_split(
    h5_path: Path,
    vocab_path: Path,
    captions_json: Path,
    output_path: Path,
    max_samples: int | None = None,
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
        image_ids = f["image_ids"]
        object_ids = f["object_ids"]
        object_names = f["object_names"]
        objects_per_img = f["objects_per_image"]
        rel_subjects = f["relationship_subjects"]
        rel_objects = f["relationship_objects"]
        rel_predicates = f["relationship_predicates"]
        relationships_per_img = f["relationships_per_image"]
        attributes_per_obj = f["attributes_per_object"]
        object_attributes = f["object_attributes"]

        total = len(image_ids)
        for idx in range(total):
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
                attr_count = int(attributes_per_obj[idx, slot]) if attributes_per_obj.ndim == 2 else 0
                attributes = []
                if attr_count > 0 and attr_names_list:
                    attr_indices = object_attributes[idx, slot, :attr_count]
                    for attr_idx in attr_indices:
                        attr_idx = int(attr_idx)
                        if attr_idx >= 0 and attr_idx < len(attr_names_list):
                            attributes.append(attr_names_list[attr_idx])
                item = {
                    "item_id": len(items),
                    "label": label,
                    "attributes": attributes,
                    "global_item_id": int(object_ids[idx, slot]),
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
            output.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(output)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert VG HDF5 splits to LAION-SG JSON.")
    parser.add_argument("--h5_path", type=Path, required=True, help="Path to VG split HDF5 file.")
    parser.add_argument("--vocab_json", type=Path, default=Path("datasets/vg/vocab.json"))
    parser.add_argument("--captions_json", type=Path, default=Path("datasets/vg/captions.json"))
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit for debugging.")
    args = parser.parse_args()

    convert_split(
        h5_path=args.h5_path,
        vocab_path=args.vocab_json,
        captions_json=args.captions_json,
        output_path=args.output_json,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

