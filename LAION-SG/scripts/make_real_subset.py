import json
import os
import shutil
from pathlib import Path


JSON_PATH = "vg_val_500_subset.json" 
VG_ROOT = "../datasets/vg/images" 
OUT_DIR = "../datasets/vg/images/val_real_subset"

# output directory
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries from {JSON_PATH}")

num_copied = 0

for item in data:
    rel_path = item["name"]

    if rel_path is None:
        continue

    src = Path(VG_ROOT) / rel_path
    dst = Path(OUT_DIR) / Path(rel_path).name

    if src.exists():
        shutil.copy(src, dst)
        num_copied += 1
    else:
        print(f"Missing image: {src}")

print(f"\nDone! Copied {num_copied} images into {OUT_DIR}")
