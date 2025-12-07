#!/usr/bin/env python3
"""
Utility for sampling a small VG subset, cleaning captions, converting to LAION,
and kicking off the downstream pipeline.

Usage (example):

    python scripts/run_subset_pipeline.py \
        --dataset_dir datasets/vg \
        --captions_json datasets/vg/captions.json \
        --train_size 2000 \
        --val_size 500 \
        --seed 42 \
        --work_dir LAION-SG \
        --clean_model gpt-4o-mini \
        --clean_temp 0.2 \
        --convert_output_dir LAION-SG \
        --train_h5 train.h5 \
        --val_h5 val.h5 \
        --log_dir logs_vg_subset
"""

import argparse
import json
import random
import subprocess
from pathlib import Path


def sample_entries(entries, count, seed):
    rng = random.Random(seed)
    if count >= len(entries):
        return entries[:]
    return rng.sample(entries, count)


def write_subset(captions, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps({"captions": captions}, indent=2))


def run_clean(input_json, output_json, dataset_dir, vocab_json, model, temp):
    script_dir = Path(__file__).resolve().parent
    clean_script = script_dir / "clean_captions_with_sg.py"
    if not clean_script.exists():
        clean_script = script_dir.parent / "LAION-SG" / "scripts" / "clean_captions_with_sg.py"
    subprocess.run([
        "python", str(clean_script),
        "--dataset_dir", dataset_dir,
        "--vocab_json", vocab_json,
        "--captions_in", input_json,
        "--captions_out", output_json,
        "--model", model,
        "--temperature", str(temp),
    ], check=True)


def run_convert(h5_path, vocab_json, captions_json, output_json, max_samples=None):
    cmd = [
        "python", "scripts/convert_vg_to_laion.py",
        "--h5_path", h5_path,
        "--vocab_json", vocab_json,
        "--captions_json", captions_json,
        "--output_json", output_json,
    ]
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    subprocess.run(cmd, check=True)

    # print basic constraint stats
    with open(output_json, "r") as f:
        data = json.load(f)
    total = len(data)
    if total == 0:
        print(f"Converted {output_json}: 0 samples")
        return
    total_constraints = sum(len(entry.get("constraints", [])) for entry in data)
    avg_constraints = total_constraints / total if total else 0
    print(f"Converted {output_json}: {total} samples, {total_constraints} total constraints (avg {avg_constraints:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Sample VG subset and continue pipeline.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--captions_json", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", type=Path, default=Path("LAION-SG"))
    parser.add_argument("--clean_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--clean_temp", type=float, default=0.2)
    parser.add_argument("--vocab_json", type=Path, default=Path("datasets/vg/vocab.json"))
    parser.add_argument("--train_h5", type=str, default="train.h5")
    parser.add_argument("--val_h5", type=str, default="val.h5")
    parser.add_argument("--log_dir", type=str, default="logs_vg_subset")
    args = parser.parse_args()

    captions = json.loads(args.captions_json.read_text())["captions"]
    train_caps = sample_entries(captions, args.train_size, args.seed)
    val_caps = sample_entries([c for c in captions if c not in train_caps], args.val_size, args.seed + 1)

    train_raw = args.dataset_dir / "captions_train_subset.json"
    val_raw = args.dataset_dir / "captions_val_subset.json"
    write_subset(train_caps, train_raw)
    write_subset(val_caps, val_raw)

    train_clean = args.dataset_dir / "captions_train_subset_clean.json"
    val_clean = args.dataset_dir / "captions_val_subset_clean.json"

    run_clean(str(train_raw), str(train_clean), str(args.dataset_dir), str(args.vocab_json),
              args.clean_model, args.clean_temp)
    run_clean(str(val_raw), str(val_clean), str(args.dataset_dir), str(args.vocab_json),
              args.clean_model, args.clean_temp)

    train_laion = args.work_dir / "vg_train_subset.json"
    val_laion = args.work_dir / "vg_val_subset.json"

    run_convert(str(args.dataset_dir / args.train_h5), str(args.vocab_json), str(train_clean), str(train_laion))
    run_convert(str(args.dataset_dir / args.val_h5), str(args.vocab_json), str(val_clean), str(val_laion))

    print("Subsets created:")
    print("  train Laion:", train_laion)
    print("  val Laion:", val_laion)
    print(f"Next: run `trainer_laion.py --train_json_path {train_laion} --val_json_path {val_laion} ...`")


if __name__ == "__main__":
    main()

