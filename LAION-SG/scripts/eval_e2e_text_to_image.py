#!/usr/bin/env python3
"""
End-to-end evaluation: Text (caption) -> Scene Graph (LLM) -> Image (SDXL-SG) -> Metrics (FID/IS).

This script evaluates the full pipeline, not just SG->Image.

Per run:
  1) Build a "real subset" directory by copying the real images referenced by input_json.
  2) Run caption_to_sg_llm.py to predict scene graphs from caption_ori (optionally cached).
  3) For each checkpoint, run test_laion2.py on the predicted SG JSON to generate images.
  4) Compute FID vs real subset using pytorch-fid, and optionally Inception Score.

Notes:
  - FID requires two directories of images (real vs generated). We create a real subset dir for
    the exact set of samples being evaluated.
  - IS is optional and requires pytorch-gan-metrics.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

try:
    from pytorch_gan_metrics import get_inception_score
    HAS_PYTORCH_GAN_METRICS = True
except ImportError:
    HAS_PYTORCH_GAN_METRICS = False


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_real_subset_dir(
    *,
    input_json: Path,
    image_dir: Path,
    out_real_dir: Path,
    max_samples: int | None,
) -> list[dict]:
    """
    Copy real images referenced by input_json into out_real_dir.
    Returns the subset list of entries used (preserving order).
    """
    data = json.loads(input_json.read_text())
    if max_samples is not None:
        data = data[:max_samples]

    out_real_dir.mkdir(parents=True, exist_ok=True)

    used = []
    for idx, entry in enumerate(data):
        name = entry.get("name")
        img_id = entry.get("img_id") or f"sample_{idx}"
        if not name:
            continue
        src = image_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing real image for {img_id}: {src}")

        # Save as {img_id}.jpg to match the generated naming convention in test_laion2.py
        dst = out_real_dir / f"{img_id}.jpg"
        _safe_copy(src, dst)
        used.append(entry)

    return used


def write_subset_json(entries: list[dict], out_path: Path) -> None:
    out_path.write_text(json.dumps(entries, indent=2))


def run_caption_to_sg_llm(
    *,
    examples_json: Path,
    input_json: Path,
    output_json: Path,
    num_shots: int,
    max_samples: int | None,
    model: str,
    temperature: float,
) -> None:
    cmd = [
        "python",
        "scripts/caption_to_sg_llm.py",
        "--examples_json",
        str(examples_json),
        "--input_json",
        str(input_json),
        "--output_json",
        str(output_json),
        "--num_shots",
        str(num_shots),
        "--model",
        model,
        "--temperature",
        str(temperature),
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]

    print("\n=== Running caption->scenegraph (LLM) ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_sg_to_image(
    *,
    checkpoint_path: Path,
    test_json_path: Path,
    image_dir: Path,
    output_dir: Path,
    num_samples: int,
    stable_diffusion_checkpoint: str,
    cache_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "test_laion2.py",
        "--pretrained_sgencoder_path",
        str(checkpoint_path),
        "--test_json_path",
        str(test_json_path),
        "--image_dir",
        str(image_dir),
        "--output_dir",
        str(output_dir),
        "--num_samples",
        str(num_samples),
        "--stable_diffusion_checkpoint",
        stable_diffusion_checkpoint,
        "--cache_dir",
        str(cache_dir),
    ]
    print(f"\n=== Generating images for checkpoint: {checkpoint_path} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compute_fid(real_dir: Path, gen_dir: Path, batch_size: int = 50, device: str | None = None, num_workers: int = 0) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = calculate_fid_given_paths(
        [str(real_dir), str(gen_dir)],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=num_workers,
    )
    return float(fid)


def compute_inception_score(gen_dir: Path, batch_size: int = 32, splits: int = 10, device: str | None = None):
    if not HAS_PYTORCH_GAN_METRICS:
        raise RuntimeError(
            "pytorch-gan-metrics is not installed. Install with `pip install pytorch-gan-metrics`."
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_mean, is_std = get_inception_score(
        str(gen_dir),
        batch_size=batch_size,
        splits=splits,
        device=device,
    )
    return float(is_mean), float(is_std)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end eval: text->sg->image->FID/IS")

    p.add_argument("--input_json", type=Path, required=True, help="JSON with {name,img_id,caption_ori} entries.")
    p.add_argument("--examples_json", type=Path, required=True, help="Few-shot examples JSON for caption_to_sg_llm.py.")
    p.add_argument("--image_dir", type=Path, required=True, help="Root dir where input_json['name'] images live.")

    p.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="One or more sgEncoder checkpoints.")

    p.add_argument("--output_root", type=Path, default=Path("e2e_eval_outputs"), help="Root output directory.")
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of samples evaluated.")
    p.add_argument("--num_samples", type=int, default=None, help="Alias for max_samples (kept for convenience).")

    # LLM config
    p.add_argument("--num_shots", type=int, default=8, help="Number of few-shot examples in the LLM prompt.")
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model name (OpenAI).")
    p.add_argument("--llm_temperature", type=float, default=0.2, help="LLM temperature.")
    p.add_argument("--reuse_sg_json", action="store_true", help="If SG JSON already exists, do not call the LLM again.")

    # SDXL config
    p.add_argument("--stable_diffusion_checkpoint", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--cache_dir", type=Path, default=Path("/tmp/.cache/huggingface"))
    p.add_argument("--image_size", type=int, default=1024, help="Output resolution (passed to test_laion2 via JSON loader defaults).")

    # Metrics
    p.add_argument("--compute_is", action="store_true", help="Also compute Inception Score (requires pytorch-gan-metrics).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Support --num_samples as alias for --max_samples
    max_samples = args.max_samples if args.max_samples is not None else args.num_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Build real subset directory + subset JSON
    real_dir = out_root / "real"
    subset_entries = build_real_subset_dir(
        input_json=args.input_json,
        image_dir=args.image_dir,
        out_real_dir=real_dir,
        max_samples=max_samples,
    )
    subset_json = out_root / "subset_input.json"
    write_subset_json(subset_entries, subset_json)

    # 2) Predict SG from captions via LLM
    sg_json = out_root / "pred_sg.json"
    if args.reuse_sg_json and sg_json.exists():
        print(f"\nReusing existing SG JSON: {sg_json}")
    else:
        run_caption_to_sg_llm(
            examples_json=args.examples_json,
            input_json=subset_json,
            output_json=sg_json,
            num_shots=args.num_shots,
            max_samples=max_samples,
            model=args.llm_model,
            temperature=args.llm_temperature,
        )

    # 3) Generate images + compute metrics per checkpoint
    results = []
    for ckpt in args.checkpoints:
        ckpt = ckpt.resolve()
        ckpt_name = ckpt.stem
        gen_dir = out_root / "generated" / ckpt_name

        run_sg_to_image(
            checkpoint_path=ckpt,
            test_json_path=sg_json,
            image_dir=args.image_dir,
            output_dir=gen_dir,
            num_samples=len(subset_entries),
            stable_diffusion_checkpoint=args.stable_diffusion_checkpoint,
            cache_dir=args.cache_dir,
        )

        fid = compute_fid(real_dir=real_dir, gen_dir=gen_dir, device=device)

        is_mean = is_std = None
        if args.compute_is:
            if not HAS_PYTORCH_GAN_METRICS:
                print("WARNING: pytorch-gan-metrics not installed; skipping Inception Score.")
            else:
                is_mean, is_std = compute_inception_score(gen_dir=gen_dir, device=device)

        results.append(
            {
                "checkpoint": str(ckpt),
                "generated_dir": str(gen_dir),
                "real_dir": str(real_dir),
                "fid": fid,
                "is_mean": is_mean,
                "is_std": is_std,
            }
        )

        print(f"\n=== Metrics for {ckpt_name} ===")
        print(f"FID: {fid:.4f}")
        if is_mean is not None:
            print(f"IS:  {is_mean:.4f} ± {is_std:.4f}")

    # 4) Save results
    results_path = out_root / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    for r in results:
        line = f"{Path(r['checkpoint']).name}: FID={r['fid']:.4f}"
        if r["is_mean"] is not None:
            line += f", IS={r['is_mean']:.4f} ± {r['is_std']:.4f}"
        print(line)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()


