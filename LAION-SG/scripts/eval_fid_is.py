#!/usr/bin/env python3
"""
Evaluate FID and Inception Score for multiple sgEncoder checkpoints.

Workflow per checkpoint:
 1) Call test_laion2.py to generate a fixed number of images.
 2) Compute FID vs a directory of real images using pytorch-fid.
 3) Optionally compute Inception Score using pytorch-gan-metrics.
"""

import argparse
import os
from pathlib import Path
import subprocess
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
import wandb
from pytorch_gan_metrics import get_inception_score


def run_generation_for_checkpoint(
    checkpoint_path: str,
    test_json_path: str,
    image_dir: str,
    output_dir: str,
    num_samples: int,
    stable_diffusion_checkpoint: str,
    cache_dir: str,
    use_lora: bool = False,
    lora_weights_path: str = "",
) -> None:
    """
    Call test_laion2.py to generate images for a given sgEncoder checkpoint.
    """
    ckpt = os.path.abspath(checkpoint_path)
    out = os.path.abspath(output_dir)
    json_path = os.path.abspath(test_json_path)
    img_dir = os.path.abspath(image_dir)
    cache_dir = os.path.abspath(cache_dir)

    os.makedirs(out, exist_ok=True)

    cmd = [
        "python",
        "test_laion2.py",
        "--pretrained_sgencoder_path",
        ckpt,
        "--test_json_path",
        json_path,
        "--image_dir",
        img_dir,
        "--output_dir",
        out,
        "--num_samples",
        str(num_samples),
        "--stable_diffusion_checkpoint",
        stable_diffusion_checkpoint,
        "--cache_dir",
        cache_dir,
    ]

    if use_lora:
        cmd += ["--use_lora", "--lora_weights_path", lora_weights_path]

    print(f"\n=== Generating images for checkpoint: {checkpoint_path} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compute_fid(real_dir: str, gen_dir: str, batch_size: int = 50, device: str | None = None, num_workers: int = 0) -> float:
    """
    Compute FID between real_dir and gen_dir using pytorch-fid.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = [os.path.abspath(real_dir), os.path.abspath(gen_dir)]
    fid = calculate_fid_given_paths(
        paths,
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=num_workers,
    )
    return float(fid)


def compute_inception_score(
    gen_dir: str,
    batch_size: int = 32,
    splits: int = 10,
    device: str | None = None,
) -> tuple[float, float]:
    """
    Compute Inception Score for gen_dir using pytorch-gan-metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_mean, is_std = get_inception_score(
        os.path.abspath(gen_dir),
        batch_size=batch_size,
        splits=splits,
        device=device,
    )
    return float(is_mean), float(is_std)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FID and IS for multiple sgEncoder checkpoints.")
    p.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Directory with real images (e.g., VG val subset).",
    )
    p.add_argument(
        "--test_json_path",
        type=str,
        required=True,
        help="LAION-SG JSON used by test_laion2.py (same for all checkpoints).",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory for dataset images referenced in JSON.",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of sgEncoder checkpoint paths to evaluate.",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="eval_outputs",
        help="Root directory to store generated images for each checkpoint.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate per checkpoint.",
    )
    p.add_argument(
        "--stable_diffusion_checkpoint",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL base checkpoint identifier or local path.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/.cache/huggingface",
        help="HF cache directory.",
    )
    p.add_argument(
        "--compute_is",
        action="store_true",
        help="Also compute Inception Score (requires pytorch-gan-metrics).",
    )
    p.add_argument(
        "--fid_batch_size",
        type=int,
        default=1,
        help="Batch size for pytorch-fid. Use 1 to avoid crashes when images have different sizes.",
    )
    p.add_argument(
        "--use_lora",
        action="store_true",
        help="Pass --use_lora to test_laion2.py.",
    )
    p.add_argument(
        "--lora_weights_path",
        type=str,
        default="",
        help="Path to LoRA weights if --use_lora is set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb setup
    wandb.init(
        project="sgdiff-eval",
        name="fid-is-eval",
        config={
            "num_samples": args.num_samples,
            "checkpoints": args.checkpoints,
            "compute_is": args.compute_is,
            "fid_batch_size": args.fid_batch_size,
        },
    )
    wandb.define_metric("*", step_metric="step")
    summary_table = wandb.Table(columns=["checkpoint", "fid", "is_mean", "is_std"])

    real_dir = os.path.abspath(args.real_dir)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    results: list[dict] = []

    for step_idx, ckpt in enumerate(args.checkpoints):
        ckpt_name = Path(ckpt).stem  # e.g., epoch_49_iter_10583
        gen_dir = os.path.join(output_root, ckpt_name)

        # 1) Generate images for this checkpoint
        run_generation_for_checkpoint(
            checkpoint_path=ckpt,
            test_json_path=args.test_json_path,
            image_dir=args.image_dir,
            output_dir=gen_dir,
            num_samples=args.num_samples,
            stable_diffusion_checkpoint=args.stable_diffusion_checkpoint,
            cache_dir=args.cache_dir,
            use_lora=args.use_lora,
            lora_weights_path=args.lora_weights_path,
        )

        # 2) Compute FID
        print(f"\n=== Computing FID for checkpoint: {ckpt} ===")
        fid = compute_fid(real_dir=real_dir, gen_dir=gen_dir, batch_size=args.fid_batch_size, device=device)
        print(f"FID ({ckpt_name}): {fid:.4f}")

        # 3) Optionally compute Inception Score
        is_mean = is_std = None
        if args.compute_is:
            print(f"Computing Inception Score for checkpoint: {ckpt}")
            is_mean, is_std = compute_inception_score(
                gen_dir=gen_dir,
                batch_size=32,
                splits=10,
                device=device,
            )
            print(f"Inception Score ({ckpt_name}): {is_mean:.4f} ± {is_std:.4f}")
        log_data = {
            "fid": fid,
            "checkpoint": ckpt_name,
            "step": step_idx,
        }
        if is_mean is not None:
            log_data["is_mean"] = is_mean
            log_data["is_std"] = is_std
        wandb.log(log_data)

        results.append(
            {
                "checkpoint": ckpt,
                "generated_dir": gen_dir,
                "fid": fid,
                "is_mean": is_mean,
                "is_std": is_std,
            }
        )

        summary_table.add_data(
            ckpt_name,
            fid,
            is_mean,
            is_std,
        )


    # Print summary table and log to wandb
    print("\n=== Evaluation Summary ===")
    for r in results:
        line = f"{Path(r['checkpoint']).name}: FID={r['fid']:.4f}"
        if r["is_mean"] is not None:
            line += f", IS={r['is_mean']:.4f} ± {r['is_std']:.4f}"
        print(line)
    
    wandb.log({"evaluation_summary": summary_table})
    wandb.finish()

if __name__ == "__main__":
    main()



