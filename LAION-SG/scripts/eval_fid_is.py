#!/usr/bin/env python3
"""
IS AND FID Evaluation
"""

import os
import subprocess
from pathlib import Path
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch_fidelity



REAL_DIR = "../datasets/vg/val_imgs_subset"  
TEST_JSON = "vg_val_2k.json"  
IMAGE_DIR = "../datasets/vg/VG_100K"  

CHECKPOINTS = [
    "vg_contrastive_run_final/checkpoints/epoch_49_iter_10583.pt",
    "2025_12_07-08_02_26-lr_0.0001-b_8-j_1-p_amp_bfloat16/checkpoints/epoch_56_iter_13999.pt",
    "../logs_vg/2025_12_05-06_45_44-lr_0.0001-b_4-j_4-p_amp/checkpoints/epoch_71_iter_17749.pt",
]                                        
OUTPUT_ROOT = "eval_outputs"                 
NUM_SAMPLES = 50                           
SDXL_CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
CACHE_DIR = "/tmp/.cache/hf"
COMPUTE_IS = True


# metrics functions
def generate_images(ckpt, out_dir):
    cmd = [
        "python", "test_laion2.py",
        "--pretrained_sgencoder_path", ckpt,
        "--test_json_path", os.path.abspath(TEST_JSON),
        "--image_dir", os.path.abspath(IMAGE_DIR),
        "--output_dir", os.path.abspath(out_dir),
        "--num_samples", str(NUM_SAMPLES),
        "--stable_diffusion_checkpoint", SDXL_CKPT,
        "--cache_dir", CACHE_DIR,
    ]
    print("\n[GEN] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def fid(real_dir, gen_dir):
    paths = [os.path.abspath(real_dir), os.path.abspath(gen_dir)]
    return float(
        calculate_fid_given_paths(paths, batch_size=1, device="cuda" if torch.cuda.is_available() else "cpu")
    )

def inception_score(gen_dir):
    metrics = torch_fidelity.calculate_metrics(
        input1=os.path.abspath(gen_dir),
        isc=True,
        cuda=torch.cuda.is_available(),
        batch_size=32,
        verbose=False,
    )
    return float(metrics["inception_score_mean"]), float(metrics["inception_score_std"])

# main code to evaluate FID and IS
os.makedirs(OUTPUT_ROOT, exist_ok=True)
results = []

for ckpt in CHECKPOINTS:
    ckpt_name = Path(ckpt).stem
    gen_dir = os.path.join(OUTPUT_ROOT, ckpt_name)

    generate_images(ckpt, gen_dir)

    score_fid = fid(REAL_DIR, gen_dir)
    print(f"FID[{ckpt_name}] = {score_fid:.4f}")

    score_is = (None, None)
    if COMPUTE_IS:
        mean, std = inception_score(gen_dir)
        score_is = (mean, std)
        print(f"IS[{ckpt_name}] = {mean:.4f} ± {std:.4f}")

    results.append((ckpt_name, score_fid, *score_is))


# print the stats
for ckpt_name, fid_val, is_mean, is_std in results:
    line = f"{ckpt_name}: FID={fid_val:.4f}"
    if is_mean is not None:
        line += f", IS={is_mean:.4f} ± {is_std:.4f}"
    print(line)
