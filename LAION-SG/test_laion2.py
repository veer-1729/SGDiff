from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel
from sgEncoderTraining.datasets.laion_dataset import build_laion_loaders
from sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms
from transformers import AutoTokenizer, PretrainedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SDXL-SG outputs on a LAION-style JSON file.")
    parser.add_argument("--pretrained_sgencoder_path", type=str, required=True, help="Checkpoint to load sgEncoder from.")
    parser.add_argument("--test_json_path", type=str, required=True, help="LAION-style JSON containing scene graphs.")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory for images referenced in JSON.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated images.")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to render.")
    parser.add_argument(
        "--stable_diffusion_checkpoint",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF repo or local path for SDXL base.",
    )
    parser.add_argument("--cache_dir", type=str, default="/tmp/.cache/huggingface", help="Cache directory for HF downloads.")
    parser.add_argument("--image_size", type=int, default=1024, help="Output resolution for generated images.")
    parser.add_argument("--max_objects_per_image", type=int, default=10)
    parser.add_argument("--use_orphaned_objects", type=bool, default=True)
    parser.add_argument("--include_relationships", type=bool, default=True)
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--graph_width", type=int, default=512)
    parser.add_argument("--num_graph_layer", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--model_config_json", type=str, default="")
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--force_quick_gelu", action="store_true")
    parser.add_argument("--pretrained_image", action="store_true")
    return parser.parse_args()


def import_text_encoder(model_id: str, subfolder: str = "text_encoder"):
    config = PretrainedConfig.from_pretrained(model_id, subfolder=subfolder)
    cls_name = config.architectures[0]
    if cls_name == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if cls_name == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    raise ValueError(f"{cls_name} is not supported.")


def compute_time_ids(original_size, target_size, device):
    add_time_ids = list(original_size + (0, 0) + target_size)
    add_time_ids = torch.tensor([add_time_ids], device=device, dtype=torch.float16)
    return add_time_ids


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build loaders by reusing training utilities.
    args.batch_size = args.val_batch_size = 1
    args.train_json_path = args.test_json_path
    args.val_json_path = args.test_json_path
    args.logs = "none"
    args.model_config_json = ""
    args.precision = "fp16"
    args.force_quick_gelu = False
    args.pretrained_image = False

    _, val_loader = build_laion_loaders(args)

    vae = AutoencoderKL.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="tokenizer", use_fast=False, cache_dir=args.cache_dir
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="tokenizer_2", use_fast=False, cache_dir=args.cache_dir
    )

    text_cls = import_text_encoder(args.stable_diffusion_checkpoint)
    text_cls_2 = import_text_encoder(args.stable_diffusion_checkpoint, subfolder="text_encoder_2")

    text_encoder = text_cls.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="text_encoder",
        variant="fp16",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    ).to(device)
    text_encoder_2 = text_cls_2.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="text_encoder_2",
        variant="fp16",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="unet",
        variant="fp16",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    ).to(device)

    model = create_model_and_transforms(
        args,
        text_encoders=[text_encoder, text_encoder_2],
        tokenizers=[tokenizer, tokenizer_2],
        model_config_json=args.model_config_json,
        precision="fp16",
        device=device,
        force_quick_gelu=False,
        pretrained_image=False,
    ).to(device)
    model = model.half().eval()

    ckpt = torch.load(args.pretrained_sgencoder_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.stable_diffusion_checkpoint,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=unet,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_size = (args.image_size, args.image_size)

    for idx, batch in enumerate(val_loader):
        if idx >= args.num_samples:
            break

        (
            all_imgs,
            all_triples,
            all_global_ids,
            all_isolated_items,
            all_text_prompts,
            all_original_sizes,
            all_crop_top_lefts,
            all_img_ids,
        ) = batch

        prompt_embeds, pooled_embeds = model(all_triples, all_isolated_items, all_global_ids)
        original_size = all_original_sizes[0]
        add_time_ids = compute_time_ids(original_size, target_size, device)

        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            added_cond_kwargs={"time_ids": add_time_ids},
            num_inference_steps=40,
            height=args.image_size,
            width=args.image_size,
        ).images[0]

        out_path = out_dir / f"{all_img_ids[0]}.jpg"
        image.save(out_path, format="JPEG")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

