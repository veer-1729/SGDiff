import argparse
import json
from pathlib import Path

import h5py
from PIL import Image
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate captions for the current VG subset using BLIP.')
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        default=Path('datasets/vg'),
        help='Directory containing {train,val,test}.h5 and images/.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='List of splits whose images should be captioned.')
    parser.add_argument(
        '--image_root',
        type=Path,
        default=Path('datasets/vg/images'),
        help='Root directory that stores VG_100K and VG_100K_2 folders.')
    parser.add_argument(
        '--output_path',
        type=Path,
        default=Path('datasets/vg/captions.json'),
        help='Where to save the generated captions.')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Torch device to run the captioning model on.')
    parser.add_argument(
        '--model_name',
        default='Salesforce/blip-image-captioning-large',
        help='HF model checkpoint used for captioning.')
    parser.add_argument(
        '--max_length',
        type=int,
        default=30,
        help='Maximum length for generated captions.')
    parser.add_argument(
        '--num_beams',
        type=int,
        default=5,
        help='Beam search width.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output file if present.')
    return parser.parse_args()


def load_image_list(dataset_dir: Path, splits):
    image_paths = []
    seen = set()
    for split in splits:
        split_path = dataset_dir / f'{split}.h5'
        if not split_path.exists():
            raise FileNotFoundError(f'Missing split file: {split_path}')
        with h5py.File(split_path, 'r') as h5f:
            paths = h5f['image_paths'][:]
            for raw in paths:
                path = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                if path not in seen:
                    seen.add(path)
                    image_paths.append(path)
    return image_paths


def generate_captions(args):
    output_path = args.output_path
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f'{output_path} already exists. Use --overwrite to replace it.')

    print(f'Loading BLIP model "{args.model_name}" on {args.device}')
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    rel_paths = load_image_list(args.dataset_dir, args.splits)
    print(f'Found {len(rel_paths)} unique images across {args.splits}')

    captions = []
    missing = 0

    for rel_path in tqdm(rel_paths, desc='Captioning images'):
        abs_path = args.image_root / rel_path
        if not abs_path.exists():
            missing += 1
            continue
        image = Image.open(abs_path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt').to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=args.num_beams,
                max_length=args.max_length)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append({'image_path': rel_path, 'caption': caption})

    if missing:
        print(f'Warning: {missing} images were missing on disk and skipped.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'captions': captions}, f, indent=2)
    print(f'Wrote {len(captions)} captions to {output_path}')


if __name__ == '__main__':
    generate_captions(parse_args())

