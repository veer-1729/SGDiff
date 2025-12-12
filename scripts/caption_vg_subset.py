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
        default=Path('datasets/vg'))
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'])
    parser.add_argument(
        '--image_root',
        type=Path,
        default=Path('datasets/vg/images'))
    parser.add_argument(
        '--output_path',
        type=Path,
        default=Path('datasets/vg/captions.json'),
        help='Where to save the generated captions.')
    parser.add_argument(
        '--device',
        default= 'cpu')
    parser.add_argument(
        '--model_name',
        default='Salesforce/blip-image-captioning-large')
    parser.add_argument(
        '--max_length',
        type=int,
        default=30)
    parser.add_argument(
        '--num_beams',
        type=int,
        default=5)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4)
    parser.add_argument(
        '--overwrite',
        action='store_true')
    parser.add_argument(
        '--vocab_json',
        type=Path,
        default=Path('datasets/vg/vocab.json'))
    parser.add_argument(
        '--use_sg_prompt',
        action='store_true')
    return parser.parse_args()

def load_image_list(dataset_dir: Path, splits):
    image_paths = []
    seen = set()

    for split in splits:
        split_path = dataset_dir / f'{split}.h5'
        if not split_path.exists():
            raise FileNotFoundError(f'ERROR: File not found')

        with h5py.File(split_path, 'r') as h5f:
            paths = h5f['image_paths'][:]

            for raw in paths:
                path = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                if path not in seen:
                    seen.add(path)
                    image_paths.append(path)
    return image_paths

def build_allowed_objects_map(dataset_dir: Path, splits, vocab_json: Path):
    """
    Build a mapping from image relative path to sorted list of object labels present in the VG scene graph for that image.
    """

    with vocab_json.open('r') as f:
        vocab = json.load(f)
    obj_names = vocab['object_idx_to_name']

    img_to_labels = {}

    for split in splits:
        split_path = dataset_dir / f'{split}.h5'

        with h5py.File(split_path, 'r') as h5f:
            image_paths = h5f['image_paths'][:]
            object_names = h5f['object_names'][:]  # (num_images, max_objects) of indices
            objects_per_image = h5f['objects_per_image'][:]

            for idx in range(len(image_paths)):
                raw = image_paths[idx]
                rel_path = raw.decode('utf-8') if isinstance(raw, bytes) else raw

                # Skip if we already collected labels from another split or run 
                if rel_path in img_to_labels:
                    continue
                num_obj = int(objects_per_image[idx])
                labels = set()
                for slot in range(num_obj):
                    name_idx = int(object_names[idx, slot])
                    if name_idx < 0 or name_idx >= len(obj_names):
                        continue
                    labels.add(obj_names[name_idx])
                
                # edge case
                if labels:
                    img_to_labels[rel_path] = sorted(labels)

    return img_to_labels

def generate_captions(args):
    output_path = args.output_path
    if output_path.exists() and not args.overwrite:
        raise FileExistsError("Use --overwrite!!!!!")

    print(f'Loading BLIP model "{args.model_name}"')
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    rel_paths = load_image_list(args.dataset_dir, args.splits)
    print(f'Found {len(rel_paths)} unique images across {args.splits}')

    sg_map = {}
    if args.use_sg_prompt:
        print('Building scene-graph-conditioned prompts from VG HDF5 + vocab...')
        sg_map = build_allowed_objects_map(args.dataset_dir, args.splits, args.vocab_json)
        print(f'Found SG object lists for {len(sg_map)} images.')

    captions = []
    missing = 0

    def process_batch(batch_images, batch_paths):
        if not batch_images:
            return
        if args.use_sg_prompt:
            batch_prompts = []
            for rel_path in batch_paths:
                labels = sg_map.get(rel_path, [])
                if labels:
                    obj_list = ', '.join(labels)
                    prompt = (
                        "Describe this image in one or two sentences as a global caption. "
                        "Only mention objects from this list and do NOT invent new object "
                        "categories or actions: "
                        f"{obj_list}."
                    )

                else:
                    # Edge case: no sg info available
                    prompt = "Describe this image in one or two sentences."
                batch_prompts.append(prompt)
            inputs = processor(
                images=batch_images,
                text=batch_prompts,
                return_tensors='pt',
                padding=True).to(args.device)
        else:
            inputs = processor(
                images=batch_images,
                return_tensors='pt',
                padding=True).to(args.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=args.num_beams, do_sample=False, max_length=args.max_length)

        for rel_path, sequence in zip(batch_paths, outputs):
            caption = processor.decode(sequence, skip_special_tokens=True)
            captions.append({'image_path': rel_path, 'caption': caption})

    batch_images, batch_paths = [], []

    for rel_path in tqdm(rel_paths, desc='Captioning images'):
        abs_path = args.image_root / rel_path

        if not abs_path.exists():
            missing += 1
            continue

        with Image.open(abs_path) as img:
            batch_images.append(img.convert('RGB'))

        batch_paths.append(rel_path)
        if len(batch_images) >= args.batch_size:
            process_batch(batch_images, batch_paths)
            batch_images, batch_paths = [], []

    # Edge case procees remainder
    process_batch(batch_images, batch_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'captions': captions}, f, indent=2)
    print(f'Wrote {len(captions)} captions to {output_path}')


if __name__ == '__main__':
    generate_captions(parse_args())

