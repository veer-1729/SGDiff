import json
from pathlib import Path

from sg2im.data.vg import VgSceneGraphDataset

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None


class VgSceneGraphWithCaptions(VgSceneGraphDataset):
    """
    Extends the standard VG scene graph dataset to return captions for each
    image, enabling (caption, scene graph, image) triplets.
    """

    def __init__(self, *args, captions_json: str, **kwargs):
        super().__init__(*args, **kwargs)

        captions_path = Path(captions_json)
        if not captions_path.exists():
            raise FileNotFoundError(f"Missing captions file: {captions_path}")

        with open(captions_path, "r") as f:
            data = json.load(f)
        entries = data["captions"] if isinstance(data, dict) else data
        self.caption_map = {entry["image_path"]: entry["caption"] for entry in entries}

    def __getitem__(self, index):
        image, objs, boxes, triples = super().__getitem__(index)

        rel_path = self.image_paths[index]
        if isinstance(rel_path, bytes):
            rel_path = rel_path.decode("utf-8")

        caption = self.caption_map.get(rel_path, "")
        return caption, image, objs, boxes, triples


def build_dataloader(
    vocab_path: str,
    h5_path: str,
    image_dir: str,
    captions_json: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Convenience helper to instantiate the dataset + DataLoader.
    """
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to build the dataloader.")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    dataset = VgSceneGraphWithCaptions(
        vocab=vocab,
        h5_path=h5_path,
        image_dir=image_dir,
        captions_json=captions_json,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

