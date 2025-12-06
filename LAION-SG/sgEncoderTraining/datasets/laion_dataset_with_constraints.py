"""
Extended LAION dataset that supports constraint nodes.

Constraints are parsed from the JSON and injected as additional triples
into the scene graph, allowing the GNN to learn constraint-conditioned generation.
"""

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL
import os
import json
import sys

# Add parent path for constraint imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sgEncoderTraining.constraints import constraints_to_triples
    from sgEncoderTraining.constraints.heuristic_constraints import (
        generate_constraints_from_caption_and_sg,
    )
except ImportError:
    # Fallback for different import contexts
    from constraints import constraints_to_triples
    from heuristic_constraints import generate_constraints_from_caption_and_sg


class LAIONSceneGraphDatasetWithConstraints(Dataset):
    """
    LAION-SG dataset extended with constraint node support.
    
    When `use_constraints=True`, constraints from the JSON are converted
    to additional triples and appended to the scene graph.
    
    JSON format expected:
    {
        "name": "image.jpg",
        "img_id": "123",
        "items": [...],
        "relations": [...],
        "constraints": [
            {"type": "presence", "object": "car", "polarity": "positive"},
            {"type": "attribute", "object": "car", "attribute": "red"},
            ...
        ],
        "caption_ori": "A red car..."
    }
    """
    
    def __init__(
        self,
        image_dir: str,
        text_json_path: str,
        image_size: tuple = (256, 256),
        max_objects: int = 10,
        max_samples: int = None,
        include_relationships: bool = True,
        use_orphaned_objects: bool = True,
        use_constraints: bool = True,  # New parameter
        max_constraint_triples: int = 5,  # Limit constraint injection
    ):
        super().__init__()
        
        self.image_dir = image_dir
        self.image_size = image_size[0]
        self.text_json_path = text_json_path
        
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        
        # Constraint settings
        self.use_constraints = use_constraints
        self.max_constraint_triples = max_constraint_triples
        
        self.text_json_data = self._load_json()
        
        transform = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform)
    
    def __len__(self):
        num = len(self.text_json_data)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num
    
    def __getitem__(self, index):
        json_data = self.text_json_data[index]
        
        img_path = os.path.join(self.image_dir, str(json_data['name']))
        
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                width, height = image.size
                image = image.resize((self.image_size, self.image_size), PIL.Image.BILINEAR)
                image = self.transform(image.convert('RGB'))
        
        triples = []
        global_ids = []
        related_items = set()
        
        items = json_data["items"]
        relations = json_data["relations"]
        img_id = json_data["img_id"]
        
        item_dict = {item["item_id"]: item for item in items}
        
        # Build standard triples from relations
        for relation in relations:
            item1_id = relation["item1"]
            item2_id = relation["item2"]
            
            related_items.add(item1_id)
            related_items.add(item2_id)
            
            item1_attributes = " ".join(item_dict[item1_id]["attributes"]) + " " + item_dict[item1_id]["label"]
            item2_attributes = " ".join(item_dict[item2_id]["attributes"]) + " " + item_dict[item2_id]["label"]
            
            triples.append({
                "item1": item1_attributes,
                "relation": relation["relation"],
                "item2": item2_attributes
            })
            
            global_ids.append({
                "item1": item_dict[item1_id]["global_item_id"],
                "item2": item_dict[item2_id]["global_item_id"]
            })
        
        # Inject constraint triples (explicit from JSON, plus heuristic extras)
        if self.use_constraints:
            # 1) Constraints that may already be present in JSON (e.g., from LLM or conversion script)
            raw_constraints = list(json_data.get("constraints", []))

            # 2) Always add heuristic constraints from caption + scene graph
            heuristic_constraints = generate_constraints_from_caption_and_sg(
                caption=json_data.get("caption_ori", ""),
                items=items,
                relations=relations,
                max_presence=3,
                max_attribute=3,
                max_relation=3,
                max_count=1,
            )
            raw_constraints.extend(heuristic_constraints)

            # 3) Deduplicate constraints by semantic key
            deduped = []
            seen_keys = set()
            for c in raw_constraints:
                ctype = c.get("type")
                if ctype == "presence":
                    key = ("presence", c.get("object"), c.get("polarity", "positive"))
                elif ctype == "attribute":
                    key = ("attribute", c.get("object"), c.get("attribute"))
                elif ctype == "relation":
                    key = ("relation", c.get("subject"), c.get("predicate"), c.get("object"))
                elif ctype == "count":
                    key = ("count", c.get("object"), c.get("operator"), c.get("value"))
                else:
                    # Unknown type; keep as-is but avoid crashing
                    key = ("other", json.dumps(c, sort_keys=True))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(c)

            if deduped:
                constraint_triples = constraints_to_triples(deduped)
                # Limit number of constraint triples
                constraint_triples = constraint_triples[: self.max_constraint_triples]

                for ct in constraint_triples:
                    triples.append(ct)
                    # Use special global IDs for constraint nodes
                    global_ids.append({
                        "item1": -1,  # Special ID for constraint node
                        "item2": -2,  # Special ID for constraint target
                    })
        
        # Isolated items
        isolated_items = []
        for item in items:
            if item["item_id"] not in related_items:
                isolated_items.append(" ".join(item["attributes"]) + " " + item["label"])
        
        img_size = (width, height)
        img_text_prompt = json_data['caption_ori']
        
        return image, triples, global_ids, isolated_items, img_text_prompt, img_size, img_id
    
    def _load_json(self):
        with open(self.text_json_path, 'r') as file:
            json_data = json.load(file)
        
        # Filter entries that have relations
        filtered_json_data = [entry for entry in json_data if entry.get("relations")]
        
        return filtered_json_data


def collate_fn_with_constraints(batch):
    """Collate function compatible with constraint-extended dataset."""
    all_imgs = []
    all_triples = []
    all_global_ids = []
    all_isolated_items = []
    all_text_prompts = []
    all_original_sizes = []
    all_crop_top_lefts = []
    all_img_ids = []
    
    for (image, triples, global_ids, isolated_items, img_text_prompt, img_size, img_id) in batch:
        all_imgs.append(image[None])
        all_triples.append(triples)
        all_global_ids.append(global_ids)
        all_isolated_items.append(isolated_items)
        all_text_prompts.append(img_text_prompt)
        all_original_sizes.append(img_size)
        all_crop_top_lefts.append((0, 0))
        all_img_ids.append(img_id)
    
    return (
        all_imgs,
        all_triples,
        all_global_ids,
        all_isolated_items,
        all_text_prompts,
        all_original_sizes,
        all_crop_top_lefts,
        all_img_ids
    )


def build_constraint_loaders(args):
    """
    Build data loaders with constraint support.
    
    Falls back to standard loaders if `use_constraints` is not set.
    """
    use_constraints = getattr(args, 'use_constraints', True)
    max_constraint_triples = getattr(args, 'max_constraint_triples', 5)
    
    dset_kwargs = {
        'image_dir': args.image_dir,
        'text_json_path': args.train_json_path,
        'image_size': (args.image_size, args.image_size),
        'max_samples': None,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
        'use_constraints': use_constraints,
        'max_constraint_triples': max_constraint_triples,
    }
    
    train_dset = LAIONSceneGraphDatasetWithConstraints(**dset_kwargs)
    
    iter_per_epoch = len(train_dset) // args.batch_size
    print(f'There are {iter_per_epoch} iterations per epoch')
    print(f'Constraints enabled: {use_constraints}')
    
    dset_kwargs['text_json_path'] = args.val_json_path
    del dset_kwargs['max_samples']
    val_dset = LAIONSceneGraphDatasetWithConstraints(**dset_kwargs)
    
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'shuffle': True,
        'collate_fn': collate_fn_with_constraints,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    train_loader.num_samples = len(train_dset)
    
    print(f'There are {len(train_dset)} train samples')
    
    loader_kwargs['batch_size'] = args.val_batch_size
    loader_kwargs['shuffle'] = False
    
    val_loader = DataLoader(val_dset, **loader_kwargs)
    val_loader.num_samples = len(val_dset)
    
    print(f'There are {len(val_dset)} validation samples')
    
    return train_loader, val_loader


# For backward compatibility, also export with original names
LAIONSceneGraphDataset = LAIONSceneGraphDatasetWithConstraints
my_laion_collate_fn = collate_fn_with_constraints
build_laion_loaders = build_constraint_loaders

