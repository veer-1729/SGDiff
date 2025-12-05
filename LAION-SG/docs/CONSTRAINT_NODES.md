# GNN Constraint Nodes Design

## Overview

This document describes how constraints are injected into scene graphs as special GNN nodes,
allowing the sgEncoder to learn constraint-conditioned image generation.

## Constraint Categories

Given our dataset size (1-2k images), we focus on **grounded, learnable** constraints:

### 1. Object Presence Constraints
- **MUST_HAVE**: The image must contain this object type
- **MUST_NOT_HAVE**: The image should not contain this object type (negative constraint)
- **Format**: `{"type": "presence", "object": "car", "polarity": "positive"}`

### 2. Attribute Constraints  
- Specifies required attributes for objects
- **Format**: `{"type": "attribute", "object": "car", "attribute": "red"}`
- Connects constraint node to the actual object node in the graph

### 3. Relation/Spatial Constraints
- Enforces specific relationships between objects
- **Format**: `{"type": "relation", "subject": "man", "predicate": "on", "object": "skateboard"}`
- Creates edges from constraint node to both subject and object nodes

### 4. Count Constraints
- Specifies minimum/maximum/exact counts
- **Format**: `{"type": "count", "object": "person", "operator": ">=", "value": 2}`

## JSON Schema Extension

```json
{
  "name": "image_001.jpg",
  "img_id": "001",
  "caption_ori": "A red car parked on a street",
  "items": [...],
  "relations": [...],
  "constraints": [
    {"type": "presence", "object": "car", "polarity": "positive"},
    {"type": "attribute", "object": "car", "attribute": "red"},
    {"type": "relation", "subject": "car", "predicate": "on", "object": "street"}
  ]
}
```

## GNN Node Injection Strategy

### Approach: Constraints as Special Triples

Constraints are converted to triples that the sgEncoder can process:

1. **Presence Constraint** → `{"item1": "[MUST_HAVE]", "relation": "requires", "item2": "car"}`
2. **Attribute Constraint** → `{"item1": "[ATTR]", "relation": "describes", "item2": "red car"}`
3. **Relation Constraint** → `{"item1": "man", "relation": "[MUST_BE] on", "item2": "skateboard"}`
4. **Count Constraint** → `{"item1": "[COUNT>=2]", "relation": "requires", "item2": "person"}`

### Why This Works

- The sgEncoder's GNN processes all triples uniformly
- Special tokens like `[MUST_HAVE]`, `[ATTR]`, `[COUNT>=2]` become learned embeddings
- The GNN propagates constraint information through message passing
- Constraint nodes connect to relevant object nodes, creating information flow

## Implementation Components

1. **`sgEncoderTraining/constraints/constraints.py`**: Constraint schema definitions and parsing utilities
2. **`scripts/add_constraints_to_json.py`**: Add constraints to existing LAION-SG JSON files
3. **`sgEncoderTraining/datasets/laion_dataset_with_constraints.py`**: Extended dataset loader
4. **`scripts/convert_vg_to_laion.py`**: VG→LAION conversion with built-in constraint generation

## Usage

### Option 1: Generate constraints during VG→LAION conversion

```bash
# Convert VG HDF5 to LAION-SG JSON with auto-generated constraints
python scripts/convert_vg_to_laion.py \
    --h5_path datasets/vg/train.h5 \
    --vocab_json datasets/vg/vocab.json \
    --captions_json datasets/vg/captions.json \
    --output_json LAION-SG/vg_train_constrained.json \
    --max_samples 1000 \
    --max_presence 2 \
    --max_attribute 2 \
    --max_relation 1 \
    --constraint_seed 42
```

### Option 2: Add constraints to existing LAION-SG JSON

```bash
# Add constraints to existing JSON file
cd LAION-SG
python scripts/add_constraints_to_json.py \
    --input_json vg_train.json \
    --output_json vg_train_with_constraints.json \
    --max_presence 2 \
    --max_attribute 2 \
    --max_relation 1 \
    --seed 42
```

### Option 3: Use constraint-aware dataset loader

```python
# In trainer_laion.py, replace:
from sgEncoderTraining.datasets.laion_dataset import build_laion_loaders

# With:
from sgEncoderTraining.datasets.laion_dataset_with_constraints import build_constraint_loaders as build_laion_loaders

# Add to args:
args.use_constraints = True
args.max_constraint_triples = 5
```

## Training Recommendations

| Setup | Pros | Cons |
|-------|------|------|
| 1k images × 100 epochs | Deep learning per example | Risk of memorization |
| 2k images × 50 epochs | More diversity, better generalization | Each constraint seen fewer times |

**Recommendation**: 2k × 50 for better constraint generalization.

With constraints, each image effectively becomes multiple training samples
(one per constraint subset), increasing effective diversity.

## Constraint Generation Strategy

For each image in the dataset, we auto-generate constraints from its scene graph:

1. **Sample 1-3 objects** → presence constraints
2. **For objects with attributes** → attribute constraints  
3. **Sample 1-2 relations** → relation constraints
4. **Count objects of same type** → count constraints (if applicable)

This ensures constraints are **grounded in actual image content** and learnable.

