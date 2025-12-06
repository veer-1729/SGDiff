#!/usr/bin/env python3
"""
Caption → Scene Graph (+ optional constraints) via LLM with many-shot VG examples.

This script does NOT train a model; it uses in-context learning (few-shot prompting)
on top of an external LLM (e.g., OpenAI GPT) to convert captions into LAION-SG style
scene graphs.

Usage (example with OpenAI):

    export OPENAI_API_KEY=sk-...
    python caption_to_sg_llm.py \
        --examples_json vg_train.json \
        --input_json vg_val.json \
        --output_json vg_val_llm_sg.json \
        --num_shots 8 \
        --max_samples 50 \
        --model gpt-4o-mini

For each entry in input_json, we read:
    - caption_ori   (caption)
and ask the LLM to generate:
    - items         (list of {item_id, label, attributes, global_item_id})
    - relations     (list of {item1, item2, relation})
    - optionally constraints (if ExtraConstraints are provided in the prompt)

At inference-time you can also provide user constraints in natural language, e.g.:
    "Constraints: must contain a red car; at least two people"
and the LLM is instructed to map those into the 4 constraint types
(presence, attribute, relation, count) when possible.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # OpenAI >=1.0 style client
    from openai import OpenAI
except ImportError:  # pragma: no cover - library may not be installed by default
    OpenAI = None  # type: ignore

SYSTEM_PROMPT = """You are a model that converts image captions into structured scene graphs
in the following JSON format (LAION-SG style), and you MUST think step by step
before producing the final JSON.

For each caption you receive, you should do two stages:

STAGE 1 - STRUCTURED REASONING
- Infer the important objects, attributes, and relations in a structured way.
- Represent your chain-of-thought as a JSON object with this schema:

  "reasoning": {
    "explicit_objects": [ "object mentioned directly in caption (e.g., couch)", ... ],
    "implied_objects":  [ "typical/background objects consistent with the caption (e.g., floor, wall)", ... ],
    "attributes": [
      { "object": "couch", "attribute": "purple" },
      { "object": "wall", "attribute": "white" }
    ],
    "relations": [
      { "subject": "couch", "predicate": "next to", "object": "window" },
      { "subject": "vase",  "predicate": "on",      "object": "table" }
    ]
  }

- explicit_objects: ONLY objects explicitly named in the caption.
- implied_objects: plausible objects given the scene type (e.g., floor/wall/window in a living room),
  but do NOT introduce objects that contradict the caption.
- attributes: colors/materials/sizes/poses that are explicit or very typical.
- relations: spatial/functional relations that are clearly suggested by the caption and objects.

STAGE 2 - SCENE GRAPH CONSTRUCTION
- From your reasoning, build a final scene graph with these keys:

  - "items": a list of objects with attributes:
      {
        "item_id": 0,
        "label": "couch",
        "attributes": ["purple"],
        "global_item_id": 0
      }

  - "relations": a list of relations between objects:
      {
        "item1": 0,
        "item2": 1,
        "relation": "next to"
      }

  - "constraints": OPTIONAL list of constraints:
      * presence:  {"type": "presence", "object": "...", "polarity": "positive"|"negative"}
      * attribute: {"type": "attribute", "object": "...", "attribute": "..."}
      * relation:  {"type": "relation", "subject": "...", "predicate": "...", "object": "..."}
      * count:     {"type": "count", "object": "...", "operator": ">=|<=|==", "value": N}

Rules for items and relations:
- Use simple singular labels like "person", "dog", "car", "tree", "couch", "table".
- attributes is a list of short adjectives (colors, sizes, materials, states).
- item_id are integers starting from 0; global_item_id can equal item_id.
- relations[i].item1 and relations[i].item2 must refer to item_id values.
- If your reasoning or constraints indicate there are MULTIPLE instances of an
  object (e.g., "several desks", "two computers"), you SHOULD create multiple
  items with the same label but different item_id values (0,1,2,...) instead of
  representing all of them by a single item.

Rules for constraints:
- constraints is OPTIONAL, but when present must use the schemas above.
- You SHOULD convert the most important facts from your reasoning into constraints
  (e.g., presence of key objects, critical relations, important counts).
- If the user provides an ExtraConstraints section, interpret each natural-language
  constraint and, when possible, map it to exactly ONE of:
    * presence: {"type": "presence", "object": "...", "polarity": "positive"|"negative"}
    * attribute: {"type": "attribute", "object": "...", "attribute": "..."}
    * relation:  {"type": "relation", "subject": "...", "predicate": "...", "object": "..."}
    * count:     {"type": "count", "object": "...", "operator": ">=|<=|==", "value": N}
  Ignore nonsense or unsupported constraints.

Output:
- ALWAYS return a single JSON object with keys:
  - "reasoning" (structured as above),
  - "items",
  - "relations",
  - and optionally "constraints".
"""


def build_few_shot_prompt(
    examples: List[Dict[str, Any]],
    target_caption: str,
    extra_constraints_text: Optional[str] = None,
) -> str:
    """Build a many-shot prompt from VG examples, now with structured CoT reasoning.

    Each example shows:
      - a caption
      - a (template) structured reasoning block
      - the corresponding scene graph (items/relations/constraints)
    """

    parts: List[str] = ["Here are examples of caption → structured reasoning → scene graph:\n"]

    for ex in examples:
        # Use provided reasoning if present; otherwise fall back to an empty template
        reasoning = ex.get("reasoning") or {
            "explicit_objects": [],
            "implied_objects": [],
            "attributes": [],
            "relations": [],
        }

        sg = {
            "reasoning": reasoning,
            "items": ex.get("items", []),
            "relations": ex.get("relations", []),
        }
        if "constraints" in ex:
            sg["constraints"] = ex["constraints"]

        parts.append(
            "Caption:\n"
            f"{ex.get('caption_ori', '')}\n"
            "SceneGraphWithReasoning:\n"
            f"{json.dumps(sg, indent=2)}\n"
        )

    # Target query
    parts.append("Now for the following caption, first do structured reasoning, then build the scene graph.\n")
    parts.append(f"Caption:\n{target_caption}\n")
    if extra_constraints_text:
        parts.append("ExtraConstraints:\n")
        parts.append(extra_constraints_text.strip() + "\n")
    parts.append("SceneGraphWithReasoning:\n")

    return "\n".join(parts)



def call_llm(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """Call the external LLM and return the raw text response."""
    if OpenAI is None:
        raise RuntimeError(
            "openai package not installed. Please `pip install openai` and "
            "set OPENAI_API_KEY."
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Try to extract the first JSON object from the LLM output.
    Falls back to raising ValueError if parsing fails.
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in LLM output.")
    snippet = text[first : last + 1]
    return json.loads(snippet)


def load_examples(path: Path, num_shots: int, seed: int = 42) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not data:
        raise ValueError(f"No data in examples_json: {path}")
    rng = random.Random(seed)
    return rng.sample(data, min(num_shots, len(data)))


def iterate_targets(path: Path, max_samples: Optional[int] = None):
    data = json.loads(path.read_text())
    if max_samples is not None:
        data = data[:max_samples]
    for entry in data:
        yield entry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption → Scene Graph via LLM (many-shot).")
    p.add_argument("--examples_json", type=Path, required=True,
                   help="JSON with VG-style entries used as few-shot examples.")
    p.add_argument("--input_json", type=Path, required=True,
                   help="JSON with captions to convert (VG-style).")
    p.add_argument("--output_json", type=Path, required=True,
                   help="Where to save LLM-generated scene graphs.")
    p.add_argument("--num_shots", type=int, default=8,
                   help="Number of few-shot examples to include in prompt.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional cap on number of input captions.")
    p.add_argument("--model", type=str, default="gpt-4o-mini",
                   help="LLM model name (for OpenAI client).")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="Sampling temperature.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for example sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.examples_json} as few-shot examples...")
    examples = load_examples(args.examples_json, num_shots=args.num_shots, seed=args.seed)

    outputs: List[Dict[str, Any]] = []

    print(f"Processing captions from {args.input_json}...")
    for i, entry in enumerate(iterate_targets(args.input_json, max_samples=args.max_samples)):
        caption = entry.get("caption_ori", "")
        if not caption:
            continue

        # Optionally, a future extension: read user constraints from entry["extra_constraints"]
        extra_constraints_text = entry.get("extra_constraints", None)

        prompt = build_few_shot_prompt(
            examples=examples,
            target_caption=caption,
            extra_constraints_text=extra_constraints_text,
        )

        try:
            raw = call_llm(prompt, model=args.model, temperature=args.temperature)
            sg = extract_json_block(raw)
        except Exception as e:  # pragma: no cover - robust to LLM quirks
            print(f"[WARN] Failed to parse LLM output for index {i}: {e}")
            continue

        # Attach original caption + id for traceability
        sg_entry = {
            "name": entry.get("name"),
            "img_id": entry.get("img_id"),
            "caption_ori": caption,
            "items": sg.get("items", []),
            "relations": sg.get("relations", []),
        }
        if "constraints" in sg:
            sg_entry["constraints"] = sg["constraints"]

        outputs.append(sg_entry)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} captions...")

    print(f"Saving {len(outputs)} entries to {args.output_json} ...")
    args.output_json.write_text(json.dumps(outputs, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()


