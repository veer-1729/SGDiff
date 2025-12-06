#!/usr/bin/env python3
"""
Interactive caption → scene-graph → image chat pipeline (controller).

This script does NOT train models. It provides a thin LLM-based controller
for test-time interaction:

1) Initialize a scene graph from a user caption (+ optional constraints):

   python chat_sg_pipeline.py init \
       --examples_json vg_train_clean_laion.json \
       --caption "a purple couch next to a window in a cozy living room" \
       --extra_constraints "must contain a purple couch; at least one window" \
       --sg_out user_sg_state.json \
       --model gpt-4o-mini

2) Edit an existing scene graph with a follow-up instruction:

   python chat_sg_pipeline.py edit \
       --sg_in user_sg_state.json \
       --instruction "change the red bag to a blue bag" \
       --sg_out user_sg_state.json \
       --model gpt-4o-mini

Then you can render images from the current scene graph state using test_laion2.py:

   python test_laion2.py \
       --pretrained_sgencoder_path checkpoints/baseline3_100.pt \
       --test_json_path user_sg_state.json \
       --output_dir output/user_chat_session \
       --num_samples 1 \
       --image_dir /content/SGDiff/datasets/vg/images \
       --cache_dir /content/.cache/huggingface \
       --stable_diffusion_checkpoint stabilityai/stable-diffusion-xl-base-1.0 \
       --image_size 512
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


SYSTEM_PROMPT_INIT = """You are a scene graph designer for images.

Your job:
- Given a caption (and optional extra constraints), produce a structured
  scene graph in JSON with the following keys:
    - "items": list of objects { "item_id", "label", "attributes", "global_item_id" }
    - "relations": list of relations { "item1", "item2", "relation" }
    - "constraints": OPTIONAL list of constraints
      (types: presence, attribute, relation, count).

Rules:
- Items:
  - Use simple, singular labels like "person", "dog", "car", "tree".
  - attributes is a list of short adjectives (colors, sizes, materials, states).
  - item_id are integers starting from 0.
  - global_item_id can be the same as item_id (we just need a stable id).
- Relations:
  - "item1" and "item2" are item_id indices.
  - "relation" is a short verb or preposition phrase (e.g., "on", "next to", "behind").
- Constraints:
  - Optional, but when present must match one of these schemas:
    * presence:  {"type": "presence", "object": "...", "polarity": "positive"|"negative"}
    * attribute: {"type": "attribute", "object": "...", "attribute": "..."}
    * relation:  {"type": "relation", "subject": "...", "predicate": "...", "object": "..."}
    * count:     {"type": "count", "object": "...", "operator": ">=|<=|==", "value": N}
- Stay faithful to the caption and extra constraints.
- Be detailed but do NOT invent entirely new unrelated scenes.

Output:
- Return a SINGLE JSON object with keys: "items", "relations", and optionally "constraints".
"""


SYSTEM_PROMPT_EDIT = """You are a scene graph editor for images.

You will receive:
- CURRENT_SCENE_GRAPH: a JSON object with "items", "relations", and optional "constraints".
- USER_INSTRUCTION: a natural-language request to modify the scene, e.g.
  "change the red bag to a blue bag", "add a tree on the left", "remove the dog".

Your job:
- Apply the instruction by minimally modifying the existing scene graph.
- Keep the scene graph structure and item_id assignments as stable as possible:
  - Prefer changing attributes/relations instead of deleting and recreating objects.
  - Only add new items if the instruction clearly adds a new object.
  - Only remove items if the instruction clearly removes an object.
- Update constraints to reflect the new state when appropriate.

Output:
- Return a SINGLE JSON object with updated "items", "relations", and optionally "constraints".
"""


def _load_examples(path: Path, num_shots: int, seed: int = 42) -> List[Dict[str, Any]]:
    import random

    data = json.loads(path.read_text())
    if not data:
        raise ValueError(f"No data in examples_json: {path}")
    rng = random.Random(seed)
    return rng.sample(data, min(num_shots, len(data)))


def _build_init_user_prompt(
    caption: str,
    extra_constraints: Optional[str],
    examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a few-shot style prompt for initial caption→SG."""
    parts: List[str] = []

    if examples:
        parts.append("Here are examples of caption → scene graph:\n")
        for ex in examples:
            sg = {
                "items": ex.get("items", []),
                "relations": ex.get("relations", []),
            }
            if "constraints" in ex:
                sg["constraints"] = ex["constraints"]
            parts.append(
                "Caption:\n"
                f"{ex.get('caption_ori', '')}\n"
                "SceneGraph:\n"
                f"{json.dumps(sg, indent=2)}\n"
            )

    parts.append("Now design a scene graph for the following caption.\n")
    parts.append("Caption:\n" + caption.strip() + "\n")
    if extra_constraints:
        parts.append("ExtraConstraints:\n" + extra_constraints.strip() + "\n")
    parts.append("SceneGraph:\n")

    return "\n".join(parts)


def _build_edit_user_prompt(current_sg: Dict[str, Any], instruction: str) -> str:
    """Build a prompt to edit an existing SG with a user instruction."""
    parts: List[str] = []
    parts.append("CURRENT_SCENE_GRAPH:\n")
    parts.append(json.dumps(current_sg, indent=2))
    parts.append("\nUSER_INSTRUCTION:\n")
    parts.append(instruction.strip())
    parts.append(
        "\nNow apply the instruction by minimally editing CURRENT_SCENE_GRAPH. "
        "Return ONLY the updated JSON scene graph.\n"
    )
    return "\n".join(parts)


def _call_llm(system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract first JSON object from LLM output."""
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in LLM output.")
    snippet = text[first : last + 1]
    return json.loads(snippet)


def run_init(args: argparse.Namespace) -> None:
    caption = args.caption.strip()
    extra_constraints = (args.extra_constraints or "").strip() or None

    examples = None
    if args.examples_json is not None:
        examples = _load_examples(args.examples_json, num_shots=args.num_shots, seed=args.seed)

    user_prompt = _build_init_user_prompt(caption, extra_constraints, examples)
    raw = _call_llm(SYSTEM_PROMPT_INIT, user_prompt, model=args.model, temperature=args.temperature)
    sg = _extract_json(raw)

    # Normalize into a single-entry JSON compatible with test_laion2.py
    sg_entry = {
        "name": args.sg_name or "user_0001.png",
        "img_id": args.sg_id or "user_0001",
        "caption_ori": caption,
        "items": sg.get("items", []),
        "relations": sg.get("relations", []),
    }
    if "constraints" in sg:
        sg_entry["constraints"] = sg["constraints"]

    out_list = [sg_entry]
    args.sg_out.write_text(json.dumps(out_list, indent=2))
    print(f"Initialized scene graph saved to {args.sg_out}")


def run_edit(args: argparse.Namespace) -> None:
    data = json.loads(args.sg_in.read_text())
    if not data:
        raise ValueError(f"No entries in sg_in: {args.sg_in}")
    current_sg = data[0]

    # We only expose the SG part to the editor (not name/img_id)
    sg_core = {
        "items": current_sg.get("items", []),
        "relations": current_sg.get("relations", []),
    }
    if "constraints" in current_sg:
        sg_core["constraints"] = current_sg["constraints"]

    user_prompt = _build_edit_user_prompt(sg_core, args.instruction)
    raw = _call_llm(SYSTEM_PROMPT_EDIT, user_prompt, model=args.model, temperature=args.temperature)
    new_sg = _extract_json(raw)

    # Merge back into entry, keeping name/img_id/caption_ori
    updated_entry = dict(current_sg)
    updated_entry["items"] = new_sg.get("items", [])
    updated_entry["relations"] = new_sg.get("relations", [])
    if "constraints" in new_sg:
        updated_entry["constraints"] = new_sg["constraints"]

    out_list = [updated_entry]
    args.sg_out.write_text(json.dumps(out_list, indent=2))
    print(f"Updated scene graph saved to {args.sg_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat-style caption↔SG controller for SDXL-SG.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # init mode: caption (+constraints) → new SG
    p_init = subparsers.add_parser("init", help="Initialize a scene graph from a caption.")
    p_init.add_argument("--examples_json", type=Path, default=None,
                        help="LAION-style JSON with training SG examples for few-shot.")
    p_init.add_argument("--caption", type=str, required=True,
                        help="User caption describing the desired scene.")
    p_init.add_argument("--extra_constraints", type=str, default=None,
                        help="Optional natural-language constraints to bias the SG.")
    p_init.add_argument("--sg_out", type=Path, required=True,
                        help="Output JSON file to store the initialized SG.")
    p_init.add_argument("--sg_name", type=str, default=None,
                        help="Optional 'name' field for the SG entry.")
    p_init.add_argument("--sg_id", type=str, default=None,
                        help="Optional 'img_id' field for the SG entry.")
    p_init.add_argument("--num_shots", type=int, default=8,
                        help="Number of few-shot examples.")
    p_init.add_argument("--seed", type=int, default=42,
                        help="Random seed for few-shot selection.")
    p_init.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model name.")
    p_init.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature.")

    # edit mode: modify existing SG with instruction
    p_edit = subparsers.add_parser("edit", help="Edit an existing scene graph with a user instruction.")
    p_edit.add_argument("--sg_in", type=Path, required=True,
                        help="Input JSON with current SG state (single entry).")
    p_edit.add_argument("--instruction", type=str, required=True,
                        help="User instruction to modify the scene.")
    p_edit.add_argument("--sg_out", type=Path, required=True,
                        help="Output JSON with updated SG.")
    p_edit.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model name.")
    p_edit.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "init":
        run_init(args)
    elif args.mode == "edit":
        run_edit(args)
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()


