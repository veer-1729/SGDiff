import re
from collections import Counter
from typing import List, Dict, Any, Optional

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
}

def _normalize(text: str) -> str:
    return text.lower().strip()

def _simple_tokenize(text: str) -> List[str]:
    # very lightweight tokenizer
    return re.findall(r"\w+", text.lower())

def generate_constraints_from_caption_and_sg(
    caption: str,
    items: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    alias_map: Optional[Dict[str, str]] = None,
    max_presence: Optional[int] = None,
    max_attribute: Optional[int] = None,
    max_relation: Optional[int] = None,
    max_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Heuristic constraint generator:
    - PRESENCE: objects whose label appears in caption
    - ATTRIBUTE: attributes whose word appears in caption near/with that object
    - RELATION: subject-predicate-object phrases that appear in caption
    - COUNT: numeric mentions like 'two dogs', 'three trees'
    """
    caption_norm = " " + _normalize(caption) + " "
    tokens = _simple_tokenize(caption)

    # optional aliasing: "puppy" -> "dog", etc.
    alias_map = alias_map or {}

    def canon_label(label: str) -> str:
        lab = _normalize(label)
        return alias_map.get(lab, lab)

    # Map labels to items
    label_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        lab = canon_label(it["label"])
        label_to_items.setdefault(lab, []).append(it)

    constraints: List[Dict[str, Any]] = []

    # -------------------------
    # 1) PRESENCE constraints
    # -------------------------
    presence_candidates = []
    for lab in label_to_items.keys():
        if f" {lab} " in caption_norm:
            presence_candidates.append(lab)

    if max_presence is not None:
        presence_candidates = presence_candidates[:max_presence]

    for lab in presence_candidates:
        constraints.append({
            "type": "presence",
            "object": lab,
            "polarity": "positive",
        })

    # -------------------------
    # 2) ATTRIBUTE constraints
    # -------------------------
    attr_constraints = []
    for it in items:
        lab = canon_label(it["label"])
        for attr in it.get("attributes", []):
            attr_norm = _normalize(attr)
            # simple check: both words appear somewhere in caption
            if f" {lab} " in caption_norm and f" {attr_norm} " in caption_norm:
                attr_constraints.append({
                    "type": "attribute",
                    "object": lab,
                    "attribute": attr_norm,
                })

    if max_attribute is not None:
        attr_constraints = attr_constraints[:max_attribute]

    constraints.extend(attr_constraints)

    # -------------------------
    # 3) RELATION constraints
    # -------------------------
    # need item_id -> label map
    id_to_label = {it["item_id"]: canon_label(it["label"]) for it in items}
    rel_constraints = []
    for rel in relations:
        subj = id_to_label.get(rel["item1"])
        obj = id_to_label.get(rel["item2"])
        pred = _normalize(rel["relation"])
        if subj is None or obj is None:
            continue

        # naive pattern "dog on couch"
        pattern = f" {subj} {pred} {obj} "
        if pattern in caption_norm:
            rel_constraints.append({
                "type": "relation",
                "subject": subj,
                "predicate": pred,
                "object": obj,
            })

    if max_relation is not None:
        rel_constraints = rel_constraints[:max_relation]

    constraints.extend(rel_constraints)

    # -------------------------
    # 4) COUNT constraints
    # -------------------------
    # very simple: "two dogs", "three trees", or "2 dogs"
    token_counts = Counter(tokens)
    count_constraints = []

    # word numbers
    for word, num in NUMBER_WORDS.items():
        if word in token_counts:
            for lab in label_to_items.keys():
                # phrase like "two dog" or "two dogs"
                if f"{word} {lab}" in caption_norm or f"{word} {lab}s" in caption_norm:
                    count_constraints.append({
                        "type": "count",
                        "object": lab,
                        "operator": ">=",
                        "value": num,
                    })

    # digit numbers
    for t in tokens:
        if t.isdigit():
            num = int(t)
            for lab in label_to_items.keys():
                if f"{t} {lab}" in caption_norm or f"{t} {lab}s" in caption_norm:
                    count_constraints.append({
                        "type": "count",
                        "object": lab,
                        "operator": ">=",
                        "value": num,
                    })

    # dedup count constraints by (object, operator, value)
    seen = set()
    dedup_counts = []
    for c in count_constraints:
        key = (c["object"], c["operator"], c["value"])
        if key not in seen:
            seen.add(key)
            dedup_counts.append(c)

    if max_count is not None:
        dedup_counts = dedup_counts[:max_count]

    constraints.extend(dedup_counts)

    return constraints
