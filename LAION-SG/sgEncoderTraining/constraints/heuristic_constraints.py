import re
from collections import Counter
from typing import List, Dict, Any, Optional

NUMBER_WORDS = {
    # explicit cardinal numbers
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
    # fuzzy quantifiers we map to approximate counts
    "couple": 2,
    "pair": 2, "pairs": 2,
    "few": 2,
    "several": 3,
    "many": 3,
    "dozen": 12,
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
    - RELATION: subject-predicate-object patterns that are clearly suggested by caption
                (subject and object words near each other with predicate between/near them)
    - COUNT: numeric mentions like 'two dogs', 'three trees', 'couple of people', 'many cars'
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
    # 3) RELATION constraints (looser, token-based matching)
    # -------------------------
    # need item_id -> canonical label map
    id_to_label = {it["item_id"]: canon_label(it["label"]) for it in items}
    rel_constraints: List[Dict[str, Any]] = []

    # helper: positions of a label (singular/plural) in token list
    def _label_positions(label: str) -> List[int]:
        positions = []
        # allow simple plural variants: dog / dogs
        forms = {label}
        if not label.endswith("s"):
            forms.add(label + "s")
        else:
            # crude singularization: dogs -> dog
            forms.add(label[:-1])
        for i, tok in enumerate(tokens):
            if tok in forms:
                positions.append(i)
        return positions

    # build positions for each label once
    label_pos_cache: Dict[str, List[int]] = {}

    def _get_positions(label: str) -> List[int]:
        if label not in label_pos_cache:
            label_pos_cache[label] = _label_positions(label)
        return label_pos_cache[label]

    for rel in relations:
        subj = id_to_label.get(rel["item1"])
        obj = id_to_label.get(rel["item2"])
        pred = _normalize(rel["relation"])
        if subj is None or obj is None:
            continue

        subj_pos = _get_positions(subj)
        obj_pos = _get_positions(obj)
        if not subj_pos or not obj_pos:
            continue

        # tokens that make up the predicate (e.g., "in front of" -> ["in","front","of"])
        pred_tokens = _simple_tokenize(pred)
        pred_set = set(pred_tokens)

        # windowed proximity heuristic: subject and object appear within a small window,
        # and at least one predicate token appears between or very near them.
        found = False
        max_span = 8  # max distance between subj and obj tokens
        for si in subj_pos:
            for oi in obj_pos:
                if abs(si - oi) > max_span:
                    continue
                lo, hi = sorted((si, oi))
                # search region a bit beyond subject/object
                region_start = max(0, lo - 2)
                region_end = min(len(tokens), hi + 3)
                region = tokens[region_start:region_end]
                if any(tok in pred_set for tok in region):
                    found = True
                    break
            if found:
                break

        if found:
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
    # 4) COUNT constraints (looser patterns)
    # -------------------------
    # We now look for number/quantifier tokens followed within a small window by an object label.
    count_constraints: List[Dict[str, Any]] = []

    def _add_count_constraint(obj_label: str, value: int):
        count_constraints.append({
            "type": "count",
            "object": obj_label,
            "operator": ">=",
            "value": value,
        })

    # precompute label forms for quick matching
    label_forms: Dict[str, List[str]] = {}
    for lab in label_to_items.keys():
        forms = [lab]
        if not lab.endswith("s"):
            forms.append(lab + "s")
        else:
            forms.append(lab[:-1])
        label_forms[lab] = forms

    # scan tokens with a small look-ahead window
    window = 4
    for i, tok in enumerate(tokens):
        # numeric word or fuzzy quantifier
        if tok in NUMBER_WORDS:
            num = NUMBER_WORDS[tok]
        elif tok.isdigit():
            num = int(tok)
        else:
            continue

        # look ahead a few tokens for any object label
        lookahead = tokens[i + 1 : i + 1 + window]
        for lab, forms in label_forms.items():
            if any(t in forms for t in lookahead):
                _add_count_constraint(lab, num)

    # dedup count constraints by (object, operator, value)
    seen = set()
    dedup_counts: List[Dict[str, Any]] = []
    for c in count_constraints:
        key = (c["object"], c["operator"], c["value"])
        if key not in seen:
            seen.add(key)
            dedup_counts.append(c)

    if max_count is not None:
        dedup_counts = dedup_counts[:max_count]

    constraints.extend(dedup_counts)

    return constraints
