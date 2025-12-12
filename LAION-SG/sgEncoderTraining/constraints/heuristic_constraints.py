"""
Heuristic constraint extraction from caption + scene graph.

This module provides functions to extract constraints that reflect what the
caption explicitly mentions, making them suitable for training constraint-aware
image generation models.
"""

import re
from collections import Counter
from typing import List, Dict, Any, Optional, Set, Tuple

# Cardinal numbers and fuzzy quantifiers
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10,
    "couple": 2, "pair": 2, "pairs": 2,
    "few": 2, "several": 3, "many": 3, "dozen": 12,
}

# Common spatial / action prepositions for relation mining
RELATION_WORDS = {
    # spatial
    "on", "in", "at", "by", "near", "beside", "behind", "under", "above",
    "below", "over", "inside", "outside", "between", "among", "through",
    "across", "along", "around", "against", "towards", "toward",
    "next", "front", "top", "bottom", "left", "right", "side",
    "sitting", "standing", "lying", "walking", "running", "riding",
    "holding", "wearing", "eating", "playing", "watching", "looking",
    "facing", "leaning", "hanging", "flying", "parked", "placed",
}

# Common visual attributes (colors, materials, states) for caption mining
COMMON_ATTRIBUTES = {
    # colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "gold", "silver", "beige", "tan",
    # materials
    "wooden", "wood", "metal", "glass", "plastic", "leather", "fabric",
    "stone", "brick", "concrete", "ceramic",
    # sizes
    "big", "large", "small", "tiny", "tall", "short", "long", "wide", "narrow",
    # states
    "old", "new", "clean", "dirty", "wet", "dry", "open", "closed", "empty",
    "full", "broken", "worn", "bright", "dark", "shiny", "dull",
    # textures/patterns
    "striped", "spotted", "plaid", "checkered", "floral", "plain",
}

def _normalize(text: str) -> str:
    return text.lower().strip()

def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def _get_label_forms(label: str) -> Set[str]:
    """Return singular and simple plural forms of a label."""
    forms = {label}
    if not label.endswith("s"):
        forms.add(label + "s")
    elif label.endswith("ss"):
        forms.add(label + "es")  # glass -> glasses
    else:
        forms.add(label[:-1])  # dogs -> dog

    if label.endswith("y") and len(label) > 2 and label[-2] not in "aeiou":
        forms.add(label[:-1] + "ies")
    return forms


def _find_label_positions(tokens: List[str], label: str) -> List[int]:
    """Find all token indices where the label (or its plural) appears."""
    forms = _get_label_forms(label)
    return [i for i, tok in enumerate(tokens) if tok in forms]

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
    Extract constraints from caption + scene graph.
    Returns: List of constraint dicts with types: presence, attribute, relation, count
    """
    caption_norm = " " + _normalize(caption) + " "
    tokens = _simple_tokenize(caption)
    token_set = set(tokens)
    
    alias_map = alias_map or {}
    
    def canon_label(label: str) -> str:
        lab = _normalize(label)
        return alias_map.get(lab, lab)
    
    # Build label to items mapping
    label_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        lab = canon_label(it["label"])
        label_to_items.setdefault(lab, []).append(it)
    
    # Precompute label positions in tokens
    label_positions: Dict[str, List[int]] = {}
    labels_in_caption: Set[str] = set()
    for lab in label_to_items.keys():
        positions = _find_label_positions(tokens, lab)
        label_positions[lab] = positions
        if positions:
            labels_in_caption.add(lab)
    
    constraints: List[Dict[str, Any]] = []
    seen_constraints: Set[Tuple] = set()  # for deduplication
    
    def _add_constraint(c: Dict[str, Any]) -> bool:
        """Add constraint if not duplicate. Returns True if added."""
        if c["type"] == "presence":
            key = ("presence", c["object"], c["polarity"])
        elif c["type"] == "attribute":
            key = ("attribute", c["object"], c["attribute"])
        elif c["type"] == "relation":
            key = ("relation", c["subject"], c["predicate"], c["object"])
        elif c["type"] == "count":
            key = ("count", c["object"], c["operator"], c["value"])
        else:
            return False
        
        if key not in seen_constraints:
            seen_constraints.add(key)
            constraints.append(c)
            return True
        return False
    

    # presence constraints
    presence_count = 0
    for lab in labels_in_caption:
        if max_presence is not None and presence_count >= max_presence:
            break
        if _add_constraint({"type": "presence", "object": lab, "polarity": "positive"}):
            presence_count += 1
    
    # att constraints
    attr_count = 0
    
    # From SG attributes that appear in caption
    for it in items:
        if max_attribute is not None and attr_count >= max_attribute:
            break
        lab = canon_label(it["label"])
        if lab not in labels_in_caption:
            continue
        for attr in it.get("attributes", []):
            if max_attribute is not None and attr_count >= max_attribute:
                break
            attr_norm = _normalize(attr)
            if attr_norm in token_set:
                if _add_constraint({"type": "attribute", "object": lab, "attribute": attr_norm}):
                    attr_count += 1
    
    # Caption-mined attributes: common visual attributes near object labels
    for lab in labels_in_caption:
        if max_attribute is not None and attr_count >= max_attribute:
            break
        lab_pos = label_positions[lab]
        for pos in lab_pos:
            if max_attribute is not None and attr_count >= max_attribute:
                break
            # Look in a window around the object mention
            window_start = max(0, pos - 3)
            window_end = min(len(tokens), pos + 2)
            window_tokens = tokens[window_start:window_end]
            for tok in window_tokens:
                if tok in COMMON_ATTRIBUTES:
                    if _add_constraint({"type": "attribute", "object": lab, "attribute": tok}):
                        attr_count += 1
    
    # relation constraints
   
    id_to_label = {it["item_id"]: canon_label(it["label"]) for it in items}
    rel_count = 0
    
    # From SG relations that are supported by caption
    for rel in relations:
        if max_relation is not None and rel_count >= max_relation:
            break
        subj = id_to_label.get(rel["item1"])
        obj = id_to_label.get(rel["item2"])
        pred = _normalize(rel["relation"])
        if subj is None or obj is None:
            continue
        
        subj_pos = label_positions.get(subj, [])
        obj_pos = label_positions.get(obj, [])
        if not subj_pos or not obj_pos:
            continue
        
        pred_tokens = set(_simple_tokenize(pred))
        
        # Check if subj and obj appear close together with pred nearby
        max_span = 10
        found = False
        for si in subj_pos:
            for oi in obj_pos:
                if abs(si - oi) > max_span:
                    continue
                lo, hi = min(si, oi), max(si, oi)
                region_start = max(0, lo - 2)
                region_end = min(len(tokens), hi + 3)
                region = set(tokens[region_start:region_end])
                if pred_tokens & region:
                    found = True
                    break
            if found:
                break
        
        if found:
            if _add_constraint({"type": "relation", "subject": subj, "predicate": pred, "object": obj}):
                rel_count += 1
    
    # Caption-mined relations: pairs of objects with spatial words between
    # This catches relations mentioned in caption but not in SG
    labels_list = list(labels_in_caption)
    for i, lab1 in enumerate(labels_list):
        if max_relation is not None and rel_count >= max_relation:
            break
        for lab2 in labels_list[i+1:]:
            if max_relation is not None and rel_count >= max_relation:
                break
            if lab1 == lab2:
                continue
            
            pos1_list = label_positions[lab1]
            pos2_list = label_positions[lab2]
            
            for p1 in pos1_list:
                if max_relation is not None and rel_count >= max_relation:
                    break
                for p2 in pos2_list:
                    if abs(p1 - p2) > 8:
                        continue
                    
                    lo, hi = min(p1, p2), max(p1, p2)
                    between = tokens[lo:hi+1]
                    
                    # Find relation words between them
                    found_rels = [t for t in between if t in RELATION_WORDS]
                    if found_rels:
                        # Use the first relation word found
                        pred = found_rels[0]
                        # Handle compound prepositions
                        if pred == "next" and "to" in between:
                            pred = "next to"
                        elif pred == "front" and "of" in between:
                            pred = "in front of"
                        elif pred == "top" and "of" in between:
                            pred = "on top of"
                        
                        # Determine direction (which is subject, which is object)
                        if p1 < p2:
                            subj, obj = lab1, lab2
                        else:
                            subj, obj = lab2, lab1
                        
                        if _add_constraint({"type": "relation", "subject": subj, "predicate": pred, "object": obj}):
                            rel_count += 1
                        break
    
    # COUNT constraints
   
    count_count = 0
    
    # Build label forms for matching
    label_forms_map: Dict[str, Set[str]] = {
        lab: _get_label_forms(lab) for lab in label_to_items.keys()
    }
    
    window = 4
    for i, tok in enumerate(tokens):
        if max_count is not None and count_count >= max_count:
            break
        
        # Check if token is a number
        if tok in NUMBER_WORDS:
            num = NUMBER_WORDS[tok]
        elif tok.isdigit():
            num = int(tok)
        else:
            continue
        
        # Look ahead for object labels
        lookahead = tokens[i + 1 : i + 1 + window]
        for lab, forms in label_forms_map.items():
            if any(t in forms for t in lookahead):
                if _add_constraint({"type": "count", "object": lab, "operator": ">=", "value": num}):
                    count_count += 1
    
    return constraints
