"""
Constraint definitions and utilities for GNN constraint nodes.

Constraints are special nodes injected into the scene graph that encode
user requirements (object presence, attributes, relations, counts).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import random


class ConstraintType(Enum):
    PRESENCE = "presence"       # Object must/must not exist
    ATTRIBUTE = "attribute"     # Object must have attribute
    RELATION = "relation"       # Relation must exist between objects
    COUNT = "count"             # Object count requirement


@dataclass
class PresenceConstraint:
    """Constraint: image must contain this object or not"""
    object: str
    polarity: Literal["positive", "negative"] = "positive"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": ConstraintType.PRESENCE.value,
            "object": self.object,
            "polarity": self.polarity
        }
    
    def to_triple(self) -> Dict[str, str]:
        """Convert to sgEncoder-compatible triple format"""
        if self.polarity == "positive":
            return {
                "item1": "[MUST_HAVE]",
                "relation": "requires",
                "item2": self.object
            }
        else:
            return {
                "item1": "[MUST_NOT_HAVE]",
                "relation": "excludes",
                "item2": self.object
            }


@dataclass
class AttributeConstraint:
    """Constraint: object must have this attribute"""
    object: str
    attribute: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": ConstraintType.ATTRIBUTE.value,
            "object": self.object,
            "attribute": self.attribute
        }
    
    def to_triple(self) -> Dict[str, str]:
        """Convert to sgEncoder-compatible triple format"""
        return {
            "item1": "[ATTR]",
            "relation": "describes",
            "item2": f"{self.attribute} {self.object}"
        }


@dataclass  
class RelationConstraint:
    """Constraint: this relation must exist."""
    subject: str
    predicate: str
    object: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": ConstraintType.RELATION.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object
        }
    
    def to_triple(self) -> Dict[str, str]:
        """Convert to sgEncoder-compatible triple format."""
        return {
            "item1": self.subject,
            "relation": f"[MUST_BE] {self.predicate}",
            "item2": self.object
        }


@dataclass
class CountConstraint:
    """Constraint: object count requirement."""
    object: str
    operator: Literal[">=", "<=", "=="]
    value: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": ConstraintType.COUNT.value,
            "object": self.object,
            "operator": self.operator,
            "value": self.value
        }
    
    def to_triple(self) -> Dict[str, str]:
        """Convert to sgEncoder-compatible triple format."""
        op_text = {
            ">=": "at_least",
            "<=": "at_most",
            "==": "exactly"
        }[self.operator]
        return {
            "item1": f"[COUNT_{op_text}_{self.value}]",
            "relation": "requires",
            "item2": self.object
        }


def parse_constraint(constraint_dict: Dict[str, Any]):
    """Parse a constraint dictionary into a Constraint object."""
    ctype = constraint_dict.get("type")
    
    if ctype == ConstraintType.PRESENCE.value:
        return PresenceConstraint(
            object=constraint_dict["object"],
            polarity=constraint_dict.get("polarity", "positive")
        )
    elif ctype == ConstraintType.ATTRIBUTE.value:
        return AttributeConstraint(
            object=constraint_dict["object"],
            attribute=constraint_dict["attribute"]
        )
    elif ctype == ConstraintType.RELATION.value:
        return RelationConstraint(
            subject=constraint_dict["subject"],
            predicate=constraint_dict["predicate"],
            object=constraint_dict["object"]
        )
    elif ctype == ConstraintType.COUNT.value:
        return CountConstraint(
            object=constraint_dict["object"],
            operator=constraint_dict["operator"],
            value=constraint_dict["value"]
        )
    else:
        raise ValueError(f"Unknown constraint type: {ctype}")


def constraints_to_triples(constraints: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert a list of constraint dicts to sgEncoder-compatible triples."""
    triples = []
    for c_dict in constraints:
        constraint = parse_constraint(c_dict)
        triples.append(constraint.to_triple())
    return triples

def generate_constraints_from_scene_graph(
    items: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    max_presence: int = 2,
    max_attribute: int = 2,
    max_relation: int = 1,
    include_count: bool = True,
    seed: Optional[int] = None
):
    """
    Auto-generate constraints from an existing scene graph.
    
    """
    if seed is not None:
        random.seed(seed)
    
    constraints = []
    item_dict = {item["item_id"]: item for item in items}
    
    # 1. Presence constraints 
    if items and max_presence > 0:
        sampled_items = random.sample(items, min(max_presence, len(items)))
        for item in sampled_items:
            constraints.append(
                PresenceConstraint(object=item["label"]).to_dict()
            )
    
    # 2. Attribute constraints 
    items_with_attrs = [i for i in items if i.get("attributes")]
    if items_with_attrs and max_attribute > 0:
        sampled = random.sample(items_with_attrs, min(max_attribute, len(items_with_attrs)))
        for item in sampled:
            attr = random.choice(item["attributes"])
            constraints.append(
                AttributeConstraint(object=item["label"], attribute=attr).to_dict()
            )
    
    # 3. Relation constraints 
    if relations and max_relation > 0:
        sampled_rels = random.sample(relations, min(max_relation, len(relations)))
        for rel in sampled_rels:
            subj = item_dict.get(rel["item1"], {}).get("label", "object")
            obj = item_dict.get(rel["item2"], {}).get("label", "object")
            constraints.append(
                RelationConstraint(
                    subject=subj,
                    predicate=rel["relation"],
                    object=obj
                ).to_dict()
            )
    
    # 4. Count constraints
    if include_count:
        label_counts = {}
        for item in items:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Only create count constraints for objects appearing 2+ times
        multi_objects = [(label, count) for label, count in label_counts.items() if count >= 2]
        if multi_objects:
            label, count = random.choice(multi_objects)
            constraints.append(
                CountConstraint(object=label, operator=">=", value=count).to_dict()
            )
    
    return constraints


# Special tokens for constraint nodes (for reference/embedding initialization)
CONSTRAINT_SPECIAL_TOKENS = [
    "[MUST_HAVE]",
    "[MUST_NOT_HAVE]",
    "[ATTR]",
    "[MUST_BE]",
    "[COUNT_at_least_2]",
    "[COUNT_at_least_3]",
    "[COUNT_at_most_2]",
    "[COUNT_exactly_1]",
    "[COUNT_exactly_2]",
]

