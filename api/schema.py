from typing import Dict, Any

DIFFERENCE_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "region": {"type": "string"},
        "bbox": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                {"type": "null"}
            ]
        },
        "type": {"type": "string"},
        "description": {"type": "string"},
        "severity": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "explainability": {"type": "array", "items": {"type": "string"}},
        "suggested_action": {"type": "string"},
        "tis_delta": {"type": "integer"},
    },
    "required": ["id", "region", "type", "description", "severity", "confidence", "explainability", "suggested_action", "tis_delta"],
    "additionalProperties": True,
}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "differences": {
            "type": "array",
            "items": DIFFERENCE_ITEM_SCHEMA
        }
    },
    "required": ["differences"],
    "additionalProperties": True,
}

