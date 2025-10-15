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

ANALYSIS_METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_differences": {"type": "integer", "minimum": 0},
        "high_severity_count": {"type": "integer", "minimum": 0},
        "medium_severity_count": {"type": "integer", "minimum": 0},
        "low_severity_count": {"type": "integer", "minimum": 0},
        "analysis_timestamp": {"type": "string"}
    },
    "required": ["total_differences", "high_severity_count", "medium_severity_count", "low_severity_count", "analysis_timestamp"],
    "additionalProperties": True,
}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "differences": {
            "type": "array",
            "items": DIFFERENCE_ITEM_SCHEMA
        },
        "aggregate_tis": {"type": "integer", "minimum": 0, "maximum": 100},
        "overall_assessment": {"type": "string", "enum": ["SAFE", "MODERATE_RISK", "HIGH_RISK"]},
        "confidence_overall": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "notes": {"type": "string"},
        "baseline_image_info": {"type": "object"},
        "current_image_info": {"type": "object"},
        "analysis_metadata": ANALYSIS_METADATA_SCHEMA
    },
    "required": ["differences", "aggregate_tis", "overall_assessment", "confidence_overall", "notes"],
    "additionalProperties": True,
}

