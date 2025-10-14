import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import RESPONSE_SCHEMA

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from jsonschema import validate, ValidationError
except Exception:
    validate = None
    ValidationError = Exception


def _configure_genai():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return False
    genai.configure(api_key=api_key)
    return True


FEW_SHOT = (
    "Return STRICT JSON as {\"differences\":[...]}. Example:\n"
    "{\n  \"differences\": [\n    {\n      \"id\": \"d1\", \"region\": \"top seam\", \"bbox\": [0.12,0.03,0.76,0.08], \"type\": \"seal_tamper\",\n"
    "      \"description\": \"Seal gap visible with lifted flap.\", \"severity\": \"HIGH\", \"confidence\": 0.84,\n"
    "      \"explainability\": [\"gap at seam\", \"edge discontinuity\"], \"suggested_action\": \"Supervisor review\", \"tis_delta\": -32\n"
    "    },\n    {\n      \"id\": \"d2\", \"region\": \"left side\", \"bbox\": [0.06,0.42,0.18,0.12], \"type\": \"dent\",\n"
    "      \"description\": \"Concave deformation on side panel.\", \"severity\": \"MEDIUM\", \"confidence\": 0.78,\n"
    "      \"explainability\": [\"shading collapse\", \"curvature change\"], \"suggested_action\": \"Supervisor review\", \"tis_delta\": -12\n"
    "    }\n  ]\n}"
)


def _build_model(name: str):
    generation_config = {
        "temperature": 0.15,
        "top_k": 20,
        "top_p": 0.8,
        "response_mime_type": "application/json",
    }
    return genai.GenerativeModel(name, generation_config=generation_config)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"differences": []}
    if text.startswith("```)":
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text)
    if not text.lstrip().startswith('{'):
        match = re.search(r"\{[\s\S]*\}", text)
        text = match.group(0) if match else '{"differences": []}'
    try:
        return json.loads(text)
    except Exception:
        return {"differences": []}


def _validate_or_repair(payload: Dict[str, Any], model) -> Dict[str, Any]:
    if validate is None:
        return payload
    try:
        validate(instance=payload, schema=RESPONSE_SCHEMA)
        return payload
    except ValidationError:
        # Ask model to repair to match the schema
        try:
            result = model.generate_content([
                "Repair this JSON to match the schema {differences:[...] with required fields}:",
                json.dumps(payload)
            ])
            repaired = _extract_json(result.text or "")
            validate(instance=repaired, schema=RESPONSE_SCHEMA)
            return repaired
        except Exception:
            return {"differences": []}


def call_gemini_ensemble(baseline: Tuple[Optional[bytes], Optional[str]], current: Tuple[Optional[bytes], Optional[str]]) -> List[Dict[str, Any]]:
    if not _configure_genai():
        return []

    baseline_bytes, baseline_mime = baseline
    current_bytes, current_mime = current
    if not baseline_bytes or not current_bytes:
        return []

    system = (
        "You are a multimodal forensic/computer-vision assistant specialized in package integrity analysis.\n"
        "Task: Compare baseline vs current package photos and detect concrete issues (dent, scratch, seal_tamper, label_mismatch, repackaging, stain, color_shift, missing_item, digital_edit).\n"
        "Rules:\n- Return STRICT JSON: {\"differences\":[...]} with NO prose.\n- Prefer multiple localized regions with realistic bboxes.\n- If confidence is high and changes are localized, avoid a single \"overall\" item.\n"
        "- Only report digital_edit if you see artifacts; list indicators in explainability.\n"
        + FEW_SHOT
    )

    parts = [
        system,
        "Weighting/TIS guidance: dent (-8..-18), scratch (-3..-7), seal_tamper (-25..-45), label_mismatch (-20..-40), digital_edit (-30..-60), missing_item (-100).",
        "Ensure bbox is [x,y,w,h] in 0..1, or null when unsure.",
        "Baseline:", {"mime_type": baseline_mime or "image/jpeg", "data": baseline_bytes},
        "Current:", {"mime_type": current_mime or "image/jpeg", "data": current_bytes},
    ]

    model_pro = _build_model("gemini-1.5-pro-latest")
    model_flash = _build_model("gemini-1.5-flash-latest")

    try:
        r1 = model_pro.generate_content(parts)
        r2 = model_flash.generate_content(parts)
        p1 = _extract_json(r1.text or "")
        p2 = _extract_json(r2.text or "")
        v1 = _validate_or_repair(p1, model_pro)
        v2 = _validate_or_repair(p2, model_pro)

        list1 = v1.get("differences", [])
        list2 = v2.get("differences", [])

        # Merge: keep items with matching region/type (rough consensus) first
        merged: List[Dict[str, Any]] = []
        def key(d: Dict[str, Any]) -> Tuple[str, str]:
            return (str(d.get("region") or ""), str(d.get("type") or ""))

        seen = set()
        for item in list1 + list2:
            k = key(item)
            if k in seen:
                continue
            seen.add(k)
            merged.append(item)

        return merged[:8]
    except Exception:
        return []

