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
    "{\n  \"differences\": [\n    {\n      \"id\": \"d1\", \"region\": \"top edge\", \"bbox\": [0.12,0.03,0.76,0.08], \"type\": \"seal_tamper\",\n"
    "      \"description\": \"Seal gap visible with lifted flap indicating potential tampering.\", \"severity\": \"HIGH\", \"confidence\": 0.84,\n"
    "      \"explainability\": [\"gap at seam\", \"edge discontinuity\", \"lifted flap\"], \"suggested_action\": \"Immediate quarantine\", \"tis_delta\": -40\n"
    "    },\n    {\n      \"id\": \"d2\", \"region\": \"left side\", \"bbox\": [0.06,0.42,0.18,0.12], \"type\": \"dent\",\n"
    "      \"description\": \"Concave deformation on left side panel suggesting impact damage.\", \"severity\": \"MEDIUM\", \"confidence\": 0.78,\n"
    "      \"explainability\": [\"shading collapse\", \"curvature change\", \"impact pattern\"], \"suggested_action\": \"Supervisor review\", \"tis_delta\": -15\n"
    "    },\n    {\n      \"id\": \"d3\", \"region\": \"right side\", \"bbox\": [0.75,0.35,0.15,0.25], \"type\": \"scratch\",\n"
    "      \"description\": \"Linear scratch mark on right side panel.\", \"severity\": \"LOW\", \"confidence\": 0.72,\n"
    "      \"explainability\": [\"linear mark\", \"surface abrasion\", \"edge contrast\"], \"suggested_action\": \"Proceed\", \"tis_delta\": -8\n"
    "    },\n    {\n      \"id\": \"d4\", \"region\": \"front panel\", \"bbox\": [0.2,0.1,0.6,0.2], \"type\": \"label_mismatch\",\n"
    "      \"description\": \"Label appears altered or replaced with different product information.\", \"severity\": \"HIGH\", \"confidence\": 0.82,\n"
    "      \"explainability\": [\"text mismatch\", \"font difference\", \"color variation\"], \"suggested_action\": \"Quarantine batch\", \"tis_delta\": -40\n"
    "    },\n    {\n      \"id\": \"d5\", \"region\": \"top-left corner\", \"bbox\": [0.0,0.0,0.15,0.15], \"type\": \"dent\",\n"
    "      \"description\": \"Corner damage detected in top-left area.\", \"severity\": \"MEDIUM\", \"confidence\": 0.75,\n"
    "      \"explainability\": [\"corner deformation\", \"impact damage\", \"structural change\"], \"suggested_action\": \"Supervisor review\", \"tis_delta\": -15\n"
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
    if text.startswith("```"):
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
        "You are an expert multimodal forensic analyst specializing in package integrity and tampering detection.\n"
        "\nMISSION: Compare baseline vs current package photos to detect security breaches and integrity violations.\n"
        "\nDETECTION TARGETS:\n"
        "- seal_tamper: Broken, lifted, or altered seals (CRITICAL SECURITY RISK)\n"
        "- repackaging: Different packaging, missing elements, or structural changes\n"
        "- label_mismatch: Altered, replaced, or counterfeit labels\n"
        "- digital_edit: Photo manipulation, cloning, or artificial modifications\n"
        "- dent: Physical damage from impact or compression\n"
        "- scratch: Surface abrasions or cuts\n"
        "- stain: Discoloration or contamination\n"
        "- color_shift: Significant color changes indicating tampering\n"
        "- missing_item: Absent components or contents\n"
        "\nREGION SPECIFICATION:\n"
        "Be VERY specific about damage locations:\n"
        "- 'left side': Left edge/panel of the package\n"
        "- 'right side': Right edge/panel of the package\n"
        "- 'top edge': Upper portion/seal area\n"
        "- 'bottom edge': Lower portion/base\n"
        "- 'front panel': Main visible surface\n"
        "- 'back panel': Rear surface\n"
        "- 'corner': Specific corner (top-left, top-right, etc.)\n"
        "- 'center': Middle area of package\n"
        "\nANALYSIS RULES:\n"
        "1. Return STRICT JSON: {\"differences\":[...]} with NO additional text\n"
        "2. Focus on security-critical issues first (seal_tamper, repackaging, digital_edit)\n"
        "3. Provide precise bbox coordinates [x,y,w,h] in 0..1 range\n"
        "4. Use HIGH severity for security breaches, MEDIUM for damage, LOW for minor issues\n"
        "5. Confidence should reflect certainty: >0.8 for clear evidence, 0.6-0.8 for likely, <0.6 for uncertain\n"
        "6. Explainability must list specific visual evidence\n"
        "7. TIS delta: seal_tamper(-40), repackaging(-35), digital_edit(-50), label_mismatch(-40), dent(-15), scratch(-8), others(-5)\n"
        "8. ALWAYS specify exact region (left side, right side, top edge, etc.) - never use generic terms\n"
        "\n" + FEW_SHOT
    )

    parts = [
        system,
        "\nCRITICAL: Focus on security threats. A single seal_tamper or digital_edit should trigger immediate quarantine.\n"
        "Be conservative with confidence scores - only use >0.8 when evidence is unequivocal.\n"
        "\nBaseline Image (Reference):", {"mime_type": baseline_mime or "image/jpeg", "data": baseline_bytes},
        "\nCurrent Image (Under Analysis):", {"mime_type": current_mime or "image/jpeg", "data": current_bytes},
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

