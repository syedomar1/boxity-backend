from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None
import os
import io
import base64
from typing import Any, Dict, List, Optional, Tuple

# Optional: Pillow for basic image probing
try:
    from PIL import Image, ExifTags
except Exception:  # Pillow may not be installed yet in dev
    Image = None
    ExifTags = None

try:
    import requests
except Exception:
    requests = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional: OpenCV for classical vision fallback
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

app = Flask(__name__)
if CORS is not None:
    # Allow cross-origin requests for the analyzer endpoint during development
    CORS(app, resources={r"/analyze": {"origins": "*"}})

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'


def _load_image_bytes(source: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Loads image bytes and MIME type from a URL or base64 data URI.

    Returns: (bytes|None, mime_type|None)
    """
    if not source:
        return None, None
    # Base64 data URI
    if source.startswith('data:'):
        try:
            header, b64 = source.split(',', 1)
            mime = header.split(';', 1)[0].replace('data:', '') or 'application/octet-stream'
            return base64.b64decode(b64), mime
        except Exception:
            return None, None
    # Heuristic: very long string without data: header is likely base64 (assume jpeg)
    if len(source) > 256 and not source.startswith('http'):
        try:
            return base64.b64decode(source), 'image/jpeg'
        except Exception:
            return None, None
    # Otherwise, treat as URL
    if requests is None:
        return None, None
    try:
        resp = requests.get(source, timeout=20)
        if resp.status_code == 200:
            mime = resp.headers.get('Content-Type', '').split(';')[0] or None
            return resp.content, mime
    except Exception:
        return None, None
    return None, None


def _get_image_info(img_bytes: Optional[bytes]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"resolution": None, "exif_present": False, "camera_make": None, "camera_model": None, "datetime": None}
    if Image is None or not img_bytes:
        return info
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            info["resolution"] = [im.width, im.height]
            exif = getattr(im, "_getexif", lambda: None)()
            if exif:
                info["exif_present"] = True
                inv = {v: k for k, v in ExifTags.TAGS.items()} if ExifTags else {}
                def get_tag(tag_name: str) -> Optional[str]:
                    key = inv.get(tag_name)
                    return str(exif.get(key)) if key in exif else None
                info["camera_make"] = get_tag("Make")
                info["camera_model"] = get_tag("Model")
                info["datetime"] = get_tag("DateTimeOriginal") or get_tag("DateTime")
    except Exception:
        pass
    return info


def _normalize_diff_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure strict schema fields exist with fallbacks
    return {
        "id": str(item.get("id")) if item.get("id") is not None else "diff-unknown",
        "region": item.get("region") or "unknown",
        "bbox": item.get("bbox"),
        "type": item.get("type") or "other",
        "description": item.get("description") or "",
        "severity": item.get("severity") or "LOW",
        "confidence": float(item.get("confidence") or 0.5),
        "explainability": item.get("explainability") or [],
        "suggested_action": item.get("suggested_action") or "Review",
        "tis_delta": int(item.get("tis_delta") or 0),
    }


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _compute_overall(differences: List[Dict[str, Any]]) -> Tuple[int, str]:
    tis = 100
    for d in differences:
        try:
            tis += int(d.get("tis_delta", 0))
        except Exception:
            continue
    tis = _clamp(tis, 0, 100)
    if tis >= 85:
        return tis, "OK"
    if tis >= 70:
        return tis, "REVIEW_REQUIRED"
    return tis, "QUARANTINE"


def _call_gemini(baseline: Tuple[Optional[bytes], Optional[str]], current: Tuple[Optional[bytes], Optional[str]]) -> List[Dict[str, Any]]:
    """Calls Gemini to detect differences and return normalized items.

    Falls back to empty if API/key not configured. The prompt instructs the model to
    return only the differences array in the specified schema.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    baseline_bytes, baseline_mime = baseline
    current_bytes, current_mime = current
    if genai is None or not api_key or not baseline_bytes or not current_bytes:
        return []
    try:
        genai.configure(api_key=api_key)
        generation_config = {"temperature": 0.2, "response_mime_type": "application/json"}
        model = genai.GenerativeModel("gemini-1.5-pro-latest", generation_config=generation_config)
        system_prompt = (
            "You are a multimodal forensic/computer-vision assistant specialized in package integrity analysis.\n"
            "Task: Compare the baseline vs current package photos and detect concrete issues (e.g., dents, scratches, seal tamper, label mismatch, repackaging, stains, color shifts, missing_item, digital_edit).\n"
            "Rules:\n"
            "- Return STRICT JSON: {\"differences\":[...]} with NO prose.\n"
            "- For each difference, include: id, region (human label), bbox [x,y,w,h] normalized 0..1 (omit or null if unsure), type (one of [dent, scratch, seal_tamper, label_mismatch, color_shift, repackaging, stain, missing_item, digital_edit, other]), description (<=2 sentences), severity [LOW,MEDIUM,HIGH,CRITICAL], confidence 0.0-1.0, explainability (1-3 short evidences), suggested_action, tis_delta (integer; follow weight guidance).\n"
            "- Prefer multiple localized regions with bboxes rather than one global note.\n"
            "- Identify open vs closed box state, seal integrity, and label/logo/QR differences.\n"
            "- Only report digital_edit if you see artifacts (resampling grids, copy-move, inconsistent noise, double-JPEG) and mention indicators in explainability.\n"
        )
        baseline_part = {"mime_type": baseline_mime or "image/jpeg", "data": baseline_bytes}
        current_part = {"mime_type": current_mime or "image/jpeg", "data": current_bytes}

        prompt = (
            "Weighting/TIS: dent (-8..-18), scratch (-3..-7), seal_tamper (-25..-45), label_mismatch (-20..-40), digital_edit (-30..-60), missing_item (-100). "
            "Ensure bboxes are normalized. If unsure, set bbox to null and reduce confidence."
        )
        result = model.generate_content([
            system_prompt,
            prompt,
            "Baseline image:", baseline_part,
            "Current image:", current_part,
        ])
        text = (result.text or "").strip()
        # Robust JSON extraction
        import json, re
        if not text:
            return []
        # Remove code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text)
        # If still not a JSON object, try to locate the first {...}
        if not text.lstrip().startswith('{'):
            match = re.search(r"\{[\s\S]*\}", text)
            text = match.group(0) if match else '{"differences": []}'
        payload = json.loads(text)
        items = payload.get("differences", [])
        normalized = [_normalize_diff_item(it) for it in items if isinstance(it, dict)]
        return normalized
    except Exception:
        return []


def _orb_similarity(img1: "cv2.Mat", img2: "cv2.Mat") -> float:
    """Returns a 0..1 similarity score using ORB feature matching (higher = more similar)."""
    try:
        orb = cv2.ORB_create(nfeatures=800)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)
        if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if not matches:
            return 0.0
        distances = [m.distance for m in matches]
        # Normalize by max Hamming distance (256) and invert
        mean_dist = sum(distances) / len(distances)
        return max(0.0, min(1.0, 1.0 - (mean_dist / 256.0)))
    except Exception:
        return 0.0


def _detect_qr_presence(img: "cv2.Mat") -> bool:
    try:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img)
        return bool(data)
    except Exception:
        return False


def _classical_diff_regions(b_bytes: bytes, c_bytes: bytes) -> List[Dict[str, Any]]:
    """OpenCV-based regional difference detection with simple heuristics."""
    if cv2 is None or Image is None:
        return []
    try:
        b_arr = np.frombuffer(b_bytes, dtype=np.uint8)
        c_arr = np.frombuffer(c_bytes, dtype=np.uint8)
        b = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)
        c = cv2.imdecode(c_arr, cv2.IMREAD_COLOR)
        if b is None or c is None:
            return []
        h1, w1 = b.shape[:2]
        c_resized = cv2.resize(c, (w1, h1), interpolation=cv2.INTER_AREA)

        # Similarity to detect completely different objects
        sim = _orb_similarity(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c_resized, cv2.COLOR_BGR2GRAY))

        # Difference image and threshold
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(c_resized, cv2.COLOR_BGR2GRAY)
        blur_b = cv2.GaussianBlur(gray_b, (5, 5), 0)
        blur_c = cv2.GaussianBlur(gray_c, (5, 5), 0)
        diff = cv2.absdiff(blur_b, blur_c)
        _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)

        # Contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[Dict[str, Any]] = []
        img_area = float(w1 * h1)
        top_band = int(0.25 * h1)  # top 25% region for seal detection

        # QR presence change
        qr_base = _detect_qr_presence(b)
        qr_curr = _detect_qr_presence(c_resized)
        if qr_base != qr_curr:
            regions.append(_normalize_diff_item({
                "id": "qr-change",
                "region": "label area",
                "bbox": None,
                "type": "label_mismatch",
                "description": "QR code presence differs between baseline and current image.",
                "severity": "MEDIUM",
                "confidence": 0.8,
                "explainability": ["QR detected mismatch"],
                "suggested_action": "Supervisor review",
                "tis_delta": -22,
            }))

        # Repackaging / different object if similarity is very low
        if sim < 0.15:
            regions.append(_normalize_diff_item({
                "id": "repackaging-1",
                "region": "overall",
                "bbox": None,
                "type": "repackaging",
                "description": "Current image appears to be a different object or repackaged item compared to baseline.",
                "severity": "HIGH",
                "confidence": 0.85,
                "explainability": [f"low ORB similarity: {sim:.2f}"],
                "suggested_action": "Quarantine batch",
                "tis_delta": -35,
            }))

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if area / img_area < 0.005:  # ignore tiny specks <0.5%
                continue
            # Heuristics
            aspect = max(w, h) / max(1.0, min(w, h))
            near_top = y < top_band
            long_thin = aspect > 6.0 and area / img_area < 0.05
            region_label = "top region" if near_top else (
                "left side" if x < w1 * 0.33 else ("right side" if x + w > w1 * 0.66 else "center"))
            bbox = [x / w1, y / h1, w / w1, h / h1]

            if near_top and h > 0.08 * h1 and w > 0.25 * w1:
                diff_item = {
                    "id": f"seal-{i}",
                    "region": "center seal" if x + w/2 > w1*0.33 and x + w/2 < w1*0.66 else "top edge",
                    "bbox": bbox,
                    "type": "seal_tamper",
                    "description": "Seal area shows opening/gap inconsistent with baseline.",
                    "severity": "HIGH",
                    "confidence": 0.78,
                    "explainability": ["large change along top seam", "thresholded contour"],
                    "suggested_action": "Supervisor review",
                    "tis_delta": -32,
                }
            elif long_thin:
                diff_item = {
                    "id": f"scratch-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "scratch",
                    "description": "Linear high-contrast mark consistent with scratch.",
                    "severity": "LOW",
                    "confidence": 0.7,
                    "explainability": ["long thin contour", "edge contrast"],
                    "suggested_action": "Proceed",
                    "tis_delta": -4,
                }
            else:
                diff_item = {
                    "id": f"dent-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "dent",
                    "description": "Localized deformation/region change likely indicating a dent or indentation.",
                    "severity": "MEDIUM" if area / img_area < 0.08 else "HIGH",
                    "confidence": 0.72,
                    "explainability": ["blob-like contour", "local intensity change"],
                    "suggested_action": "Supervisor review" if area / img_area >= 0.08 else "Proceed",
                    "tis_delta": -12 if area / img_area < 0.08 else -18,
                }
            regions.append(_normalize_diff_item(diff_item))

        # Merge/limit to top 5 by severity area
        return regions[:5]
    except Exception:
        return []


@app.route('/analyze', methods=['POST'])
def analyze():
    """POST JSON: { baseline_url|baseline_b64, current_url|current_b64 }

    Returns single JSON object following the requested schema.
    """
    try:
        data = request.get_json(silent=True) or {}
        baseline_src = data.get('baseline_url') or data.get('baseline_b64')
        current_src = data.get('current_url') or data.get('current_b64')
        baseline_bytes, baseline_mime = _load_image_bytes(baseline_src)
        current_bytes, current_mime = _load_image_bytes(current_src)

        baseline_info = _get_image_info(baseline_bytes)
        current_info = _get_image_info(current_bytes)

        differences = _call_gemini((baseline_bytes, baseline_mime), (current_bytes, current_mime))

        # Classical CV fallback for richer localized differences
        if (not differences or sum(d.get("confidence", 0) for d in differences) / max(1, len(differences)) < 0.6) and baseline_bytes and current_bytes:
            cv_regions = _classical_diff_regions(baseline_bytes, current_bytes)
            if cv_regions:
                # If Gemini returned something, merge distinct regions by ID
                if differences:
                    seen = {d.get("id") for d in differences}
                    for r in cv_regions:
                        if r.get("id") not in seen:
                            differences.append(r)
                else:
                    differences = cv_regions
        tis, assessment = _compute_overall(differences)

        # Heuristic confidence overall
        conf_overall = 0.9 if differences else 0.7

        response: Dict[str, Any] = {
            "differences": differences,
            "baseline_image_info": baseline_info,
            "current_image_info": current_info,
            "aggregate_tis": tis,
            "overall_assessment": assessment,
            "confidence_overall": conf_overall,
            "notes": (
                "No significant issues detected." if not differences else
                "Detected potential integrity issues; follow suggested actions."
            ),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "differences": [],
            "baseline_image_info": {"resolution": None, "exif_present": False, "camera_make": None, "camera_model": None, "datetime": None},
            "current_image_info": {"resolution": None, "exif_present": False, "camera_make": None, "camera_model": None, "datetime": None},
            "aggregate_tis": 100,
            "overall_assessment": "OK",
            "confidence_overall": 0.5,
            "notes": "Analyzer error; defaulting to OK.",
            "error": str(e),
        }), 200