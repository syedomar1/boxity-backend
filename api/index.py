# api/index.py (improved logging & runtime checks)
import os
import sys
import traceback
import json
import io
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify

# CORS
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# Pillow
try:
    from PIL import Image, ExifTags
except Exception:
    Image = None
    ExifTags = None

# requests
try:
    import requests
except Exception:
    requests = None

# google generative ai library
try:
    import google.generativeai as genai
except Exception:
    genai = None

# modular helpers (ai / vision)
try:
    from .ai import call_gemini_ensemble
except Exception as e:
    call_gemini_ensemble = None
    print("AI helper import failed:", e, file=sys.stderr)

try:
    from .vision import align_and_normalize
except Exception as e:
    align_and_normalize = None
    print("Vision helper import failed:", e, file=sys.stderr)

# opencv / numpy may be heavy -> check
try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None
    print("cv2 import failed:", str(e), file=sys.stderr)

try:
    import numpy as np  # type: ignore
except Exception as e:
    np = None
    print("numpy import failed:", str(e), file=sys.stderr)

app = Flask(__name__)
if CORS is not None:
    CORS(app, resources={r"/analyze": {"origins": "*"}})

def _configure_genai():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GOOGLE_API_KEY/GEMINI_API_KEY set in environment", file=sys.stderr)
        return False
    if genai is None:
        print("google.generativeai not installed or failed to import", file=sys.stderr)
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print("genai.configure error:", str(e), file=sys.stderr)
        return False

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


def _compute_overall(differences: List[Dict[str, Any]]) -> Tuple[int, str, float, str]:
    """Enhanced TIS calculation with proper difference weighting.
    
    Returns: (tis_score, assessment, confidence, notes)
    """
    if not differences:
        return 100, "SAFE", 0.95, "No differences detected - product integrity maintained"
    
    # Start with perfect score
    tis = 100
    total_confidence = 0.0
    severity_weights = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
    critical_issues = []
    high_severity_count = 0
    medium_severity_count = 0
    
    for d in differences:
        try:
            # Apply TIS delta - this should reduce the score
            tis_delta = int(d.get("tis_delta", 0))
            tis += tis_delta  # tis_delta is negative, so this reduces the score
            
            # Weight confidence by severity
            severity = str(d.get("severity", "LOW")).upper()
            weight = severity_weights.get(severity, 0.3)
            confidence = float(d.get("confidence", 0.5))
            total_confidence += confidence * weight
            
            # Count severities
            if severity == "HIGH":
                high_severity_count += 1
            elif severity == "MEDIUM":
                medium_severity_count += 1
            
            # Track critical issues
            if severity == "HIGH" and confidence > 0.6:
                issue_type = str(d.get("type", "unknown"))
                if issue_type in ["seal_tamper", "repackaging", "digital_edit"]:
                    critical_issues.append(issue_type)
                    
        except Exception:
            continue
    
    # Calculate weighted confidence
    avg_confidence = total_confidence / max(1, len(differences)) if differences else 0.0
    
    # Clamp TIS score to ensure it's never above 100
    tis = _clamp(tis, 0, 100)
    
    # Enhanced assessment logic based on actual TIS score
    if tis >= 80:
        assessment = "SAFE"
        notes = "Product integrity maintained - safe to proceed"
    elif tis >= 40:
        assessment = "MODERATE_RISK"
        notes = "Moderate risk detected - supervisor review recommended"
    else:
        assessment = "HIGH_RISK"
        notes = "High risk detected - immediate quarantine required"
    
    # Additional overrides for critical security issues
    if critical_issues:
        if "seal_tamper" in critical_issues:
            tis = min(tis, 20)  # Force very high risk for seal tampering
            assessment = "HIGH_RISK"
            notes = f"Critical security breach detected: {', '.join(critical_issues)} - immediate quarantine required"
        elif "repackaging" in critical_issues:
            tis = min(tis, 15)  # Force highest risk for repackaging
            assessment = "HIGH_RISK"
            notes = f"Product substitution detected: {', '.join(critical_issues)} - immediate quarantine required"
        elif "digital_edit" in critical_issues:
            tis = min(tis, 10)  # Force highest risk for digital tampering
            assessment = "HIGH_RISK"
            notes = "Digital tampering detected - highest security risk"
    
    # Additional logic for multiple high-severity issues
    if high_severity_count >= 2:
        tis = min(tis, 30)
        assessment = "HIGH_RISK"
        notes = f"Multiple high-severity issues detected ({high_severity_count} issues) - immediate quarantine required"
    elif high_severity_count >= 1 and medium_severity_count >= 2:
        tis = min(tis, 35)
        assessment = "HIGH_RISK"
        notes = f"Multiple damage issues detected - immediate quarantine required"
    
    return tis, assessment, avg_confidence, notes


def _call_gemini(baseline: Tuple[Optional[bytes], Optional[str]], current: Tuple[Optional[bytes], Optional[str]]) -> List[Dict[str, Any]]:
    if call_gemini_ensemble is None:
        return []
    try:
        items = call_gemini_ensemble(baseline, current)
        return [_normalize_diff_item(it) for it in items if isinstance(it, dict)]
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
    """Enhanced OpenCV-based regional difference detection with advanced heuristics."""
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

        # Enhanced similarity analysis
        sim = _orb_similarity(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c_resized, cv2.COLOR_BGR2GRAY))
        
        # Multi-scale difference detection
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.cvtColor(c_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple blur levels for different detail levels
        blur_levels = [(3, 3), (5, 5), (7, 7)]
        all_diffs = []
        
        for blur_size in blur_levels:
            blur_b = cv2.GaussianBlur(gray_b, blur_size, 0)
            blur_c = cv2.GaussianBlur(gray_c, blur_size, 0)
            diff = cv2.absdiff(blur_b, blur_c)
            all_diffs.append(diff)
        
        # Combine differences from multiple scales
        combined_diff = np.maximum.reduce(all_diffs)
        
        # Enhanced thresholding with adaptive methods
        _, th_otsu = cv2.threshold(combined_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th_adaptive = cv2.threshold(combined_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Combine both thresholding methods
        th = cv2.bitwise_or(th_otsu, th_adaptive)
        
        # Enhanced morphological operations
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        # Remove noise
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Fill gaps
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        # Dilate to connect nearby regions
        th = cv2.dilate(th, kernel_medium, iterations=1)

        # Enhanced contour detection
        contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[Dict[str, Any]] = []
        img_area = float(w1 * h1)
        
        # Define regions of interest
        top_band = int(0.25 * h1)  # top 25% for seal detection
        center_region = (int(0.2 * w1), int(0.2 * h1), int(0.6 * w1), int(0.6 * h1))  # center 60% for main content
        
        # Calculate image quality metrics
        baseline_sharpness = cv2.Laplacian(gray_b, cv2.CV_64F).var()
        current_sharpness = cv2.Laplacian(gray_c, cv2.CV_64F).var()
        sharpness_ratio = current_sharpness / max(baseline_sharpness, 1.0)

        # Enhanced QR and barcode detection
        qr_base = _detect_qr_presence(b)
        qr_curr = _detect_qr_presence(c_resized)
        if qr_base != qr_curr:
            regions.append(_normalize_diff_item({
                "id": "qr-change",
                "region": "label area",
                "bbox": None,
                "type": "label_mismatch",
                "description": "QR code presence differs between baseline and current image - potential label tampering.",
                "severity": "HIGH",
                "confidence": 0.85,
                "explainability": ["QR code mismatch", "label area change"],
                "suggested_action": "Immediate quarantine",
                "tis_delta": -35,
            }))
        
        # Detect significant image quality degradation (potential digital tampering)
        if sharpness_ratio < 0.5 or sharpness_ratio > 2.0:
            regions.append(_normalize_diff_item({
                "id": "quality-degradation",
                "region": "overall",
                "bbox": None,
                "type": "digital_edit",
                "description": f"Significant image quality change detected (sharpness ratio: {sharpness_ratio:.2f}) - potential digital manipulation.",
                "severity": "HIGH",
                "confidence": 0.75,
                "explainability": [f"sharpness change: {sharpness_ratio:.2f}", "image quality degradation"],
                "suggested_action": "Quarantine for digital forensics",
                "tis_delta": -45,
            }))

        # Enhanced repackaging detection
        if sim < 0.15:
            regions.append(_normalize_diff_item({
                "id": "repackaging-1",
                "region": "overall",
                "bbox": None,
                "type": "repackaging",
                "description": "Current image appears to be a completely different object or heavily repackaged item compared to baseline.",
                "severity": "HIGH",
                "confidence": 0.95,
                "explainability": [f"extremely low ORB similarity: {sim:.2f}", "structural mismatch", "different object detected"],
                "suggested_action": "Immediate quarantine - potential product substitution",
                "tis_delta": -60,  # Higher penalty for completely different objects
            }))
        elif sim < 0.4:
            regions.append(_normalize_diff_item({
                "id": "moderate-repackaging",
                "region": "overall",
                "bbox": None,
                "type": "repackaging",
                "description": "Significant structural changes detected - possible repackaging or major damage.",
                "severity": "HIGH",
                "confidence": 0.80,
                "explainability": [f"low ORB similarity: {sim:.2f}", "structural changes"],
                "suggested_action": "Immediate quarantine - major structural changes",
                "tis_delta": -35,
            }))

        # Enhanced contour analysis with better heuristics
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if area / img_area < 0.003:  # ignore tiny specks <0.3%
                continue
                
            # Enhanced geometric analysis
            aspect = max(w, h) / max(1.0, min(w, h))
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Enhanced region classification for better specificity
            near_top = y < top_band
            in_center = (center_region[0] <= x <= center_region[2] and 
                        center_region[1] <= y <= center_region[3])
            near_edge = (x < w1 * 0.1 or x + w > w1 * 0.9 or 
                        y < h1 * 0.1 or y + h > h1 * 0.9)
            
            # More specific region labeling
            if near_top:
                if x < w1 * 0.33:
                    region_label = "top-left corner"
                elif x + w > w1 * 0.66:
                    region_label = "top-right corner"
                else:
                    region_label = "top edge"
            elif x < w1 * 0.2:
                region_label = "left side"
            elif x + w > w1 * 0.8:
                region_label = "right side"
            elif y < h1 * 0.2:
                region_label = "top edge"
            elif y + h > h1 * 0.8:
                region_label = "bottom edge"
            elif in_center:
                region_label = "center"
            else:
                region_label = "side region"
            
            bbox = [x / w1, y / h1, w / w1, h / h1]
            
            # Enhanced damage type classification with better region specificity
            if near_top and h > 0.06 * h1 and w > 0.2 * w1:
                # Seal tampering detection
                diff_item = {
                    "id": f"seal-tamper-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "seal_tamper",
                    "description": f"Seal area shows significant change - potential tampering detected in {region_label} (area: {area/img_area*100:.1f}% of image).",
                    "severity": "HIGH",
                    "confidence": 0.85,
                    "explainability": ["large change in seal area", "top region modification", f"area coverage: {area/img_area*100:.1f}%"],
                    "suggested_action": "Immediate quarantine - security breach",
                    "tis_delta": -40,
                }
            elif aspect > 8.0 and area / img_area < 0.03:
                # Scratch detection
                diff_item = {
                    "id": f"scratch-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "scratch",
                    "description": f"Linear scratch mark detected on {region_label} (length: {max(w,h):.0f}px).",
                    "severity": "LOW",
                    "confidence": 0.75,
                    "explainability": ["high aspect ratio", "linear pattern", "edge contrast"],
                    "suggested_action": "Proceed with caution",
                    "tis_delta": -8,
                }
            elif circularity > 0.6 and area / img_area > 0.02:
                # Circular damage (dents, holes)
                diff_item = {
                    "id": f"circular-damage-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "dent",
                    "description": f"Circular dent detected on {region_label} - likely impact damage (diameter: {max(w,h):.0f}px).",
                    "severity": "MEDIUM" if area / img_area < 0.05 else "HIGH",
                    "confidence": 0.80,
                    "explainability": ["circular shape", "high circularity", "impact pattern"],
                    "suggested_action": "Supervisor review" if area / img_area >= 0.05 else "Proceed",
                    "tis_delta": -15 if area / img_area < 0.05 else -25,
                }
            elif area / img_area > 0.08:
                # Large damage area
                diff_item = {
                    "id": f"major-damage-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "dent",
                    "description": f"Major structural damage detected on {region_label} - significant area affected ({area/img_area*100:.1f}% of image).",
                    "severity": "HIGH",
                    "confidence": 0.85,
                    "explainability": ["large affected area", "structural change", f"coverage: {area/img_area*100:.1f}%"],
                    "suggested_action": "Immediate quarantine - major damage",
                    "tis_delta": -30,
                }
            else:
                # General damage
                diff_item = {
                    "id": f"damage-{i}",
                    "region": region_label,
                    "bbox": bbox,
                    "type": "dent",
                    "description": f"Localized damage detected on {region_label} (area: {area/img_area*100:.1f}% of image).",
                    "severity": "MEDIUM" if area / img_area < 0.03 else "HIGH",
                    "confidence": 0.70,
                    "explainability": ["localized change", "intensity difference", f"area: {area/img_area*100:.1f}%"],
                    "suggested_action": "Supervisor review" if area / img_area >= 0.03 else "Proceed",
                    "tis_delta": -12 if area / img_area < 0.03 else -20,
                }
            regions.append(_normalize_diff_item(diff_item))

        # Merge/limit to top 5 by severity area
        return regions[:5]
    except Exception:
        return []


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # quick runtime sanity
        if not _configure_genai():
            # If you do not want to require gemini for fallback, change logic here.
            # For debugging, return an explicit error.
            return jsonify({
                "error": "GOOGLE_API_KEY / GEMINI_API_KEY not configured or google.generativeai import failed.",
                "differences": [],
                "aggregate_tis": 100,
                "overall_assessment": "UNKNOWN"
            }), 500

        data = request.get_json(silent=True) or {}
        baseline_src = data.get("baseline_url") or data.get("baseline_b64")
        current_src = data.get("current_url") or data.get("current_b64")

        baseline_bytes, baseline_mime = _load_image_bytes(baseline_src)
        current_bytes, current_mime = _load_image_bytes(current_src)

        baseline_info = _get_image_info(baseline_bytes)
        current_info = _get_image_info(current_bytes)

        differences = _call_gemini((baseline_bytes, baseline_mime), (current_bytes, current_mime))

        # fallback to classical CV if gemini empty or low confidence
        if (not differences or sum(d.get("confidence", 0) for d in differences) / max(1, len(differences)) < 0.6) and baseline_bytes and current_bytes:
            cv_regions = []
            try:
                if align_and_normalize is not None and cv2 is not None:
                    ab, ac = align_and_normalize(baseline_bytes, current_bytes)
                    if ab is not None and ac is not None:
                        cv_regions = _classical_diff_regions(cv2.imencode('.jpg', ab)[1].tobytes(), cv2.imencode('.jpg', ac)[1].tobytes())
                if not cv_regions:
                    cv_regions = _classical_diff_regions(baseline_bytes, current_bytes)
            except Exception as e:
                print("classical diff error:", str(e), file=sys.stderr)
                cv_regions = _classical_diff_regions(baseline_bytes, current_bytes)
            if cv_regions:
                if differences:
                    seen = {d.get("id") for d in differences}
                    for r in cv_regions:
                        if r.get("id") not in seen:
                            differences.append(r)
                else:
                    differences = cv_regions

        tis, assessment, conf_overall, notes = _compute_overall(differences)

        response = {
            "differences": differences,
            "baseline_image_info": baseline_info,
            "current_image_info": current_info,
            "aggregate_tis": tis,
            "overall_assessment": assessment,
            "confidence_overall": conf_overall,
            "notes": notes,
            "analysis_metadata": {
                "total_differences": len(differences),
                "high_severity_count": len([d for d in differences if str(d.get("severity", "")).upper() == "HIGH"]),
                "medium_severity_count": len([d for d in differences if str(d.get("severity", "")).upper() == "MEDIUM"]),
                "low_severity_count": len([d for d in differences if str(d.get("severity", "")).upper() == "LOW"]),
                "analysis_timestamp": str(datetime.now().isoformat()) if 'datetime' in globals() else "unknown"
            }
        }
        return jsonify(response)
    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in /analyze:", tb, file=sys.stderr)
        # return error info (status 500) so client sees problem
        return jsonify({
            "error": "Analyzer internal error",
            "details": str(e),
            "traceback": tb,
            "differences": [],
            "aggregate_tis": 100,
            "overall_assessment": "UNKNOWN"
        }), 500