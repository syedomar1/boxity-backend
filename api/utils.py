import base64
import io
from typing import Any, Dict, Optional, Tuple

try:
    import requests
except Exception:
    requests = None

try:
    from PIL import Image, ExifTags
except Exception:
    Image = None
    ExifTags = None


def load_image_bytes(source: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not source:
        return None, None
    if source.startswith('data:'):
        try:
            header, b64 = source.split(',', 1)
            mime = header.split(';', 1)[0].replace('data:', '') or 'application/octet-stream'
            return base64.b64decode(b64), mime
        except Exception:
            return None, None
    if len(source) > 256 and not source.startswith('http'):
        try:
            return base64.b64decode(source), 'image/jpeg'
        except Exception:
            return None, None
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


def get_image_info(img_bytes: Optional[bytes]) -> Dict[str, Any]:
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
                def get_tag(tag_name: str):
                    key = inv.get(tag_name)
                    return str(exif.get(key)) if key in exif else None
                info["camera_make"] = get_tag("Make")
                info["camera_model"] = get_tag("Model")
                info["datetime"] = get_tag("DateTimeOriginal") or get_tag("DateTime")
    except Exception:
        pass
    return info


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))

