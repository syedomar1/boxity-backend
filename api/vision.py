from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None
    np = None


def align_and_normalize(b_bytes: bytes, c_bytes: bytes) -> Tuple[Optional["cv2.Mat"], Optional["cv2.Mat"]]:
    if cv2 is None:
        return None, None
    b_arr = np.frombuffer(b_bytes, dtype=np.uint8)
    c_arr = np.frombuffer(c_bytes, dtype=np.uint8)
    b = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)
    c = cv2.imdecode(c_arr, cv2.IMREAD_COLOR)
    if b is None or c is None:
        return None, None
    h, w = b.shape[:2]
    c_resized = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)

    # Feature-based homography
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        k1, d1 = orb.detectAndCompute(b, None)
        k2, d2 = orb.detectAndCompute(c_resized, None)
        if d1 is not None and d2 is not None and len(k1) >= 10 and len(k2) >= 10:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = matcher.knnMatch(d1, d2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) >= 8:
                src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    c_resized = cv2.warpPerspective(c_resized, H, (w, h))
    except Exception:
        pass

    # Illumination normalization (CLAHE)
    b_yuv = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
    c_yuv = cv2.cvtColor(c_resized, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b_yuv[:, :, 0] = clahe.apply(b_yuv[:, :, 0])
    c_yuv[:, :, 0] = clahe.apply(c_yuv[:, :, 0])
    b_norm = cv2.cvtColor(b_yuv, cv2.COLOR_YCrCb2BGR)
    c_norm = cv2.cvtColor(c_yuv, cv2.COLOR_YCrCb2BGR)
    return b_norm, c_norm


