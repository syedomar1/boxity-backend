import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None
    np = None


def align_and_normalize(b_bytes: bytes, c_bytes: bytes) -> Tuple[Optional["cv2.Mat"], Optional["cv2.Mat"]]:
    """Enhanced image alignment and normalization for better comparison accuracy."""
    if cv2 is None:
        return None, None
    
    try:
        b_arr = np.frombuffer(b_bytes, dtype=np.uint8)
        c_arr = np.frombuffer(c_bytes, dtype=np.uint8)
        b = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)
        c = cv2.imdecode(c_arr, cv2.IMREAD_COLOR)
        if b is None or c is None:
            return None, None
            
        h, w = b.shape[:2]
        c_resized = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)

        # Enhanced feature-based alignment with multiple detectors
        try:
            # Use multiple feature detectors for better alignment
            detectors = [
                cv2.ORB_create(nfeatures=1500),
                cv2.SIFT_create(nfeatures=1000) if hasattr(cv2, 'SIFT_create') else None
            ]
            
            best_homography = None
            best_match_count = 0
            
            for detector in detectors:
                if detector is None:
                    continue
                    
                k1, d1 = detector.detectAndCompute(b, None)
                k2, d2 = detector.detectAndCompute(c_resized, None)
                
                if d1 is not None and d2 is not None and len(k1) >= 15 and len(k2) >= 15:
                    # Use appropriate matcher based on detector type
                    if 'ORB' in str(type(detector)):
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                        matches = matcher.knnMatch(d1, d2, k=2)
                        good = []
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:  # Stricter ratio
                                good.append(m)
                    else:
                        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                        matches = matcher.knnMatch(d1, d2, k=2)
                        good = []
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                    
                    if len(good) >= 12:  # Require more matches for better alignment
                        src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        
                        # Use RANSAC with stricter parameters
                        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0, maxIters=2000)
                        
                        if H is not None and np.sum(mask) > best_match_count:
                            best_homography = H
                            best_match_count = np.sum(mask)
            
            # Apply best homography if found
            if best_homography is not None and best_match_count >= 8:
                c_resized = cv2.warpPerspective(c_resized, best_homography, (w, h))
                
        except Exception as e:
            print(f"Alignment failed: {e}", file=sys.stderr)
            pass

        # Enhanced illumination normalization
        # Convert to LAB color space for better perceptual uniformity
        b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
        c_lab = cv2.cvtColor(c_resized, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel (luminance)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        b_lab[:, :, 0] = clahe.apply(b_lab[:, :, 0])
        c_lab[:, :, 0] = clahe.apply(c_lab[:, :, 0])
        
        # Convert back to BGR
        b_norm = cv2.cvtColor(b_lab, cv2.COLOR_LAB2BGR)
        c_norm = cv2.cvtColor(c_lab, cv2.COLOR_LAB2BGR)
        
        # Additional histogram equalization for better contrast
        b_norm = cv2.addWeighted(b_norm, 0.8, cv2.equalizeHist(cv2.cvtColor(b_norm, cv2.COLOR_BGR2GRAY)), 0.2, 0)
        c_norm = cv2.addWeighted(c_norm, 0.8, cv2.equalizeHist(cv2.cvtColor(c_norm, cv2.COLOR_BGR2GRAY)), 0.2, 0)
        
        return b_norm, c_norm
        
    except Exception as e:
        print(f"Normalization failed: {e}", file=sys.stderr)
        return None, None


