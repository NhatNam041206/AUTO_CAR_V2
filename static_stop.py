# static_stop.py
import cv2 as cv
import numpy as np

class StaticParams:
    # === FIXED threshold on LAB L* (floor assumed brighter) ===
    # Increase if your floor is very bright (try 200 → 210 → 220).
    THR_L = 200

    # === General blob guards ===
    MIN_AREA   = 1200   # reject tiny blobs (px)
    MIN_THICK  = 6      # min(w, h) must be >= this (px)
    ASPECT_MAX = 12.0   # hard cap on absurd aspect ratio (failsafe)

    # === Line-specific rejection (tile/grout lines) ===
    # If component is elongated AND has very low fill, treat as a line → reject.
    LINE_AR_REJECT = 6.0    # elongated if max(w/h, h/w) >= this
    LINE_FILL_MAX  = 0.22   # fill = area/(w*h) must be > this if elongated

    # === Global STOP decision threshold ===
    AREA_PCT = 1.0      # % of ROI pixels (not full image) to trigger STOP

    # === Morphology cleanup ===
    MORPH = (3, 3)

def _shape_metrics(w, h, area):
    """Return (fill, thickness, elongation)."""
    box_area = max(1, w * h)
    fill = area / float(box_area)                # pixels / bbox area
    thickness = min(w, h)                        # skinny lines have tiny thickness
    elong = max(w / max(1, h), h / max(1, w))    # >= 1.0; higher = more elongated
    return fill, thickness, elong

def _passes_shape_filters(w, h, area, p: StaticParams) -> bool:
    # General guards
    if area < p.MIN_AREA:       return False
    if min(w, h) < p.MIN_THICK: return False

    fill, thick, elong = _shape_metrics(w, h, area)

    # Failsafe: absurd aspect
    if elong > p.ASPECT_MAX:
        return False

    # Tile/grout line rejection
    if (elong >= p.LINE_AR_REJECT) and (fill <= p.LINE_FILL_MAX):
        return False

    return True

def static_stop_detect(frame_bgr, roi_mask, danger_mask, params: StaticParams = StaticParams()):
    """
    Static obstacle check with FIXED L* threshold and shape filters.
    Returns: (stop: bool, bbox: (x,y,w,h) or None, debug dict)
    """
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Restrict to ROI polygon first
    L_roi = cv.bitwise_and(L, L, mask=roi_mask)

    # Fixed threshold on L* (floor bright → non-floor dark)
    thr = int(np.clip(params.THR_L, 0, 255))
    _, floor = cv.threshold(L_roi, thr, 255, cv.THRESH_BINARY)
    nonfloor = cv.bitwise_not(floor)

    # Cleanup small speckle (keep cheap)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, params.MORPH)
    nonfloor = cv.morphologyEx(nonfloor, cv.MORPH_OPEN, kernel, iterations=1)
    nonfloor = cv.morphologyEx(nonfloor, cv.MORPH_CLOSE, kernel, iterations=1)

    # Consider only danger band within ROI
    nf_danger = cv.bitwise_and(nonfloor, danger_mask)

    # Connected components + shape filtering
    num, lbl, stats, _ = cv.connectedComponentsWithStats(nf_danger, connectivity=8)
    best_bbox, best_area = None, 0
    for i in range(1, num):
        x, y, w2, h2, area = stats[i]
        if _passes_shape_filters(w2, h2, area, params):
            if area > best_area:
                best_bbox, best_area = (x, y, w2, h2), int(area)

    # Global decision: area% over ROI polygon
    roi_pix = max(1, int(np.count_nonzero(roi_mask)))
    area_pct = (100.0 * best_area / roi_pix) if best_area > 0 else 0.0
    stop = area_pct >= params.AREA_PCT

    debug = {
        "thr": thr,
        "nonfloor": nonfloor,
        "nf_danger": nf_danger,
        "area_pct": area_pct,
    }
    return stop, best_bbox, debug
