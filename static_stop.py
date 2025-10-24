# static_stop.py
import cv2 as cv
import numpy as np

class StaticParams:
    # --- FIXED threshold on LAB L* (floor assumed brighter) ---
    THR_L = 200             # try 200/210/220 depending on floor brightness

    # --- blob acceptance (shape/size) ---
    MIN_AREA = 1200         # reject tiny thin stuff
    MIN_FILL = 0.15         # area / (w*h) must be >= this
    MIN_THICK = 6           # min(w, h) in pixels
    ASPECT_MAX = 8.0        # reject extreme aspect ratios (w>8h or h>8w)

    # --- global decision threshold ---
    AREA_PCT = 1.0          # % of ROI pixels to trigger STOP

    # --- morphology cleanup ---
    MORPH = (3, 3)

def _passes_shape_filters(x, y, w, h, area, p: StaticParams) -> bool:
    if area < p.MIN_AREA:
        return False
    if w * h <= 0:
        return False
    fill = area / float(w * h)
    if fill < p.MIN_FILL:
        return False
    if min(w, h) < p.MIN_THICK:
        return False
    if (w > p.ASPECT_MAX * h) or (h > p.ASPECT_MAX * w):
        return False
    return True

def static_stop_detect(frame_bgr, roi_mask, danger_mask, params: StaticParams = StaticParams()):
    """
    Static obstacle check with FIXED L* threshold and shape filters.
    Returns: (stop: bool, bbox: (x,y,w,h) or None, debug dict)
    """
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Restrict to ROI polygon
    L_roi = cv.bitwise_and(L, L, mask=roi_mask)

    # Fixed threshold on L*
    thr = int(np.clip(params.THR_L, 0, 255))
    _, floor = cv.threshold(L_roi, thr, 255, cv.THRESH_BINARY)
    nonfloor = cv.bitwise_not(floor)

    # Cleanup
    kernel = cv.getStructuringElement(cv.MORPH_RECT, params.MORPH)
    nonfloor = cv.morphologyEx(nonfloor, cv.MORPH_OPEN, kernel, iterations=1)
    nonfloor = cv.morphologyEx(nonfloor, cv.MORPH_CLOSE, kernel, iterations=1)

    # Only consider the danger band inside the polygon
    nf_danger = cv.bitwise_and(nonfloor, danger_mask)

    # Connected components + shape filters
    num, lbl, stats, _ = cv.connectedComponentsWithStats(nf_danger, connectivity=8)
    best_bbox, best_area = None, 0
    for i in range(1, num):
        x, y, w2, h2, area = stats[i]
        if _passes_shape_filters(x, y, w2, h2, area, params):
            if area > best_area:
                best_bbox, best_area = (x, y, w2, h2), int(area)

    # Area percentage relative to ROI polygon
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
