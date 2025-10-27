# static_stop.py
import cv2 as cv
import numpy as np

class StaticParams:
    # ---------------- DEBUG ----------------
    DEBUG = True  # True → print ratios & decisions per component

    # --- FIXED threshold on LAB L* (floor assumed brighter) ---
    THR_L = 150

    # --- General blob guards ---
    MIN_AREA   = 500
    MIN_THICK  = 6
    ASPECT_MAX = 1.4

    # --- Line-specific rejection (tile/grout) ---
    LINE_AR_REJECT = 3.0
    LINE_FILL_MAX  = 0.22

    # --- Global STOP decision threshold ---
    AREA_PCT = 2.5 # percentage of detected bbox with ROI_area

    # --- Morphology cleanup ---
    MORPH = (3, 3)

# ---------- DEBUG HELPERS (no logic changes) ----------
def _shape_metrics(w, h, area):
    box_area = max(1, w * h)
    fill  = area / float(box_area)                 # pixels / bbox area
    elong = max(w / max(1, h), h / max(1, w))      # ≥1.0; big → elongated
    return fill, elong

def _passes_shape_filters(w, h, area, p: StaticParams):
    """Same logic as before, but prints debug info."""
    fill, elong = _shape_metrics(w, h, area)

    passed = True
    reason = "PASS"
    if area < p.MIN_AREA:
        passed, reason = False, "small"
    elif min(w, h) < p.MIN_THICK:
        passed, reason = False, "thin"
    elif elong > p.ASPECT_MAX:
        passed, reason = False, "absurd"
    elif (elong >= p.LINE_AR_REJECT) or (fill <= p.LINE_FILL_MAX):
        passed, reason = False, "line"

    if p.DEBUG:
        # w,h, fill, elong + verdict
        print(f"[Static] area={area:5d} w={w:3d} h={h:3d} "
              f"fill={fill:.3f} elong={elong:.2f} → {reason}")

    return passed,fill,elong

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

    # Only danger band within ROI
    nf_danger = cv.bitwise_and(nonfloor, danger_mask)

    # Connected components + shape filtering
    num, lbl, stats, _ = cv.connectedComponentsWithStats(nf_danger, connectivity=4)
    best_bbox, best_area, fill, elong = None, 0, 0, 0
    params_dbg=None
    for i in range(1, num):
        x, y, w2, h2, area = stats[i]
        passed,fill,elong=_passes_shape_filters(w2, h2, area, params)
        if passed:
            if area > best_area:
                best_bbox, best_area, params_dbg = (x, y, w2, h2), int(area), (elong, fill)

    # Global decision: area% over ROI polygon
    roi_pix = max(1, int(np.count_nonzero(roi_mask)))
    area_pct = (100.0 * best_area / roi_pix) if best_area > 0 else 0.0
    stop = area_pct >= params.AREA_PCT

    if params.DEBUG:
        print(f"[Static] Best={best_bbox} area_pct={area_pct:.2f}% STOP={stop}")

    debug = {
        "nonfloor": nonfloor,
        "nf_danger": nf_danger,
        "area_pct": area_pct,
        "elong" : params_dbg[0] if params_dbg is not None else 0,
        "fill": params_dbg[1] if params_dbg is not None else 0
    }
    return stop, best_bbox, debug
