# static_stop.py
import cv2 as cv
import numpy as np

class StaticParams:
    # --- FIXED threshold on LAB L* (floor assumed brighter) ---
    THR_L = 170             # try 200/210/220 depending on your floor brightness
    # --- cleanup / filtering ---
    MIN_AREA = 300          # ignore tiny blobs
    AREA_PCT = 25         # % of ROI pixels to trigger STOP (0.75% default)
    MORPH = (3, 3)
    # (kept for compatibility if you later want adaptive)
    SAFE_STRIP_FRAC = 0.12
    K = 1.5

def static_stop_detect(frame_bgr, roi_mask, danger_mask, params: StaticParams = StaticParams()):
    """
    Static obstacle check with FIXED L* threshold:
      - Convert to LAB, threshold L* (bright floor -> non-floor is dark).
      - Consider ONLY the region within your polygon ROI and danger band.
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

    # Find biggest component
    num, lbl, stats, _ = cv.connectedComponentsWithStats(nf_danger, connectivity=8)
    best_bbox, best_area = None, 0
    for i in range(1, num):
        x, y, w2, h2, area = stats[i]
        if area >= params.MIN_AREA and area > best_area:
            best_bbox, best_area = (x, y, w2, h2), int(area)

    # Area percentage relative to ROI polygon (simple + stable)
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
