# static_stop.py
import cv2 as cv
import numpy as np

class StaticParams:
    # Threshold = mean_floor - K * std_floor  (on bottom "safe strip")
    K = 1.5
    # ignore tiny blobs
    MIN_AREA = 250
    # stop if non-floor area in danger zone exceeds this % of ROI pixels
    AREA_PCT = 0.75
    # morphology kernel for cleanup
    MORPH = (3, 3)
    # bottom strip used to estimate floor brightness (fraction of image height)
    SAFE_STRIP_FRAC = 0.12

def static_stop_detect(frame_bgr, roi_mask, danger_mask, params: StaticParams = StaticParams()):
    """
    Your static obstacle method:
    - model floor as bright region in LAB L*
    - threshold adaptively using a bottom safe strip
    - blobs in danger zone -> STOP if big enough
    Returns: (stop: bool, bbox: (x,y,w,h) or None, debug: dict)
    """
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # Mask to ROI polygon
    L_roi = cv.bitwise_and(L, L, mask=roi_mask)

    h, w = L.shape
    # Safe strip at bottom of image, intersect with ROI polygon
    strip = np.zeros((h, w), np.uint8)
    y0 = int(h * (1.0 - params.SAFE_STRIP_FRAC))
    strip[y0:, :] = 255
    strip = cv.bitwise_and(strip, roi_mask)

    # Estimate floor brightness stats in the strip
    m, s = cv.meanStdDev(L, mask=strip)
    mean_floor = float(m[0, 0]) if m is not None else 220.0
    std_floor  = float(s[0, 0]) if s is not None else 5.0

    thr = int(np.clip(mean_floor - params.K * std_floor, 0, 255))

    # Floor = bright; Non-floor = dark
    _, floor = cv.threshold(L_roi, thr, 255, cv.THRESH_BINARY)
    nonfloor = cv.bitwise_not(floor)

    # Clean noise
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

    # Area percentage relative to ROI polygon (not full image)
    roi_pix = max(1, int(np.count_nonzero(roi_mask)))
    area_pct = (100.0 * best_area / roi_pix) if best_area > 0 else 0.0
    stop = area_pct >= params.AREA_PCT

    debug = {
        "thr": thr,
        "mean_floor": mean_floor,
        "std_floor": std_floor,
        "nonfloor": nonfloor,
        "nf_danger": nf_danger,
        "area_pct": area_pct,
    }
    return stop, best_bbox, debug
