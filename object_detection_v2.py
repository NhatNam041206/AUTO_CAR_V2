# lite_stop_base.py
import cv2 as cv
import numpy as np
from ROI import ROI
from helpers import rotate
import math
from static_stop import static_stop_detect, StaticParams

# ---------------- DEBUG ----------------
DEBUG_MOTION = True    # True → print motion-component metrics
SHOW_STATIC_DEBUG = False  # show "Nonfloor" & "NF Danger" windows

# ---------- Scene-idle detection ----------
MOT_IDLE_FRAC  = 0.001
IDLE_FRAMES    = 5

# ---------- Config ----------
CAM_INDEX = int(input('Camera Source (0/1/2): '))
WIDTH, HEIGHT = 640, 480
EDGE_PAD = 4
DIFF_THR = 18
DANGER_YFRAC = 0.45
STOP_AREA_PCT = 1.0
STOP_BOTTOM_Y = 0.85
MORPH_KERNEL = (3, 3)

# ---- Motion-path shape filters (unchanged) ----
M_MIN_AREA       = 1200
M_MIN_THICK      = 6
M_ASPECT_MAX     = 12.0
M_LINE_AR_REJECT = 6.0
M_LINE_FILL_MAX  = 0.22

# ---------- DEBUG HELPERS (no logic changes) ----------
def _shape_metrics(w, h, area):
    box_area = max(1, w * h)
    fill  = area / float(box_area)
    elong = max(w / max(1, h), h / max(1, w))
    return fill, elong

def _passes_motion_shape_filters(w, h, area):
    """Same logic as before, but prints debug info."""
    fill, elong = _shape_metrics(w, h, area)

    passed = True
    reason = "PASS"
    if area < M_MIN_AREA:
        passed, reason = False, "small"
    elif min(w, h) < M_MIN_THICK:
        passed, reason = False, "thin"
    elif elong > M_ASPECT_MAX:
        passed, reason = False, "absurd"
    elif (elong >= M_LINE_AR_REJECT) and (fill <= M_LINE_FILL_MAX):
        passed, reason = False, "line"

    if DEBUG_MOTION:
        print(f"[Motion] area={area:5d} w={w:3d} h={h:3d} "
              f"fill={fill:.3f} elong={elong:.2f} → {reason}")

    return passed

# ---------- ROI helper ----------
roi_helper = ROI(
    saved_path="roi_points.txt",
    ROTATE_CW_DEG=0,
    FLIPCODE=1,
    ANGLE_TRIANGLE=math.radians(60),
    W=640, H=480
)
roi_ok = roi_helper.get_roi()
if not roi_ok:
    raise SystemExit("ROI not set. Exiting.")

def connected_components(binmask):
    num, lbl, stats, _ = cv.connectedComponentsWithStats(binmask, connectivity=8)
    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        comps.append(((x, y, w, h), area))
    return comps

def main():
    cap = cv.VideoCapture(CAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ok, frame0 = cap.read()
    if not ok:
        print("Camera error"); return

    # SAME transforms as ROI picker
    frame0 = rotate(frame0, roi_helper.ROTATE_CW_DEG)
    frame0 = cv.flip(frame0, roi_helper.FLIPCODE)
    frame0 = cv.resize(frame0, (roi_helper.W, roi_helper.H))

    # Build ROI & danger masks
    roi_mask, danger_mask = roi_helper.build_masks(
        frame0.shape, danger_frac=DANGER_YFRAC, edge_pad=EDGE_PAD
    )

    kernel = cv.getStructuringElement(cv.MORPH_RECT, MORPH_KERNEL)
    prev_gray = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

    roi_pix = max(1, int(np.count_nonzero(roi_mask)))
    idle_count = 0

    stop_latch = False
    clear_count = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = rotate(frame, roi_helper.ROTATE_CW_DEG)
        frame = cv.flip(frame, roi_helper.FLIPCODE)
        frame = cv.resize(frame, (roi_helper.W, roi_helper.H))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Motion path
        diff = cv.absdiff(gray, prev_gray)
        diff = cv.bitwise_and(diff, roi_mask)

        _, mot = cv.threshold(diff, DIFF_THR, 255, cv.THRESH_BINARY)
        mot = cv.morphologyEx(mot, cv.MORPH_OPEN, kernel, iterations=1)
        mot = cv.morphologyEx(mot, cv.MORPH_CLOSE, kernel, iterations=1)
        mot_danger = cv.bitwise_and(mot, danger_mask)

        # Idle detection
        mot_ratio = float(np.count_nonzero(mot)) / float(roi_pix)
        idle_count = idle_count + 1 if mot_ratio < MOT_IDLE_FRAC else 0
        use_static = (idle_count >= IDLE_FRAMES)
        if DEBUG_MOTION:
            print(f"[Motion] mot_ratio={mot_ratio:.4f} idle_count={idle_count} use_static={use_static}")

        stop = False
        bbox = None

        if not use_static:
            comps = connected_components(mot_danger)

            # IMPORTANT: reset best_* every frame
            best_bbox, best_area = None, 0

            for (x, y, w, h), area in comps:
                if _passes_motion_shape_filters(w, h, area):
                    if area > best_area:
                        best_bbox, best_area = (x, y, w, h), area

            if best_bbox is not None:
                bbox = best_bbox
                x, y, bw, bh = bbox
                area_pct = 100.0 * (bw * bh) / roi_pix
                bottom_pass = (y + bh) > int(mot.shape[0] * STOP_BOTTOM_Y)

                if DEBUG_MOTION:
                    print(f"[Motion] best_area={best_area} "
                          f"area_pct={area_pct:.2f} bottom_pass={bottom_pass}")

                if area_pct >= STOP_AREA_PCT or bottom_pass:
                    stop = True
            else:
                stop = False
                bbox = None
        else:
            sp = StaticParams()  # DEBUG flag is inside StaticParams
            stop, bbox, dbg = static_stop_detect(frame, roi_mask, danger_mask, sp)
            if SHOW_STATIC_DEBUG:
                cv.imshow("Nonfloor", dbg["nonfloor"])
                cv.imshow("NF Danger", dbg["nf_danger"])

        if bbox is not None:
            x, y, bw, bh = bbox
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        if stop:
            stop_latch = True
            clear_count = 0
        else:
            clear_count += 1
            if clear_count > 5:
                stop_latch = False

        if stop_latch:
            cv.putText(frame, "STOP", (10, 24),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.imshow("Motion(masked ROI)", mot)
        cv.imshow("Danger(masked)", mot_danger)
        cv.imshow("View", frame)

        prev_gray = gray
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
