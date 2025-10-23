# lite_stop_base.py
import cv2 as cv
import numpy as np
from ROI import ROI
from helpers import rotate
import math
from static_stop import static_stop_detect, StaticParams

# Scene-idle detection (when to switch to "static" path)
MOT_IDLE_FRAC  = 0.001   # fraction of ROI pixels considered "no motion"
IDLE_FRAMES    = 5       # how many consecutive frames → consider stopped
STATIC_CHECK_EVERY = 1   # run static check each idle frame (keep light)

# ---------- Config (edit these) ----------
CAM_INDEX = int(input('Camera Source: '))              # your camera index (0/1/2...)
WIDTH, HEIGHT = 640, 480   # capture size (keep modest for speed)
ROI_FRACTION = 0.5         # use bottom half of the frame
EDGE_PAD = 4               # ignore thin border to reduce edge flicker
DIFF_THR = 18              # threshold for frame differencing (8-bit)
MIN_BLOB_AREA = 150        # reject tiny flicker blobs
DANGER_YFRAC = 0.45        # bottom DANGER_YFRAC% of ROI is "danger band"
STOP_AREA_PCT = 1.0        # stop if motion blob area > this % of ROI
STOP_BOTTOM_Y = 0.85       # or if bbox bottom passes 85% of ROI height
MORPH_KERNEL = (3, 3)      # morphology kernel


# 1) Instantiate ROI helper (set your file path & camera transforms)
roi_helper = ROI(saved_path="roi_points.txt",
                 ROTATE_CW_DEG=0,     # keep same transform in runtime
                 FLIPCODE=1,          # 1: horizontal
                 ANGLE_TRIANGLE=math.radians(60),
                 W=640, H=480, CAMERA_SOURCE=CAM_INDEX)

# 2) Let the user decide to load or create ROI (your existing UI)
roi_helper.get_roi()  # runs its own short capture for clicking if needed


def connected_components(binmask, min_area=150):
    num, lbl, stats, _ = cv.connectedComponentsWithStats(binmask, connectivity=8)
    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            comps.append(((x, y, w, h), area))
    return comps

def main():
    cap = cv.VideoCapture(CAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ok, frame0 = cap.read()
    if not ok:
        print("Camera error"); return

    # Apply SAME transforms as ROI picker
    frame0 = rotate(frame0, roi_helper.ROTATE_CW_DEG)
    frame0 = cv.flip(frame0, roi_helper.FLIPCODE)
    frame0 = cv.resize(frame0, (roi_helper.W, roi_helper.H))

    # Build masks from your 3-point polygon
    roi_mask, danger_mask = roi_helper.build_masks(
        frame0.shape, danger_frac=DANGER_YFRAC, edge_pad=EDGE_PAD
    )

    kernel = cv.getStructuringElement(cv.MORPH_RECT, MORPH_KERNEL)
    prev_gray = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

    # Precompute masked pixel count for area%
    roi_pix = max(1, int(np.count_nonzero(roi_mask)))
    idle_count=0
    
    stop_latch = False
    clear_count = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = rotate(frame, roi_helper.ROTATE_CW_DEG)
        frame = cv.flip(frame, roi_helper.FLIPCODE)
        frame = cv.resize(frame, (roi_helper.W, roi_helper.H))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ---- motion path (as before) ----
        diff = cv.absdiff(gray, prev_gray)
        diff = cv.bitwise_and(diff, roi_mask)
        _, mot = cv.threshold(diff, DIFF_THR, 255, cv.THRESH_BINARY)
        mot = cv.morphologyEx(mot, cv.MORPH_OPEN, kernel, iterations=1)
        mot = cv.morphologyEx(mot, cv.MORPH_CLOSE, kernel, iterations=1)
        mot_danger = cv.bitwise_and(mot, danger_mask)

        # Motion ratio in ROI
        mot_ratio = float(np.count_nonzero(mot)) / float(roi_pix)

        # Decide whether the scene is "idle" (car stopped / objects static)
        if mot_ratio < MOT_IDLE_FRAC:
            idle_count += 1
        else:
            idle_count = 0

        use_static = (idle_count >= IDLE_FRAMES)

        stop = False
        bbox = None

        if not use_static:
            # ------- MOVING CASE: use differencing path -------
            comps = connected_components(mot_danger, MIN_BLOB_AREA)
            if comps:
                bbox, area = max(comps, key=lambda c: c[1])
                x, y, bw, bh = bbox
                area_pct = 100.0 * (bw * bh) / roi_pix
                bottom_pass = (y + bh) > int(mot.shape[0] * STOP_BOTTOM_Y)
                if area_pct >= STOP_AREA_PCT or bottom_pass:
                    stop = True
        else:
            # ------- IDLE CASE: use YOUR static floor method -------
            stop, bbox, dbg = static_stop_detect(frame, roi_mask, danger_mask, StaticParams())
            # (optional) visualize debug:
            cv.imshow("Nonfloor", dbg["nonfloor"])
            cv.imshow("NF Danger", dbg["nf_danger"])

        # Draw bbox & STOP
        if bbox is not None:
            x,y,bw,bh = bbox
            cv.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)

        # Latch with hysteresis (unchanged)
        if stop:
            stop_latch = True
            clear_count = 0
        else:
            clear_count += 1
            if clear_count > 5:
                stop_latch = False

        if stop_latch:
            cv.putText(frame, "STOP", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Separate windows
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
