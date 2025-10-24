# lite_stop_base.py
import cv2 as cv
import numpy as np
from ROI import ROI
from helpers import rotate
import math
from static_stop import static_stop_detect, StaticParams
from signal import Car


# ---------- Scene-idle detection (when to switch to static path) ----------
MOT_IDLE_FRAC  = 0.001   # fraction of ROI pixels considered "no motion"
IDLE_FRAMES    = 5       # consecutive frames below MOT_IDLE_FRAC => idle
SHOW_STATIC_DEBUG = True  # set True to show "Nonfloor" & "NF Danger"

# ---------- Config (edit these) ----------
CAM_INDEX = int(input('Camera Source (0/1/2): '))
WIDTH, HEIGHT = 640, 480      # capture size
EDGE_PAD = 4                  # ignore thin border to reduce edge flicker
DIFF_THR = 18                 # threshold for frame differencing (8-bit)
MIN_BLOB_AREA = 300           # reject tiny flicker blobs (motion path)
DANGER_YFRAC = 0.45           # bottom fraction of ROI is "danger band"
STOP_AREA_PCT = 25.0           # motion STOP if bbox area% >= this
STOP_BOTTOM_Y = 0.85          # or bbox bottom passes this frac of ROI height
MORPH_KERNEL = (3, 3)         # morphology kernel

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

    # Build masks from your 3-point polygon (ROI + danger band)
    roi_mask, danger_mask = roi_helper.build_masks(
        frame0.shape, danger_frac=DANGER_YFRAC, edge_pad=EDGE_PAD
    )

    kernel = cv.getStructuringElement(cv.MORPH_RECT, MORPH_KERNEL)
    prev_gray = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

    # Precompute masked pixel count for area%
    roi_pix = max(1, int(np.count_nonzero(roi_mask)))
    idle_count = 0

    stop_latch = False
    clear_count = 0
    
    car=Car()

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = rotate(frame, roi_helper.ROTATE_CW_DEG)
        frame = cv.flip(frame, roi_helper.FLIPCODE)
        frame = cv.resize(frame, (roi_helper.W, roi_helper.H))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ---- Motion path (simple, lightweight) ----
        diff = cv.absdiff(gray, prev_gray)
        diff = cv.bitwise_and(diff, roi_mask)  # restrict to polygon

        _, mot = cv.threshold(diff, DIFF_THR, 255, cv.THRESH_BINARY)
        mot = cv.morphologyEx(mot, cv.MORPH_OPEN, kernel, iterations=1)
        mot = cv.morphologyEx(mot, cv.MORPH_CLOSE, kernel, iterations=1)
        mot_danger = cv.bitwise_and(mot, danger_mask)

        # Motion ratio in ROI
        mot_ratio = float(np.count_nonzero(mot)) / float(roi_pix)

        # Idle detection
        idle_count = idle_count + 1 if mot_ratio < MOT_IDLE_FRAC else 0
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
            # ------- IDLE CASE: use STATIC fixed-threshold method (whole danger band) -------
            # Tune THR_L once per site; start with 200 and adjust to 210/220 if floor is very bright
            sp = StaticParams()
            stop, bbox, dbg = static_stop_detect(frame, roi_mask, danger_mask, sp)
            if bbox is not None:
                x, y, bw, bh = bbox
                area_pct = 100.0 * (bw * bh) / roi_pix
            
            if SHOW_STATIC_DEBUG:
                cv.imshow("Nonfloor", dbg["nonfloor"])
                cv.imshow("NF Danger", dbg["nf_danger"])

        # Draw bbox & STOP
        if bbox is not None:
            x, y, bw, bh = bbox
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.putText(frame,f'Area: {area_pct}',(x,y),2,1,(0,255,0))

        # Latch with hysteresis
        if stop:
            stop_latch = True
            clear_count = 0
        else:
            clear_count += 1
            if clear_count > 5:
                stop_latch = False

        if stop_latch:
            cv.putText(frame, "STOP", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            car.send_command('mctl 0 0')
        else:
            car.send_command('mctl 100 100')
        # Views
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
