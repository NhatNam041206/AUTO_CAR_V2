import cv2 as cv
import numpy as np
import math
import os
import time
from datetime import datetime
from ROI import ROI
from helpers import rotate
from static_stop import static_stop_detect, StaticParams

# -------------------- CONFIG --------------------
CAM_DEVICE = "/dev/video0"    # change if needed
VIDEO_PATH = r"test\7152851634197.mp4"
W, H       = 640, 480
FPS        = 30
OUT_SCALE  = 0.7

SHOW_DEBUG_WINDOWS = True
USE_BLUR   = True
BLUR_KSIZE = 5
BLUR_SIGMA = 5
SAFE_FLUSH = 0
# ------------------------------------------------

def safe_read(cap, flush=0):
    """Grab and retrieve a COMPLETE frame (avoids partial updates)."""
    for _ in range(flush):
        cap.grab()
    ok = cap.grab()
    if not ok:
        return False, None
    ok, frame = cap.retrieve()
    return ok, frame

def log_message(logfile, msg):
    """Append message with timestamp to log file."""
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{t}] {msg}\n")

def main():
    roi_helper = ROI(
        saved_path="roi_points.txt",
        ROTATE_CW_DEG=0,
        FLIPCODE=1,
        ANGLE_TRIANGLE=math.radians(60),
        W=W, H=H
    )
    roi_helper.get_roi()
    if not getattr(roi_helper, "corner_points", None) or len(roi_helper.corner_points) != 3:
        raise SystemExit("[ERR] ROI not set (need 3 points).")

    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"[ERR] Could not open {VIDEO_PATH or CAM_DEVICE}")

    ok, frame0 = safe_read(cap, flush=SAFE_FLUSH)
    if not ok:
        raise SystemExit("[ERR] Camera read failed at start.")
    frame0 = rotate(frame0, roi_helper.ROTATE_CW_DEG)
    frame0 = cv.flip(frame0, roi_helper.FLIPCODE)
    frame0 = cv.resize(frame0, (roi_helper.W, roi_helper.H))



    DANGER_YFRAC = 0.85
    EDGE_PAD = 4
    roi_mask, danger_mask = roi_helper.build_masks(
        frame0.shape, danger_frac=DANGER_YFRAC, edge_pad=EDGE_PAD
    )

    os.makedirs("output", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)

    out_path = os.path.join("output", "result_combined.avi")
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out_w = int(W * OUT_SCALE * 3)
    out_h = int(H * OUT_SCALE)
    writer = cv.VideoWriter(out_path, fourcc, FPS, (out_w, out_h))
    print(f"[INFO] Writing combined video to: {out_path}")

    log_file = os.path.join("output/logs", "detection_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Object Detection Log ===\n")
    print(f"[INFO] Logging to: {log_file}")

    sp = StaticParams()
    frame_id = 0
    while True:
        ok, frame = safe_read(cap, flush=SAFE_FLUSH)
        if not ok:
            print("[INFO] End of stream.")
            break
        frame_id += 1

        frame = rotate(frame, roi_helper.ROTATE_CW_DEG)
        frame = cv.flip(frame, roi_helper.FLIPCODE)
        frame = cv.resize(frame, (roi_helper.W, roi_helper.H))

        if USE_BLUR:
            frame = cv.GaussianBlur(frame, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)

        start_t = time.time()
        stop, bbox, dbg = static_stop_detect(frame, roi_mask, danger_mask, sp)
        elapsed_ms = (time.time() - start_t) * 1000

        # --- Draw overlay ---
        vis = frame.copy()
        bbox_info = "None"
        if bbox is not None:
            x, y, bw, bh = bbox
            cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            bbox_info = f"x={x},y={y},w={bw},h={bh}"

        if stop:
            cv.putText(vis, "STOP", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Prepare masks for combine ---
        nf_color = cv.cvtColor(dbg["nonfloor"], cv.COLOR_GRAY2BGR)
        nd_color = cv.cvtColor(dbg["nf_danger"], cv.COLOR_GRAY2BGR)

        vis_s = cv.resize(vis, (int(W * OUT_SCALE), int(H * OUT_SCALE)))
        nf_s  = cv.resize(nf_color, (int(W * OUT_SCALE), int(H * OUT_SCALE)))
        nd_s  = cv.resize(nd_color, (int(W * OUT_SCALE), int(H * OUT_SCALE)))
        combined = np.hstack((vis_s, nf_s, nd_s))
        writer.write(combined)

        # --- Display ---
        if SHOW_DEBUG_WINDOWS:
            cv.imshow("Combined", combined)

        # --- Log debug info ---
        # Area% is the percentage of the counted detected / ROI region
        log_msg = (
            f"Frame {frame_id:05d} | STOP={stop} | {bbox_info} | "
            f"area%={dbg['area_pct']:.2f} | elong={dbg['elong']:.2f} | fill={dbg['fill']:.2f} | elapsed={elapsed_ms:.1f}ms"
        )
        log_message(log_file, log_msg)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()
    print(f"[INFO] Finished. Logs saved at: {log_file}")

if __name__ == "__main__":
    main()
