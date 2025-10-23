import cv2,os,math
import numpy as np
from helpers import rotate
# ---------------- ROI utils ----------------
class ROI:
    def __init__(self, saved_path, ROTATE_CW_DEG=0, FLIPCODE=1, ANGLE_TRIANGLE=math.radians(60),W=480,H=640):
        self.saved_path=saved_path
        self.ROTATE_CW_DEG=ROTATE_CW_DEG
        self.FLIPCODE=FLIPCODE
        self.ANGLE_TRIANGLE=ANGLE_TRIANGLE
        self.W,self.H=W,H
        self.corner_points = []
        self.roi_created = False

    def _pt(self, p):
        # accept list/tuple/np types and return (int,int)
        return (int(p[0]), int(p[1]))


    def on_click_roi(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corner_points) < 2:
            if len(self.corner_points) == 0:
                self.corner_points.append([x, y])
            elif x > self.corner_points[0][0]:
                self.corner_points.append([x, self.corner_points[0][1]])

    def save_points(self):
        np.savetxt(self.saved_path, np.array(self.corner_points), fmt='%d')
        print(f"Saved corner points to {self.saved_path}")

    # GPT INCLUDED
    def _ensure_three_points(self):
        # If only two base points are stored, compute the apex using ANGLE_TRIANGLE
        if len(self.corner_points) == 2:
            p1, p2 = self.corner_points
            distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            x_3 = int(round((p1[0] + p2[0]) / 2))
            height = (distance / 2) * math.tan(self.ANGLE_TRIANGLE)
            y_3 = int(round(p1[1] - height))
            apex = [x_3, y_3]
            self.corner_points.append(apex)

    def load_points(self):
        if not os.path.exists(self.saved_path):
            print(f"[ROI] File not found: {self.saved_path}")
            return
        try:
            pts = np.loadtxt(self.saved_path, dtype=int)
        except Exception as e:
            print(f"[ROI] Failed to read {self.saved_path}: {e}")
            return
        # Normalize shapes: (2,) → (1,2), (N,2) stays
        pts = np.array(pts).reshape(-1, 2)
        self.corner_points = pts.tolist()
        self._ensure_three_points()
        self.roi_created = (len(self.corner_points) == 3)
        print(f"Loaded corner points from {self.saved_path}: {self.corner_points}")

    # GPT 

    def build_masks(self, frame_shape, danger_frac=0.35, edge_pad=0):
        h, w = frame_shape[:2]
        poly = np.array(self.corner_points, dtype=np.int32)

        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(roi_mask, poly, 255)

        band_mask = np.zeros((h, w), np.uint8)
        y0 = int(h * (1.0 - danger_frac))
        cv2.rectangle(band_mask, (0, y0), (w, h), 255, -1)

        if edge_pad > 0:
            inner = np.zeros_like(roi_mask)
            inner[edge_pad:h - edge_pad, edge_pad:w - edge_pad] = 255
            roi_mask = cv2.bitwise_and(roi_mask, inner)
            band_mask = cv2.bitwise_and(band_mask, inner)

        danger_mask = cv2.bitwise_and(roi_mask, band_mask)
        return roi_mask, danger_mask


def get_roi(self):
    """
    Interactively pick a triangular ROI (2 base clicks -> auto apex) or load it from file.
    - Press 'c' to confirm once the triangle is shown.
    - Press 'r' to reset points.
    - Press 'q' to quit without saving.
    Returns True if ROI is ready (loaded or created), else False.
    """
    # --- choose camera & load/create option ---
    try:
        camera_option = int(input('Camera Source (0/1/2): ').strip())
    except Exception:
        camera_option = 0

    choice = input("Use saved corner points? (y/n): ").strip().lower()

    # Try load from file if requested
    if choice == 'y':
        self.load_points()
        if self.roi_created and len(self.corner_points) == 3:
            # Normalize to plain Python ints
            self.corner_points = [[int(x), int(y)] for x, y in self.corner_points]
            print(f"[ROI] Using saved points: {self.corner_points}")
            return True
        else:
            print("[ROI] No valid saved ROI. Switching to interactive selection...")

    # Otherwise, create new interactively
    self.corner_points = []
    self.roi_created = False

    cap = cv2.VideoCapture(camera_option)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    # Make preview match your runtime size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)

    cv2.namedWindow('inputted')
    cv2.namedWindow('raw')
    cv2.setMouseCallback('inputted', self.on_click_roi)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Apply SAME transforms you will use at runtime
            frame = rotate(frame, self.ROTATE_CW_DEG)
            frame = cv2.flip(frame, self.FLIPCODE)
            frame = cv2.resize(frame, (self.W, self.H))

            raw = frame.copy()

            # Draw any already-clicked base points
            for p in self.corner_points:
                cv2.circle(raw, self._pt(p), 5, (0, 0, 255), -1)
                cv2.putText(raw, f"{int(p[0])},{int(p[1])}", self._pt(p),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # If 2 base points chosen, compute apex once and draw helpers
            if len(self.corner_points) == 2:
                p1, p2 = self.corner_points
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                x_3 = int(round((p1[0] + p2[0]) / 2))
                height = (distance / 2) * math.tan(self.ANGLE_TRIANGLE)
                y_3 = int(round(p1[1] - height))
                apex = [x_3, y_3]
                # Append apex exactly once
                self.corner_points.append([int(apex[0]), int(apex[1])])

                cv2.circle(raw, self._pt((x_3, p1[1])), 5, (0, 0, 255), -1)
                cv2.circle(raw, self._pt((x_3, y_3)), 5, (0, 0, 255), -1)

            # If we have a full triangle, draw its edges
            if len(self.corner_points) == 3:
                cv2.line(raw, self._pt(self.corner_points[0]), self._pt(self.corner_points[1]), (0, 255, 255), 2)
                cv2.line(raw, self._pt(self.corner_points[0]), self._pt(self.corner_points[2]), (0, 255, 255), 2)
                cv2.line(raw, self._pt(self.corner_points[1]), self._pt(self.corner_points[2]), (0, 255, 255), 2)

            cv2.imshow('raw', frame)
            cv2.imshow('inputted', raw)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                self.roi_created = False
                break
            if k in (ord('r'), ord('R')):
                # Reset to empty (start over)
                self.corner_points.clear()
            if k in (ord('c'), ord('C')) and len(self.corner_points) == 3:
                self.roi_created = True
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if self.roi_created and len(self.corner_points) == 3:
        # Normalize to python ints
        self.corner_points = [[int(x), int(y)] for x, y in self.corner_points]
        save = input("Save corner points to file? (y/n): ").strip().lower()
        if save == 'y':
            self.save_points()
        return True

    print("[ROI] ROI not created.")
    return False
