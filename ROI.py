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


    def on_click_roi(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corner_points) < 2:
            if len(self.corner_points) == 0:
                self.corner_points.append([x, y])
            elif x > self.corner_points[0][0]:
                self.corner_points.append([x, self.corner_points[0][1]])

    def save_points(self):
        np.savetxt(self.saved_path, np.array(self.corner_points), fmt='%d')
        print(f"Saved corner points to {self.saved_path}")

    def load_points(self):
        if os.path.exists(self.saved_path):
            self.corner_points = np.loadtxt(self.saved_path, dtype=int).tolist()
            if isinstance(self.corner_points[0], int):
                self.corner_points = [self.corner_points]
            print(f"Loaded corner points from {self.saved_path}")
            if len(self.corner_points) == 3:
                self.roi_created = True

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
        camera_option=int(input('Camera Source (0/1/2): '))
        choice = input("Use saved corner points? (y/n): ").strip().lower()
        if choice == 'y':
            self.load_points()
            return

        cap = cv2.VideoCapture(camera_option)
        if not cap.isOpened():
            raise IOError("Cannot open camera")

        cv2.namedWindow('inputted')
        cv2.namedWindow('raw')
        cv2.setMouseCallback('inputted', self.on_click_roi)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = rotate(frame, self.ROTATE_CW_DEG)
            frame = cv2.flip(frame, self.FLIPCODE)
            frame = cv2.resize(frame, (self.W, self.H))

            raw = frame.copy()
            for p in self.corner_points:
                cv2.circle(raw, tuple(p), 5, (0, 0, 255), -1)
                cv2.putText(raw, f"{p[0]},{p[1]}", tuple(p),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if len(self.corner_points) == 2:
                p1, p2 = self.corner_points
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                x_3 = int(round((p1[0] + p2[0]) / 2))
                height = (distance / 2) * math.tan(self.ANGLE_TRIANGLE)
                y_3 = int(round(p1[1] - height))
                apex = [x_3, y_3]
                self.corner_points.append(apex)
                cv2.circle(raw, (x_3, p1[1]), 5, (0, 0, 255), -1)
                cv2.circle(raw, (x_3, y_3), 5, (0, 0, 255), -1)

            if len(self.corner_points) == 3:
                cv2.line(raw, self.corner_points[0], self.corner_points[1], (0, 255, 255), 2)
                cv2.line(raw, self.corner_points[0], self.corner_points[2], (0, 255, 255), 2)
                cv2.line(raw, self.corner_points[1], self.corner_points[2], (0, 255, 255), 2)

            cv2.imshow('raw', frame)
            cv2.imshow('inputted', raw)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            if k in (ord('c'), ord('C')) and len(self.corner_points) == 3:
                self.roi_created = True
                break
            if k in (ord('r'), ord('R')):
                self.corner_points.clear()

        cap.release()
        cv2.destroyAllWindows()

        if self.roi_created:
            save = input("Save corner points to file? (y/n): ").strip().lower()
            if save == 'y':
                self.save_points()
