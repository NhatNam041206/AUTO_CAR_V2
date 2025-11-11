import numpy as np
from helpers import convert_angle, cluster_lines, detect_edges, detect_lines


class Calibrate:
    def __init__(self,
        CANNY_T1 = 40,
        CANNY_T2 = 120,
        CANNY_APER = 3,
        HOUGH_RHO = 1,
        HOUGH_THETA = np.pi / 180,
        HOUGH_THRESH = 40,
        ANGLE_BIAS = 0.3,
        RHO_BIAS = 20):

        # Assign every parameters to class environment

        self.CANNY_T1=CANNY_T1
        self.CANNY_T2=CANNY_T2
        self.CANNY_APER=CANNY_APER
        self.HOUGH_RHO=HOUGH_RHO
        self.HOUGH_THETA=HOUGH_THETA
        self.HOUGH_THRESH=HOUGH_THRESH
        self.RHO_BIAS=RHO_BIAS
        self.ANGLE_BIAS=ANGLE_BIAS

    # ---------------- Main ----------------

    def update(self,frame):

        frame_ori = frame.copy()  # full-size camera frame

        # ---- Preprocess & analysis frames ----
        edges = detect_edges(frame,self.CANNY_T1, self.CANNY_T2, self.CANNY_APER)
        lines = detect_lines(edges, self.HOUGH_RHO, self.HOUGH_THETA, self.HOUGH_THRESH)

        # ---- Hough / control ----

        if lines is not None:
            flines = cluster_lines(lines, self.RHO_BIAS, self.ANGLE_BIAS).reshape(-1, 2)
            angles = np.array(convert_angle(flines))
            
            # Feat: change min_ang from taking min angles to take the angle with the smallest difference compared to the base angle (90)
            min_ang=min(angles, key=lambda x:abs(x-np.pi/2)) # Angle with smallest difference to 90 deg or pi/2 (rad)
            # If any near-horizontal line was found
            log = f'<LOG> max_angle: {np.rad2deg(np.max(angles))} min_angle: {np.rad2deg(np.min(angles))} taken_angle: '
            
            #The formula determined by acceptance_angle +- 90 (since we are taking raw theta instead of angle from the x-axis {converting the raw theta into this concept is still faced some troubles})
            if (np.pi/2-self.ANGLE_BIAS)<min_ang and min_ang<(self.ANGLE_BIAS+np.pi/2):
                return min_ang,log+str(np.rad2deg(min_ang)) # Only return the horizontal line's angle
            return None,log+'None'
        return None,'<LOG> Angle Not Found!' # Return None values if the lines are not found
