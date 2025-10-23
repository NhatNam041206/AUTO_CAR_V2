import numpy as np
import cv2
import math


def rotate(img,ROTATE_CW_DEG):
    d = ROTATE_CW_DEG % 360
    if d == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def cluster_lines(lines,RHO_BIAS,ANGLE_BIAS):
    visited=[]
    lines_filtered=[]
    for i in range(len(lines)):
        if list(lines[i][0]) in visited:
            continue

        cluster, clustered = [], False
        for j in range(i, len(lines)):
            angleD = abs(lines[i][0][1] - lines[j][0][1])
            pD     = abs(lines[i][0][0] - lines[j][0][0])

            if angleD < ANGLE_BIAS and pD < RHO_BIAS:
                cluster.extend([lines[j]])
                visited.append(list(lines[j][0]))
                clustered = True

        # keep average   or   raw line
        out = np.mean(cluster, axis=0) if clustered else lines[i]
        lines_filtered.append(out)
    return np.array(lines_filtered)

def create_binary_quad(points, img_size=(480,640)):

    mask = np.zeros(img_size, dtype=np.uint8)

    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask, [pts], 255)

    return mask

'''def draw_detected_lines(frame, flines):
    min_ang = 180
    for rho, theta in flines:
        angle_x_axis = math.degrees(np.pi / 2 - theta)
        if -self.ACCEPT <= angle_x_axis <= self.ACCEPT:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            min_ang = min(min_ang, angle_x_axis)
    return frame, min_ang  '''  

def convert_angle(flines): # just normalized angles, not converting
    angles=[]
    for _,theta in flines:

        if theta>np.pi: #norm
            theta-=np.pi        
        # angle_x_axis=max(np.pi/2-normalized_angle,normalized_angle-np.pi/2) #Not used
        angles.append(theta)
    return angles


def apply_roi(frame, mask, corner_points, CROP_SIZE): # Feat: Move all steps in applying ROI from preprocessing_frame() to apply_roi()
    # Ensure mask is binary 0 or 255
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    # Apply ROI (keep region, paint outside white)
    mask_3ch = cv2.merge([mask] * 3)
    frame_roi = cv2.bitwise_and(frame, frame, mask=mask)
    frame_roi[mask_3ch == 0] = 255

    # Crop bottom band based on base points
    y_bottom = max(corner_points[0][1], corner_points[1][1])
    y_crop = max(0, y_bottom - CROP_SIZE)
    frame_roi = frame_roi[:y_crop, :]


    return frame_roi

def preprocess_frame(frame, ROTATE_CW_DEG, FLIPCODE, W, H):
    frame = rotate(frame, ROTATE_CW_DEG)
    frame = cv2.flip(frame, FLIPCODE)
    frame = cv2.resize(frame, (W, H))

    return frame

def detect_edges(frame, CANNY_T1, CANNY_T2, CANNY_APER):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)
    return edges

def detect_lines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH):
    return cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)
