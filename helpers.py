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

def convert_angle(flines): # just normalized angles, not converting
    angles=[]
    for _,theta in flines:

        if theta>np.pi: #norm
            theta-=np.pi        
        # angle_x_axis=max(np.pi/2-normalized_angle,normalized_angle-np.pi/2) #Not used
        angles.append(theta)
    return angles

def detect_edges(frame, CANNY_T1, CANNY_T2, CANNY_APER):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2, apertureSize=CANNY_APER)
    return edges

def detect_lines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH):
    return cv2.HoughLines(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH)


def draw_arrow_by_angle(image, start_point, angle_degrees, length, color, thickness):
    
    angle_radians = math.radians(angle_degrees)
    
    x1, y1 = start_point
    x2 = int(x1 + length * math.cos(angle_radians))
    y2 = int(y1 - length * math.sin(angle_radians))
    end_point = (x2, y2)
    
    cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.2)