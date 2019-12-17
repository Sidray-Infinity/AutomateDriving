import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
from keymap import W, A, S, D, PressKey, ReleaseKey
import math

bounding_box = {'top': 32,
                'left': 3,
                'width': 800,
                'height': 600}

low_threshold_max = 500
upper_threshold_max = 600
appSize_max = 20
blurr_max = 20
tol_max = 150
title_window = "Dynamic Canny"

vertices = np.array([[0, 600],
                     [0, 400],
                     [230, 305],
                     [580, 305],
                     [800, 400],
                     [800, 600]])

sct = mss()


def nothing(x):
    pass


def only_ver_linesP(lines):
    """
    ASSUMPTION: Probabilistic Hough transform is used.
    """
    new_lines = []
    MIN_ANGLE = 60
    MAX_ANGLE = 120
    print("MIN: ", np.radians(MIN_ANGLE))
    print("MAX: ", np.radians(MAX_ANGLE))
    if lines is not None:
        for line in lines:
            slope = (line[0][3]-line[0][1])/(line[0][2]-line[0][0])
            angle = np.arctan(slope)
            print(angle)
            if angle >= np.radians(MIN_ANGLE) and angle <= np.radians(MAX_ANGLE):
                new_lines.append(line)
    print("LEN NEW LINES:", len(new_lines))
    return new_lines


def draw_linesP(frame, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(frame, (l[0], l[1]),
                     (l[2], l[3]), (0, 255, 0), 3, cv2.LINE_AA)
    return frame


def detect_lines(frame, l_thresh, u_thresh, app_size):
    new_frame = frame
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.Canny(new_frame, l_thresh, u_thresh,
                          apertureSize=app_size,
                          L2gradient=True)
    new_frame = cv2.GaussianBlur(new_frame, (5, 5), 0)
    new_frame = roi(new_frame, [vertices])

    lines = cv2.HoughLines(new_frame, 1, np.pi/90, 300,
                           np.array([]))
    return lines


def draw_lines(frame, lines, tolerance):
    if lines is not None:
        for i in range(0, len(lines)):
            theta = lines[i][0][1]
            rho = lines[i][0][0]
            if (theta <= np.deg2rad(0+tolerance) or theta >= np.deg2rad(180 - tolerance)):
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

    return frame


def roi(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(frame, mask)
    return masked


if __name__ == "__main__":

    for i in range(1, 4):
        print(i)
        time.sleep(1)
    """
    * L: 126 U: 195
    * L: 250 UL 350
    * L: 156 U: 324
    """
    cv2.namedWindow(title_window)
    cv2.createTrackbar("Low Thresh", title_window, 250,
                       low_threshold_max, nothing)
    cv2.createTrackbar("Up Thresh", title_window, 350,
                       upper_threshold_max, nothing)
    cv2.createTrackbar("App size", title_window, 3,
                       appSize_max, nothing)
    cv2.createTrackbar("Tolerance", title_window, 40,
                       tol_max, nothing)

    while True:
        start = time.time()
        sct_img = sct.grab(bounding_box)
        frame = np.array(sct_img)

        l_thresh = cv2.getTrackbarPos('Low Thresh', title_window)
        u_thresh = cv2.getTrackbarPos('Up Thresh', title_window)
        app_size = cv2.getTrackbarPos('App Size', title_window)
        tolerance = cv2.getTrackbarPos('Tolerance', title_window)

        lines = detect_lines(
            frame, l_thresh, u_thresh, app_size)

        processed_frame = draw_lines(frame, lines, tolerance)

        cv2.imshow(title_window, processed_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
