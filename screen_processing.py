import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
from keymap import W, A, S, D, PressKey, ReleaseKey
import math

"""
Some functions needed to detect the lanes on the road.
Used to enhance the performance of the model.
"""


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


def sort_lines(set):
    def sort_on_rho(line_bloc):
        return abs(line_bloc[0][0])

    set.sort(key=sort_on_rho)
    return set


def almost_ver(lines, tolerance):
    new_lines = []
    if lines is not None:
        for line in lines:
            theta = line[0][1]
            if (theta <= np.deg2rad(0 + tolerance) or theta >= np.deg2rad(180 - tolerance)):
                new_lines.append(line)
    return new_lines


def reduce_lines(lines, tolerance):
    """
    * Reduce the set of lines to just the two lanes.
    * Return just two lines
    if (theta <= np.deg2rad(0+tolerance) or theta >= np.deg2rad(180 - tolerance)):
    """

    new_lines = almost_ver(lines, tolerance)
    left = []
    right = []

    for line in new_lines:
        if line[0][0] >= 0:
            left.append(line)
        else:
            right.append(line)

    rho1 = 0
    theta1 = 0
    rho2 = 0
    theta2 = 0

    for line in left:
        rho1 += line[0][0]
        theta1 += line[0][1]

    for line in right:
        rho2 += line[0][0]
        theta2 += line[0][1]

    try:
        rho1 = rho1/len(left)
        theta1 = theta1/len(left)
        rho2 = rho2/len(right)
        theta2 = theta2/len(right)
    except:
        pass

    new_lines = [[[rho1, theta1]], [[rho2, theta2]]]

    return new_lines


def detect_lines(frame, l_thresh, u_thresh, app_size, tolerance):
    new_frame = frame
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.Canny(new_frame, l_thresh, u_thresh,
                          apertureSize=app_size,
                          L2gradient=True)
    new_frame = cv2.GaussianBlur(new_frame, (5, 5), 0)
    new_frame = roi(new_frame, [vertices])

    lines = cv2.HoughLines(new_frame, 1, np.pi/180, 300,
                           np.array([]))

    lines = reduce_lines(lines, tolerance)
    return lines


def draw_lines(frame, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            theta = lines[i][0][1]
            rho = lines[i][0][0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 6, cv2.LINE_AA)

    return frame


def roi(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(frame, mask)
    return masked


"""
* L: 126 U: 195
* L: 250 UL 350
* L: 156 U: 324
    cv2.namedWindow(title_window)
    cv2.createTrackbar("Low Thresh", title_window, 250,
                       low_threshold_max, nothing)
    cv2.createTrackbar("Up Thresh", title_window, 350,
                       upper_threshold_max, nothing)
    cv2.createTrackbar("App size", title_window, 3,
                       appSize_max, nothing)
    cv2.createTrackbar("Tolerance", title_window, 55,
                       tol_max, nothing)

    start = time.time()
        sct_img = sct.grab(bounding_box)
        frame = np.array(sct_img)

        l_thresh = cv2.getTrackbarPos('Low Thresh', title_window)
        u_thresh = cv2.getTrackbarPos('Up Thresh', title_window)
        app_size = cv2.getTrackbarPos('App Size', title_window)
        tolerance = cv2.getTrackbarPos('Tolerance', title_window)

        lines = detect_lines(
            frame, l_thresh, u_thresh, app_size, tolerance)

        processed_frame = draw_lines(frame, lines)

        cv2.imshow(title_window, processed_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break

"""
