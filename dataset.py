import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
from keymap import W, A, S, D, PressKey, ReleaseKey

bounding_box = {'top': 32,
                'left': 3,
                'width': 800,
                'height': 600}

low_threshold_max = 500
upper_threshold_max = 600
appSize_max = 20
blurr_max = 20
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


def roi(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(frame, mask)
    return masked


def draw_lines(frame, lines):
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

    lines = cv2.HoughLinesP(new_frame, 1, np.pi/180, 50, np.array([]), 150, 5)

    return lines


if __name__ == "__main__":

    for i in range(1, 4):
        print(i)
        time.sleep(1)
    """
    * L: 126 U: 195

    q* L: 156 U: 324
    """
    cv2.namedWindow(title_window)
    cv2.createTrackbar("Low Thresh", title_window, 156,
                       low_threshold_max, nothing)
    cv2.createTrackbar("Up Thresh", title_window, 324,
                       upper_threshold_max, nothing)
    cv2.createTrackbar("App size", title_window, 3,
                       appSize_max, nothing)

    while True:
        start = time.time()
        sct_img = sct.grab(bounding_box)
        frame = np.array(sct_img)

        l_thresh = cv2.getTrackbarPos('Low Thresh',
                                      title_window)
        u_thresh = cv2.getTrackbarPos('Up Thresh',
                                      title_window)
        app_size = cv2.getTrackbarPos('App Size',
                                      title_window)

        lines = detect_lines(
            frame, l_thresh, u_thresh, app_size)

        processed_frame = draw_lines(frame, lines)

        cv2.imshow(title_window, processed_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
