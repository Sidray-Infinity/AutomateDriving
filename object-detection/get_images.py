import numpy as np
import cv2
import time
import math
import win32gui
import win32ui
import win32con
import win32api
import os


REGION = (3, 32, 800, 600)
HEIGHT = 600
WIDTH = 800


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()
    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    img = cv2.resize(img, (WIDTH, HEIGHT))
    return img


if __name__ == "__main__":

    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    count = 222
    img_count = len(os.listdir("Images"))

    # if len(os.listdir("Images")) != 0:
    #     """
    #     If some files already exists, adjust the value of count.
    #     """
    #     print("Existing Data found.")
    #     files = os.listdir("Images")
    #     paths = [os.path.join("Images", basename) for basename in files]
    #     latest_file = max(paths, key=os.path.getctime)
    #     print("LATEST FILE:", latest_file)
    #     count = int(latest_file.split('.')[0].split('-')[1]) + 1
    #     print("COUNT:", count)
    #     print("--------------------------------------------------------")

    while True:
        frame = grab_screen(REGION)
        # cv2.imshow("Win", frame)
        cv2.imwrite(f"Images/image-{count}.jpg", frame)
        print("Image Count:", count)
        #img_count += 1
        count += 1
        time.sleep(5)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
