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
HEIGHT = 120
WIDTH = 160
TOTAL_TRAINING_SIZE = 400000
MAX_FILE_SIZE = 16000
DIR = "TrainingData/400K"

if(TOTAL_TRAINING_SIZE % MAX_FILE_SIZE != 0):
    print("Illegal training and file size combination.")
    exit(0)


KEY_MAP = {
    'W': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    'S': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    'A': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'WS': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'WD': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'SA': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'SD': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'NK': [0, 0, 0, 0, 0, 0, 0, 0, 1],
    'default': [0, 0, 0, 0, 0, 0, 0, 0, 0],
}

KEY_LIST = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    KEY_LIST.append(char)


def key_check():
    """
    Returns the currently active keys.
    """
    keys = []
    for key in KEY_LIST:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    if ''.join(keys) in KEY_MAP:
        return KEY_MAP[''.join(keys)]
    return KEY_MAP['default']


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
    img = cv2.resize(img, (160, 120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.
    return img


if __name__ == "__main__":

    paused = False
    training_data = []
    count = 0
    file_count = 0

    if len(os.listdir(DIR)) != 0:
        """
        If some files already exists, adjust the value of count and file_count.
        """
        print("Existing Data found.")
        files = os.listdir(DIR)
        print("Existing files:")
        print("--------------------------------------------------------")
        for i, f in enumerate(files):
            print('{}.'.format(i+1), f)

        paths = [os.path.join(DIR, basename) for basename in files]
        latest_file = max(paths, key=os.path.getctime)
        print("LATEST FILE:", latest_file)
        file_count = int(latest_file.split('_')[3].split('.')[0]) + 1
        count = file_count * MAX_FILE_SIZE
        print("FILE COUNT:", file_count)
        print("TRAINING DATA ITEMS:", count)
        print("--------------------------------------------------------")

    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)

    while True:
        if not paused:
            start = time.time()
            frame = grab_screen(REGION)
            # cv2.imshow("Test", img)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([frame, output])

            len_train = len(training_data)

            if count % 50 == 0:
                print("TRAINING SIZE:", count, end='\r', flush=True)

            if len_train == MAX_FILE_SIZE:
                file_name = os.path.join(
                    DIR, "CAR_TD_400K_{}.npy".format(file_count))
                print("--------------------------------------------------------")
                print(file_name)
                print("--------------------------------------------------------")
                np.save(file_name, training_data)
                file_count += 1
                training_data = []

            if count == TOTAL_TRAINING_SIZE:
                print("COMPLETED.")
                break

            count += 1

        keys = key_check()
        if 'Q' in keys:
            if paused:
                paused = False
                print('UNPAUSED')
                time.sleep(1)
            else:
                print('PAUSED')
                paused = True
                time.sleep(1)

        # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
