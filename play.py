from dataset import grab_screen, REGION, WIDTH, HEIGHT, key_check
import numpy as np
from models import alex_net_keras
from keymap import W, S, A, D, PressKey, ReleaseKey
from keras.models import load_model
import cv2
import time

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)


def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


model = load_model(
    "TrainedModels/SHUFFLED-240K-16%W/model-SHUFFLED-240K-16%W-16.h5")

if __name__ == "__main__":

    classes = ['W', 'S', 'A', 'D', 'WA', 'WD', 'SA', 'SD', 'NK']

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=REGION)

            prediction = model.predict([screen.reshape(-1, 160, 120, 1)])[0]
            print(classes[np.argmax(prediction)])

            if np.argmax(prediction) == np.argmax(w):
                straight()

            elif np.argmax(prediction) == np.argmax(s):
                reverse()
            if np.argmax(prediction) == np.argmax(a):
                left()
            if np.argmax(prediction) == np.argmax(d):
                right()
            if np.argmax(prediction) == np.argmax(wa):
                forward_left()
            if np.argmax(prediction) == np.argmax(wd):
                forward_right()
            if np.argmax(prediction) == np.argmax(sa):
                reverse_left()
            if np.argmax(prediction) == np.argmax(sd):
                reverse_right()
            if np.argmax(prediction) == np.argmax(nk):
                no_keys()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'Q' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
