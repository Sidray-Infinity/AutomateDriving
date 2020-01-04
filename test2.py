import numpy as np
import cv2
import time

data = np.load(
    "D:\AutomateDriving\TrainingData\SHUFFLED-240K-16%W\CAR_160x120_5.npy", allow_pickle=True)

for d in data:
    cv2.imshow("test", d[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.4)
