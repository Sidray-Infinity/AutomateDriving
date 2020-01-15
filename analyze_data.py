import numpy as np
import cv2
import os
import time

DIR = "TrainingData/SHUFFLED-460k-15%W"

file = "CAR_160x120_0.npy"


if __name__ == "__main__":
    data = np.load(os.path.join(DIR, file), allow_pickle=True)

    for d in data:
        frame = cv2.resize(d[0], (800, 600))
        cv2.imshow("test", frame)

        time.sleep(0.3)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
