from models import alexnet
import os
import numpy as np
import pandas as pd

"""
[W, S, A, D, WA, WD, SA, SD, NK]
"""
W = [1, 0, 0, 0, 0, 0, 0, 0, 0]
S = [0, 1, 0, 0, 0, 0, 0, 0, 0]
A = [0, 0, 1, 0, 0, 0, 0, 0, 0]
D = [0, 0, 0, 1, 0, 0, 0, 0, 0]
WA = [0, 0, 0, 0, 1, 0, 0, 0, 0]
WD = [0, 0, 0, 0, 0, 1, 0, 0, 0]
SA = [0, 0, 0, 0, 0, 0, 1, 0, 0]
SD = [0, 0, 0, 0, 0, 0, 0, 1, 0]
NK = [0, 0, 0, 0, 0, 0, 0, 0, 1]
HEIGHT = 120
WIDTH = 160
DIR = "TrainingData/SHUFFLED-240K-16%W"
EPOCHS = 5
MODEL_NAME = ''
SAVE_DIR = 'D:/AutomateDriving/TrainedModels/SHUFFLED-240K-16%W'

model = alexnet(WIDTH, HEIGHT, 1e-3, output=9)

for epoch in range(EPOCHS):
    for f in os.listdir(DIR):
        print("--------------------------------------------")
        file_name = os.path.join(DIR, f)
        print(epoch, file_name)
        data = np.load(file_name, allow_pickle=True)
        train = data[:-50]
        test = data[-50:]

        train_x = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
        train_y = np.array([i[1] for i in train])

        test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
        test_y = np.array([i[1] for i in test])

        model.fit({'input': train_x}, {'targets': train_y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=2500, show_metric=False, run_id=MODEL_NAME)
        print("--------------------------------------------")

    model.save(os.path.join(
        SAVE_DIR, "model-{}-TEpochs-{}.tfl".format(epoch, EPOCHS)))

# tensorboard --logdir=foo:D:\AutomateDriving\log
