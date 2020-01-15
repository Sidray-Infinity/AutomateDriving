
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from random import shuffle
from collections import Counter

DIR = "TrainingData/RAW"
TAR = "TrainingData/model"


W = [1, 0, 0, 0, 0, 0, 0, 0, 0]
S = [0, 1, 0, 0, 0, 0, 0, 0, 0]
A = [0, 0, 1, 0, 0, 0, 0, 0, 0]
D = [0, 0, 0, 1, 0, 0, 0, 0, 0]
WA = [0, 0, 0, 0, 1, 0, 0, 0, 0]
WD = [0, 0, 0, 0, 0, 1, 0, 0, 0]
SA = [0, 0, 0, 0, 0, 0, 1, 0, 0]
SD = [0, 0, 0, 0, 0, 0, 0, 1, 0]
NK = [0, 0, 0, 0, 0, 0, 0, 0, 1]

classes = ['W', 'S', 'A', 'D', 'WA', 'WD', 'SA', 'SD', 'NK']

GLOBAL_COUNTER = {
    'W': 0,
    'S': 0,
    'A': 0,
    'D': 0,
    'WA': 0,
    'WD': 0,
    'SA': 0,
    'SD': 0,
    'NK': 0,
}


def get_key(val):
    return classes[np.argmax(val)]


# def analyze():
#     for i in range(2):
#         vals = []
#         LOCAL_COUNTER = {
#            'W': 0,
#             'S': 0,
#             'A': 0,
#             'D': 0,
#             'WA': 0,
#             'WD': 0,
#             'SA': 0,
#             'SD': 0,
#             'NK': 0,
#         }

#         print("------------------------------------------------------")
#         print(os.path.join(DIR, f"CAR_160x120_{i}.npy"))
#         data = np.load(os.path.join(DIR, f"CAR_160x120_{i}.npy"), allow_pickle=True)

#         for i in range(16000):
#             LOCAL_COUNTER[get_key(data[i][1])] += 1
#             GLOBAL_COUNTER[get_key(data[i][1])] += 1
#         print(LOCAL_COUNTER)

# #         for c in classes:
# #             vals.append(GLOBAL_COUNTER[c])
#         plt.clf()
#         plt.bar(range(len(GLOBAL_COUNTER)), list(GLOBAL_COUNTER.values()), align='center')
#         plt.xticks(range(len(GLOBAL_COUNTER)), list(GLOBAL_COUNTER.keys()))
#         plt.show()

#     print("GLOBAL: ", GLOBAL_COUNTER)


# analyze()


def BalanceEachFile(DIR):

    MAX_W = 2560
    GLOBAL = {
        'W': 0,
        'S': 0,
        'A': 0,
        'D': 0,
        'WA': 0,
        'WD': 0,
        'SA': 0,
        'SD': 0,
        'NK': 0,
    }
    GLOBAL_TOTAL = 0

    for j in range(63):
        w_count = 0
        file_total = 0
        file_name = os.path.join(DIR, f"CAR_160x120_{j}.npy")
        key_counter = {
            'W': 0,
            'S': 0,
            'A': 0,
            'D': 0,
            'WA': 0,
            'WD': 0,
            'SA': 0,
            'SD': 0,
            'NK': 0,
        }

        print(file_name)
        data = np.load(file_name, allow_pickle=True)
        shuffle(data)
        new_data = []

        for i in range(16000):
            if data[i][1] == W:
                if w_count < MAX_W:
                    w_count += 1
                    new_data.append(data[i])
                    GLOBAL[get_key(data[i][1])] += 1
                    key_counter[get_key(data[i][1])] += 1
                    file_total += 1
            else:
                new_data.append(data[i])
                GLOBAL[get_key(data[i][1])] += 1
                key_counter[get_key(data[i][1])] += 1
                file_total += 1

        GLOBAL_TOTAL += file_total

        print("FILE:", key_counter)
        print("FILE TOTAL:", file_total)
        np.save(os.path.join(TAR, f"CAR_160x120_{j}.npy"), new_data)

        print("------------------------------------------------------")

    print("GLOBAL:", GLOBAL)
    print("GLOBAL TOTAL:", GLOBAL_TOTAL)

    plt.bar(range(len(GLOBAL)), list(GLOBAL.values()), align='center')
    plt.xticks(range(len(GLOBAL)), list(GLOBAL.keys()))
    plt.show()


BalanceEachFile(DIR)
