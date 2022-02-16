# import
import math
import pickle
from pprint import pprint
from threading import Thread
from time import sleep, time

import cv2
import numpy as np
import pandas as pd
import pyautogui as pg
import seaborn as sns
from joblib import Parallel, delayed
from line_profiler import LineProfiler
from matplotlib import pyplot as plt
from numba import jit
from sklearn import neural_network, svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sqlalchemy import true
from tqdm import tqdm
from tqdm.notebook import trange

camera = cv2.VideoCapture(0)
CAMERA_SIZE = (50, 85)

LEFT_ACTCODE = 0
RIGHT_ACTCODE = 1
LEFT90_ACTCODE = 2
RIGHT90_ACTCODE = 3
STRAIGHT_ACTCODE = 4


def SaveInstances(instance, path):
    '''
    PickleモジュールのWrapper
    instanceのオブジェクトをpathで指定したファイルパスに保存する。
    '''
    with open(path, mode='wb') as file:
        pickle.dump(instance, file, protocol=2)


def LoadInstances(path):
    '''
    PickleモジュールのWrapper
    pathに保存されているpickle形式のファイルをオブジェクトとして読み込む。
    '''
    with open(path, 'rb') as ins:
        return pickle.load(ins)


model_color = LoadInstances("C:/Users/Haya/OneDrive/DevlopingProjects/RoboCup_Cam/model_one/ColorModel.pickle")
model_decision = LoadInstances("C:/Users/Haya/OneDrive/DevlopingProjects/RoboCup_Cam/model_one/NeuralDecision.pickle")

# カメラで撮影する
camera = cv2.VideoCapture(0)
while True:
    ret, raw_frame = camera.read()
    if ret:
        frame = cv2.resize(src=raw_frame, dsize=CAMERA_SIZE)
        height, width = frame.shape[:2]

        result = [model_color.predict(row) for row in reversed(frame)]

        """
        # カメラからの写真を表示する
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig = plt.figure(1, (5., 5.))
        plt.imshow(img)
        plt.pause(0.1)
        # 写真を色識別する
        plt.clf()
        plt.pcolormesh(result)
        plt.pause(0.01)
        """

        res = model_decision.predict([np.ravel(result)])[0]

        acttype = [
            [LEFT_ACTCODE, "LEFT", ],
            [RIGHT_ACTCODE, "RIGHT", ],
            [LEFT90_ACTCODE, "LEFT90", ],
            [RIGHT90_ACTCODE, "RIGHT90", ],
            [STRAIGHT_ACTCODE, "STRAIGHT", ],
        ]

        for at in acttype:
            if res == at[0]:
                print(at[1])
