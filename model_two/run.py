# import
from matplotlib.colors import ListedColormap
import math
import os
import pickle
import random
import socket
import sys
import warnings
from glob import glob
from pprint import pprint
from threading import Thread
from time import sleep, time

import cv2
import numpy as np
import pandas as pd
import pyautogui as pg
import seaborn as sns
import sklearn
from line_profiler import LineProfiler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm


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


colors = ["white", "black", "green", "red"]
cmap = ListedColormap(colors, name="custom")

warnings.simplefilter('ignore')  # すべての警告を非表示にする
col_model = LoadInstances(".RoboCup_Cam/model_two/ColorReco2.pickle")
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

features_for_video = []

OFFSET_CHECK_EXIST_BLACK = 12
BLACK_COUNT = 3
HOST = '127.0.0.1'
PORT = 50007

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

start = time()

while start + 45 > time():
    # 写真を読み込む
    ret, frame = capture.read()
    #
    frame = cv2.resize(frame, (64, 36))
    height, width = frame.shape[:2]
    #
    # 色を識別する
    recognized_frame = np.flipud(
        np.reshape(
            col_model.predict(np.reshape(frame, (-1, 3))),
            np.shape(frame)[:2]
        )
    )
    #
    # 画面上部と画面下部における黒の総量
    temp_frame_height_middle = math.floor(len(recognized_frame) / 2)
    black_amount_bottom = np.sum(np.where(recognized_frame[:temp_frame_height_middle, :] == 1, 1, 0))
    black_amount_upper = np.sum(np.where(recognized_frame[temp_frame_height_middle:, :] == 1, 1, 0))
    #
    # 画像縦列における黒の出現割合
    black_distribution_weighted_bottom = np.sum(np.array(np.where(recognized_frame == 1, 1, 0)) *
                                                np.reshape(np.arange(len(recognized_frame), 0, -1), (-1, 1)), axis=0)
    black_distribution_weighted_upper = np.sum(np.array(np.where(recognized_frame == 1, 1, 0)) *
                                               np.reshape(np.arange(0, len(recognized_frame), 1), (-1, 1)), axis=0)
    #
    # 画像縦列における黒の出現位置の中心インデックス
    black_center_position_weighted_bottom = np.round((np.sum(black_distribution_weighted_bottom * np.arange(
        0, len(black_distribution_weighted_bottom), 1))) / np.sum(black_distribution_weighted_bottom) / width - 0.5, 2) * 100
    black_center_position_weighted_upper = np.round((np.sum(black_distribution_weighted_upper * np.arange(0,
                                                    len(black_distribution_weighted_upper), 1))) / np.sum(black_distribution_weighted_upper) / width - 0.5, 2) * 100
    #
    # 画像全体における赤の出現インデックスを抽出
    raw_red_appeared = np.where(recognized_frame == 3)
    red_appeared = np.hstack((np.reshape(raw_red_appeared[0], (-1, 1)), np.reshape(raw_red_appeared[1], (-1, 1))))  # ["height", "width"]
    occupy_red_appeared = np.round(len(red_appeared) / (height * width), 2)
    #
    # 画像全体における緑の出現インデックスを抽出
    is_green_found = len(np.where(recognized_frame == 2)[0]) > 4
    temp_recognized_frame = recognized_frame

    green_KMeans_center = None
    recognized_frame_for_display = np.copy(recognized_frame)
    if is_green_found:
        # 緑の周囲に黒が存在するかを確認するための配列を作る
        temp_recognized_frame = np.zeros_like(recognized_frame)
        temp_enabled_zone = np.s_[:OFFSET_CHECK_EXIST_BLACK * -1, OFFSET_CHECK_EXIST_BLACK:OFFSET_CHECK_EXIST_BLACK * -1]
        temp_recognized_frame[temp_enabled_zone] = recognized_frame[temp_enabled_zone]
        #
        # 緑色であるピクセルの位置インデックス
        temp_green_appeared = np.where(temp_recognized_frame == 2)
        green_appeared = (
            np.hstack((np.reshape(temp_green_appeared[0], (-1, 1)), np.reshape(temp_green_appeared[1], (-1, 1)))))  # ["height", "width"]
        #
        if len(green_appeared) > 4:
            # 緑色のピクセルインデックスを４つに絞り込む
            green_KMeans_center = np.array(MiniBatchKMeans(n_clusters=4).fit(green_appeared).cluster_centers_).astype(np.int16)
            #
            # 全体に占める緑色の割合
            occupy_green_appeared = np.round(len(green_appeared) / (height * width), 2)
            #
            green_cluster_around = np.zeros((1, 3))
            print(green_KMeans_center)
            for i in range(len(green_KMeans_center)):
                x = green_KMeans_center[i, 0]
                y = green_KMeans_center[i, 1]
                of = OFFSET_CHECK_EXIST_BLACK
                ones = np.ones(OFFSET_CHECK_EXIST_BLACK).astype(np.int16)
                upper = np.sum(np.where(recognized_frame == 1, 1, 0)[np.arange(x, x + of, 1), ones * y])
                right = np.sum(np.where(recognized_frame == 1, 1, 0)[ones * x, np.arange(y, y + of, 1)])
                left = np.sum(np.where(recognized_frame == 1, 1, 0)[ones * x, np.arange(y, y - of, -1)])
                print([left, upper, right])
                green_cluster_around = np.append(green_cluster_around, [[left, upper, right]], axis=0)
                # @for anchor
            green_cluster_around = np.round(green_cluster_around[1:, :], 2)
            temp = np.where(green_cluster_around[1] >= BLACK_COUNT)
            is_this_meant_right = True in (green_cluster_around[temp, 0] >= BLACK_COUNT)
            print(green_cluster_around)
            is_this_meant_left = True in (green_cluster_around[temp, 2] >= BLACK_COUNT)
            is_this_meant_turn = is_this_meant_right and is_this_meant_left   # is_this_meant_left and is_this_meant_right
            print(is_this_meant_left, is_this_meant_right)
            # 表示用に加工（方向判断のあとに実行）
            recognized_frame_for_display[green_KMeans_center[:, 0], green_KMeans_center[:, 1]] = 3
            for i in range(len(green_KMeans_center)):
                x = green_KMeans_center[i, 0]
                y = green_KMeans_center[i, 1]
                of = OFFSET_CHECK_EXIST_BLACK
                recognized_frame[x, y] = 3
                recognized_frame[np.arange(x, x + of, 1), y] = 3
                recognized_frame[x, np.arange(y, y + of, 1)] = 3
                recognized_frame[x, np.arange(y, y - of, -1)] = 3
            #
    else:
        # タイルに緑色がない
        occupy_green_appeared = 0.0
        is_this_meant_right = False
        is_this_meant_left = False
        is_this_meant_turn = False
    #
    # 表示
    # メッシュ図の表示用に調整する
    for array in [recognized_frame, temp_recognized_frame, recognized_frame_for_display]:
        array[0][0] = 0  # meant white
        array[0][1] = 1  # meant black
        array[0][2] = 2  # meant green
        array[0][3] = 3  # meant red
    #
    # 表示
    features = {
        "black_amount_bottom": black_amount_bottom,
        "black_amount_upper": black_amount_upper,
        "黒出現中心(下)": black_center_position_weighted_bottom,
        "黒出現中心(上)": black_center_position_weighted_upper,
        "緑": occupy_green_appeared,
        "赤": occupy_red_appeared,
        "交差点 右折": is_this_meant_right,
        "交差点 左折": is_this_meant_left,
        "交差点 Uターン": is_this_meant_turn
    }
    # pprint(features, width=1)
    # recognized_frame_for_display = np.copy(temp_recognized_frame)
    # plt.figure(figsize=(16, 10))
    # plt.subplot(221)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # -> 注意
    # plt.subplot(222)
    # plt.pcolormesh(recognized_frame, cmap=cmap)
    # plt.subplot(223)
    # plt.plot(black_distribution_weighted_bottom)
    # plt.plot(black_distribution_weighted_upper)
    # plt.subplot(224)
    # plt.pcolormesh(recognized_frame_for_display, cmap=cmap)
    # plt.savefig(f"K:/{time()}.png", dpi=200)
    features_for_video.append([
        frame,
        recognized_frame,
        black_distribution_weighted_bottom,
        black_distribution_weighted_upper,
        recognized_frame_for_display])
    #
    # 操作
    control = (black_center_position_weighted_bottom * 1.5 + black_center_position_weighted_upper * 1) / 2.5
    if features["交差点 右折"]:
        message = "right"  # 右に曲がりたい
    elif features["交差点 左折"]:
        message = "left"  # 左に曲がりたい
    else:
        message = "normal"
    if control is not np.nan:
        client.sendto((str(control) + "," + message).encode('utf-8'), (HOST, PORT))


client.sendto("0,stop".encode('utf-8'), (HOST, PORT))

for i, feature in enumerate(tqdm(features_for_video)):
    plt.figure(figsize=(6, 6))
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(feature[0], cv2.COLOR_BGR2RGB))  # -> 注意
    plt.subplot(222)
    plt.pcolormesh(feature[1], cmap=cmap)
    plt.subplot(223)
    plt.plot(feature[2])
    plt.plot(feature[3])
    plt.subplot(224)
    plt.pcolormesh(feature[4], cmap=cmap)
    plt.savefig(f"K:/{time()}.png", dpi=100)
