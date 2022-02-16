import itertools
from threading import Thread
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
from line_profiler import LineProfiler
from numba import jit
from pyautogui import keyDown, keyUp

camera = cv2.VideoCapture(0)
fig, ax = plt.subplots(1, 1)

Can_thread_continue = True
act_type = None

FRAME_HEIGHT = 50
FRAME_WIDTH = 85

CODE_STOP = 0
CODE_GO = 1
CODE_RIGHT = 2
CODE_LEFT = 3
CODE_FRONT_RIGHT = 4
CODE_FRONT_LEFT = 5
CODE_GO_RIGHT = 6
CODE_GO_LEFT = 7

for_debug = [
    ["STOP", 0],
    ["GO", 1],
    ["RIGHT", 2],
    ["LEFT", 3],
    ["FRONT_RIGHT", 4],
    ["FRONT_LEFT", 5],
    ["GO_RIGHT", 6],
    ["GO_LEFT", 7],
]


def controll():
    global act_type, Can_thread_continue
    while Can_thread_continue:
        if act_type == CODE_STOP:  # 完全に停止
            keyUp("up")
            keyUp("left")
            keyUp("right")
        elif act_type == CODE_GO:  # 真っ直ぐ進む
            keyDown("up")
            keyUp("left")
            keyUp("right")
        elif act_type == CODE_RIGHT:  # 右に回転する
            keyUp("up")
            keyUp("left")
            keyDown("right")
        elif act_type == CODE_LEFT:  # 左に回転する
            keyUp("up")
            keyDown("left")
            keyUp("right")
        elif act_type == CODE_FRONT_RIGHT:  # 右前に進む
            keyDown("up")
            keyUp("left")
            keyDown("right")
        elif act_type == CODE_FRONT_LEFT:  # 左前に進む
            keyDown("up")
            keyDown("left")
            keyUp("right")
        elif act_type == CODE_GO_RIGHT:  # 前に進んで右に曲がる
            keyDown("up")
            keyUp("left")
            keyUp("right")
            sleep(0.7)
            keyUp("up")
            keyUp("left")
            keyDown("right")
            sleep(2)
            while act_type != CODE_LEFT and act_type != CODE_FRONT_LEFT:
                sleep(0.1)
            print("前に進んで右 - 終了")
        elif act_type == CODE_GO_LEFT:  # 前に進んで左に曲がる
            keyDown("up")
            keyUp("left")
            keyUp("right")
            sleep(0.7)
            keyUp("up")
            keyDown("left")
            keyUp("right")
            sleep(2)
            while act_type != CODE_RIGHT and act_type != CODE_FRONT_RIGHT:
                sleep(0.1)
            print("前に進んで左 - 終了")
        else:
            keyUp("up")
            keyUp("left")
            keyUp("right")


def main(repeat=400):
    global act_type
    for _ in range(repeat):
        ret, raw_frame = camera.read()
        # cv2.imshow('camera', raw_frame)
        frame = cv2.resize(src=raw_frame, dsize=(FRAME_HEIGHT, FRAME_WIDTH))

        height, width = frame.shape[:2]
        color_recognized_result = np.zeros((85, 50))

        green_count = 1
        black_count = 1
        green_appear_x = 0
        green_appear_y = 0
        black_appear_x = 0
        black_appear_y = 0
        for h in range(height):
            for w in range(width):
                red = frame[h][w][0]
                green = frame[h][w][1]
                blue = frame[h][w][2]
                if (red < 150 and blue < 150):  # 緑もしくは黒
                    if (green < 120):
                        # black
                        black_appear_x += w
                        black_appear_y += h
                        black_count += 1
                        color_recognized_result[height - h - 1][w] = 1
                    elif (red * 1.1 < green and blue * 1.1 < green):
                        # green
                        green_appear_x += w
                        green_appear_y += h
                        green_count += 1
                        color_recognized_result[height - h - 1][w] = 2
        green_median_x = int(green_appear_x / green_count * 100) / 100
        green_median_y = int(green_appear_y / green_count * 100) / 100
        black_median_x = int(black_appear_x / black_count * 100) / 100
        black_median_y = int(black_appear_y / black_count * 100) / 100

        if 20 < green_count:
            # green exist
            None
        else:
            if 50 < sum(itertools.chain.from_iterable([[num for num in row] for row in color_recognized_result[20:]])):
                if 0 <= black_median_y and black_median_y < 20:
                    act_type = CODE_RIGHT
                elif 20 <= black_median_y and black_median_y < 30:
                    act_type = CODE_FRONT_RIGHT
                elif 50 <= black_median_y and black_median_y < 60:
                    act_type = CODE_FRONT_LEFT
                elif 60 <= black_median_y and black_median_y <= 85:
                    act_type = CODE_LEFT
                else:
                    act_type = CODE_GO
            else:
                if sum(itertools.chain.from_iterable([[num for num in row] for row in color_recognized_result])) < 50:
                    act_type = CODE_GO
                else:
                    if 0 <= black_median_y and black_median_y < 20:
                        act_type = CODE_GO_RIGHT
                    elif 20 <= black_median_y and black_median_y < 30:
                        act_type = CODE_RIGHT
                    elif 50 <= black_median_y and black_median_y < 60:
                        act_type = CODE_LEFT
                    elif 60 <= black_median_y and black_median_y <= 85:
                        act_type = CODE_GO_LEFT
                    else:
                        act_type = CODE_GO
        for row in for_debug:
            if act_type == row[1]:
                print((row[0], black_median_y))


if __name__ == "__main__":
    # 矢印操作の関数を起動
    # th = Thread(target=controll)
    # th.start()

    #
    pr = LineProfiler()
    pr.add_function(main)
    pr.runcall(main)
    pr.print_stats()

    # 終了
    Can_thread_continue = False
    camera.release()
    cv2.destroyAllWindows()
    keyUp("up")
    keyUp("down")
    keyUp("left")
