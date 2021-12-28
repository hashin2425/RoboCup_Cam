import cv2
import matplotlib.pyplot as plt
from importlib import reload

import WebCam_judge

from pyautogui import keyDown, keyUp
import itertools

from threading import Thread
from time import sleep

camera = cv2.VideoCapture(0)

left_sum = 0
right_sum = 0


def controll():
    global left_sum, right_sum
    while True:
        if left_sum > right_sum:
            keyDown('left')
            keyUp('right')
            keyUp('up')
        elif left_sum < right_sum:
            keyUp('left')
            keyDown('right')
            keyUp('up')
        else:
            keyUp('left')
            keyUp('right')
            keyDown('up')


th = Thread(target=controll)
th.start()

while True:
    ret, raw_frame = camera.read()
    # cv2.imshow('camera', raw_frame)
    frame = cv2.resize(src=raw_frame, dsize=(50, 85))

    reload(WebCam_judge)
    res = WebCam_judge.judgement(frame)

    # plt.clf()
    # plt.pcolormesh(res)
    # plt.pause(1)

    left_sum = int(sum(itertools.chain.from_iterable([row[:20] for row in res[30:]])))
    right_sum = int(sum(itertools.chain.from_iterable([row[-20:] for row in res[30:]])))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
th.stop()
