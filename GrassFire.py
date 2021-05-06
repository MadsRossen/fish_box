import cv2 as cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
from random import randint


def GrassFire(img):
    """ Only input binary images of 0 and 255 """
    mask = copy.copy(img)

    h, w = mask.shape[:2]
    h = h-1
    w = w-1
    grassfire = np.zeros_like(mask, dtype=np.uint8)
    save_array = []
    zero_array = []
    blob_array = []
    temp_cord = []


    for y in range(h):
        for x in range(w):
            if mask[y][x] == 0 and x <= h:
                zero_array.append(mask[y][x])
            elif mask[y][x] == 0 and x >= w:
                zero_array.append(mask[y][x])

    # Looping if x == 1, and some pixels has to be burned
            while mask[y][x] > 0 or len(save_array) > 0:
                mask[y][x] = 0
                temp_cord.append([y, x])
                if mask[y - 1][x] > 0:
                    if [y - 1, x] not in save_array:
                        save_array.append([y - 1, x])
                if mask[y][x - 1] > 0:
                    if [y, x - 1] not in save_array:
                        save_array.append([y, x - 1])
                if mask[y + 1][x] > 0:
                    if [y + 1, x] not in save_array:
                        save_array.append([y + 1, x])
                if mask[y][x + 1] > 0:
                    if [y, x + 1] not in save_array:
                        save_array.append([y, x + 1])
                if len(save_array)>0:
                    y,x = save_array.pop()

                else:
                    blob_array.append(temp_cord)
                    temp_cord = []
                    break
    maskColor = np.zeros((h,w, 3), np.uint8)
    for blob in range(len(blob_array)):
        B, G, R = randint(0, 255), randint(0, 255), randint(0, 255)
        for cord in blob_array[blob]:
            y,x = cord
            maskColor[y][x][0] = B
            maskColor[y][x][1] = G
            maskColor[y][x][2] = R

    return blob_array
