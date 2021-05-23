import cv2 as cv
import numpy as np
from fractions import Fraction as frac 
import copy

def blur(img):   
    '''
    blurs input image

    '''

    img = copy.copy(img)

    kernel = np.ones(shape=(3,3))

    imgy, imgx = np.shape(img)

    blur_img = np.zeros(shape=(imgy, imgx))

    row, col = np.shape(kernel)

    blur = frac(1, col**2 * row**2) * kernel # Blurring kernel 

    for y in range(imgy - (row + 1)):
        for x in range(imgx - (col + 1)):

            blur_img[y, x] = np.sum(np.multiply(blur, img[y:y+row, x:x+col]))

    print("Done blurring")
    return blur_img
