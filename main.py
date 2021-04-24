import cv2
import numpy as np
from matplotlib import pyplot as plt

from Kasperfunctions import crop, resizeImg, highestPixelValue, showCompariHist, histColor, showCompari3Hist, \
    meanEdgeRGB, white_balance
from BenjaminFunctions import replaceHighlights

'''
# load image
left = cv2.imread('fishpics/direct2pic/GOPR1591.JPG', 1)
left_crop = crop(left, 650, 500, 1000, 3000)
left_re = resizeImg(left_crop, 30)
left_eq = equalizeColoredImage(left_re)

right = cv2.imread('fishpics/direct2pic/GOPR1590.JPG', 1)
right_crop = crop(right, 650, 500, 1000, 3000)
right_re = resizeImg(right_crop, 30)
right_eq = equalizeColoredImage(right_re)

replaceHighlights(left_eq, right_eq, 215, 100)

# display image
cv2.imshow("left", left_eq)
cv2.imshow("right", right_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

########################### Histogram ##########################
# load image
control = cv2.imread('fishpics/histogram_design1_and_control/GOPR1542.JPG', 1)

paperdome = cv2.imread('fishpics/histogram_design1_and_control/papirdome.jpg', 1)
paperdome2 = cv2.imread('fishpics/histogram_design1_and_control/papirdome2.jpeg', 1)

diffuse = cv2.imread('fishpics/histogram_design1_and_control/diffuse.jpg', 1)

print('diffuse')
highestPixelValue(diffuse, False)
highestPixelValue(diffuse, True)

print('paperdome')
highestPixelValue(paperdome, False)
highestPixelValue(paperdome, True)

print('paperdome2')
highestPixelValue(paperdome2, False)
highestPixelValue(paperdome2, True)

print('control')
highestPixelValue(control, False)
highestPixelValue(control, True)

########################### Color histogram Histogram halogen VS  ##########################
currentMethod = cv2.imread('fishpics/Lightsetup_ite_2/CurrentMethod.JPG', 1)
domelighting_ite2 = cv2.imread('fishpics/Lightsetup_ite_2/domelight.JPG', 1)
domelighting_ite1 = cv2.imread('fishpics/Lightsetup_ite_2/domelight_ite1_ROI.JPG', 1)

cv2.imshow('domelighting_ite2',domelighting_ite2)

showCompariHist(currentMethod, domelighting_ite2,'Current Method', 'domelighting_ite2')

highestPixelValue(currentMethod, True)
highestPixelValue(currentMethod, False)
highestPixelValue(domelighting_ite2, True)
highestPixelValue(domelighting_ite2, False)
'''
grayScaleCard = cv2.imread('fishpics/grayScaleCard/gray_scale_cardCropped.JPG',1)
cv2.imshow('grayScaleCard', grayScaleCard)
whiteBalancedImg = white_balance(grayScaleCard)
cv2.imshow('whiteBalancedImg', whiteBalancedImg)
cv2.waitKey(0)

########################### Find the mean of the color of the  ##########################
EdgesPic = cv2.imread('fishpics/Edges/EdgesPic.png', 1)
meanEdgeRGB(EdgesPic)

EdgesPic2 = cv2.imread('fishpics/Edges/EdgesCodPic.png', 1)
meanEdgeRGB(EdgesPic2)

########################### Merge two pictures to one - Replacing only highlights  ##########################
Bund = cv2.imread('fishpics/direct2pic/Iteration2/Bund.JPG', 1)
Top = cv2.imread('fishpics/direct2pic/Iteration2/Top.JPG', 1)
replaceHighlights(Bund,Top,127,127)





