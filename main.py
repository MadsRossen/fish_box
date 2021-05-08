import cv2
import numpy as np
import glob
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

grayScaleCard = cv2.imread('fishpics/grayScaleCard/gray_scale_cardCropped.JPG',1)
cv2.imshow('grayScaleCard', grayScaleCard)
whiteBalancedImg = white_balance(grayScaleCard)
cv2.imshow('whiteBalancedImg', whiteBalancedImg)
cv2.waitKey(0)

'''

########################### Find the mean of the color of#655458 the  ####655458###51473E#####################
EdgesPic = cv2.imread('fishpics/Edges/EdgesPic.png', 1)
meanEdgeRGB(EdgesPic, 1, middleBelly=130)

EdgesPic2 = cv2.imread('fishpics/Edges/EdgesCod-GOPR1659.png', 1)
meanEdgeRGB(EdgesPic2, 1, middleBelly=650)

########################### Merge two pictures to one - Replacing only highlights  ##########################
Bund = cv2.imread('fishpics/direct2pic/Iteration2/Bund.JPG', 1)
Top = cv2.imread('fishpics/direct2pic/Iteration2/Top.JPG', 1)
replaceHighlights(Bund,Top,127,127)


'''
########################### Checkerboard calibration using openCV  ####655458###51473E#####################

print("Camera calibration initiated\n")

chessboardSize = (9,6)
frameSize = (3000, 2250)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('checkerboards_test2/*.jpg')
print('Number of images with checkerboards :')
print(len(images), '\n')

for fname in images:
    print('Analyzing: ', fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        imgScale = img
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(" ")
print("Camera matrix :  \n", mtx,   '\n')
print("dist :           \n", dist,  '\n')
print("rvecs :          \n", rvecs, '\n')
print("tvecs :          \n", tvecs, '\n')

img = cv2.imread('checkerboards_test2/control/Kontrol.JPG')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort method 1 Using cv.undistort()
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
cv2.imwrite('calibresult_method_1.png', dst)

# undistort method 2 Using remapping
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_method_2.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)) )

########################### Checkerboard calibration using openCV method with own build  ####655458###51473E#####################
'https://docs.opencv.org/master/d9/d0c/group__calib3d.html'

#Find chessboard corners - objPoints and imgPoints


########################### Zangs method #####################
'https://docs.opencv.org/master/d9/d0c/group__calib3d.html'


A = ([1,    1,  1],
     [1,    1,  1],
     [1,    1,  1])

B = ([1,    1,  1],
     [1,    1,  1],
     [1,    1,  1])

res = np.dot(A,B)
print(res)

res = np.add(A,B)
print(res)

H = ([x],
     [y],
     [1])

K = ([c,    cs,         x_H],
     [0,    c*(1+m),    y_H],
     [0,    0,          1])

C = [[r_11,    r_12,   t_1],
     [r_21,    r_22,   t_2],
     [r_31,    r_32,   t_3]]

D = [[X],
     [Y],
     [1]]

H = K*C



x = K*R*[I_3| X_O]*X
x = PX
'''