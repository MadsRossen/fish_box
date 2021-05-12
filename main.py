import math

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

########################### Find the mean of the color of#655458 the  ####655458###51473E#####################
EdgesPic = cv2.imread('fishpics/Edges/EdgesPic.png', 1)
meanEdgeRGB(EdgesPic, 1, middleBelly=130)

EdgesPic2 = cv2.imread('fishpics/Edges/EdgesCod-GOPR1659.png', 1)
meanEdgeRGB(EdgesPic2, 1, middleBelly=650)

########################### Merge two pictures to one - Replacing only highlights  ##########################
Bund = cv2.imread('fishpics/direct2pic/Iteration2/Bund.JPG', 1)
Top = cv2.imread('fishpics/direct2pic/Iteration2/Top.JPG', 1)
replaceHighlights(Bund,Top,127,127)



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

#### Checkerboard calibration using fisheye method in MATLAB ##########
CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('checkerboards_test2/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

img = cv2.imread('checkerboards_test2/control/Kontrol.JPG', 0)
img_dim = img.shape[:2][::-1]

DIM = img_dim
balance = 1

scaled_K = K * img_dim[0] / DIM[0]
scaled_K[2][2] = 1.0
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
    img_dim, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
    new_K, img_dim, cv2.CV_16SC2)
undist_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT)

#percent by which the image is resized
scale_percent = 30

#calculate the 50 percent of original dimensions
width = int(undist_image.shape[1] * scale_percent / 100)
height = int(undist_image.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
undist_image_scaled = cv2.resize(undist_image, dsize)

cv2.imshow('undist_image_scaled',undist_image_scaled)
cv2.waitKey(0)
'''

#### Checkerboard calibration using strech matrix found in MATLAB using scarramuzza (fisheye method)##########
distorted = cv2.imread('checkerboards_test2/control/Kontrol.JPG', 0)

rows, cols = distorted.shape

undistorted = np.zeros(distorted.shape, dtype=distorted.dtype)

stretchMatrix = np.array([[1.00682131035995      , -0.00586627807785578],
                         [0.00628286256688552   , 1]], dtype=np.float64)

InvStrechmatrix = np.linalg.inv(stretchMatrix)

distortionCenter = np.array([[1443.77390922027], [1122.35038919366]], dtype=np.float64)

for v_m in range(rows):
    for u_m in range(cols):
        output_pos = stretchMatrix.dot(np.array([[u_m],[v_m]])) + distortionCenter
        output_pos_x = round(output_pos[0,0])
        output_pos_y = round(output_pos[1,0])

        if output_pos_x < cols and output_pos_y < rows:
            undistorted[v_m,u_m] = distorted[output_pos_y%rows,output_pos_x%cols]
        else:
            undistorted[v_m,u_m] = 0

plt.imshow(distorted)
plt.show()
plt.imshow(undistorted)
plt.show()

for v_m in range(rows):
    for u_m in range(cols):
        output_pos = InvStrechmatrix.dot(np.array([[u_m],[v_m]])) - distortionCenter
        output_pos_x = round(output_pos[0,0])
        output_pos_y = round(output_pos[1,0])

        if output_pos_x < cols and output_pos_y < rows:
            undistorted[v_m,u_m] = distorted[(output_pos_y),(output_pos_x)]
        else:
            undistorted[v_m,u_m] = 0

plt.imshow(distorted)
plt.show()
plt.imshow(undistorted)
plt.show()

for i in range(rows):
    for j in range(cols):
        offset_x = round(c * i + d * j + c_x)
        offset_y = round(e * i + l * j + c_y)
        if offset_x < rows & offset_y < cols:
            undistorted[i,j] = distorted[(offset_x),(offset_y)]
        else:
            undistorted[i,j] = 0

cv2.undistort(distorted, distCoeffs=stretchMatrix, cameraMatrix=0)

#percent by which the image is resized
scale_percent = 30

#calculate the 50 percent of original dimensions
width = int(distorted.shape[1] * scale_percent / 100)
height = int(distorted.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

distorted_small = cv2.resize(undistorted, dsize)
undistorted_small = cv2.resize(distorted, dsize)

cv2.imshow('distorted',distorted_small)
cv2.waitKey(0)
cv2.imshow('undistorted',undistorted_small)
cv2.waitKey(0)

'https://docs.opencv.org/master/d9/d0c/group__calib3d.html'

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