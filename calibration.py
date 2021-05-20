
#### Checkerboard calibration using fisheye method in openCV ##########
'https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0'

import glob
import warnings

import cv2
import numpy as np

import extremeImageProcessing as eip

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time

def undistort(inputImg):
    '''
    Undistorts images using parameters found in MATLAB calibration using 'standard'

    :param inputImg:
    :return:
    '''
    start_time = time.time()
    h, w, ch = inputImg.shape

    # Parameters found in MATLAB calibration session
    k_1 = -0.2666
    k_2 = 0.0845
    imgCenterX = 1441.64880195002
    imgCenterY = 1126.65141577051
    Fx = 1757.26695467940
    Fy = 1743.87124082603

    undistorted = np.zeros(inputImg.shape, np.uint8)

    for y in np.arange(-1, 1, 1 / h):
        for x in np.arange(-1, 1, 1 / w):
            xorig = x
            yorig = y

            r = np.sqrt(xorig ** 2 + yorig ** 2)
            output_pos_x = round(Fx * (xorig * (1 + k_1 * r ** 2 + k_2 * r ** 4)) + imgCenterX);
            output_pos_y = round(Fy * (yorig * (1 + k_1 * r ** 2 + k_2 * r ** 4)) + imgCenterY);

            input_pos_x = round(Fx * x + imgCenterX)
            input_pos_y = round(Fy * y + imgCenterY)

            if input_pos_x < w - 1 and input_pos_y < h - 1 and output_pos_x < w - 1 and output_pos_y < h - 1:
                if input_pos_x >= 0 and input_pos_y >= 0 and output_pos_x >= 0 and output_pos_y >= 0:
                    undistorted.itemset((input_pos_y, input_pos_x, 0), inputImg.item((output_pos_y, output_pos_x, 0)))
                    undistorted.itemset((input_pos_y, input_pos_x, 1), inputImg.item((output_pos_y, output_pos_x, 1)))
                    undistorted.itemset((input_pos_y, input_pos_x, 2), inputImg.item((output_pos_y, output_pos_x, 2)))
    print("Execution time for optimized item/itemset function: ", "--- %s seconds ---" % (time.time() - start_time))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(inputImg)
    ax2.imshow(undistorted)
    plt.show()

def undistortImg(distortedImg, recalibrate=False):
    '''
    Undistorts images using openCV's cv2.fisheye.calibrate function.

    :param distortedImg: The distorted image that is to be undistorted.
    :param recalibrate: set to True if recalibration is needed.
    :return: The undistorted image.
    '''

    if recalibrate == True:
        print('Calibrating camera please wait ... \n')
        CHECKERBOARD = (6,9) # size of checkerboard

        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('calibration/checkerboard_pics/*.JPG') #loaded images from folder in work tree
        #Run through list of images of checkerboards
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2] #All images must share the same size
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

        # Use the fisheye model to calibrate
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

        # Save calibration session parametres
        N_OK_array = np.array(N_OK)
        _img_shape_array = np.array(_img_shape)
        np.save('calibration/parameters_calibration_session/matrixK.npy', K)
        np.save('calibration/parameters_calibration_session/matrixD.npy', D)
        np.save('calibration/N_OK.npy', _img_shape_array)
        np.save('calibration/_img_shape.npy', _img_shape)
        print("Found " + str(N_OK_array) + " valid images for calibration")
        print("DIM = Dimension of images = " + str(_img_shape_array[::-1]))

    K = np.load('calibration/parameters_calibration_session/matrixK.npy')
    D = np.load('calibration/parameters_calibration_session/matrixD.npy')
    N_OK_array = np.load('calibration/N_OK.npy')
    _img_shape_array = np.load('calibration/_img_shape.npy')

    print("\nIntrinsic parameters")
    print("Camera matrix: K =")
    print(K)
    print("D =")
    print(D)

    img_dim = distortedImg.shape[:2][::-1]

    DIM = img_dim
    balance = 1

    scaled_K = K * img_dim[0] / DIM[0]
    scaled_K[2][2] = 1.0

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
        img_dim, np.eye(3), balance=balance)

    print('\n Undistorting image ... ')
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
        new_K, img_dim, cv2.CV_16SC2)
    undist_image = cv2.remap(distortedImg, map1, map2, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)

    print('\n Image has been undistorted')

    return undist_image

def checkerboard_calibrateOPENCV(dimensions, images_distort, images_checkerboard, show_img=False, recalibrate=False):
    """
    Undistorts images by a checkerboard calibration.

    SRC: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    :param show_img: Debug to see if all the images are loaded and all the edges are found
    :param dimensions: The dimensions of the checkerboard from a YAML file
    :param images_distort: The images the needs to be undistorted
    :param images_checkerboard: The images of the checkerboard to calibrate by
    :return: If it succeeds, returns the undistorted images, if it fails, returns the distorted images with a warning
    """
    print('Undistorting images ... \n')

    if recalibrate:
        print('Calibrating camera please wait ... \n')

        chessboardSize = (6, 9)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((dimensions[0][1] * dimensions[1][1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for img in images_checkerboard:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret is True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
                if show_img:
                    print(imgpoints)
                    cv2.imshow('img', img)
                    cv2.imshow('gray', gray)
                    cv2.waitKey(0)

        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:2], None, None)

        print("Done calibrating")

        # Save calibration session parameters
        print('Saving calibration session parameters in calibration/parameters_calibration_session ... \n')
        np.save('calibration/parameters_calibration_session/mtx.npy', mtx)
        np.save('calibration/parameters_calibration_session/dist.npy', dist)

    # Loading in parameters from previous calibration session
    mtx = np.load('calibration/parameters_calibration_session/mtx.npy')
    dist = np.load('calibration/parameters_calibration_session/dist.npy')

    print("\n Intrinsic parameters")
    print("Camera matrix: K =")
    print(mtx)
    print("\nDistortion coefficients =")
    print(dist)

    # Go through all the images and undistort them
    img_undst = []
    for n in images_distort:
        # Get image shape
        h, w = n.shape[:2]

        # Refine the camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort using remapping
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        undst = cv2.remap(n, mapx, mapy, cv2.INTER_LINEAR)

        img_undst.append(undst)
        if show_img:
            cv2.imshow('calibresult.png', undst)
            cv2.waitKey(0)

    print("Done undistorting images")

    return img_undst