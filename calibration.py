
#### Checkerboard calibration using fisheye method in openCV ##########
'https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0'

import glob
import cv2
import numpy as np

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
        np.save('calibration/intrinsic_parameters/matrixK.npy', K)
        np.save('calibration/intrinsic_parameters/matrixD.npy', D)
        np.save('calibration/N_OK.npy', _img_shape_array)
        np.save('calibration/_img_shape.npy', _img_shape)
        print("Found " + str(N_OK_array) + " valid images for calibration")
        print("DIM = Dimension of images = " + str(_img_shape_array[::-1]))

    K = np.load('calibration/intrinsic_parameters/matrixK.npy')
    D = np.load('calibration/intrinsic_parameters/matrixD.npy')
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