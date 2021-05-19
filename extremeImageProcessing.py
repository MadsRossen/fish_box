import numpy as np
import cv2
import warnings
import glob

from random import randint


def crop(images, y, x, height, width):

    cropped_images = []
    for n in images:
        ROI = n[y:y + height, x:x + width]
        cropped_images.append(ROI)

    return cropped_images


def bitwise_and(img, mask):
    '''
    A bitwise operation to stitch a picture to a mask

    :param img: The image to reference in color
    :param mask: The mask to reference in grayscale
    :return: An image where the mask decides
    '''

    # Get the height and width of the image to make an array filled with zeroes to have a black image
    height, width = img.shape[:2]
    main_clone = np.zeros((height, width, 3), dtype=np.uint8)

    # Go through each pixel and change the clones pixel values to the ones of the original image, as long as the same
    # pixel on the mask is not black
    for y in range(height):
        for x in range(width):
            mask_val = mask.item(y, x)
            if mask_val != 0:
                main_clone.itemset((y, x, 0), img.item(y, x, 0))
                main_clone.itemset((y, x, 1), img.item(y, x, 1))
                main_clone.itemset((y, x, 2), img.item(y, x, 2))
            else:
                main_clone.itemset((y, x, 0), 0)
                main_clone.itemset((y, x, 1), 0)
                main_clone.itemset((y, x, 2), 0)

    return main_clone


def erosion(mask, kernel_ero):
    """
    A standard erosion solver, shrinks the given mask.

    :param mask: The mask to shrink
    :param kernel_ero: The kenerl to shrink the erosion by
    :return: Returns the erosied mask
    """
    print("Started erosion...")

    # Acquire size of the image
    height, width = mask.shape[0], mask.shape[1]
    # Define the structuring element
    k = kernel_ero.shape[0]
    SE = np.ones((k, k), dtype=np.uint8)
    # kernel_ero = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2

    # Define new image
    imgErode = np.zeros((height, width), dtype=np.uint8)

    # Erosion
    if k % 2 >= 1:
        for y in range(constant, height - constant):
            for x in range(constant, width - constant):
                temp = mask[y - constant:y + constant + 1, x - constant:x + constant + 1]
                product = temp * SE
                imgErode[y, x] = np.min(product)
    else:
        warnings.warn("Kernel shape is even, it should be uneven!")

    print("Done with erosion!")

    return imgErode


def dilation(mask, kernel_di):
    '''
    A standard dilation solver, expands the given mask.

    :param mask: The mask to dilate
    :param kernel_di: The kernel to dilate by
    :return: The dilated mask
    '''

    print("Started dilating...")

    # Acquire size of the image
    height, width = mask.shape[0], mask.shape[1]
    # Define new image to store the pixels of dilated image
    imgDilate = np.zeros((height, width), dtype=np.uint8)
    # Define the kernel shape
    ks = kernel_di.shape[0]
    # Use that to define the constant for the middle part
    constant1 = (ks - 1) // 2
    # Dilation
    if ks % 2 >= 1:
        for y in range(constant1, height - constant1):
            for x in range(constant1, width - constant1):
                temp = mask[y - constant1:y + constant1 + 1, x - constant1:x + constant1 + 1]
                product = temp * kernel_di
                imgDilate[y, x] = np.max(product)
    else:
        warnings.warn("Kernel shape is even, it should be uneven!")

    print("Done with dilation!")

    return imgDilate


def morph_close(mask, kernel):
    """
    Close morphology on a mask and a given kernel.

    :param kernel:
    :param mask: The mask to use the morphology on
    :return: Close morphology on a mask
    """

    dilate = dilation(mask, kernel)
    ero = erosion(dilate, kernel)

    return ero


def morph_open(mask, kernel):
    """
    Open morphology on a mask and a given kernel.

    :param mask: The mask to use the morphology on
    :return: Open morphology on a mask
    """
    ero = erosion(mask, kernel)
    dilate = dilation(ero, kernel)

    return dilate


def find_contours():
    print("Find contours")


def findInRange(images):
    '''
    Finds pixels in a pre-determined lower and upper bound.

    :param images: And array with images to create masks off of
    :return: A list with masks created from the images
    '''
    print("Creating masks...")

    img_iso = []
    for img in images:
        # Make this function ourself
        img_hsv = convert_RGB_to_HSV(img)

        height, width = img.shape[0], img.shape[1]
        img_copy = np.zeros((height, width), dtype=np.uint8)

        # Define upper and lower
        lower = np.array([0, 16, 16])
        upper = np.array([94, 255, 255])

        # Find the pixels within the value of the lower and upper bounds using numpy
        img_hsv_iso = np.where(((img_hsv[:, :, 0] >= lower[0]) & (img_hsv[:, :, 0] <= upper[0])) &
                               ((img_hsv[:, :, 1] >= lower[1]) & (img_hsv[:, :, 1] <= upper[1])) &
                               ((img_hsv[:, :, 2] >= lower[2]) & (img_hsv[:, :, 2] <= upper[2])))

        # Replace the pixels where we have those values found with white pixels
        img_copy[img_hsv_iso] = 255

        # Turn every other pixel which is not white to black
        img_copy[img_copy != 255] = 0

        img_iso.append(img_copy)

    print("Done creating masks!")

    return img_iso


def grayScaling(img):
    """
    Function that will convert a BGR image to a mean valued greyscale image.
    :param img: BGR image that will be converted to greyscale
    :return: The converted greyscale image.
    """

    # Get the height and width of the image to create a cop of the other image in an array of zeros
    h, w, = img.shape[:2]
    greyscale_img1 = np.zeros((h, w, 1), np.uint8)

    # Go through each pixel in the image and record the intensity, then safe it for the same pixel in the image copy
    for y in range(h):
        for x in range(w):
            I1 = (img.item(y, x, 0) + img.item(y, x, 1) + img.item(y, x, 2))/3
            greyscale_img1.itemset((y, x, 0), I1)
    return greyscale_img1


def convert_RGB_to_HSV(img):
    """
    Converts an RGB image to HSV.

    :param img: The image to convert
    :return: HSV image
    """

    width, height, channel = img.shape

    B, G, R = img[:, :, 0]/255, img[:, :, 1]/255, img[:, :, 2]/255

    hsv_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(width):
        for j in range(height):

            # Defining Hue
            h, s, v = 0.0, 0.0, 0.0
            r, g, b = R[i][j], G[i][j], B[i][j]

            max_rgb, min_rgb = max(r, g, b), min(r, g, b)
            dif_rgb = (max_rgb-min_rgb)

            if r == g == b:
                h = 0
            elif max_rgb == r:
                h = ((60*(g-b))/dif_rgb)
            elif max_rgb == g:
                h = (((60*(b-r))/dif_rgb)+120)
            elif max_rgb == b:
                h = (((60*(r-g))/dif_rgb)+240)
            if h < 0:
                h = h+360

            # Defining Saturation
            if max_rgb == 0:
                s = 0
            else:
                s = ((max_rgb-min_rgb)/max_rgb)

            # Defining Value
            hsv_img[i][j][0], hsv_img[i][j][1], hsv_img[i][j][2] = h/2, s * 255, s * 255

    return hsv_img


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


def grassfire_v2(mask):
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
                    print("Burn is done")
                    blob_array.append(temp_cord)
                    temp_cord = []
                    break
    maskColor = np.zeros((h,w, 3), np.uint8)
    for blob in range(len(blob_array)):
        B, G, R = randint(0, 255), randint(0, 255), randint(0, 255)
        for cord in blob_array[blob]:
            y, x = cord
            maskColor[y][x][0] = B
            maskColor[y][x][1] = G
            maskColor[y][x][2] = R
    cv2.imshow("grasfire", maskColor)
    cv2.waitKey(0)

    return 0