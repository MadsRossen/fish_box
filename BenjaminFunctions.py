import cv2
import numpy as np
import os
import specularity_removal as sr
import math
import glob

from Kasperfunctions import resizeImg, crop

# A library that has a equalize matcher!
from skimage.exposure import match_histograms


def loadImages(path, edit_images, show_img=False, scaling_percentage=30):
    '''
    Loads all the images inside a file.

    :return: All the images in a list and its file names.
    '''

    images = []
    class_names = []
    img_list = os.listdir(path)
    print("Total images found:", len(img_list))

    for cl in img_list:
        # Find all the images in the file and save them in a list without the ".jpg"
        cur_img = cv2.imread(f"{path}/{cl}", 1)
        img_name = os.path.splitext(cl)[0]

        # Do some quick images processing to get better pictures if the user wants to
        if edit_images:
            #cur_img_crop = crop(cur_img, 650, 500, 1000, 3000)
            cur_img_re = resizeImg(cur_img, scaling_percentage)
            cur_img = cur_img_re

        # Show the image before we append it, to make sure it is read correctly
        if show_img:
            cv2.imshow(f"Loaded image: {img_name}", cur_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Append them into the list
        images.append(cur_img)
        class_names.append(img_name)

    # Remove the image window after we have checked all the pictures
    cv2.destroyAllWindows()

    return images, class_names


def replaceHighlights(main_img, spec_img, limit):
    '''
    This functions replaces the highlights from a main picture with the pixels from a specular image pixels

    :param main_img: The image of which will get the pixels replaced with the specular image
    :param spec_img: The image of which will be used to replace the pixels of the main image
    :param limit: The limits of a pixel value before it is classified as a specular highlight
    :return: The image that has the highlights replaced
    '''

    # Create copy
    img_main_cop = np.copy(main_img)
    img_spec_cop = np.zeros((spec_img.shape[0],spec_img.shape[1],3), np.uint8)

    # Isolate the areas where the color is white
    main_img_spec = np.where((img_main_cop[:, :, 0] >= limit) & (img_main_cop[:, :, 1] >= limit) &
                             (img_main_cop[:, :, 2] >= limit))

    img_spec_cop[main_img_spec] = spec_img[main_img_spec]
    img_main_cop[main_img_spec] = (0, 0, 0)

    img_main_cop[main_img_spec] = img_spec_cop[main_img_spec]

    # Different methods, find out what works best later
    match = match_histograms(img_spec_cop, img_main_cop, multichannel=True)
    match2 = match_histograms(spec_img, img_main_cop, multichannel=True)

    # Replace pixels, replace the matches to use different methods
    img_main_cop[main_img_spec] = match2[main_img_spec]

    cv2.imshow("match", match2)
    cv2.imshow("spec_img", spec_img)
    cv2.imshow("final", img_main_cop)
    cv2.imshow("Spec", img_spec_cop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_main_cop


def equalizeColoredImage(img):
    '''
    This function returns the equalized image of a color image using YUV

    :param img: The image to equalize
    :return: Equalized image
    '''
    # Turn into YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def morphological_trans(mask):
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    return dilation


def find_blood_damage(img):
    '''
    Returns the mask of the found blood damage

    :param img: Image to check for blood spots
    :return: The mask of blood damage
    '''

    # Bounds for the red in the wound, use gimp to find the right values
    lower_red = np.array([0, 90, 90])
    upper_red = np.array([7, 100, 100])

    # Convert to hsv, check for red and return the found mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    return mask


### KAJ FUNCTIONS ###

def createDict():
    '''
    Custom dictionary for storing length and angle values for each contour.
    :return: Dictionary with length and angle array.
    '''
    dict = {
        "length": [],
        "angle": []
    }
    return dict


def cropToROI(orig_img, contours):
    '''
    Finds the minimum (x,y) and maximum (x,y) coordinates for each contour and then crops the original image to fit the contours.
    It also computes the center of mass of each contour.

    :param orig_img: original image that will be cropped.
    :param contours: (x,y) coordinates for the contours in the orig_img.
    :return: (xcm, ycm) array with the (x,y) coordinates for the contours center of mass. crop_img: array of the cropped images (one image for each contour)
    '''

    height, width = orig_img.shape[:2]
    xcm = []
    ycm = []
    crop_img = []
    for nr in range(len(contours)):
        ymax, ymin = 0, height
        xmax, xmin = 0, width
        for point in range(len(contours[nr])):
            if contours[nr][point][0][0] > xmax:
                xmax = contours[nr][point][0][0]
            if contours[nr][point][0][0] < xmin:
                xmin = contours[nr][point][0][0]
            if contours[nr][point][0][1] > ymax:
                ymax = contours[nr][point][0][1]
            if contours[nr][point][0][1] < ymin:
                ymin = contours[nr][point][0][1]
        # Computing the approximate center of mass:
        # From Thomas B. Moeslund "Introduction to Video and Image Processing"
        # (Page 109 Eq: 7.3 and 7.4)
        xcm.append(int((xmin+xmax)/2))
        ycm.append(int((ymin+ymax)/2))
        crop_img.append(orig_img[ymin-50:ymax+50, xmin-50:xmax+50])
        #rsize = 6
        #crop_img[nr] = cv2.resize(crop_img[nr], (rsize, rsize))
    return xcm, ycm, crop_img


def find_contours(images):
    # Bounds for the fish
    lower = np.array([0, 0, 16])
    upper = np.array([61, 176, 255])
    kernel = np.ones((6, 6), np.uint8)

    def nothing(x):
        pass

    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("kernel", "Adjust_Hue_Satuation_Value", 0, 10, nothing)

    contours = []
    for n in images:
        while True:
            kernel_val = cv2.getTrackbarPos("kernel", "Adjust_Hue_Satuation_Value")
            kernel = np.ones((kernel_val, kernel_val), np.uint8)
            # Convert to hsv, check for red and return the found mask
            hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphology
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            cv2.imshow("Mask", closing)

            key = cv2.waitKey(1)
            if key == 27:
                break

        contours.append(mask)


def rotateImages(img, rotate_img, xcm, ycm, contours):
    '''
    A function that can rotate a set of images, so the longest part has an angle of zero relative to the
    positive x-axis. This is done by line tracing each point on the contours relative to
    the center of mass of the contours. The angle for each line relative to the positive
    x-axis is then computed, and the angle for the longest length is then used to rotate the image.

    NOTE: The approximation method for the findContour() function should be CHAIN_APPROX_NONE, otherwise if
    CHAIN_APPROX_SIMPLE is used the results might vary negatively.

    Optional: It is possible to draw the line tracing in the img (line 164 should then be included).
    It is also possible to plot the distribution of the contour coordinates length
    relative to the angle (line 174-177 should then be included).

    :param img: Image that you want the line tracing to be drawn upon.
    :param rotate_img: Images that needs to be rotated.
    :param xcm: x-coordinates for the center of mass of the contours
    :param ycm: y-coordinates for the center of mass of the contours
    :param contours: Contours from the images that needs to be rotated.
    :return: The rotated images.
    '''

    # Variable where the length and angle will be stored.
    data = []
    # Variable to store the rotated images.
    img_output = []
    for nr in range(len(contours)):
        maxLength = 0
        angleMaxLength = 0
        data.append(createDict())
        for point in range(len(contours[nr])):
            # Compute x and y coordinate relative to the contours center of mass.
            x_delta = contours[nr][point][0][0] - xcm[nr]
            y_delta = contours[nr][point][0][1] - ycm[nr]
            # Compute the length and angle of each coordinate in the contours.
            data[nr]["length"].append(math.sqrt(pow(x_delta, 2) + pow(y_delta, 2)))
            data[nr]["angle"].append(math.atan2(y_delta, x_delta)*(180/math.pi))
            # Finding the longest length and at what angle.
            if data[nr]["length"][point] > maxLength:
                maxLength = data[nr]["length"][point]
                angleMaxLength = data[nr]["angle"][point]
            # Draw the line tracing on the contours (optional)
            cv2.line(img, (xcm[nr], ycm[nr]), (contours[nr][point][0][0], contours[nr][point][0][1]), (255, 0, 0), 1)
        # Rotating the images so the longest part of the resistor has an angle of 0 relative to the positive x-axis.
        if angleMaxLength != 0:
            (height, width) = rotate_img[nr].shape[:2]
            (cX, cY) = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D((cX, cY), angleMaxLength, 1.0)
            img_output.append(cv2.warpAffine(rotate_img[nr], M, (width, height), borderValue=(0, 128, 128)))
        else:
            img_output.append(rotate_img[nr])
        resize = 600
        img_output[nr] = cv2.resize(img_output[nr], (resize, resize))
        # Plot the contour coordinates length relative to the angle (optional):
        """plt.subplot(int("1" + str(len(contours)) + str(nr + 1)))
        plt.bar(data[nr]["angle"], data[nr]["length"])
        plt.axis([-180, 180, 0, 100])
    plt.show()"""
    return img_output


def checkerboard_calibrate(images_distort, images_checkerboard):
    print("Started calibrating...")
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for img in images_checkerboard:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            print("No ret")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
    print(imgpoints)
    print(objpoints)
    img_undst = []
    for n in images_distort:
        h, w = n.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv2.undistort(n, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        img_undst.append(dst)
        cv2.imshow('calibresult.png', dst)
        cv2.waitKey(0)
        print("Done calibrating")
    return img_undst


### MATHIAS FUNCTIONS ###

def isolate_img(resized_input_image):

    hsv_image = cv2.cvtColor(resized_input_image, cv2.COLOR_BGR2HSV)

    def nothing(x):
        pass

    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("lowerH", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("lowerS", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("lowerV", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    cv2.createTrackbar("upperH", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("upperS", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("upperV", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    # while loop to adjust the HSV detection in the image.

    while True:

        lowerH = cv2.getTrackbarPos("lowerH", "Adjust_Hue_Satuation_Value")
        lowerS = cv2.getTrackbarPos("lowerS", "Adjust_Hue_Satuation_Value")
        lowerV = cv2.getTrackbarPos("lowerV", "Adjust_Hue_Satuation_Value")

        upperH = cv2.getTrackbarPos("upperH", "Adjust_Hue_Satuation_Value")
        upperS = cv2.getTrackbarPos("upperS", "Adjust_Hue_Satuation_Value")
        upperV = cv2.getTrackbarPos("upperV", "Adjust_Hue_Satuation_Value")

        lowerRange_blue = np.array([lowerH, lowerS, lowerV])
        upperRange_blue = np.array([upperH, upperS, upperV])

        mask = cv2.inRange(hsv_image, lowerRange_blue, upperRange_blue)

        res = cv2.bitwise_and(resized_input_image, resized_input_image, mask=mask)

        cv2.imshow("res", res)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1)
        if key == 27:
            break
    return res