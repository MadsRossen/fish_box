import cv2
import numpy as np
import os
import math
import warnings
import extremeImageProcessing as eip
import copy

from matplotlib import pyplot as plt  # --- Is this still used in final?


def loadImages(path, edit_images, show_img=False, scaling_percentage=30):
    """
    Loads all the images inside a file.

    :return: All the images in an array and its file names.
    """

    images = []
    class_names = []
    img_list = os.listdir(path)
    print("Loading in images...")
    print("Total images found:", len(img_list))
    for cl in img_list:
        # Find all the images in the file and save them in an array without the ".jpg"
        cur_img = cv2.imread(f"{path}/{cl}", 1)
        img_name = os.path.splitext(cl)[0]

        # Do some quick images processing to get better pictures if the user wants to
        if edit_images:
            cur_img_re = resizeImg(cur_img, scaling_percentage)
            cur_img = cur_img_re

        # Show the image before we append it, to make sure it is read correctly
        if show_img:
            cv2.imshow(f"Loaded image: {img_name}", cur_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Append them into the array
        images.append(cur_img)
        class_names.append(img_name)

    # Remove the image window after we have checked all the pictures
    cv2.destroyAllWindows()

    print("Done loading the images!")

    return images, class_names


def save_img(img, save_image_path):
    """
    Saves image(s) in a file directory.

    :param save_image_path: The path to the directory to save images
    :param img: An image or an array of images to save in a file
    """

    count = 0
    # Check if we have one or several images we wanna save
    if len(img) < 1:
        for n in img:
            # Count one for each image so we can name them differently and with numbers
            count = count+1
            cv2.imwrite(save_image_path+f"\\fish{count}.jpg", n)
    else:
        cv2.imwrite(save_image_path+"\\fish.jpg", img)


def claheHSL(img, clipLimit, tileGridSize):
    """
    Turns an image into a claheHSL image.

    :param img: The main image
    :param clipLimit: Set in parameters
    :param tileGridSize: Set in parameters
    :return: The claheHSL image
    """

    # Convert BGR to HSL
    fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Get the first channel
    LChannelHLS = fiskHLS2[:, :, 1]

    # Create clahe using the cliplimit and tileGridSize set from parameters
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Apply the first channel
    claheLchannel1 = clahe.apply(LChannelHLS)
    fiskHLS2[:, :, 1] = claheLchannel1

    # Convert back to BGR from HLS
    fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)

    return fiskClahe


def resizeImg(img, scale_percent):
    """
    Resizes the image by a scaling percent.

    :param img: The image to resize
    :param scale_percent: The percent to scale by
    :return: The resized image
    """

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def detect_bloodspots(imgs, show_img=False):

    print("Started detecting blood spots...")

    mask_bloodspots = []
    segmented_blodspots_imgs = []
    marked_bloodspots_imgs = []
    booleans_bloodspot = []         # List of boolean values for each image classification
    count = 0

    for n in imgs:
        # hsv_img = cv2.cvtColor(copy.copy(n), cv2.COLOR_BGR2HSV)
        hsv_img = eip.convert_RGB_to_HSV(copy.copy(n))

        booleans_bloodspot.append(False)
        marked_bloodspots_imgs.append(copy.copy(n))

        # Threshold for blood spots
        H_range = (0, 10)
        S_range = (90, 255)
        V_range = (90, 255)

        # Find the blood in the specified range
        frame_threshold1_own = eip.findInRange(hsv_img, [H_range, S_range, V_range])

        # Combining the masks
        mask_bloodspots.append(frame_threshold1_own)

        # Create kernels for morphology
        kernelOpen = 3
        kernelClose = 5

        # Perform morphology
        masks, _ = morphology_operations(mask_bloodspots, imgs, kernelOpen, kernelClose, False, False)

        # Perform bitwise operation to show bloodspots instead of BLOBS
        segmented_blodspots_imgs.append(eip.bitwise_and(n, masks[count]))

        # Make representation of BLOB / bloodspots
        # Find contours
        contours, _ = cv2.findContours(masks[count], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # For debugging
        if show_img:
            cv2.imshow("IMGHSV", masks[count])
            cv2.waitKey(0)

        # Classify as blood spots if the spots are big enough
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 0:
                x, y, w, h = cv2.boundingRect(cont)
                # Create tag
                cv2.putText(marked_bloodspots_imgs[count], 'Wound', (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                # Draw green contour
                cv2.rectangle(marked_bloodspots_imgs[count],(x-5,y-5),(x+w+5,y+h+5),(0,255,0), 2);
                booleans_bloodspot.append(True)

        count = count + 1

    print("Finished detecting blood spots!")

    return mask_bloodspots, segmented_blodspots_imgs, marked_bloodspots_imgs, booleans_bloodspot


# Used for the trackbars
def nothing(x):
    pass


def open_close_trackbars():
    """
    This function allows for kernel value editing with trackbars in find_contours.

    :return: Kernel values for open and close
    """

    # Set kernel values for open and close
    kernel_val_open_val = cv2.getTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value")
    kernel_val_close_val = cv2.getTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value")

    # Make sure it's only uneven numbers for the kernels
    if kernel_val_open_val % 2 == 0:
        cv2.setTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value", kernel_val_open_val + 1)
        kernel_val_open_val = cv2.getTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value")

    if kernel_val_close_val % 2 == 0:
        cv2.setTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value", kernel_val_close_val + 1)
        kernel_val_close_val = cv2.getTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value")

    return kernel_val_open_val, kernel_val_close_val


def morphology_operations(masks, images, open_kern_val, close_kern_val, change_kernel=False, show_img=False):

    """
    Uses open and close morphology on a given array of masks. Also shows the res images when the masks is combined
    with the normal images.

    :param close_kern_val: An integer for the close kernel value
    :param open_kern_val: An integer for the open kernel value
    :param masks: The masks to use morphology on
    :param images: The images to add to the masks when creating the res
    :param change_kernel: Enables trackbars to change kernel values
    :param show_img: To either show the results or not
    :return: An array of masks with morphology applied and the res images
    """

    # To make editing the kernel possible
    if change_kernel:
        # Create trackbar window and trackbars for open and close
        cv2.namedWindow("Adjust_Hue_Satuation_Value")
        cv2.createTrackbar("kernel open", "Adjust_Hue_Satuation_Value", 2, 20, nothing)
        cv2.createTrackbar("kernel close", "Adjust_Hue_Satuation_Value", 2, 20, nothing)

    # Values for each image
    old_open_val, old_closes_val = 0, 0
    image_n = 0
    closing = None
    closing_array = []
    res_array = []
    # Looping through each mask
    for n in masks:
        while True:
            # If we wanna change the kernel by trackbar
            if change_kernel:
                kernel_val_open_val, kernel_val_close_val = open_close_trackbars()
            else:
                kernel_val_open_val, kernel_val_close_val = open_kern_val, close_kern_val

            # Make kernels for each morph type
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_open_val, kernel_val_open_val))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_close_val, kernel_val_close_val))

            # Only use opening and closing when the slider is moved instead of every frame
            if old_open_val != kernel_val_open_val or old_closes_val != kernel_val_close_val or change_kernel is False:
                opening = eip.morph_open(n, kernel_open)
                closing = eip.morph_close(opening, kernel_close)
                old_open_val = kernel_val_open_val
                old_closes_val = kernel_val_close_val

            # To see how much of the fish we are keeping
            if closing is not None:
                res = eip.bitwise_and(images[image_n], closing)
            else:
                warnings.warn("The closing or open operation is None!")

            # If the user wanna change the kernel values
            if change_kernel:
                cv2.imshow("Adjust_Hue_Satuation_Value", closing)

            # For debugging
            if show_img:
                cv2.imshow("Mask Closing", closing)
                cv2.imshow("Res", res)

            # Break the while loop if esc is pressed
            key = cv2.waitKey(1)
            if key == 27 or change_kernel is False:
                break

        closing_array.append(closing)
        res_array.append(res)

    return closing_array, res_array


############# THE REST FROM HERE IS EXPERIMENTAL AND ALSO CONTIANS OPENCV FUNCTIONS ##########
############# THEY ARE NOT PART OF THE MAIN PROGRAM.                                 ##########


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


def contour_MOC(orig_img, contours):
    '''
    Finds the minimum (x,y) and maximum (x,y) coordinates for each contour and computes the center of mass of each
    contour.

    :param orig_img: original image that will be cropped.
    :param contours: (x,y) coordinates for the contours in the orig_img.
    :return: (xcm, ycm) array with the (x,y) coordinates for the contours center of mass. crop_img: array of the cropped
    images (one image for each contour)
    '''

    print("Finding maximum and minimum coordinates for each contours and then cropping...")

    # To see if both are the correct amount
    print(f"Original images length: {len(orig_img)}")
    print(f"Contours len: {len(contours)}")

    # Save the height and width for each image in an array
    height = []
    width = []
    for n in orig_img:
        height.append(n.shape[0])
        width.append(n.shape[1])

    xcm = []
    ycm = []
    for nr in range(len(orig_img)):
        ymax, ymin = 0, height[nr]
        xmax, xmin = 0, width[nr]
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

    print("Found all the contours and cropped the image!")

    return xcm, ycm


def find_biggest_contour(cnt):
    """
    Returns the biggest contour in an array of contours.

    :param cnt: An array of contours
    :return: The biggest contour inside the array
    """
    print("Finding the biggest contours...")

    biggest_contour = 0
    biggest_cnt = None
    for n in cnt:
        if cv2.contourArea(n) > biggest_contour:
            biggest_cnt = n
        else:
            continue

    print("Found the biggest contours!")

    return biggest_cnt


def find_contours(masks, images, change_kernel=False, show_img=False):
    """
    Returns the biggest contour for an array of images.

    :param show_img: Weather or not to display the morphed images
    :param change_kernel: Changes weather or not to change the kernels by trackbars. If left false, it will use the
    default parameters 5 and 7 for open and close respectively
    :param masks: Masks to find contours of
    :param images: An array of images to find contours inside
    :return: An array with the biggest contour for each image
    """

    print("Finding contours...")

    # To make editing the kernel possible
    if change_kernel:
        # Create trackbar window and trackbars for open and close
        cv2.namedWindow("Adjust_Hue_Satuation_Value")
        cv2.createTrackbar("kernel open", "Adjust_Hue_Satuation_Value", 2, 20, nothing)
        cv2.createTrackbar("kernel close", "Adjust_Hue_Satuation_Value", 2, 20, nothing)

    # Values for each image
    old_open_val, old_closes_val = 0, 0
    contours = []
    image_n = 0
    closing = None

    # Looping through each mask
    for n in masks:
        while True:
            # If we wanna change the kernel by trackbar
            if change_kernel:
                kernel_val_open_val, kernel_val_close_val = open_close_trackbars()
            else:
                kernel_val_open_val, kernel_val_close_val = 5, 7

            # Make kernels for each morph type
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_open_val, kernel_val_open_val))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_close_val, kernel_val_close_val))

            # Only use opening and closing when the slider is moved instead of every frame
            if old_open_val != kernel_val_open_val or old_closes_val != kernel_val_close_val or change_kernel is False:
                opening = eip.morph_open(n, kernel_open)
                closing = eip.morph_close(opening, kernel_close)
                old_open_val = kernel_val_open_val
                old_closes_val = kernel_val_close_val

            # To see how much of the fish we are keeping
            if closing is not None:
                res = eip.bitwise_and(images[image_n], closing)
            else:
                warnings.warn("The closing or open operation is None!")

            # If the user wanna change the kernel values
            if change_kernel:
                cv2.imshow("Adjust_Hue_Satuation_Value", closing)

            # For debugging
            if show_img:
                cv2.imshow("Mask Closing", closing)
                cv2.imshow("Res", res)

            # Break the while loop if esc is pressed
            key = cv2.waitKey(1)
            if key == 27 or change_kernel is False:
                break

        # Find contours, implement grassfire algorithm
        contours_c, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        print(f"Contours length:{len(contours)}")

        # Getting the biggest contour which is always the fish
        if contours_c is not None:
            contours.append(find_biggest_contour(contours_c))
        else:
            print("Can't find contour")

        # Increment the image number so we have the right bitwise
        image_n = image_n + 1

    print("Found all the contours!")

    return contours


def raytracing(rotate_img, xcm, ycm, contours):
    """
    A function that can rotate a set of images, so the longest part has an angle of zero relative to the
    positive x-axis. This is done by line tracing each point on the contours relative to
    the center of mass of the contours. The angle for each line relative to the positive
    x-axis is then computed, and the angle for the longest length is then used to rotate the image.

    NOTE: The approximation method for the findContour() function should be CHAIN_APPROX_NONE, otherwise if
    CHAIN_APPROX_SIMPLE is used the results might vary negatively.

    Optional: It is possible to draw the line tracing in the img (line 164 should then be included).
    It is also possible to plot the distribution of the contour coordinates length
    relative to the angle (line 174-177 should then be included).

    :param rotate_img: Images that needs to be rotated.
    :param xcm: x-coordinates for the center of mass of the contours.
    :param ycm: y-coordinates for the center of mass of the contours.
    :param contours: Contours from the images that needs to be rotated.
    :return: The rotated images.
    """

    print("Raytracing on the image...")

    # Variable where the length and angle will be stored.
    data = []
    # Variable to store the rotated images.
    img_output = []
    for nr in range(len(contours)):
        maxLength = 0
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
            # Draw the line tracing on the contours and point (optional)
            cv2.line(rotate_img[nr], (xcm[nr], ycm[nr]), (contours[nr][point][0][0], contours[nr][point][0][1]),
                     (255, 0, 0), 1)

        # Show COF contour
        cv2.circle(rotate_img[nr], (xcm[nr], ycm[nr]), radius=4, color=(0, 0, 255), thickness=-1)

        # Plot the contour coordinates length relative to the angle (optional):
        plt.subplot(int("1" + str(len(contours)) + str(nr + 1)))
        plt.bar(data[nr]["angle"], data[nr]["length"])
        plt.axis([-180, 180, 0, 500])
        cv2.imshow("Traced", rotate_img[nr])

    print("Done raytracing!")
    plt.show()
    return rotate_img


def checkerboard_calibrate(dimensions, images_distort, images_checkerboard, show_img=False):
    """
    Undistorts images by a checkerboard calibration.

    :param show_img: Debug to see if all the images are loaded and all the edges are found
    :param dimensions: The dimensions of the checkerboard from a YAML file
    :param images_distort: The images the needs to be undistorted
    :param images_checkerboard: The images of the checkerboard to calibrate by
    :return: If it succeeds, returns the undistorted images, if it fails, returns the distorted images with a warning
    """

    print("Started calibrating...")

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((dimensions[0][1] * dimensions[1][1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for img in images_checkerboard:
        gray = eip.grayScaling(img)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
            if show_img:
                cv2.imshow('img', img)
                cv2.imshow('gray', gray)
                cv2.waitKey(0)
        else:
            warnings.warn("No ret! This might lead to a crash.")
    # The function doesn't always find the checkerboard, therefore we have to try, and if not, pass exception
    try:
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:2], None, None)

        # Go through all the images and undistort them
        img_undst = []
        for n in images_distort:
            # Get image shape
            h, w = n.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # undistorted
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
            dst = cv2.remap(n, mapx, mapy, cv2.INTER_LINEAR)

            # crop the image back to original shape
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            img_undst.append(dst)
            if show_img:
                cv2.imshow('calibresult.png', dst)
                cv2.waitKey(0)

        print("Done calibrating")

        return img_undst

    except ValueError:
        # If the calibration fails, inform us and tell us the error
        warnings.warn(f"Could not calibrate camera. Check images. Error {ValueError}")
        mean_error = 0
        tot_error = 0

        # Show the total error
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        print("total error: ", mean_error / len(objpoints))

        # If the function fails, return the input arguments
        return images_distort
