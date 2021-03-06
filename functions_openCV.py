import copy
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


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

    print("Intrinsic parameters:")
    print("Camera matrix: K =")
    print(mtx)
    print("\nDistortion coefficients =")
    print(dist, "\n")

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


def detect_woundspotsOPENCV(imgs, maskCod):
    '''
    Detect bloodspots, mark and tag them and find the coverage of bloodspots on hte cod

    :param imgs: Images with cod
    :param maskCod: The mask showing only the cod area
    :return: mask of blood spots, segmented blood spots, marked and tagged blood spots, coverage of blood spots on the
    cod
    '''
    mask_woundspots = []
    segmented_blodspots_imgs = []
    marked_woundspots_imgs = []
    percSpotCoverage = []
    count = 0

    # Find biggest contour
    for n in imgs:
        hsv_img = cv2.cvtColor(copy.copy(n), cv2.COLOR_BGR2HSV)
        biggestarea = 0
        fishContours, __ = cv2.findContours(maskCod[count], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cont in fishContours:
            area = cv2.contourArea(cont)
            if area > biggestarea:
                biggestarea = area

        fishArea = biggestarea

        spotcount = 0

        marked_woundspots_imgs.append(copy.copy(n))

        # Threshold for blood spots
        frame_threshold1 = cv2.inRange(hsv_img, (0, 90, 90), (10, 255, 255))

        # Combining the masks
        mask_woundspots.append(frame_threshold1)

        # Create kernels for morphology
        kernelClose = np.ones((30, 30), np.uint8)

        # Perform morphology
        close = cv2.morphologyEx(mask_woundspots[count], cv2.MORPH_CLOSE, kernelClose)

        # Perform bitwise operation to show woundspots instead of BLOBS
        segmented_blodspots_imgs.append(cv2.bitwise_and(n, n, mask=close))

        # Make representation of BLOB / woundspots
        # Find contours
        contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Classify as blood spots if the spots are big enought
        totalSpotArea = 0
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 50:
                x, y, w, h = cv2.boundingRect(cont)
                # Create tag
                cv2.putText(marked_woundspots_imgs[count], 'Wound', (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                            3)
                # Draw green contour
                cv2.rectangle(marked_woundspots_imgs[count], (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2);

                # Because the biologists have put a red tag on the cod, there will always be detected at least 1 blood
                # spot
                spotcount = spotcount + 1
                if spotcount > 1:
                    totalSpotArea = totalSpotArea + area

        percSpotCoverage.append(totalSpotArea / fishArea * 100)

        count = count + 1

    return mask_woundspots, marked_woundspots_imgs, segmented_blodspots_imgs, percSpotCoverage


def resizeImg(img, scale_percent):
    """
    Resizes the image by a scalar

    :param img: the image to resize
    :param scale_percent: The scaling percentage
    :return: the image scaled by the scalar
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # resize image
    return resized


def segment_codOPENCV(images, show_images=False):
    print("Started segmenting the cod!")

    inRangeImages = []
    segmentedImages = []

    for n in images:
        hsv_img = copy.copy(n)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

        # Create threshold for segmenting cod
        mask = cv2.inRange(hsv_img, (100, 21, 65), (180, 255, 255))

        # Invert the mask
        mask = (255 - mask)

        # Create kernels for morphology
        # kernelOpen = np.ones((4, 4), np.uint8)
        # kernelClose = np.ones((7, 7), np.uint8)

        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((5, 5), np.uint8)

        # Perform morphology
        open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen, iterations=3)
        close2 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernelClose, iterations=5)

        segmented_cods = cv2.bitwise_and(n, n, mask=close2)

        segmented_cods[close2 == 0] = (255, 255, 255)

        if show_images:
            cv2.imshow("res", segmented_cods)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cods)

    print("Finished segmenting the cod!")

    return inRangeImages, segmentedImages


def showSteps(stepsList, CLAHE=False):
    '''
    Create subplots showing main steps in algorithm

    :return: None
    '''

    # OpenCV loads pictures in BGR, but the this step is plotted in RGB:
    count = 0
    img_rgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    img_undistorted_rgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    img_cropped_rgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    if CLAHE:
        cod_CLAHErgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
        count = count + 1
    img_segmented_codrgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    bloodspotsrgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)
    count = count + 1
    marked_bloodspotssrgb = cv2.cvtColor(stepsList[count], cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    fig.suptitle('Steps in algorithm', fontsize=16)

    subplot = 1
    plt.subplot(3, 3, subplot)
    plt.imshow(img_rgb)
    plt.title('Original image')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(img_undistorted_rgb)
    plt.title('Undistorted image')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(img_cropped_rgb)
    plt.title('ROI')
    subplot = subplot + 1

    if CLAHE:
        plt.subplot(3, 3, subplot)
        plt.imshow(cod_CLAHErgb)
        plt.title('CLAHE')
        subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(img_segmented_codrgb)
    plt.title('Segmented cod')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(bloodspotsrgb)
    plt.title('Blood spots segmented')
    subplot = subplot + 1

    plt.subplot(3, 3, subplot)
    plt.imshow(marked_bloodspotssrgb)
    plt.title('Blood spots tagged')
    subplot = subplot + 1

    plt.show()


def save_imgOPENCV(imgs, path, originPathNameList, img_tag=None):
    '''
    Saves a list of images in the folder that the path is set to.

    :param originPathName: The path of the original path of the images that have been manipulated.
    :param imgs: A list of images.
    :param path: The path that the images will be saved to.
    :param img_tag: Tag that is is added to the original image file name e.g. filname + img_tag = filnameimg_tag.JPG
    :return: None
    '''

    print('Saving images')

    count = 0
    if len(imgs) > 0:
        for n in imgs:
            cv2.imwrite(path + f"\\{originPathNameList[count] + img_tag}.JPG", n)
            count = count + 1

    print('Done saving images')


def crop(images, y, x, height, width):
    '''
    Crops images.

    :param images: The images to crop
    :return: Cropped images
    '''

    print("Cropping images ... ")
    cropped_images = []
    for n in images:
        ROI = n[y:y + height, x:x + width]
        cropped_images.append(ROI)

    print("Done cropping images!")

    return cropped_images


def loadImages(path, edit_images=False, show_img=False, scaling_percentage=30):
    """
    Loads all the images inside a file.

    :return: All the images in a list and its file names.
    """

    images = []
    class_names = []
    img_list = os.listdir(path)
    print("Loading in images...")
    print("Total images found:", len(img_list))
    for cl in img_list:
        # Find all the images in the file and save them in a list without the ".jpg"
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

        # Append them into the list
        images.append(cur_img)
        class_names.append(img_name)

    # Remove the image window after we have checked all the pictures
    cv2.destroyAllWindows()

    print("Done loading the images!")

    return images, class_names


def saveCDI(img_list_fish, percSpotCoverage):
    """
    Saves a CDI of each fish in a .txt file.

    :param img_list_fish: The names of each fish
    :param percSpotCoverage: The percentage the wound covers the surface area of one side of the fish
    :return:
    """

    print("Started CDI...")

    f = open("output_CDI/CDI.txt", "w+")

    line = ("-----------------------------------------------------------------\n")

    # Layout for CDI
    f.write("CDI\n\n")
    f.write("              |Category: \t|Wounds \n")
    f.write(line)
    f.write("FISH:\n")

    # Fill CDI
    for i in range(len(img_list_fish)):
        name = os.path.splitext(img_list_fish[i])[0]

        if percSpotCoverage[i] > 0:
            f.write(line)
            f.write("%s \t\t\t|x, coverage = " % name)
            f.write("%.3f" % percSpotCoverage[i])
            f.write("%\n")
        elif percSpotCoverage[i] == 0:
            f.write(line)
            f.write("%s \t\t\t| \n" % name)

    f.write(line)
    f.close()

    print("Done writing the CDI!")


def segment_cod_CLAHEOPENCV(images, show_images=False):
    print("Started segmenting the cod!")

    inRangeImages = []
    segmentedImages = []

    for n in images:
        hsv_img = copy.copy(n)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

        # Create threshold for segmenting cod
        mask = cv2.inRange(hsv_img, (99, 15, 30), (123, 255, 255))

        # Invert the mask
        mask = (255 - mask)

        # Create kernels for morphology
        # kernelOpen = np.ones((4, 4), np.uint8)
        # kernelClose = np.ones((7, 7), np.uint8)

        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((5, 5), np.uint8)

        # Perform morphology
        open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen, iterations=3)
        close2 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernelClose, iterations=5)

        segmented_cods = cv2.bitwise_and(n, n, mask=close2)

        segmented_cods[close2 == 0] = (255, 255, 255)

        if show_images:
            cv2.imshow("res", segmented_cods)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cods)

    print("Finished segmenting the cod!")

    return inRangeImages, segmentedImages


def claheHSL(imgList, clipLimit, tileGridSize):
    '''
    Performs CLAHE on a list of images
    '''
    fiskClaheList = []
    for img in imgList:
        fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        LChannelHLS = fiskHLS2[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        claheLchannel1 = clahe.apply(LChannelHLS)
        fiskHLS2[:, :, 1] = claheLchannel1
        fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)
        fiskClaheList.append(fiskClahe)
    return fiskClaheList
