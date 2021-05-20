import cv2
import numpy as np
import os
import math
import warnings
import extremeImageProcessing as eip

from matplotlib import pyplot as plt
# A library that has a equalize matcher!
from skimage.exposure import match_histograms


def loadImages(path, edit_images, show_img=False, scaling_percentage=30):
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


def save_img(img):
    out_folder_processed_images_path = "C:\\Users\\MOCst\\PycharmProjects\\Nye_filer"    # Make into Yaml parameter path
    count = 0
    if len(img) < 1:
        for n in img:
            count = count+1
            cv2.imwrite(out_folder_processed_images_path+f"\\fish{count}.jpg", n)
    else:
        cv2.imwrite(out_folder_processed_images_path+"\\fish.jpg", img)


def replaceHighlights(main_img, spec_img, limit):
    """
    This functions replaces the highlights from a main picture with the pixels from a specular image pixels.

    :param main_img: The image of which will get the pixels replaced with the specular image
    :param spec_img: The image of which will be used to replace the pixels of the main image
    :param limit: The limits of a pixel value before it is classified as a specular highlight
    :return: The image that has the highlights replaced
    """

    print("Replacing highlights...")

    # Create copy
    img_main_cop = np.copy(main_img)
    img_spec_cop = np.zeros((spec_img.shape[0], spec_img.shape[1], 3), np.uint8)

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

    print("Done replacing the highlights!")

    return img_main_cop


def claheHSL(img, clipLimit, tileGridSize):
    fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannelHLS = fiskHLS2[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelHLS)
    fiskHLS2[:, :, 1] = claheLchannel1
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

    print(f"Original images length: {len(orig_img)}")
    print(f"Contours len: {len(contours)}")

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
    Returns the biggest contour in a list of contours.

    :param cnt: A list of contours
    :return: The biggest contour inside the list
    """
    print("Finding the biggest contours...")
    biggest_area = 0
    biggest_cnt = None
    for n in cnt:
        if cv2.contourArea(n) > biggest_area:
            biggest_cnt = n
        else:
            continue

    print("Found the biggest contours!")

    return biggest_cnt


# Used for the trackbars
def nothing(x):
    pass


def open_close_trackbars():
    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("kernel open", "Adjust_Hue_Satuation_Value", 2, 20, nothing)
    cv2.createTrackbar("kernel close", "Adjust_Hue_Satuation_Value", 2, 20, nothing)

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


def find_contours(masks, images, change_kernel=False, show_img=False):
    """
    Returns the biggest contour for a list of images.

    :param show_img: Weather or not to display the morphed images
    :param change_kernel: Changes weather or not to change the kernels by trackbars. If left false, it will use the
    default parameters 5 and 7 for open and close respectively
    :param masks: Masks to find contours of
    :param images: A list of images to find contours inside
    :return: A list with the biggest contour for each image
    """

    print("Finding contours...")

    old_open_val, old_closes_val = 0, 0
    contours = []
    image_n = 0
    opening = None
    closing = None
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

            if change_kernel:
                cv2.imshow("Adjust_Hue_Satuation_Value", closing)

            if show_img:
                cv2.imshow("Mask", n)
                cv2.imshow("Res", res)

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


def rotateImages(rotate_img, xcm, ycm, contours):
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

    print("Done raytracing!")
    plt.show()
    return rotate_img


def isolate_img(resized_input_image):
    """
    Returns the image of which pixels was in range of the selected HSV values.

    :param resized_input_image: The image to find pixels in range of the HSV values
    :return: A mask of the isolated pixels
    """

    hsv_image = eip.convert_RGB_to_HSV(resized_input_image)

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

        res = eip.bitwise_and(resized_input_image, mask)

        cv2.imshow("res", res)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1)
        if key == 27:
            break
    return res


def segment_cod(images, cahe_clipSize, titleSize, show_images=False):

    print("Started segmenting the cods!")

    inRangeImages = []
    segmentedImages = []

    # def nothing(x):
    #    pass

    # cv2.namedWindow("res")
    # cv2.createTrackbar("lowerHseg", "res", 0, 255, nothing)
    # cv2.createTrackbar("higherHseg", "res", 0, 255, nothing)
    # cv2.createTrackbar("lowerSseg", "res", 0, 255, nothing)
    # cv2.createTrackbar("higherSseg", "res", 0, 255, nothing)

    for n in images:
        while True:
            hsv_img = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)

            # lowerHseg = cv2.getTrackbarPos("lowerHseg", "res")
            # higherHseg = cv2.getTrackbarPos("higherHseg", "res")
            # lowerSseg = cv2.getTrackbarPos("lowerSseg", "res")
            # higherSseg = cv2.getTrackbarPos("higherSseg", "res")

            # lowerH = (lowerHseg, higherHseg)
            # lowerV = (lowerSseg, higherSseg)
            lowerH = (90, 128)
            lowerV = (0, 40)
            h, w, ch = hsv_img.shape[:3]

            mask = np.zeros((h, w), np.uint8)
            # We start segmenting
            for y in range(h):
                for x in range(w):
                    H = hsv_img.item(y, x, 0)
                    V = hsv_img.item(y, x, 2)
                    # If Hue lies in the lowerHueRange(Blue hue range) we want to segment it out
                    if lowerH[1] > H > lowerH[0]:
                        mask.itemset((y, x), 0)
                    # If Hue lies in the lowerValRange(black value range) we want to segment it out
                    elif lowerV[1] > V > lowerV[0]:
                        mask.itemset((y, x), 0)
                    else:
                        mask.itemset((y, x), 255)

            # NEEDS TO BE CHANGED TO OUR OWN BITWISE
            segmentedImg = cv2.bitwise_and(clahe, clahe, mask=mask)

            if show_images:
                cv2.imshow("res", segmentedImg)
                cv2.imshow("mask", mask)

                key = cv2.waitKey(1)
                if key == 27:
                    break

            break

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmentedImg)

    print("Finished segmenting the cods!")

    return inRangeImages, segmentedImages


def detect_bloodspots(hsv_img):
    lowerH = (80, 125)
    lowerS = (153, 204)
    lowerv = (115, 140)

    h, w, ch = hsv_img.shape[:3]

    segmentedImg = np.zeros((h, w), np.uint8)
    # We start segmenting
    for y in range(h):
        for x in range(w):
            H = hsv_img.item(y, x, 0)
            S = hsv_img.item(y, x, 1)
            V = hsv_img.item(y, x, 2)
            # If Hue lies in the lowerHueRange(Blue hue range) we want to segment it out
            if lowerH[1] > H > lowerH[0]:
                segmentedImg.itemset((y, x), 0)
            # If Hue lies in the lowerValRange(black value range) we want to segment it out
            elif lowerv[1] > V > lowerv[0]:
                segmentedImg.itemset((y, x), 0)
            elif lowerS[1] > S > lowerS[0]:
                segmentedImg.itemset((y, x), 0)
            else:
                segmentedImg.itemset((y, x), 255)


# Histogram check
def images_for_rapport(images):
    equalized_images = []
    image = 0
    for n in images:
        # Equalized images
        gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        equalized_images.append(equalize)

        # Plotting
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].hist(n.ravel(), 256, [0, 256])
        axs[1].hist(equalize.ravel(), 256, [0, 256])
        cv2.imshow(f"Image Equalized: {image}", equalize)
        cv2.imshow(f"Image Not Equalized: {image}", gray)
        plt.show()

        image += 1
        cv2.destroyAllWindows()

    return 0
