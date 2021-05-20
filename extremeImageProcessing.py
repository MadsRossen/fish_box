import numpy as np
import cv2
import warnings
import glob
import copy


def undistort(inputImgs, k_1, k_2, imgCenterX, imgCenterY, Fx, Fy, show_img=False):
    '''
    Undistorts images using parameters found in MATLAB calibration using 'standard'.

    :param inputImgs: The distorted images
    :return: The undistorted image
    '''

    print("Started camera calibration...")

    undistortedImgs = []

    for img in inputImgs:
        h, w, ch = img.shape

        undistorted = np.zeros(img.shape, np.uint8)

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
                        undistorted.itemset((input_pos_y, input_pos_x, 0), img.item((output_pos_y, output_pos_x, 0)))
                        undistorted.itemset((input_pos_y, input_pos_x, 1), img.item((output_pos_y, output_pos_x, 1)))
                        undistorted.itemset((input_pos_y, input_pos_x, 2), img.item((output_pos_y, output_pos_x, 2)))

        undistortedImgs.append(undistorted)

        if show_img:
            cv2.imshow("Undistorted img", undistorted)
            cv2.waitKey(1)

    print("Done with calibration!")

    return undistortedImgs


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


def grassFire(mask):
    """ Only input binary images of 0 and 255 """
    mask_copy = copy.copy(mask)

    h, w = mask_copy.shape[:2]

    h = h-1
    w = w-1

    save_array = []
    zero_array = []
    blob_array = []
    temp_cord = []

    for y in range(h):
        for x in range(w):
            if mask_copy.item(y, x) == 0 and x <= h:
                zero_array.append(mask_copy.item(y, x))
            elif mask_copy.item(y, x) == 0 and x >= w:
                zero_array.append(mask_copy.item(y, x))

    # Looping if x == 1, and some pixels has to be burned
            while mask_copy.item(y, x) > 0 or len(save_array) > 0:
                mask_copy.itemset((y, x), 0)
                temp_cord.append([y, x])

                if mask_copy.item(y - 1, x) > 0:
                    if [y - 1, x] not in save_array:
                        save_array.append([y - 1, x])

                if mask_copy.item(y, x - 1) > 0:
                    if [y, x - 1] not in save_array:
                        save_array.append([y, x - 1])

                if mask_copy.item(y + 1, x) > 0:
                    if [y + 1, x] not in save_array:
                        save_array.append([y + 1, x])

                if mask_copy.item(y, x + 1) > 0:
                    if [y, x + 1] not in save_array:
                        save_array.append([y, x + 1])

                if len(save_array) > 0:
                    y, x = save_array.pop()

                else:
                    blob_array.append(temp_cord)
                    print(temp_cord)
                    temp_cord = []
                    break

    return blob_array


def segment_cod(images, show_images=False):

    print("Started segmenting the cods!")

    inRangeImages = []
    segmentedImages = []

    for img in images:
        hsv_img = copy.copy(img)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

        lowerH = (99, 117)

        h, w, ch = hsv_img.shape[:3]

        mask = np.zeros((h, w), np.uint8)
        # We start segmenting
        for y in range(h):
            for x in range(w):
                H = hsv_img.item(y, x, 0)
                S = hsv_img.item(y, x, 1)
                V = hsv_img.item(y, x, 2)
                # If Hue lies in th lowerHueRange(Blue hue range) we want to segment it out
                if lowerH[1] > H > lowerH[0]:
                    mask.itemset((y, x), 0)
                else:
                    mask.itemset((y, x), 255)

        # Create kernels for morphology
        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((7, 7), np.uint8)

        # Perform morphology
        open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        close2 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernelClose)

        # NEEDS TO BE CHANGED TO OUR OWN BITWISE
        segmented_cod = cv2.bitwise_and(img, img, mask=close2)

        if show_images:
            cv2.imshow("res", segmented_cod)
            cv2.imshow("mask", mask)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cod)

    print("Finished segmenting the cods!")

    return inRangeImages, segmentedImages