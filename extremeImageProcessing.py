import numpy as np
import cv2


def bitwise_and(img, mask):
    '''
    A bitwise operation to stitch a picture to a mask

    :param img: The image to reference in color
    :param mask: The mask to reference in grayscale
    :return: An image where the mask decides
    '''

    # print("Doing bitwise and operation...")

    height, width = img.shape[:2]
    main_clone = np.zeros((height, width, 3), dtype=np.uint8)
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

    # print("Done with the operation!")

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
    # kernel_ero = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2

    # Define new image
    imgErode = np.zeros((height, width), dtype=np.uint8)

    # Erosion
    for y in range(constant, height - constant):
        for x in range(constant, width - constant):
            temp = mask[y - constant:y + constant + 1, x - constant:x + constant + 1]
            product = temp * kernel_ero
            imgErode[y, x] = np.min(product)

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
    for y in range(constant1, height - constant1):
        for x in range(constant1, width - constant1):
            temp = mask[y - constant1:y + constant1 + 1, x - constant1:x + constant1 + 1]
            product = temp * kernel_di
            imgDilate[y, x] = np.max(product)

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

    h, w, = img.shape[:2]
    greyscale_img1 = np.zeros((h, w, 1), np.uint8)

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
