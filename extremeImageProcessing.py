import numpy as np
import cv2
import BenjaminFunctions as bf


def bitwise_and(img, mask):
    '''
    A bitwise operation to stitch a picture to a mask

    :param img: The image to reference in color
    :param mask: The mask to reference in grayscale
    :return: An image where the mask decides
    '''

    print("Doing bitwise and operation...")

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

    print("Done with the operation!")

    return main_clone


def make_img_bit(images):
    bit_images = []
    for img in images:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        height, width = img.shape[:2]
        bit_img = np.zeros((height, width, 1), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                value = img.item(y, x, 0)
                if value == 255:
                    bit_img.itemset((y, x, 0), 1)
                else:
                    bit_img.itemset((y, x, 0), 0)

        bit_images.append(bit_img)
        cv2.waitKey(0)

    return bit_images


def erosion():
    print("Fit")


def dilation():
    print("Hit")


def morph_close():

    print("close")


def morph_open():
    print("Open")


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
        img_hsv = bf.convert_RGB_to_HSV(img)
        img_copy = img.copy()

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

        cv2.imshow("Mask", img_copy)
        cv2.waitKey(0)
        img_iso.append(img_copy)

    print("Done creating masks!")

    return img_iso
