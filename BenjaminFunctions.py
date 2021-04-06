import cv2
import numpy as np
import os

from Kasperfunctions import resizeImg, crop


def loadImages(edit_images, scaling_percentage=30):
    '''
    Loads all the images inside a file.

    :return: All the images in a list and its file names.
    '''
    path = "fishpics/direct2pic"
    images = []
    class_names = []
    img_list = os.listdir(path)
    print("Total images found", len(img_list))

    for cl in img_list:
        # Find all the images in the file and save them in a list without the ".jpg"
        cur_img = cv2.imread(f"{path}/{cl}", 1)

        # Do some quick images processing to get better pictures if the user wants to
        if edit_images:
            cur_img_crop = crop(cur_img, 650, 500, 1000, 3000)
            cur_img_re = resizeImg(cur_img_crop, scaling_percentage)
            cur_img = cur_img_re

        # Show the image before we append it, to make sure it is read correctly
        img_name = os.path.splitext(cl)[0]
        cv2.imshow(f"Loaded image: {img_name}", cur_img)
        cv2.waitKey(0)

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

    # Copy
    img_main_cop = np.copy(main_img)

    # Isolate the areas where the color is white
    main_img_spec = np.where((img_main_cop[:, :, 0] >= limit) & (img_main_cop[:, :, 1] >= limit) &
                             (img_main_cop[:, :, 2] >= limit))

    # Replace pixels
    img_main_cop[main_img_spec] = spec_img[main_img_spec]

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

