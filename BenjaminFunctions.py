import cv2
import numpy as np


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

    print(main_img_spec)

    # Replace pixels with
    img_main_cop[main_img_spec] = spec_img[main_img_spec]

    cv2.imshow("spec", img_main_cop)


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
