import cv2
import numpy as np


def replaceHighlights(main_img, spec_img):
    main_img_spec = np.where((main_img[:, :, 0] == 255) & (main_img[:, :, 1] == 255) & (main_img[:, :, 2] == 255))
    cv2.imshow("spec", main_img_spec)