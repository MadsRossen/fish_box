import cv2
import functions as ft
import yamlLoader as yamlL
import extremeImageProcessing as eip
import numpy as np

from scipy import ndimage

# Load in yaml data from the file
yaml_data = yamlL.yaml_loader("parameters.yaml")

# Load in the yaml parameters from the data
kernels, checkerboard_dimensions, paths = yamlL.setup_parameters(yaml_data)

# load images into memory
images, names = ft.loadImages(paths[0][1], True, False, 40)

# Calibrate images
img_cali, names_cali = ft.loadImages(paths[1][1], True, False, 40)

# Calibrate camera
fish_cali = ft.checkerboard_calibrate(checkerboard_dimensions, images, img_cali, False)

# Calibrated fish images
left = fish_cali[0]
right = fish_cali[1]

# Specular highlights
img_spec_rem = [ft.replaceHighlights(left, right, 225), ft.replaceHighlights(right, left, 225)]

# Threshold to create a mask for each image
masks = eip.findInRange(img_spec_rem)

# Erosion
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ero = eip.erosion(masks[0], kernel_open)
cv2.imshow("Ero", ero)
cv2.waitKey(0)

# Get the contours
contour = ft.find_contours(masks, img_spec_rem)

# Find contours middle point
xcm, ycm = ft.contour_MOC(img_spec_rem, contour)

# Raytracing
rot_img = ft.rotateImages(img_spec_rem, xcm, ycm, contour)

# display images and it's names
cv2.imshow(f"Left: {names[0]}", left)
cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", rot_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
