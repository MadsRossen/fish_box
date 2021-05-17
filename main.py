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

# load Fish images into memory
images, names = ft.loadImages(paths[2][1], True, False, 40)

# Load calibration images
img_cali, names_cali = ft.loadImages(paths[1][1], True, False, 40)

# load Fish images into memory
# images_histo_test, names_histo_test = ft.loadImages(paths[2][1], True, False, 40)

# Rapport images, Histogram
# img = ft.images_for_rapport(images_histo_test)

# Calibrate camera
fish_cali = ft.checkerboard_calibrate(checkerboard_dimensions, images, img_cali, False)

# Calibrated fish images for spec remove
# left = fish_cali[0]
# right = fish_cali[1]

# Specular highlights
# img_spec_rem = [ft.replaceHighlights(left, right, 225), ft.replaceHighlights(right, left, 225)]

# Threshold to create a mask for each image
# masks = eip.findInRange(img_spec_rem)
masks = eip.findInRange(fish_cali)

# Get the contours
contour = ft.find_contours(masks, fish_cali)

# Find contours middle point
xcm, ycm = ft.contour_MOC(fish_cali, contour)

# Raytracing
rot_img = ft.rotateImages(fish_cali, xcm, ycm, contour)

# display images and it's names
#cv2.imshow(f"Left: {names[0]}", left)
#cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", rot_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
