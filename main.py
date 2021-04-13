import cv2
import BenjaminFunctions as bf
import yamlLoader as yamlL
import extremeImageProcessing as eip

# Load in yaml data from the file
yaml_data = yamlL.yaml_loader("parameters.yaml")

# Load in the yaml parameters from the data
kernels, checkerboard_dimensions, paths = yamlL.setup_parameters(yaml_data)

# load images into memory
images, names = bf.loadImages(paths[0][1], True, False, 40)

# Calibrate images
img_cali, names_cali = bf.loadImages(paths[1][1], True, False, 40)

# Calibrate camera
fish_cali = bf.checkerboard_calibrate(checkerboard_dimensions, images, img_cali, True)

# Calibrated fish images
left = fish_cali[0]
right = fish_cali[1]

# Specular highlights
img_spec_rem = [bf.replaceHighlights(left, right, 225), bf.replaceHighlights(right, left, 225)]

# Threshold to create a mask for each image
img = eip.findInRange(img_spec_rem)

# Make bit image for contour and morph
bit = eip.make_img_bit(img)
cv2.imshow("bit img", bit[0])
cv2.waitKey(0)

# Get the contours
contour = bf.find_contours(img, img_spec_rem)

# Find contours middle point
xcm, ycm = bf.cropToROI(img_spec_rem, contour)

# Raytracing
rot_img = bf.rotateImages(img_spec_rem, xcm, ycm, contour)

# display images and it's names
cv2.imshow(f"Left: {names[0]}", left)
cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", rot_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
