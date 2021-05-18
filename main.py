import cv2
import functions as ft
import yamlLoader as yamlL
import extremeImageProcessing as eip
import time

# Check the runtime
start_time = time.time()

# Load in yaml data from the file
yaml_data = yamlL.yaml_loader("parameters.yaml")

# Load in the yaml parameters from the data
kernels, checkerboard_dimensions, paths, clahe = yamlL.setup_parameters(yaml_data)

# load Fish images
images, names = ft.loadImages(paths[0][1], False, False, 40)

# Load checkerboard images
img_cali, names_cali = ft.loadImages(paths[1][1], False, False, 40)

# Calibrate camera and undistort images
fish_cali = ft.checkerboard_calibrate(checkerboard_dimensions, images, img_cali, False)

# Crop to ROI
cropped_images = eip.crop(images, 700, 450, 600, 2200)

# Threshold to create a mask for each image
# masks = eip.findInRange(fish_cali) - Might need to get removed

# Threshold to create a mask for each image
masks, segmented_images = ft.segment_cod(cropped_images, clahe[0][1], clahe[1][1], False)

# Bloodspot detection
# blodspot_img = ft.detect_bloodspots()

# Get the contours
contour = ft.find_contours(masks, cropped_images)

# Find contours middle point
xcm, ycm = ft.contour_MOC(cropped_images, contour)

# Raytracing
rot_img = ft.rotateImages(cropped_images, xcm, ycm, contour)

# Print runtime
print("Execution time: ","--- %s seconds ---" % (time.time() - start_time))

# display images and it's names
#cv2.imshow(f"Left: {names[0]}", left)
#cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", rot_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
