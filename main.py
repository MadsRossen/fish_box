import cv2
import matplotlib.pyplot as plt

from basic_image_functions import crop, claheHSL
from calibration import undistortImg
from mathias_functions import convert_RGB_to_HSV, smallrange_isolate_img_content, detect_bloodspots, isolate_img

'Step 1: Load image'
img = cv2.imread(f"fish_pics/GOPR1911.JPG", 1)

'Step 2: undistort image'
img_undistorted = undistortImg(img, False)

'Step 3: Crop to ROI'
img_cropped = crop(img_undistorted, 900, 650, 500, 1500)

'Step 4: Apply CLAHE'
img_CLAHE = claheHSL(img_cropped, 2, (20,20))

'Step 5: Show pre-processing steps in one plot'
# OpenCV loads pictures in BGR, but the this step is plotted in RGB:
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_undistorted_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
img_CLAHE_rgb = cv2.cvtColor(img_CLAHE, cv2.COLOR_BGR2RGB)

# Create subplots
fig = plt.figure()
fig.suptitle('Steps in pre-processing', fontsize=16)

plt.title('Original image')
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)

plt.title('Undistorted image')
plt.subplot(2, 2, 2)
plt.imshow(img_undistorted_rgb)

plt.title('ROI')
plt.subplot(2, 2, 3)
plt.imshow(img_cropped_rgb)

plt.title('ROI with CLAHE applied')
plt.subplot(2, 2, 4)
plt.imshow(img_CLAHE_rgb)

plt.show()

'Step 6: Segment'
img_HSV = cv2.cvtColor(img_CLAHE, cv2.COLOR_BGR2HSV)
img_segmented_cod = smallrange_isolate_img_content(img_CLAHE, img_HSV)
cv2.imshow('img_HSV', img_HSV)

cv2.imshow('img_segmented_cod',img_segmented_cod)

'Step 7: Classify'
'''
# load images into memory
#images, names = loadImages("fishpics/direct2pic", True, True, 40)

# Calibrate images
#img_cali, names_cali = loadImages("fishpics/CheckerBoards", True, False, 40)

# Calibrate camera
#fish_cali = checkerboard_calibrate(images, img_cali)

# Calibrated fish images
#left = images[0]
#right = images[1]

# Specular highlights
#img_spec_rem = replaceHighlights(left, right, 225)  # 230


with_polfilter = cv2.imread("fishpics/direct2pic/with_polfilter.jpg")

mask_gras = cv2.imread("fishpics/direct2pic/TestGrassFire.png", 0)

#making gray scale image without opencv
#grayscale = grayScaling(with_polfilter)

#making gray scale image 8 bit without 8 bit
#grayscaleimg8bit = grayScaling8bit(with_polfilter)

#making hsv image without opencv
hsv_img = convert_RGB_to_HSV(with_polfilter)
#cv2.imshow("hvs_own", hsv_img)
# Calibrating hsv mask to mask to grassfire
#isolated_mask = isolate_img(with_polfilter, hsv_img)
#runnig mask with hsv img setup
#mask_to_grassfire = creating_mask_input_hsv(with_polfilter, hsv_img)
#running grassfire algorithm
segmentedimg = smallrange_isolate_img_content(with_polfilter, hsv_img)

hsv_img_segmented = convert_RGB_to_HSV(segmentedimg)
detect_bloodspots(segmentedimg, hsv_img_segmented)
#grassfire_transform(mask_to_grassfire, with_polfilter)
#grassfirealgorithm
#grassfire_v2(smallrange)
# segmentation
#isolated_mask = isolate_img(with_polfilter, hsv_img)

#erosion
#erosion(isolated_mask)

#save each processed image
#save_img(left)

#canny detection
#canny_edge_detection(with_polfilter, mask_to_grassfire)

# Get the contours
#find_contours(fish_cali)

# Blood spots
#blood_spot = find_blood_damage(img_spec_rem)
#morph = morphological_trans(blood_spot)

# Convert morph to color so we can add it together with our main image
#morph_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
#final = cv2.addWeighted(img_spec_rem, 1, morph_color, 0.6, 0)

# display images and it's names
#cv2.imshow(f"Left: {names[0]}", left)
#cv2.imshow(f"Right: {names[1]}", right)
#cv2.imshow("Final", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''