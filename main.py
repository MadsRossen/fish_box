import cv2

import matplotlib.pyplot as plt
from basic_image_functions import crop, resizeImg, claheHSL
from BenjaminFunctions import replaceHighlights, equalizeColoredImage, find_blood_damage, morphological_trans, \
     loadImages
from mathias_functions import isolate_img
from calibration import undistortImg

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


'Step 7: Classify'
# load images into memory
images, names = loadImages(True, 10)

left = images[0]
right = images[1]

# Specular highlights
img_spec_rem = replaceHighlights(left, right, 210)

# isolate image

isolated_img = isolate_img(images)

# Blood spots
blood_spot = find_blood_damage(img_spec_rem)
morph = morphological_trans(blood_spot)

# Convert morph to color so we can add it together with our main image
morph_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
final = cv2.addWeighted(img_spec_rem, 1, morph_color, 0.6, 0)

# display images and it's names
cv2.imshow(f"Left: {names[0]}", left)
cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
