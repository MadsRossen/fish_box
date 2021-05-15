import cv2

from Kasperfunctions import crop, resizeImg
from BenjaminFunctions import replaceHighlights, equalizeColoredImage, find_blood_damage, morphological_trans, \
     loadImages
from mathias_functions import isolate_img

'Step 1: Load image'
img = cv2.imread(f"fish_pics/GOPR1911.JPG", 1)

'Step 2: undistort image'


'Step 3: Crop to ROI'

'Step 4: Apply CLAHE'

'Step 5: Apply CLAHE'


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
