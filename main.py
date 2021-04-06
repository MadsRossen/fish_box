import cv2

from Kasperfunctions import crop, resizeImg
from BenjaminFunctions import replaceHighlights, equalizeColoredImage, find_blood_damage, morphological_trans, \
     loadImages

# load images
left = cv2.imread('fishpics/direct2pic/Left.JPG', 1)
left_crop = crop(left, 650, 500, 1000, 3000)
left_re = resizeImg(left_crop, 40)
left_eq = equalizeColoredImage(left_re)

right = cv2.imread('fishpics/direct2pic/Right.JPG', 1)
right_crop = crop(right, 650, 500, 1000, 3000)
right_re = resizeImg(right_crop, 40)
right_eq = equalizeColoredImage(right_re)

# Specular highlights
img_spec_rem = replaceHighlights(left_eq, right_eq, 210)

# Blood spots
blood_spot = find_blood_damage(img_spec_rem)
morph = morphological_trans(blood_spot)

# Convert morph to color so we can add it together with our main image
morph_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
final = cv2.addWeighted(img_spec_rem, 1, morph_color, 0.6, 0)

# display image
cv2.imshow("Left", left_eq)
cv2.imshow("Right", right_eq)
cv2.imshow("Final", final)
cv2.waitKey(0)
cv2.destroyAllWindows()