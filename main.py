import cv2

from BenjaminFunctions import replaceHighlights, equalizeColoredImage, find_blood_damage, morphological_trans, \
     loadImages, checkerboard_calibrate, find_contours, isolate_img

# load images into memory
images, names = loadImages("fishpics/direct2pic", True, True, 40)

# Calibrate images
img_cali, names_cali = loadImages("fishpics/CheckerBoards", True, False, 40)

# Calibrate camera
fish_cali = checkerboard_calibrate(images, img_cali)

# Calibrated fish images
left = fish_cali[0]
right = fish_cali[1]

# HSV changer
# isolate_img(left)

# Specular highlights
img_spec_rem = replaceHighlights(left, right, 225)  # 230

# Get the contours
find_contours(fish_cali)

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
