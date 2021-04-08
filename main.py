import cv2
import BenjaminFunctions as bf

# load images into memory
images, names = bf.loadImages("fishpics/direct2pic", True, True, 40)

# Calibrate images
img_cali, names_cali = bf.loadImages("fishpics/CheckerBoards", True, False, 40)

# Calibrate camera
fish_cali = bf.checkerboard_calibrate(images, img_cali)

# Calibrated fish images
left = fish_cali[0]
right = fish_cali[1]

# Specular highlights
img_spec_rem = [bf.replaceHighlights(left, right, 225), bf.replaceHighlights(right, left, 225)]

# Get the contours
contour = bf.find_contours(img_spec_rem)

xcm, ycm = bf.cropToROI(img_spec_rem, contour)

rot_img = bf.rotateImages(left, img_spec_rem, xcm, ycm, contour)

# display images and it's names
cv2.imshow(f"Left: {names[0]}", left)
cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", rot_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
