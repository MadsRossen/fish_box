import cv2

from Kasperfunctions import crop, resizeImg
from BenjaminFunctions import replaceHighlights, equalizeColoredImage

# load image
left = cv2.imread('fishpics/direct2pic/GOPR1591.JPG', 1)
left_crop = crop(left, 650, 500, 1000, 3000)
left_re = resizeImg(left_crop, 30)
left_eq = equalizeColoredImage(left_re)

right = cv2.imread('fishpics/direct2pic/GOPR1590.JPG', 1)
right_crop = crop(right, 650, 500, 1000, 3000)
right_re = resizeImg(right_crop, 30)
right_eq = equalizeColoredImage(right_re)

replaceHighlights(left_eq, right_eq, 215, 100)

# display image
cv2.imshow("left", left_eq)
cv2.imshow("right", right_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()