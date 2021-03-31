import cv2

from Kasperfunctions import SURFalignment, meanEdgeRGB, doClaheLAB2, limitLchannel, crop, doClaheLAB1, resizeImg
from BenjaminFunctions import replaceHighlights

# load image
fisk = cv2.imread('fishpics/direct2pic\\GOPR1591.JPG', 1)
top = cv2.imread('fishpics/direct2pic\\GOPR1591.JPG', 1)
top = crop(top, 650, 500, 1000, 3000)
top_re = resizeImg(top, 30)

bund = cv2.imread('fishpics/direct2pic\\GOPR1590.JPG', 1)

# display image
cv2.imshow("ResultHLS", img)
cv2.waitKey(0)
cv2.destroyAllWindows()