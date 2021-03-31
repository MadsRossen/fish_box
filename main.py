import cv2

from Kasperfunctions import SURFalignment, meanEdgeRGB, doClaheLAB2, limitLchannel, crop, doClaheLAB1, resizeImg

# load images
fisk = cv2.imread('fishpics/direct2pic\\GOPR1591.JPG', 1)

top = cv2.imread('fishpics/direct2pic\\GOPR1591.JPG', 1)
top = crop(top, 650,500,1000,3000)

bund = cv2.imread('fishpics/direct2pic\\GOPR1590.JPG', 1)
bund = crop(bund, 650,500,1000,3000)

edge1 = cv2.imread('fishpics/edgeFisk.png', 1)

meanEdgeRGB(edge1, 130)

SURFalignment(top,bund)

fisk = resizeImg(fisk, 70)
fiskROI = crop(fisk, 410, 500, 800, 1000)

cv2.imshow('fiskROI',fiskROI)
cv2.waitKey(0)

removeSpcHighTest = limitLchannel(fiskROI, 150)
cv2.imshow('removeSpcHighTest',removeSpcHighTest)
cv2.waitKey(0)

# Load image that will be performed CLAHE on
img = removeSpcHighTest

# create window and add trackbar
cv2.namedWindow('ResultHLS')

global maxCliplimit, maxTilesize
maxCliplimit = 100 # Is devided by 10 in output
maxTilesize = 100

cv2.createTrackbar('cliplimit','ResultHLS', 1, maxCliplimit, doClaheLAB1)
cv2.createTrackbar('tilesize','ResultHLS', 2, maxTilesize, doClaheLAB2)

# Display image
cv2.imshow("ResultHLS", img)
cv2.waitKey(0)
cv2.destroyAllWindows()