import cv2

from donefunctions import SURFalignment, meanEdgeRGB, doClaheLAB2, limitLchannel, crop, doClaheLAB1, resizeImg

# load image
fisk = cv2.imread('fishpics/direct2pic\\GOPR1591.JPG', 1)

top = cv2.imread('fishpics/direct2pic\\GOPR1591.JPG', 1)
top = crop(top, 650,500,1000,3000)

bund = cv2.imread('fishpics/direct2pic\\GOPR1590.JPG', 1)
bund = crop(bund, 650,500,1000,3000)

edge1 = cv2.imread('fishpics/direct2pic', 1)

meanEdgeRGB(edge1)

SURFalignment(top,bund)

fisk = resizeImg(fisk, 70)

y = 410
x = 500
h = 800
w = 1000

fiskROI = crop(fisk, y, x, h, w)
cv2.imshow('fiskROI',fiskROI)
cv2.waitKey(0)

val1 = 0
val2 = 1
kernel = (1,1)

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

# display image
cv2.imshow("ResultHLS", img)
cv2.waitKey(0)
cv2.destroyAllWindows()