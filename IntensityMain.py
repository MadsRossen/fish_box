from __future__ import print_function

import argparse
import glob

import cv2
import cv2 as cv
from matplotlib import pyplot as plt

import Funcs as F

TestControl = glob.glob("GOPR1706.JPG")
Testdiffuse = glob.glob("GOPR1716.JPG")
Testpapirdome = glob.glob("GOPR1656.jpg")
fisk = cv2.imread("fish_pics/input_images/step1_nytest_GOPR1956.JPG")

# Compute intensity histogram and mean value:
controlMeanHisto, controlMeanVal = F.calcMeanHist(TestControl)
diffuseMeanHisto, diffuseMeanVal = F.calcMeanHist(Testdiffuse)
papirdomeMeanHisto, papirdomeMeanVal = F.calcMeanHist(Testpapirdome)
#HSV_H_Histo = F.calcHueChanHist(fisk[:,:,0])
#RGB_B_Histo = F.calcHueChanHist(fisk[:,:,2])

Test1_Text = "Dome light (Halogen)"
Test6_Text = "Control (Halogen)"
Test7_Text = "Diffused light (Halogen)"
Test8_Text = ""
#Test1_Text = "White dome background"
#Test6_Text = "White dome + shielding background"
#Test7_Text = "Gray background"

'''
# Uanset hvor mange plot jeg laver,
# så er det første plot altid en combination af dem, ved ikke hvorfor det sker :/
plt.plot(controlMeanHisto, 'r', label = Test6_Text)
plt.xlim([0, 256])
plt.legend()
plt.xlabel('Intensity (unweighted)')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()

# Show histograms
plt.plot(controlMeanHisto, 'r', label = Test6_Text)
plt.xlim([0, 255])
plt.legend()
plt.xlabel('Intensity (unweighted)')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()

# Show histograms
plt.plot(diffuseMeanHisto, 'g', label = Test7_Text)
plt.xlim([0, 255])
plt.legend()
plt.xlabel('Intensity (unweighted)')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()

# Show histograms
plt.plot(papirdomeMeanHisto, 'b', label = Test1_Text)
plt.xlim([0, 255])
plt.legend()
plt.xlabel('Intensity (unweighted)')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()

# Show histograms
#plt.plot(HSV_H_Histo, 'black')
plt.xlim([0, 255])
plt.legend()
plt.xlabel('Hue')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()

# Show histograms
#plt.plot(RGB_B_Histo, 'black')
plt.xlim([0, 255])
plt.legend()
plt.xlabel('Hue')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()
'''

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
while True:

    frame = cv2.imread('fish_pics/segment_this.JPG', 1)

    # percent by which the image is resized
    scale_percent = 70

    # calculate the 50 percent of original dimensions
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    frame = cv2.resize(frame, dsize)
    if frame is None:
        break
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # try 1 : frame_threshold = cv.inRange(frame_HSV, (99, 30, 70), (180, 255, 255))

    cv.imshow(window_capture_name, frame)
    segmented_blodspots = cv2.bitwise_or(frame, frame, mask=frame_threshold)
    cv.imshow(window_detection_name, segmented_blodspots)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break