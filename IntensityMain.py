import numpy as np
import cv2
import Funcs as F
import glob
import csv

from matplotlib import pyplot as plt

TestControl = glob.glob("GOPR1706.JPG")
Testdiffuse = glob.glob("GOPR1716.JPG")
Testpapirdome = glob.glob("GOPR1656.jpg")
fisk = cv2.imread("fish_pics/step1_haandholdt_closeup_GOPR1886.JPG")

# Compute intensity histogram and mean value:
controlMeanHisto, controlMeanVal = F.calcMeanHist(TestControl)
diffuseMeanHisto, diffuseMeanVal = F.calcMeanHist(Testdiffuse)
papirdomeMeanHisto, papirdomeMeanVal = F.calcMeanHist(Testpapirdome)
HSV_H_Histo = F.calcOneChanHist(fisk[:,:,0])

Test1_Text = "Dome light (Halogen)"
Test6_Text = "Control (Halogen)"
Test7_Text = "Diffused light (Halogen)"
#Test1_Text = "White dome background"
#Test6_Text = "White dome + shielding background"
#Test7_Text = "Gray background"

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
plt.plot(HSV_H_Histo, 'b', label = Test1_Text)
plt.xlim([0, 255])
plt.legend()
plt.xlabel('Intensity (unweighted)')
plt.ylabel('Number of pixels in percentage')
plt.subplots_adjust(left=0.083, right=0.995, top=0.994, bottom=0.092)
plt.show()