import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import GrassFire as gf

filename = cv.imread("File path", 0)
img = cv.cvtColor(filename, cv.COLOR_GRAY2RGB)

# Parameters
windowSize = 3  
k = 0.04 # Parameter between 0.04 - 0.06
threshold = 10000

CheckPoints = 54

# Til test af billeder set til true eller false:
test = False

offset = int(windowSize/2)

x_size = filename.shape[1] - offset
y_size = filename.shape[0] - offset

nul = np.zeros((img.shape[0], img.shape[1]), np.uint8)

# mean blur
blur = cv.blur(filename, (3, 3))

# Partial differentiation hvor ** = ^2
Iy, Ix = np.gradient(blur)
# Repræsentation af M matricen
Ixx = Ix**2
Ixy = Iy*Ix
Iyy = Iy**2

CornerList = []

# Fra offset til y_size og offset til x_size
print("Start running corner detection . . . ")

for y in range(offset, y_size):
    for x in range(offset, x_size):

        # Variabler for det window den kører over hver windowSize
        start_x = x - offset
        end_x = x + offset + 1
        start_y = y - offset
        end_y = y + offset + 1

        # Laver det window den køre med
        windowIxx = Ixx[start_y : end_y, start_x : end_x]
        windowIxy = Ixy[start_y : end_y, start_x : end_x]
        windowIyy = Iyy[start_y : end_y, start_x : end_x]

        # Summed af det enkelte window
        Sxx = windowIxx.sum()
        Sxy = windowIxy.sum()
        Syy = windowIyy.sum()

        # Beregner determinanten og dirgonalen(tracen) for mere info --> se Jacobian formula
        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy

        # finder r for harris corner detection equation
        r = det - k * (trace**2)

        if bool(test):
            CornerList.append([x, y, Ix[x,y], Iy[x,y], r])

        if r > threshold:
            nul[y,x] = 255
            img[y,x] = (0,255,0)

print("Starting GrassFire . . .")
Objects = gf.GrassFire(nul)

# Sort the list by the mass of objects
print("Number of objects: ", len(Objects))
ObjectsH = sorted(Objects, key=len, reverse=True)

# Take the 54 biggest objects and make a circle around it. 54 = number of points at the checkerboard
for h in range(CheckPoints):
    corner = np.array(ObjectsH[h])
    y_min = min(corner[:,0])
    y_max = max(corner[:,0])
    x_min = min(corner[:,1])
    x_max = max(corner[:,1])

    # Calculate the center of mass for each object
    xbb = int((x_min + x_max)/2)
    ybb = int((y_min + y_max)/2)

    img[ybb, xbb] = ( 255, 0, 0)

    # Draw a circle around the center
    cv.circle(img, (xbb, ybb), 30, (255, 0, 0), thickness = 2, lineType = cv.LINE_8)
    

if bool(test):
    print('Creating file')

    CornerFile = open('CornersFound.txt', 'w')
    CornerFile.write('x, \t y, \t Ix, \t Iy, \t R \n')
    for i in range(len(CornerList)):
        CornerFile.write(str(CornerList[i][0]) + ' , ' + str(CornerList[i][1]) + ' , ' + str(CornerList[i][2]) + ' , ' + str(CornerList[i][3]) + ' , ' + str(CornerList[i][4]) + '\n')
    CornerFile.close() 

print('Done!')

plt.subplot(2,2,1)
plt.title("Billede")
plt.imshow(img, cmap='gray')

plt.subplot(2,2,2)
plt.title("Ixx")
plt.imshow(Ixx, cmap='gray')

plt.subplot(2,2,3)
plt.title("Iyy")
plt.imshow(Iyy, cmap='gray')

plt.subplot(2,2,4)
plt.title("Nul")
plt.imshow(nul, cmap='gray')

plt.show()
