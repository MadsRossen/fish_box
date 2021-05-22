import GrassFire as gf
import blur as bl
import cv2 as cv
import glob
import numpy as np
import os
import scipy.io
from matplotlib import pyplot as plt

img_dir = "checkerboard_pics"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*JPG')
files = glob.glob(data_path)
filename_Imgs = []
checkerboard_Imgs = []
for f1 in files:
    img = cv.imread(f1, 0)
    filename_Imgs.append(img)
    checkerboard_Imgs.append(cv.cvtColor(img, cv.COLOR_GRAY2RGB))

numImgs = len(checkerboard_Imgs)
print("Number of checkerboard images found:", numImgs)
print("Brew a cup of coffee or something, this is going to take some time")

# Parameters
windowSize = 3
k = 0.04  # Parameter between 0.04 - 0.06
threshold = 10  # Threshold for R value representing corner response value
checkerboard_size = [9, 6]
CheckPoints = checkerboard_size[0] * checkerboard_size[1]  # Number of checkpoints
imgcount = 0  # Count of images
enoughPointimgs = 0  # Number of images used

# List of booleans for images that are used and not used
usedimgsNum = []

# Til test af cornerbilleder set til true eller false:
test = False
# Få corner coordinate set til true eller false:
CornerCor = True

for n in range(len(checkerboard_Imgs)):
    if n == 0:
        imgspoints = np.zeros([CheckPoints, 2, numImgs])

    filename = filename_Imgs[n]
    img = checkerboard_Imgs[n]

    offset = int(windowSize / 2)

    x_size = filename.shape[1] - offset
    y_size = filename.shape[0] - offset

    nul = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # mean blur
    blur = bl.blur(filename)

    # Partial differentiation hvor ** = ^2
    Iy, Ix = np.gradient(blur)

    # Repræsentation af M matricen
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2

    CornerCoordinate = []
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
            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            # Summed af det enkelte window
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Beregner determinanten og dirgonalen(tracen) for mere info --> se Jacobian formula
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            # finder r for harris corner detection equation
            r = det - k * (trace ** 2)

            if bool(test):
                CornerCoordinate.append([x, y, Ix[y, x], Iy[y, x], r])

            if r > threshold:
                nul.itemset((y, x), 255)
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 255)
                img.itemset((y, x, 2), 0)

    # Create a list of corner coordinates
    if bool(CornerCor):
        print("Starting GrassFire . . .")
        Objects = gf.GrassFire(nul)

        # Sort the list by the mass of objects
        print("Number of objects: ", len(Objects))
        ObjectsH = sorted(Objects, key=len, reverse=True)

        # Corner list for all corner candidates
        CornerList = []

        # Corner list for sorted and best corner candidates
        CornerListSortedXY = []

        usedimgsNum.append(False)

        # If there is not enough coroner candidates found, then do not use the image or img points
        if len(ObjectsH) >= CheckPoints:
            print('enough coroner ')
            usedimgsNum[imgcount] = True
            enoughPointimgs = enoughPointimgs + 1
            # Take the 54 biggest objects and make a circle around it. 54 = number of points at the checkerboard
            for h in range(CheckPoints):
                corner = np.array(ObjectsH[h])
                y_min = min(corner[:, 0])
                y_max = max(corner[:, 0])
                x_min = min(corner[:, 1])
                x_max = max(corner[:, 1])

                # Calculate the center of mass for each object
                xbb = int((x_min + x_max) / 2)
                ybb = int((y_min + y_max) / 2)

                img.itemset((ybb, xbb, 0), 255)
                img.itemset((ybb, xbb, 1), 0)
                img.itemset((ybb, xbb, 2), 0)

                CornerList.append([ybb, xbb])

                columnIndex = 0

                # Sort 2D numpy array by 1nd Column from high to low
                CornerListX = sorted(CornerList, key=lambda x: x[0], reverse=True)

                # Sort 2D numpy array by 2nd Column from high to low, but only for 6 at a time
                for n in range(int(CheckPoints / checkerboard_size[1])):
                    start = n * checkerboard_size[1]
                    stop = n * checkerboard_size[1]
                    nextRow = CornerListX[start:stop]
                    CornerListSortedXY.append(sorted(nextRow, key=lambda x: x[1], reverse=True))

            imgspoints[:, :, n] = CornerList

            # Draw a circle around the center
            cv.circle(img, (xbb, ybb), 30, (255, 0, 0), thickness=2, lineType=cv.LINE_8)

        print('one ')

    imgcount = imgcount + 1

# Some images where not used because the corners where not found
# Therefore make a new array, where only all image points in the checkerboard where detected
print(usedimgsNum)
ar = 0
for n in range(len(usedimgsNum)):
    if n == 0 and usedimgsNum[n] == True:
        imgspoints_print = np.zeros([CheckPoints, 2, enoughPointimgs])
        imgspoints_print[:, :, ar] = imgspoints[:, :, n]
        ar = ar + 1
    elif usedimgsNum[n] == True:
        imgspoints_print[:, :, ar] = imgspoints[:, :, n]
        ar = ar + 1
    else:
        print('Delete tha first image in checkerboard_pics or replace checkerboard images')

scipy.io.savemat('imgPoints.mat', mdict={'imgspoints_print': imgspoints_print})

'''
        print('Creating cornerlist file')
        CornerFileList = open('CornerFileList', 'w')
        CornerFileList.write('x, \t y \n')
        for i in range(len(CornerList)):
            CornerFileList.write(str(CornerList[i][0]) + ' , ' + str(CornerList[i][1]) + '\n')
        CornerFileList.close()

    # Create a list of Response value with the corrosponding x and y coordinate
    if bool(test):
        print('Creating corner file')

        CornerFile = open('CornersFoundCoordniate.txt', 'w')
        CornerFile.write('x, \t y, \t Ix, \t Iy, \t R \n')
        for i in range(len(CornerCoordinate)):
            CornerFile.write(str(CornerCoordinate[i][0]) + ' , ' + str(CornerCoordinate[i][1]) + ' , ' + str(CornerCoordinate[i][2]) + ' , ' + str(CornerCoordinate[i][3]) + ' , ' + str(CornerCoordinate[i][4]) + '\n')
        CornerFile.close()

        scipy.io.savemat('imgpoint.mat', mdict={'CornerCoordinate': CornerCoordinate})
    '''

print('Done!')

print('Number of used images: ', usedimgsNum)

plt.subplot(2, 2, 1)
plt.title("Billede")
plt.imshow(img, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Ixx")
plt.imshow(Ixx, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Iyy")
plt.imshow(Iyy, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Nul")
plt.imshow(nul, cmap='gray')

plt.show()
