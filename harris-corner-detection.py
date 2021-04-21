import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

<<<<<<< HEAD
filename = cv.imread("/Users/madsrossen/Documents/4. Semester/Projekt/code/images/MedStøjCorner.png", 0)
img = cv.cvtColor(filename, cv.COLOR_GRAY2RGB)

# Parameters
windowSize = 3
k = 0.06
threshold = 159952

# Til test af billeder set til true eller false:
test = True
=======
filename = cv.imread("/filelocation/", 0)
img = cv.cvtColor(filename, cv.COLOR_GRAY2RGB)

# Parameters
windowSize = 5
k = 0.04
threshold = 100000
>>>>>>> a3c8d21fa22201b1df78fe4f493470b5680a0281

offset = int(windowSize/2)

x_size = filename.shape[1] - offset
y_size = filename.shape[0] - offset

<<<<<<< HEAD
# mean blur
gBlur = cv.blur(filename, (5, 5))

# Partial differentiation hvor ** = ^2
Iy, Ix = np.gradient(gBlur)
Ixx = Ix**2
Ixy = Iy*Ix
Iyy = Iy**2
=======
# Gaussian blur
gBlur = cv.GaussianBlur(filename,(windowSize,windowSize),1.4)

# Partial differentiation hvor ** = ^2
dy, dx = np.gradient(gBlur)
Ixx = dx**2
Ixy = dy*dx
Iyy = dy**2
>>>>>>> a3c8d21fa22201b1df78fe4f493470b5680a0281

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
<<<<<<< HEAD
            img[y,x] = (255,0,0)
            CornerList.append([x, y, Ix[x,y], Iy[x,y], r])


if bool(test):

    print('Creating file')

    CornerFile = open('CornersFound.txt', 'w')
    CornerFile.write('x, \t y, \t Ix, \t Iy, \t R \n')
    for i in range(len(CornerList)):
        CornerFile.write(str(CornerList[i][0]) + ' , ' + str(CornerList[i][1]) + ' , ' + str(CornerList[i][2]) + ' , ' + str(CornerList[i][3]) + ' , ' + str(CornerList[i][4]) + '\n')
    CornerFile.close() 
=======
            print("R: ", r)
            img[y,x] = (0,0,255)
>>>>>>> a3c8d21fa22201b1df78fe4f493470b5680a0281

print('Done!')

plt.subplot(2,2,1)
plt.title("Billede")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)

plt.subplot(2,2,2)
plt.title("Ixx")
plt.imshow(Ixx, cmap='gray', vmin=0, vmax=255)

plt.subplot(2,2,3)
plt.title("Iyy")
plt.imshow(Iyy, cmap='gray', vmin=0, vmax=255)

plt.subplot(2,2,4)
plt.title("Ixy")
plt.imshow(Ixy, cmap='gray', vmin=0, vmax=255)

plt.show()
