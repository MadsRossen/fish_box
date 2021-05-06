import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

file = cv.imread('/Users/madsrossen/Documents/4. Semester/Projekt/code/images/fP0Im.jpg', 0)

col, row = np.shape(file)
print('Row: ', row)
print('Col: ', col)

sobel_image = np.zeros(shape=(col, row))
Iyy = np.zeros(shape=(col, row))
Ixx = np.zeros(shape=(col, row))

# Define sobel oprator matrix
gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


for y in range(row - 2):
    for x in range(col - 2):

        Iy = np.sum(np.multiply(gy, file[x:x+3, y:y+3])) # multiply two 3x3 matrix together in y
        Ix = np.sum(np.multiply(gx, file[x:x+3, y:y+3])) # multiply two 3x3 matrix together in x

        Iyy[x, y] = Iy
        Ixx[x, y] = Ix

        sobel_image[x, y] = np.sqrt(Ix**2 + Iy**2)


plt.subplot(2,2,1)
plt.title("Billede")
plt.imshow(file, cmap='gray')

plt.subplot(2,2,2)
plt.title("Ixx")
plt.imshow(Ixx, cmap='gray')

plt.subplot(2,2,3)
plt.title("Iyy")
plt.imshow(Iyy, cmap='gray')

plt.subplot(2,2,4)
plt.title("IxIy")
plt.imshow(sobel_image, cmap='gray')

plt.show()