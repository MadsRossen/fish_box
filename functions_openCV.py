import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

def GrassFire(img):
    """ Only input binary images of 0 and 255 """
    mask = copy.copy(img)

    h, w = mask.shape[:2]
    h = h-1
    w = w-1

    save_array = []
    zero_array = []
    blob_array = []
    temp_cord = []

    for y in range(h):
        for x in range(w):
            if mask.item(y, x) == 0 and x <= h:
                zero_array.append(mask.item(y, x))
            elif mask.item(y, x) == 0 and x >= w:
                zero_array.append(mask.item(y, x))

    # Looping if x == 1, and some pixels has to be burned
            while mask.item(y, x) > 0 or len(save_array) > 0:
                mask.itemset((y, x), 0)
                temp_cord.append([y, x])

                if mask.item(y - 1, x) > 0:
                    if [y - 1, x] not in save_array:
                        save_array.append([y - 1, x])

                if mask.item(y, x - 1) > 0:
                    if [y, x - 1] not in save_array:
                        save_array.append([y, x - 1])

                if mask.item(y + 1, x) > 0:
                    if [y + 1, x] not in save_array:
                        save_array.append([y + 1, x])

                if mask.item(y, x + 1) > 0:
                    if [y, x + 1] not in save_array:
                        save_array.append([y, x + 1])

                if len(save_array) > 0:
                    y, x = save_array.pop()

                else:
                    blob_array.append(temp_cord)
                    temp_cord = []
                    break

    return blob_array

def doClaheLAB2(null):
    global val2, kernel
    val2 = cv2.getTrackbarPos('tilesize', 'ResultHLS')
    if val2 <= 1:
        val2 = 2
    kernel = (val2,val2)
    res = claheHSL(img, val1/10,kernel)
    cv2.imshow("ResultHLS", res)
    plt.hist(res.ravel(), 256, [0, 256]);
    plt.hist(res.ravel(), 256, [0, 256]);
    plt.show()
    cv2.waitKey(0)

def harrisCorner(checkeboardImg, test=False, CornerCor=True):
    '''

    :param CornerCor: Få corner coordinate set til true eller false:
    :param checkeboardImg:
    :param test: Til test af cornerbilleder set til true eller false:
    :return:
    '''
    img = cv2.cvtColor(checkeboardImg, cv2.COLOR_GRAY2RGB)

    # Parameters
    windowSize = 3
    k = 0.06  # Parameter between 0.04 - 0.06
    threshold = 10000

    CheckPoints = 54

    offset = int(windowSize / 2)

    x_size = checkeboardImg.shape[1] - offset
    y_size = checkeboardImg.shape[0] - offset

    nul = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # mean blur
    blur = cv2.blur(checkeboardImg, (5, 5))

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

            # Create window
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
        Objects = GrassFire(nul)

        # Sort the list by the mass of objects
        print("Number of objects: ", len(Objects))
        ObjectsH = sorted(Objects, key=len, reverse=True)

        CornerList = []

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

            # Draw a circle around the center
            cv2.circle(img, (xbb, ybb), 30, (255, 0, 0), thickness=2, lineType=cv2.LINE_8)

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
            CornerFile.write(str(CornerCoordinate[i][0]) + ' , ' + str(CornerCoordinate[i][1]) + ' , ' + str(
                CornerCoordinate[i][2]) + ' , ' + str(CornerCoordinate[i][3]) + ' , ' + str(
                CornerCoordinate[i][4]) + '\n')
        CornerFile.close()

    print('Done!')

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

    return CornerCoordinate

def normHistEqualizeHLS(img):
    fiskHLS1 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannel = fiskHLS1[:,:,1]
    HistEqualize = cv2.equalizeHist(LChannel)
    fiskHLS1[:,:,1] = HistEqualize
    fiskNomrHistEq = cv2.cvtColor(fiskHLS1,cv2.COLOR_HLS2BGR)
    return fiskNomrHistEq


def claheHSL(img,clipLimit,tileGridSize):
    fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannelHLS = fiskHLS2[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelHLS)
    fiskHLS2[:, :, 1] = claheLchannel1
    fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)
    return fiskClahe


def claheLAB(img,clipLimit,tileGridSize):
    fiskLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    LChannelLAB = fiskLAB[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelLAB)
    fiskLAB[:, :, 0] = claheLchannel1
    fiskClaheLAB = cv2.cvtColor(fiskLAB, cv2.COLOR_LAB2BGR)
    return fiskClaheLAB


def limitLchannel(img, limit):
    max = 0
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    height, width, channels = imgHLS.shape
    # Creating new empty image
    newLchannel = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if imgHLS.item(y, x, 1) >= limit:
                newLchannel.itemset((y, x), limit)
                current = imgHLS.item(y, x, 1)
                if current > max:
                    max = current
            else:
                newLchannel.itemset((y, x), imgHLS.item(y, x, 1))
    imgHLS[:, :, 1] = newLchannel
    imgHLS = cv2.cvtColor(imgHLS, cv2.COLOR_HLS2BGR)
    print(max)
    return imgHLS


def crop(img,y,x,height,width):
    ROI = img[y:y + height, x:x + width]
    return ROI


def doClaheLAB1(null):
    global val1
    val1 = cv2.getTrackbarPos('cliplimit', 'ResultHLS')
    res = claheHSL(img, val1/10,kernel)
    cv2.imshow("ResultHLS", res)
    plt.hist(res.ravel(), 256, [0, 256]);
    plt.hist(res.ravel(), 256, [0, 256]);
    plt.show()


def resizeImg(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # resize image
    return resized

def meanEdgeRGB(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,imgGrayBin = cv2.threshold(imgGray,0, 255,cv2.THRESH_BINARY)

    kernel = np.ones((4, 4), np.uint8)
    erosionBefore = cv2.erode(imgGrayBin, kernel)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(erosionBefore, kernel)
    outline = cv2.subtract(erosionBefore, erosion)

    res = cv2.bitwise_and(img,img,mask = outline)
    cv2.imshow('res', res)
    cv2.waitKey(0)

    blue = 0
    green = 0
    red = 0

    bellyBlue = 0
    bellyGreen = 0
    bellyRed = 0

    backBlue = 0
    backGreen = 0
    backRed = 0

    height, width, channel = res.shape

    i = 0; bi = 0; bki = 0
    u = 0; bu = 0; bku = 0
    k = 0; bk = 0; bkk = 0

    for chan in range(channel):
        for y in range(height):
            for x in range(width):
                if res.item(y,x,0) > 0:
                    blue = blue + res.item(y,x,0)
                    i = i + 1
                    if y >= 130:
                        bellyBlue = bellyBlue + res.item(y,x,0)
                        bi = bi + 1
                    if y < 130:
                        backBlue = backBlue + res.item(y, x, 0)
                        bki = bki + 1

                if res.item(y,x,1) > 0:
                    green = green + res.item(y,x,1)
                    u = u + 1
                    if y >= 130:
                        bellyGreen = bellyGreen + res.item(y,x,1)
                        bu = bu + 1
                    if y < 130:
                        backGreen = backGreen + res.item(y, x, 1)
                        bku = bku + 1

                if res.item(y,x,2) > 0:
                    red = red + res.item(y,x,2)
                    k = k + 1
                    if y >= 130:
                        bellyRed = bellyRed + res.item(y, x, 2)
                        bk = bk + 1
                    if y < 130:
                        backRed = backRed + res.item(y, x, 2)
                        bkk = bkk + 1

    meanBlue = blue/i
    meanGreen = green/u
    meanRed = red/k

    print('mean blue: ',meanBlue)
    print('mean green: ',meanGreen)
    print(' ')

    meanBlueBelly = bellyBlue / bi
    meanGreenBelly = bellyGreen / bu
    meanRedBelly = bellyRed / bk

    print('mean blue belly: ', meanBlueBelly)
    print('mean green belly: ', meanGreenBelly)
    print('mean red belly: ', meanRedBelly)
    print(' ')

    meanBlueBack = backBlue / bki
    meanGreenBack = backGreen / bku
    meanRedBack = backRed / bkk

    print('mean blue back: ', meanBlueBack)
    print('mean green back: ', meanGreenBack)
    print('mean red back: ', meanRedBack)

# SURFAlignment aligns two images. Based on https://www.youtube.com/watch?v=cA8K8dl-E6k&t=131s
def SURFalignment(img1, img2):
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(50)

    kp1, des1 = orb.detectAndCompute(img1Gray,None)
    kp2, des2 = orb.detectAndCompute(img2Gray, None)

    img1Kp = cv2.drawKeypoints(img1, kp1, None, flags=None)
    img2Kp = cv2.drawKeypoints(img2, kp2, None, flags=None)

    cv2.imshow('img1Kp', img1Kp)
    cv2.imshow('img2Kp', img2Kp)
    cv2.waitKey(0)





