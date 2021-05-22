import copy
import glob
import math
import time
import warnings
from random import randint

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms

import extremeImageProcessing as eip


def GrassFire(img):
    """ Only input binary images of 0 and 255 """
    mask = copy.copy(img)

    h, w = mask.shape[:2]
    h = h - 1
    w = w - 1

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
    kernel = (val2, val2)
    res = claheHSL(img, val1 / 10, kernel)
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
    LChannel = fiskHLS1[:, :, 1]
    HistEqualize = cv2.equalizeHist(LChannel)
    fiskHLS1[:, :, 1] = HistEqualize
    fiskNomrHistEq = cv2.cvtColor(fiskHLS1, cv2.COLOR_HLS2BGR)
    return fiskNomrHistEq


def claheHSL(img, clipLimit, tileGridSize):
    fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannelHLS = fiskHLS2[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelHLS)
    fiskHLS2[:, :, 1] = claheLchannel1
    fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)
    return fiskClahe


def claheLAB(img, clipLimit, tileGridSize):
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


def doClaheLAB1(null):
    global val1
    val1 = cv2.getTrackbarPos('cliplimit', 'ResultHLS')
    res = claheHSL(img, val1 / 10, kernel)
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

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, imgGrayBin = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((4, 4), np.uint8)
    erosionBefore = cv2.erode(imgGrayBin, kernel)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(erosionBefore, kernel)
    outline = cv2.subtract(erosionBefore, erosion)

    res = cv2.bitwise_and(img, img, mask=outline)
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

    i = 0;
    bi = 0;
    bki = 0
    u = 0;
    bu = 0;
    bku = 0
    k = 0;
    bk = 0;
    bkk = 0

    for chan in range(channel):
        for y in range(height):
            for x in range(width):
                if res.item(y, x, 0) > 0:
                    blue = blue + res.item(y, x, 0)
                    i = i + 1
                    if y >= 130:
                        bellyBlue = bellyBlue + res.item(y, x, 0)
                        bi = bi + 1
                    if y < 130:
                        backBlue = backBlue + res.item(y, x, 0)
                        bki = bki + 1

                if res.item(y, x, 1) > 0:
                    green = green + res.item(y, x, 1)
                    u = u + 1
                    if y >= 130:
                        bellyGreen = bellyGreen + res.item(y, x, 1)
                        bu = bu + 1
                    if y < 130:
                        backGreen = backGreen + res.item(y, x, 1)
                        bku = bku + 1

                if res.item(y, x, 2) > 0:
                    red = red + res.item(y, x, 2)
                    k = k + 1
                    if y >= 130:
                        bellyRed = bellyRed + res.item(y, x, 2)
                        bk = bk + 1
                    if y < 130:
                        backRed = backRed + res.item(y, x, 2)
                        bkk = bkk + 1

    meanBlue = blue / i
    meanGreen = green / u
    meanRed = red / k

    print('mean blue: ', meanBlue)
    print('mean green: ', meanGreen)
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


def SURFalignment(img1, img2):
    # SURFAlignment aligns two images. Based on https://www.youtube.com/watch?v=cA8K8dl-E6k&t=131s
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(50)

    kp1, des1 = orb.detectAndCompute(img1Gray, None)
    kp2, des2 = orb.detectAndCompute(img2Gray, None)

    img1Kp = cv2.drawKeypoints(img1, kp1, None, flags=None)
    img2Kp = cv2.drawKeypoints(img2, kp2, None, flags=None)

    cv2.imshow('img1Kp', img1Kp)
    cv2.imshow('img2Kp', img2Kp)
    cv2.waitKey(0)


def smallrange_isolate_img(hsv_img):
    def nothing(x):
        pass

    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("lowerH1", "Adjust_Hue_Satuation_Value", 0, 179, nothing)
    cv2.createTrackbar("lowerH2", "Adjust_Hue_Satuation_Value", 0, 179, nothing)

    cv2.createTrackbar("lowerv1", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("lowerv2", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    # while loop to adjust the HSV detection in the image.
    h, w, ch = hsv_img.shape[:3]

    while True:

        lowerH1 = cv2.getTrackbarPos("lowerH1", "Adjust_Hue_Satuation_Value")
        lowerH2 = cv2.getTrackbarPos("lowerH2", "Adjust_Hue_Satuation_Value")

        lowerv1 = cv2.getTrackbarPos("lowerv1", "Adjust_Hue_Satuation_Value")
        lowerv2 = cv2.getTrackbarPos("lowerv2", "Adjust_Hue_Satuation_Value")

        segmentedImg = np.zeros((h, w), np.uint8)
        # We start segmenting
        for y in range(h):
            for x in range(w):
                H = hsv_img.item(y, x, 0)
                S = hsv_img.item(y, x, 1)
                V = hsv_img.item(y, x, 2)
                # If Hue lies in the lowerHueRange(Blue hue range) we want to segment it out
                if lowerH1 > H > lowerH2:
                    segmentedImg.itemset((y, x), 0)
                # If Hue lies in the lowerValRange(black value range) we want to segment it out
                elif lowerv1 > V > lowerv2:
                    segmentedImg.itemset((y, x), 0)
                else:
                    segmentedImg.itemset((y, x), 255)
        cv2.imshow("segmentedimg", segmentedImg)

        key = cv2.waitKey(1)

        if key == 27:
            break
        # plt.imshow(segmentedImg, cmap="gray")
        # plt.show()


def isolate_img(resized_input_image, hsv_image):
    # hsv_image = cv2.cvtColor(resized_input_image, cv2.COLOR_BGR2HSV)

    def nothing(x):
        pass

    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("lowerH", "Adjust_Hue_Satuation_Value", 0, 179, nothing)
    cv2.createTrackbar("lowerS", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("lowerV", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    cv2.createTrackbar("upperH", "Adjust_Hue_Satuation_Value", 0, 179, nothing)
    cv2.createTrackbar("upperS", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("upperV", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    # while loop to adjust the HSV detection in the image.

    while True:

        lowerH = cv2.getTrackbarPos("lowerH", "Adjust_Hue_Satuation_Value")
        lowerS = cv2.getTrackbarPos("lowerS", "Adjust_Hue_Satuation_Value")
        lowerV = cv2.getTrackbarPos("lowerV", "Adjust_Hue_Satuation_Value")

        upperH = cv2.getTrackbarPos("upperH", "Adjust_Hue_Satuation_Value")
        upperS = cv2.getTrackbarPos("upperS", "Adjust_Hue_Satuation_Value")
        upperV = cv2.getTrackbarPos("upperV", "Adjust_Hue_Satuation_Value")

        lowerRange_blue = np.array([lowerH, lowerS, lowerV])
        upperRange_blue = np.array([upperH, upperS, upperV])

        mask = cv2.inRange(hsv_image, lowerRange_blue, upperRange_blue)

        res = cv2.bitwise_and(resized_input_image, resized_input_image, mask=mask)

        cv2.imshow("res", res)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1)

        if key == 27:
            print("Lower Hue:", lowerH)
            print("Lower Satuation:", lowerS)
            print("Lower value:", lowerV)

            print("Higher Hue:", upperH)
            print("Higher Satuation:", upperS)
            print("Higher value:", upperV)

            break
    return mask


def creating_mask_input_hsv(img, hsv_image):
    def nothing(x):
        pass

    cv2.createTrackbar("kernel1", "Canny_detection", 5, 50, nothing)
    cv2.createTrackbar("kernel2", "Canny_detection", 5, 50, nothing)

    while True:

        k1 = cv2.getTrackbarPos("kernel1", "Canny_detection")
        k2 = cv2.getTrackbarPos("kernel2", "Canny_detection")

        lowerRange_blue = np.array([0, 0, 40])
        upperRange_blue = np.array([183, 100, 255])

        mask = cv2.inRange(hsv_image, lowerRange_blue, upperRange_blue)

        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((11, 11), np.uint8)

        close1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        open2 = cv2.morphologyEx(close1, cv2.MORPH_CLOSE, kernel2)

        res = cv2.bitwise_and(img, img, mask=open2)
        cv2.imshow("damagemask", res)
        cv2.imshow("open", close1)
        cv2.imshow("close", open2)

        key = cv2.waitKey(1)

        if key == 27:
            break

    return mask


def canny_edge_detection(img, mask):
    def nothing(x):
        pass

    cv2.namedWindow("Canny_detection")
    cv2.createTrackbar("c1", "Canny_detection", 0, 200, nothing)
    cv2.createTrackbar("c2", "Canny_detection", 0, 500, nothing)
    # cv2.createTrackbar("kernel1", "Canny_detection", 0, 50, nothing)
    # cv2.createTrackbar("kernel2", "Canny_detection", 0, 50, nothing)

    while True:

        c1 = cv2.getTrackbarPos("c1", "Canny_detection")
        c2 = cv2.getTrackbarPos("c2", "Canny_detection")
        # k1 = cv2.getTrackbarPos("kernel1", "Canny_detection")
        # k2 = cv2.getTrackbarPos("kernel2", "Canny_detection")

        edges = cv2.Canny(mask, c1, c2)

        cv2.imshow("edges", edges)
        plt.imshow(edges, cmap="gray")
        plt.show()

        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)

        open1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)
        close1 = cv2.morphologyEx(open1, cv2.MORPH_OPEN, kernel2)
        close2 = cv2.morphologyEx(close1, cv2.MORPH_CLOSE, kernel1)

        res = cv2.bitwise_and(img, img, mask=close2)

        plt.imshow(close1, cmap="gray")
        plt.show()
        plt.imshow(close2, cmap="gray")
        plt.show()
        cv2.imshow("res", res)

        key = cv2.waitKey(1)

        if key == 27:
            break

    return res


def grayScaling(img):
    width, height = img.shape[:2]

    new_img = np.zeros([width, height, 3])

    for i in range(width):
        for j in range(height):
            list = [float(img[i][j][0]), float(img[i][j][1]), float(img[i][j][2])]
            avg = float(((list[0] + list[1] + list[2]) / 3) / 255)
            new_img[i][j][0] = avg
            new_img[i][j][1] = avg
            new_img[i][j][2] = avg

    cv2.imshow("newimagegray", new_img)

    cv2.waitKey(0)


def grayScaling8bit(img):
    width, height = img.shape[:2]

    new_img8bit = np.zeros([width, height, 1], dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            list = [float(img[i][j][0]), float(img[i][j][1]), float(img[i][j][2])]
            avg = float(((list[0] + list[1] + list[2]) / 3))
            new_img8bit[i][j][0] = avg

    cv2.imshow("newimagegray8bit", new_img8bit)

    cv2.waitKey(0)

    return new_img8bit


def convert_RGB_to_HSV(img):
    width, height, channel = img.shape

    B, G, R = img[:, :, 0] / 255, img[:, :, 1] / 255, img[:, :, 2] / 255

    hsv_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(width):
        for j in range(height):

            # Defining Hue
            h, s, v = 0.0, 0.0, 0.0
            r, g, b = R[i][j], G[i][j], B[i][j]

            max_rgb, min_rgb = max(r, g, b), min(r, g, b)
            dif_rgb = (max_rgb - min_rgb)

            if r == g == b:
                h = 0
            elif max_rgb == r:
                h = ((60 * (g - b)) / dif_rgb)
            elif max_rgb == g:
                h = (((60 * (b - r)) / dif_rgb) + 120)
            elif max_rgb == b:
                h = (((60 * (r - g)) / dif_rgb) + 240)
            if h < 0:
                h = h + 360

            # Defining Satuation
            if max_rgb == 0:
                s = 0
            else:
                s = ((max_rgb - min_rgb) / max_rgb)
            # Defining Value

            v = max_rgb
            # print(h, s, v)
            hsv_img[i][j][0], hsv_img[i][j][1], hsv_img[i][j][2] = h / 2, s * 255, v * 255

    return hsv_img


def erosion(img):
    k1 = 5
    k2 = 5
    c1 = (k1 - 1)
    c2 = (k2 - 1)

    width, height = img.shape

    # structuring_element = np.array([[1, 1, 1],
    # [1, 1, 1],
    # [1, 1, 1]])

    imgErode = np.zeros((width, height), dtype=img.dtype)

    kernel = np.ones(k1, k2)

    for i in range(c1, width - c1):
        for j in range(c2, height - c2):
            temp = img[i - c1:i + c1 + 1][j - c2:j + c2 + 1]
            product = temp * kernel
            imgErode[i][j] = np.min(product)

    cv2.imshow("erodeIMG", imgErode)

    cv2.waitKey(0)


def grassfire_transform(mask, img):
    """
    Apply the grassfire transform to a binary mask array.
    """
    # imgGray = grayScaling8bit(img)

    h, w = mask.shape
    # Use uint32 to avoid overflow
    grassfire = np.zeros_like(mask, dtype=np.uint8)

    # 1st pass
    # Left to right, top to bottom
    for x in range(w):
        for y in range(h):
            if mask[y, x] != 0:  # Pixel in contour
                north = 0 if y == 0 else grassfire[y - 1, x]
                west = 0 if x == 0 else grassfire[y, x - 1]
                if x == 3 and y == 3:
                    print(north, west)
                grassfire[y, x] = 1 + min(west, north)

    # 2nd pass
    # Right to left, bottom to top
    for x in range(w - 1, -1, -1):
        for y in range(h - 1, -1, -1):
            if mask[y, x] != 0:  # Pixel in contour
                south = 0 if y == (h - 1) else grassfire[y + 1, x]
                east = 0 if x == (w - 1) else grassfire[y, x + 1]
                grassfire[y, x] = min(grassfire[y, x],
                                      1 + min(south, east))

    cv2.imshow("grasfire", grassfire)
    cv2.waitKey(0)
    return grassfire


def grasfire_algorithm(mask):
    h, w = mask.shape[:2]
    h = h - 1
    w = w - 1
    grassfire = np.zeros_like(mask, dtype=np.uint8)
    save_array = []
    zero_array = []
    pop_array = []
    for y in range(h):
        for x in range(w):
            # x = x + 1
            # y = y + 1
            # try:
            # print(mask[y, x])
            # except:
            # print(y, x)
            if mask[y][x] == 0 and x <= h:
                zero_array.append([y, x])
                zero_array.append(mask[y][x])
            elif mask[y][x] == 0 and x >= w:
                zero_array.append([y, x])
                zero_array.append(mask[y][x])

            # Looping if x == 1, and some pixels has to be burned
            while mask[y][x] == 255:
                save_array.append([y, x])
                if mask[y][x + 1] == 255:
                    save_array.append(mask[y][x + 1])
                    x = x + 1
                    break
                if mask[y][x - 1] == 255:
                    save_array.append(mask[y][x - 1])
                    x = x - 1
                    break
                if mask[y + 1][x] == 255:
                    save_array.append(mask[y + 1][x])
                    y = y + 1
                    break
                if mask[y - 1][x] == 255:
                    save_array.append(mask[y - 1][x])
                    y = y - 1
                break
            while len(save_array) > 0:
                mask[y][x] = save_array.pop()
                y, x = save_array.pop()
                print("yx", y, x)

    print("savearray", save_array)
    print("zeroarray", zero_array)
    cv2.imshow("grasfire", grassfire)
    cv2.waitKey(0)

    return grassfire


def grassfire_v2(mask):
    h, w = mask.shape[:2]
    h = h - 1
    w = w - 1
    grassfire = np.zeros_like(mask, dtype=np.uint8)
    save_array = []
    zero_array = []
    blob_array = []
    temp_cord = []

    for y in range(h):
        for x in range(w):
            if mask[y][x] == 0 and x <= h:
                zero_array.append(mask[y][x])
            elif mask[y][x] == 0 and x >= w:
                zero_array.append(mask[y][x])

            # Looping if x == 1, and some pixels has to be burned
            while mask[y][x] > 0 or len(save_array) > 0:
                mask[y][x] = 0
                temp_cord.append([y, x])
                if mask[y - 1][x] > 0:
                    if [y - 1, x] not in save_array:
                        save_array.append([y - 1, x])
                if mask[y][x - 1] > 0:
                    if [y, x - 1] not in save_array:
                        save_array.append([y, x - 1])
                if mask[y + 1][x] > 0:
                    if [y + 1, x] not in save_array:
                        save_array.append([y + 1, x])
                if mask[y][x + 1] > 0:
                    if [y, x + 1] not in save_array:
                        save_array.append([y, x + 1])
                if len(save_array) > 0:
                    y, x = save_array.pop()

                else:
                    print("Burn is done")
                    blob_array.append(temp_cord)
                    temp_cord = []
                    break
    maskColor = np.zeros((h, w, 3), np.uint8)
    for blob in range(len(blob_array)):
        B, G, R = randint(0, 255), randint(0, 255), randint(0, 255)
        for cord in blob_array[blob]:
            y, x = cord
            maskColor[y][x][0] = B
            maskColor[y][x][1] = G
            maskColor[y][x][2] = R
    cv2.imshow("grasfire", maskColor)
    cv2.waitKey(0)


def replaceHighlights(main_img, spec_img, limit):
    """
    This functions replaces the highlights from a main picture with the pixels from a specular image pixels.

    :param main_img: The image of which will get the pixels replaced with the specular image
    :param spec_img: The image of which will be used to replace the pixels of the main image
    :param limit: The limits of a pixel value before it is classified as a specular highlight
    :return: The image that has the highlights replaced
    """

    # Create copy
    img_main_cop = np.copy(main_img)
    img_spec_cop = np.zeros((spec_img.shape[0], spec_img.shape[1], 3), np.uint8)

    # Isolate the areas where the color is white
    main_img_spec = np.where((img_main_cop[:, :, 0] >= limit) & (img_main_cop[:, :, 1] >= limit) &
                             (img_main_cop[:, :, 2] >= limit))

    img_spec_cop[main_img_spec] = spec_img[main_img_spec]
    img_main_cop[main_img_spec] = (0, 0, 0)

    img_main_cop[main_img_spec] = img_spec_cop[main_img_spec]

    # Different methods, find out what works best later
    match = match_histograms(img_spec_cop, img_main_cop, multichannel=True)
    match2 = match_histograms(spec_img, img_main_cop, multichannel=True)

    # Replace pixels, replace the matches to use different methods
    img_main_cop[main_img_spec] = match2[main_img_spec]

    cv2.imshow("match", match2)
    cv2.imshow("spec_img", spec_img)
    cv2.imshow("final", img_main_cop)
    cv2.imshow("Spec", img_spec_cop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_main_cop


def equalizeColoredImage(img):
    '''
    This function returns the equalized image of a color image using YUV.

    :param img: The image to equalize
    :return: Equalized image
    '''
    # Turn into YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def find_blood_damage(img):
    '''
    Returns the mask of the found blood damage

    :param img: Image to check for blood spots
    :return: The mask of blood damage
    '''

    # Bounds for the red in the wound, use gimp to find the right values
    lower_red = np.array([0, 90, 90])
    upper_red = np.array([7, 100, 100])

    # Convert to hsv, check for red and return the found mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    return mask


def createDict():
    '''
    Custom dictionary for storing length and angle values for each contour.
    :return: Dictionary with length and angle array.
    '''
    dict = {
        "length": [],
        "angle": []
    }
    return dict


def cropToROI(orig_img, contours):
    '''
    Finds the minimum (x,y) and maximum (x,y) coordinates for each contour and then crops the original image to fit the contours.
    It also computes the center of mass of each contour.

    :param orig_img: original image that will be cropped.
    :param contours: (x,y) coordinates for the contours in the orig_img.
    :return: (xcm, ycm) array with the (x,y) coordinates for the contours center of mass. crop_img: array of the cropped images (one image for each contour)
    '''

    height = []
    width = []
    for n in orig_img:
        height.append(n.shape[0])
        width.append(n.shape[1])

    xcm = []
    ycm = []
    crop_img = []
    for nr in range(len(contours)):
        ymax, ymin = 0, height[nr]
        xmax, xmin = 0, width[nr]
        for point in range(len(contours[nr])):
            if contours[nr][point][0][0] > xmax:
                xmax = contours[nr][point][0][0]
            if contours[nr][point][0][0] < xmin:
                xmin = contours[nr][point][0][0]
            if contours[nr][point][0][1] > ymax:
                ymax = contours[nr][point][0][1]
            if contours[nr][point][0][1] < ymin:
                ymin = contours[nr][point][0][1]
        # Computing the approximate center of mass:
        # From Thomas B. Moeslund "Introduction to Video and Image Processing"
        # (Page 109 Eq: 7.3 and 7.4)
        xcm.append(int((xmin + xmax) / 2))
        ycm.append(int((ymin + ymax) / 2))
        # crop_img.append(orig_img[nr][ymin-50:ymax+50, xmin-50:xmax+50])
        # rsize = 6
        # crop_img[nr] = cv2.resize(crop_img[nr], (rsize, rsize))
    return xcm, ycm


def find_biggest_contour(cnt):
    """
    Returns the biggest contour in a list of contours.

    :param cnt: A list of contours
    :return: The biggest contour inside the list
    """
    biggest_area = 0
    biggest_cnt = None
    for n in cnt:
        if cv2.contourArea(n) > biggest_area:
            biggest_cnt = n
        else:
            continue

    return biggest_cnt


def find_contours(images):
    """
    Returns the biggest contour for a list of images.

    :param images: A list of images to find contours inside
    :return: A list with the biggest contour for each image
    """
    # Bounds for the fish
    lower = np.array([0, 16, 16])
    upper = np.array([94, 255, 255])

    def nothing(x):
        pass

    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("kernel open", "Adjust_Hue_Satuation_Value", 1, 20, nothing)
    cv2.createTrackbar("kernel close", "Adjust_Hue_Satuation_Value", 1, 20, nothing)

    contours = []
    for n in images:
        while True:
            kernel_val_open_val = cv2.getTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value")
            kernel_val_close_val = cv2.getTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value")

            # Find a better method than this
            if kernel_val_open_val == 0:
                kernel_val_open_val = 1
            if kernel_val_close_val == 0:
                kernel_val_close_val = 1

            # Make kernels for each morph type
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_open_val, kernel_val_open_val))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_close_val, kernel_val_close_val))

            # Convert to hsv, check for red and return the found mask
            hsv = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphology
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)

            # To see how much of the fish we are keeping
            if closing is not None:
                res = cv2.bitwise_and(n, n, mask=closing)

            cv2.imshow("Adjust_Hue_Satuation_Value", closing)
            cv2.imshow("Res", res)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # Find contours
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Getting the biggest contour which is always the fish
        if contours is not None:
            contours.append(find_biggest_contour(contours))
        else:
            print("Can't find contour")

    return contours


def rotateImages(img, rotate_img, xcm, ycm, contours):
    """
    A function that can rotate a set of images, so the longest part has an angle of zero relative to the
    positive x-axis. This is done by line tracing each point on the contours relative to
    the center of mass of the contours. The angle for each line relative to the positive
    x-axis is then computed, and the angle for the longest length is then used to rotate the image.

    NOTE: The approximation method for the findContour() function should be CHAIN_APPROX_NONE, otherwise if
    CHAIN_APPROX_SIMPLE is used the results might vary negatively.

    Optional: It is possible to draw the line tracing in the img (line 164 should then be included).
    It is also possible to plot the distribution of the contour coordinates length
    relative to the angle (line 174-177 should then be included).

    :param img: Image that you want the line tracing to be drawn upon.
    :param rotate_img: Images that needs to be rotated.
    :param xcm: x-coordinates for the center of mass of the contours
    :param ycm: y-coordinates for the center of mass of the contours
    :param contours: Contours from the images that needs to be rotated.
    :return: The rotated images.
    """

    # Variable where the length and angle will be stored.
    data = []
    # Variable to store the rotated images.
    img_output = []
    for nr in range(len(contours)):
        maxLength = 0
        angleMaxLength = 0
        data.append(createDict())
        for point in range(len(contours[nr])):
            # Compute x and y coordinate relative to the contours center of mass.
            x_delta = contours[nr][point][0][0] - xcm[nr]
            y_delta = contours[nr][point][0][1] - ycm[nr]
            # Compute the length and angle of each coordinate in the contours.
            data[nr]["length"].append(math.sqrt(pow(x_delta, 2) + pow(y_delta, 2)))
            data[nr]["angle"].append(math.atan2(y_delta, x_delta) * (180 / math.pi))
            # Finding the longest length and at what angle.
            if data[nr]["length"][point] > maxLength:
                maxLength = data[nr]["length"][point]
                angleMaxLength = data[nr]["angle"][point]
            # Draw the line tracing on the contours and point (optional)
            cv2.line(rotate_img[nr], (xcm[nr], ycm[nr]), (contours[nr][point][0][0], contours[nr][point][0][1]),
                     (255, 0, 0), 1)
        # Rotating the images so the longest part of the resistor has an angle of 0 relative to the positive x-axis.
        '''if angleMaxLength != 0:
            (height, width) = rotate_img[nr].shape[:2]
            (cX, cY) = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D((cX, cY), angleMaxLength, 1.0)
            img_output.append(cv2.warpAffine(rotate_img[nr], M, (width, height), borderValue=(0, 128, 128)))
        else:
            img_output.append(rotate_img[nr])'''
        # Show COF contour
        cv2.circle(rotate_img[nr], (xcm[nr], ycm[nr]), radius=4, color=(0, 0, 255), thickness=-1)

        # Plot the contour coordinates length relative to the angle (optional):
        plt.subplot(int("1" + str(len(contours)) + str(nr + 1)))
        plt.bar(data[nr]["angle"], data[nr]["length"])
        plt.axis([-180, 180, 0, 500])
    plt.show()
    return rotate_img


def checkerboard_calibrate(images_distort, images_checkerboard):
    """
    Undistorts images by a checkerboard calibration.

    :param images_distort: The images the needs to be undistorted
    :param images_checkerboard: The images of the checkerboard to calibrate by
    :return: If it succeeds, returns the undistorted images, if it fails, returns the distorted images with a warning
    """

    print("Started calibrating...")

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for img in images_checkerboard:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        else:
            warnings.warn("No ret! This might lead to a crash.")
    # The function doesn't always find the checkerboard, therefore we have to try, and if not, pass exception
    try:
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Go through all the images and undistort them
        img_undst = []
        for n in images_distort:
            # Get image shape
            h, w = n.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # undistorted
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
            dst = cv2.remap(n, mapx, mapy, cv2.INTER_LINEAR)

            # crop the image back to original shape
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            img_undst.append(dst)
            cv2.imshow('calibresult.png', dst)
            cv2.waitKey(0)

        print("Done calibrating")

        return img_undst
    except ValueError:
        # If the calibration fails, inform us and tell us the error
        warnings.warn(f"Could not calibrate camera. Check images. Error {ValueError}")
        mean_error = 0
        tot_error = 0

        # Show the total error
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        print("total error: ", mean_error / len(objpoints))

        # If the function fails, return the input arguments
        return images_distort


def showCompariHist(img1, img2, stringImg1, stringImg2, mode):
    '''
    Compares histogram of R+B+G histogram

    :param img1: Image 1
    :param img2: Image 2
    :param stringImg1: String that is shown to say that this is histogram for Image 1
    :param stringImg2: String that is shown to say that this is histogram for Image 2
    :param mode: "BGR" or "Grey". Defines whether the function will calculate histogram on a BGR image or greyscale image
    :return:
    '''
    histSize = 256
    if mode == "BGR":
        color = ('b', 'g', 'r')
        img1histr = 0
        img2histr = 0
        h1, w1, chan1 = img1.shape
        h2, w2, chan2 = img2.shape

        # Calculate histograms
        for i, col in enumerate(color):
            img1histrchannel = cv2.calcHist([img1], [i], None, [histSize], [0, histSize])
            img1histr = img1histr + img1histrchannel

            img2histrchannel = cv2.calcHist([img2], [i], None, [histSize], [0, histSize])
            img2histr = img2histr + img2histrchannel

        # Normalize histograms
        img1histrNorm = img1histr / (h1 * w1 * chan1)
        img2histrNorm = img2histr / (h2 * w2 * chan2)
        print(sum(img2histrNorm), sum(img1histrNorm))
        plt.plot(img1histrNorm, color='orange'), plt.plot(img2histrNorm, color='blue')
    else:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img1histr = cv2.calcHist([img1], [0], None, [histSize], [0, histSize])
        img2histr = cv2.calcHist([img2], [0], None, [histSize], [0, histSize])
        sum1 = 0
        sum2 = 0
        for i in range(len(img1histr)):
            sum1 = sum1 + (i * img1histr[i])
            sum2 = sum2 + (i * img2histr[i])

        # Normalize histograms
        img1histrNorm = (img1histr / (h1 * w1)) * 100
        img2histrNorm = (img2histr / (h2 * w2)) * 100
        print(sum(img1histrNorm), sum(img2histrNorm))
        plt.plot(img1histrNorm, color='orange'), plt.plot(img2histrNorm, color='blue')
        print("mean value for img1 = ", sum1 / (h1 * w1))
        print("mean value for img2 = ", sum2 / (h2 * w2))

    # Show histograms
    plt.xlim([0, 256])
    plt.text(50, 0.02, stringImg1, color='orange')
    plt.text(150, 0.02, stringImg2, fontsize='medium', color='blue')
    plt.xlabel('Intensity (unweighted)')
    plt.ylabel('Number of pixels in percentage')
    plt.show()


def calcHueChanHist(imageChannel):
    '''
    Calculates histogram for only one channel
    :param images:
    :return:
    '''

    histSize = 0
    imgHistogram = (cv2.calcHist([imageChannel], [0], None, [180], [0, 180]))

    h1, w1 = imageChannel.shape

    # Normalize histograms
    pixelsTotal1 = h1 * w1
    img1histrNorm = imgHistogram / pixelsTotal1
    plt.plot(img1histrNorm, color='orange')

    return img1histrNorm


def calcMeanHist(images):
    '''
    Compares histogram of R+B+G histogram

    :param img1: Image 1
    :param stringImg1: String that is shown to say that this is histogram for Image 1
    :param mode: "BGR" or "Grey". Defines whether the function will calculate histogram on a BGR image or greyscale image
    :return:
    '''
    if len(images) == 0:
        return 0, 0

    sum = 0
    histSize = 256
    imgHistograms = []
    imgMeanHistograms = []
    for img in images:
        n_img = cv2.imread(img)
        grey_img = BGR2MeanGreyscale(n_img)
        # B, G, R = cv2.split(n_img)
        imgHistograms.append(cv2.calcHist([grey_img], [0], None, [histSize], [0, histSize]))
    for i in range(histSize):
        for histo in imgHistograms:
            sum = sum + histo[i]
        mean = sum / len(images)
        sum = 0
        imgMeanHistograms.append(mean)
    imgMeanHistograms = np.array(imgMeanHistograms)

    h1, w1 = n_img.shape[:2]
    sum1 = 0
    for i in range(len(imgMeanHistograms)):
        sum1 = sum1 + (i * imgMeanHistograms[i])

    # Normalize histograms
    img1histrNorm = (imgMeanHistograms / (h1 * w1)) * 100
    plt.plot(img1histrNorm, color='orange')
    mean_value = sum1 / (h1 * w1)

    return img1histrNorm, mean_value


def BGR2MeanGreyscale(img):
    """
    Function that will convert a BGR image to a mean valued greyscale image.
    :param img: BGR image that will be converted to greyscale
    :return: The converted greyscale image.
    """

    h, w, = img.shape[:2]
    greyscale_img1 = np.zeros((h, w), np.uint8)
    greyscale_img2 = np.zeros((h, w), np.uint8)
    start_time = time.time()

    for y in range(h):
        for x in range(w):
            I1 = (img.item(y, x, 0) + img.item(y, x, 1) + img.item(y, x, 2)) / 3
            greyscale_img1.itemset((y, x), I1)
    print("Execution time for optimized item/itemset function: ", "--- %s seconds ---" % (time.time() - start_time))

    """for y in range(h):
        for x in range(w):
            I2 = (int(img[y][x][0]) + int(img[y][x][1]) + int(img[y][x][2]))/3
            greyscale_img2[y][x][0] = I2
    print("Execution time for non optimized function: ", "--- %s seconds ---" % (time.time() - start_time))"""
    return greyscale_img1


def imgMinusimg(img1, img2):
    h, w = img1.shape[:2]
    greyscale_img1 = np.zeros((h, w, 1), np.uint8)
    for y in range(h):
        for x in range(w):
            greyscale_img1.itemset((y, x, 0), abs(img1.item(y, x, 0) - img2.item(y, x, 0)))

    return greyscale_img1


def undistort(inputImg):
    '''
    Undistorts images using parameters found in MATLAB calibration using 'standard'

    :param inputImg:
    :return:
    '''
    start_time = time.time()
    h, w, ch = inputImg.shape

    # Parameters found in MATLAB calibration session
    k_1 = -0.2666
    k_2 = 0.0845
    imgCenterX = 1441.64880195002
    imgCenterY = 1126.65141577051
    Fx = 1757.26695467940
    Fy = 1743.87124082603

    undistorted = np.zeros(inputImg.shape, np.uint8)

    for y in np.arange(-1, 1, 1 / h):
        for x in np.arange(-1, 1, 1 / w):
            xorig = x
            yorig = y

            r = np.sqrt(xorig ** 2 + yorig ** 2)
            output_pos_x = round(Fx * (xorig * (1 + k_1 * r ** 2 + k_2 * r ** 4)) + imgCenterX);
            output_pos_y = round(Fy * (yorig * (1 + k_1 * r ** 2 + k_2 * r ** 4)) + imgCenterY);

            input_pos_x = round(Fx * x + imgCenterX)
            input_pos_y = round(Fy * y + imgCenterY)

            if input_pos_x < w - 1 and input_pos_y < h - 1 and output_pos_x < w - 1 and output_pos_y < h - 1:
                if input_pos_x >= 0 and input_pos_y >= 0 and output_pos_x >= 0 and output_pos_y >= 0:
                    undistorted.itemset((input_pos_y, input_pos_x, 0), inputImg.item((output_pos_y, output_pos_x, 0)))
                    undistorted.itemset((input_pos_y, input_pos_x, 1), inputImg.item((output_pos_y, output_pos_x, 1)))
                    undistorted.itemset((input_pos_y, input_pos_x, 2), inputImg.item((output_pos_y, output_pos_x, 2)))
    print("Execution time for optimized item/itemset function: ", "--- %s seconds ---" % (time.time() - start_time))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(inputImg)
    ax2.imshow(undistorted)
    plt.show()


def undistortImg(distortedImg, recalibrate=False):
    '''
    Undistorts images using openCV's cv2.fisheye.calibrate function.

    :param distortedImg: The distorted image that is to be undistorted.
    :param recalibrate: set to True if recalibration is needed.
    :return: The undistorted image.
    '''

    if recalibrate == True:
        print('Calibrating camera please wait ... \n')
        CHECKERBOARD = (6, 9)  # size of checkerboard

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('calibration/checkerboard_pics/*.JPG')  # loaded images from folder in work tree
        # Run through list of images of checkerboards
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2]  # All images must share the same size
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                imgpoints.append(corners)
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        # Use the fisheye model to calibrate
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

        # Save calibration session parametres
        N_OK_array = np.array(N_OK)
        _img_shape_array = np.array(_img_shape)
        np.save('calibration/parameters_calibration_session/matrixK.npy', K)
        np.save('calibration/parameters_calibration_session/matrixD.npy', D)
        np.save('calibration/N_OK.npy', _img_shape_array)
        np.save('calibration/_img_shape.npy', _img_shape)
        print("Found " + str(N_OK_array) + " valid images for calibration")
        print("DIM = Dimension of images = " + str(_img_shape_array[::-1]))

    K = np.load('calibration/parameters_calibration_session/matrixK.npy')
    D = np.load('calibration/parameters_calibration_session/matrixD.npy')
    N_OK_array = np.load('calibration/N_OK.npy')
    _img_shape_array = np.load('calibration/_img_shape.npy')

    print("\nIntrinsic parameters")
    print("Camera matrix: K =")
    print(K)
    print("D =")
    print(D)

    img_dim = distortedImg.shape[:2][::-1]

    DIM = img_dim
    balance = 1

    scaled_K = K * img_dim[0] / DIM[0]
    scaled_K[2][2] = 1.0

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
                                                                   img_dim, np.eye(3), balance=balance)

    print('\n Undistorting image ... ')
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
                                                     new_K, img_dim, cv2.CV_16SC2)
    undist_image = cv2.remap(distortedImg, map1, map2, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

    print('\n Image has been undistorted')

    return undist_image


def save_img(img):
    out_folder_processed_images_path = "C:\\Users\\MOCst\\PycharmProjects\\Nye_filer"  # Make into Yaml parameter path
    count = 0
    if len(img) < 1:
        for n in img:
            count = count + 1
            cv2.imwrite(out_folder_processed_images_path + f"\\fish{count}.jpg", n)
    else:
        cv2.imwrite(out_folder_processed_images_path + "\\fish.jpg", img)


def replaceHighlights2(main_img, spec_img, limit):
    """
    This functions replaces the highlights from a main picture with the pixels from a specular image pixels.

    :param main_img: The image of which will get the pixels replaced with the specular image
    :param spec_img: The image of which will be used to replace the pixels of the main image
    :param limit: The limits of a pixel value before it is classified as a specular highlight
    :return: The image that has the highlights replaced
    """

    print("Replacing highlights...")

    # Create copy
    img_main_cop = np.copy(main_img)
    img_spec_cop = np.zeros((spec_img.shape[0], spec_img.shape[1], 3), np.uint8)

    # Isolate the areas where the color is white
    main_img_spec = np.where((img_main_cop[:, :, 0] >= limit) & (img_main_cop[:, :, 1] >= limit) &
                             (img_main_cop[:, :, 2] >= limit))

    img_spec_cop[main_img_spec] = spec_img[main_img_spec]
    img_main_cop[main_img_spec] = (0, 0, 0)

    img_main_cop[main_img_spec] = img_spec_cop[main_img_spec]

    # Different methods, find out what works best later
    match = match_histograms(img_spec_cop, img_main_cop, multichannel=True)
    match2 = match_histograms(spec_img, img_main_cop, multichannel=True)

    # Replace pixels, replace the matches to use different methods
    img_main_cop[main_img_spec] = match2[main_img_spec]

    cv2.imshow("match", match2)
    cv2.imshow("spec_img", spec_img)
    cv2.imshow("final", img_main_cop)
    cv2.imshow("Spec", img_spec_cop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Done replacing the highlights!")

    return img_main_cop


def claheHSL2(img, clipLimit, tileGridSize):
    fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannelHLS = fiskHLS2[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelHLS)
    fiskHLS2[:, :, 1] = claheLchannel1
    fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)

    return fiskClahe


def resizeImg2(img, scale_percent):
    """
    Resizes the image by a scaling percent.

    :param img: The image to resize
    :param scale_percent: The percent to scale by
    :return: The resized image
    """

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


def createDict2():
    '''
    Custom dictionary for storing length and angle values for each contour.
    :return: Dictionary with length and angle array.
    '''
    dict = {
        "length": [],
        "angle": []
    }
    return dict


def contour_MOC(orig_img, contours):
    '''
    Finds the minimum (x,y) and maximum (x,y) coordinates for each contour and computes the center of mass of each
    contour.

    :param orig_img: original image that will be cropped.
    :param contours: (x,y) coordinates for the contours in the orig_img.
    :return: (xcm, ycm) array with the (x,y) coordinates for the contours center of mass. crop_img: array of the cropped
    images (one image for each contour)
    '''

    print("Finding maximum and minimum coordinates for each contours and then cropping...")

    print(f"Original images length: {len(orig_img)}")
    print(f"Contours len: {len(contours)}")

    height = []
    width = []
    for n in orig_img:
        height.append(n.shape[0])
        width.append(n.shape[1])

    xcm = []
    ycm = []
    for nr in range(len(orig_img)):
        ymax, ymin = 0, height[nr]
        xmax, xmin = 0, width[nr]
        for point in range(len(contours[nr])):
            if contours[nr][point][0][0] > xmax:
                xmax = contours[nr][point][0][0]
            if contours[nr][point][0][0] < xmin:
                xmin = contours[nr][point][0][0]
            if contours[nr][point][0][1] > ymax:
                ymax = contours[nr][point][0][1]
            if contours[nr][point][0][1] < ymin:
                ymin = contours[nr][point][0][1]
        # Computing the approximate center of mass:
        # From Thomas B. Moeslund "Introduction to Video and Image Processing"
        # (Page 109 Eq: 7.3 and 7.4)
        xcm.append(int((xmin + xmax) / 2))
        ycm.append(int((ymin + ymax) / 2))

    print("Found all the contours and cropped the image!")

    return xcm, ycm


def find_biggest_contour2(cnt):
    """
    Returns the biggest contour in a list of contours.

    :param cnt: A list of contours
    :return: The biggest contour inside the list
    """
    print("Finding the biggest contours...")
    biggest_area = 0
    biggest_cnt = None
    for n in cnt:
        if cv2.contourArea(n) > biggest_area:
            biggest_cnt = n
        else:
            continue

    print("Found the biggest contours!")

    return biggest_cnt


def nothing(x):
    pass


def open_close_trackbars():
    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("kernel open", "Adjust_Hue_Satuation_Value", 2, 20, nothing)
    cv2.createTrackbar("kernel close", "Adjust_Hue_Satuation_Value", 2, 20, nothing)

    kernel_val_open_val = cv2.getTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value")
    kernel_val_close_val = cv2.getTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value")

    # Make sure it's only uneven numbers for the kernels
    if kernel_val_open_val % 2 == 0:
        cv2.setTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value", kernel_val_open_val + 1)
        kernel_val_open_val = cv2.getTrackbarPos("kernel open", "Adjust_Hue_Satuation_Value")

    if kernel_val_close_val % 2 == 0:
        cv2.setTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value", kernel_val_close_val + 1)
        kernel_val_close_val = cv2.getTrackbarPos("kernel close", "Adjust_Hue_Satuation_Value")

    return kernel_val_open_val, kernel_val_close_val


def find_contours2(masks, images, change_kernel=False, show_img=False):
    """
    Returns the biggest contour for a list of images.

    :param show_img: Weather or not to display the morphed images
    :param change_kernel: Changes weather or not to change the kernels by trackbars. If left false, it will use the
    default parameters 5 and 7 for open and close respectively
    :param masks: Masks to find contours of
    :param images: A list of images to find contours inside
    :return: A list with the biggest contour for each image
    """

    print("Finding contours...")

    old_open_val, old_closes_val = 0, 0
    contours = []
    image_n = 0
    opening = None
    closing = None
    for n in masks:
        while True:
            # If we wanna change the kernel by trackbar
            if change_kernel:
                kernel_val_open_val, kernel_val_close_val = open_close_trackbars()
            else:
                kernel_val_open_val, kernel_val_close_val = 5, 7

            # Make kernels for each morph type
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_open_val, kernel_val_open_val))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_val_close_val, kernel_val_close_val))

            # Only use opening and closing when the slider is moved instead of every frame
            if old_open_val != kernel_val_open_val or old_closes_val != kernel_val_close_val or change_kernel is False:
                opening = eip.morph_open(n, kernel_open)
                closing = eip.morph_close(opening, kernel_close)
                old_open_val = kernel_val_open_val
                old_closes_val = kernel_val_close_val

            # To see how much of the fish we are keeping
            if closing is not None:
                res = eip.bitwise_and(images[image_n], closing)
            else:
                warnings.warn("The closing or open operation is None!")

            if change_kernel:
                cv2.imshow("Adjust_Hue_Satuation_Value", closing)

            if show_img:
                cv2.imshow("Mask", n)
                cv2.imshow("Res", res)

            key = cv2.waitKey(1)
            if key == 27 or change_kernel is False:
                break

        # Find contours, implement grassfire algorithm
        contours_c, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        print(f"Contours length:{len(contours)}")

        # Getting the biggest contour which is always the fish
        if contours_c is not None:
            contours.append(find_biggest_contour(contours_c))
        else:
            print("Can't find contour")

        # Increment the image number so we have the right bitwise
        image_n = image_n + 1

    print("Found all the contours!")

    return contours


def rotateImages2(rotate_img, xcm, ycm, contours):
    """
    A function that can rotate a set of images, so the longest part has an angle of zero relative to the
    positive x-axis. This is done by line tracing each point on the contours relative to
    the center of mass of the contours. The angle for each line relative to the positive
    x-axis is then computed, and the angle for the longest length is then used to rotate the image.

    NOTE: The approximation method for the findContour() function should be CHAIN_APPROX_NONE, otherwise if
    CHAIN_APPROX_SIMPLE is used the results might vary negatively.

    Optional: It is possible to draw the line tracing in the img (line 164 should then be included).
    It is also possible to plot the distribution of the contour coordinates length
    relative to the angle (line 174-177 should then be included).

    :param rotate_img: Images that needs to be rotated.
    :param xcm: x-coordinates for the center of mass of the contours.
    :param ycm: y-coordinates for the center of mass of the contours.
    :param contours: Contours from the images that needs to be rotated.
    :return: The rotated images.
    """

    print("Raytracing on the image...")

    # Variable where the length and angle will be stored.
    data = []
    # Variable to store the rotated images.
    img_output = []
    for nr in range(len(contours)):
        maxLength = 0
        data.append(createDict())
        for point in range(len(contours[nr])):
            # Compute x and y coordinate relative to the contours center of mass.
            x_delta = contours[nr][point][0][0] - xcm[nr]
            y_delta = contours[nr][point][0][1] - ycm[nr]
            # Compute the length and angle of each coordinate in the contours.
            data[nr]["length"].append(math.sqrt(pow(x_delta, 2) + pow(y_delta, 2)))
            data[nr]["angle"].append(math.atan2(y_delta, x_delta) * (180 / math.pi))
            # Finding the longest length and at what angle.
            if data[nr]["length"][point] > maxLength:
                maxLength = data[nr]["length"][point]
            # Draw the line tracing on the contours and point (optional)
            cv2.line(rotate_img[nr], (xcm[nr], ycm[nr]), (contours[nr][point][0][0], contours[nr][point][0][1]),
                     (255, 0, 0), 1)

        # Show COF contour
        cv2.circle(rotate_img[nr], (xcm[nr], ycm[nr]), radius=4, color=(0, 0, 255), thickness=-1)

        # Plot the contour coordinates length relative to the angle (optional):
        plt.subplot(int("1" + str(len(contours)) + str(nr + 1)))
        plt.bar(data[nr]["angle"], data[nr]["length"])
        plt.axis([-180, 180, 0, 500])

    print("Done raytracing!")
    plt.show()
    return rotate_img


def isolate_img2(resized_input_image):
    """
    Returns the image of which pixels was in range of the selected HSV values.

    :param resized_input_image: The image to find pixels in range of the HSV values
    :return: A mask of the isolated pixels
    """

    hsv_image = eip.convert_RGB_to_HSV(resized_input_image)

    def nothing(x):
        pass

    cv2.namedWindow("Adjust_Hue_Satuation_Value")
    cv2.createTrackbar("lowerH", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("lowerS", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("lowerV", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    cv2.createTrackbar("upperH", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("upperS", "Adjust_Hue_Satuation_Value", 0, 255, nothing)
    cv2.createTrackbar("upperV", "Adjust_Hue_Satuation_Value", 0, 255, nothing)

    # while loop to adjust the HSV detection in the image.

    while True:

        lowerH = cv2.getTrackbarPos("lowerH", "Adjust_Hue_Satuation_Value")
        lowerS = cv2.getTrackbarPos("lowerS", "Adjust_Hue_Satuation_Value")
        lowerV = cv2.getTrackbarPos("lowerV", "Adjust_Hue_Satuation_Value")

        upperH = cv2.getTrackbarPos("upperH", "Adjust_Hue_Satuation_Value")
        upperS = cv2.getTrackbarPos("upperS", "Adjust_Hue_Satuation_Value")
        upperV = cv2.getTrackbarPos("upperV", "Adjust_Hue_Satuation_Value")

        lowerRange_blue = np.array([lowerH, lowerS, lowerV])
        upperRange_blue = np.array([upperH, upperS, upperV])

        mask = cv2.inRange(hsv_image, lowerRange_blue, upperRange_blue)

        res = eip.bitwise_and(resized_input_image, mask)

        cv2.imshow("res", res)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1)
        if key == 27:
            break
    return res


def segment_cod2(images, cahe_clipSize, titleSize, show_images=False):
    print("Started segmenting the cods!")

    inRangeImages = []
    segmentedImages = []

    # def nothing(x):
    #    pass

    # cv2.namedWindow("res")
    # cv2.createTrackbar("lowerHseg", "res", 0, 255, nothing)
    # cv2.createTrackbar("higherHseg", "res", 0, 255, nothing)
    # cv2.createTrackbar("lowerSseg", "res", 0, 255, nothing)
    # cv2.createTrackbar("higherSseg", "res", 0, 255, nothing)

    for n in images:
        while True:
            hsv_img = cv2.cvtColor(n, cv2.COLOR_BGR2HSV)

            # lowerHseg = cv2.getTrackbarPos("lowerHseg", "res")
            # higherHseg = cv2.getTrackbarPos("higherHseg", "res")
            # lowerSseg = cv2.getTrackbarPos("lowerSseg", "res")
            # higherSseg = cv2.getTrackbarPos("higherSseg", "res")

            # lowerH = (lowerHseg, higherHseg)
            # lowerV = (lowerSseg, higherSseg)
            lowerH = (90, 128)
            lowerV = (0, 40)
            h, w, ch = hsv_img.shape[:3]

            mask = np.zeros((h, w), np.uint8)
            # We start segmenting
            for y in range(h):
                for x in range(w):
                    H = hsv_img.item(y, x, 0)
                    V = hsv_img.item(y, x, 2)
                    # If Hue lies in the lowerHueRange(Blue hue range) we want to segment it out
                    if lowerH[1] > H > lowerH[0]:
                        mask.itemset((y, x), 0)
                    # If Hue lies in the lowerValRange(black value range) we want to segment it out
                    elif lowerV[1] > V > lowerV[0]:
                        mask.itemset((y, x), 0)
                    else:
                        mask.itemset((y, x), 255)

            # NEEDS TO BE CHANGED TO OUR OWN BITWISE
            segmentedImg = cv2.bitwise_and(clahe, clahe, mask=mask)

            if show_images:
                cv2.imshow("res", segmentedImg)
                cv2.imshow("mask", mask)

                key = cv2.waitKey(1)
                if key == 27:
                    break

            break

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmentedImg)

    print("Finished segmenting the cods!")

    return inRangeImages, segmentedImages


def detect_bloodspots2(hsv_img):
    lowerH = (80, 125)
    lowerS = (153, 204)
    lowerv = (115, 140)

    h, w, ch = hsv_img.shape[:3]

    segmentedImg = np.zeros((h, w), np.uint8)
    # We start segmenting
    for y in range(h):
        for x in range(w):
            H = hsv_img.item(y, x, 0)
            S = hsv_img.item(y, x, 1)
            V = hsv_img.item(y, x, 2)
            # If Hue lies in the lowerHueRange(Blue hue range) we want to segment it out
            if lowerH[1] > H > lowerH[0]:
                segmentedImg.itemset((y, x), 0)
            # If Hue lies in the lowerValRange(black value range) we want to segment it out
            elif lowerv[1] > V > lowerv[0]:
                segmentedImg.itemset((y, x), 0)
            elif lowerS[1] > S > lowerS[0]:
                segmentedImg.itemset((y, x), 0)
            else:
                segmentedImg.itemset((y, x), 255)


def images_for_rapport2(images):
    equalized_images = []
    image = 0
    for n in images:
        # Equalized images
        gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        equalized_images.append(equalize)

        # Plotting
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].hist(n.ravel(), 256, [0, 256])
        axs[1].hist(equalize.ravel(), 256, [0, 256])
        cv2.imshow(f"Image Equalized: {image}", equalize)
        cv2.imshow(f"Image Not Equalized: {image}", gray)
        plt.show()

        image += 1
        cv2.destroyAllWindows()

    return 0
