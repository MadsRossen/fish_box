import cv2
import numpy as np
from matplotlib import pyplot as plt

def normHistEqualizeHLS(img):
    """
        Performs histogram equalization on images

        Input images must be BGR
    """
    fiskHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannel = fiskHLS[:,:,1]
    HistEqualize = cv2.equalizeHist(LChannel)
    fiskHLS[:,:,1] = HistEqualize
    fiskNomrHistEq = cv2.cvtColor(fiskHLS,cv2.COLOR_HLS2BGR)
    return fiskNomrHistEq


def claheHSL(img,clipLimit,tileGridSize):
    """
        Performs CLAHE on images

        Input images must be BGR
    """
    fiskHLS2 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    LChannelHLS = fiskHLS2[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelHLS)
    fiskHLS2[:, :, 1] = claheLchannel1
    fiskClahe = cv2.cvtColor(fiskHLS2, cv2.COLOR_HLS2BGR)
    return fiskClahe


def claheLAB(img,clipLimit,tileGridSize):
    """
    Performs CLAHE on images

    Input images must be BGR
    """
    fiskLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    LChannelLAB = fiskLAB[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLchannel1 = clahe.apply(LChannelLAB)
    fiskLAB[:, :, 0] = claheLchannel1
    fiskClaheLAB = cv2.cvtColor(fiskLAB, cv2.COLOR_LAB2BGR)
    return fiskClaheLAB


def limitLchannel(img, limit):
    """
    Sets a limit on the light channel in a HSL version of input image
    """

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
    return imgHLS

def crop(img,y,x,height,width):
    """
    Crops images
    """
    ROI = img[y:y + height, x:x + width]
    return ROI

def doClaheLAB1(null):
    """
    Creates slider for performing CLAHE
    """
    global val1
    val1 = cv2.getTrackbarPos('cliplimit', 'ResultHLS')
    res = claheHSL(img, val1/10,kernel)
    cv2.imshow("ResultHLS", res)
    plt.hist(res.ravel(), 256, [0, 256]);
    plt.hist(res.ravel(), 256, [0, 256]);
    plt.show()

def doClaheLAB2(null):
    """
    Creates slider for performing CLAHE
    """
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


def resizeImg(img, scale_percent):
    """
        Resizes images with a scale percentage
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # resize image
    return resized

def meanEdgeRGB(img, middleYValue):
    """
        Finds the mean RGB values of the edge of fish in a .png image

        middleYValue: The y value of the line that separates the back and belly region of the fish
    """

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
                    if y >= middleYValue:
                        bellyBlue = bellyBlue + res.item(y,x,0)
                        bi = bi + 1
                    if y < middleYValue:
                        backBlue = backBlue + res.item(y, x, 0)
                        bki = bki + 1

                if res.item(y,x,1) > 0:
                    green = green + res.item(y,x,1)
                    u = u + 1
                    if y >= middleYValue:
                        bellyGreen = bellyGreen + res.item(y,x,1)
                        bu = bu + 1
                    if y < middleYValue:
                        backGreen = backGreen + res.item(y, x, 1)
                        bku = bku + 1

                if res.item(y,x,2) > 0:
                    red = red + res.item(y,x,2)
                    k = k + 1
                    if y >= middleYValue:
                        bellyRed = bellyRed + res.item(y, x, 2)
                        bk = bk + 1
                    if y < middleYValue:
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

def SURFalignment(img1, img2):
    """
    # SURFAlignment aligns two images. Based on https://www.youtube.com/watch?v=cA8K8dl-E6k&t=131s
    """

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





