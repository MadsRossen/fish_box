import cv2
import numpy as np
from matplotlib import pyplot as plt

def white_balance(frame):
    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(1)
    new_frame = wb.balanceWhite(frame)
    return new_frame

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

def normHistEqualizeHLS(img):
    fiskHLS1 = img
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

def meanEdgeRGB(img, widthOfEdge, middleBelly=0):
    '''
    Finds mean RGB values of the edge of a  cod.
    :param img: The image with the object. Mut have removed background, showing only the object, NOTE: The cod must be
    horizontally placed in the image
    :param widthOfEdge: The width of the edge that is taken.
    :param middleOfCod: The distance between the top of the image to the middle of the cod in pixels.
    :return:
    '''

    print('Calculating mean edge value')
    cv2.imshow('img', img)
    cv2.waitKey(0)

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,imgGrayBin = cv2.threshold(imgGray,0, 255,cv2.THRESH_BINARY)
    cv2.imshow('imgGrayBin',imgGrayBin)

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
                    if y >= middleBelly:
                        bellyBlue = bellyBlue + res.item(y,x,0)
                        bi = bi + 1
                    if y < middleBelly:
                        backBlue = backBlue + res.item(y, x, 0)
                        bki = bki + 1

                if res.item(y,x,1) > 0:
                    green = green + res.item(y,x,1)
                    u = u + 1
                    if y >= middleBelly:
                        bellyGreen = bellyGreen + res.item(y,x,1)
                        bu = bu + 1
                    if y < middleBelly:
                        backGreen = backGreen + res.item(y, x, 1)
                        bku = bku + 1

                if res.item(y,x,2) > 0:
                    red = red + res.item(y,x,2)
                    k = k + 1
                    if y >= middleBelly:
                        bellyRed = bellyRed + res.item(y, x, 2)
                        bk = bk + 1
                    if y < middleBelly:
                        backRed = backRed + res.item(y, x, 2)
                        bkk = bkk + 1

    meanBlue = blue/i
    meanGreen = green/u
    meanRed = red/k

    print('mean blue: ',meanBlue)
    print('mean green: ',meanGreen)
    print('mean red: ', meanRed)
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
    print(' ')

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

def highestPixelValue(imgOrg, invrt):
    cv2.imshow('imgOrg',imgOrg)
    img = imgOrg
    width,height,channels = img.shape

    high = 0
    highcount = 0
    limit = 100
    limitchan = 10

    for chan in range(channels):
        for x in range(width):
            for y in range(height):
                if invrt != True:
                    if img.item(x,y,chan) > high:
                        high = img.item(x,y,chan)
                elif invrt == True:
                    if limitchan > img.item(x,y,chan):

                        '''
                        for i in range(1):
                            for k in range(1):
                                img.itemset((x - i, y - k, chan),255)
                        '''

                    '''                    
                    if limit > img[x, y, 0] & limit > img[x, y, 1] & limit > img[x, y, 2]:
                        for i in range(3):
                            for k in range(3):
                                img[x - i, y - k, 0] = 0
                                img[x - i, y - k, 1] = 0
                                img[x - i, y - k, 2] = 0
                    '''

    for chan in range(channels):
        for x in range(width):
            for y in range(height):
                if high == img[x,y,chan]:
                    highcount = highcount+1

    total = height*width*chan
    percentageOfPictureHighlights = highcount/total*100

    print(high)
    print(total)
    print(percentageOfPictureHighlights)
    cv2.imshow('img',img)


def showCompariHist(img1,img2, stringImg1, stringImg2):
    color = ('b', 'g', 'r')
    img1histr = 0
    img2histr = 0
    h1, w1, chan1 = img1.shape
    h2, w2, chan2 = img2.shape

    # Create histograms
    for i, col in enumerate(color):
        img1histrchannel = cv2.calcHist([img1], [i], None, [256], [0, 256])
        img1histr = img1histr + img1histrchannel

        img2histrchannel = cv2.calcHist([img2], [i], None, [256], [0, 256])
        img2histr = img2histr + img2histrchannel

    # Normalize histogram
    pixelsTotal1 = h1 * w1 * chan1
    pixelsTotal2 = h2 * w2 * chan2
    img1histrNorm = img1histr / pixelsTotal1
    img2histrNorm = img2histr/pixelsTotal2

    plt.plot(img1histrNorm, color='orange',label=stringImg1), plt.plot(img2histrNorm, color='blue',label=stringImg2)
    plt.legend(loc = 'upper right')
    plt.xlim([0, 256])
    plt.show()

def showCompari3Hist(img1,img2, img3, stringImg1, stringImg2, stringImg3):
    color = ('b', 'g', 'r')
    img1histr = 0
    img2histr = 0
    img3histr = 0
    h1, w1, chan1 = img1.shape
    h2, w2, chan2 = img2.shape
    h3, w3, chan3 = img3.shape

    # Create histograms
    for i, col in enumerate(color):
        img1histrchannel = cv2.calcHist([img1], [i], None, [256], [0, 256])
        img1histr = img1histr + img1histrchannel

        img2histrchannel = cv2.calcHist([img2], [i], None, [256], [0, 256])
        img2histr = img2histr + img2histrchannel

        img3histrchannel = cv2.calcHist([img3], [i], None, [256], [0, 256])
        img3histr = img3histr + img3histrchannel

    # Normalize histogram
    pixelsTotal1 = h1 * w1 * chan1
    pixelsTotal2 = h2 * w2 * chan2
    pixelsTotal3 = h3 * w3 * chan3
    img1histrNorm = img1histr / pixelsTotal1
    img2histrNorm = img2histr/pixelsTotal2
    img3histrNorm = img3histr / pixelsTotal3

    plt.plot(img1histrNorm, color='orange'), plt.plot(img2histrNorm, color='blue'), plt.plot(img3histrNorm, color='green')
    plt.xlim([0, 256])
    plt.text(50, 0.001, stringImg1, color='orange')
    plt.text(150, 0.001, stringImg2, color='blue')
    plt.text(200, 0.001, stringImg3, color='green')
    plt.show()

def histColor(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

