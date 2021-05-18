import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def showCompariHist(img1,img2, stringImg1, stringImg2, mode):
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
        img1histrNorm = (img1histr / (h1 * w1))*100
        img2histrNorm = (img2histr / (h2 * w2))*100
        print(sum(img1histrNorm), sum(img2histrNorm))
        plt.plot(img1histrNorm, color='orange'), plt.plot(img2histrNorm, color='blue')
        print("mean value for img1 = ", sum1/(h1 * w1))
        print("mean value for img2 = ", sum2/(h2 * w2))


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
        return 0,0

    sum = 0
    histSize = 256
    imgHistograms = []
    imgMeanHistograms = []
    for img in images:
            n_img = cv2.imread(img)
            grey_img = BGR2MeanGreyscale(n_img)
            #B, G, R = cv2.split(n_img)
            imgHistograms.append(cv2.calcHist([grey_img], [0], None, [histSize], [0, histSize]))
    for i in range(histSize):
        for histo in imgHistograms:
            sum = sum + histo[i]
        mean = sum/len(images)
        sum = 0
        imgMeanHistograms.append(mean)
    imgMeanHistograms = np.array(imgMeanHistograms)

    h1, w1 = n_img.shape[:2]
    sum1 = 0
    for i in range(len(imgMeanHistograms)):
        sum1 = sum1 + (i * imgMeanHistograms[i])

    # Normalize histograms
    img1histrNorm = (imgMeanHistograms / (h1 * w1))*100
    plt.plot(img1histrNorm, color='orange')
    mean_value = sum1/(h1 * w1)

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
            I1 = (img.item(y, x, 0) + img.item(y, x, 1) + img.item(y, x, 2))/3
            greyscale_img1.itemset((y, x), I1)
    print("Execution time for optimized item/itemset function: ","--- %s seconds ---" % (time.time() - start_time))


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