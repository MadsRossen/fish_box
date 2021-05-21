import copy
import os
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np

def showSteps(stepsList):
    '''
    Create subplots showing main steps in algorithm

    :return: None
    '''

    # OpenCV loads pictures in BGR, but the this step is plotted in RGB:
    img_rgb = cv2.cvtColor(stepsList[0], cv2.COLOR_BGR2RGB)
    img_undistorted_rgb = cv2.cvtColor(stepsList[1], cv2.COLOR_BGR2RGB)
    img_cropped_rgb = cv2.cvtColor(stepsList[2], cv2.COLOR_BGR2RGB)
    img_segmented_codrgb = cv2.cvtColor(stepsList[3], cv2.COLOR_BGR2RGB)
    bloodspotsrgb = cv2.cvtColor(stepsList[4], cv2.COLOR_BGR2RGB)
    marked_bloodspotssrgb = cv2.cvtColor(stepsList[5], cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    fig.suptitle('Steps in algorithm', fontsize=16)

    plt.subplot(3, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original image')

    plt.subplot(3, 3, 2)
    plt.imshow(img_undistorted_rgb)
    plt.title('Undistorted image')

    plt.subplot(3, 3, 3)
    plt.imshow(img_cropped_rgb)
    plt.title('ROI')

    plt.subplot(3, 3, 4)
    plt.imshow(img_segmented_codrgb)
    plt.title('Segmented cod')

    plt.subplot(3, 3, 5)
    plt.imshow(bloodspotsrgb)
    plt.title('Blood spots segmented')

    plt.subplot(3, 3, 6)
    plt.imshow(marked_bloodspotssrgb)
    plt.title('Blood spots tagged')

    plt.show()

def save_imgOPENCV(imgs, path, originPathNameList):
    '''
    Saves a list of images in the folder that the path is set to.

    :param originPathName: The path of the original path of the images that have been manipulated.
    :param imgs: A list of images.
    :param path: The path that the images will be saved to.
    :return: None
    '''

    print('Saving images')

    count = 0
    if len(imgs) > 1:
        for n in imgs:
            cv2.imwrite(path + f"\\{originPathNameList[count]}_marked.JPG", n)
            count = count + 1

    print('Done saving images')

def detect_bloodspotsOPENCV(imgs):

    mask_bloodspots = []
    segmented_blodspots_imgs = []
    marked_bloodspots_imgs = []
    booleans_bloodspot = []         # List of boolean values for each image classification
    count = 0

    for n in imgs:
        hsv_img = cv2.cvtColor(copy.copy(n), cv2.COLOR_BGR2HSV)

        booleans_bloodspot.append(False)
        marked_bloodspots_imgs.append(copy.copy(n))

        '''
        # Mean
        frame_threshold1 = cv2.inRange(hsv_img, (0, 70, 50), (10, 255, 255))
        frame_threshold2 = cv2.inRange(hsv_img, (170, 70, 50), (180, 255, 255))

        # Combining the masks
        mask_bloodspots.append(frame_threshold1 | frame_threshold2)
        
        # Threshold for blood spots best yet
        frame_threshold1 = cv2.inRange(hsv_img, (0,90, 90), (10, 255, 255))
        frame_threshold2 = cv2.inRange(hsv_img, (0, 90, 90), (10, 255, 255))
        '''

        # Threshold for blood spots
        frame_threshold1 = cv2.inRange(hsv_img, (0, 90, 90), (10, 255, 255))
        frame_threshold2 = cv2.inRange(hsv_img, (0, 90, 90), (10, 255, 255))

        #cv2.inRange(hsv_img, (170, 70, 50), (180, 255, 255))

        # Combining the masks
        mask_bloodspots.append(frame_threshold1 | frame_threshold2)

        # Create kernels for morphology
        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((50, 50), np.uint8)

        # Perform morphology
        open = cv2.morphologyEx(mask_bloodspots[count], cv2.MORPH_OPEN, kernelOpen)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernelClose)

        # Perform bitwise operation to show bloodspots instead of BLOBS
        segmented_blodspots_imgs.append(cv2.bitwise_and(n, n, mask=close))

        # Make representation of BLOB / bloodspots
        # Find contours
        contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Classify as blood spots if the spots are big enought
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > 0:
                x, y, w, h = cv2.boundingRect(cont)
                # Create tag
                cv2.putText(marked_bloodspots_imgs[count], 'Wound', (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                # Draw green contour
                cv2.rectangle(marked_bloodspots_imgs[count],(x-5,y-5),(x+w+5,y+h+5),(0,255,0), 2);
                #cv2.drawContours(marked_bloodspots_imgs[count], [cont], -1, (0,255,0), 2)
                booleans_bloodspot.append(True)

        count = count + 1

    return mask_bloodspots, segmented_blodspots_imgs, marked_bloodspots_imgs, booleans_bloodspot

def segment_codOPENCV(images, show_images=False):

    print("Started segmenting the cod!")

    inRangeImages = []
    segmentedImages = []

    for n in images:
        hsv_img = copy.copy(n)
        hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

        # Create threshold for segmenting cod
        mask = cv2.inRange(hsv_img, (101, 21, 65), (180, 255, 255))

        # Invert the mask
        mask = (255 - mask)

        # Create kernels for morphology
        #kernelOpen = np.ones((4, 4), np.uint8)
        #kernelClose = np.ones((7, 7), np.uint8)

        kernelOpen = np.ones((9, 9), np.uint8)
        kernelClose = np.ones((20, 20), np.uint8)

        # Perform morphology
        open1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        close2 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernelClose)

        segmented_cods = cv2.bitwise_and(n, n, mask=close2)

        if show_images:
            cv2.imshow("res", segmented_cods)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

        # add to lists
        inRangeImages.append(mask)
        segmentedImages.append(segmented_cods)

    print("Finished segmenting the cod!")

    return inRangeImages, segmentedImages

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
        #plt.imshow(segmentedImg, cmap="gray")
        #plt.show()

def isolate_img(resized_input_image, hsv_image):

    #hsv_image = cv2.cvtColor(resized_input_image, cv2.COLOR_BGR2HSV)

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
    #cv2.createTrackbar("kernel1", "Canny_detection", 0, 50, nothing)
    #cv2.createTrackbar("kernel2", "Canny_detection", 0, 50, nothing)

    while True:

        c1 = cv2.getTrackbarPos("c1", "Canny_detection")
        c2 = cv2.getTrackbarPos("c2", "Canny_detection")
        #k1 = cv2.getTrackbarPos("kernel1", "Canny_detection")
        #k2 = cv2.getTrackbarPos("kernel2", "Canny_detection")

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
            avg = float(((list[0]+list[1]+list[2])/3)/255)
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

    B, G, R = img[:, :, 0]/255, img[:, :, 1]/255, img[:, :, 2]/255

    hsv_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(width):
        for j in range(height):

        # Defining Hue
            h, s, v = 0.0, 0.0, 0.0
            r, g, b = R[i][j], G[i][j], B[i][j]

            max_rgb, min_rgb = max(r, g, b), min(r, g, b)
            dif_rgb = (max_rgb-min_rgb)

            if r == g == b:
                h = 0
            elif max_rgb == r:
                h = ((60*(g-b))/dif_rgb)
            elif max_rgb == g:
                h = (((60*(b-r))/dif_rgb)+120)
            elif max_rgb == b:
                h = (((60*(r-g))/dif_rgb)+240)
            if h < 0:
                h = h+360

        # Defining Satuation
            if max_rgb == 0:
                s = 0
            else:
                s = ((max_rgb-min_rgb)/max_rgb)
        # Defining Value

            v = max_rgb
            #print(h, s, v)
            hsv_img[i][j][0], hsv_img[i][j][1], hsv_img[i][j][2] = h/2, s*255, v*255

    return hsv_img

def erosion (img):

    k1 = 5
    k2 = 5
    c1 = (k1-1)
    c2 = (k2-1)

    width, height = img.shape

    #structuring_element = np.array([[1, 1, 1],
                                    #[1, 1, 1],
                                    #[1, 1, 1]])

    imgErode = np.zeros((width, height), dtype=img.dtype)

    kernel = np.ones(k1, k2)

    for i in range(c1, width-c1):
        for j in range(c2, height-c2):
            temp = img[i-c1:i+c1+1][j-c2:j+c2+1]
            product = temp*kernel
            imgErode[i][j] = np.min(product)

    cv2.imshow("erodeIMG", imgErode)

    cv2.waitKey(0)

def grassfire_transform(mask, img):
    """
    Apply the grassfire transform to a binary mask array.
    """
    #imgGray = grayScaling8bit(img)

    h, w = mask.shape
    # Use uint32 to avoid overflow
    grassfire = np.zeros_like(mask, dtype=np.uint8)

    # 1st pass
    # Left to right, top to bottom
    for x in range(w):
        for y in range(h):
            if mask[y, x] != 0: # Pixel in contour
                north = 0 if y == 0 else grassfire[y - 1, x]
                west = 0 if x == 0 else grassfire[y, x - 1]
                if x == 3 and y == 3:
                    print(north, west)
                grassfire[y, x] = 1 + min(west, north)

    # 2nd pass
    # Right to left, bottom to top
    for x in range(w - 1, -1, -1):
        for y in range(h - 1, -1, -1):
            if mask[y, x] != 0: # Pixel in contour
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
            #x = x + 1
            #y = y + 1
            #try:
                #print(mask[y, x])
            #except:
                #print(y, x)
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
    h = h-1
    w = w-1
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
                if len(save_array)>0:
                    y,x = save_array.pop()

                else:
                    print("Burn is done")
                    blob_array.append(temp_cord)
                    temp_cord = []
                    break
    maskColor = np.zeros((h,w, 3), np.uint8)
    for blob in range(len(blob_array)):
        B, G, R = randint(0, 255), randint(0, 255), randint(0, 255)
        for cord in blob_array[blob]:
            y,x = cord
            maskColor[y][x][0] = B
            maskColor[y][x][1] = G
            maskColor[y][x][2] = R
    cv2.imshow("grasfire", maskColor)
    cv2.waitKey(0)