import cv2
import matplotlib.pyplot as plt

from functions_openCV import crop, claheHSL, harrisCorner
from calibration import undistortImg
from mathias_functions import smallrange_isolate_img_content, detect_bloodspots
import functions_analyse as ft
import yamlLoader as yamlL
import extremeImageProcessing as eip

# User options
save_steps     = True
recalibrate    = False
openCV    = True

if openCV == False:
     checkeboardImg = cv2.imread('calibration/checkerboard_pics/GOPR1840.JPG',0)
     cornercoordinateImg = harrisCorner(checkeboardImg)
     print(cornercoordinateImg)

if openCV == True:
     'Step 1: Load image'
     img = cv2.imread("fish_pics/step1_haandholdt_closeup_GOPR1886.JPG", 1)

     'Step 2: undistort image'
     img_undistorted = undistortImg(img, recalibrate)

     'Step 3: Crop to ROI'
     img_cropped = crop(img_undistorted, 700, 450, 600, 2200) # for fish_pics/step1_haandholdt_closeup_GOPR1886.JPG
     #img_cropped = crop(img_undistorted, 900, 650, 500, 1500)

     'Step 4: Apply CLAHE'
     img_CLAHE = claheHSL(img_cropped, 2, (25,25))

     'Step 5: Segment'
     img_HSV = cv2.cvtColor(img_CLAHE, cv2.COLOR_BGR2HSV)
     img_segmented_cod = smallrange_isolate_img_content(img_CLAHE, img_HSV)

     'Step 6: Segment'
     img_HSV = cv2.cvtColor(img_CLAHE, cv2.COLOR_BGR2HSV)
     mask_cod, img_segmented_cod  = smallrange_isolate_img_content(img_CLAHE, img_HSV)

     'Step 7: Detect bloodspots'
     img_segmented_cod_HSV = cv2.cvtColor(img_segmented_cod, cv2.COLOR_BGR2HSV)
     mask_bloodspots, bloodspots, marked_bloodspots = detect_bloodspots(img_segmented_cod, img_segmented_cod_HSV)

     'Step ??: Show pre-processing steps in one plot'
     # OpenCV loads pictures in BGR, but the this step is plotted in RGB:
     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     img_undistorted_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
     img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
     img_CLAHE_rgb = cv2.cvtColor(img_CLAHE, cv2.COLOR_BGR2RGB)
     img_segmented_codrgb = cv2.cvtColor(img_segmented_cod, cv2.COLOR_BGR2RGB)
     bloodspotsrgb = cv2.cvtColor(bloodspots, cv2.COLOR_BGR2RGB)
     marked_bloodspotssrgb = cv2.cvtColor(marked_bloodspots, cv2.COLOR_BGR2RGB)

     # Create subplots
     fig = plt.figure()
     fig.suptitle('Step in algorithm', fontsize=16)

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
     plt.imshow(img_CLAHE_rgb)
     plt.title('ROI with CLAHE applied')

     plt.subplot(3, 3, 5)
     plt.imshow(img_segmented_codrgb)
     plt.title('Segmented cod')

     plt.subplot(3, 3, 6)
     plt.imshow(bloodspotsrgb)
     plt.title('Blood spots segmented')

     plt.subplot(3, 3, 7)
     plt.imshow(marked_bloodspotssrgb)
     plt.title('Blood spots tagged')

     plt.show()

'Step ??: Save output image for inspection by biologists'

'Step ??: Save all images in step categories'
if save_steps == True:
     cv2.imwrite('fish_pics/step2.JPG',img_undistorted)
     cv2.imwrite('fish_pics/step3.JPG',img_cropped)
     cv2.imwrite('fish_pics/step4.JPG',img_CLAHE)
     cv2.imwrite('fish_pics/step6.JPG',img_segmented_cod)
     cv2.imwrite('fish_pics/step7.JPG', bloodspots)
     cv2.imwrite('fish_pics/step8.JPG', marked_bloodspots)

'''
# load images into memory
#images, names = loadImages("fishpics/direct2pic", True, True, 40)

# Calibrate images
#img_cali, names_cali = loadImages("fishpics/CheckerBoards", True, False, 40)

# Calibrate camera
#fish_cali = checkerboard_calibrate(images, img_cali)

# Calibrated fish images
#left = images[0]
#right = images[1]
import time

# Specular highlights
#img_spec_rem = replaceHighlights(left, right, 225)  # 230


with_polfilter = cv2.imread("fishpics/direct2pic/with_polfilter.jpg")

mask_gras = cv2.imread("fishpics/direct2pic/TestGrassFire.png", 0)

#making gray scale image without opencv
#grayscale = grayScaling(with_polfilter)

#making gray scale image 8 bit without 8 bit
#grayscaleimg8bit = grayScaling8bit(with_polfilter)

#making hsv image without opencv
hsv_img = convert_RGB_to_HSV(with_polfilter)
#cv2.imshow("hvs_own", hsv_img)
# Calibrating hsv mask to mask to grassfire
#isolated_mask = isolate_img(with_polfilter, hsv_img)
#runnig mask with hsv img setup
#mask_to_grassfire = creating_mask_input_hsv(with_polfilter, hsv_img)
#running grassfire algorithm
segmentedimg = smallrange_isolate_img_content(with_polfilter, hsv_img)

hsv_img_segmented = convert_RGB_to_HSV(segmentedimg)
detect_bloodspots(segmentedimg, hsv_img_segmented)
#grassfire_transform(mask_to_grassfire, with_polfilter)
#grassfirealgorithm
#grassfire_v2(smallrange)
# segmentation
#isolated_mask = isolate_img(with_polfilter, hsv_img)

#erosion
#erosion(isolated_mask)

#save each processed image
#save_img(left)

#canny detection
#canny_edge_detection(with_polfilter, mask_to_grassfire)

# Get the contours
#find_contours(fish_cali)

# Blood spots
#blood_spot = find_blood_damage(img_spec_rem)
#morph = morphological_trans(blood_spot)

# Convert morph to color so we can add it together with our main image
#morph_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
#final = cv2.addWeighted(img_spec_rem, 1, morph_color, 0.6, 0)

# display images and it's names
#cv2.imshow(f"Left: {names[0]}", left)
#cv2.imshow(f"Right: {names[1]}", right)
#cv2.imshow("Final", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Rapport images
# img = ft.images_for_rapport()
# Check the runtime
start_time = time.time()

# Load in yaml data from the file
yaml_data = yamlL.yaml_loader("parameters.yaml")

# Load in the yaml parameters from the data
kernels, checkerboard_dimensions, paths, clahe = yamlL.setup_parameters(yaml_data)

# load Fish images
images, names = ft.loadImages(paths[0][1], False, False, 40)

# Load checkerboard images
img_cali, names_cali = ft.loadImages(paths[1][1], False, False, 40)

# Calibrate camera and undistort images
fish_cali = ft.checkerboard_calibrate(checkerboard_dimensions, images, img_cali, False)

# Crop to ROI
cropped_images = eip.crop(images, 700, 450, 600, 2200)

# Threshold to create a mask for each image
# masks = eip.findInRange(fish_cali) - Might need to get removed

# Threshold to create a mask for each image
masks, segmented_images = ft.segment_cod(cropped_images, clahe[0][1], clahe[1][1], False)

# Bloodspot detection
# blodspot_img = ft.detect_bloodspots()      Not done yet

# Get the contours
contour = ft.find_contours(masks, cropped_images)

# Find contours middle point
xcm, ycm = ft.contour_MOC(cropped_images, contour)

# Raytracing
rot_img = ft.rotateImages(cropped_images, xcm, ycm, contour)

# Print runtime
print("Execution time: ","--- %s seconds ---" % (time.time() - start_time))

# display images and it's names
#cv2.imshow(f"Left: {names[0]}", left)
#cv2.imshow(f"Right: {names[1]}", right)
cv2.imshow("Final", rot_img[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
