import time

import cv2
from matplotlib import pyplot as plt
import calibration
import extremeImageProcessing as eip
import functions_analyse as ft
import yamlLoader as yamlL
from mathias_functions import segment_codOPENCV, detect_bloodspotsOPENCV, save_imgOPENCV

# User options
openCV         = True

# Run program using openCV functions
if openCV:
     start_time = time.time()

     # Load in yaml data from the file
     yaml_data = yamlL.yaml_loader("parameters.yaml")

     # Load in the yaml parameters from the data
     kernels, checkerboard_dimensions, paths, clahe = yamlL.setup_parameters(yaml_data, printParameters=False)

     # Load Fish images
     images, names, img_list_fish = ft.loadImages(paths[0][1], False, False, 40)

     # Load checkerboard images
     img_cali, names_cali, _ = ft.loadImages(paths[1][1], False, False, 40)

     # Calibrate camera and undistort images
     fish_cali = calibration.checkerboard_calibrateOPENCV(checkerboard_dimensions, images, img_cali,
                                                          show_img=False, recalibrate=False)
     # Crop to ROI
     cropped_images = eip.crop(fish_cali, 710, 200, 720, 2500)

     # Threshold to create a mask for each image
     mask_cod, img_segmented_cod = segment_codOPENCV(cropped_images)
     cv2.imwrite('fish_pics/segment_this.JPG', img_segmented_cod[5])

     # Bloodspot detection
     mask_bloodspots, bloodspots, marked_bloodspots, boolean_bloodspot = detect_bloodspotsOPENCV(img_segmented_cod)

     # Save marked bloodspots images in folder
     save_imgOPENCV(marked_bloodspots, 'fish_pics/output_images', img_list_fish)

     # Create subplots of steps
     # OpenCV loads pictures in BGR, but the this step is plotted in RGB:
     fish_nr = 2
     img_rgb = cv2.cvtColor(images[fish_nr], cv2.COLOR_BGR2RGB)
     img_undistorted_rgb = cv2.cvtColor(fish_cali[fish_nr], cv2.COLOR_BGR2RGB)
     img_cropped_rgb = cv2.cvtColor(cropped_images[fish_nr], cv2.COLOR_BGR2RGB)
     img_segmented_codrgb = cv2.cvtColor(img_segmented_cod[fish_nr], cv2.COLOR_BGR2RGB)
     bloodspotsrgb = cv2.cvtColor(bloodspots[fish_nr], cv2.COLOR_BGR2RGB)
     marked_bloodspotssrgb = cv2.cvtColor(marked_bloodspots[fish_nr], cv2.COLOR_BGR2RGB)

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

# Run program using own built functions
     if openCV == False:
          openCV = True

'''
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

if openCV == False:
     checkeboardImg = cv2.imread('calibration/checkerboard_pics/GOPR1840.JPG',0)
     cornercoordinateImg = harrisCorner(checkeboardImg)
     print(cornercoordinateImg)

if openCV == True:
     start_time = time.time()
     'Step 1: Load image'
     img = cv2.imread("fish_pics/input_images/step1_nytest_GOPR1956.JPG", 1)

     'Step 2: undistort image'
     img_undistorted = undistortImg(img, recalibrate)

     'Step 3: Crop to ROI'
     img_cropped = crop(img_undistorted, 670, 250, 950, 2650)  # for fish_pics/input_images/step1_nytest_GOPR1956.JPG

     'Step 5: Segment'
     img_HSV = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
     mask_cod, img_segmented_cod  = smallrange_isolate_img_content(img_cropped, img_HSV)

     'Step 6: Detect bloodspots'
     img_segmented_cod_HSV = cv2.cvtColor(img_segmented_cod, cv2.COLOR_BGR2HSV)
     mask_bloodspots, bloodspots, marked_bloodspots, boolean_bloodspot = detect_bloodspots(img_segmented_cod, img_segmented_cod_HSV)

     'Step ??: Show pre-processing steps in one plot'
     # OpenCV loads pictures in BGR, but the this step is plotted in RGB:
     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     img_undistorted_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
     img_cropped_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
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

     plt.subplot(3, 3, 5)
     plt.imshow(img_segmented_codrgb)
     plt.title('Segmented cod')

     plt.subplot(3, 3, 6)
     plt.imshow(bloodspotsrgb)
     plt.title('Blood spots segmented')

     plt.subplot(3, 3, 7)
     plt.imshow(marked_bloodspotssrgb)
     plt.title('Blood spots tagged')

'Step ??: Save output image for inspection by biologists'

'Step ??: Save all images in step categories'
if save_steps == True:
     cv2.imwrite('fish_pics/step1.JPG', img)
     cv2.imwrite('fish_pics/step2.JPG',img_undistorted)
     cv2.imwrite('fish_pics/step3.JPG',img_cropped)

     cv2.imwrite('fish_pics/step6.JPG',img_segmented_cod)
     cv2.imwrite('fish_pics/step6_2.JPG',mask_cod)
     cv2.imwrite('fish_pics/step7.JPG', bloodspots)
     cv2.imwrite('fish_pics/step8.JPG', marked_bloodspots)

end_time = time.time()
print("Execution time for optimized item/itemset function: ", "--- %s seconds ---" % (end_time - start_time))
'''