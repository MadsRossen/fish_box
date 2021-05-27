import time

import cv2

# Imports of functions where own built functions have been used:
from extremeImageProcessing import saveCDI
# Imports of functions where openCV functions have been used:
from functions_openCV import checkerboard_calibrateOPENCV, detect_bloodspotsOPENCV, segment_codOPENCV, showSteps, \
    save_imgOPENCV, crop, loadImages, claheHSL, segment_cod_CLAHEOPENCV
from yamlLoader import yaml_loader, setup_parameters

'''User options'''

# Run algorithm with opencvV or own built functions
openCV = True

# Choose which fish the steps are shown for
showFish = 0

'''User options'''

# Run program using openCV functions
if openCV:
    stepsList = []

    # Start timer for algorithm
    start_time = time.time()

    # Load in yaml data from the file
    yaml_data = yaml_loader("parameters.yaml")

    # Load in the yaml parameters from the data
    kernels, checkerboard_dimensions, paths, clahe = setup_parameters(yaml_data, printParameters=False)

    # Load Fish images
    images, names, img_list_fish = loadImages(paths[0][1], False, False, 40)
    stepsList.append(images[showFish])

    # Load checkerboard images
    img_cali, names_cali, _ = loadImages(paths[1][1], False, False, 40)

    # Calibrate camera and undistort images
    fish_cali = checkerboard_calibrateOPENCV(checkerboard_dimensions, images, img_cali,
                                             show_img=False, recalibrate=False)
    stepsList.append(fish_cali[showFish])

    # Apply CLAHE
    CLAHE = claheHSL(fish_cali, 2, (25,25))
    stepsList.append(CLAHE[showFish])

    # Crop to ROI
    cropped_images = crop(fish_cali, 710, 200, 720, 2500)
    stepsList.append(cropped_images[showFish])
    cropped_images_CLAHE = crop(CLAHE, 710, 200, 720, 2500)
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/ccc.JPG', cropped_images[2])


    # Threshold to create a mask for each image
    mask_cod, img_segmented_cod = segment_codOPENCV(cropped_images)
    stepsList.append(img_segmented_cod[showFish])
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/segment.JPG', img_segmented_cod[2])


    cv2.imwrite('segment_this.JPG', cropped_images_CLAHE[0])
    mask_cod, img_segmented_cod_CLAHE = segment_cod_CLAHEOPENCV(cropped_images_CLAHE)
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/clahe1.JPG', img_segmented_cod_CLAHE[0])
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/clahe2.JPG', img_segmented_cod_CLAHE[1])
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/clahe3.JPG', img_segmented_cod_CLAHE[2])
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/clahe4.JPG', img_segmented_cod_CLAHE[3])
    cv2.imwrite('fish_pics/output_images/manual_inspection_CLAHE/mask_cod.JPG', mask_cod[2])

    # Blood spot detection
    mask_bloodspots, bloodspots, marked_bloodspots, \
    percSpotCoverage = detect_bloodspotsOPENCV(img_segmented_cod, mask_cod)
    stepsList.append(bloodspots[showFish])
    stepsList.append(marked_bloodspots[showFish])


    # Save marked blood spots images in folder
    save_imgOPENCV(marked_bloodspots, 'fish_pics/output_images/marked_images', img_list_fish)

    # Save a .txt file with CDI (catch damage index)
    saveCDI(img_list_fish, percSpotCoverage)

    # Check how long it took for algorithm to finish
    end_time = time.time()
    print('Total time for algoritm: ', end_time - start_time, 'sec')

    # Create subplots of main steps
    showSteps(stepsList)

# Run program using own built functions
if openCV == False:
    0
